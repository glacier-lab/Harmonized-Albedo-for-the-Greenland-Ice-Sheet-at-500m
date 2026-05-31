"""
Build daily mosaics of VIIRS surface reflectance over Greenland.

This script processes VIIRS SR products (VNP09GA, VJ109GA, VJ209GA) and
creates daily orthomosaics reprojected to EPSG:3413 (NSIDC Polar Stereographic
North) at the PROMICE mask grid resolution.

Processing workflow:
- Reads 500 m I-bands (I1, I2, I3) and 1 km M-bands (M1, M2, M3, M4, M5,
  M7, M8, M10, M11) from HDF5 files.
- Applies scale/offset calibration from dataset attributes.
- Applies QA masks from SurfReflect_QF1_1 ... SurfReflect_QF7_1.
- Reprojects each tile to the Greenland mask grid (EPSG:3413).
- Averages overlapping pixels per day.
- Applies PROMICE ice mask.
- Exports one multi-band GeoTIFF per day and product.

Notes:
- Input root is expected to contain product subfolders:
  VNP09GA, VJ109GA, VJ209GA.
- Output band order is written in dataset-level tag BAND_NAMES.

Shunan Feng (shunan.feng@envs.au.dk)
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import rasterio as rio
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


SUPPORTED_PRODUCTS = ("VNP09GA", "VJ109GA", "VJ209GA")

# 500 m imagery bands
I_BANDS = ("I1", "I2", "I3")
# 1 km imagery bands
M_BANDS = ("M1", "M2", "M3", "M4", "M5", "M7", "M8", "M10", "M11")

BAND_NAMES = tuple([f"SurfReflect_{b}" for b in I_BANDS + M_BANDS])
N_BANDS = len(BAND_NAMES)

SINUSOIDAL_CRS = CRS.from_proj4(
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)


def read_mask(mask_path):
    with rio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

        valid_rows = np.where(np.any(mask != src.nodata, axis=1))[0]
        valid_cols = np.where(np.any(mask != src.nodata, axis=0))[0]

        row_start, row_end = valid_rows[0], valid_rows[-1] + 1
        col_start, col_end = valid_cols[0], valid_cols[-1] + 1

        mask_cropped = mask[row_start:row_end, col_start:col_end]
        new_transform = transform * Affine.translation(col_start, row_start)
        mask_cropped = np.where(mask_cropped <= 0, 0, 1)

    return mask_cropped, new_transform, crs, mask_cropped.shape


def parse_struct_metadata(struct_text):
    """Parse HDFEOS StructMetadata.0 text to a dict of key-value strings."""
    metadata = {}
    for line in struct_text.split("\n"):
        line = line.strip()
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip().strip('"')
    return metadata


def parse_point_mtrs(point_string):
    # Example: "(-4447802.078667,8895604.157333)"
    point_string = point_string.strip().strip("()")
    x_str, y_str = point_string.split(",")
    return float(x_str), float(y_str)


def get_eqa_transform(metadata_dict, cols, rows):
    """Construct sinusoidal transform using StructMetadata tile bounds."""
    west, north = parse_point_mtrs(metadata_dict["UpperLeftPointMtrs"])
    east, south = parse_point_mtrs(metadata_dict["LowerRightMtrs"])

    if west == east or north == south:
        raise ValueError(
            f"Invalid geometry bounds. W:{west} E:{east} N:{north} S:{south}"
        )

    return from_bounds(west, south, east, north, cols, rows)


def _expand_qf_for_shape(qf_1km, target_shape):
    """Expand 1 km QA to match 500 m shape (2x in each axis)."""
    if qf_1km.shape == target_shape:
        return qf_1km

    if target_shape[0] == qf_1km.shape[0] * 2 and target_shape[1] == qf_1km.shape[1] * 2:
        return np.repeat(np.repeat(qf_1km, 2, axis=0), 2, axis=1)

    raise ValueError(
        f"Unsupported QA reshape from {qf_1km.shape} to {target_shape}."
    )


def apply_viirs_sr_qa_mask(data, qf1, qf2, qf3, qf4, qf5, qf6, qf7, band_key):
    """
    Apply VIIRS SR QA mask.

    Common invalid conditions:
    - Cloud confidence: probably/confident cloudy
    - Night pixels
    - Low sun
    - Heavy aerosol, cloud shadow
    - Cirrus or adjacent-to-cloud contamination

        Band-specific quality:
        - Uses bad SDR bits in QF3/QF4
        - QF5/QF6 overall-quality masking is intentionally disabled for Greenland
            because it can remove almost all ice pixels. The original strict logic is
            kept below as commented lines for future restoration.
    """
    band_qf1 = _expand_qf_for_shape(qf1, data.shape)
    band_qf2 = _expand_qf_for_shape(qf2, data.shape)
    band_qf3 = _expand_qf_for_shape(qf3, data.shape)
    band_qf4 = _expand_qf_for_shape(qf4, data.shape)
    # Strict overall-quality masks (QF5/QF6) are intentionally disabled.
    # They can flag nearly all cryosphere pixels as bad in this workflow.
    # Uncomment the following two lines to restore strict behavior.
    # band_qf5 = _expand_qf_for_shape(qf5, data.shape)
    # band_qf6 = _expand_qf_for_shape(qf6, data.shape)
    band_qf7 = _expand_qf_for_shape(qf7, data.shape)

    cloud_conf = (band_qf1 >> 2) & 0b11
    is_night = ((band_qf1 >> 4) & 0b1) == 1
    is_low_sun = ((band_qf1 >> 5) & 0b1) == 1

    heavy_aerosol = ((band_qf2 >> 4) & 0b1) == 1
    cloud_shadow = ((band_qf2 >> 3) & 0b1) == 1
    cirrus_emissive = ((band_qf2 >> 7) & 0b1) == 1
    cirrus_reflective = ((band_qf2 >> 6) & 0b1) == 1

    thin_cirrus = ((band_qf7 >> 4) & 0b1) == 1
    adjacent_cloud = ((band_qf7 >> 1) & 0b1) == 1

    invalid_common = (
        (cloud_conf >= 2)
        | is_night
        | is_low_sun
        | heavy_aerosol
        | cloud_shadow
        | cirrus_emissive
        | cirrus_reflective
        | thin_cirrus
        | adjacent_cloud
    )

    # Overall SR quality bits (0=good, 1=bad) for strict mode.
    # Uncomment this block and the mask application lines below to restore.
    # qf5_quality_bit = {
    #     "M1": 2,
    #     "M2": 3,
    #     "M3": 4,
    #     "M4": 5,
    #     "M5": 6,
    #     "M7": 7,
    # }
    # qf6_quality_bit = {
    #     "M8": 0,
    #     "M10": 1,
    #     "M11": 2,
    #     "I1": 3,
    #     "I2": 4,
    #     "I3": 5,
    # }

    # Bad SDR bits (0=good, 1=bad)
    qf3_bad_sdr_bit = {
        "M1": 0,
        "M2": 1,
        "M3": 2,
        "M4": 3,
        "M5": 4,
        "M7": 5,
        "M8": 6,
        "M10": 7,
    }
    qf4_bad_sdr_bit = {
        "M11": 0,
        "I1": 1,
        "I2": 2,
        "I3": 3,
    }

    invalid_band = np.zeros(data.shape, dtype=bool)

    # Strict QF5/QF6 overall-quality mask application (disabled by default).
    # if band_key in qf5_quality_bit:
    #     invalid_band |= ((band_qf5 >> qf5_quality_bit[band_key]) & 0b1) == 1
    # if band_key in qf6_quality_bit:
    #     invalid_band |= ((band_qf6 >> qf6_quality_bit[band_key]) & 0b1) == 1
    if band_key in qf3_bad_sdr_bit:
        invalid_band |= ((band_qf3 >> qf3_bad_sdr_bit[band_key]) & 0b1) == 1
    if band_key in qf4_bad_sdr_bit:
        invalid_band |= ((band_qf4 >> qf4_bad_sdr_bit[band_key]) & 0b1) == 1

    masked = data.copy()
    masked[invalid_common | invalid_band] = np.nan
    return masked


def collect_files_for_product(product_dir):
    files = sorted(glob.glob(os.path.join(product_dir, "*.h5")))
    rows = []
    for fpath in files:
        base = os.path.basename(fpath)
        # Example: VNP09GA.A2012019.h14v01.002.2023122182523.h5
        parts = base.split(".")
        if len(parts) < 2 or not parts[1].startswith("A"):
            continue
        year_doy = parts[1][1:]
        date = pd.to_datetime(year_doy, format="%Y%j").strftime("%Y%m%d")
        rows.append({"filepath": fpath, "date": date})

    if not rows:
        return pd.DataFrame(columns=["filepath", "date"])

    return pd.DataFrame(rows)


def process_single_date(
    date,
    daily_files,
    product,
    out_folder,
    mask_shape,
    mask_transform,
    mask_crs,
    ice_mask,
):
    try:
        daily_sum = np.zeros((N_BANDS, *mask_shape), dtype=np.float32)
        daily_count = np.zeros((N_BANDS, *mask_shape), dtype=np.float32)

        for fpath in daily_files:
            try:
                with h5py.File(fpath, "r") as f:
                    struct_text = f["HDFEOS INFORMATION"]["StructMetadata.0"][()].decode("utf-8")
                    metadata = parse_struct_metadata(struct_text)

                    # 1 km QA fields
                    qf_base = "HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields"
                    qf1 = f[f"{qf_base}/SurfReflect_QF1_1"][:]
                    qf2 = f[f"{qf_base}/SurfReflect_QF2_1"][:]
                    qf3 = f[f"{qf_base}/SurfReflect_QF3_1"][:]
                    qf4 = f[f"{qf_base}/SurfReflect_QF4_1"][:]
                    qf5 = f[f"{qf_base}/SurfReflect_QF5_1"][:]
                    qf6 = f[f"{qf_base}/SurfReflect_QF6_1"][:]
                    qf7 = f[f"{qf_base}/SurfReflect_QF7_1"][:]

                    # Process I-bands from 500 m grid
                    i_base = "HDFEOS/GRIDS/VIIRS_Grid_500m_2D/Data Fields"
                    for band_idx, band_key in enumerate(I_BANDS):
                        ds = f[f"{i_base}/SurfReflect_{band_key}_1"]
                        rows, cols = ds.shape
                        src_transform = get_eqa_transform(metadata, cols, rows)

                        fill_value = ds.attrs.get("_FillValue", -28672)
                        scale = float(ds.attrs.get("scale_factor", 1.0))
                        offset = float(ds.attrs.get("add_offset", 0.0))

                        data = ds[:].astype(np.float32)
                        data[data == fill_value] = np.nan
                        data = data * scale + offset

                        data = apply_viirs_sr_qa_mask(
                            data, qf1, qf2, qf3, qf4, qf5, qf6, qf7, band_key
                        )

                        data[(data < 0) | (data > 1)] = np.nan

                        reproj = np.full(mask_shape, np.nan, dtype=np.float32)
                        reproject(
                            source=data,
                            destination=reproj,
                            src_transform=src_transform,
                            src_crs=SINUSOIDAL_CRS,
                            dst_transform=mask_transform,
                            dst_crs=mask_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )

                        valid = ~np.isnan(reproj)
                        daily_sum[band_idx][valid] += reproj[valid]
                        daily_count[band_idx][valid] += 1

                    # Process M-bands from 1 km grid
                    m_base = "HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields"
                    for m_idx, band_key in enumerate(M_BANDS):
                        band_idx = len(I_BANDS) + m_idx
                        ds = f[f"{m_base}/SurfReflect_{band_key}_1"]
                        rows, cols = ds.shape
                        src_transform = get_eqa_transform(metadata, cols, rows)

                        fill_value = ds.attrs.get("_FillValue", -28672)
                        scale = float(ds.attrs.get("scale_factor", 0.0001))
                        offset = float(ds.attrs.get("add_offset", 0.0))

                        data = ds[:].astype(np.float32)
                        data[data == fill_value] = np.nan
                        data = data * scale + offset

                        data = apply_viirs_sr_qa_mask(
                            data, qf1, qf2, qf3, qf4, qf5, qf6, qf7, band_key
                        )

                        data[(data < 0) | (data > 1)] = np.nan

                        reproj = np.full(mask_shape, np.nan, dtype=np.float32)
                        reproject(
                            source=data,
                            destination=reproj,
                            src_transform=src_transform,
                            src_crs=SINUSOIDAL_CRS,
                            dst_transform=mask_transform,
                            dst_crs=mask_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )

                        valid = ~np.isnan(reproj)
                        daily_sum[band_idx][valid] += reproj[valid]
                        daily_count[band_idx][valid] += 1

            except Exception as exc:
                return date, False, f"Error processing {os.path.basename(fpath)}: {exc}"

        with np.errstate(divide="ignore", invalid="ignore"):
            final_daily = np.divide(
                daily_sum,
                daily_count,
                where=daily_count > 0,
                out=np.full_like(daily_sum, np.nan),
            )

        final_daily[:, ice_mask == 0] = np.nan

        out_path = os.path.join(out_folder, f"VIIRS_SR_{product}_{date}_500m.tif")
        with rio.open(
            out_path,
            "w",
            driver="GTiff",
            height=mask_shape[0],
            width=mask_shape[1],
            count=N_BANDS,
            dtype=np.float32,
            crs=mask_crs,
            transform=mask_transform,
            nodata=np.nan,
            compress="lzw",
        ) as dst:
            dst.write(final_daily)
            dst.update_tags(BAND_NAMES=",".join(BAND_NAMES))

        return date, True, None

    except Exception as exc:
        return date, False, str(exc)


def process_product(
    product,
    input_root,
    output_root,
    mask_shape,
    mask_transform,
    mask_crs,
    ice_mask,
    workers,
    max_dates,
):
    product_dir = os.path.join(input_root, product)
    if not os.path.isdir(product_dir):
        print(f"[WARN] Missing product folder: {product_dir}")
        return

    df_files = collect_files_for_product(product_dir)
    if df_files.empty:
        print(f"[WARN] No files found for {product} in {product_dir}")
        return

    unique_dates = sorted(df_files["date"].unique())
    if max_dates is not None:
        unique_dates = unique_dates[:max_dates]

    out_folder = os.path.join(output_root, product)
    os.makedirs(out_folder, exist_ok=True)

    print(f"[{product}] Processing {len(unique_dates)} dates with {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_date,
                date,
                df_files[df_files["date"] == date]["filepath"].tolist(),
                product,
                out_folder,
                mask_shape,
                mask_transform,
                mask_crs,
                ice_mask,
            ): date
            for date in unique_dates
        }

        with tqdm(total=len(unique_dates), desc=f"{product} dates") as pbar:
            for future in as_completed(futures):
                date, success, error_msg = future.result()
                if not success:
                    tqdm.write(f"[ERROR] {product} {date}: {error_msg}")
                pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build daily VIIRS surface reflectance mosaics over Greenland."
    )
    parser.add_argument(
        "--input-root",
        default="/data_3/shunan_2/AU/hsa500m/VIIRS_SR",
        help="Root folder containing VNP09GA/VJ109GA/VJ209GA subfolders.",
    )
    parser.add_argument(
        "--output-root",
        default="/data_3/shunan_2/AU/hsa500m/VIIRS_SR_mosaics",
        help="Output root. Product subfolders are created automatically.",
    )
    parser.add_argument(
        "--mask-path",
        default="/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif",
        help="PROMICE ice mask GeoTIFF path.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=list(SUPPORTED_PRODUCTS),
        choices=list(SUPPORTED_PRODUCTS),
        help="Products to process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of process workers.",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Optional limit for quick testing (per product).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    ice_mask, mask_transform, mask_crs, mask_shape = read_mask(args.mask_path)

    for product in args.products:
        process_product(
            product=product,
            input_root=args.input_root,
            output_root=args.output_root,
            mask_shape=mask_shape,
            mask_transform=mask_transform,
            mask_crs=mask_crs,
            ice_mask=ice_mask,
            workers=args.workers,
            max_dates=args.max_dates,
        )

    print("Processing complete.")


if __name__ == "__main__":
    main()
