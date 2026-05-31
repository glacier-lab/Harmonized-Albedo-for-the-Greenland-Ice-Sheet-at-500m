"""
Inspect and test one VIIRS SR HDF5 file.

This script helps validate VIIRS SR file structure, QA bit fields, and one-file
reprojection workflow before running full mosaics.

What it does:
- Prints HDF5 groups/datasets for key VIIRS SR fields.
- Prints attributes and scale/fill metadata for selected reflectance bands.
- Summarizes QA bit statistics from SurfReflect_QF1_1 ... SurfReflect_QF7_1.
- Optionally writes one orthoreprojected test GeoTIFF for quick validation.

Example:
python src/data_acquisition/test_viirs_sr_single_file.py \
  --file /data_3/shunan_2/AU/hsa500m/VIIRS_SR/VNP09GA/VNP09GA.A2012019.h14v01.002.2023122182523.h5 \
  --write-test-tif \
  --out-tif /tmp/VIIRS_SR_single_test.tif
"""

import argparse
import os
import glob

import h5py
import numpy as np
import rasterio as rio
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling


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
    metadata = {}
    for line in struct_text.split("\n"):
        line = line.strip()
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip().strip('"')
    return metadata


def parse_point_mtrs(point_string):
    point_string = point_string.strip().strip("()")
    x_str, y_str = point_string.split(",")
    return float(x_str), float(y_str)


def get_eqa_transform(metadata_dict, cols, rows):
    west, north = parse_point_mtrs(metadata_dict["UpperLeftPointMtrs"])
    east, south = parse_point_mtrs(metadata_dict["LowerRightMtrs"])
    return from_bounds(west, south, east, north, cols, rows)


def bit_fraction(bit_array, bit_idx):
    return float((((bit_array >> bit_idx) & 1) == 1).mean())


def print_qa_summary(qf1, qf2, qf3, qf4, qf5, qf6, qf7):
    print("\n=== QA SUMMARY (fraction of pixels) ===")

    cloud_conf = (qf1 >> 2) & 0b11
    print(f"Cloud probably/confident cloudy (QF1 bits2-3 >= 2): {(cloud_conf >= 2).mean():.4f}")
    print(f"Night (QF1 bit4): {bit_fraction(qf1, 4):.4f}")
    print(f"Low sun (QF1 bit5): {bit_fraction(qf1, 5):.4f}")

    print(f"Shadow (QF2 bit3): {bit_fraction(qf2, 3):.4f}")
    print(f"Heavy aerosol (QF2 bit4): {bit_fraction(qf2, 4):.4f}")
    print(f"Thin cirrus reflective (QF2 bit6): {bit_fraction(qf2, 6):.4f}")
    print(f"Thin cirrus emissive (QF2 bit7): {bit_fraction(qf2, 7):.4f}")

    print(f"Bad M1 SDR (QF3 bit0): {bit_fraction(qf3, 0):.4f}")
    print(f"Bad M7 SDR (QF3 bit5): {bit_fraction(qf3, 5):.4f}")
    print(f"Bad M8 SDR (QF3 bit6): {bit_fraction(qf3, 6):.4f}")
    print(f"Bad M10 SDR (QF3 bit7): {bit_fraction(qf3, 7):.4f}")

    print(f"Bad M11 SDR (QF4 bit0): {bit_fraction(qf4, 0):.4f}")
    print(f"Bad I1 SDR (QF4 bit1): {bit_fraction(qf4, 1):.4f}")
    print(f"Bad I2 SDR (QF4 bit2): {bit_fraction(qf4, 2):.4f}")
    print(f"Bad I3 SDR (QF4 bit3): {bit_fraction(qf4, 3):.4f}")

    print(f"M1 overall bad (QF5 bit2): {bit_fraction(qf5, 2):.4f}")
    print(f"M5 overall bad (QF5 bit6): {bit_fraction(qf5, 6):.4f}")
    print(f"M7 overall bad (QF5 bit7): {bit_fraction(qf5, 7):.4f}")

    print(f"M8 overall bad (QF6 bit0): {bit_fraction(qf6, 0):.4f}")
    print(f"M10 overall bad (QF6 bit1): {bit_fraction(qf6, 1):.4f}")
    print(f"M11 overall bad (QF6 bit2): {bit_fraction(qf6, 2):.4f}")
    print(f"I1 overall bad (QF6 bit3): {bit_fraction(qf6, 3):.4f}")
    print(f"I2 overall bad (QF6 bit4): {bit_fraction(qf6, 4):.4f}")
    print(f"I3 overall bad (QF6 bit5): {bit_fraction(qf6, 5):.4f}")

    print(f"Thin cirrus flag (QF7 bit4): {bit_fraction(qf7, 4):.4f}")
    print(f"Adjacent to cloud (QF7 bit1): {bit_fraction(qf7, 1):.4f}")


def inspect_file(file_path):
    print(f"Inspecting: {file_path}")

    with h5py.File(file_path, "r") as f:
        print("\nRoot keys:")
        print(list(f.keys()))

        grids = list(f["HDFEOS"]["GRIDS"].keys())
        print("\nHDFEOS/GRIDS:")
        print(grids)

        for grid in grids:
            fields = list(f["HDFEOS"]["GRIDS"][grid]["Data Fields"].keys())
            print(f"\n{grid} Data Fields ({len(fields)}):")
            print(fields)

        struct_text = f["HDFEOS INFORMATION"]["StructMetadata.0"][()].decode("utf-8")
        metadata = parse_struct_metadata(struct_text)
        print("\nStructMetadata summary:")
        for k in ["UpperLeftPointMtrs", "LowerRightMtrs", "XDim", "YDim"]:
            if k in metadata:
                print(f"  {k}: {metadata[k]}")

        for band_path in [
            "HDFEOS/GRIDS/VIIRS_Grid_500m_2D/Data Fields/SurfReflect_I1_1",
            "HDFEOS/GRIDS/VIIRS_Grid_500m_2D/Data Fields/SurfReflect_I2_1",
            "HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields/SurfReflect_M1_1",
            "HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields/SurfReflect_M5_1",
        ]:
            ds = f[band_path]
            print(f"\n{band_path}")
            print(f"  shape={ds.shape}, dtype={ds.dtype}")
            print(f"  scale_factor={ds.attrs.get('scale_factor')}")
            print(f"  add_offset={ds.attrs.get('add_offset')}")
            print(f"  _FillValue={ds.attrs.get('_FillValue')}")
            print(f"  valid_range={ds.attrs.get('valid_range')}")

        qf_base = "HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields"
        qf1 = f[f"{qf_base}/SurfReflect_QF1_1"][:]
        qf2 = f[f"{qf_base}/SurfReflect_QF2_1"][:]
        qf3 = f[f"{qf_base}/SurfReflect_QF3_1"][:]
        qf4 = f[f"{qf_base}/SurfReflect_QF4_1"][:]
        qf5 = f[f"{qf_base}/SurfReflect_QF5_1"][:]
        qf6 = f[f"{qf_base}/SurfReflect_QF6_1"][:]
        qf7 = f[f"{qf_base}/SurfReflect_QF7_1"][:]

        print("\nQF dataset dtypes and value ranges:")
        for name, qf in [
            ("QF1", qf1),
            ("QF2", qf2),
            ("QF3", qf3),
            ("QF4", qf4),
            ("QF5", qf5),
            ("QF6", qf6),
            ("QF7", qf7),
        ]:
            print(f"  {name}: dtype={qf.dtype}, min={qf.min()}, max={qf.max()}")

        print_qa_summary(qf1, qf2, qf3, qf4, qf5, qf6, qf7)



def write_test_tif(file_path, out_tif, mask_path):
    print("\nWriting one-file reprojection test GeoTIFF...")

    ice_mask, mask_transform, mask_crs, mask_shape = read_mask(mask_path)

    with h5py.File(file_path, "r") as f:
        struct_text = f["HDFEOS INFORMATION"]["StructMetadata.0"][()].decode("utf-8")
        metadata = parse_struct_metadata(struct_text)

        i1 = f["HDFEOS/GRIDS/VIIRS_Grid_500m_2D/Data Fields/SurfReflect_I1_1"]
        m5 = f["HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields/SurfReflect_M5_1"]

        i1_arr = i1[:].astype(np.float32)
        i1_arr[i1_arr == i1.attrs.get("_FillValue", -28672)] = np.nan
        i1_arr = i1_arr * float(i1.attrs.get("scale_factor", 1.0)) + float(i1.attrs.get("add_offset", 0.0))

        m5_arr = m5[:].astype(np.float32)
        m5_arr[m5_arr == m5.attrs.get("_FillValue", -28672)] = np.nan
        m5_arr = m5_arr * float(m5.attrs.get("scale_factor", 1.0)) + float(m5.attrs.get("add_offset", 0.0))

        i1_arr[(i1_arr < 0) | (i1_arr > 1)] = np.nan
        m5_arr[(m5_arr < 0) | (m5_arr > 1)] = np.nan

        i_rows, i_cols = i1_arr.shape
        m_rows, m_cols = m5_arr.shape
        i_transform = get_eqa_transform(metadata, i_cols, i_rows)
        m_transform = get_eqa_transform(metadata, m_cols, m_rows)

        i1_reproj = np.full(mask_shape, np.nan, dtype=np.float32)
        m5_reproj = np.full(mask_shape, np.nan, dtype=np.float32)

        reproject(
            source=i1_arr,
            destination=i1_reproj,
            src_transform=i_transform,
            src_crs=SINUSOIDAL_CRS,
            dst_transform=mask_transform,
            dst_crs=mask_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        reproject(
            source=m5_arr,
            destination=m5_reproj,
            src_transform=m_transform,
            src_crs=SINUSOIDAL_CRS,
            dst_transform=mask_transform,
            dst_crs=mask_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        i1_reproj[ice_mask == 0] = np.nan
        m5_reproj[ice_mask == 0] = np.nan

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    with rio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=mask_shape[0],
        width=mask_shape[1],
        count=2,
        dtype=np.float32,
        crs=mask_crs,
        transform=mask_transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(i1_reproj, 1)
        dst.write(m5_reproj, 2)
        dst.set_band_description(1, "SurfReflect_I1")
        dst.set_band_description(2, "SurfReflect_M5")

    print(f"Test GeoTIFF written: {out_tif}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one VIIRS SR file and run one-file test.")
    parser.add_argument(
        "--file",
        default=None,
        help="Path to one VIIRS SR .h5 file. If omitted, a sample file is auto-selected.",
    )
    parser.add_argument(
        "--input-root",
        default="/data_3/shunan_2/AU/hsa500m/VIIRS_SR",
        help="Root folder containing VNP09GA/VJ109GA/VJ209GA subfolders for auto file selection.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=["VNP09GA", "VJ109GA", "VJ209GA"],
        help="Product search order when --file is omitted.",
    )
    parser.add_argument(
        "--mask-path",
        default="/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif",
        help="PROMICE ice mask path",
    )
    parser.add_argument(
        "--write-test-tif",
        action="store_true",
        help="Write one test orthoreprojected GeoTIFF (I1 + M5)",
    )
    parser.add_argument(
        "--out-tif",
        default="/tmp/VIIRS_SR_single_test.tif",
        help="Output path for the one-file test GeoTIFF",
    )
    return parser.parse_args()


def resolve_input_file(cli_file, input_root, products):
    if cli_file:
        if not os.path.exists(cli_file):
            raise FileNotFoundError(f"Input file not found: {cli_file}")
        return cli_file

    for product in products:
        pattern = os.path.join(input_root, product, "*.h5")
        candidates = sorted(glob.glob(pattern))
        if candidates:
            selected = candidates[0]
            print(f"No --file provided. Using sample: {selected}")
            return selected

    raise FileNotFoundError(
        "No VIIRS SR .h5 files found for auto-selection. "
        f"Checked products {products} under {input_root}."
    )


def main():
    args = parse_args()
    input_file = resolve_input_file(args.file, args.input_root, args.products)
    inspect_file(input_file)

    if args.write_test_tif:
        write_test_tif(input_file, args.out_tif, args.mask_path)


if __name__ == "__main__":
    main()
