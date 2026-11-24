"""
Build daily mosaics of GCOM-C surface reflectance over the Arctic or Antarctica.

This script processes GCOM-C/SGLI surface reflectance data (RSRFQ product) in HDF5
format and creates daily mosaics on the selected ice mask grid: EPSG:3413
or EPSG:3031.

Key operations:
- Reads multiple GCOM-C/SGLI surface reflectance bands from HDF5 files
- Applies per-band calibration (slope and offset)
- Reprojects from sinusoidal to polar stereographic projection
- Handles overlapping pixels by averaging
- Masks to the selected polar ice extent
- Exports daily composites as multi-band GeoTIFF files
- Saves reflectance multiplied by 10000 as uint16 (scale 0.0001, nodata 65535)
- Uses the input and output roots as supplied

Band order in output GeoTIFF is stored in the dataset-level tag BAND_NAMES.

Excluded bands:
  Rs_SW02  — TOA reflectance (not surface)
  Rs_VN08P — co-registered variant of Rs_VN08 (for PI01 channel)
  Rs_VN11P — co-registered variant of Rs_VN11 (for PI02 channel)
  Rs_PI01/02, Rp_PL01/02 — polarimetric; add to REFLECTANCE_BANDS if needed

Shunan Feng (shunan.feng@envs.au.dk)
"""
# %%
import argparse
import os
import glob
import numpy as np
import h5py
import pandas as pd
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import CRS
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from build_mosaic_VIIRS_SR import OUTPUT_NODATA, REGIONS, read_mask


# %%
def get_eqa_transform(geom_attrs, cols, rows):
    """Calculates the correct Affine transform using EQA metadata."""
    R = 6371007.181

    upper_lat = float(geom_attrs['Upper_left_latitude'][0])
    lower_lat = float(geom_attrs['Lower_left_latitude'][0])

    north = R * np.radians(upper_lat)
    south = R * np.radians(lower_lat)

    ref_lat = lower_lat if abs(lower_lat) < abs(upper_lat) else upper_lat

    if abs(ref_lat) == 90:
        width_m = (2 * np.pi * R) / 36
    else:
        left_lon = float(geom_attrs['Lower_left_longitude'][0])
        right_lon = float(geom_attrs['Lower_right_longitude'][0])
        west = R * np.radians(left_lon) * np.cos(np.radians(ref_lat))
        east = R * np.radians(right_lon) * np.cos(np.radians(ref_lat))
        width_m = east - west

    anchor_x = R * np.radians(float(geom_attrs['Lower_right_longitude'][0])) * np.cos(np.radians(ref_lat))
    east = anchor_x
    west = anchor_x - abs(width_m)

    if west == east or north == south:
        raise ValueError(f"Invalid geometry: W:{west} E:{east} N:{north} S:{south}")

    return from_bounds(west, south, east, north, cols, rows)


'''
QA flag bit definitions for GCOM-C RSRFQ surface reflectance bands
(Rs_VN*, Rs_SW*, Rs_PI*) — "Rs_PI,SW,VN_L3_mask*" column:

Bit  Meaning                        Mask for Rs?
    0  no data                        YES (=1 → invalid)
    1  land (0=ocean, 1=land)         NO  (ice sheet is land)
    2  coast                          NO
    3  sunglint flag (>0.005)         NO
    4  sunglint mask (>0.12)          YES
    5  snow or ice                    NO  (we want snow/ice!)
    6  cloud (single-day)             YES (masked in daily composite)
    7  probably cloud (multi-day)     YES
    8  high aerosol tau-a (>0.8)      YES
    9  saturation recovery            NO
    10  BRF samples (<=3)              NO
    11  stray light flag               NO
    12  shadow                         YES
    13  pol cloud or hi-tau            YES
    14  recovery by pre-days           NO
    15  recovery (pol)                 NO
'''


def apply_qa_mask(data, qa_flag):
    """
    Apply QA masking to a GCOM-C surface reflectance band.

    Masked conditions (bit = 1 means invalid):
        Bit 0  — no data
        Bit 4  — sunglint mask (reflectance > 0.12)
        Bit 6  — cloud (single-day)
        Bit 7  — probably cloud (multi-day composite criterion)
        Bit 8  — high aerosol optical depth (> 0.8)
        Bit 12 — shadow
        Bit 13 — polarimetric cloud or high aerosol tau
    """
    data_masked = data.copy()

    no_data       = (qa_flag & (1 << 0))  > 0
    sunglint_mask = (qa_flag & (1 << 4))  > 0
    cloud_mask    = (qa_flag & (1 << 6))  > 0
    # prob_cloud    = (qa_flag & (1 << 7))  > 0
    pol_cloud     = (qa_flag & (1 << 13)) > 0
    high_tau      = (qa_flag & (1 << 8))  > 0
    shadow        = (qa_flag & (1 << 12)) > 0

    invalid = (no_data | sunglint_mask | cloud_mask | 
                pol_cloud | high_tau | shadow)
    data_masked[invalid] = np.nan

    return data_masked


# %%
# --- Band configuration ---
# Surface reflectance bands to process.
# Rs_SW02 is TOA reflectance (excluded). Rs_VN08P/VN11P are co-registered
# duplicates (excluded). Add Rs_PI01/Rs_PI02 here if polarimetric bands are needed.
REFLECTANCE_BANDS = [
    'Rs_VN01', 'Rs_VN02', 'Rs_VN03', 'Rs_VN04', 'Rs_VN05', 'Rs_VN06',
    'Rs_VN07', 'Rs_VN08', 'Rs_VN09', 'Rs_VN10', 'Rs_VN11',
    'Rs_SW03'#, 'Rs_SW01', 'Rs_SW04', two bands are 1000m resolution so excluding for now
]
n_bands = len(REFLECTANCE_BANDS)

sinusoidal_crs = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")

def process_single_date(date, daily_files, out_folder, immask, mask_transform, mask_crs, mask_shape):
    """
    Process all files for a single date and write the output mosaic.
    
    Parameters
    ----------
    date : str
        Date string (YYYYMMDD)
    daily_files : list
        List of file paths for this date
    
    Returns
    -------
    tuple: (date, success, error_msg)
        date: date string
        success: bool
        error_msg: str or None
    """
    try:
        # Shape: (n_bands, height, width)
        daily_mosaics = np.zeros((n_bands, *mask_shape), dtype=np.float32)
        daily_counts  = np.zeros((n_bands, *mask_shape), dtype=np.float32)

        for f_path in daily_files:
            try:
                with h5py.File(f_path, 'r') as f:
                    qa_flag = f['/Image_data/QA_flag'][:]
                    geom    = f['/Geometry_data'].attrs
                    rows, cols = qa_flag.shape
                    src_transform = get_eqa_transform(geom, cols, rows)

                    for band_idx, band_name in enumerate(REFLECTANCE_BANDS):
                        band_path = f'/Image_data/{band_name}'
                        if band_path not in f:
                            continue

                        ds = f[band_path]
                        slope  = ds.attrs['Slope']
                        offset = ds.attrs['Offset']
                        data   = ds[:].astype(np.float32) * slope + offset

                        # QA masking
                        data = apply_qa_mask(data, qa_flag)

                        # Physical range check
                        data[data < 0] = np.nan
                        data[data > 1] = np.nan

                        # Reproject granule to mask grid
                        temp_reprojected = np.full(mask_shape, np.nan, dtype=np.float32)
                        reproject(
                            source=data,
                            destination=temp_reprojected,
                            src_transform=src_transform,
                            src_crs=sinusoidal_crs,
                            dst_transform=mask_transform,
                            dst_crs=mask_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )

                        valid = ~np.isnan(temp_reprojected)
                        daily_mosaics[band_idx][valid] += temp_reprojected[valid]
                        daily_counts[band_idx][valid]  += 1

            except Exception as e:
                return date, False, f"Error processing {os.path.basename(f_path)}: {e}"

        # Average overlapping pixels
        with np.errstate(divide='ignore', invalid='ignore'):
            final_daily = np.divide(
                daily_mosaics,
                daily_counts,
                where=daily_counts > 0,
                out=np.full_like(daily_mosaics, np.nan),
            )

        # Apply selected ice mask
        final_daily[:, immask == 0] = np.nan

        # Save as multi-band GeoTIFF
        out_path = os.path.join(out_folder, f"GCOMC_SR_{date}_500m.tif")
        with rio.open(
            out_path, 'w',
            driver='GTiff',
            height=mask_shape[0],
            width=mask_shape[1],
            count=n_bands,
            dtype=np.uint16,
            crs=mask_crs,
            transform=mask_transform,
            nodata=OUTPUT_NODATA,
            compress='lzw',
        ) as dst:
            for band_idx, data in enumerate(final_daily, 1):
                valid = np.isfinite(data)
                output = np.full(mask_shape, OUTPUT_NODATA, dtype=np.uint16)
                output[valid] = np.rint(data[valid] * 10000).astype(np.uint16)
                dst.write(output, band_idx)
            dst.scales = (0.0001,) * n_bands
            dst.update_tags(BAND_NAMES=','.join(REFLECTANCE_BANDS))

        return date, True, None

    except Exception as e:
        return date, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Build polar GCOM-C surface reflectance mosaics.")
    parser.add_argument("--region", required=True, choices=list(REGIONS))
    parser.add_argument("--input-root", default="/data_3/shunan_2/AU/hsa500m/GCOMC_SR",
                        help="Folder containing GCOM-C RSRFQ HDF5 files.")
    parser.add_argument("--output-root", default="/data_3/shunan_2/AU/hsa500m/GCOMC_SR_mosaics",
                        help="Output root for mosaics.")
    parser.add_argument("--mask-path", default=None, help="Override the region's ice mask.")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-dates", type=int, default=None)
    args = parser.parse_args()

    region = REGIONS[args.region]
    immask, mask_transform, mask_crs, mask_shape = read_mask(
        args.mask_path or region["mask_path"], region["epsg"], region["ice_values"]
    )
    imfiles = sorted(glob.glob(os.path.join(args.input_root, '**/*RSRFQ_*.h5'), recursive=True))
    if not imfiles:
        print(f"[WARN] No GCOM-C RSRFQ files found in {args.input_root}")
        return
    file_data = [
        {'filepath': f, 'date': os.path.basename(f).split('_')[1][:8]}
        for f in imfiles
    ]
    df_files = pd.DataFrame(file_data)
    unique_dates = sorted(df_files['date'].unique())[:args.max_dates]
    out_folder = args.output_root
    os.makedirs(out_folder, exist_ok=True)
    print(f"Processing {len(unique_dates)} dates with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_date,
                date,
                df_files[df_files['date'] == date]['filepath'].tolist(),
                out_folder, immask, mask_transform, mask_crs, mask_shape
            ): date
            for date in unique_dates
        }

        with tqdm(total=len(unique_dates), desc="Processing Days") as pbar:
            for future in as_completed(futures):
                date, success, error_msg = future.result()
                if not success:
                    tqdm.write(f"Error processing {date}: {error_msg}")
                pbar.update(1)

    print("Processing Complete.")


if __name__ == '__main__':
    main()

#%%
# testfile = imfiles[0]
# with h5py.File(testfile, 'r') as f:
#     band_path = f'/Image_data/{'Rs_VN01'}'
                

#     ds = f[band_path]
#     slope  = ds.attrs['Slope']
#     offset = ds.attrs['Offset']
#     print(f"Band: Rs_VN01, Slope: {slope}, Offset: {offset}")
    # print("Available datasets:")
    # for key in f.keys():
    #     print(f"  {key}")
    # print("\nGeometry attributes:")
    # for attr in f['/Geometry_data'].attrs:
    #     print(f"  {attr}: {f['/Geometry_data'].attrs[attr]}")
    # print("\nImage data attributes:")
    # for attr in f['/Image_data'].attrs:
    #     print(f"  {attr}: {f['/Image_data'].attrs[attr]}")
    # print("\nQA_flag attributes")
    # for attr in f['/Image_data/QA_flag'].attrs:
    #     print(f"  {attr}: {f['/Image_data/QA_flag'].attrs[attr]}")

# %%
# file_path = imfiles[0]
# with h5py.File(file_path, 'r') as f:
#     # List all items in the root
#     print("Root items:", list(f.keys()))
    
#     # List all datasets under Image_data
#     if 'Image_data' in f:
#         print("\nDatasets in Image_data:")
#         for key in f['/Image_data'].keys():
#             ds = f[f'/Image_data/{key}']
#             # Printing the name and shape helps identify 250m vs 1km bands
#             print(f" - {key:15} Shape: {ds.shape}")
# %%
