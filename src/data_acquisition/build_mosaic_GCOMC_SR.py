"""
Build daily mosaics of GCOM-C surface reflectance images over Greenland.

This script processes GCOM-C/SGLI surface reflectance data (RSRFQ product) in HDF5
format and creates daily mosaics reprojected to EPSG:3413 (NSIDC Polar Stereographic
North) at 500m resolution.

Key operations:
- Reads multiple GCOM-C/SGLI surface reflectance bands from HDF5 files
- Applies per-band calibration (slope and offset)
- Reprojects from sinusoidal to polar stereographic projection
- Handles overlapping pixels by averaging
- Masks to Greenland ice sheet extent using PROMICE mask
- Exports daily composites as multi-band GeoTIFF files

Band order in output GeoTIFF is stored in the dataset-level tag BAND_NAMES.

Excluded bands:
  Rs_SW02  — TOA reflectance (not surface)
  Rs_VN08P — co-registered variant of Rs_VN08 (for PI01 channel)
  Rs_VN11P — co-registered variant of Rs_VN11 (for PI02 channel)
  Rs_PI01/02, Rp_PL01/02 — polarimetric; add to REFLECTANCE_BANDS if needed

Shunan Feng (shunan.feng@envs.au.dk)
"""
# %%
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
from affine import Affine


# %%
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

# --- Setup Paths ---
im_mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"
imfolder     = '/data_3/shunan_2/AU/hsa500m/GCOMC_SR/'
out_folder   = '/data_3/shunan_2/AU/hsa500m/GCOMC_SR_mosaics/'
os.makedirs(out_folder, exist_ok=True)

# --- Load Mask ---
immask, mask_transform, mask_crs, mask_shape = read_mask(im_mask_path)

# --- Organize Files by Date ---
# Filename example: GC1SG1_20180103A01D_T0015_L2SG_RSRFQ_3001.h5
imfiles = sorted(glob.glob(os.path.join(imfolder, '**/*RSRFQ_*.h5'), recursive=True))
file_data = []
for f in imfiles:
    date_str = os.path.basename(f).split('_')[1][:8]  # YYYYMMDD
    file_data.append({'filepath': f, 'date': date_str})

df_files = pd.DataFrame(file_data)
unique_dates = df_files['date'].unique()
sinusoidal_crs = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")

NUM_WORKERS = 10


def process_single_date(date, daily_files):
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

        # Apply PROMICE ice mask
        final_daily[:, immask == 0] = np.nan

        # Save as multi-band GeoTIFF
        out_path = os.path.join(out_folder, f"GCOMC_SR_{date}_500m.tif")
        with rio.open(
            out_path, 'w',
            driver='GTiff',
            height=mask_shape[0],
            width=mask_shape[1],
            count=n_bands,
            dtype=np.float32,
            crs=mask_crs,
            transform=mask_transform,
            nodata=np.nan,
            compress='lzw',
        ) as dst:
            dst.write(final_daily)
            dst.update_tags(BAND_NAMES=','.join(REFLECTANCE_BANDS))

        return date, True, None

    except Exception as e:
        return date, False, str(e)


# --- Processing Loop with Parallelization ---
print(f"Processing {len(unique_dates)} dates with {NUM_WORKERS} workers...")

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(
            process_single_date,
            date,
            df_files[df_files['date'] == date]['filepath'].tolist()
        ): date
        for date in unique_dates
    }

    with tqdm(total=len(unique_dates), desc="Processing Days") as pbar:
        for future in as_completed(futures):
            date, success, error_msg = future.result()

            if not success:
                tqdm.write(f"✗ Error processing {date}: {error_msg}")

            pbar.update(1)

print("Processing Complete.")

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
