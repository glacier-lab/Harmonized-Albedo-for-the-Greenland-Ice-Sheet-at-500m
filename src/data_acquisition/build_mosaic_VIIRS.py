"""
Build daily mosaics of VIIRS albedo images over Greenland.

This script processes VIIRS albedo data in HDF5 format and creates daily
mosaics reprojected to EPSG:3413 (NSIDC Polar Stereographic North) at 500m resolution.

Key operations:
- Reads VIIRS albedo from HDF5 files
- Apply QA masking to ensure only high-quality pixels are used
- Applies calibration (slope and offset)
- Reprojects from sinusoidal to polar stereographic projection
- Handles overlapping pixels by averaging
- Masks to Greenland ice sheet extent using PROMICE mask
- Exports daily composites as GeoTIFF files
- Saves both BS (band1) and WS (band2) shortwave albedo in separate bands of the same GeoTIFF


Shunan Feng (shunan.feng@envs.au.dk)
"""
#%%
import os
import glob
import numpy as np
import h5py
import pandas as pd
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import CRS
from tqdm import tqdm # For progress tracking
from concurrent.futures import ProcessPoolExecutor, as_completed
from affine import Affine
# from rasterio.plot import show
# %%
def read_mask(mask_path):
    with rio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
        # Crop to valid extent (removes empty borders)
        valid_rows = np.where(np.any(mask != src.nodata, axis=1))[0]
        valid_cols = np.where(np.any(mask != src.nodata, axis=0))[0]
        
        row_start, row_end = valid_rows[0], valid_rows[-1] + 1
        col_start, col_end = valid_cols[0], valid_cols[-1] + 1
        
        mask_cropped = mask[row_start:row_end, col_start:col_end]
        # Calculate new transform for cropped window
        new_transform = transform * Affine.translation(col_start, row_start)
        # binary mask: 1 for ice, 0 for land/ocean
        mask_cropped = np.where(mask_cropped <= 0, 0, 1)
    
    return mask_cropped, new_transform, crs, mask_cropped.shape

def inspect_h5_structure(filepath):
    """
    Print the internal structure of an HDF5 file to debug path issues.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\n=== HDF5 Structure for {os.path.basename(filepath)} ===")
            def print_structure(name, obj):
                print(f"  {name}")
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error inspecting file: {e}")

def get_eqa_transform(im_attrs, cols, rows):
    """
    Calculates the correct Affine transform using EQA metadata.
    For VIIRS sinusoidal grid, uses UpperLeftPointMtrs and LowerRightMtrs directly.
    """
    # Extract bounds directly from sinusoidal coordinates
    upper_left_str = im_attrs['UpperLeftPointMtrs'][0].strip('()')
    lower_right_str = im_attrs['LowerRightMtrs'][0].strip('()')
    
    upper_left = list(map(float, upper_left_str.split(',')))
    lower_right = list(map(float, lower_right_str.split(',')))
    
    west = upper_left[0]
    north = upper_left[1]
    east = lower_right[0]
    south = lower_right[1]
    
    # Sanity check
    if west == east or north == south:
        raise ValueError(f"Invalid geometry: Bounds are zero. W:{west} E:{east} N:{north} S:{south}")
    
    return from_bounds(west, south, east, north, cols, rows)


def apply_qa_mask(albedo_data, qa_flag):
    """
    Apply QA masking to VIIRS albedo data.
    
    Mandatory QA:
      0 = processed, good quality (full BRDF inversions)
      1 = processed, see other QA (magnitude BRDF inversions)
    
    Parameters
    ----------
    albedo_data : ndarray
        Albedo values (2D)
    qa_flag : ndarray
        QA flag values (2D) with same shape as albedo_data
    
    Returns
    -------
    albedo_masked : ndarray
        Albedo with QA-based masking applied
    """
    albedo_masked = albedo_data.copy()
    
    # Mask out pixels that are not "processed, good quality"
    albedo_masked[qa_flag != 0] = np.nan
    
    return albedo_masked

#%%
# --- Setup Paths ---
im_mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"
# imfolder = '/data_3/shunan_2/AU/hsa500m/VIIRS/VJ143MA3/'
# out_folder = '/data_3/shunan_2/AU/hsa500m/VIIRS_mosaics/VJ143MA3/'
imfolder = '/data_3/shunan_2/AU/hsa500m/VIIRS/VNP43MA3/'
out_folder = '/data_3/shunan_2/AU/hsa500m/VIIRS_mosaics/VNP43MA3/'
os.makedirs(out_folder, exist_ok=True)

# --- Load Mask ---
immask, mask_transform, mask_crs, mask_shape = read_mask(im_mask_path)

# --- Organize Files by Date ---
imfiles = sorted(glob.glob(os.path.join(imfolder, '**/*.h5'), recursive=True))
imdates = [os.path.basename(f).split('.')[1] for f in imfiles]
imdates = [d[1:8] for d in imdates]
imdates = pd.to_datetime(imdates, format='%Y%j').strftime('%Y%m%d')
df_files = pd.DataFrame({'filepath': imfiles, 'date': imdates})
unique_dates = np.unique(df_files['date'])

sinusoidal_crs = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")
NUM_WORKERS = 10

def process_single_date(date, daily_files):
    """Process all files for one date and write one output mosaic."""
    try:
        daily_mosaic_bs = np.zeros(mask_shape, dtype=np.float32)
        daily_mosaic_ws = np.zeros(mask_shape, dtype=np.float32)
        daily_count_bs = np.zeros(mask_shape, dtype=np.float32)
        daily_count_ws = np.zeros(mask_shape, dtype=np.float32)

        for f_path in daily_files:
            try:
                with h5py.File(f_path, 'r') as f:
                    required_paths = [
                        'HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/Albedo_BSA_shortwave',
                        'HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/Albedo_WSA_shortwave',
                        'HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/BRDF_Albedo_Band_Mandatory_Quality_shortwave',
                        'HDFEOS INFORMATION/StructMetadata.0'
                    ]
                    missing_paths = [p for p in required_paths if p not in f]
                    if missing_paths:
                        continue

                    bs_albedo = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['Albedo_BSA_shortwave']
                    ws_albedo = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['Albedo_WSA_shortwave']
                    qa_flag = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['BRDF_Albedo_Band_Mandatory_Quality_shortwave']
                    im_metadata = f['HDFEOS INFORMATION']['StructMetadata.0'][()].decode('utf-8')

                    metadata_dict = {}
                    for line in im_metadata.split('\n'):
                        line = line.strip()
                        if '=' in line:
                            key, value = line.split('=', 1)
                            metadata_dict[key.strip()] = value.strip().strip('"').replace('(', '').replace(')', '')
                    df_metadata = pd.DataFrame([metadata_dict])

                    slope_bs = bs_albedo.attrs['scale_factor']
                    offset_bs = bs_albedo.attrs['add_offset']
                    bs_albedo = bs_albedo[:].astype(np.float32) * slope_bs + offset_bs

                    slope_ws = ws_albedo.attrs['scale_factor']
                    offset_ws = ws_albedo.attrs['add_offset']
                    ws_albedo = ws_albedo[:].astype(np.float32) * slope_ws + offset_ws

                    # bs_albedo_masked = apply_qa_mask(bs_albedo, qa_flag[:])
                    # ws_albedo_masked = apply_qa_mask(ws_albedo, qa_flag[:])
                    bs_albedo_masked = bs_albedo
                    ws_albedo_masked = ws_albedo

                    bs_albedo_masked[bs_albedo_masked <= 0] = np.nan
                    bs_albedo_masked[bs_albedo_masked >= 1] = np.nan
                    ws_albedo_masked[ws_albedo_masked <= 0] = np.nan
                    ws_albedo_masked[ws_albedo_masked >= 1] = np.nan

                    rows, cols = int(df_metadata.YDim[0]), int(df_metadata.XDim[0])
                    src_transform = get_eqa_transform(df_metadata, cols, rows)

                    temp_reprojected_bs = np.full(mask_shape, np.nan, dtype=np.float32)
                    temp_reprojected_ws = np.full(mask_shape, np.nan, dtype=np.float32)

                    reproject(
                        source=bs_albedo_masked,
                        destination=temp_reprojected_bs,
                        src_transform=src_transform,
                        src_crs=sinusoidal_crs,
                        dst_transform=mask_transform,
                        dst_crs=mask_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=np.nan,
                        dst_nodata=np.nan
                    )

                    reproject(
                        source=ws_albedo_masked,
                        destination=temp_reprojected_ws,
                        src_transform=src_transform,
                        src_crs=sinusoidal_crs,
                        dst_transform=mask_transform,
                        dst_crs=mask_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=np.nan,
                        dst_nodata=np.nan
                    )

                    mask_valid_bs = ~np.isnan(temp_reprojected_bs)
                    mask_valid_ws = ~np.isnan(temp_reprojected_ws)

                    daily_mosaic_bs[mask_valid_bs] += temp_reprojected_bs[mask_valid_bs]
                    daily_mosaic_ws[mask_valid_ws] += temp_reprojected_ws[mask_valid_ws]
                    daily_count_bs[mask_valid_bs] += 1
                    daily_count_ws[mask_valid_ws] += 1

            except Exception as e:
                return date, False, f"Error processing {os.path.basename(f_path)}: {e}"

        with np.errstate(divide='ignore', invalid='ignore'):
            final_bs = np.divide(
                daily_mosaic_bs,
                daily_count_bs,
                where=daily_count_bs > 0,
                out=np.full_like(daily_mosaic_bs, np.nan),
            )
            final_ws = np.divide(
                daily_mosaic_ws,
                daily_count_ws,
                where=daily_count_ws > 0,
                out=np.full_like(daily_mosaic_ws, np.nan),
            )

        final_bs[immask == 0] = np.nan
        final_ws[immask == 0] = np.nan

        out_path = os.path.join(out_folder, f"VIIRS_Albedo_{date}_500m.tif")
        with rio.open(
            out_path, 'w',
            driver='GTiff',
            height=mask_shape[0],
            width=mask_shape[1],
            count=2,
            dtype=np.float32,
            crs=mask_crs,
            transform=mask_transform,
            nodata=np.nan,
            compress='lzw'
        ) as dst:
            dst.write(final_bs, 1)
            dst.write(final_ws, 2)
            dst.set_band_description(1, 'BSA_shortwave')
            dst.set_band_description(2, 'WSA_shortwave')

        return date, True, None

    except Exception as e:
        return date, False, str(e)


def main():
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
                    tqdm.write(f"Error processing {date}: {error_msg}")
                pbar.update(1)

    print("Processing Complete.")


if __name__ == '__main__':
    main()

#%%
# testfile = imfiles[11]
# with h5py.File(testfile, 'r') as f:
#     # print("Root Keys: %s" % list(f.keys()))
#     im_metadata = f['HDFEOS INFORMATION']['StructMetadata.0'][()].decode('utf-8')
#     # Parse metadata string into key-value pairs
#     bs_albedo = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['Albedo_BSA_shortwave']
#     ws_albedo = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['Albedo_WSA_shortwave']
#     qa_flag = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['BRDF_Albedo_Band_Mandatory_Quality_shortwave']
#     im_metadata = f['HDFEOS INFORMATION']['StructMetadata.0'][()].decode('utf-8')
    
#     # Parse metadata string into key-value pairs
#     metadata_dict = {}
#     for line in im_metadata.split('\n'):
#         line = line.strip()
#         if '=' in line:
#             key, value = line.split('=', 1)
#             metadata_dict[key.strip()] = value.strip().strip('"').replace('(', '').replace(')', '')
#     # Convert to DataFrame
#     df_metadata = pd.DataFrame([metadata_dict])

#     # Apply scale and offset
#     slope_bs = bs_albedo.attrs['scale_factor']
#     offset_bs = bs_albedo.attrs['add_offset']
#     bs_albedo = bs_albedo[:].astype(np.float32) * slope_bs + offset_bs

#     slope_ws = ws_albedo.attrs['scale_factor']
#     offset_ws = ws_albedo.attrs['add_offset']
#     ws_albedo = ws_albedo[:].astype(np.float32) * slope_ws + offset_ws

#     # 2. Apply QA Mask (IMPORTANT: do this BEFORE value range checks)
#     bs_albedo_masked = apply_qa_mask(bs_albedo, qa_flag[:])
#     ws_albedo_masked = apply_qa_mask(ws_albedo, qa_flag[:])
    
#     # 3. Additional value range checks (after QA masking)
#     bs_albedo_masked[bs_albedo_masked <= 0] = np.nan
#     bs_albedo_masked[bs_albedo_masked >= 1] = np.nan
#     ws_albedo_masked[ws_albedo_masked <= 0] = np.nan
#     ws_albedo_masked[ws_albedo_masked >= 1] = np.nan
    
#     # 4. Get Geolocation Attributes
#     rows, cols = int(df_metadata.YDim[0]), int(df_metadata.XDim[0])
    
#     # 5. Define Source Transform (Sinusoidal)
#     src_transform = get_eqa_transform(df_metadata, cols, rows)

    
    # print(list(f['HDFEOS'])) 
    # print(list(f['HDFEOS']['GRIDS'])) 
    # print(list(f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF'])) 
    # print(list(f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields'])) 
    # print(list(f['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'])) # is empty
    # SouthBoundingCoord = f.attrs['SouthBoundingCoord']
# %%
