"""
Build daily mosaics of GCOM-C albedo images over Greenland.

This script processes GCOM-C/SGLI albedo data in HDF5 format and creates daily
mosaics reprojected to EPSG:3413 (NSIDC Polar Stereographic North) at 500m resolution.

Key operations:
- Reads GCOM-C albedo from HDF5 files
- Applies calibration (slope and offset)
- Reprojects from sinusoidal to polar stereographic projection
- Handles overlapping pixels by averaging
- Masks to Greenland ice sheet extent using PROMICE mask
- Exports daily composites as GeoTIFF files

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

#%%
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
        new_transform = rio.windows.transform(
            rio.windows.Window(col_start, row_start, col_end-col_start, row_end-row_start), 
            transform
        )
        # binary mask: 1 for ice, 0 for land/ocean
        mask_cropped = np.where(mask_cropped <= 0, 0, 1)
    
    return mask_cropped, new_transform, crs, mask_cropped.shape
def get_eqa_transform(geom_attrs, cols, rows):
    """
    Calculates the correct Affine transform using EQA metadata.
    """
    R = 6371007.181
    
    # 1. Get latitudes
    upper_lat = float(geom_attrs['Upper_left_latitude'][0])
    lower_lat = float(geom_attrs['Lower_left_latitude'][0])
    
    # 2. Calculate Y (North/South) - this is always safe
    north = R * np.radians(upper_lat)
    south = R * np.radians(lower_lat)
    
    # 3. Calculate X (East/West)
    # To avoid the "North Pole 0 width" error, we calculate width at 
    # the latitude furthest from the pole.
    ref_lat = lower_lat if abs(lower_lat) < abs(upper_lat) else upper_lat
    
    # Check for the extreme case (if both are 90, which shouldn't happen)
    if abs(ref_lat) == 90:
        # Fallback to the standard 10-degree tile width in meters
        width_m = (2 * np.pi * R) / 36   # 360/10 = 36 tiles globally
    else:
        left_lon = float(geom_attrs['Lower_left_longitude'][0])
        right_lon = float(geom_attrs['Lower_right_longitude'][0])
        
        # Calculate width in meters at the reference latitude
        west = R * np.radians(left_lon) * np.cos(np.radians(ref_lat))
        east = R * np.radians(right_lon) * np.cos(np.radians(ref_lat))
        width_m = east - west

    # 4. Re-calculate west/east based on the constant width
    # This ensures the transform is a perfect square
    # We use the right_lon (usually 0 or fixed) as the anchor
    anchor_x = R * np.radians(float(geom_attrs['Lower_right_longitude'][0])) * np.cos(np.radians(ref_lat))
    east = anchor_x
    west = anchor_x - abs(width_m)

    # Sanity check to prevent "Cannot invert geotransform"
    if west == east or north == south:
        raise ValueError(f"Invalid geometry: Bounds are zero. W:{west} E:{east} N:{north} S:{south}")

    return from_bounds(west, south, east, north, cols, rows)

'''
Notes on QA band:
Data_description: 
[b'Bit-0:no input data, 
1:land/water flag 
2: cloudy/clear flag, 
3:day/night(shadow) flag, 
4-6:snow over land or seaice, 
snow mixed w/t vegetation or bare ice, melting snow over land or seaice, (111) no snow, 
7-9:stray light correction (VN,SW,IR), 
10:radiance saturation, 
11:sun-glint area, 
12-14:missing channel(VN,SW,IR), 
15:(reserved)']
'''
def apply_qa_mask(albedo_data, qa_flag):
    """
    Apply QA masking to GCOM-C albedo data.
    
    QA Flag Bit Structure:
    - Bit 0: No input data flag (1 = invalid)
    - Bit 1: Land/water flag (not critical, but can filter if needed)
    - Bit 2: Cloudy/clear flag (1 = cloudy, 0 = clear)
    - Bit 3: Day/night (shadow) flag (1 = night/shadow, 0 = day)
    - Bits 4-6: Snow conditions (for snow over land/ice)
    - Bits 7-9: Stray light correction status
    - Bit 10: Radiance saturation (1 = saturated)
    - Bit 11: Sun-glint area (1 = glint)
    - Bits 12-14: Missing channel flags
    - Bit 15: Reserved
    
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
    
    # Extract individual bit flags
    no_input = (qa_flag & 1) > 0                    # Bit 0
    cloudy = (qa_flag & (1 << 2)) > 0              # Bit 2
    night_shadow = (qa_flag & (1 << 3)) > 0        # Bit 3
    radiance_saturated = (qa_flag & (1 << 10)) > 0  # Bit 10
    sun_glint = (qa_flag & (1 << 11)) > 0          # Bit 11
    # missing_channel = (qa_flag & (1 << 12)) > 0    # Bit 12 (check VN channel)
    any_missing_channel = (qa_flag & (7 << 12)) != 0 # Binary 111 (7) shifted left by 12 covers all three bits
    
    # Error DN values (typically 65535 for uint16)
    error_dn = qa_flag == 65535
    
    # Create mask: set to NaN for invalid pixels
    invalid_mask = (no_input | cloudy | night_shadow | radiance_saturated | 
                    sun_glint | any_missing_channel | error_dn)
    
    albedo_masked[invalid_mask] = np.nan
    
    return albedo_masked

# --- Setup Paths ---
im_mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"
imfolder = '/data_3/shunan_2/AU/hsa500m/GCOMC/'
out_folder = '/data_3/shunan_2/AU/hsa500m/GCOMC_mosaics/'
os.makedirs(out_folder, exist_ok=True)

# --- Load Mask ---
immask, mask_transform, mask_crs, mask_shape = read_mask(im_mask_path)

# --- Organize Files by Date ---
imfiles = sorted(glob.glob(os.path.join(imfolder, '**/*Q_3000.h5'), recursive=True))
file_data = []
for f in imfiles:
    # Extracts YYYYMMDD from filename
    date_str = os.path.basename(f).split('_')[1][:8]
    file_data.append({'filepath': f, 'date': date_str})

df_files = pd.DataFrame(file_data)
unique_dates = df_files['date'].unique()
sinusoidal_crs = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")

# --- Processing Loop ---
for date in tqdm(unique_dates, desc="Processing Days"):
    daily_files = df_files[df_files['date'] == date]['filepath'].tolist()
    
    # Initialize empty daily mosaic and a count array for averaging overlapping pixels
    daily_mosaic = np.zeros(mask_shape, dtype=np.float32)
    daily_count = np.zeros(mask_shape, dtype=np.float32)

    for f_path in daily_files:
        try:
            with h5py.File(f_path, 'r') as f:
                # 1. Read Data
                albedo_ds = f['/Image_data/SALB']
                qa_flag = f['/Image_data/QA_flag'][:]
                geom = f['/Geometry_data'].attrs
                
                # Apply scale and offset
                slope = albedo_ds.attrs['Slope']
                offset = albedo_ds.attrs['Offset']
                data = albedo_ds[:].astype(np.float32) * slope + offset
                
                # 2. Apply QA Mask (IMPORTANT: do this BEFORE value range checks)
                data = apply_qa_mask(data, qa_flag)
                
                # 3. Additional value range checks (after QA masking)
                data[data <= 0] = np.nan  # Mask negative/zero values
                data[data > 1] = np.nan   # Mask values > 1
                
                # 4. Get Geolocation Attributes
                rows, cols = data.shape
                
                # 5. Define Source Transform (Sinusoidal)
                src_transform = get_eqa_transform(geom, cols, rows)

                # 6. Reproject this granule to the mask grid
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
                    dst_nodata=np.nan
                )

                # 7. Add to daily aggregate
                mask_valid = ~np.isnan(temp_reprojected)
                daily_mosaic[mask_valid] += temp_reprojected[mask_valid]
                daily_count[mask_valid] += 1
                
        except Exception as e:
            tqdm.write(f"Error processing {os.path.basename(f_path)}: {e}")

    # Calculate mean for overlaps
    with np.errstate(divide='ignore', invalid='ignore'):
        final_daily = np.divide(daily_mosaic, daily_count, 
                            where=daily_count>0, 
                            out=np.full_like(daily_mosaic, np.nan))
    
    # Apply the PROMICE mask (assuming 1=ice, 0=land/ocean)
    final_daily[immask == 0] = np.nan
    
    # 6. Save as GeoTIFF
    out_path = os.path.join(out_folder, f"GCOMC_Albedo_{date}_500m.tif")
    with rio.open(
        out_path, 'w',
        driver='GTiff',
        height=mask_shape[0],
        width=mask_shape[1],
        count=1,
        dtype=np.float32,
        crs=mask_crs,
        transform=mask_transform,
        nodata=np.nan,
        compress='lzw'
    ) as dst:
        dst.write(final_daily, 1)

print("Processing Complete.")