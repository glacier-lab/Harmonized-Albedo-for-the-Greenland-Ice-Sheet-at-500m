"""
Build daily mosaics of GCOM-C albedo images.
Processes GCOM-C albedo data (HDF5) to EPSG:3413 (Greenland) at 500m.
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
                
                # Apply scale and offset
                slope = albedo_ds.attrs['Slope']
                offset = albedo_ds.attrs['Offset']
                data = albedo_ds[:].astype(np.float32) * slope + offset
                
                # Basic QA Mask (example: mask out values using QA_flag if needed)
                # data[qa_flag != 0] = np.nan 
                data[data <= 0] = np.nan # Mask fill values
                data[data >= 1] = np.nan  # Mask invalid albedo values
                
                # 2. Get Geolocation Attributes
                geom = f['/Geometry_data'].attrs
                l_lat = float(geom['Lower_left_latitude'][0])
                l_lon = float(geom['Lower_left_longitude'][0])
                u_lat = float(geom['Upper_right_latitude'][0])
                u_lon = float(geom['Upper_right_longitude'][0])
                rows, cols = data.shape
                
                # 3. Define Source Transform (WGS84)
                # Note: 'u_lat' is usually top, 'l_lat' is bottom. 
                src_transform = from_bounds(l_lon, l_lat, u_lon, u_lat, cols, rows)
                src_crs = CRS.from_epsg(4326)

                # 4. Reproject this granule to the mask grid
                temp_reprojected = np.full(mask_shape, np.nan, dtype=np.float32)
                
                reproject(
                    source=data,
                    destination=temp_reprojected,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=mask_transform,
                    dst_crs=mask_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan
                )

                # 5. Add to daily aggregate
                mask_valid = ~np.isnan(temp_reprojected)
                daily_mosaic[mask_valid] += temp_reprojected[mask_valid]
                daily_count[mask_valid] += 1
                
        except Exception as e:
            print(f"Error processing {f_path}: {e}")

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