"""
Rebuild SICE albedo products from TOA reflectance bands.

This script calculates broadband albedo from SICE TOA reflectance bands
(01, 06, 17, 21) and saves as GeoTIFF files at 500m resolution.

Formula: albedo = 1.003 * mean(band01, band06, band17, band21) + 0.058

Author: Shunan Feng (shunan.feng@envs.au.dk)
"""
#%%
import rasterio as rio
from rasterio.transform import from_bounds
import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
# Configuration
sice_path = '/data_3/shunan_2/AU/hsa500m/SICE'
sice_rebuild_path = '/data_3/shunan_2/AU/hsa500m/SICE_rebuild'
os.makedirs(sice_rebuild_path, exist_ok=True)

# Albedo coefficients
a = 1.003
b = 0.058

# %%
# Find all SICE files
sice_files = sorted(glob.glob(os.path.join(sice_path, '**/*.nc'), recursive=True))

print(f"\n{'='*60}")
print("SICE Albedo Rebuild")
print(f"{'='*60}")
print(f"Found {len(sice_files)} SICE files to process\n")

if len(sice_files) == 0:
    print("No files found!")
    exit()

# %%
# Process all files with progress bar
processed = 0
failed = 0
skipped = 0

for file_path in tqdm(sice_files, desc="Processing Files", unit="file"):
    imname = os.path.basename(file_path)
    
    try:
        # Parse date from filename
        imtime = os.path.splitext(imname)[0].split('_')[-1]
        imtime = pd.to_datetime(imtime)
        
        # Check if output already exists
        out_path = os.path.join(sice_rebuild_path, f'SICE_Albedo_{imtime.strftime("%Y%m%d")}_500m.tif')
        if os.path.exists(out_path):
            skipped += 1
            tqdm.write(f"Skipping {imname}")
            continue
        
        # Open dataset
        ds_sice = xr.open_dataset(file_path)
        
        # Check if required bands exist
        required_bands = ['r_TOA_01', 'r_TOA_06', 'r_TOA_17', 'r_TOA_21']
        missing_bands = [band for band in required_bands if band not in ds_sice.variables]
        
        if missing_bands:
            tqdm.write(f"✗ {imname}: Missing bands {missing_bands}")
            ds_sice.close()
            failed += 1
            continue
        
        # Extract TOA reflectance bands
        band01 = ds_sice['r_TOA_01'].values
        band06 = ds_sice['r_TOA_06'].values
        band17 = ds_sice['r_TOA_17'].values
        band21 = ds_sice['r_TOA_21'].values
        
        # Calculate broadband albedo
        ds_albedo = a * ((band01 + band06 + band17 + band21) / 4) + b
        
        # Get coordinates
        x_coord = ds_sice['x'].values
        y_coord = ds_sice['y'].values
        
        # Get dimensions
        height, width = ds_albedo.shape
        
        # Create transform
        transform = from_bounds(
            west=x_coord.min(),
            south=y_coord.min(),
            east=x_coord.max(),
            north=y_coord.max(),
            width=width,
            height=height
        )
        
        # Save as GeoTIFF
        with rio.open(
            out_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=ds_albedo.dtype,
            crs='EPSG:3413',
            transform=transform,
            compress='lzw',
            nodata=np.nan
        ) as dst:
            dst.write(ds_albedo, 1)
        
        ds_sice.close()
        
        # Calculate statistics
        valid_pixels = np.sum(~np.isnan(ds_albedo))
        coverage = valid_pixels / ds_albedo.size * 100
        
        tqdm.write(f"✓ {imname} (coverage: {coverage:>5.1f}%)")
        processed += 1
        
    except Exception as e:
        tqdm.write(f"✗ {imname}: {type(e).__name__}: {e}")
        failed += 1
        continue

# %%
# Print summary
print(f"\n{'='*60}")
print("Processing Summary")
print(f"{'='*60}")
print(f"Total files:             {len(sice_files):>5}")
print(f"Successfully processed:  {processed:>5}")
print(f"Skipped (existing):      {skipped:>5}")
print(f"Failed:                  {failed:>5}")
print(f"{'='*60}\n")
