"""
Extract and reproject SICE albedo products.

This script extracts the existing BBA_combination broadband albedo from SICE
NetCDF files (already in EPSG:3413), reprojects to match the PROMICE mask grid
at 500m resolution, and masks to Greenland ice sheet extent.

Key operations:
- Extracts BBA_combination from SICE NetCDF files (EPSG:3413)
- Reprojects to PROMICE mask grid resolution
- Masks to Greenland ice sheet extent using PROMICE mask
- Exports daily composites as single-band GeoTIFF files
- Parallelized processing with multiple workers

Author: Shunan Feng (shunan.feng@envs.au.dk)
"""
#%%
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from pyproj import CRS
from tqdm import tqdm
from affine import Affine
from concurrent.futures import ProcessPoolExecutor, as_completed

# %%
# Configuration
im_mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"
sice_path = '/data_3/shunan_2/AU/hsa500m/SICE'
sice_rebuild_path = '/data_3/shunan_2/AU/hsa500m/SICE_rebuild'
os.makedirs(sice_rebuild_path, exist_ok=True)

NUM_WORKERS = 10

# Target CRS (SICE data is already in EPSG:3413)
target_crs = CRS.from_epsg(3413)  # NSIDC Polar Stereographic North

# %%
def read_mask(mask_path):
    """Read PROMICE ice mask and return mask array, transform, CRS, and shape."""
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

# %%
# Load mask once
immask, mask_transform, mask_crs, mask_shape = read_mask(im_mask_path)

def process_single_file(file_path, immask, mask_transform, mask_crs, mask_shape, sice_rebuild_path):
    """
    Process a single SICE file and save reprojected albedo output.
    
    Parameters
    ----------
    file_path : str
        Path to SICE NetCDF file
    immask : ndarray
        PROMICE ice mask array
    mask_transform : Affine
        Transform of mask grid
    mask_crs : CRS
        CRS of mask grid
    mask_shape : tuple
        Shape of mask grid (height, width)
    sice_rebuild_path : str
        Output directory path
    
    Returns
    -------
    tuple: (filename, success, error_msg)
        filename: basename of file
        success: bool
        error_msg: str or None
    """
    imname = os.path.basename(file_path)
    
    try:
        # Parse date from filename
        imtime = os.path.splitext(imname)[0].split('_')[-1]
        imtime = pd.to_datetime(imtime)
        
        # Check if output already exists
        out_path = os.path.join(sice_rebuild_path, f'SICE_Albedo_{imtime.strftime("%Y%m%d")}_500m.tif')
        if os.path.exists(out_path):
            return imname, None, "skipped"
        
        # Open dataset
        ds_sice = xr.open_dataset(file_path)
        
        # Check if BBA_combination exists
        if 'BBA_combination' not in ds_sice.variables:
            ds_sice.close()
            return imname, False, "Missing BBA_combination variable"
        
        # Extract existing broadband albedo
        bba_data = ds_sice['BBA_combination'].values.astype(np.float32)
        
        # Get coordinates and dimensions
        x_coord = ds_sice['x'].values
        y_coord = ds_sice['y'].values
        rows, cols = bba_data.shape
        
        # Create source transform from SICE coordinates (already in EPSG:3413)
        src_transform = from_bounds(
            west=x_coord.min(),
            south=y_coord.min(),
            east=x_coord.max(),
            north=y_coord.max(),
            width=cols,
            height=rows
        )
        
        # Reproject to mask grid
        reprojected_albedo = np.full(mask_shape, np.nan, dtype=np.float32)
        reproject(
            source=bba_data,
            destination=reprojected_albedo,
            src_transform=src_transform,
            src_crs=target_crs,
            dst_transform=mask_transform,
            dst_crs=mask_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        
        # Apply PROMICE ice mask
        reprojected_albedo[immask == 0] = np.nan
        
        # Save as GeoTIFF
        with rio.open(
            out_path,
            'w',
            driver='GTiff',
            height=mask_shape[0],
            width=mask_shape[1],
            count=1,
            dtype=np.float32,
            crs=mask_crs,
            transform=mask_transform,
            nodata=np.nan,
            compress='lzw',
        ) as dst:
            dst.write(reprojected_albedo, 1)
        
        ds_sice.close()
        
        # Calculate statistics
        valid_pixels = np.sum(~np.isnan(reprojected_albedo))
        coverage = valid_pixels / reprojected_albedo.size * 100
        
        return imname, True, f"coverage: {coverage:.1f}%"
        
    except Exception as e:
        return imname, False, f"{type(e).__name__}: {e}"


# %%
# Find all SICE files
sice_files = sorted(glob.glob(os.path.join(sice_path, '**/*.nc'), recursive=True))

print(f"\n{'='*60}")
print("SICE Albedo Extraction and Reprojection")
print(f"{'='*60}")
print(f"Found {len(sice_files)} SICE files to process\n")

if len(sice_files) == 0:
    print("No files found!")
    exit()

# %%
# Process all files with parallelization
print(f"Processing {len(sice_files)} files with {NUM_WORKERS} workers...\n")

processed = 0
failed = 0
skipped = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(
            process_single_file,
            file_path,
            immask,
            mask_transform,
            mask_crs,
            mask_shape,
            sice_rebuild_path
        ): file_path
        for file_path in sice_files
    }

    with tqdm(total=len(sice_files), desc="Processing Files", unit="file") as pbar:
        for future in as_completed(futures):
            imname, success, msg = future.result()
            
            if success is None:  # Skipped
                tqdm.write(f"⊘ Skipping {imname}")
                skipped += 1
            elif success:
                tqdm.write(f"✓ {imname} ({msg})")
                processed += 1
            else:
                tqdm.write(f"✗ {imname}: {msg}")
                failed += 1
            
            pbar.update(1)

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
