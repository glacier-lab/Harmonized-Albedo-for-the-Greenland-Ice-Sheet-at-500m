#%%
"""
Apply narrow-to-broadband regression to GCOM-C SR mosaics.

Regression coefficients from narrowband-to-broadband calibration:
albedo_pred = 0.0443 + 1.0587*Rs_VN01 + 0.1336*Rs_VN02 + -0.7252*Rs_VN03
            + 0.3253*Rs_VN05 + -1.7284*Rs_VN07 + 1.5388*Rs_VN08
            + 0.0843*Rs_VN09 + 0.4209*Rs_VN10 + -0.3269*Rs_VN11 + 0.3677*Rs_SW03

This script:
1. Reads GCOM-C SR multiband GeoTIFF mosaics
2. Extracts the specific bands used in the regression
3. Applies the band math to compute broadband albedo
4. Saves single-band broadband albedo GeoTIFFs to an output directory
"""

import os
import glob
import numpy as np
import rasterio as rio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Regression coefficients ---
INTERCEPT = 0.0443
COEFFICIENTS = {
    'Rs_VN01': 1.0587,
    'Rs_VN02': 0.1336,
    'Rs_VN03': -0.7252,
    'Rs_VN05': 0.3253,
    'Rs_VN07': -1.7284,
    'Rs_VN08': 1.5388,
    'Rs_VN09': 0.0843,
    'Rs_VN10': 0.4209,
    'Rs_VN11': -0.3269,
    'Rs_SW03': 0.3677,
}
REQUIRED_BANDS = list(COEFFICIENTS.keys())

# --- Setup Paths ---
sr_mosaic_folder = '/data_3/shunan_2/AU/hsa500m/GCOMC_SR_mosaics'
output_folder = '/data_3/shunan_2/AU/hsa500m/GCOMC_SR_albedo'
os.makedirs(output_folder, exist_ok=True)

NUM_WORKERS = 10


def apply_n2b_to_file(file_path):
    """
    Apply narrow-to-broadband regression to a single SR mosaic file.
    
    Parameters
    ----------
    file_path : str
        Path to the SR mosaic GeoTIFF
    
    Returns
    -------
    tuple: (filename, success, error_msg)
        filename: str
        success: bool
        error_msg: str or None
    """
    imname = os.path.basename(file_path)
    
    try:
        # Open and read metadata
        with rio.open(file_path) as src:
            # Read BAND_NAMES tag to identify band indices
            band_names_tag = src.tags().get('BAND_NAMES', '')

            if not band_names_tag:
                return imname, False, "No BAND_NAMES tag found"

            band_names = [b.strip() for b in band_names_tag.split(',') if b.strip()]
            transform = src.transform
            crs = src.crs

        # Validate that all required bands are present
        missing_bands = [b for b in REQUIRED_BANDS if b not in band_names]
        if missing_bands:
            return imname, False, f"Missing bands {missing_bands}"

        # Read the multiband GeoTIFF
        with rio.open(file_path) as src:
            all_bands = src.read()

        # Extract required bands by index
        band_data = {}
        for band_name in REQUIRED_BANDS:
            band_idx = band_names.index(band_name)
            band_data[band_name] = all_bands[band_idx, :, :].astype(np.float32)

        # Apply the regression (band math)
        # albedo = intercept + sum(coeff * band)
        albedo = np.full_like(band_data[REQUIRED_BANDS[0]], INTERCEPT, dtype=np.float32)

        for band_name, coeff in COEFFICIENTS.items():
            albedo += coeff * band_data[band_name]

        # Clip to valid range [0, 1]
        albedo = np.clip(albedo, 0, 1)

        # Preserve NaN where input bands are NaN (any required band is NaN means output NaN)
        valid_mask = ~np.full_like(band_data[REQUIRED_BANDS[0]], False, dtype=bool)
        for band_name in REQUIRED_BANDS:
            valid_mask &= ~np.isnan(band_data[band_name])

        albedo[~valid_mask] = np.nan

        # Save as single-band GeoTIFF
        out_path = os.path.join(output_folder, imname.replace('GCOMC_SR_', 'GCOMC_SRalbedo_'))

        with rio.open(
            out_path, 'w',
            driver='GTiff',
            height=albedo.shape[0],
            width=albedo.shape[1],
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress='lzw',
        ) as dst:
            dst.write(albedo, 1)

        return imname, True, None

    except Exception as e:
        return imname, False, f"{type(e).__name__}: {e}"


# Find all SR mosaic files
sr_files = sorted(glob.glob(os.path.join(sr_mosaic_folder, 'GCOMC_SR_*.tif')))

if len(sr_files) == 0:
    print(f"No GCOMC SR mosaic files found in {sr_mosaic_folder}, exiting.")
else:
    print(f"Found {len(sr_files)} SR mosaic files to process with {NUM_WORKERS} workers.\n")

    # --- Processing Loop with Parallelization ---
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(apply_n2b_to_file, fpath): fpath for fpath in sr_files}

        with tqdm(total=len(sr_files), desc="Processing SR Mosaics") as pbar:
            for future in as_completed(futures):
                imname, success, error_msg = future.result()

                if not success:
                    tqdm.write(f"✗ {imname}: {error_msg}")

                pbar.update(1)

    print(f"\nProcessing Complete. Broadband albedo mosaics saved to {output_folder}")

# %%
