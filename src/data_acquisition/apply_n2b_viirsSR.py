#%%
"""
Apply sensor-specific narrow-to-broadband regression to VIIRS SR mosaics.

Regression equations are derived from narrow2broadband_viirsSR.py and applied
per sensor/product:
- VNP09GA
- VJ109GA
- VJ209GA

Only the selected VIIRS SR bands are used in the conversion:
SurfReflect_I1, SurfReflect_I2, SurfReflect_I3,
SurfReflect_M7, SurfReflect_M8, SurfReflect_M10, SurfReflect_M11

This script:
1. Reads VIIRS SR multiband GeoTIFF mosaics
2. Detects product from filename
3. Extracts required bands from BAND_NAMES tag
4. Applies product-specific regression to compute broadband albedo
5. Saves single-band broadband albedo GeoTIFFs
"""

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio as rio
from tqdm import tqdm

# --- Sensor-specific regression coefficients ---
MODEL_BY_PRODUCT = {
    "VNP09GA": {
        "intercept": 0.0981,
        "coefficients": {
            "SurfReflect_I1": 0.4284,
            "SurfReflect_I2": 0.3015,
            "SurfReflect_I3": -0.0781,
            "SurfReflect_M7": -0.0114,
            "SurfReflect_M8": 0.1146,
            "SurfReflect_M10": 0.2036,
            "SurfReflect_M11": -0.0777,
        },
    },
    "VJ109GA": {
        "intercept": 0.0787,
        "coefficients": {
            "SurfReflect_I1": 0.4502,
            "SurfReflect_I2": 0.2601,
            "SurfReflect_I3": -0.0481,
            "SurfReflect_M7": 0.0608,
            "SurfReflect_M8": 0.0345,
            "SurfReflect_M10": 0.3930,
            "SurfReflect_M11": -0.1457,
        },
    },
    "VJ209GA": {
        "intercept": 0.0620,
        "coefficients": {
            "SurfReflect_I1": 0.5894,
            "SurfReflect_I2": 0.1437,
            "SurfReflect_I3": -0.0037,
            "SurfReflect_M7": -0.1019,
            "SurfReflect_M8": 0.4591,
            "SurfReflect_M10": 1.0021,
            "SurfReflect_M11": -1.1247,
        },
    },
}

# --- Setup Paths ---
sr_mosaic_root = "/data_3/shunan_2/AU/hsa500m/VIIRS_SR_mosaics"
output_root = "/data_3/shunan_2/AU/hsa500m/VIIRS_SR_albedo"
os.makedirs(output_root, exist_ok=True)

NUM_WORKERS = 10


def detect_product_from_filename(imname):
    """Infer VIIRS product from filename tokens."""
    parts = imname.split("_")
    if len(parts) < 5:
        return None
    product = parts[2]
    if product in MODEL_BY_PRODUCT:
        return product
    return None


def apply_n2b_to_file(file_path):
    """Apply narrow-to-broadband regression to one VIIRS SR mosaic file."""
    imname = os.path.basename(file_path)

    try:
        product = detect_product_from_filename(imname)
        if product is None:
            return imname, False, "Unable to detect product from filename"

        model = MODEL_BY_PRODUCT[product]
        intercept = model["intercept"]
        coefficients = model["coefficients"]
        required_bands = list(coefficients.keys())

        with rio.open(file_path) as src:
            band_names_tag = src.tags().get("BAND_NAMES", "")

            if not band_names_tag:
                return imname, False, "No BAND_NAMES tag found"

            band_names = [b.strip() for b in band_names_tag.split(",") if b.strip()]
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()

            missing_bands = [b for b in required_bands if b not in band_names]
            if missing_bands:
                return imname, False, f"Missing bands {missing_bands}"

            all_bands = src.read().astype(np.float32)

        band_data = {}
        for band_name in required_bands:
            band_idx = band_names.index(band_name)
            band_data[band_name] = all_bands[band_idx, :, :]

        albedo = np.full_like(band_data[required_bands[0]], intercept, dtype=np.float32)
        for band_name, coeff in coefficients.items():
            albedo += coeff * band_data[band_name]

        albedo = np.clip(albedo, 0, 1)

        valid_mask = np.ones_like(albedo, dtype=bool)
        for band_name in required_bands:
            valid_mask &= ~np.isnan(band_data[band_name])
        albedo[~valid_mask] = np.nan

        product_out_dir = os.path.join(output_root, product)
        os.makedirs(product_out_dir, exist_ok=True)

        out_name = imname.replace("VIIRS_SR_", "VIIRS_SRalbedo_")
        out_path = os.path.join(product_out_dir, out_name)

        out_profile = profile
        out_profile.update(
            {
                "count": 1,
                "dtype": np.float32,
                "transform": transform,
                "crs": crs,
                "nodata": np.nan,
                "compress": "lzw",
            }
        )

        with rio.open(out_path, "w", **out_profile) as dst:
            dst.write(albedo, 1)
            dst.update_tags(
                PRODUCT=product,
                ALGORITHM="narrow_to_broadband_viirs_sr",
                INTERCEPT=str(intercept),
                COEFFICIENTS=",".join([f"{k}:{v}" for k, v in coefficients.items()]),
            )

        return imname, True, None

    except Exception as e:
        return imname, False, f"{type(e).__name__}: {e}"


def main():
    sr_files = sorted(glob.glob(os.path.join(sr_mosaic_root, "*", "VIIRS_SR_*.tif")))

    if len(sr_files) == 0:
        print(f"No VIIRS SR mosaic files found in {sr_mosaic_root}, exiting.")
        return

    print(f"Found {len(sr_files)} VIIRS SR mosaic files to process with {NUM_WORKERS} workers.\n")

    n_fail = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(apply_n2b_to_file, fpath): fpath for fpath in sr_files}

        with tqdm(total=len(sr_files), desc="Processing VIIRS SR Mosaics") as pbar:
            for future in as_completed(futures):
                imname, success, error_msg = future.result()

                if not success:
                    n_fail += 1
                    tqdm.write(f"x {imname}: {error_msg}")

                pbar.update(1)

    print(f"\nProcessing complete. Broadband albedo mosaics saved under {output_root}")
    if n_fail > 0:
        print(f"Failed files: {n_fail}")


if __name__ == "__main__":
    main()
# %%
