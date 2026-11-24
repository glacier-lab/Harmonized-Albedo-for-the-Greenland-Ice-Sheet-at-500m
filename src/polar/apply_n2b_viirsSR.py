#%%
"""
Apply sensor-specific narrow-to-broadband regression to polar VIIRS SR mosaics.

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
4. Converts scaled uint16 reflectance to 0-1 and applies the regression
5. Saves single-band broadband albedo GeoTIFFs as uint16, scaled by 10000

Inputs and outputs use nodata 65535 and a GeoTIFF scale of 0.0001.
The input and output roots are used as supplied; each contains product
subfolders, matching the polar build_mosaic_VIIRS_SR.py output layout.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
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

REGION_EPSG = {"arctic": 3413, "antarctica": 3031}
SCALE = 0.0001
NODATA = 65535


def detect_product_from_filename(imname):
    """Infer VIIRS product from filename tokens."""
    parts = imname.split("_")
    if len(parts) < 5:
        return None
    product = parts[2]
    if product in MODEL_BY_PRODUCT:
        return product
    return None


def apply_n2b_to_file(file_path, output_root, expected_epsg):
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
            if src.crs != CRS.from_epsg(expected_epsg):
                raise ValueError(
                    f"Input CRS {src.crs} does not match EPSG:{expected_epsg}"
                )
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

            band_indexes = [band_names.index(band_name) + 1 for band_name in required_bands]
            if src.nodata != NODATA or any(
                src.dtypes[i - 1] != "uint16"
                or not np.isclose(src.scales[i - 1], SCALE)
                or src.offsets[i - 1] != 0
                for i in band_indexes
            ):
                raise ValueError(
                    "Expected uint16 reflectance with nodata 65535, scale 0.0001 and offset 0"
                )

            albedo = np.full((src.height, src.width), intercept, dtype=np.float32)
            valid_mask = np.ones(albedo.shape, dtype=bool)
            for band_name, band_idx in zip(required_bands, band_indexes):
                band = src.read(band_idx, masked=True)
                valid_mask &= ~np.ma.getmaskarray(band)
                reflectance = band.astype(np.float32).filled(0) * SCALE
                albedo += coefficients[band_name] * reflectance

        albedo = np.clip(albedo, 0, 1)
        scaled_albedo = np.full(albedo.shape, NODATA, dtype=np.uint16)
        scaled_albedo[valid_mask] = np.rint(albedo[valid_mask] / SCALE).astype(np.uint16)

        product_out_dir = os.path.join(output_root, product)
        os.makedirs(product_out_dir, exist_ok=True)

        out_name = imname.replace("VIIRS_SR_", "VIIRS_SRalbedo_")
        out_path = os.path.join(product_out_dir, out_name)

        out_profile = profile
        out_profile.update(
            {
                "count": 1,
                "dtype": np.uint16,
                "transform": transform,
                "crs": crs,
                "nodata": NODATA,
                "compress": "lzw",
            }
        )

        with rio.open(out_path, "w", **out_profile) as dst:
            dst.write(scaled_albedo, 1)
            dst.scales = (SCALE,)
            dst.update_tags(
                PRODUCT=product,
                ALGORITHM="narrow_to_broadband_viirs_sr",
                INTERCEPT=str(intercept),
                COEFFICIENTS=",".join([f"{k}:{v}" for k, v in coefficients.items()]),
            )

        return imname, True, None

    except Exception as e:
        return imname, False, f"{type(e).__name__}: {e}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert polar VIIRS surface reflectance mosaics to albedo."
    )
    parser.add_argument(
        "--region", required=True, choices=list(REGION_EPSG),
        help="Polar region: arctic (EPSG:3413) or antarctica (EPSG:3031).",
    )
    parser.add_argument(
        "--input-root",
        default="/data_3/shunan_2/AU/hsa500m/VIIRS_SR_mosaics",
        help="Input root containing product subfolders.",
    )
    parser.add_argument(
        "--output-root",
        default="/data_3/shunan_2/AU/hsa500m/VIIRS_SR_albedo",
        help="Output root; product subfolders are created automatically.",
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of process workers."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sr_mosaic_root = args.input_root
    output_root = args.output_root
    sr_files = sorted(glob.glob(os.path.join(sr_mosaic_root, "*", "VIIRS_SR_*.tif")))

    if len(sr_files) == 0:
        print(f"No VIIRS SR mosaic files found in {sr_mosaic_root}, exiting.")
        return

    print(f"Found {len(sr_files)} VIIRS SR mosaic files to process with {args.workers} workers.\n")

    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                apply_n2b_to_file, fpath, output_root, REGION_EPSG[args.region]
            ): fpath
            for fpath in sr_files
        }

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
