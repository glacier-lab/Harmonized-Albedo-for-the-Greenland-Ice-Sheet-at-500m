"""
Calculate blue-sky albedo from black-sky (BSA) and white-sky (WSA) albedo
using CARRA-derived daily diffuse fraction.

    blue_sky = (1 - f) * BSA + f * WSA

Supports MCD43A3 and VIIRS sensors. Select SENSOR at the configuration block below.

Shunan Feng (shunan.feng@envs.au.dk)
"""
# %%
import os
import glob
import re
import numpy as np
import rasterio as rio
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# =============================================================================
# CONFIGURATION — set SENSOR to "MCD43A3" or "VIIRS_VJ143MA3" or "VIIRS_VNP43MA3"
# =============================================================================
SENSOR = "MCD43A3"

CONFIGS = {
    "MCD43A3": {
        "albedo_dir":  "/data_3/shunan_2/AU/hsa500m/MCD43A3_061_mosaics",
        "albedo_glob": "MCD43A3_Albedo_*_500m.tif",
        "date_re":     r"MCD43A3_Albedo_(\d{8})_500m\.tif",
        "output_dir":  "/data_3/shunan_2/AU/hsa500m/MCD43A3_061_bluesky",
        "prefix":      "MCD43A3_BlueskyAlbedo",
    },
    "VIIRS_VJ143MA3": {
        "albedo_dir":  "/data_3/shunan_2/AU/hsa500m/VIIRS_mosaics/VJ143MA3",
        "albedo_glob": "VIIRS_Albedo_*_500m.tif",
        "date_re":     r"VIIRS_Albedo_(\d{8})_500m\.tif",
        "output_dir":  "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VJ143MA3",
        "prefix":      "VJ143MA3_BlueskyAlbedo",
    },
    "VIIRS_VNP43MA3": {
        "albedo_dir":  "/data_3/shunan_2/AU/hsa500m/VIIRS_mosaics/VNP43MA3",
        "albedo_glob": "VIIRS_Albedo_*_500m.tif",
        "date_re":     r"VIIRS_Albedo_(\d{8})_500m\.tif",
        "output_dir":  "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VNP43MA3",
        "prefix":      "VNP43MA3_BlueskyAlbedo",
    },
}

DIFFUSE_DIR = "/data_3/shunan_2/AU/hsa500m/CARRA_diffuse_fraction"
NUM_WORKERS = 20

# =============================================================================

cfg = CONFIGS[SENSOR]
os.makedirs(cfg["output_dir"], exist_ok=True)

# Build date-indexed lookup of albedo files
albedo_files = sorted(glob.glob(os.path.join(cfg["albedo_dir"], cfg["albedo_glob"])))
albedo_by_date = {}
for fp in albedo_files:
    m = re.search(cfg["date_re"], os.path.basename(fp))
    if m:
        albedo_by_date[m.group(1)] = fp

# Build date-indexed lookup of diffuse fraction files
diffuse_files = sorted(glob.glob(os.path.join(DIFFUSE_DIR, "CARRA_diffuse_fraction_*_500m.tif")))
diffuse_by_date = {}
for fp in diffuse_files:
    m = re.search(r"CARRA_diffuse_fraction_(\d{8})_500m\.tif", os.path.basename(fp))
    if m:
        diffuse_by_date[m.group(1)] = fp

# Only process dates present in both datasets
common_dates = sorted(set(albedo_by_date) & set(diffuse_by_date))
print(f"Sensor       : {SENSOR}")
print(f"Albedo files : {len(albedo_by_date)}")
print(f"Diffuse files: {len(diffuse_by_date)}")
print(f"Matched dates: {len(common_dates)}")


def process_date(date, albedo_path, diffuse_path, output_dir, prefix):
    try:
        with rio.open(albedo_path) as src_a:
            bsa = src_a.read(1).astype(np.float32)  # band 1: BSA shortwave
            wsa = src_a.read(2).astype(np.float32)  # band 2: WSA shortwave
            profile = src_a.profile.copy()

        with rio.open(diffuse_path) as src_f:
            f = src_f.read(1).astype(np.float32)   # band 1: diffuse fraction

        # blue_sky = (1 - f) * BSA + f * WSA
        blue_sky = (1.0 - f) * bsa + f * wsa

        # Propagate NaN: if any input is NaN, output is NaN
        no_data = np.isnan(bsa) | np.isnan(wsa) | np.isnan(f)
        blue_sky[no_data] = np.nan

        # Physical range safety clip
        blue_sky = np.where(no_data, np.nan, np.clip(blue_sky, 0.0, 1.0))

        out_path = os.path.join(output_dir, f"{prefix}_{date}_500m.tif")
        profile.update(count=1, compress="lzw", nodata=np.nan)

        with rio.open(out_path, "w", **profile) as dst:
            dst.write(blue_sky, 1)
            dst.set_band_description(1, "blue_sky_albedo_shortwave")

        return date, True, None

    except Exception as e:
        return date, False, str(e)


# %%
print(f"\nProcessing {len(common_dates)} dates with {NUM_WORKERS} workers...")

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(
            process_date,
            date,
            albedo_by_date[date],
            diffuse_by_date[date],
            cfg["output_dir"],
            cfg["prefix"],
        ): date
        for date in common_dates
    }

    with tqdm(total=len(futures), desc=f"Blue-sky albedo ({SENSOR})") as pbar:
        for future in as_completed(futures):
            date, success, err = future.result()
            if not success:
                tqdm.write(f"✗ {date}: {err}")
            pbar.update(1)

print("Done.")