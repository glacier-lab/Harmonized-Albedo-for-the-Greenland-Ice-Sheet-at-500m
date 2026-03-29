"""
Unified point-to-pixel extractor for reprojected GeoTIFF albedo products.

This script samples selected GeoTIFF bands at AWS station locations and writes one
CSV per sensor into a common output folder:
    /data_3/shunan_2/AU/hsa500m/point2pix

Design goals:
- One script for multiple sensors (switch by SENSORS_TO_RUN)
- Keep existing CEN reinstallation handling (use 2016 coords before 2017-07-25)
- Support single-band and multi-band products
- Keep NaN values in output to preserve missing-data information

Shunan Feng (shunan.feng@envs.au.dk)
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio as rio
from pyproj import CRS, Transformer
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
AWS_PATH = "/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/point2pix"

CEN_NAME = "CEN"
CEN_REINSTALLATION_DATE = pd.Timestamp("2017-07-25")

# Choose one or more sensors from SENSOR_CONFIGS keys.
SENSORS_TO_RUN = [
    "MOD10A1",
    "MYD10A1",
    "MCD43A3_BLUESKY",
    "VIIRS_VJ143MA3_BLUESKY",
    "VIIRS_VNP43MA3_BLUESKY",
    "GCOMC_SR_ALBEDO",
    "SICE_REBUILD",
]

# Band mapping supports two modes:
# 1) index-based: {"column": "...", "index": <1-based band index>}
# 2) name-based (for datasets with BAND_NAMES tag): {"column": "...", "name": "BandName"}
SENSOR_CONFIGS = {
    "MOD10A1": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/MOD10A1_cropped",
        "glob": "*.tif",
        "date_regex": r"(\d{4}-\d{2}-\d{2})",
        "output_csv": "point2pix_mod10a1.csv",
        "bands": [
            {"column": "albedo_mod10", "index": 1},
        ],
    },
    "MYD10A1": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/MYD10A1_cropped",
        "glob": "*.tif",
        "date_regex": r"(\d{4}-\d{2}-\d{2})",
        "output_csv": "point2pix_myd10a1.csv",
        "bands": [
            {"column": "albedo_myd10", "index": 1},
        ],
    },
    "MCD43A3_BLUESKY": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/MCD43A3_061_bluesky",
        "glob": "MCD43A3_BlueskyAlbedo_*_500m.tif",
        "date_regex": r"MCD43A3_BlueskyAlbedo_(\d{8})_500m\.tif",
        "output_csv": "point2pix_mcd43a3_bluesky.csv",
        "bands": [
            {"column": "albedo_mcd43a3_bluesky", "index": 1},
        ],
    },
    "VIIRS_VJ143MA3_BLUESKY": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VJ143MA3",
        "glob": "VJ143MA3_BlueskyAlbedo_*_500m.tif",
        "date_regex": r"VJ143MA3_BlueskyAlbedo_(\d{8})_500m\.tif",
        "output_csv": "point2pix_viirs_vj143ma3_bluesky.csv",
        "bands": [
            {"column": "albedo_viirs_vj143ma3_bluesky", "index": 1},
        ],
    },
    "VIIRS_VNP43MA3_BLUESKY": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VNP43MA3",
        "glob": "VNP43MA3_BlueskyAlbedo_*_500m.tif",
        "date_regex": r"VNP43MA3_BlueskyAlbedo_(\d{8})_500m\.tif",
        "output_csv": "point2pix_viirs_vnp43ma3_bluesky.csv",
        "bands": [
            {"column": "albedo_viirs_vnp43ma3_bluesky", "index": 1},
        ],
    },
    "GCOMC_SR_ALBEDO": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/GCOMC_SR_albedo",
        "glob": "GCOMC_SRalbedo_*_500m.tif",
        "date_regex": r"GCOMC_SRalbedo_(\d{8})_500m\.tif",
        "output_csv": "point2pix_gcomc_sr_albedo.csv",
        "bands": [
            {"column": "albedo_gcomc_sr", "index": 1},
        ],
    },
    "SICE_REBUILD": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/SICE_rebuild",
        "glob": "SICE_Albedo_*_500m.tif",
        "date_regex": r"SICE_Albedo_(\d{8})_500m\.tif",
        "output_csv": "point2pix_sice_rebuild.csv",
        "bands": [
            {"column": "albedo_sice_rebuild", "index": 1},
        ],
    },
}


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def parse_date_from_name(filename, date_regex):
    match = re.search(date_regex, filename)
    if not match:
        return None

    date_token = match.group(1)
    if len(date_token) == 8:
        return pd.to_datetime(date_token, format="%Y%m%d")
    if len(date_token) == 10:
        return pd.to_datetime(date_token, format="%Y-%m-%d")
    raise ValueError(f"Unsupported date token in {filename}: {date_token}")


def get_station_coords_for_date(df_aws_year, date, cen_coords_2016):
    lats = []
    lons = []
    aws_names = []

    for _, row in df_aws_year.iterrows():
        aws_name = row["aws"]

        if (
            date.year == 2017
            and aws_name == CEN_NAME
            and date < CEN_REINSTALLATION_DATE
            and cen_coords_2016 is not None
        ):
            lat, lon = cen_coords_2016
        else:
            lat = row["lat"]
            lon = row["lon"]

        aws_names.append(aws_name)
        lats.append(lat)
        lons.append(lon)

    return aws_names, np.asarray(lats), np.asarray(lons)


def resolve_band_indices(src, band_specs):
    band_names_tag = src.tags().get("BAND_NAMES", "")
    band_names = [b.strip() for b in band_names_tag.split(",") if b.strip()]

    resolved = []
    for spec in band_specs:
        col = spec["column"]

        if "index" in spec:
            idx = int(spec["index"])
            if idx < 1 or idx > src.count:
                raise ValueError(f"Band index {idx} out of range for column {col}")
            resolved.append((col, idx))
            continue

        if "name" in spec:
            band_name = spec["name"]
            if not band_names:
                raise ValueError(
                    f"BAND_NAMES tag missing, cannot resolve named band {band_name}"
                )
            if band_name not in band_names:
                raise ValueError(
                    f"Band {band_name} not found in BAND_NAMES ({band_names})"
                )
            resolved.append((col, band_names.index(band_name) + 1))
            continue

        raise ValueError(f"Invalid band specification for output column {col}: {spec}")

    return resolved


def sample_geotiff_at_points(src, lons, lats, band_idx_list):
    transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    coords = list(zip(xs, ys))

    sampled = np.asarray(list(src.sample(coords, indexes=band_idx_list)), dtype=np.float32)

    nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        sampled[sampled == nodata] = np.nan

    return sampled


def process_sensor(sensor_name, config, df_aws, cen_coords_2016):
    input_files = sorted(glob.glob(os.path.join(config["input_dir"], config["glob"])))
    output_csv = os.path.join(OUTPUT_DIR, config["output_csv"])

    if len(input_files) == 0:
        print(f"[{sensor_name}] No input files found in {config['input_dir']}")
        return

    # Write CSV header
    output_columns = ["aws", "time"] + [b["column"] for b in config["bands"]]
    with open(output_csv, "w") as f:
        f.write(",".join(output_columns) + "\n")

    total_records = 0
    skipped_files = 0

    print(f"\n{'=' * 70}")
    print(f"Sensor: {sensor_name}")
    print(f"Input files: {len(input_files)}")
    print(f"Output CSV: {output_csv}")
    print(f"{'=' * 70}")

    for file_path in tqdm(input_files, desc=f"{sensor_name}", unit="file"):
        filename = os.path.basename(file_path)

        try:
            imtime = parse_date_from_name(filename, config["date_regex"])
            if imtime is None:
                tqdm.write(f"[{sensor_name}] Skip {filename}: cannot parse date")
                skipped_files += 1
                continue

            year = imtime.year
            df_aws_year = df_aws[df_aws["year"] == year]
            if len(df_aws_year) == 0:
                continue

            aws_names, lats, lons = get_station_coords_for_date(df_aws_year, imtime, cen_coords_2016)

            with rio.open(file_path) as src:
                resolved_bands = resolve_band_indices(src, config["bands"])
                band_cols = [c for c, _ in resolved_bands]
                band_idx_list = [i for _, i in resolved_bands]

                sampled = sample_geotiff_at_points(src, lons, lats, band_idx_list)

            df_out = pd.DataFrame({
                "aws": aws_names,
                "time": imtime,
            })

            for col_idx, col_name in enumerate(band_cols):
                df_out[col_name] = sampled[:, col_idx]

            df_out.to_csv(output_csv, mode="a", header=False, index=False)
            total_records += len(df_out)

        except Exception as exc:
            tqdm.write(f"[{sensor_name}] Skip {filename}: {type(exc).__name__}: {exc}")
            skipped_files += 1
            continue

    print(f"[{sensor_name}] Wrote {total_records} rows")
    print(f"[{sensor_name}] Skipped {skipped_files} files")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_aws = pd.read_csv(AWS_PATH)
    df_cen_2016 = df_aws[(df_aws["aws"] == CEN_NAME) & (df_aws["year"] == 2016)]
    cen_coords_2016 = None
    if not df_cen_2016.empty:
        cen_coords_2016 = (df_cen_2016.iloc[0]["lat"], df_cen_2016.iloc[0]["lon"])

    invalid_sensors = [s for s in SENSORS_TO_RUN if s not in SENSOR_CONFIGS]
    if invalid_sensors:
        raise ValueError(f"Invalid sensors in SENSORS_TO_RUN: {invalid_sensors}")

    for sensor_name in SENSORS_TO_RUN:
        process_sensor(sensor_name, SENSOR_CONFIGS[sensor_name], df_aws, cen_coords_2016)

    print(f"\nDone. Point-to-pixel outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
