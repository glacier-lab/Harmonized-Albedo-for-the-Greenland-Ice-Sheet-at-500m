"""
Build bias/RMSE maps from converted model GeoTIFFs vs HSA500m daily GeoTIFFs.

This script reads daily model GeoTIFFs, resamples HSA500m to model grids,
aggregates pixel-wise differences across time, and writes one 3-band GeoTIFF per
model:
  Band 1: bias (model - hsa500m)
  Band 2: rmse
  Band 3: n_pairs
"""

import glob
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import Resampling, reproject
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HSA_DAILY_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"

MODEL_DIRS = {
    "HCLIM": "/data_3/shunan_2/AU/hsa500m/Kristiina/HCLIM_geotiff",
    "HIRHAM5": "/data_3/shunan_2/AU/hsa500m/Kristiina/HIRHAM5_geotiff",
}

OUTPUT_BASE_DIR = "/data_3/shunan_2/AU/hsa500m/Kristiina/model_hsa_comparison"
OUTPUT_MAPS_BASE = os.path.join(OUTPUT_BASE_DIR, "maps")
OUTPUT_STATS_BASE = os.path.join(OUTPUT_BASE_DIR, "stats")

YEAR_START = 2000
YEAR_END = 2025
OVERWRITE_EXISTING_FILES = True

HSA_DATE_REGEX = r"hsa500m_gapfilled_(\d{8})\.tif$"
MODEL_DATE_REGEX = r"_albedo_(\d{8})_modelgrid\.tif$"

MODEL_VALID_MIN = 0.0
MODEL_VALID_MAX = 1.0
HSA_VALID_MIN = 0.0
HSA_VALID_MAX = 1.0


def parse_date_token(file_name: str, pattern: str) -> Optional[pd.Timestamp]:
    m = re.search(pattern, file_name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d").normalize()


def build_hsa_index(input_dir: str) -> Dict[pd.Timestamp, str]:
    idx: Dict[pd.Timestamp, str] = {}
    for fp in sorted(glob.glob(os.path.join(input_dir, "hsa500m_gapfilled_*.tif"))):
        ts = parse_date_token(os.path.basename(fp), HSA_DATE_REGEX)
        if ts is not None:
            idx[ts] = fp
    return idx


def list_model_files(model_dir: str) -> list[Tuple[pd.Timestamp, str]]:
    out = []
    for fp in sorted(glob.glob(os.path.join(model_dir, "*_albedo_*_modelgrid.tif"))):
        ts = parse_date_token(os.path.basename(fp), MODEL_DATE_REGEX)
        if ts is None:
            continue
        if YEAR_START <= ts.year <= YEAR_END:
            out.append((ts, fp))
    return out


def read_model_albedo(model_fp: str) -> Tuple[np.ndarray, dict]:
    with rio.open(model_fp) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr[arr == nodata] = np.nan

        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="LZW", predictor=3)

        meta = {
            "shape": (src.height, src.width),
            "transform": src.transform,
            "crs": src.crs,
            "profile": profile,
        }

    arr[(arr <= MODEL_VALID_MIN) | (arr >= MODEL_VALID_MAX)] = np.nan
    return arr, meta


def read_hsa_resampled_to_model_grid(hsa_fp: str, shape: Tuple[int, int], transform, crs) -> np.ndarray:
    with rio.open(hsa_fp) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        if src_nodata is not None and np.isfinite(src_nodata):
            src_arr[src_arr == src_nodata] = np.nan

        src_valid = np.isfinite(src_arr) & (src_arr > HSA_VALID_MIN) & (src_arr < HSA_VALID_MAX)

        dst_arr = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        dst_valid = np.zeros(shape, dtype=np.float32)
        reproject(
            source=src_valid.astype(np.float32),
            destination=dst_valid,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0.0,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=0.0,
            resampling=Resampling.nearest,
        )

        dst_arr[dst_valid < 0.5] = np.nan

    dst_arr[(dst_arr <= HSA_VALID_MIN) | (dst_arr >= HSA_VALID_MAX)] = np.nan
    return dst_arr


def write_bias_rmse_map(output_tif: str, profile: dict, bias: np.ndarray, rmse: np.ndarray, n_pairs: np.ndarray) -> None:
    out_profile = profile.copy()
    out_profile.update(count=3, dtype="float32", nodata=np.nan, compress="LZW", predictor=3)

    with rio.open(output_tif, "w", **out_profile) as dst:
        dst.write(bias.astype(np.float32), 1)
        dst.write(rmse.astype(np.float32), 2)
        dst.write(n_pairs.astype(np.float32), 3)
        dst.set_band_description(1, "bias_model_minus_hsa500m")
        dst.set_band_description(2, "rmse_model_hsa500m")
        dst.set_band_description(3, "n_pairs")


def process_model(model_name: str, model_dir: str, hsa_index: Dict[pd.Timestamp, str]) -> None:
    model_daily = list_model_files(model_dir)
    if not model_daily:
        raise FileNotFoundError(f"[{model_name}] No model GeoTIFF files found in {model_dir}")

    out_stats_dir = os.path.join(OUTPUT_STATS_BASE, model_name)
    out_maps_dir = os.path.join(OUTPUT_MAPS_BASE, model_name)
    os.makedirs(out_stats_dir, exist_ok=True)
    os.makedirs(out_maps_dir, exist_ok=True)

    sum_diff = None
    sum_sq = None
    count = None
    reference_profile = None

    matched_days = 0
    skipped_days = 0
    failed_days = 0

    for day, model_fp in tqdm(model_daily, desc=f"{model_name}: rmse/bias", unit="day"):
        hsa_fp = hsa_index.get(day)
        if hsa_fp is None:
            skipped_days += 1
            continue

        try:
            model_arr, model_meta = read_model_albedo(model_fp)
            hsa_arr = read_hsa_resampled_to_model_grid(
                hsa_fp,
                shape=model_meta["shape"],
                transform=model_meta["transform"],
                crs=model_meta["crs"],
            )
        except Exception:
            failed_days += 1
            continue

        if reference_profile is None:
            reference_profile = model_meta["profile"]
            sum_diff = np.zeros(model_meta["shape"], dtype=np.float64)
            sum_sq = np.zeros(model_meta["shape"], dtype=np.float64)
            count = np.zeros(model_meta["shape"], dtype=np.int32)

        valid = np.isfinite(model_arr) & np.isfinite(hsa_arr)
        if not np.any(valid):
            skipped_days += 1
            continue

        diff = model_arr - hsa_arr
        sum_diff[valid] += diff[valid]
        sum_sq[valid] += np.square(diff[valid])
        count[valid] += 1
        matched_days += 1

    if reference_profile is None or count is None:
        raise RuntimeError(f"[{model_name}] No readable model files were processed.")

    has_obs = count > 0
    out_map_tif = os.path.join(out_maps_dir, f"{model_name.lower()}_hsa500m_bias_rmse_{YEAR_START}_{YEAR_END}.tif")

    if not np.any(has_obs):
        if OVERWRITE_EXISTING_FILES and os.path.exists(out_map_tif):
            os.remove(out_map_tif)
        print(f"[{model_name}] No valid paired pixels across all days. No map output written.")
        return

    bias = np.full(count.shape, np.nan, dtype=np.float32)
    rmse = np.full(count.shape, np.nan, dtype=np.float32)
    bias[has_obs] = (sum_diff[has_obs] / count[has_obs]).astype(np.float32)
    rmse[has_obs] = np.sqrt(sum_sq[has_obs] / count[has_obs]).astype(np.float32)

    write_bias_rmse_map(out_map_tif, reference_profile, bias, rmse, count)

    total_pairs = int(np.sum(count[has_obs]))
    global_bias = float(np.sum(sum_diff[has_obs]) / total_pairs)
    global_rmse = float(np.sqrt(np.sum(sum_sq[has_obs]) / total_pairs))
    summary_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "year_start": YEAR_START,
                "year_end": YEAR_END,
                "matched_days": int(matched_days),
                "skipped_days": int(skipped_days),
                "failed_days": int(failed_days),
                "total_pairs": total_pairs,
                "global_bias_model_minus_hsa500m": global_bias,
                "global_rmse_model_hsa500m": global_rmse,
            }
        ]
    )
    summary_csv = os.path.join(out_stats_dir, f"{model_name.lower()}_summary_{YEAR_START}_{YEAR_END}.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"[{model_name}] matched_days={matched_days}, skipped_days={skipped_days}, failed_days={failed_days}")
    print(f"[{model_name}] bias/RMSE map: {out_map_tif}")
    print(f"[{model_name}] stats summary: {summary_csv}")


def main() -> None:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    hsa_index = build_hsa_index(HSA_DAILY_DIR)
    if len(hsa_index) == 0:
        raise FileNotFoundError(f"No HSA files found in {HSA_DAILY_DIR}")

    print(f"HSA files indexed: {len(hsa_index)}")
    print(f"Year filter: {YEAR_START}-{YEAR_END}")

    for model_name, model_dir in MODEL_DIRS.items():
        process_model(model_name, model_dir, hsa_index)

    print("GeoTIFF-based bias/RMSE mapping completed.")


if __name__ == "__main__":
    main()
