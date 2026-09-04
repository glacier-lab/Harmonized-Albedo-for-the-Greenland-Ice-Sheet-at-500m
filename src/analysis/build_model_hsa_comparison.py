"""
Export daily paired model-vs-HSA500m pixels as Vaex HDF5 files.

This script uses converted daily model GeoTIFFs (HCLIM/HIRHAM5), resamples
HSA500m to each model grid, matches valid pixels, and writes one HDF5 file per
model-day with paired values.
"""

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import Resampling, reproject
from tqdm import tqdm
import vaex


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HSA_DAILY_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"

MODEL_DIRS = {
    "HCLIM": "/data_3/shunan_2/AU/hsa500m/Kristiina/HCLIM_geotiff",
    "HIRHAM5": "/data_3/shunan_2/AU/hsa500m/Kristiina/HIRHAM5_geotiff",
}

OUTPUT_BASE_DIR = "/data_3/shunan_2/AU/hsa500m/Kristiina/model_hsa_comparison"
OUTPUT_HDF5_BASE = os.path.join(OUTPUT_BASE_DIR, "paired_hdf5")

YEAR_START = 2000
YEAR_END = 2025
NUM_WORKERS = 10
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

        meta = {
            "shape": (src.height, src.width),
            "transform": src.transform,
            "crs": src.crs,
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


def save_pairs_hdf5(output_h5: str, model_vals: np.ndarray, hsa_vals: np.ndarray) -> int:
    df = pd.DataFrame(
        {
            "model_albedo": model_vals.astype(np.float32),
            "hsa500m_albedo": hsa_vals.astype(np.float32),
        }
    ).dropna(subset=["model_albedo", "hsa500m_albedo"])

    if df.empty:
        return 0

    vaex.from_pandas(df, copy_index=False).export_hdf5(output_h5, mode="w")
    return len(df)


def process_single_day(
    model_name: str,
    day: pd.Timestamp,
    model_fp: str,
    hsa_index: Dict[pd.Timestamp, str],
    out_hdf5_dir: str,
) -> Tuple[str, int]:
    hsa_fp = hsa_index.get(day)
    if hsa_fp is None:
        return "skipped", 0

    out_h5 = os.path.join(out_hdf5_dir, f"{model_name.lower()}_hsa500m_{day.strftime('%Y%m%d')}.h5")
    if os.path.exists(out_h5) and not OVERWRITE_EXISTING_FILES:
        return "skipped", 0

    try:
        model_arr, model_meta = read_model_albedo(model_fp)
        hsa_arr = read_hsa_resampled_to_model_grid(
            hsa_fp,
            shape=model_meta["shape"],
            transform=model_meta["transform"],
            crs=model_meta["crs"],
        )
    except Exception:
        return "failed", 0

    valid = np.isfinite(model_arr) & np.isfinite(hsa_arr)
    if not np.any(valid):
        if OVERWRITE_EXISTING_FILES and os.path.exists(out_h5):
            os.remove(out_h5)
        return "skipped", 0

    n_pairs = save_pairs_hdf5(out_h5, model_arr[valid], hsa_arr[valid])
    if n_pairs == 0:
        if OVERWRITE_EXISTING_FILES and os.path.exists(out_h5):
            os.remove(out_h5)
        return "skipped", 0

    return "written", n_pairs


def process_model(model_name: str, model_dir: str, hsa_index: Dict[pd.Timestamp, str]) -> None:
    model_daily = list_model_files(model_dir)
    if not model_daily:
        raise FileNotFoundError(f"[{model_name}] No model GeoTIFF files found in {model_dir}")

    out_hdf5_dir = os.path.join(OUTPUT_HDF5_BASE, model_name)
    os.makedirs(out_hdf5_dir, exist_ok=True)

    written_days = 0
    skipped_days = 0
    failed_days = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_single_day, model_name, day, model_fp, hsa_index, out_hdf5_dir)
            for day, model_fp in model_daily
        ]

        with tqdm(total=len(futures), desc=f"{model_name}: pairing", unit="day") as pbar:
            for future in as_completed(futures):
                try:
                    status, _ = future.result()
                except Exception:
                    status = "failed"

                if status == "written":
                    written_days += 1
                elif status == "skipped":
                    skipped_days += 1
                else:
                    failed_days += 1

                pbar.update(1)

    print(f"[{model_name}] written_days={written_days}, skipped_days={skipped_days}, failed_days={failed_days}")
    print(f"[{model_name}] daily HDF5 dir: {out_hdf5_dir}")


def main() -> None:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    hsa_index = build_hsa_index(HSA_DAILY_DIR)
    if len(hsa_index) == 0:
        raise FileNotFoundError(f"No HSA files found in {HSA_DAILY_DIR}")

    print(f"HSA files indexed: {len(hsa_index)}")
    print(f"Year filter: {YEAR_START}-{YEAR_END}")
    print(f"Workers: {NUM_WORKERS}")

    for model_name, model_dir in MODEL_DIRS.items():
        process_model(model_name, model_dir, hsa_index)

    print("GeoTIFF-based daily HDF5 pairing completed.")


if __name__ == "__main__":
    main()
