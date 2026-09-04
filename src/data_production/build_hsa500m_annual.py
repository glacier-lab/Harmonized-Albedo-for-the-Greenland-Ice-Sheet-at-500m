"""
Build annual HSA500m summary GeoTIFFs from daily gapfilled files.

Reads Band 1 (hsa500m_gapfilled) from each daily hsa500m_gapfilled_YYYYMMDD.tif,
groups by calendar year, and writes 2-band annual products:
- Band 1: annual average
- Band 2: annual standard deviation
- Band 3: number of days with albedo < 0.431 (dark ice days)
- Band 4: number of days with albedo < 0.679 (melt season days)

Output filename: hsa500m_annual_YYYY.tif
"""

import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_annual"
OVERWRITE_EXISTING_FILES = True
TILE_SIZE = 256
N_WORKERS = 20

# Optional: restrict to specific calendar months, e.g. (6, 7, 8) for JJA.
# Set to None to use all months.
FILTER_MONTHS: Optional[Tuple[int, ...]] = (5, 6, 7, 8, 9)
DARK_ICE_THRESHOLD = 0.431
MELT_SEASON_THRESHOLD = 0.679


def parse_date_from_name(filename: str) -> Optional[pd.Timestamp]:
    m = re.search(r"hsa500m_gapfilled_(\d{8})\.tif", filename)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def group_files_by_year(input_dir: str) -> Dict[int, List[str]]:
    files = sorted(glob.glob(os.path.join(input_dir, "hsa500m_gapfilled_*.tif")))
    groups: Dict[int, List[str]] = defaultdict(list)
    for fp in files:
        ts = parse_date_from_name(os.path.basename(fp))
        if ts is None:
            continue
        if FILTER_MONTHS and ts.month not in FILTER_MONTHS:
            continue
        groups[ts.year].append(fp)
    return groups


def generate_windows(height: int, width: int, tile_size: int):
    for row_off in range(0, height, tile_size):
        tile_h = min(tile_size, height - row_off)
        for col_off in range(0, width, tile_size):
            tile_w = min(tile_size, width - col_off)
            yield Window.from_slices((row_off, row_off + tile_h), (col_off, col_off + tile_w))


def compute_tile_stats(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # stack shape: (time, y, x)
    valid = np.isfinite(stack)
    count = np.sum(valid, axis=0).astype(np.int32)
    has_data = count > 0

    sum_vals = np.nansum(stack, axis=0, dtype=np.float64)
    mean = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    np.divide(sum_vals, count, out=mean, where=has_data)

    sumsq_vals = np.nansum(stack * stack, axis=0, dtype=np.float64)
    var = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    np.divide(sumsq_vals, count, out=var, where=has_data)
    var = var - mean.astype(np.float64) * mean.astype(np.float64)
    var[var < 0] = 0

    std = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    std[has_data] = np.sqrt(var[has_data]).astype(np.float32)

    dark_ice_days = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    melt_season_days = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    dark_ice_days[has_data] = np.sum(stack < DARK_ICE_THRESHOLD, axis=0)[has_data].astype(np.float32)
    melt_season_days[has_data] = np.sum(stack < MELT_SEASON_THRESHOLD, axis=0)[has_data].astype(np.float32)

    return mean, std, dark_ice_days, melt_season_days


def process_window(args):
    daily_files, row_off, col_off, tile_h, tile_w = args
    win = Window.from_slices((row_off, row_off + tile_h), (col_off, col_off + tile_w))

    stack = np.full((len(daily_files), tile_h, tile_w), np.nan, dtype=np.float32)
    for i, fp in enumerate(daily_files):
        try:
            with rio.open(fp) as src:
                data = src.read(1, window=win).astype(np.float32)
                nodata = src.nodata
            if nodata is not None and np.isfinite(nodata):
                data[data == nodata] = np.nan
            stack[i] = data
        except Exception:
            pass

    annual_mean, annual_std, dark_ice_days, melt_season_days = compute_tile_stats(stack)
    return row_off, col_off, annual_mean, annual_std, dark_ice_days, melt_season_days


def process_year(year: int, daily_files: List[str], output_dir: str) -> Tuple[bool, str]:
    output_tif = os.path.join(output_dir, f"hsa500m_annual_{year:04d}.tif")

    if os.path.exists(output_tif) and not OVERWRITE_EXISTING_FILES:
        return True, f"{year:04d} -> skipped (exists)"

    with rio.open(daily_files[0]) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width

    profile.update(
        dtype="float32",
        count=4,
        nodata=np.nan,
        compress="LZW",
        predictor=3,
    )

    windows = list(generate_windows(height, width, TILE_SIZE))
    tasks = [
        (daily_files, int(win.row_off), int(win.col_off), int(win.height), int(win.width))
        for win in windows
    ]

    with rio.open(output_tif, "w", **profile) as dst:
        dst.set_band_description(1, "annual_mean")
        dst.set_band_description(2, "annual_std")
        dst.set_band_description(3, "dark_ice_days_lt_0p431")
        dst.set_band_description(4, "melt_season_days_lt_0p679")

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_window, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{year:04d}", unit="tile", leave=False):
                row_off, col_off, annual_mean, annual_std, dark_ice_days, melt_season_days = future.result()
                win = Window.from_slices(
                    (row_off, row_off + annual_mean.shape[0]),
                    (col_off, col_off + annual_mean.shape[1]),
                )
                dst.write(annual_mean, 1, window=win)
                dst.write(annual_std, 2, window=win)
                dst.write(dark_ice_days, 3, window=win)
                dst.write(melt_season_days, 4, window=win)

    n_days = len(daily_files)
    return True, f"{year:04d} -> {n_days} days"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = group_files_by_year(INPUT_DIR)
    if not groups:
        raise FileNotFoundError(f"No daily HSA500m files found in {INPUT_DIR}")

    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Years to process: {len(groups)}")
    print(f"Month filter: {list(FILTER_MONTHS) if FILTER_MONTHS else 'all months'}")
    print(f"Dark-ice threshold: {DARK_ICE_THRESHOLD}")
    print(f"Melt-season threshold: {MELT_SEASON_THRESHOLD}")
    print(f"Overwrite existing: {OVERWRITE_EXISTING_FILES}")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Workers: {N_WORKERS}")

    years = sorted(groups.keys())

    success = 0
    failed = 0
    with tqdm(total=len(years), desc="Building annual HSA500m", unit="year") as pbar:
        for year in years:
            try:
                ok, msg = process_year(year, groups[year], OUTPUT_DIR)
            except Exception as exc:
                ok, msg = False, f"{year:04d} failed: {type(exc).__name__}: {exc}"
            if ok:
                success += 1
            else:
                failed += 1
            tqdm.write(msg)
            pbar.update(1)

    print("Done.")
    print(f"Succeeded: {success}, Failed: {failed}")
    print(f"Output GeoTIFFs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
