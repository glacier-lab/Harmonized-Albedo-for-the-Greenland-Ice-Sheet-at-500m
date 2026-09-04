"""
Build monthly mean HSA500m GeoTIFFs from daily gapfilled files.

Reads Band 1 (hsa500m_gapfilled) from each daily hsa500m_gapfilled_YYYYMMDD.tif,
groups by calendar month, computes per-pixel nanmean, and writes a single-band
float32 GeoTIFF per month.

Output filename: hsa500m_monthly_YYYYMM.tif
"""

import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio as rio
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_monthly"
OVERWRITE_EXISTING_FILES = True
NUM_WORKERS = 10


def parse_date_from_name(filename: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"hsa500m_gapfilled_(\d{4})(\d{2})\d{2}\.tif", filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def group_files_by_month(input_dir: str) -> Dict[Tuple[int, int], List[str]]:
    files = sorted(glob.glob(os.path.join(input_dir, "hsa500m_gapfilled_*.tif")))
    groups: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for fp in files:
        key = parse_date_from_name(os.path.basename(fp))
        if key is not None:
            groups[key].append(fp)
    return groups


def process_month(year: int, month: int, daily_files: List[str], output_dir: str) -> Tuple[bool, str]:
    label = f"{year:04d}-{month:02d}"
    output_tif = os.path.join(output_dir, f"hsa500m_monthly_{year:04d}{month:02d}.tif")

    if os.path.exists(output_tif) and not OVERWRITE_EXISTING_FILES:
        return True, f"{label} -> skipped (exists)"

    # Read profile from the first file
    with rio.open(daily_files[0]) as src:
        profile = src.profile.copy()
        base_shape = (src.height, src.width)

    # Stack band 1 from all daily files
    stack = np.full((len(daily_files), base_shape[0], base_shape[1]), np.nan, dtype=np.float32)
    for i, fp in enumerate(daily_files):
        try:
            with rio.open(fp) as src:
                data = src.read(1).astype(np.float32)
                nodata = src.nodata
            if nodata is not None and np.isfinite(nodata):
                data[data == nodata] = np.nan
            stack[i] = data
        except Exception as exc:
            pass  # leave as NaN

    monthly_mean = np.nanmean(stack, axis=0).astype(np.float32)
    # Pixels with no valid data across all days remain NaN via nanmean on all-NaN slices
    all_nan = np.all(~np.isfinite(stack), axis=0)
    monthly_mean[all_nan] = np.nan

    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="LZW",
        predictor=3,
    )

    with rio.open(output_tif, "w", **profile) as dst:
        dst.write(monthly_mean, 1)
        dst.set_band_description(1, "hsa500m_monthly_mean")

    n_valid = int(np.sum(np.isfinite(monthly_mean)))
    n_days = len(daily_files)
    return True, f"{label} -> {n_days} days, valid pixels={n_valid}"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = group_files_by_month(INPUT_DIR)
    if not groups:
        raise FileNotFoundError(f"No daily HSA500m files found in {INPUT_DIR}")

    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Months to process: {len(groups)}")
    print(f"Overwrite existing: {OVERWRITE_EXISTING_FILES}")
    print(f"Workers: {NUM_WORKERS}")

    month_keys = sorted(groups.keys())

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_month, year, month, groups[(year, month)], OUTPUT_DIR): (year, month)
            for year, month in month_keys
        }

        success = 0
        failed = 0
        with tqdm(total=len(futures), desc="Building monthly HSA500m", unit="month") as pbar:
            for future in as_completed(futures):
                try:
                    ok, msg = future.result()
                except Exception as exc:
                    year, month = futures[future]
                    ok, msg = False, f"{year:04d}-{month:02d} failed: {type(exc).__name__}: {exc}"
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
