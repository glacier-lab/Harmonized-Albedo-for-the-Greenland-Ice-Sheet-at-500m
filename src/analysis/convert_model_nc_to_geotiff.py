"""
Convert yearly model NetCDF albedo files (HCLIM and HIRHAM5) to daily GeoTIFFs.

This script reads each yearly NetCDF, extracts daily albedo slices from variable
`albl`, converts time values to calendar dates, and writes one GeoTIFF per day
for each model into separate output folders.
"""

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import Affine
from tqdm import tqdm
import xarray as xr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HCLIM_DIR = "/data_3/shunan_2/AU/hsa500m/Kristiina/HCLIM_ERA5_Greenland"
HIRHAM5_DIR = "/data_3/shunan_2/AU/hsa500m/Kristiina/HIRHAM5_ERA5_Greenland"

OUTPUT_BASE_DIR = "/data_3/shunan_2/AU/hsa500m/Kristiina/"
OUTPUT_HCLIM_DIR = os.path.join(OUTPUT_BASE_DIR, "HCLIM_geotiff")
OUTPUT_HIRHAM5_DIR = os.path.join(OUTPUT_BASE_DIR, "HIRHAM5_geotiff")

YEAR_START = 2000
YEAR_END = 2025
MODEL_VAR_NAME = "albl"
NUM_WORKERS = 10
OVERWRITE_EXISTING_FILES = True

# Keep physical albedo range only.
ALBEDO_MIN = 0.0
ALBEDO_MAX = 1.0


@dataclass(frozen=True)
class ModelGrid:
    shape: Tuple[int, int]
    transform: Affine
    crs: str
    y_increasing: bool


def extract_year(path: str) -> Optional[int]:
    match = re.search(r"(19\d{2}|20\d{2})", path)
    return int(match.group(1)) if match else None


def filter_by_year_range(file_list: list[str], start_year: int, end_year: int) -> list[str]:
    selected = []
    for fp in file_list:
        year = extract_year(fp)
        if year is not None and start_year <= year <= end_year:
            selected.append(fp)
    return sorted(selected)


def to_daily_timestamps(time_values: np.ndarray) -> np.ndarray:
    """Convert model time coordinate values to normalized pandas daily timestamps."""
    out = []
    for t in time_values:
        ts = pd.NaT

        # HIRHAM5 numeric time format: YYYYMMDD.fraction (e.g. 20000101.875)
        if isinstance(t, (float, np.floating, int, np.integer)):
            try:
                t_float = float(t)
                ymd = int(np.floor(t_float))
                if 10000101 <= ymd <= 99991231:
                    ts = pd.to_datetime(str(ymd), format="%Y%m%d")
            except Exception:
                ts = pd.NaT

        # cftime-style objects
        year_attr = getattr(t, "year", None)
        month_attr = getattr(t, "month", None)
        day_attr = getattr(t, "day", None)
        if pd.isna(ts) and year_attr is not None and month_attr is not None and day_attr is not None:
            try:
                ts = pd.Timestamp(year=int(year_attr), month=int(month_attr), day=int(day_attr))
            except Exception:
                ts = pd.NaT

        if pd.isna(ts):
            try:
                ts = pd.to_datetime(str(t))
            except Exception:
                ts = pd.NaT

        if pd.isna(ts):
            try:
                ts = pd.to_datetime(str(t)[:10], format="%Y-%m-%d")
            except Exception:
                ts = pd.NaT

        if pd.isna(ts):
            out.append(pd.NaT)
        else:
            out.append(pd.Timestamp(ts).normalize())

    return np.array(out, dtype=object)


def normalize_model_albedo(arr: np.ndarray) -> np.ndarray:
    """Normalize model albedo units to fraction if data appear to be in percent."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    p99 = float(np.nanpercentile(finite, 99))
    if p99 > 1.5:
        return arr / 100.0
    return arr


def infer_model_crs(ds: xr.Dataset) -> str:
    crs = ds.attrs.get("crs")
    if isinstance(crs, str) and crs.strip():
        return crs

    for coord_name in ("spatial_ref", "crs"):
        if coord_name in ds.coords:
            candidate = ds.coords[coord_name].attrs.get("spatial_ref")
            if isinstance(candidate, str) and candidate.strip():
                return candidate

    raise ValueError("Could not infer model CRS from NetCDF metadata.")


def build_model_grid(ds: xr.Dataset) -> ModelGrid:
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected 1D model coordinates x and y.")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Model x/y coordinates must have at least 2 elements.")

    x_res = float(np.median(np.abs(np.diff(x))))
    y_res = float(np.median(np.abs(np.diff(y))))
    if not np.isfinite(x_res) or not np.isfinite(y_res) or x_res <= 0 or y_res <= 0:
        raise ValueError("Invalid model grid spacing inferred from x/y coordinates.")

    x_left = float(np.min(x) - 0.5 * x_res)
    y_top = float(np.max(y) + 0.5 * y_res)
    transform = Affine(x_res, 0.0, x_left, 0.0, -y_res, y_top)

    return ModelGrid(
        shape=(len(y), len(x)),
        transform=transform,
        crs=infer_model_crs(ds),
        y_increasing=bool(y[1] > y[0]),
    )


def write_daily_geotiff(output_tif: str, grid: ModelGrid, arr: np.ndarray, model_name: str, day: pd.Timestamp) -> None:
    profile = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": np.nan,
        "compress": "LZW",
        "predictor": 3,
    }

    with rio.open(output_tif, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.set_band_description(1, "albedo")
        dst.update_tags(MODEL=model_name, DATE=day.strftime("%Y-%m-%d"), VARIABLE=MODEL_VAR_NAME)


def process_single_year_file(nc_path: str, model_name: str, output_dir: str) -> Tuple[int, int, int]:
    written = 0
    skipped = 0
    failed = 0

    try:
        ds = xr.open_dataset(nc_path)
    except Exception:
        return written, skipped, failed + 1

    if MODEL_VAR_NAME not in ds.variables:
        ds.close()
        return written, skipped, failed + 1

    try:
        grid = build_model_grid(ds)
    except Exception:
        ds.close()
        return written, skipped, failed + 1

    times = to_daily_timestamps(ds["time"].to_numpy())

    for i, day in enumerate(times):
        if pd.isna(day):
            skipped += 1
            continue

        day_ts = pd.Timestamp(day)
        out_tif = os.path.join(output_dir, f"{model_name.lower()}_albedo_{day_ts.strftime('%Y%m%d')}_modelgrid.tif")
        if os.path.exists(out_tif) and not OVERWRITE_EXISTING_FILES:
            skipped += 1
            continue

        try:
            arr = ds[MODEL_VAR_NAME].isel(time=i).to_numpy().astype(np.float32)
            # if grid.y_increasing:
            #     arr = np.flipud(arr)

            arr = normalize_model_albedo(arr)
            arr[(arr <= ALBEDO_MIN) | (arr >= ALBEDO_MAX)] = np.nan

            if not np.any(np.isfinite(arr)):
                if OVERWRITE_EXISTING_FILES and os.path.exists(out_tif):
                    os.remove(out_tif)
                skipped += 1
                continue

            write_daily_geotiff(out_tif, grid, arr, model_name, day_ts)
            written += 1
        except Exception:
            failed += 1

    ds.close()
    return written, skipped, failed


def run_model_conversion(model_name: str, model_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model_files = filter_by_year_range(
        glob.glob(os.path.join(model_dir, "*.nc")),
        YEAR_START,
        YEAR_END,
    )
    if len(model_files) == 0:
        raise FileNotFoundError(f"No model NetCDF files found in {model_dir}")

    print(f"[{model_name}] Yearly files: {len(model_files)}")
    print(f"[{model_name}] Output dir: {output_dir}")

    written_total = 0
    skipped_total = 0
    failed_total = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_single_year_file, nc_path, model_name, output_dir)
            for nc_path in model_files
        ]

        with tqdm(total=len(futures), desc=f"{model_name}: converting", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    written, skipped, failed = future.result()
                except Exception as exc:
                    written, skipped, failed = 0, 0, 1
                    tqdm.write(f"[{model_name}] worker failed: {type(exc).__name__}: {exc}")

                written_total += written
                skipped_total += skipped
                failed_total += failed
                pbar.update(1)

    print(f"[{model_name}] Done. written={written_total}, skipped={skipped_total}, failed={failed_total}")


def main() -> None:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print(f"Year filter: {YEAR_START}-{YEAR_END}")
    print(f"Workers: {NUM_WORKERS}")

    run_model_conversion("HCLIM", HCLIM_DIR, OUTPUT_HCLIM_DIR)
    run_model_conversion("HIRHAM5", HIRHAM5_DIR, OUTPUT_HIRHAM5_DIR)

    print("All model NetCDF-to-GeoTIFF conversions completed.")


if __name__ == "__main__":
    main()
