"""
High-performance Mann-Kendall trend detection for HSA500m on large servers.

Design goals:
- Use ~40% of a 255-thread / 1 TB server
- Avoid computationally intractable daily MK (O(n^2) with n=9132)
- Run MK on monthly aggregates (default JJA only) for scientific validity
- Keep file handles safe (open_mfdataset) and avoid chunk-misalignment warnings

Outputs (3-band GeoTIFF):
  Band 1: mk_tau
  Band 2: mk_pvalue
  Band 3: mk_sens_slope_per_year
"""

import glob
import os
import re
from typing import List, Optional

import dask
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray  # noqa: F401 - registers rasterio engine for xarray
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask.diagnostics.progress import ProgressBar
from numba import float32, guvectorize, int64
from scipy import stats


# -----------------------------------------------------------------------------
# Paths and filenames
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
INPUT_GLOB = "hsa500m_gapfilled_*.tif"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/trend_monthly"
OUTPUT_TIF = "hsa500m_mk_hpc_monthly_jja.tif"

BAND_INDEX = 1
DATE_REGEX = r"hsa500m_gapfilled_(\d{8})\.tif"
DATE_FMT = "%Y%m%d"


# -----------------------------------------------------------------------------
# Scientific and processing options
# -----------------------------------------------------------------------------
# Default is JJA only to remove seasonal cycle and reduce autocorrelation.
MELT_SEASON_ONLY = True
MELT_MONTHS = (6, 7, 8)

# Monthly sample count thresholds
MIN_VALID_OBS = 10

# Valid albedo range
ALBEDO_MIN = 0.0
ALBEDO_MAX = 1.0


# -----------------------------------------------------------------------------
# HPC resource configuration (40% of a 255-thread / 1 TB node)
# -----------------------------------------------------------------------------
TOTAL_THREADS = 255
USE_FRACTION = 0.40
TARGET_THREADS = int(TOTAL_THREADS * USE_FRACTION)  # 102

# 17 workers x 6 threads = 102 total threads
N_WORKERS = 17
THREADS_PER_WORKER = 6

# 40% of 1 TB ~= 409 GB. 17 x 24 GB = 408 GB.
MEMORY_PER_WORKER = "24GB"

# Larger spatial chunks are better on large RAM nodes.
# Load stage uses time=1 (one file per chunk), then we rechunk to time=-1 post-monthly.
CHUNK_XY = 1024


def parse_date_from_name(file_name: str) -> Optional[pd.Timestamp]:
    match = re.search(DATE_REGEX, file_name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format=DATE_FMT)


def list_input_files(input_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No input files found: {os.path.join(input_dir, pattern)}")
    return files


def load_daily_stack(files: List[str]) -> xr.DataArray:
    valid_files: List[str] = []
    timestamps: List[pd.Timestamp] = []

    for fp in files:
        ts = parse_date_from_name(os.path.basename(fp))
        if ts is not None:
            valid_files.append(fp)
            timestamps.append(ts.normalize())

    if not valid_files:
        raise RuntimeError("No valid files after date parsing")

    print(f"Loading {len(valid_files)} GeoTIFFs via open_mfdataset...")
    ds = xr.open_mfdataset(
        valid_files,
        concat_dim="time",
        combine="nested",
        engine="rasterio",
    )
    ds = ds.assign_coords(time=pd.to_datetime(timestamps))

    da = ds["band_data"].sel(band=BAND_INDEX, drop=True)
    da = da.astype(np.float32)
    da = da.where((da > ALBEDO_MIN) & (da < ALBEDO_MAX))
    da = da.sortby("time")

    # Rechunk after load to avoid on-disk tile misalignment warnings.
    return da.chunk({"time": 1, "x": CHUNK_XY, "y": CHUNK_XY})


def build_monthly_series(daily: xr.DataArray) -> xr.DataArray:
    print("Resampling daily -> monthly means...")
    monthly = daily.resample(time="MS").mean(skipna=True)

    if MELT_SEASON_ONLY:
        monthly = monthly.sel(time=monthly.time.dt.month.isin(list(MELT_MONTHS)))
        label = f"JJA-only months {MELT_MONTHS}"
    else:
        label = "all months"

    print(f"Monthly timesteps after filter ({label}): {len(monthly.time)}")

    # MK needs the full time vector per pixel.
    return monthly.chunk({"time": -1, "x": CHUNK_XY, "y": CHUNK_XY})


@guvectorize(
    [(float32[:], int64, float32[:], float32[:], float32[:])],
    "(n),()->(),(),()",
    nopython=True,
    cache=True,
)
def mann_kendall_numba(y, min_n, tau, z, sens_slope_per_year):
    # Count finite observations
    valid_count = 0
    for i in range(len(y)):
        if np.isfinite(y[i]):
            valid_count += 1

    if valid_count < min_n:
        tau[0] = np.nan
        z[0] = np.nan
        sens_slope_per_year[0] = np.nan
        return

    x = np.empty(valid_count, dtype=np.float32)
    idx = 0
    for i in range(len(y)):
        if np.isfinite(y[i]):
            x[idx] = y[i]
            idx += 1

    n = valid_count
    s = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    x_sorted = np.sort(x)
    tie_term = 0.0
    run = 1
    for i in range(1, n):
        if x_sorted[i] == x_sorted[i - 1]:
            run += 1
        else:
            if run > 1:
                tie_term += run * (run - 1) * (2 * run + 5)
            run = 1
    if run > 1:
        tie_term += run * (run - 1) * (2 * run + 5)

    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        tau[0] = 0.0
        z[0] = 0.0
        sens_slope_per_year[0] = 0.0
        return

    if s > 0:
        z[0] = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z[0] = (s + 1.0) / np.sqrt(var_s)
    else:
        z[0] = 0.0

    tau[0] = s / (0.5 * n * (n - 1))

    # Sen's slope in albedo per month-step
    n_pairs = n * (n - 1) // 2
    slopes = np.empty(n_pairs, dtype=np.float32)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes[k] = (x[j] - x[i]) / float(j - i)
            k += 1

    # Convert month-step slope to year slope for easier interpretation.
    sens_slope_per_year[0] = np.median(slopes) * 12.0


def run_mann_kendall(monthly: xr.DataArray) -> xr.Dataset:
    print("Computing Mann-Kendall + Sen slope (Numba, Dask parallelized)...")
    mk_tau, mk_z, mk_sens_slope_per_year = xr.apply_ufunc(
        mann_kendall_numba,
        monthly,
        MIN_VALID_OBS,
        input_core_dims=[["time"], []],
        output_core_dims=[[], [], []],
        dask="parallelized",
        output_dtypes=[np.float32, np.float32, np.float32],
    )

    print("Computing MK p-values...")
    mk_pvalue = 2.0 * (1.0 - xr.apply_ufunc(
        lambda z_stat: stats.norm.cdf(np.abs(z_stat)),
        mk_z,
        dask="parallelized",
        output_dtypes=[np.float32],
    ))

    ds = xr.Dataset(
        {
            "mk_tau": mk_tau,
            "mk_pvalue": mk_pvalue,
            "mk_sens_slope_per_year": mk_sens_slope_per_year,
        }
    )

    ds["mk_tau"].attrs["long_name"] = "Mann-Kendall Kendall tau"
    ds["mk_pvalue"].attrs["long_name"] = "Mann-Kendall two-tailed p-value"
    ds["mk_sens_slope_per_year"].attrs.update(
        {
            "long_name": "Sen slope (median pairwise slope)",
            "units": "albedo_per_year",
        }
    )

    return ds


def write_output(ds: xr.Dataset, output_path: str, template_da: xr.DataArray) -> None:
    band_names = ["mk_tau", "mk_pvalue", "mk_sens_slope_per_year"]
    stacked = xr.concat([ds[name] for name in band_names], dim="band")
    stacked = stacked.assign_coords(band=np.arange(1, len(band_names) + 1, dtype=np.int16))

    if template_da.rio.crs is not None:
        stacked = stacked.rio.write_crs(template_da.rio.crs, inplace=False)

    print(f"Writing output: {output_path}")
    with ProgressBar():
        stacked.rio.to_raster(
            output_path,
            dtype="float32",
            compress="LZW",
            predictor=3,
            nodata=np.nan,
            tiled=True,
            windowed=True,
        )

    with rio.open(output_path, "r+") as dst:
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_TIF)

    print("=" * 70)
    print("HSA500m MK HPC: Monthly aggregation + Numba + Dask Distributed")
    print("=" * 70)
    print(f"Target threads: {TARGET_THREADS} ({N_WORKERS} x {THREADS_PER_WORKER})")
    print(f"Memory per worker: {MEMORY_PER_WORKER}")

    dask.config.set({"temporary_directory": "/data_3/shunan_2/dask_temp"})
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        processes=True,
        memory_limit=MEMORY_PER_WORKER,
        local_directory="/data_3/shunan_2/dask_temp",
        dashboard_address=":0",
    )
    client = Client(cluster)

    try:
        files = list_input_files(INPUT_DIR, INPUT_GLOB)
        print(f"Input files: {len(files)}")

        daily = load_daily_stack(files)
        monthly = build_monthly_series(daily)

        ds = run_mann_kendall(monthly)
        write_output(ds, output_path, monthly)

        print("Done!")
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
