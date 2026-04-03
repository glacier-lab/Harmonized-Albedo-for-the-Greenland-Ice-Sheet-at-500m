"""
Detect per-pixel trends in the HSA500m gapfilled time series.

Optimized for 20-year daily time series across Greenland Ice Sheet:
- Linear trend: Native Xarray/Dask vectorized operations
- Mann-Kendall: Numba-compiled @guvectorize
- Memory Management: Dask Distributed Client + Optimized spatial chunks
"""

import glob
import os
import re
from typing import List, Optional, Tuple, cast

import dask
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from numba import guvectorize, float32
from scipy import stats


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
INPUT_GLOB = "hsa500m_gapfilled_*.tif"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/trend"
OUTPUT_TIF = "hsa500m_trend_ols_mk.tif"

BAND_INDEX = 1
DATE_REGEX = r"hsa500m_gapfilled_(\d{8})\.tif"
DATE_FMT = "%Y%m%d"

MIN_VALID_OBS = 30
ALBEDO_MIN = 0.0
ALBEDO_MAX = 1.0

# 256x256 * 7300 days * 4 bytes = ~1.9 GB per chunk. Safe for Dask workers.
CHUNKS = {"x": 256, "y": 256}


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


def decimal_year(times: pd.DatetimeIndex) -> np.ndarray:
    year = times.year.to_numpy(dtype=np.float64)
    day = times.dayofyear.to_numpy(dtype=np.float64)
    return year + (day - 1.0) / 365.25


def load_hsa_timeseries(files: List[str]) -> xr.DataArray:
    arrays: List[xr.DataArray] = []
    timestamps: List[pd.Timestamp] = []

    print("Loading GeoTIFFs into lazy xarray stack...")
    for fp in files:
        ts = parse_date_from_name(os.path.basename(fp))
        if ts is None:
            continue

        da = cast(xr.DataArray, rxr.open_rasterio(fp, chunks=CHUNKS))
        if "band" in da.dims:
            da = da.sel(band=BAND_INDEX, drop=True)

        da = da.astype(np.float32)
        da = da.where((da > ALBEDO_MIN) & (da < ALBEDO_MAX))

        arrays.append(da)
        timestamps.append(ts.normalize())

    if not arrays:
        raise RuntimeError("No valid input files after date parsing")

    hsa = xr.concat(arrays, dim="time")
    hsa = hsa.assign_coords(time=("time", pd.to_datetime(timestamps)))
    hsa = hsa.sortby("time")

    print("Rechunking for time-series operations (time=-1)...")
    return hsa.chunk({"time": -1, "x": CHUNKS["x"], "y": CHUNKS["y"]})


def calculate_linear_trend_vectorized(y: xr.DataArray, min_n: int) -> Tuple[xr.DataArray, xr.DataArray]:
    tvals = decimal_year(pd.DatetimeIndex(y.time.values))
    t = xr.DataArray(tvals.astype(np.float32), dims=["time"], coords={"time": y.time})
    
    valid = y.notnull() & t.notnull()
    n = valid.sum(dim="time")

    y_valid = y.where(valid)
    t_valid = t.where(valid)

    t_mean = t_valid.mean(dim="time")
    y_mean = y_valid.mean(dim="time")

    t_centered = t_valid - t_mean
    y_centered = y_valid - y_mean

    ss_t = (t_centered ** 2).sum(dim="time")
    ss_ty = (t_centered * y_centered).sum(dim="time")

    slope = ss_ty / ss_t.where(ss_t > 0)
    intercept = y_mean - slope * t_mean

    y_hat = slope * t_valid + intercept
    rss = ((y_valid - y_hat) ** 2).sum(dim="time")
    
    df = n - 2
    se_slope = xr.ufuncs.sqrt((rss / df.clip(min=1)) / ss_t.where(ss_t > 0))
    t_stat = slope / se_slope.where(se_slope > 0)

    pvalue = 2.0 * (1.0 - xr.apply_ufunc(
        lambda ts, d: stats.t.cdf(np.abs(ts), df=d),
        t_stat, df,
        dask="parallelized",
        output_dtypes=[np.float32]
    ))

    slope = slope.where(n >= min_n).astype(np.float32)
    pvalue = pvalue.where(n >= min_n).astype(np.float32)

    return slope, pvalue


@guvectorize(
    ["void(float32[:], int64, float32[:], float32[:])"],
    "(n),()->(),()",
    nopython=True,
    cache=True
)
def mann_kendall_numba(y, min_n, tau, z):
    valid_count = 0
    for i in range(len(y)):
        if np.isfinite(y[i]):
            valid_count += 1

    # FIX: Removed the [0] index. min_n is evaluated as a scalar here.
    if valid_count < min_n:
        tau[0] = np.nan
        z[0] = np.nan
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
    current_count = 1
    for i in range(1, n):
        if x_sorted[i] == x_sorted[i - 1]:
            current_count += 1
        else:
            if current_count > 1:
                tie_term += current_count * (current_count - 1) * (2 * current_count + 5)
            current_count = 1
    if current_count > 1:
        tie_term += current_count * (current_count - 1) * (2 * current_count + 5)

    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        tau[0] = 0.0
        z[0] = 0.0
        return

    if s > 0:
        z[0] = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z[0] = (s + 1.0) / np.sqrt(var_s)
    else:
        z[0] = 0.0

    tau[0] = s / (0.5 * n * (n - 1))


def run_trend_detection(hsa: xr.DataArray, min_valid_obs: int) -> xr.Dataset:
    print("Computing linear trends (vectorized)...")
    linear_slope, linear_pvalue = calculate_linear_trend_vectorized(hsa, min_valid_obs)

    print("Computing Mann-Kendall trends (Numba-compiled)...")
    mk_tau, mk_z = xr.apply_ufunc(
        mann_kendall_numba,
        hsa,
        min_valid_obs,
        input_core_dims=[["time"], []],
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[np.float32, np.float32],
    )

    print("Computing Mann-Kendall p-values...")
    mk_pvalue = 2.0 * (1.0 - xr.apply_ufunc(
        lambda z_stat: stats.norm.cdf(np.abs(z_stat)),
        mk_z,
        dask="parallelized",
        output_dtypes=[np.float32]
    ))

    ds = xr.Dataset(
        {
            "linear_slope_per_year": linear_slope,
            "linear_pvalue": linear_pvalue,
            "mk_tau": mk_tau,
            "mk_z": mk_z,
            "mk_pvalue": mk_pvalue,
        }
    )

    ds["linear_slope_per_year"].attrs.update({
        "long_name": "OLS slope per decimal year",
        "units": "albedo_per_year"
    })
    ds["linear_pvalue"].attrs["long_name"] = "OLS two-tailed p-value"
    ds["mk_tau"].attrs["long_name"] = "Mann-Kendall Kendall tau"
    ds["mk_z"].attrs["long_name"] = "Mann-Kendall Z-score"
    ds["mk_pvalue"].attrs["long_name"] = "Mann-Kendall two-tailed p-value"

    return ds


def write_multiband_trend_geotiff(ds: xr.Dataset, output_path: str, template_da: xr.DataArray) -> None:
    band_names = [
        "linear_slope_per_year",
        "linear_pvalue",
        "mk_tau",
        "mk_z",
        "mk_pvalue",
    ]

    print("Stacking bands and preparing output...")
    stacked = xr.concat([ds[name] for name in band_names], dim="band")
    stacked = stacked.assign_coords(band=np.arange(1, len(band_names) + 1, dtype=np.int16))

    if template_da.rio.crs is not None:
        stacked = stacked.rio.write_crs(template_da.rio.crs, inplace=False)

    print("Writing GeoTIFF...")
    with ProgressBar():
        stacked.rio.to_raster(
            output_path,
            dtype="float32",
            compress="LZW",
            predictor=3,
            nodata=np.nan,
            windowed=True, 
        )

    print("Adding band descriptions...")
    with rio.open(output_path, "r+") as dst:
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def main() -> None:
    # Use threaded scheduler (in-process, no TCP).
    # Avoids CommClosedError from inter-process data shuffles during rechunking.
    # Numba guvectorize releases the GIL, so threads parallelize efficiently.
    dask.config.set(scheduler='threads')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("HSA500m Trend Detection: Linear + Mann-Kendall")
    print("=" * 70)
    
    files = list_input_files(INPUT_DIR, INPUT_GLOB)
    print(f"Input files: {len(files)}")
    print(f"Initial spatial chunks: {CHUNKS}")
    print()

    hsa = load_hsa_timeseries(files)
    
    ds_trend = run_trend_detection(hsa, min_valid_obs=MIN_VALID_OBS)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_TIF)
    write_multiband_trend_geotiff(ds_trend, output_path, hsa)

    print("Done!")


if __name__ == "__main__":
    main()