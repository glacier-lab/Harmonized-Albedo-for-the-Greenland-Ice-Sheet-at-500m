"""
Detect per-pixel trends in the HSA500m gapfilled time series.
SIMPLIFIED VERSION using scipy.stats.kendalltau (like Martin Jung's reference).

This version is simpler and uses scipy's built-in Mann-Kendall (kendalltau),
which is equivalent to the manual implementation but less code and well-tested.

Output:
- One multiband GeoTIFF with trend layers:
  band 1: linear_slope_per_year (albedo change per year)
  band 2: linear_pvalue (two-tailed t-test)
  band 3: mk_tau (Kendall's tau from scipy)
  band 4: mk_pvalue (from scipy kendalltau)

Input filenames expected: hsa500m_gapfilled_YYYYMMDD.tif
"""

import glob
import os
import re
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from numba import jit
from scipy import stats


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
INPUT_GLOB = "hsa500m_gapfilled_*.tif"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/trendscipy"
OUTPUT_TIF = "hsa500m_trend_ols_mk_scipy.tif"

BAND_INDEX = 1
DATE_REGEX = r"hsa500m_gapfilled_(\d{8})\.tif"
DATE_FMT = "%Y%m%d"

MIN_VALID_OBS = 30
ALBEDO_MIN = 0.0
ALBEDO_MAX = 1.0

CHUNKS = {"x": 512, "y": 512}


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
    """Vectorized OLS trend calculation using native xarray operations."""
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


@jit(nopython=True, nogil=True)
def mann_kendall_scipy_wrapper(y, time_indices, min_n):
    """
    Numba-optimized wrapper for scipy kendalltau.
    
    This matches Martin Jung's reference implementation but returns
    full tau and p-value instead of simplified direction.
    
    Note: scipy.stats.kendalltau cannot be called from numba, so we'll
    use this via apply_ufunc without numba compilation for kendalltau itself.
    """
    # Extract valid values
    valid_mask = np.isfinite(y)
    valid_count = np.sum(valid_mask)
    
    if valid_count < min_n:
        return np.float32(np.nan), np.float32(np.nan)
    
    y_valid = y[valid_mask]
    t_valid = time_indices[valid_mask]
    
    return y_valid, t_valid


def kendalltau_pixel(y, time_indices, min_n):
    """
    Calculate Kendall's tau using scipy (like Martin Jung's reference).
    
    This is mathematically equivalent to the manual Mann-Kendall implementation,
    but uses scipy's well-tested kendalltau function.
    """
    valid_mask = np.isfinite(y)
    valid_count = np.sum(valid_mask)
    
    if valid_count < min_n:
        return np.float32(np.nan), np.float32(np.nan)
    
    y_valid = y[valid_mask]
    t_valid = time_indices[valid_mask]
    
    # scipy.stats.kendalltau(x, y) tests correlation between x and y
    # For trend detection: kendalltau(data, time) = Mann-Kendall test
    tau, p_value = stats.kendalltau(y_valid, t_valid)
    
    return np.float32(tau), np.float32(p_value)


def calculate_mk_trend_scipy(hsa: xr.DataArray, min_valid_obs: int) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate Mann-Kendall using scipy.stats.kendalltau.
    
    This is the approach used in Martin Jung's reference script.
    Equivalent to manual Mann-Kendall but simpler and uses scipy's
    built-in implementation.
    """
    # Create time index array (1, 2, 3, ..., n)
    time_indices = xr.DataArray(
        np.arange(len(hsa.time), dtype=np.float32) + 1,
        dims=["time"],
        coords={"time": hsa.time}
    )
    
    mk_tau, mk_pvalue = xr.apply_ufunc(
        kendalltau_pixel,
        hsa,
        time_indices,
        min_valid_obs,
        input_core_dims=[["time"], ["time"], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32, np.float32],
    )
    
    return mk_tau, mk_pvalue


def run_trend_detection(hsa: xr.DataArray, min_valid_obs: int) -> xr.Dataset:
    print("Computing linear trends (vectorized)...")
    linear_slope, linear_pvalue = calculate_linear_trend_vectorized(hsa, min_valid_obs)

    print("Computing Mann-Kendall trends (scipy.stats.kendalltau)...")
    mk_tau, mk_pvalue = calculate_mk_trend_scipy(hsa, min_valid_obs)

    ds = xr.Dataset(
        {
            "linear_slope_per_year": linear_slope,
            "linear_pvalue": linear_pvalue,
            "mk_tau": mk_tau,
            "mk_pvalue": mk_pvalue,
        }
    )

    ds["linear_slope_per_year"].attrs.update({
        "long_name": "OLS slope per decimal year",
        "units": "albedo_per_year"
    })
    ds["linear_pvalue"].attrs["long_name"] = "OLS two-tailed p-value"
    ds["mk_tau"].attrs.update({
        "long_name": "Mann-Kendall Kendall tau (scipy.stats.kendalltau)",
        "description": "Equivalent to Martin Jung's reference implementation"
    })
    ds["mk_pvalue"].attrs["long_name"] = "Mann-Kendall two-tailed p-value"

    return ds


def write_multiband_trend_geotiff(ds: xr.Dataset, output_path: str, template_da: xr.DataArray) -> None:
    band_names = [
        "linear_slope_per_year",
        "linear_pvalue",
        "mk_tau",
        "mk_pvalue",
    ]

    print("Stacking bands and preparing output...")
    stacked = xr.concat([ds[name] for name in band_names], dim="band")
    stacked = stacked.assign_coords(band=np.arange(1, len(band_names) + 1, dtype=np.int16))

    if template_da.rio.crs is not None:
        stacked = stacked.rio.write_crs(template_da.rio.crs, inplace=False)

    print("Writing GeoTIFF (this may take 10-30 minutes)...")
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("HSA500m Trend Detection: Linear + Mann-Kendall (scipy version)")
    print("=" * 70)
    
    files = list_input_files(INPUT_DIR, INPUT_GLOB)
    print(f"Input files: {len(files)}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output: {os.path.join(OUTPUT_DIR, OUTPUT_TIF)}")
    print(f"Min valid observations per pixel: {MIN_VALID_OBS}")
    print(f"Initial spatial chunks: {CHUNKS}")
    print()

    hsa = load_hsa_timeseries(files)
    print(f"Loaded lazy cube with dims: {dict(hsa.sizes)}")
    print(f"Date range: {pd.to_datetime(hsa.time.values[0]).date()} -> {pd.to_datetime(hsa.time.values[-1]).date()}")
    print(f"Chunk structure: {hsa.chunks}")
    print()

    ds_trend = run_trend_detection(hsa, min_valid_obs=MIN_VALID_OBS)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_TIF)
    write_multiband_trend_geotiff(ds_trend, output_path, hsa)

    print()
    print("=" * 70)
    print("Done!")
    print(f"Trend GeoTIFF: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
