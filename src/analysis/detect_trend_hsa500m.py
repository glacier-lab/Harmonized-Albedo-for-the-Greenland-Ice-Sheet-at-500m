"""
Detect per-pixel trends in the HSA500m gapfilled time series.

This script lazily loads daily GeoTIFF files (band 1: gapfilled albedo) into an
xarray+dask time stack and computes two trend diagnostics at each pixel:

1) Linear trend (ordinary least squares): slope and p-value
2) Mann-Kendall trend test: Kendall tau, Z score, and p-value

Output:
- One multiband GeoTIFF with trend layers:
  band 1: linear_slope_per_year
  band 2: linear_pvalue
  band 3: mk_tau
  band 4: mk_z
  band 5: mk_pvalue

Notes:
- Input filenames are expected as hsa500m_gapfilled_YYYYMMDD.tif.
- The script keeps processing lazy until write time.
- Minimum valid observations per pixel can be configured via MIN_VALID_OBS.
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

# Dask chunk sizes for lazy processing.
CHUNKS = {"x": 1024, "y": 1024}


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

    return hsa


def linear_trend_1d(y: np.ndarray, t: np.ndarray, min_n: int) -> Tuple[np.float32, np.float32]:
    valid = np.isfinite(y) & np.isfinite(t)
    n = int(np.sum(valid))
    if n < min_n:
        return np.float32(np.nan), np.float32(np.nan)

    x = t[valid].astype(np.float64)
    v = y[valid].astype(np.float64)

    x_mean = np.mean(x)
    y_mean = np.mean(v)
    x_centered = x - x_mean
    y_centered = v - y_mean

    ss_x = np.sum(x_centered ** 2)
    if ss_x <= 0:
        return np.float32(np.nan), np.float32(np.nan)

    slope = np.sum(x_centered * y_centered) / ss_x
    intercept = y_mean - slope * x_mean

    if n <= 2:
        return np.float32(slope), np.float32(np.nan)

    y_hat = slope * x + intercept
    rss = np.sum((v - y_hat) ** 2)
    se_slope = np.sqrt((rss / (n - 2)) / ss_x) if rss > 0 else 0.0

    if se_slope == 0:
        pvalue = 0.0
    else:
        t_stat = slope / se_slope
        pvalue = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=n - 2))

    return np.float32(slope), np.float32(pvalue)


def mann_kendall_1d(y: np.ndarray, min_n: int) -> Tuple[np.float32, np.float32, np.float32]:
    x = y[np.isfinite(y)]
    n = x.size
    if n < min_n:
        return np.float32(np.nan), np.float32(np.nan), np.float32(np.nan)

    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(x[i + 1 :] - x[i])))

    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        tau = 0.0
        z = 0.0
        p = 1.0
    else:
        if s > 0:
            z = (s - 1.0) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1.0) / np.sqrt(var_s)
        else:
            z = 0.0

        tau = s / (0.5 * n * (n - 1))
        p = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))

    return np.float32(tau), np.float32(z), np.float32(p)


def run_trend_detection(hsa: xr.DataArray, min_valid_obs: int) -> xr.Dataset:
    tvals = decimal_year(pd.DatetimeIndex(hsa.time.values))
    t_da = xr.DataArray(tvals.astype(np.float32), dims=["time"], coords={"time": hsa.time})

    linear_slope, linear_pvalue = xr.apply_ufunc(
        linear_trend_1d,
        hsa,
        t_da,
        kwargs={"min_n": min_valid_obs},
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32, np.float32],
    )

    mk_tau, mk_z, mk_pvalue = xr.apply_ufunc(
        mann_kendall_1d,
        hsa,
        kwargs={"min_n": min_valid_obs},
        input_core_dims=[["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32, np.float32, np.float32],
    )

    ds = xr.Dataset(
        {
            "linear_slope_per_year": linear_slope,
            "linear_pvalue": linear_pvalue,
            "mk_tau": mk_tau,
            "mk_z": mk_z,
            "mk_pvalue": mk_pvalue,
        }
    )

    ds["linear_slope_per_year"].attrs["long_name"] = "OLS slope per decimal year"
    ds["linear_pvalue"].attrs["long_name"] = "OLS p-value"
    ds["mk_tau"].attrs["long_name"] = "Mann-Kendall Kendall tau"
    ds["mk_z"].attrs["long_name"] = "Mann-Kendall Z score"
    ds["mk_pvalue"].attrs["long_name"] = "Mann-Kendall p-value"

    return ds


def write_multiband_trend_geotiff(ds: xr.Dataset, output_path: str, template_da: xr.DataArray) -> None:
    band_names = [
        "linear_slope_per_year",
        "linear_pvalue",
        "mk_tau",
        "mk_z",
        "mk_pvalue",
    ]

    stacked = xr.concat([ds[name] for name in band_names], dim="band")
    stacked = stacked.assign_coords(band=np.arange(1, len(band_names) + 1, dtype=np.int16))

    if template_da.rio.crs is not None:
        stacked = stacked.rio.write_crs(template_da.rio.crs, inplace=False)

    with ProgressBar():
        stacked.rio.to_raster(
            output_path,
            dtype="float32",
            compress="LZW",
            predictor=3,
            nodata=np.nan,
        )

    with rio.open(output_path, "r+") as dst:
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = list_input_files(INPUT_DIR, INPUT_GLOB)
    print(f"Input files: {len(files)}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output: {os.path.join(OUTPUT_DIR, OUTPUT_TIF)}")
    print(f"Min valid observations per pixel: {MIN_VALID_OBS}")
    print(f"Chunks: {CHUNKS}")

    hsa = load_hsa_timeseries(files)
    print(f"Loaded lazy cube with dims: {dict(hsa.sizes)}")
    print(f"Date range: {pd.to_datetime(hsa.time.values[0]).date()} -> {pd.to_datetime(hsa.time.values[-1]).date()}")

    ds_trend = run_trend_detection(hsa, min_valid_obs=MIN_VALID_OBS)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_TIF)
    write_multiband_trend_geotiff(ds_trend, output_path, hsa)

    print("Done.")
    print(f"Trend GeoTIFF: {output_path}")


if __name__ == "__main__":
    main()
