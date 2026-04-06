"""
Detect per-pixel trends in monthly-aggregated HSA500m albedo.

WHY MONTHLY INSTEAD OF DAILY?
-------------------------------
Applying Mann-Kendall to raw daily data is statistically problematic because
consecutive days are strongly autocorrelated. This inflates the MK S-statistic
and produces over-confident p-values (false positives for significance). Monthly
means substantially reduce this autocorrelation.

Seasonal autocorrelation (every July > every August regardless of trend) still
exists in an all-months series. The scientifically cleanest approach for
Greenland albedo is to restrict to the MELT SEASON (June–August, JJA), which:
  - Is when solar insolation drives ice melt and albedo is the key control
  - Removes the seasonal cycle entirely, leaving only inter-annual variability
  - Is standard in published Greenland albedo trend literature
  - Gives ~75 time points (25 yr × 3 months) — sufficient for MK power

Set MELT_SEASON_ONLY = True  → JJA monthly means  (25 yr × 3 mon = ~75 points)
Set MELT_SEASON_ONLY = False → all monthly means   (25 yr × 12 mon = ~300 points)

Outputs (6-band GeoTIFF):
  Band 1: linear_slope_per_year  (albedo/year, OLS on decimal-year axis)
  Band 2: linear_pvalue          (OLS two-tailed p-value)
  Band 3: mk_tau                 (Mann-Kendall Kendall tau)
  Band 4: mk_z                   (Mann-Kendall Z-score)
  Band 5: mk_pvalue              (Mann-Kendall two-tailed p-value)
  Band 6: mk_sens_slope          (Sen's slope, albedo/month-step)
"""

import glob
import os
import re
from typing import List, Optional, Tuple

import dask
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray  # noqa: F401 — registers the "rasterio" engine for xr.open_mfdataset
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from numba import guvectorize, float32, int64
from scipy import stats


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
INPUT_GLOB = "hsa500m_gapfilled_*.tif"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/trend_monthly"

# True  → JJA (June-Aug) monthly means only; output: *_monthly_jja.tif
# False → all calendar months;               output: *_monthly_all.tif
MELT_SEASON_ONLY = True
MELT_MONTHS = (6, 7, 8)  # June, July, August

BAND_INDEX = 1
DATE_REGEX = r"hsa500m_gapfilled_(\d{8})\.tif"
DATE_FMT = "%Y%m%d"

# With monthly data (~75–300 time steps) require at least 10 valid months.
MIN_VALID_OBS = 10
ALBEDO_MIN = 0.0
ALBEDO_MAX = 1.0

CHUNKS = {"x": 256, "y": 256}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_daily_timeseries(files: List[str]) -> xr.DataArray:
    """Load daily GeoTIFFs as a lazy Dask-backed DataArray."""
    valid_files: List[str] = []
    timestamps: List[pd.Timestamp] = []
    for fp in files:
        ts = parse_date_from_name(os.path.basename(fp))
        if ts is not None:
            valid_files.append(fp)
            timestamps.append(ts.normalize())

    if not valid_files:
        raise RuntimeError("No valid input files after date parsing")

    print(f"Loading {len(valid_files)} daily GeoTIFFs via open_mfdataset...")
    # Pass only the dimension name — xarray iterates over pd.Index as
    # 9132 individual items, causing "concat_dims has length 9132" ValueError.
    # Assign actual timestamps after loading instead.
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
    # Rechunk after load to avoid misalignment with on-disk GeoTIFF tile boundaries
    return da.chunk({"time": 1, "x": CHUNKS["x"], "y": CHUNKS["y"]})


def build_monthly_timeseries(
    daily: xr.DataArray,
    melt_season_only: bool,
    melt_months: tuple,
) -> xr.DataArray:
    """
    Aggregate daily → monthly means, then optionally keep only melt-season months.

    Monthly mean uses nanmean semantics: a month is NaN only if ALL its days
    within that pixel are NaN (cloud/off-ice). Months with some valid days
    produce a reduced-sample mean — acceptable for trend analysis.
    """
    print("Resampling daily → monthly means...")
    # 'MS' = month-start frequency; mean ignores NaN via skipna=True (default)
    monthly = daily.resample(time="MS").mean(skipna=True)

    if melt_season_only:
        monthly = monthly.sel(time=monthly.time.dt.month.isin(list(melt_months)))
        label = f"melt season (months {melt_months})"
    else:
        label = "all calendar months"

    n_steps = len(monthly.time)
    print(f"Monthly time steps ({label}): {n_steps}")

    print("Rechunking for time-series operations (time=-1)...")
    return monthly.chunk({"time": -1, "x": CHUNKS["x"], "y": CHUNKS["y"]})


# -----------------------------------------------------------------------------
# Trend statistics
# -----------------------------------------------------------------------------

def calculate_linear_trend_vectorized(
    y: xr.DataArray, min_n: int
) -> Tuple[xr.DataArray, xr.DataArray]:
    """OLS slope (albedo/year) and two-tailed p-value via vectorized Xarray math."""
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
        output_dtypes=[np.float32],
    ))

    slope = slope.where(n >= min_n).astype(np.float32)
    pvalue = pvalue.where(n >= min_n).astype(np.float32)
    return slope, pvalue


@guvectorize(
    [(float32[:], int64, float32[:], float32[:], float32[:])],
    "(n),()->(),(),()",
    nopython=True,
    cache=True,
)
def mann_kendall_numba(y, min_n, tau, z, sens_slope):
    """Numba-compiled Mann-Kendall + Sen's slope for a single pixel time series."""
    valid_count = 0
    for i in range(len(y)):
        if np.isfinite(y[i]):
            valid_count += 1

    if valid_count < min_n:
        tau[0] = np.nan
        z[0] = np.nan
        sens_slope[0] = np.nan
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
        sens_slope[0] = 0.0
        return

    if s > 0:
        z[0] = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z[0] = (s + 1.0) / np.sqrt(var_s)
    else:
        z[0] = 0.0

    tau[0] = s / (0.5 * n * (n - 1))

    # Sen's slope: median of all pairwise slopes (x[j]-x[i]) / (j-i)
    n_pairs = n * (n - 1) // 2
    pairwise = np.empty(n_pairs, dtype=np.float32)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            pairwise[k] = (x[j] - x[i]) / float(j - i)
            k += 1
    sens_slope[0] = np.median(pairwise)


def run_trend_detection(hsa: xr.DataArray, min_valid_obs: int) -> xr.Dataset:
    print("Computing linear trends (vectorized)...")
    linear_slope, linear_pvalue = calculate_linear_trend_vectorized(hsa, min_valid_obs)

    print("Computing Mann-Kendall + Sen's slope (Numba-compiled)...")
    mk_tau, mk_z, mk_sens_slope = xr.apply_ufunc(
        mann_kendall_numba,
        hsa,
        min_valid_obs,
        input_core_dims=[["time"], []],
        output_core_dims=[[], [], []],
        dask="parallelized",
        output_dtypes=[np.float32, np.float32, np.float32],
    )

    print("Computing Mann-Kendall p-values...")
    mk_pvalue = 2.0 * (1.0 - xr.apply_ufunc(
        lambda z_stat: stats.norm.cdf(np.abs(z_stat)),
        mk_z,
        dask="parallelized",
        output_dtypes=[np.float32],
    ))

    ds = xr.Dataset({
        "linear_slope_per_year": linear_slope,
        "linear_pvalue": linear_pvalue,
        "mk_tau": mk_tau,
        "mk_z": mk_z,
        "mk_pvalue": mk_pvalue,
        "mk_sens_slope": mk_sens_slope,
    })

    ds["linear_slope_per_year"].attrs.update({
        "long_name": "OLS slope per decimal year",
        "units": "albedo_per_year",
    })
    ds["linear_pvalue"].attrs["long_name"] = "OLS two-tailed p-value"
    ds["mk_tau"].attrs["long_name"] = "Mann-Kendall Kendall tau"
    ds["mk_z"].attrs["long_name"] = "Mann-Kendall Z-score"
    ds["mk_pvalue"].attrs["long_name"] = "Mann-Kendall two-tailed p-value"
    ds["mk_sens_slope"].attrs.update({
        "long_name": "Sen's slope (median pairwise slope per monthly timestep)",
        "units": "albedo_per_month",
    })
    return ds


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def write_multiband_trend_geotiff(
    ds: xr.Dataset, output_path: str, template_da: xr.DataArray
) -> None:
    band_names = [
        "linear_slope_per_year",
        "linear_pvalue",
        "mk_tau",
        "mk_z",
        "mk_pvalue",
        "mk_sens_slope",
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


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    dask.config.set(scheduler="threads")

    suffix = "monthly_jja" if MELT_SEASON_ONLY else "monthly_all"
    output_tif = f"hsa500m_trend_ols_mk_{suffix}.tif"
    output_path = os.path.join(OUTPUT_DIR, output_tif)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    mode = "JJA melt-season months" if MELT_SEASON_ONLY else "all calendar months"
    print(f"HSA500m Monthly Trend Detection: Linear + Mann-Kendall ({mode})")
    print("=" * 70)

    files = list_input_files(INPUT_DIR, INPUT_GLOB)
    print(f"Input files: {len(files)}")

    daily = load_daily_timeseries(files)
    hsa_monthly = build_monthly_timeseries(daily, MELT_SEASON_ONLY, MELT_MONTHS)

    ds_trend = run_trend_detection(hsa_monthly, min_valid_obs=MIN_VALID_OBS)
    write_multiband_trend_geotiff(ds_trend, output_path, hsa_monthly)

    print(f"Done! Output: {output_path}")


if __name__ == "__main__":
    main()
