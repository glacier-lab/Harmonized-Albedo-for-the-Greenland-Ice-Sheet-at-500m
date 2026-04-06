import os
import glob
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
from scipy.stats import linregress
import pymannkendall as mk

# --- Configuration ---
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
OUTPUT_FILE = "greenland_albedo_trends.tif"
# Chunk size: (y, x). Adjust based on your RAM. Smaller chunks = less RAM but more overhead.
# Time MUST be -1 (unchunked) to perform time-series analysis on the whole sequence.
CHUNKS = {'time': -1, 'y': 250, 'x': 250} 

# 1. Locate files and parse dates
file_pattern = os.path.join(INPUT_DIR, "hsa500m_gapfilled_*.tif")
files = sorted(glob.glob(file_pattern))

if not files:
    raise ValueError("No files found. Check your directory path.")

print(f"Found {len(files)} files. Preparing lazy dataset...")

dates = []
for f in files:
    # Extract 'YYYYMMDD' from 'hsa500m_gapfilled_YYYYMMDD.tif'
    basename = os.path.basename(f)
    date_str = basename.split('_')[2].split('.')[0]
    dates.append(pd.to_datetime(date_str, format='%Y%m%d'))

# 2. Lazily load and concatenate the dataset using Dask
# rioxarray loads each file with dimensions (band, y, x). We squeeze 'band' out.
da_list = [rioxarray.open_rasterio(f, chunks={'y': CHUNKS['y'], 'x': CHUNKS['x']}).squeeze(drop=True) for f in files]

# Concatenate along a new 'time' dimension
da = xr.concat(da_list, dim=pd.Index(dates, name='time'))

# Rechunk to ensure time is contiguous in memory for our statistical functions
da = da.chunk(CHUNKS)

# 3. Define the statistical functions
# These functions expect a 1D numpy array representing the time series for a single pixel.
def compute_trends(y_array):
    """
    Computes Linear Regression and Mann-Kendall statistics for a 1D pixel time series.
    Returns: [lin_slope, lin_pvalue, mk_tau, mk_pvalue, mk_sens_slope]
    """
    # Handle entirely NaN pixels (e.g., ocean or off-ice boundaries)
    if np.isnan(y_array).all():
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # Linear Regression (using time index 0 to N)
    time_idx = np.arange(len(y_array))
    lin_res = linregress(time_idx, y_array)
    
    # Mann-Kendall Test
    try:
        # mk.original_test is robust but slow. It returns a named tuple.
        mk_res = mk.original_test(y_array)
        mk_tau = mk_res.tau
        mk_pval = mk_res.p
        mk_sens = mk_res.slope
    except Exception:
        # Fallback if MK fails (e.g., constant values)
        mk_tau, mk_pval, mk_sens = np.nan, np.nan, np.nan

    return np.array([lin_res.slope, lin_res.pvalue, mk_tau, mk_pval, mk_sens])

# 4. Apply the function across the entire Dask array
print("Building Dask computation graph for trend analysis...")
results = xr.apply_ufunc(
    compute_trends,
    da,
    input_core_dims=[['time']],      # Core dimension is time
    output_core_dims=[['stats']],    # Output dimension for the 5 stats
    vectorize=True,                  # Vectorize over x and y
    dask='parallelized',             # Allow dask to process chunks in parallel
    output_dtypes=[float],
    dask_gufunc_kwargs={'output_sizes': {'stats': 5}} # 5 output variables
)

# 5. Format the output into separate bands
print("Formatting output dataset...")
# results has dimensions (y, x, stats). We need to transpose to (stats, y, x) for GeoTIFF
results = results.transpose('stats', 'y', 'x')

# Assign coordinate names to the 'stats' dimension to identify the bands
results = results.assign_coords(stats=['lin_slope', 'lin_pvalue', 'mk_tau', 'mk_pvalue', 'mk_sens_slope'])

# Reattach the spatial projection and transform from the first image
results = results.rio.write_crs(da_list[0].rio.crs)
results = results.rio.write_transform(da_list[0].rio.transform())

# 6. Compute and Save to GeoTIFF
# This is where the actual computation happens. It will take significant time.
print("Executing computation and saving to disk. This will take a while...")

# Save out as a multi-band GeoTIFF
results.rio.to_raster(
    OUTPUT_FILE,
    tiled=True,
    windowed=True,
    compress='lzw' # Recommended for spatial data to save disk space
)

print(f"Processing complete! Saved to {OUTPUT_FILE}")