"""
Analyze annual minimum albedo to count pixels below threshold.

Reads Band 4 (annual_min) from hsa500m_annual_YYYY.tif files,
counts pixels below 0.431 (or custom threshold), and exports results as DataFrame/CSV.

Output: summary_albedo_analysis.csv
"""
#%%
import glob
import os

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", font_scale=1.5)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_annual"
THRESHOLD = 0.431
BAND_MIN = 4  # Band 4 is annual_min



files = sorted(glob.glob(os.path.join(INPUT_DIR, "hsa500m_annual_*.tif")))
if not files:
    raise FileNotFoundError(f"No annual files found in {INPUT_DIR}")

results = []

for fp in tqdm(files, desc="Analyzing annual files", unit="file"):
    basename = os.path.basename(fp)
    # Extract year from filename: hsa500m_annual_YYYY.tif
    year_str = basename.split("_")[-1].replace(".tif", "")
    try:
        year = int(year_str)
    except ValueError:
        continue

    with rio.open(fp) as src:
        # Read band 4 (annual_min)
        band_min = src.read(BAND_MIN).astype("float32")
    # convert to one dimensional array for easier processing
    band_min = band_min.flatten()
    # drop NaN values
    band_min = band_min[~np.isnan(band_min)]
    # Count valid pixels
    n_total = band_min.size

    # Count pixels below threshold
    below_threshold = band_min < THRESHOLD
    n_below = int(np.sum(below_threshold))

    # Percentage
    pct_below = (n_below / n_total * 100) if n_total > 0 else np.nan

    results.append({
        "year": year,
        "n_total_pixels": n_total,
        "n_pixels_below_threshold": n_below,
        "percent_below_threshold": pct_below,
        "threshold": THRESHOLD,
    })

df = pd.DataFrame(results)
df = df.sort_values("year").reset_index(drop=True)


# %%
sns.lineplot(data=df, x="year", y="n_pixels_below_threshold", marker="o")
# %%
