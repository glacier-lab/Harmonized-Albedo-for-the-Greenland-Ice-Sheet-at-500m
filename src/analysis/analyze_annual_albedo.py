"""
Time series of annual HSA500m statistics.

Bands in hsa500m_annual_YYYY.tif:
    1: annual average
    2: annual standard deviation
    3: number of days with albedo < 0.451 (dark ice days)
    4: number of days with albedo < 0.565 (melt season days)

Fig 1: ice-sheet mean of the annual average, with the mean annual std as error bar.
Fig 2: ice-sheet mean dark ice days and melt season days.
"""
#%%
import glob
import os

import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sns.set_theme(style="darkgrid", font_scale=1.5)

INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_annual"

#%%
files = sorted(glob.glob(os.path.join(INPUT_DIR, "hsa500m_annual_*.tif")))
if not files:
    raise FileNotFoundError(f"No annual files found in {INPUT_DIR}")

def box_stats(values, label):
    """Exact boxplot statistics (Tukey whiskers) from every valid pixel."""
    values = values[~np.isnan(values)]
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    return {
        "label": label,
        "q1": q1, "med": med, "q3": q3,
        "whislo": values[values >= q1 - 1.5 * iqr].min(),
        "whishi": values[values <= q3 + 1.5 * iqr].max(),
        "fliers": [],
    }


rows = []
box = {"albedo": [], "dark_ice_albedo": [], "dark_ice_days": [], "melt_season_days": []}
for fp in tqdm(files, desc="Reading annual files", unit="file"):
    try:
        year = int(os.path.basename(fp).split("_")[-1].replace(".tif", ""))
    except ValueError:
        continue
    with rio.open(fp) as src:
        data = src.read([1, 2, 3, 4]).astype("float32")
        immask = data[3] == 0  # melt season days == 0 means not bare ice, so mask out those pixels
        # data[1:3, immask] = np.nan
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        pixel_area_km2 = abs(src.res[0] * src.res[1]) / 1e6  # assumes projected CRS in metres
        print(f"{year}: {np.count_nonzero(~np.isnan(data[0]))} valid pixels, "
              f"{np.count_nonzero(data[2] > 0)} dark ice pixels, "
              f"{np.count_nonzero(data[3] > 0)} melt season pixels"
              f" ({pixel_area_km2:.3f} km² per pixel)")
    box["albedo"].append(box_stats(data[0].ravel(), year))
    box["dark_ice_albedo"].append(box_stats(data[0][data[2] > 0], year))
    box["dark_ice_days"].append(box_stats(data[2].ravel(), year))
    box["melt_season_days"].append(box_stats(data[3].ravel(), year))
    rows.append({
        "year": year,
        "mean": np.nanmean(data[0]),
        "std": np.nanmean(data[1]),
        "dark_ice_days": np.nanmean(data[2]),
        "dark_ice_days_std": np.nanstd(data[2]),
        "melt_season_days": np.nanmean(data[3]),
        "melt_season_days_std": np.nanstd(data[3]),
        "dark_ice_area": np.count_nonzero(data[2] > 0) * pixel_area_km2,
        "bare_ice_area": np.count_nonzero(data[3] > 0) * pixel_area_km2,
    })

df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

#%%
# Figure 1: annual mean with std error bars
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.errorbar(df["year"], df["mean"], yerr=df["std"], fmt="o-", capsize=4,
             color=sns.color_palette()[0])
ax1.set(xlabel="Year", ylabel="Albedo", title="Annual mean albedo (± std)")
fig1.tight_layout()

#%%
# Figure 2: dark ice days and melt season days in two subplots
fig2, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
axes[0].errorbar(df["year"], df["dark_ice_days"], yerr=df["dark_ice_days_std"], fmt="o-", capsize=4,
                 color=sns.color_palette()[3])
# sns.lineplot(data=df, x="year", y="dark_ice_days", marker="o", ax=axes[0],
#              color=sns.color_palette()[3])
axes[0].set(xlabel="", ylabel="Days", title="Dark ice days (albedo < 0.451)")
axes[1].errorbar(df["year"], df["melt_season_days"], yerr=df["melt_season_days_std"], fmt="o-", capsize=4,
                 color=sns.color_palette()[1])
#
# sns.lineplot(data=df, x="year", y="melt_season_days", marker="o", ax=axes[1],
#              color=sns.color_palette()[1])
axes[1].set(xlabel="Year", ylabel="Days", title="Melt season days (albedo < 0.565)")
fig2.tight_layout()

plt.show()

# %%
# Figures 3-5: boxplots per year computed from every unmasked pixel (no sampling)
for key, ylabel, title, color, area_col in [
    ("albedo", "Albedo", "Annual mean albedo", sns.color_palette()[0], None),
    ("dark_ice_albedo", "Albedo", "Annual mean albedo of dark ice areas (dark ice days > 0)", sns.color_palette()[4], None),
    ("dark_ice_days", "Days", "Dark ice days (albedo < 0.451)", sns.color_palette()[3], "dark_ice_area"),
    ("melt_season_days", "Days", "Bare ice days / melt season days (albedo < 0.565)", sns.color_palette()[1], "bare_ice_area"),
]:
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.bxp(box[key], showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set(xlabel="Year", ylabel=ylabel, title=title)
    ax.tick_params(axis="x", rotation=45)
    if area_col is not None:
        # box positions are 1..N in the same order as df (both sorted by year)
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(df) + 1), df[area_col] / 1e3, "k-o", lw=2)
        ax2.set_ylabel("Area (10$^3$ km$^2$)")
        ax2.grid(False)
    fig.tight_layout()

plt.show()