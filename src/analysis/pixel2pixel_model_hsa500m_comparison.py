#%%
import numpy as np
import vaex as vx
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib import colors
import contextily as ctx
import cmocean as cmo
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar

sns.set_theme(style="darkgrid", font_scale=1.3)

#%%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HCLIM_MAP_PATH = "/data_3/shunan_2/AU/hsa500m/Kristiina/model_hsa_comparison/maps/HCLIM/hclim_hsa500m_bias_rmse_2000_2025.tif"
HIRHAM5_MAP_PATH = "/data_3/shunan_2/AU/hsa500m/Kristiina/model_hsa_comparison/maps/HIRHAM5/hirham5_hsa500m_bias_rmse_2000_2025.tif"

HCLIM_HDF5_GLOB = "/data_3/shunan_2/AU/hsa500m/Kristiina/model_hsa_comparison/paired_hdf5/HCLIM/*.h5"
HIRHAM5_HDF5_GLOB = "/data_3/shunan_2/AU/hsa500m/Kristiina/model_hsa_comparison/paired_hdf5/HIRHAM5/*.h5"


def read_bias_rmse(map_path: str):
    with rio.open(map_path) as src:
        bias = src.read(1).astype(np.float32)
        rmse = src.read(2).astype(np.float32)
        n_pairs = src.read(3).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if nodata is not None and np.isfinite(nodata):
        bias[bias == nodata] = np.nan
        rmse[rmse == nodata] = np.nan
        n_pairs[n_pairs == nodata] = np.nan

    # Keep statistics only where at least one paired observation exists.
    invalid = ~np.isfinite(n_pairs) | (n_pairs <= 0)
    bias[invalid] = np.nan
    rmse[invalid] = np.nan

    cropped_arrays, transform = crop_nan_borders([bias, rmse], transform)
    bias, rmse = cropped_arrays

    return bias, rmse, transform, crs


def crop_nan_borders(arrays, transform):
    if not arrays:
        return arrays, transform

    valid_mask = np.zeros_like(arrays[0], dtype=bool)
    for arr in arrays:
        valid_mask |= np.isfinite(arr)

    if not np.any(valid_mask):
        return arrays, transform

    rows = np.where(np.any(valid_mask, axis=1))[0]
    cols = np.where(np.any(valid_mask, axis=0))[0]

    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1

    cropped = [arr[r0:r1, c0:c1] for arr in arrays]
    window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
    cropped_transform = rio.windows.transform(window, transform)

    return cropped, cropped_transform


def robust_symmetric_limit(arr: np.ndarray, q: float = 99.0, fallback: float = 0.01) -> float:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return fallback
    vmax = float(np.nanpercentile(np.abs(vals), q))
    if vmax <= 0:
        return fallback
    return vmax


def robust_upper_limit(arr: np.ndarray, q: float = 99.0, fallback: float = 0.1) -> float:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return fallback
    vmax = float(np.nanpercentile(vals, q))
    if vmax <= 0:
        return fallback
    return vmax


def plot_metric_map(ax, data, transform, crs, cmap, vmin, vmax, cbar_label: str):
    gray_basemap = getattr(ctx.providers, "CartoDB").get("PositronNoLabels")
    show(data, transform=transform, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    ctx.add_basemap(ax, crs=crs, source=gray_basemap, attribution=False)
    show(data, transform=transform, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.88)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.045, pad=0.02)
    cbar.set_label(cbar_label)

    scalebar = ScaleBar(
        dx=1,
        units="m",
        fixed_value=300,
        fixed_units="km",
        location="lower right",
        frameon=False,
        color="black",
    )
    ax.add_artist(scalebar)

    ax.axis("off")


def plot_hist(ax, hdf5_glob: str, model_label: str):
    df = vx.open(hdf5_glob)
    n_bins = 120
    x_min = -0.1
    x_max = 1.2
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Use Vaex aggregations without Vaex Matplotlib helpers to avoid layout-engine conflicts.
    counts_model = np.asarray(
        df.count(binby="model_albedo", limits=[x_min, x_max], shape=n_bins)
    )
    counts_hsa = np.asarray(
        df.count(binby="hsa500m_albedo", limits=[x_min, x_max], shape=n_bins)
    )

    ax.plot(centers, counts_model, lw=1.5, label=model_label)
    ax.plot(centers, counts_hsa, lw=1.5, label="HSA500m")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"{model_label} and HSA500m albedo")
    ax.set_ylabel("Count (log-transformed)")


# -----------------------------------------------------------------------------
# Build figure (2 rows x 3 columns)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(
    nrows=2,
    ncols=4,
    width_ratios=[1.0, 1.0, 0.45, 1.0],  # col 3 is a fixed spacer between map and histogram columns
    wspace=0.06,
    hspace=0.16,
)

axes = np.empty((2, 3), dtype=object)
axes[0, 0] = fig.add_subplot(gs[0, 0])
axes[0, 1] = fig.add_subplot(gs[0, 1])
axes[0, 2] = fig.add_subplot(gs[0, 3])
axes[1, 0] = fig.add_subplot(gs[1, 0])
axes[1, 1] = fig.add_subplot(gs[1, 1])
axes[1, 2] = fig.add_subplot(gs[1, 3])

# Row 1: HCLIM
bias_hclim, rmse_hclim, tf_hclim, crs_hclim = read_bias_rmse(HCLIM_MAP_PATH)
bias_lim_hclim = robust_symmetric_limit(bias_hclim)
rmse_lim_hclim = robust_upper_limit(rmse_hclim)

plot_metric_map(
    axes[0, 0],
    bias_hclim,
    tf_hclim,
    crs_hclim,
    cmap=getattr(cmo.cm, "balance_r"),
    vmin=-bias_lim_hclim,
    vmax=bias_lim_hclim,
    cbar_label="Bias (HCLIM - HSA500m)",
)
# axes[0, 0].set_title("(a) HCLIM Bias")
axes[0, 0].text(0.02, 0.08, "a)", transform=axes[0, 0].transAxes, verticalalignment="top", horizontalalignment  ="left")

plot_metric_map(
    axes[0, 1],
    rmse_hclim,
    tf_hclim,
    crs_hclim,
    cmap=getattr(cmo.cm, "amp"),
    vmin=0.0,
    vmax=rmse_lim_hclim,
    cbar_label="RMSE (HCLIM - HSA500m)",
)
# axes[0, 1].set_title("(b) HCLIM RMSE")
axes[0, 1].text(0.02, 0.08, "b)", transform=axes[0, 1].transAxes, verticalalignment="top", horizontalalignment  ="left")
plt.sca(axes[0, 2])
plot_hist(axes[0, 2], HCLIM_HDF5_GLOB, "HCLIM")
axes[0, 2].set_xlim(0.0, 1.0)
# axes[0, 2].set_title("(c) HCLIM Histogram")
axes[0, 2].set_xlabel("Albedo")
axes[0, 2].text(0.02, 0.08, "c)", transform=axes[0, 2].transAxes, verticalalignment="top", horizontalalignment  ="left")

# Row 2: HIRHAM5
bias_hirham5, rmse_hirham5, tf_hirham5, crs_hirham5 = read_bias_rmse(HIRHAM5_MAP_PATH)
bias_lim_hirham5 = robust_symmetric_limit(bias_hirham5)
rmse_lim_hirham5 = robust_upper_limit(rmse_hirham5)

plot_metric_map(
    axes[1, 0],
    bias_hirham5,
    tf_hirham5,
    crs_hirham5,
    cmap=getattr(cmo.cm, "balance_r"),
    vmin=-bias_lim_hclim, #-bias_lim_hirham5,
    vmax=bias_lim_hclim, # bias_lim_hirham5,
    cbar_label="Bias (HIRHAM5 - HSA500m)",
)
# axes[1, 0].set_title("(d) HIRHAM5 Bias")
axes[1, 0].text(0.02, 0.08, "d)", transform=axes[1, 0].transAxes, verticalalignment="top", horizontalalignment  ="left")

plot_metric_map(
    axes[1, 1],
    rmse_hirham5,
    tf_hirham5,
    crs_hirham5,
    cmap=getattr(cmo.cm, "amp"),
    vmin=0.0,
    vmax=rmse_lim_hclim, # rmse_lim_hirham5,
    cbar_label="RMSE (HIRHAM5 - HSA500m)",
)
# axes[1, 1].set_title("(e) HIRHAM5 RMSE")
axes[1, 1].text(0.02, 0.08, "e)", transform=axes[1, 1].transAxes, verticalalignment="top", horizontalalignment  ="left")

plt.sca(axes[1, 2])
plot_hist(axes[1, 2], HIRHAM5_HDF5_GLOB, "HIRHAM5")
axes[1, 2].set_xlim(0.0, 1.0)
# axes[1, 2].set_title("(f) HIRHAM5 Histogram")
axes[1, 2].set_xlabel("Albedo")
axes[1, 2].text(0.02, 0.08, "f)", transform=axes[1, 2].transAxes, verticalalignment="top", horizontalalignment  ="left")


plt.show()

fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/model_hsa500m_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/model_hsa500m_comparison.pdf", dpi=300, bbox_inches="tight")
# %%
