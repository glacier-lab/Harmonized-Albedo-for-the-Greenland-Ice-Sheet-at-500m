#%%
# import os

# Give Vaex/NumPy backends permission to use all available cores unless the
# user already pinned these variables in the shell.
# _CPU_COUNT = str(os.cpu_count() or 1)
# os.environ.setdefault("VAEX_NUM_THREADS", _CPU_COUNT)
# os.environ.setdefault("OMP_NUM_THREADS", _CPU_COUNT)
# os.environ.setdefault("MKL_NUM_THREADS", _CPU_COUNT)
# os.environ.setdefault("OPENBLAS_NUM_THREADS", _CPU_COUNT)

import vaex as vx
import numpy as np
import cmocean as cm
import matplotlib.pyplot as plt
# import pandas as pd
from scipy import stats
import seaborn as sns
from typing import Any, NamedTuple, Optional

sns.set_theme(font_scale=1.5, style="darkgrid")

CALIBRATION_CAP = 0.83


class RegressionResult(NamedTuple):
    slope: float
    intercept: float
    rvalue: float
    pvalue: float


def linregress_vaex(df: Any, x: str, y: str, selection: Optional[str] = None) -> RegressionResult:
    """Compute linear-regression summary from Vaex aggregations (out-of-core)."""
    base_selection = f"isfinite({x}) & isfinite({y})"
    if selection:
        selection = f"({base_selection}) & ({selection})"
    else:
        selection = base_selection

    n = float(df.count(selection=selection))
    if n < 3:
        raise ValueError("Need at least 3 finite points to compute regression statistics")

    mean_x = float(df.mean(x, selection=selection))
    mean_y = float(df.mean(y, selection=selection))
    var_x = float(df.var(x, selection=selection))
    var_y = float(df.var(y, selection=selection))
    mean_xy = float(df.mean(df[x] * df[y], selection=selection))
    cov_xy = mean_xy - mean_x * mean_y

    if var_x <= 0 or var_y <= 0:
        raise ValueError("Variance is zero; regression is undefined")

    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    rvalue = cov_xy / np.sqrt(var_x * var_y)

    # Numerical guard to keep r within [-1, 1] for p-value computation.
    rvalue = float(np.clip(rvalue, -1.0, 1.0))
    if abs(rvalue) == 1.0:
        pvalue = 0.0
    else:
        t_stat = rvalue * np.sqrt((n - 2.0) / (1.0 - rvalue**2))
        pvalue = float(2.0 * stats.t.sf(abs(t_stat), df=n - 2.0))

    return RegressionResult(
        slope=float(slope),
        intercept=float(intercept),
        rvalue=float(rvalue),
        pvalue=float(pvalue),
    )

#%% training data

df = vx.open("/data_3/shunan_2/AU/hsa500m/carra_hsa_comparison_hdf5/training/*.h5")

#%% evaluate the data distribution
# fig, ax = plt.subplots(figsize=(8,3))
# df.viz.histogram(df.carra, show=False, color='blue', label='CARRA')
# df.viz.histogram(df.hsa500m, show=False, color='orange', label='HSA500m')
# ax.legend()
# ax.axvline(CALIBRATION_CAP, color='gray', alpha=0.8)
# ax.set_xlim(0, 1)
# ax.set(xlabel="Albedo", title="Training Data Distribution (Full Range)")
# fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_comparison_hdf5/figures/carra_hsa_distribution_training.png", dpi=300)
# fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_comparison_hdf5/figures/carra_hsa_distribution_training.pdf", dpi=300)
# plt.close(fig)

#%% statistics between hsa500m and carra: slope, intercept, r2, and p-value of linear regression
fit_selection = f"carra < {CALIBRATION_CAP}"
results = linregress_vaex(df, "carra", "hsa500m", selection=fit_selection)

print(f"slope: {results.slope}")
print(f"intercept: {results.intercept}")
print(f"r2: {results.rvalue**2}")
print(f"p-value: {results.pvalue}")
print(f"Number of points used in regression: {df.count(selection=fit_selection):,}")
print(f"Number of points in full dataset: {df.count():,}")
print(results)

#%%
fig, ax = plt.subplots(figsize=(8,7))
ax.plot(np.array([0,1]), results.slope * np.array([0,1]) + results.intercept, color='red') # ols regression etm+ vs oli
ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, label='1:1 line')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
df.viz.heatmap(
    x="carra", 
    y="hsa500m", 
    what=np.log(vx.stat.count()),
    show=False,
    vmin=0, vmax=20,
    colormap = cm.cm.haline
    )
ax.axvline(CALIBRATION_CAP, color='gray', alpha=0.8)
# add text annotation for the calibration cap
ax.text(CALIBRATION_CAP + 0.02, 0.0, f"Calibration Cap: {CALIBRATION_CAP}", color='gray', alpha=0.8, rotation=90, va='bottom')
ax.set_aspect('equal')
ax.set(xlabel="Downscaled CARRA", ylabel="Harmonized Satellite Albedo 500m")
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_calibration_training.png", dpi=300)
# close figure after saved
# plt.close(fig)
# %% testing data
df = vx.open("/data_3/shunan_2/AU/hsa500m/carra_hsa_comparison_hdf5/testing/*.h5")
df["carra_calibrated"] = results.slope * df.carra + results.intercept

validation_selection = f"carra < {CALIBRATION_CAP}"
results = linregress_vaex(df, "carra_calibrated", "hsa500m", selection=validation_selection)
print(f"slope: {results.slope}")
print(f"intercept: {results.intercept}")
print(f"r2: {results.rvalue**2}")
print(f"p-value: {results.pvalue}")
print(f"Number of points used in regression: {df.count(selection=validation_selection):,}")
print(f"Number of points in full dataset: {df.count():,}")
print(results)


# %%
fig, ax = plt.subplots(figsize=(8,7))
ax.plot(np.array([0,1]), results.slope * np.array([0,1]) + results.intercept, color='red') # ols regression etm+ vs oli
ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, label='1:1 line')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
df.viz.heatmap(
    x="carra_calibrated", 
    y="hsa500m", 
    what=np.log(vx.stat.count()),
    show=False,
    vmin=0, vmax=20,
    colormap = cm.cm.haline
    )
ax.set_aspect('equal')
ax.set(xlabel="Downscaled CARRA", ylabel="Harmonized Satellite Albedo 500m")
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_calibration_testing.png", dpi=300)
# close figure after saved
# plt.close(fig)
# %%
