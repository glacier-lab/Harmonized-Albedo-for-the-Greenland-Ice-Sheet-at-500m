#%%
import vaex as vx
import numpy as np
import cmocean as cm
import matplotlib.pyplot as plt
# import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np
sns.set_theme(font_scale=1.5, style="darkgrid")

#%% training data

df = vx.open("/data_3/shunan_2/AU/hsa500m/carra_hsa_comparison_hdf5/training/*.h5")

#%% statistics between hsa500m and carra: slope, intercept, r2, and p-value of linear regression
results = stats.linregress(df.carra.values, df.hsa500m.values)

print(f"slope: {results.slope}")
print(f"intercept: {results.intercept}")
print(f"r2: {results.rvalue**2}")
print(f"p-value: {results.pvalue}")
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
    show=True,
    colormap = cm.cm.haline)
ax.set_aspect('equal')
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_comparison_hdf5/figures/carra_hsa_calibration_training.png", dpi=300)
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_comparison_hdf5/figures/carra_hsa_calibration_training.pdf", dpi=300)
# close figure after saved
plt.close(fig)
# %% testing data
df = vx.open("/data_3/shunan_2/AU/hsa500m/carra_hsa_comparison_hdf5/testing/*.h5")
df["carra_calibrated"] = results.slope * df.carra + results.intercept

results = stats.linregress(df.carra_calibrated.values, df.hsa500m.values)
print(f"slope: {results.slope}")
print(f"intercept: {results.intercept}")
print(f"r2: {results.rvalue**2}")
print(f"p-value: {results.pvalue}")
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
    show=True,
    colormap = cm.cm.haline)
ax.set_aspect('equal')
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_comparison_hdf5/figures/carra_hsa_calibration_testing.png", dpi=300)
fig.savefig("/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/print/carra_hsa_comparison_hdf5/figures/carra_hsa_calibration_testing.pdf", dpi=300)
# close figure after saved
plt.close(fig)
# %%
