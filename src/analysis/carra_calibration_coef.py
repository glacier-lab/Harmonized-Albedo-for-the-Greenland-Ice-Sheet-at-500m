#%%
# Source - https://stackoverflow.com/a/76753429
# Posted by Hasan Ramezani
# Retrieved 2026-03-29, License - CC BY-SA 4.0

import vaex as vx
import numpy as np
import cmocean as cm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np

#%% training data

df = vx.open("/data_3/shunan_2/AU/hsa500m/carra_hsa_comparison_hdf5/training/*.h5")

#%%
df.viz.heatmap(x="carra", y="hsa500m", what=np.log(vx.stat.count()))
# %%
dftesting = vx.open("/data_3/shunan_2/AU/hsa500m/carra_hsa_comparison_hdf5/testing/*.h5")
# %%
