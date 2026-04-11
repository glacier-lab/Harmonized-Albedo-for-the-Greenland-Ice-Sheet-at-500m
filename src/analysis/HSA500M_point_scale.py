'''

'''

#%%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# import cmocean
sns.set_theme(style='darkgrid', font_scale=1.5)
#%%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/promice_day.csv'
mod10_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_mod10a1.csv'
myd10_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_myd10a1.csv'
mcd43a3_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_mcd43a3_bluesky.csv'
vj143ma3_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_viirs_vj143ma3_bluesky.csv'
vnp43ma3_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_viirs_vnp43ma3_bluesky.csv'
gcomc_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_gcomc_sr_albedo.csv'
sice_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_sice_rebuild.csv'
carra_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_carra.csv'
hsa500m_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_hsa500m_gapfilled.csv'

dfaws = pd.read_csv(aws_path)
# keep KAN_M only
dfaws = dfaws[dfaws['aws'] == 'KAN_M']
# filter to 2021
dfaws['time'] = pd.to_datetime(dfaws['time'])
dfaws = dfaws[(dfaws['time'] >= '2021-01-01') & (dfaws['time'] <= '2021-12-31')]

dfmod10 = pd.read_csv(mod10_path)
dfmod10 = dfmod10[dfmod10['aws'] == 'KAN_M']
dfmod10['time'] = pd.to_datetime(dfmod10['time'])
dfmod10 = dfmod10[(dfmod10['time'] >= '2021-01-01') & (dfmod10['time'] <= '2021-12-31')]


dfmyd10 = pd.read_csv(myd10_path)
dfmyd10 = dfmyd10[dfmyd10['aws'] == 'KAN_M']
dfmyd10['time'] = pd.to_datetime(dfmyd10['time'])
dfmyd10 = dfmyd10[(dfmyd10['time'] >= '2021-01-01') & (dfmyd10['time'] <= '2021-12-31')]

dfmcd43a3 = pd.read_csv(mcd43a3_path)
dfmcd43a3 = dfmcd43a3[dfmcd43a3['aws'] == 'KAN_M']
dfmcd43a3['time'] = pd.to_datetime(dfmcd43a3['time'])
dfmcd43a3 = dfmcd43a3[(dfmcd43a3['time'] >= '2021-01-01') & (dfmcd43a3['time'] <= '2021-12-31')]
    
dfvj143ma3 = pd.read_csv(vj143ma3_path)
dfvj143ma3 = dfvj143ma3[dfvj143ma3['aws'] == 'KAN_M']
dfvj143ma3['time'] = pd.to_datetime(dfvj143ma3['time'])
dfvj143ma3 = dfvj143ma3[(dfvj143ma3['time'] >= '2021-01-01') & (dfvj143ma3['time'] <= '2021-12-31')]

dfvnp43ma3 = pd.read_csv(vnp43ma3_path)
dfvnp43ma3 = dfvnp43ma3[dfvnp43ma3['aws'] == 'KAN_M']
dfvnp43ma3['time'] = pd.to_datetime(dfvnp43ma3['time'])
dfvnp43ma3 = dfvnp43ma3[(dfvnp43ma3['time'] >= '2021-01-01') & (dfvnp43ma3['time'] <= '2021-12-31')]

dfgcomc = pd.read_csv(gcomc_path)
dfgcomc = dfgcomc[dfgcomc['aws'] == 'KAN_M']
dfgcomc['time'] = pd.to_datetime(dfgcomc['time'])
dfgcomc = dfgcomc[(dfgcomc['time'] >= '2021-01-01') & (dfgcomc['time'] <= '2021-12-31')]

dfsice = pd.read_csv(sice_path)
dfsice = dfsice[dfsice['aws'] == 'KAN_M']
dfsice['time'] = pd.to_datetime(dfsice['time'])
dfsice = dfsice[(dfsice['time'] >= '2021-01-01') & (dfsice['time'] <= '2021-12-31')]

dfhsa500m = pd.read_csv(hsa500m_path)
dfhsa500m = dfhsa500m[dfhsa500m['aws'] == 'KAN_M']
dfhsa500m['time'] = pd.to_datetime(dfhsa500m['time'])
dfhsa500m = dfhsa500m[(dfhsa500m['time'] >= '2021-01-01') & (dfhsa500m['time'] <= '2021-12-31')]

dfcarra = pd.read_csv(carra_path)
dfcarra = dfcarra[dfcarra['aws'] == 'KAN_M']
dfcarra['time'] = pd.to_datetime(dfcarra['time'])
dfcarra = dfcarra[(dfcarra['time'] >= '2021-01-01') & (dfcarra['time'] <= '2021-12-31')]

#%% plot time series of all datasets
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dfaws['time'], dfaws['albedo'], label='AWS KAN_M', marker='o', linestyle='-', color='black')
sns.scatterplot(data=dfmod10, x='time', y='albedo_mod10', label='MOD10A1',  color='blue')
sns.scatterplot(data=dfmyd10, x='time', y='albedo_myd10', label='MYD10A1', color='cyan')
sns.scatterplot(data=dfmcd43a3, x='time', y='albedo_mcd43a3_bluesky', label='MCD43A3', color='green')
sns.scatterplot(data=dfvj143ma3, x='time', y='albedo_viirs_vj143ma3_bluesky', label='VIIRS VJ143MA3', color='orange')
sns.scatterplot(data=dfvnp43ma3, x='time', y='albedo_viirs_vnp43ma3_bluesky', label='VIIRS VNP43MA3',  color='red')
sns.scatterplot(data=dfgcomc, x='time', y='albedo_gcomc_sr', label='GCOMC', color='purple')
sns.scatterplot(data=dfsice, x='time', y='albedo_sice_rebuild', label='SICE', color='brown')
sns.scatterplot(data=dfhsa500m, x='time', y='albedo_hsa500m_gapfilled', label='HSA500M', color='gray')
sns.scatterplot(data=dfcarra, x='time', y='albedo_carra', label='CARRA', color='lightgray')

ax.set_xlabel('Time')
ax.set_ylabel('Albedo')
ax.set_xlim([pd.to_datetime('2021-04-01'), pd.to_datetime('2021-09-01')])
# %%
