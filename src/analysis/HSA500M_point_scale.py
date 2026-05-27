'''

'''

#%%
import pandas as pd
from datetime import timedelta
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
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

stations = ['KAN_U', 'KAN_M', 'KAN_L']
date_start = '2021-04-01'
date_end = '2021-09-01'
subplot_labels = ['a', 'b', 'c']

def load_and_filter(path, station, date_start=date_start, date_end=date_end):
    df = pd.read_csv(path)
    df = df[df['aws'] == station].copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= date_start) & (df['time'] < date_end)]
    # mask out albedo data as NaN when cloud cover is >0.3 if 'cc' column exists
    # if 'cc' in df.columns:
    #     df.loc[df['cc'] >0.3, 'albedo'] = np.nan
    # insert row with NaN albedo for missing dates to ensure continuous time series
    all_dates = pd.date_range(start=date_start, end=date_end, freq='D')
    df = df.set_index('time').reindex(all_dates).reset_index().rename(columns={'index': 'time'})

    return df

#%% plot 3-panel time series
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True, sharey=True)

print(f"{'Station':<8} {'N':>4}  {'Bias':>7}  {'RMSE':>7}  {'R':>7}  {'p':>10}")
print("-" * 50)

for i, station in enumerate(stations):
    ax = axes[i]

    dfaws     = load_and_filter(aws_path,     station)
    dfmod10   = load_and_filter(mod10_path,   station)
    dfmyd10   = load_and_filter(myd10_path,   station)
    dfmcd43a3 = load_and_filter(mcd43a3_path, station)
    dfvj143ma3= load_and_filter(vj143ma3_path,station)
    dfvnp43ma3= load_and_filter(vnp43ma3_path,station)
    dfgcomc   = load_and_filter(gcomc_path,   station)
    dfsice    = load_and_filter(sice_path,    station)
    dfcarra   = load_and_filter(carra_path,   station)
    dfhsa500m = load_and_filter(hsa500m_path, station)

    ax.plot(dfaws['time'], dfaws['albedo'], label='AWS', marker='.', linestyle='-', color='black', alpha=0.7)
    sns.scatterplot(data=dfmod10,    x='time', y='albedo_mod10',                    label='MOD10A1',  color='#083962', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfmyd10,    x='time', y='albedo_myd10',                    label='MYD10A1',  color='#2062ac', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfmcd43a3,  x='time', y='albedo_mcd43a3_bluesky',          label='MCD43A3',  color='#94ace6', marker='d', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfvj143ma3, x='time', y='albedo_viirs_vj143ma3_bluesky',   label='VJ143MA3', color='#5a3918', marker='d', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfvnp43ma3, x='time', y='albedo_viirs_vnp43ma3_bluesky',   label='VNP43MA3', color='#d5ac4a', marker='d', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfgcomc,    x='time', y='albedo_gcomc_sr',                 label='GCOM-C',   color='#942010', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfsice,     x='time', y='albedo_sice_rebuild',             label='SICE',     color='#ff836a', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfcarra,    x='time', y='albedo_carra',                    label='CARRA',    color='#c0c0c0', marker='s', alpha=0.7, ax=ax)
    sns.scatterplot(data=dfhsa500m,  x='time', y='albedo_hsa500m_gapfilled',        label='HSA500m',  color='#c52018', marker='X', alpha=1, ax=ax)

    # ax.text(0.01, 0.97, f"{subplot_labels[i]}) {station}", transform=ax.transAxes,
    #         ha='left', va='top', fontsize=14, fontweight='bold')
    ax.set_ylabel('Albedo')
    ax.set_xlabel('')
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(pd.to_datetime(date_start)+timedelta(days=-1), pd.to_datetime(date_end)+timedelta(days=1))
    legend = ax.get_legend()
    if legend:
        legend.remove()

    # Statistics: HSA500m vs PROMICE AWS
    merged = pd.merge(
        dfaws[['time', 'albedo']],
        dfhsa500m[['time', 'albedo_hsa500m_gapfilled']],
        on='time', how='inner'
    ).dropna()
    if len(merged) > 1:
        bias = (merged['albedo_hsa500m_gapfilled'] - merged['albedo']).mean()
        rmse = np.sqrt(((merged['albedo_hsa500m_gapfilled'] - merged['albedo']) ** 2).mean())
        r, p = stats.pearsonr(merged['albedo'], merged['albedo_hsa500m_gapfilled'])
        print(f"{station:<8} {len(merged):>4}  {bias:>7.4f}  {rmse:>7.4f}  {r:>7.4f}  {p:>10.4e}")
        stats_text = (
            f"{subplot_labels[i]}) {station}:\n"
            f"N = {len(merged)}\n"
            f"Bias = {bias:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"R(p<0.05) = {r:.3f}"
        )
    else:
        print(f"{station:<8} -- no matching data --")
        stats_text = 'No matching\ndata'

    ax.text(
        0.01,
        0.05,
        stats_text,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        # fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none')
    )

    # # add reference line to mark the rainfall event on 2021-08-14
    # ax.vlines(pd.to_datetime('2021-08-14'), 0.2, 1.0, colors='k', linestyles='--')
    # # add annotation
    # ax.text(pd.to_datetime('2021-08-14') + timedelta(days=1), 0.3, 'Rainfall at the Summit', color='k', fontsize=10, ha='left', va='center')

# axes[-1].set_xlabel('Time')

# Single legend on top of the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.96))

fig.savefig('HSA500m_point_scale_time_series.png', dpi=300, bbox_inches='tight')
fig.savefig('HSA500m_point_scale_time_series.pdf', dpi=300, bbox_inches='tight')

# %%
