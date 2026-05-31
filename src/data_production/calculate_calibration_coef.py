'''
This script calculates calibration coefficients to harmonize remote sensing albedo data with AWS albedo measurements. The workflow includes:
1. Loading and preprocessing albedo data from various sources.
2. Generating scenarios based on different combinations of remote sensing sensors, including handling MODIS orbital drift by creating year-specific scenarios.
3. For each scenario, calculating average daily remote sensing albedo, merging with AWS data, and splitting into training and testing sets.
4. Deriving calibration coefficients from training data, applying calibration to test data, and calculating evaluation metrics.
5. Visualizing the results with hexbin plots showing the relationship between remote sensing and AWS albedo before and after calibration, including statistics in text boxes.
6. Storing results in a DataFrame and saving to CSV, as well as saving figures for each scenario.
The script is designed to be flexible, allowing for easy addition of new sensors or scenarios by modifying the `sensors_expanded` dictionary and ensuring that the data is properly preprocessed. The `is_valid_combination` function ensures that only valid combinations of sensors are processed, avoiding mixing of incompatible sensor variants.

Shunan Feng (shunan.feng@envs.au.dk)
'''
#%% 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import cmocean 
import os
from itertools import combinations
from sklearn.model_selection import train_test_split

sns.set_theme(style="darkgrid", font_scale=1.5, font="Arial")
plt.ioff()
# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/promice_day.csv'
mod10_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_mod10a1.csv'
myd10_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_myd10a1.csv'
mcd43a3_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_mcd43a3_bluesky.csv'
vj143ma3_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_viirs_vj143ma3_bluesky.csv'
vnp43ma3_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_viirs_vnp43ma3_bluesky.csv'
gcomc_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_gcomc_sr_albedo.csv'
sice_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_sice_rebuild.csv'
viirs_sr_vnp09ga_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_viirs_sr_albedo_vnp09ga.csv'
viirs_sr_vj109ga_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_viirs_sr_albedo_vj109ga.csv'
viirs_sr_vj209ga_path = '/data_3/shunan_2/AU/hsa500m/point2pix/point2pix_viirs_sr_albedo_vj209ga.csv'
PLOT_FIGURES = False # Set to False to skip plotting and only calculate metrics

# Folder to save calibration figures
calibration_output_folder = '/data_3/shunan_2/AU/hsa500m/calibration'
figure_output_folder = '/data_3/shunan_2/AU/hsa500m/calibration/print'
os.makedirs(calibration_output_folder, exist_ok=True)
os.makedirs(figure_output_folder, exist_ok=True)

def load_and_preprocess_data(albedo_path):
    """Load and preprocess albedo data."""
    # Load data
    df = pd.read_csv(albedo_path)

    # exclude local glaciers and aws with mean albedo > 0.8
    aws_exclude = ['FRE', 'LYN_L', 'LYN_T', 'MIT', 'NUK_K', 'ZAC_A', 'ZAC_L', 'ZAC_U',
                   'CEN', 'CP1', 'DY2', 'EGP', 'HUM', 'KAN_U', 'NAE', 'NAU', 'NEM', 
                   'NSE', 'SDL', 'SDM', 'TUN']
    df = df[~df['aws'].isin(aws_exclude)]
    
    df['time'] = pd.to_datetime(df['time'])

    # determine if the data is from AWS or remote sensing based on the filename
    if 'promice' in albedo_path.lower():
        df = df.rename(columns={'albedo': 'albedo_aws'})
        df.loc[(df['albedo_aws'] <= 0) | (df['albedo_aws'] >= 1), 'albedo_aws'] = np.nan
    else:
        rs_albedo_cols = [col for col in df.columns if col.startswith('albedo_') and col != 'albedo']
        if not rs_albedo_cols:
            raise ValueError(f"No remote sensing albedo column found in {albedo_path} (expected column starting with 'albedo_')")
        rs_albedo_col = rs_albedo_cols[0]
        df = df.rename(columns={rs_albedo_col: 'albedo_rs'})
        df.loc[(df['albedo_rs'] <= 0) | (df['albedo_rs'] >= 1), 'albedo_rs'] = np.nan
    
    df = df.dropna()

    return df
    
def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    y_true = pd.to_numeric(pd.Series(y_true), errors='coerce').to_numpy(dtype=float)
    y_pred = pd.to_numeric(pd.Series(y_pred), errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) < 2:
        raise ValueError("Not enough valid points to compute metrics")

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'std_err': std_err,
        'n': len(y_true)
    }

def create_calibration_plot(scenario_id, scenario_name, df_train, df_test, slope, intercept, train_metrics, test_metrics):
    """Create figure with subplots showing training data and calibrated test data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Training data with calibration line
    ax = axes[0]
    hexbin = ax.hexbin(df_train['albedo_rs'], df_train['albedo_aws'], 
                       gridsize=50, cmap=cmocean.cm.haline, bins='log', mincnt=1)
    
    # Add regression line from training
    sns.regplot(x='albedo_rs', y='albedo_aws', data=df_train, scatter=False, ax=ax, color='red')
    # x_line = np.array([0, 1])
    # y_line = slope * x_line + intercept
    # ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression (R²={train_metrics["r_squared"]:.3f})')
    
    # Add 1:1 line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='1:1 line')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('RS Albedo')
    ax.set_ylabel('AWS Albedo')
    ax.set_title('(a) Before Calibration (Training Data)', weight='normal')
    # ax.legend(fontsize=9, loc='upper left')
    cb = fig.colorbar(hexbin, ax=ax)
    cb.set_label('log(Count+1)')
    
    # Add statistics text box for training data
    stats_text = f'N = {train_metrics["n"]}\n'
    stats_text += f'R² = {train_metrics["r_squared"]:.3f}\n'
    stats_text += f'RMSE = {train_metrics["rmse"]:.3f}\n'
    stats_text += f'MAE = {train_metrics["mae"]:.3f}\n'
    stats_text += f'Bias = {train_metrics["bias"]:.3f}\n'
    stats_text += f'Slope = {train_metrics["slope"]:.3f}\n'
    stats_text += f'Intercept = {train_metrics["intercept"]:.3f}'
    
    ax.text(0.05, 0.50, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel B: Test data with calibrated RS albedo
    ax = axes[1]
    
    # Apply calibration coefficients to test data
    # Calibrated albedo = slope * albedo_rs + intercept
    df_test_calib = df_test.copy()
    df_test_calib['albedo_rs_calibrated'] = slope * df_test['albedo_rs'] + intercept
    
    # Clip to [0, 1] range
    df_test_calib['albedo_rs_calibrated'] = df_test_calib['albedo_rs_calibrated'].clip(0, 1)
    
    hexbin = ax.hexbin(df_test_calib['albedo_rs_calibrated'], df_test['albedo_aws'], 
                       gridsize=50, cmap=cmocean.cm.haline, bins='log', mincnt=1)
    sns.regplot(x='albedo_rs_calibrated', y='albedo_aws', data=df_test_calib, scatter=False, ax=ax, color='red')
    # Add 1:1 line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='1:1 line')
    
    # Calculate test metrics
    test_calib_metrics = calculate_metrics(df_test['albedo_aws'], df_test_calib['albedo_rs_calibrated'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('Harmonized Albedo')
    ax.set_ylabel('AWS Albedo')
    ax.set_title(f'(b) After Calibration (Test Data)', weight='normal')
    # ax.legend(fontsize=9, loc='upper left')
    cb = fig.colorbar(hexbin, ax=ax)
    cb.set_label('log(Count+1)')
    
    # Add statistics text box for test data
    stats_text = f'N = {test_calib_metrics["n"]}\n'
    stats_text += f'R² = {test_calib_metrics["r_squared"]:.3f}\n'
    stats_text += f'RMSE = {test_calib_metrics["rmse"]:.3f}\n'
    stats_text += f'MAE = {test_calib_metrics["mae"]:.3f}\n'
    stats_text += f'Bias = {test_calib_metrics["bias"]:.3f}\n'
    stats_text += f'Slope = {test_calib_metrics["slope"]:.3f}\n'
    stats_text += f'Intercept = {test_calib_metrics["intercept"]:.3f}'
    
    ax.text(0.05, 0.50, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f'Scenario {scenario_id}: {scenario_name}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    # Save figure
    fig_path = os.path.join(figure_output_folder, f'scenario_{scenario_id:03d}_calibration.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved to: {fig_path}")
    fig_path = os.path.join(figure_output_folder, f'scenario_{scenario_id:03d}_calibration.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return test_calib_metrics

def build_combined_rs_dataset(sensor_sources):
    """Build one RS albedo series by averaging all sensors in a scenario."""
    frames = []
    for sensor_name, df_sensor in sensor_sources.items():
        if len(df_sensor) == 0:
            continue
        tmp = df_sensor[['time', 'aws', 'albedo_rs']].copy()
        tmp['sensor'] = sensor_name
        frames.append(tmp)

    if len(frames) == 0:
        return pd.DataFrame(columns=['time', 'aws', 'albedo_rs'])

    df_all = pd.concat(frames, ignore_index=True)
    df_avg = (
        df_all
        .groupby(['time', 'aws'], as_index=False)['albedo_rs']
        .mean()
    )
    df_avg['albedo_rs'] = pd.to_numeric(df_avg['albedo_rs'], errors='coerce')
    return df_avg


def split_sensor_by_orbital_drift(df_sensor, base_name, drift_start_year):
    """Split one sensor into pre-drift and per-year post-drift datasets."""
    out = {}

    df_pre = df_sensor[df_sensor['time'].dt.year < drift_start_year].copy()
    if len(df_pre) > 0:
        out[base_name] = df_pre

    drift_years = sorted([y for y in df_sensor['time'].dt.year.unique() if y >= drift_start_year])
    for year in drift_years:
        df_year = df_sensor[df_sensor['time'].dt.year == year].copy()
        if len(df_year) > 0:
            out[f'{base_name}_{year}'] = df_year

    return out

#%% calibration coefficient calculation
'''
The workflow for calculating calibration coefficients involves the following steps:
1. Load and preprocess albedo data from various remote sensing sources and AWS data.
2. Iterate through the combination of each sensor and mark the scenario using unique identifiers.
3. For each combination, calculate the average daily remote sensing albedo value.
4. Merge the remote sensing albedo data with AWS albedo data based on AWS station and time.
5. Split data into training and testing sets.
6. Derive calibration coefficients from training data.
7. Apply calibration to test data and calculate metrics.
8. Visualize training and calibrated test data.
'''
df_aws = load_and_preprocess_data(aws_path)
df_mod10 = load_and_preprocess_data(mod10_path)
df_myd10 = load_and_preprocess_data(myd10_path)
df_mcd43a3 = load_and_preprocess_data(mcd43a3_path)
df_vj143ma3 = load_and_preprocess_data(vj143ma3_path)
df_vnp43ma3 = load_and_preprocess_data(vnp43ma3_path)
df_sice = load_and_preprocess_data(sice_path)
df_gcomc = load_and_preprocess_data(gcomc_path)
df_viirs_sr_vnp09ga = load_and_preprocess_data(viirs_sr_vnp09ga_path)
df_viirs_sr_vj109ga = load_and_preprocess_data(viirs_sr_vj109ga_path)
df_viirs_sr_vj209ga = load_and_preprocess_data(viirs_sr_vj209ga_path)

# Restore orbital-drift handling for MODIS-family products.
# MOD10A1 drift starts 2020, MYD10A1 drift starts 2021.
mod10_split = split_sensor_by_orbital_drift(df_mod10, 'mod10', drift_start_year=2020)
myd10_split = split_sensor_by_orbital_drift(df_myd10, 'myd10', drift_start_year=2021)

# MCD43A3 (Terra+Aqua blend) follows MOD10 drift timeline in this workflow.
mcd43a3_split = split_sensor_by_orbital_drift(df_mcd43a3, 'mcd43a3_bluesky', drift_start_year=2020)

# Group sensors for scenario generation. Daily and 16-day products are calibrated
# separately; they are not mixed within the same scenario.
daily_sensor_groups = {
    'mod10': mod10_split,
    'myd10': myd10_split,
    'sice': {'sice': df_sice},
    'gcomc': {'gcomc': df_gcomc},
    'viirs_sr_vnp09ga': {'viirs_sr_vnp09ga': df_viirs_sr_vnp09ga},
    'viirs_sr_vj109ga': {'viirs_sr_vj109ga': df_viirs_sr_vj109ga},
    'viirs_sr_vj209ga': {'viirs_sr_vj209ga': df_viirs_sr_vj209ga},
}

fallback_sensor_groups = {
    'mcd43a3_bluesky': mcd43a3_split,
    'viirs_vj143ma3_bluesky': {'viirs_vj143ma3_bluesky': df_vj143ma3},
    'viirs_vnp43ma3_bluesky': {'viirs_vnp43ma3_bluesky': df_vnp43ma3},
}

daily_group_names = list(daily_sensor_groups.keys())
fallback_group_names = list(fallback_sensor_groups.keys())

scenario_specs = []
for r_daily in range(1, len(daily_group_names) + 1):
    for daily_combo in combinations(daily_group_names, r_daily):
        scenario_specs.append({
            'family': 'daily',
            'group_names': list(daily_combo),
        })

for r_fb in range(1, len(fallback_group_names) + 1):
    for fallback_combo in combinations(fallback_group_names, r_fb):
        scenario_specs.append({
            'family': '16day',
            'group_names': list(fallback_combo),
        })

print(f"Total scenarios to process: {len(scenario_specs)}")
print(f"Plot figures: {PLOT_FIGURES}")

#%% move on to processing each scenario and calculating calibration coefficients
# Initialize results storage with scenario IDs
calibration_results = []

for scenario_id, spec in enumerate(scenario_specs, start=1):
    scenario_family = spec['family']
    group_names = spec['group_names']
    scenario_name = f"{scenario_family}[{'+'.join(group_names)}]"
    print(f"\nProcessing scenario {scenario_id}/{len(scenario_specs)}: {scenario_name}")

    sensor_sources = {}
    if scenario_family == 'daily':
        for group_name in group_names:
            sensor_sources.update(daily_sensor_groups[group_name])
    else:
        for group_name in group_names:
            sensor_sources.update(fallback_sensor_groups[group_name])

    df_rs_combined = build_combined_rs_dataset(sensor_sources)

    # Merge with AWS data
    df_merged = pd.merge(
        df_aws[['time', 'aws', 'albedo_aws']],
        df_rs_combined,
        on=['time', 'aws'],
        how='inner'
    )

    # Remove NaN values
    df_merged = df_merged.dropna(subset=['albedo_aws', 'albedo_rs'])

    if len(df_merged) < 10:
        print(f"  Insufficient data points after merge: {len(df_merged)}, skipping")
        continue

    # Split into training and testing data (70-30 split)
    df_train, df_test = train_test_split(df_merged, test_size=0.3, random_state=42)

    if len(df_train) < 5 or len(df_test) < 5:
        print("  Insufficient training or test data after split, skipping")
        continue

    # Calculate metrics on training data
    train_metrics = calculate_metrics(df_train['albedo_aws'], df_train['albedo_rs'])

    # Derive calibration coefficients from training data
    slope = train_metrics['slope']
    intercept = train_metrics['intercept']

    if PLOT_FIGURES:
        test_calib_metrics = create_calibration_plot(
            scenario_id, scenario_name, df_train, df_test,
            slope, intercept, train_metrics, train_metrics
        )
    else:
        df_test_calib = df_test.copy()
        df_test_calib['albedo_rs_calibrated'] = slope * df_test['albedo_rs'] + intercept
        df_test_calib['albedo_rs_calibrated'] = df_test_calib['albedo_rs_calibrated'].clip(0, 1)
        test_calib_metrics = calculate_metrics(df_test['albedo_aws'], df_test_calib['albedo_rs_calibrated'])

    result = {
        'scenario_id': scenario_id,
        'scenario': scenario_name,
        'workflow': 'daily_only_or_16day_only_combinations',
        'scenario_family': scenario_family,
        'daily_sensor_groups': ', '.join(group_names) if scenario_family == 'daily' else '',
        'fallback_sensor_groups': ', '.join(group_names) if scenario_family == '16day' else '',
        'daily_sensors': ', '.join(sensor_sources.keys()) if scenario_family == 'daily' else '',
        'fallback_sensors': ', '.join(sensor_sources.keys()) if scenario_family == '16day' else '',
        'n_train': len(df_train),
        'n_test': len(df_test),
        'n_total': len(df_merged),
        'n_daily_used': len(df_merged) if scenario_family == 'daily' else 0,
        'n_fallback_used': len(df_merged) if scenario_family == '16day' else 0,
        'train_r_squared': train_metrics['r_squared'],
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_bias': train_metrics['bias'],
        'test_calib_r_squared': test_calib_metrics['r_squared'],
        'test_calib_rmse': test_calib_metrics['rmse'],
        'test_calib_mae': test_calib_metrics['mae'],
        'test_calib_bias': test_calib_metrics['bias'],
        'train_slope': train_metrics['slope'],
        'train_intercept': train_metrics['intercept'],
        'test_calib_slope': test_calib_metrics['slope'],
        'test_calib_intercept': test_calib_metrics['intercept'],
        'slope': slope,
        'intercept': intercept,
    }

    calibration_results.append(result)

    print(f"  Rows merged: {len(df_merged)}")
    print(f"  Scenario family: {scenario_family}")
    print(f"  Training: N={len(df_train)}, R²={train_metrics['r_squared']:.3f}")
    print(f"  Test (calibrated): N={len(df_test)}, R²={test_calib_metrics['r_squared']:.3f}")

# Convert results to DataFrame
df_calibration = pd.DataFrame(calibration_results)

# Sort by test R² value (descending)
df_calibration = df_calibration.sort_values('test_calib_r_squared', ascending=False)

print(f"\n{'='*80}")
print(f"Calibration coefficient calculation completed!")
print(f"Total scenarios processed: {len(df_calibration)}")
print(f"\nTop 10 performing scenarios (by test R²):")
print(df_calibration[
    [
        'scenario_id',
        'scenario',
        'train_r_squared',
        'test_calib_r_squared',
        'train_slope',
        'train_intercept',
        'test_calib_slope',
        'test_calib_intercept',
    ]
].head(10).to_string(index=False))

# Save detailed results to CSV
output_path = os.path.join(calibration_output_folder, 'calibration_coefficients.csv')
df_calibration.to_csv(output_path, index=False)
print(f"\nDetailed results saved to: {output_path}")

#%% Summary visualization
print(f"\nFigures saved to: {figure_output_folder}")

# %%

