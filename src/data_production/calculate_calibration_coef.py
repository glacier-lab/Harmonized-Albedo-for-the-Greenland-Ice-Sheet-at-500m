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
from itertools import combinations
import os
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
PLOT_FIGURES = True # Set to False to skip plotting and only calculate metrics

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

def is_valid_combination(sensors):
    """
    Check if a combination of sensors is valid.
    Rules:
    - Cannot mix base sensor (e.g., 'mod10') with its year-specific variants (e.g., 'mod10_2020')
    - Cannot mix different year variants of the same sensor (e.g., 'mod10_2020' with 'mod10_2021')
    - Cannot mix year-specific sensors from different years (e.g., 'mod10_2020' with 'myd10_2021')
    - Cannot mix pre-drift mod10 with year-specific myd10 (e.g., 'mod10' with 'myd10_2021')
    - Exception: pre-drift myd10 CAN be combined with mod10_2020 (since myd10 drift starts in 2021)
    """
    # Extract base sensor names and years
    sensor_info = {}
    years_in_combo = set()
    has_mod10_predrift = False
    has_myd10_predrift = False
    has_mod10_year = False
    has_myd10_year = False
    mod10_year_value = None
    myd10_year_value = None
    
    for sensor in sensors:
        # Get the base name and year (if any)
        if '_' in sensor and sensor.split('_')[-1].isdigit():
            base = sensor.rsplit('_', 1)[0]
            year = sensor.rsplit('_', 1)[1]
            years_in_combo.add(year)
            
            # Track if we have year-specific MODIS sensors
            if base == 'mod10':
                has_mod10_year = True
                mod10_year_value = int(year)
            elif base == 'myd10':
                has_myd10_year = True
                myd10_year_value = int(year)
        else:
            base = sensor
            year = None
            
            # Track if we have pre-drift MODIS sensors
            if base == 'mod10':
                has_mod10_predrift = True
            elif base == 'myd10':
                has_myd10_predrift = True
        
        if base not in sensor_info:
            sensor_info[base] = []
        sensor_info[base].append((sensor, year))
    
    # Check if any base sensor has multiple variants
    for base, variants in sensor_info.items():
        if len(variants) > 1:
            return False
    
    # Check if there are multiple different years in the combination
    if len(years_in_combo) > 1:
        return False
    
    # Check for invalid MODIS combinations across sensors
    # Cannot mix pre-drift mod10 with year-specific myd10 (any year >= 2021)
    if has_mod10_predrift and has_myd10_year:
        if myd10_year_value >= 2021:
            return False
    
    # Exception: pre-drift myd10 (pre-2021) CAN be combined with mod10_2020
    # This is valid, so no check needed for this case
    
    # Cannot mix pre-drift myd10 with year-specific mod10 >= 2021
    if has_myd10_predrift and has_mod10_year:
        if mod10_year_value >= 2021:
            return False
    
    return True

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

# Prepare sensor data with expanded MODIS scenarios for orbital drift
# First, identify years for MODIS orbital drift scenarios
mod10_years = sorted([y for y in df_mod10['time'].dt.year.unique() if y >= 2020])
myd10_years = sorted([y for y in df_myd10['time'].dt.year.unique() if y >= 2021])

# Create expanded sensor list with MODIS year-specific scenarios
sensors_expanded = {}

# Add base sensors with no orbital drift split
sensors_expanded['sice'] = {'data': df_sice, 'year_filter': None}
sensors_expanded['gcomc'] = {'data': df_gcomc, 'year_filter': None}
sensors_expanded['viirs_vj143ma3_bluesky'] = {'data': df_vj143ma3, 'year_filter': None}
sensors_expanded['viirs_vnp43ma3_bluesky'] = {'data': df_vnp43ma3, 'year_filter': None}

# Add MOD10 scenarios - pre-drift and yearly drift scenarios
mod10_pre_drift = df_mod10[df_mod10['time'].dt.year < 2020]
if len(mod10_pre_drift) > 0:
    sensors_expanded['mod10'] = {'data': mod10_pre_drift, 'year_filter': None}

for year in mod10_years:
    sensors_expanded[f'mod10_{year}'] = {
        'data': df_mod10[df_mod10['time'].dt.year == year],
        'year_filter': year
    }

# Add MYD10 scenarios - pre-drift and yearly drift scenarios  
myd10_pre_drift = df_myd10[df_myd10['time'].dt.year < 2021]
if len(myd10_pre_drift) > 0:
    sensors_expanded['myd10'] = {'data': myd10_pre_drift, 'year_filter': None}

for year in myd10_years:
    sensors_expanded[f'myd10_{year}'] = {
        'data': df_myd10[df_myd10['time'].dt.year == year],
        'year_filter': year
    }

# Add MCD43A3 scenarios - pre-drift (before 2020) and yearly drift scenarios (2020 onwards)
# MCD43A3 combines Terra and Aqua, so it follows MOD10A1 drift timeline (drift starts 2020)
mcd43a3_years = sorted([y for y in df_mcd43a3['time'].dt.year.unique() if y >= 2020])

mcd43a3_pre_drift = df_mcd43a3[df_mcd43a3['time'].dt.year < 2020]
if len(mcd43a3_pre_drift) > 0:
    sensors_expanded['mcd43a3_bluesky'] = {'data': mcd43a3_pre_drift, 'year_filter': None}

for year in mcd43a3_years:
    sensors_expanded[f'mcd43a3_bluesky_{year}'] = {
        'data': df_mcd43a3[df_mcd43a3['time'].dt.year == year],
        'year_filter': year
    }

print(f"Expanded sensor list: {list(sensors_expanded.keys())}")

# Generate all possible combinations (1 to n sensors)

scenarios = []

# Scenario 0 placeholder (empty), so CARRA can be added later if needed.
scenarios.append([])

# All combinations of configured sensors
non_carra_sensors = [s for s in sensors_expanded.keys()]
sensor_indicator_columns = sorted(non_carra_sensors)

for r in range(1, len(non_carra_sensors) + 1):
    for combo in combinations(non_carra_sensors, r):
        # Check if this is a valid combination (no mixing of base and year-specific variants)
        if is_valid_combination(combo):
            scenarios.append(list(combo))

print(f"Total number of valid scenarios: {len(scenarios)}")
print(f"Plot figures: {PLOT_FIGURES}")

#%% move on to processing each scenario and calculating calibration coefficients
# Initialize results storage with scenario IDs
calibration_results = []

# Process each scenario
for scenario_id, scenario in enumerate(scenarios, start=0):
    scenario_name = '_'.join(scenario)
    print(f"\nProcessing scenario {scenario_id}/{len(scenarios)}: {scenario_name}")
    
    # Collect and combine remote sensing data for this scenario
    combined_rs_data = []
    
    for sensor in scenario:
        # Use expanded sensor data
        df_sensor = sensors_expanded[sensor]['data'].copy()
        
        if len(df_sensor) > 0:
            combined_rs_data.append(df_sensor[['time', 'aws', 'albedo_rs']])
    
    if len(combined_rs_data) == 0:
        print(f"  No data available for this scenario")
        continue
    
    # Concatenate all sensor data
    df_rs_combined = pd.concat(combined_rs_data, ignore_index=True)
    
    # Calculate average daily remote sensing albedo
    df_rs_avg = df_rs_combined.groupby(['time', 'aws']).agg({
        'albedo_rs': 'mean'
    }).reset_index()
    
    # Merge with AWS data
    df_merged = pd.merge(
        df_aws[['time', 'aws', 'albedo_aws']],
        df_rs_avg,
        on=['time', 'aws'],
        how='inner'
    )
    
    # Remove NaN values
    df_merged = df_merged.dropna(subset=['albedo_aws', 'albedo_rs'])
    
    if len(df_merged) < 10:  # Minimum sample size requirement
        print(f"  Insufficient data points ({len(df_merged)}), skipping")
        continue
    
    # Split into training and testing data (70-30 split)
    df_train, df_test = train_test_split(df_merged, test_size=0.3, random_state=42)
    
    if len(df_train) < 5 or len(df_test) < 5:
        print(f"  Insufficient training or test data after split, skipping")
        continue
    
    # Calculate metrics on training data
    train_metrics = calculate_metrics(df_train['albedo_aws'], df_train['albedo_rs'])
    
    # Derive calibration coefficients from training data
    slope = train_metrics['slope']
    intercept = train_metrics['intercept']
    
    try:
        if PLOT_FIGURES:
            # Create calibration plot
            test_calib_metrics = create_calibration_plot(
                scenario_id, scenario_name, df_train, df_test,
                slope, intercept, train_metrics, train_metrics
            )
        else:
            # Calculate calibrated test metrics without plotting.
            df_test_calib = df_test.copy()
            df_test_calib['albedo_rs_calibrated'] = slope * df_test['albedo_rs'] + intercept
            df_test_calib['albedo_rs_calibrated'] = df_test_calib['albedo_rs_calibrated'].clip(0, 1)
            test_calib_metrics = calculate_metrics(df_test['albedo_aws'], df_test_calib['albedo_rs_calibrated'])
        
        # Store results
        result = {
            'scenario_id': scenario_id,
            'scenario': scenario_name,
            'sensors': ', '.join(scenario),
            'n_sensors': len(scenario),
            'n_train': len(df_train),
            'n_test': len(df_test),
            'n_total': len(df_merged),
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
            'intercept': intercept
        }

        for sensor_name in sensor_indicator_columns:
            result[sensor_name] = int(sensor_name in scenario)
        
        calibration_results.append(result)
        
        print(f"  Training: N={len(df_train)}, R²={train_metrics['r_squared']:.3f}")
        print(f"  Test (calibrated): N={len(df_test)}, R²={test_calib_metrics['r_squared']:.3f}")
        
    except Exception as e:
        print(f"  Error processing scenario: {e}")
        continue

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

