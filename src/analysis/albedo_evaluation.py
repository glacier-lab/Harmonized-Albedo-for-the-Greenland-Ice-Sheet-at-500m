#%% 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np

#%% 
def setup_plotting_style():
    """Set up the default plotting style."""
    sns.set_theme(style="darkgrid", font_scale=1.5)

def load_and_preprocess_data(aws_path, rs_path):
    """Load and preprocess AWS and remote sensing albedo data."""
    # Load data
    aws_data = pd.read_csv(aws_path)
    rs_data = pd.read_csv(rs_path)
    
    # Process remote sensing data
    rs_data['time'] = pd.to_datetime(rs_data['time'])
    
    # Process AWS data
    aws_data['time'] = pd.to_datetime(aws_data['time'])
    
    # Merge data on time and aws
    df = pd.merge(
        aws_data, 
        rs_data, 
        on=['time', 'aws'], 
        how='inner'
    )
    
    # Identify the remote sensing albedo column (starts with 'albedo_')
    rs_albedo_cols = [col for col in df.columns if col.startswith('albedo_') and col != 'albedo']
    if not rs_albedo_cols:
        raise ValueError("No remote sensing albedo column found (expected column starting with 'albedo_')")
    
    # Use the first matching column as remote sensing albedo
    rs_albedo_col = rs_albedo_cols[0]
    
    # Rename columns for consistency
    df = df.rename(columns={'albedo': 'albedo_aws', rs_albedo_col: 'albedo_rs'})

    # if any albedo values are negative or greater than 1, set them to NaN
    df.loc[(df['albedo_aws'] <= 0) | (df['albedo_aws'] >= 1), 'albedo_aws'] = np.nan
    df.loc[(df['albedo_rs'] <= 0) | (df['albedo_rs'] >= 1), 'albedo_rs'] = np.nan
    
    return df.dropna(subset=['albedo_aws', 'albedo_rs'])

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
        'n': len(y_true)
    }

def create_overall_regression_plot(df):
    """Create overall regression plot comparing remote sensing and AWS albedo."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate regression statistics
    metrics = calculate_metrics(df['albedo_aws'], df['albedo_rs'])
    
    # Create scatter plot with regression line
    sns.scatterplot(ax=ax, data=df, x='albedo_rs', y='albedo_aws', hue='aws', alpha=0.5)
    sns.regplot(ax=ax, data=df, x='albedo_rs', y='albedo_aws', 
                scatter=False, color='red', label=f'Regression (R²={metrics["r_squared"]:.3f})')
    
    # Add 1:1 reference line
    min_val = min(df['albedo_rs'].min(), df['albedo_aws'].min())
    max_val = max(df['albedo_rs'].max(), df['albedo_aws'].max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.8, label='1:1 line')
    
    # Customize plot
    ax.set_aspect('equal')
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    plt.ylabel('AWS Albedo')
    plt.xlabel('Remote Sensing Albedo')
    plt.title('Remote Sensing vs AWS Albedo Comparison')
    plt.legend()
    
    # Add statistics text box
    stats_text = f'N = {metrics["n"]}\n'
    stats_text += f'R² = {metrics["r_squared"]:.3f}\n'
    stats_text += f'RMSE = {metrics["rmse"]:.3f}\n'
    stats_text += f'MAE = {metrics["mae"]:.3f}\n'
    stats_text += f'Bias = {metrics["bias"]:.3f}\n'
    stats_text += f'Slope = {metrics["slope"]:.3f}\n'
    stats_text += f'Intercept = {metrics["intercept"]:.3f}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Print statistics
    print(f"Overall Statistics:")
    for key, value in metrics.items():
        if key != 'n':
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return fig

def create_station_subplots(df):
    """Create subplots for each AWS station."""
    unique_aws = df['aws'].unique()
    n_aws = len(unique_aws)
    n_cols = 3
    n_rows = (n_aws + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, aws in enumerate(unique_aws):
        plot_single_station(df, aws, axes[idx])
    
    # Remove empty subplots
    for idx in range(len(unique_aws), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig

def plot_single_station(df, aws, ax):
    """Plot regression for a single AWS station."""
    aws_data = df[df['aws'] == aws]
    
    # Calculate regression statistics
    metrics = calculate_metrics(aws_data['albedo_aws'], aws_data['albedo_rs'])
    
    # Create plots
    sns.scatterplot(data=aws_data, x='albedo_rs', y='albedo_aws', ax=ax, alpha=0.5)
    sns.regplot(data=aws_data, x='albedo_rs', y='albedo_aws', 
                scatter=False, color='red', ax=ax)
    
    # Add 1:1 line
    min_val = min(aws_data['albedo_rs'].min(), aws_data['albedo_aws'].min())
    max_val = max(aws_data['albedo_rs'].max(), aws_data['albedo_aws'].max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.8)
    
    # Customize plot
    ax.set_title(f'AWS: {aws} (N={metrics["n"]}, R²={metrics["r_squared"]:.2f})')
    ax.set_xlabel('Remote Sensing Albedo')
    ax.set_ylabel('AWS Albedo')
    ax.set_aspect('equal')
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    
    # Print statistics
    print(f"\nStatistics for AWS {aws}:")
    for key, value in metrics.items():
        if key != 'n':
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

def create_time_series_plots(df):
    """Create time series plots for each AWS station."""
    unique_aws = df['aws'].unique()
    n_aws = len(unique_aws)
    n_cols = 2
    n_rows = (n_aws + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=True)
    axes = axes.flatten()

    for idx, aws in enumerate(unique_aws):
        plot_time_series(df, aws, axes[idx])

    # Remove empty subplots
    for idx in range(len(unique_aws), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig

def plot_time_series(df, aws, ax):
    """Plot time series for a single AWS station."""
    aws_data = df[df['aws'] == aws].sort_values('time')

    # Plot AWS data as a line
    ax.plot(aws_data['time'], aws_data['albedo_aws'], marker='', linestyle='-', 
            color='blue', label='AWS Albedo', linewidth=2)

    # Plot remote sensing data as scatter points
    ax.scatter(aws_data['time'], aws_data['albedo_rs'], marker='o', 
               color='red', label='Remote Sensing Albedo', alpha=0.7, s=30)

    # Customize plot
    ax.set_title(f'AWS: {aws}')
    ax.set_ylabel('Albedo')
    ax.set_xlabel('Date')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True)

def main():
    """Main function to run the analysis."""
    setup_plotting_style()
    
    # File paths
    aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/promice_day.csv'
    rs_path = '/data_3/shunan_2/AU/hsa500m/SICE/albedo_sice.csv'

    # Load and process data
    df = load_and_preprocess_data(aws_path, rs_path)
    
    print(f"Total number of matched observations: {len(df)}")
    print(f"Number of AWS stations: {df['aws'].nunique()}")
    print(f"AWS stations: {df['aws'].unique()}\n")
    
    # Create plots
    create_overall_regression_plot(df)
    plt.show()
    
    create_station_subplots(df)
    plt.show()
    
    create_time_series_plots(df)
    plt.show()

if __name__ == '__main__':
    main()
# %%
