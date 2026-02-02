#%% 
import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from pyproj import Transformer, CRS

# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)
sice_path = '/data_3/shunan_2/AU/hsa500m/SICE'
csv_output_path = '/data_3/shunan_2/AU/hsa500m/SICE/albedo_sice_exception.csv'

# AWS CEN reinstallation date
CEN_REINSTALLATION_DATE = '2017-07-25'

# Identify AWS sites that have 2016 data (these need special handling in 2017)
aws_with_2016 = set(df_aws[df_aws['year'] == 2016]['aws'].unique())

# Convert AWS lat/lon to EPSG:3413 (x, y in meters)
crs_wgs84 = CRS.from_epsg(4326)  # WGS84
crs_3413 = CRS.from_epsg(3413)   # EPSG:3413

transformer = Transformer.from_crs(crs_wgs84, crs_3413, always_xy=True)
x_proj, y_proj = transformer.transform(df_aws['lon'].values, df_aws['lat'].values)
df_aws['x_proj'] = x_proj
df_aws['y_proj'] = y_proj

# Create a lookup dictionary for 2016 coordinates (name -> (x_proj, y_proj))
df_aws_2016 = df_aws[df_aws['year'] == 2016]
coords_2016 = {row['aws']: (row['x_proj'], row['y_proj']) 
               for _, row in df_aws_2016.iterrows()}

# Create output CSV with header
with open(csv_output_path, 'w') as f:
    f.write('aws,time,albedo_sice\n')
# %%
years = np.arange(2017, 2026)
total_records = 0
skipped_files = 0

for year in years:

    sice_files_year = sorted(glob.glob(os.path.join(sice_path, f'**/*{year}*.nc'), recursive=True))
    df_aws_year = df_aws[df_aws['year'] == year]

    if len(sice_files_year) == 0:
        print(f"No files found for year {year}, skipping.")
        continue

    for i in range(len(sice_files_year)):

        file_path = sice_files_year[i]
        imname = os.path.basename(file_path)

        try:
            imtime = os.path.splitext(imname)[0].split('_')[-1]
            imtime = pd.to_datetime(imtime)
            print(f"Processing {i+1}/{len(sice_files_year)} for year {year}: {imname}")

            ds_sice = xr.open_dataset(file_path)
            ds_albedo = ds_sice['albedo_bb_planar_sw']
            x_coord = ds_sice['x'].values  # 1D array
            y_coord = ds_sice['y'].values  # 1D array

            # Create 2D meshgrid for distance calculations
            x_grid, y_grid = np.meshgrid(x_coord, y_coord)

            # Find nearest grid indices for each AWS site
            grid_indices = []
            aws_names = []
            
            for aws_idx, row in df_aws_year.iterrows():
                aws_name = row['aws']
                aws_names.append(aws_name)
                
                # Special handling for 2017: use 2016 coordinates if before CEN reinstallation
                if year == 2017 and aws_name in aws_with_2016 and imtime < pd.Timestamp(CEN_REINSTALLATION_DATE):
                    x_aws, y_aws = coords_2016[aws_name]
                else:
                    x_aws = row['x_proj']
                    y_aws = row['y_proj']
                
                # Find nearest grid point
                dist2 = (x_grid - x_aws) ** 2 + (y_grid - y_aws) ** 2
                min_idx = np.unravel_index(dist2.argmin(), dist2.shape)
                grid_indices.append(min_idx)

            # Extract albedo values at nearest grid points
            albedo_values = []
            for y_idx, x_idx in grid_indices:
                albedo_at_point = ds_albedo[y_idx, x_idx]
                albedo_values.append(albedo_at_point.values)
            
            albedo_array = np.vstack(albedo_values)

            df_sice = pd.DataFrame({
                'aws': aws_names,
                'time': imtime,
                'albedo_sice': albedo_array.flatten()
            })
            
            df_sice.to_csv(csv_output_path, mode='a', header=False, index=False)
            total_records += len(df_sice)
            
            ds_sice.close()
            print(f"  ✓ Extracted {len(df_sice)} records")
            
        except Exception as e:
            print(f"  ✗ Error processing {imname}: {type(e).__name__}: {e}")
            skipped_files += 1
            continue

print(f"\n{'='*60}")
print(f"Done. Wrote {total_records} rows to {csv_output_path}")
print(f"Skipped {skipped_files} files due to errors")
print(f"{'='*60}")

