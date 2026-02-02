#%%
import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import re

# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)
carra_path = '/data_3/shunan_2/AU/hsa500m/CARRA/GL500m'
csv_output_path = '/data_3/shunan_2/AU/hsa500m/CARRA/albedo_carra_exception.csv'

# AWS CEN reinstallation date
CEN_REINSTALLATION_DATE = '2017-07-25'

# Identify AWS sites that have 2016 data (these need special handling in 2017)
aws_with_2016 = set(df_aws[df_aws['year'] == 2016]['aws'].unique())

# Create a lookup dictionary for 2016 coordinates (name -> (lat, lon))
df_aws_2016 = df_aws[df_aws['year'] == 2016]
coords_2016 = {row['aws']: (row['lat'], row['lon']) 
               for _, row in df_aws_2016.iterrows()}

# create csv file for storing output, overwrite if exists
with open(csv_output_path, 'w') as f:
    f.write('aws,time,albedo_carra\n')

# %%
years = np.arange(2000, 2026)
total_records = 0
skipped_files = 0

for year in years:

    carra_files_year = sorted(glob.glob(os.path.join(carra_path, f'**/*{year}*.nc'), recursive=True))
    df_aws_year = df_aws[df_aws['year'] == year]

    if len(carra_files_year) == 0:
        print(f"No files found for year {year}, skipping.")
        continue

    for i in range(len(carra_files_year)):

        file_path = carra_files_year[i]
        imname = os.path.basename(file_path)

        try:
            imtime = os.path.splitext(imname)[0].split('_')[3]
            imtime = re.split(r'\D+', imtime)[-1]
            imtime = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(imtime)-1)

            print(f"Processing {i+1}/{len(carra_files_year)} for year {year}: {imname}")

            ds_carra = xr.open_dataset(file_path)
            ds_albedo = ds_carra['al']

            # Get the 2D coordinate arrays
            Y_2d = ds_carra['Y'].values  # shape (Y, X)
            X_2d = ds_carra['X'].values  # shape (Y, X)
            
            # Find nearest grid indices for each AWS site
            grid_indices = []
            aws_names = []
            
            for aws_idx, row in df_aws_year.iterrows():
                aws_name = row['aws']
                aws_names.append(aws_name)
                
                # Special handling for 2017: use 2016 coordinates if before CEN reinstallation
                if year == 2017 and aws_name in aws_with_2016 and imtime < pd.Timestamp(CEN_REINSTALLATION_DATE):
                    lat, lon = coords_2016[aws_name]
                else:
                    lat = row['lat']
                    lon = row['lon']
                
                # Calculate the squared distance to find the nearest point
                dist2 = (Y_2d - lat) ** 2 + (X_2d - lon) ** 2
                # Find the 2D index of the minimum distance
                min_idx = np.unravel_index(dist2.argmin(), dist2.shape)
                grid_indices.append(min_idx)

            # Extract albedo values at the nearest grid points
            albedo_values = []
            for y_idx, x_idx in grid_indices:
                # Extract albedo at this grid point: dims (time, zlev)
                albedo_at_point = ds_albedo[:, :, y_idx, x_idx]
                albedo_values.append(albedo_at_point.values)
            # Stack into a 2D array: shape (time, sites)
            albedo_array = np.vstack(albedo_values)

            df_carra = pd.DataFrame({
                'aws': aws_names,
                'time': imtime,
            })
            df_carra['albedo_carra'] = albedo_array
            
            # append to csv file
            df_carra.to_csv(csv_output_path, mode='a', header=False, index=False)
            total_records += len(df_carra)
            
            ds_carra.close()
            print(f"  ✓ Extracted {len(df_carra)} records")

        except Exception as e:
            print(f"  ✗ Error processing {imname}: {type(e).__name__}: {e}")
            skipped_files += 1
            continue

print(f"\n{'='*60}")
print(f"Done. Wrote {total_records} rows to {csv_output_path}")
print(f"Skipped {skipped_files} files due to errors")
print(f"{'='*60}")