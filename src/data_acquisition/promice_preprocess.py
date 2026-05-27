#%%
import pandas as pd
import numpy as np
import os
import glob
import re
import utm
# %%
folder_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/day'
all_files = glob.glob(os.path.join(folder_path, '*.csv'))
csv_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/promice_day.csv'
annual_drift_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
csv4qgis_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/promice_location_for_qgis.csv'
# %%
aws_list = sorted({re.sub(r'_day$', '', os.path.splitext(os.path.basename(f))[0]) for f in all_files})
# exclude AWSs on bedrock (end with 'B'), and LYN_L
aws_list = [aws for aws in aws_list if not aws.endswith('B') and aws != 'LYN_L']
csv_files = {aws: os.path.join(folder_path, f"{aws}_day.csv") for aws in aws_list}

#%%
# create csv file for storing output, overwrite if exists
with open(csv_path, 'w') as f:
    f.write('time,albedo,lat,lon,alt,cc,aws\n')
# %%
for i in range(len(aws_list)):
    aws = aws_list[i]
    file_path = csv_files[aws]
    print(f"Processing {i+1}/{len(aws_list)}: {aws}")
    df = pd.read_csv(file_path)
    # select relevant columns
    df = df[['time', 'albedo', 'lat', 'lon', 'alt', 'cc']]
    # drop nan values if any, count and print how many rows are dropped
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"Dropped {n_before - n_after} rows with NaN values")
    # add aws column
    df['aws'] = aws
    # append to csv file
    df.to_csv(csv_path, mode='a', header=False, index=False)
print("All done!")

# %% post-process aws data
df = pd.read_csv(csv_path)
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
# exclude data acquired before 2000-01-01
df = df[df['time'] >= pd.Timestamp('2000-01-01')]
# average lat and lon for each aws in each year
# for CEN exclude gps records before reinstall date (2017-07-25). This AWS needs special treatment due to relocation.
df_filtered = df.copy()
cutoff = pd.Timestamp('2017-07-25')
mask = ~((df_filtered['aws'] == 'CEN') & (df_filtered['year'] == 2017) & (df_filtered['time'] < cutoff))
df_filtered = df_filtered[mask]
avg_coords = df_filtered.groupby(['aws', 'year'])[['lat', 'lon', 'alt']].mean().reset_index()
# add UTM coordinates 
avg_coords[['utm_easting', 'utm_northing', 'utm_zone_number', 'utm_zone_letter']] = avg_coords.apply(
    lambda row: pd.Series(utm.from_latlon(row['lat'], row['lon'])), axis=1)
# report the annual AWS drift in meters
drift_report = []
for aws in aws_list:
    aws_data = avg_coords[avg_coords['aws'] == aws].sort_values('year')
    aws_data = aws_data.reset_index(drop=True)
    for j in range(1, len(aws_data)):
        prev_row = aws_data.loc[j-1]
        curr_row = aws_data.loc[j]
        drift = np.sqrt((curr_row['utm_easting'] - prev_row['utm_easting'])**2 + 
                        (curr_row['utm_northing'] - prev_row['utm_northing'])**2)
        drift_report.append((aws, prev_row['year'], curr_row['year'], drift))
# add to avg_coords dataframe
drift_df = pd.DataFrame(drift_report, columns=['aws', 'year_start', 'year_end', 'drift_meters'])
avg_coords = avg_coords.merge(drift_df, left_on=['aws', 'year'], right_on=['aws', 'year_end'], how='left')
avg_coords = avg_coords.drop(columns=['year_start', 'year_end'])
avg_coords.to_csv(annual_drift_path, index=False)
# %%
df_location = avg_coords.groupby('aws')[['lat', 'lon', 'alt']].mean().reset_index()
df_location.to_csv(csv4qgis_path, index=False)