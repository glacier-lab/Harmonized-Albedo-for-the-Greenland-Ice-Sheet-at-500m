#%%
"""
Extract albedo from SICE rebuild GeoTIFF files at AWS locations.
Handles CEN AWS reinstallation on 2017-07-25.
"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer, CRS
from tqdm import tqdm

# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)
sice_rebuild_path = '/data_3/shunan_2/AU/hsa500m/SICE_rebuild'
csv_output_path = '/data_3/shunan_2/AU/hsa500m/SICE/albedo_sice_rebuild.csv'

# AWS CEN reinstallation date
CEN_REINSTALLATION_DATE = '2017-07-25'
CEN_NAME = 'CEN'  # The specific station affected by the reinstallation

# Get CEN coordinates for 2016 (if available)
df_cen_2016 = df_aws[(df_aws['aws'] == CEN_NAME) & (df_aws['year'] == 2016)]
cen_coords_2016 = None
if not df_cen_2016.empty:
    cen_coords_2016 = (df_cen_2016.iloc[0]['lat'], df_cen_2016.iloc[0]['lon'])

# Set up coordinate transformer (WGS84 to EPSG:3413)
crs_wgs84 = CRS.from_epsg(4326)
crs_3413 = CRS.from_epsg(3413)
transformer = Transformer.from_crs(crs_wgs84, crs_3413, always_xy=True)

# Create output CSV with header
with open(csv_output_path, 'w') as f:
    f.write('aws,time,albedo_sice_rebuild\n')

# %%
# Find all SICE rebuild mosaic files
sice_files = sorted(glob.glob(os.path.join(sice_rebuild_path, 'SICE_Albedo_*.tif')))

print(f"\n{'='*60}")
print("Extract SICE Rebuild Albedo at AWS Locations")
print(f"{'='*60}")
print(f"Found {len(sice_files)} SICE rebuild files\n")

if len(sice_files) == 0:
    print(f"No SICE rebuild files found in {sice_rebuild_path}, exiting.")
else:
    total_records = 0
    skipped_files = 0

    for file_path in tqdm(sice_files, desc="Processing Files", unit="file"):
        imname = os.path.basename(file_path)
        
        try:
            # Extract date from filename: SICE_Albedo_YYYYMMDD_500m.tif
            date_str = imname.split('_')[2]
            imtime = pd.to_datetime(date_str, format='%Y%m%d')
            year = imtime.year
            
            # Open GeoTIFF with rioxarray
            ds = xr.open_dataarray(file_path, engine='rasterio')
            
            # Get AWS data for this year
            df_aws_year = df_aws[df_aws['year'] == year]
            
            if len(df_aws_year) == 0:
                tqdm.write(f"⚠ {imname}: No AWS data for year {year}")
                skipped_files += 1
                continue
            
            albedo_records = []
            
            for aws_idx, row in df_aws_year.iterrows():
                aws_name = row['aws']
                
                # Special handling only for CEN station in 2017 before reinstallation
                if (year == 2017 and aws_name == CEN_NAME and 
                    imtime < pd.Timestamp(CEN_REINSTALLATION_DATE) and 
                    cen_coords_2016 is not None):
                    lat, lon = cen_coords_2016
                else:
                    lat = row['lat']
                    lon = row['lon']
                
                # Transform from WGS84 to EPSG:3413
                x_proj, y_proj = transformer.transform(lon, lat)
                
                try:
                    # Use nearest neighbor selection
                    albedo_value = ds.sel(x=x_proj, y=y_proj, method='nearest').values.item()
                    
                    # Keep NaN values to track cloudy/missing data
                    albedo_records.append({
                        'aws': aws_name,
                        'time': imtime,
                        'albedo_sice_rebuild': albedo_value
                    })
                except Exception as e:
                    tqdm.write(f"✗ {imname}: Failed to extract value for AWS {aws_name}: {e}")
            
            if albedo_records:
                df_sice = pd.DataFrame(albedo_records)
                df_sice.to_csv(csv_output_path, mode='a', header=False, index=False)
                total_records += len(df_sice)
                tqdm.write(f"✓ {imname}: Extracted {len(df_sice)} records")
            else:
                tqdm.write(f"⚠ {imname}: No valid records extracted")
            
            ds.close()
            
        except Exception as e:
            tqdm.write(f"✗ {imname}: {type(e).__name__}: {e}")
            skipped_files += 1
            continue

    print(f"\n{'='*60}")
    print("Extraction Summary")
    print(f"{'='*60}")
    print(f"Total files:         {len(sice_files):>5}")
    print(f"Records written:     {total_records:>5}")
    print(f"Skipped/Failed:      {skipped_files:>5}")
    print(f"Output file:         {csv_output_path}")
    print(f"{'='*60}\n")