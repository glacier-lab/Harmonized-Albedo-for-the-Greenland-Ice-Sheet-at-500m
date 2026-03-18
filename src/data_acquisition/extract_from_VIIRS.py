#%%
"""
Extract albedo from VIIRS daily mosaic GeoTIFF files at AWS locations.
Handles both black-sky albedo (BSA) and white-sky albedo (WSA) from dual-band files.
Handles CEN AWS reinstallation on 2017-07-25.
"""
import os
import glob
# import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer, CRS

# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)
# viirs_path = '/data_3/shunan_2/AU/hsa500m/VIIRS_mosaics/VJ143MA3/'
# csv_output_path = '/data_3/shunan_2/AU/hsa500m/VIIRS/albedo_VJ143MA3.csv'
viirs_path = '/data_3/shunan_2/AU/hsa500m/VIIRS_mosaics/VNP43MA3/'
csv_output_path = '/data_3/shunan_2/AU/hsa500m/VIIRS/albedo_VNP43MA3.csv'

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
    f.write('aws,time,albedo_viirs_bsa,albedo_viirs_wsa\n')

# %%
# Find all VIIRS mosaic files
viirs_files = sorted(glob.glob(os.path.join(viirs_path, 'VIIRS_Albedo_*.tif')))

if len(viirs_files) == 0:
    print(f"No VIIRS files found in {viirs_path}, exiting.")
else:
    total_records = 0
    skipped_files = 0

    for i, file_path in enumerate(viirs_files):
        imname = os.path.basename(file_path)
        
        try:
            # Extract date from filename: VIIRS_Albedo_YYYYMMDD_500m.tif
            date_str = imname.split('_')[2]
            imtime = pd.to_datetime(date_str, format='%Y%m%d')
            year = imtime.year
            
            print(f"Processing {i+1}/{len(viirs_files)}: {imname}")

            # Open GeoTIFF with rioxarray (reads all bands)
            ds = xr.open_dataarray(file_path, engine='rasterio')
            
            # Get AWS data for this year
            df_aws_year = df_aws[df_aws['year'] == year]
            
            if len(df_aws_year) == 0:
                print(f"  ⚠ No AWS data for year {year}, skipping")
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
                    # Extract both bands using nearest neighbor
                    # Band 1: Black-sky albedo (BSA)
                    albedo_bsa = ds.sel(band=1).sel(x=x_proj, y=y_proj, method='nearest').values.item()
                    # Band 2: White-sky albedo (WSA)
                    albedo_wsa = ds.sel(band=2).sel(x=x_proj, y=y_proj, method='nearest').values.item()
                    
                    # Keep NaN values to track cloudy/missing data
                    albedo_records.append({
                        'aws': aws_name,
                        'time': imtime,
                        'albedo_viirs_bsa': albedo_bsa,
                        'albedo_viirs_wsa': albedo_wsa
                    })
                except Exception as e:
                    print(f"  ⚠ Failed to extract value for AWS {aws_name}: {e}")
            
            if albedo_records:
                df_viirs = pd.DataFrame(albedo_records)
                df_viirs.to_csv(csv_output_path, mode='a', header=False, index=False)
                total_records += len(df_viirs)
                print(f"  ✓ Extracted {len(df_viirs)} records")
            
            ds.close()
            
        except Exception as e:
            print(f"  ✗ Error processing {imname}: {type(e).__name__}: {e}")
            skipped_files += 1
            continue

    print(f"\n{'='*60}")
    print(f"Done. Wrote {total_records} rows to {csv_output_path}")
    print(f"Skipped {skipped_files} files due to errors")
    print(f"{'='*60}")