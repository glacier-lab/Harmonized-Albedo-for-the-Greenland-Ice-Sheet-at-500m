#%%
"""
Extract albedo from GCOM-C daily mosaic GeoTIFF files at AWS locations.
Handles CEN AWS reinstallation on 2017-07-25.
"""
import os
import glob
import numpy as np
import pandas as pd
import rasterio as rio
from pyproj import Transformer, CRS

# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)
gcomc_path = '/data_3/shunan_2/AU/hsa500m/GCOMC_mosaics'
csv_output_path = '/data_3/shunan_2/AU/hsa500m/GCOMC/albedo_gcomc.csv'

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

# Convert AWS lat/lon to pixel coordinates (for rasterio)
def latlon_to_pixel(lat, lon, transform):
    """Convert lat/lon (WGS84) to pixel coordinates using rasterio transform (EPSG:3413)."""
    # First transform from WGS84 (lat/lon) to EPSG:3413 (x/y in meters)
    x_proj, y_proj = transformer.transform(lon, lat)
    
    # Then use rasterio's inverse transform to get pixel coordinates
    col, row = ~transform * (x_proj, y_proj)
    return int(np.round(row)), int(np.round(col))

# Create output CSV with header
with open(csv_output_path, 'w') as f:
    f.write('aws,time,albedo_gcomc\n')

# %%
# Find all GCOMC mosaic files
gcomc_files = sorted(glob.glob(os.path.join(gcomc_path, 'GCOMC_Albedo_*.tif')))

if len(gcomc_files) == 0:
    print(f"No GCOMC files found in {gcomc_path}, exiting.")
else:
    total_records = 0
    skipped_files = 0

    for i, file_path in enumerate(gcomc_files):
        imname = os.path.basename(file_path)
        
        try:
            # Extract date from filename: GCOMC_Albedo_YYYYMMDD_500m.tif
            date_str = imname.split('_')[2]
            imtime = pd.to_datetime(date_str, format='%Y%m%d')
            year = imtime.year
            
            print(f"Processing {i+1}/{len(gcomc_files)}: {imname}")

            with rio.open(file_path) as src:
                albedo_data = src.read(1)
                transform = src.transform
                
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
                    
                    # Convert lat/lon to pixel coordinates
                    row_idx, col_idx = latlon_to_pixel(lat, lon, transform)
                    
                    # Check bounds
                    if 0 <= row_idx < albedo_data.shape[0] and 0 <= col_idx < albedo_data.shape[1]:
                        albedo_value = albedo_data[row_idx, col_idx]
                        
                        # Handle NaN/nodata values
                        if np.isnan(albedo_value):
                            albedo_value = np.nan
                        
                        albedo_records.append({
                            'aws': aws_name,
                            'time': imtime,
                            'albedo_gcomc': albedo_value
                        })
                    else:
                        print(f"  ⚠ AWS {aws_name} out of bounds (row={row_idx}, col={col_idx})")
                
                if albedo_records:
                    df_gcomc = pd.DataFrame(albedo_records)
                    df_gcomc.to_csv(csv_output_path, mode='a', header=False, index=False)
                    total_records += len(df_gcomc)
                    print(f"  ✓ Extracted {len(df_gcomc)} records")
                
        except Exception as e:
            print(f"  ✗ Error processing {imname}: {type(e).__name__}: {e}")
            skipped_files += 1
            continue

    print(f"\n{'='*60}")
    print(f"Done. Wrote {total_records} rows to {csv_output_path}")
    print(f"Skipped {skipped_files} files due to errors")
    print(f"{'='*60}")