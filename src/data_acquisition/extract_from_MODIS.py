#%%
import geemap
import ee
import pandas as pd
from datetime import datetime
# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)
mod10_output_path = '/data_3/shunan_2/AU/hsa500m/MODIS/albedo_mod10.csv'
myd10_output_path = '/data_3/shunan_2/AU/hsa500m/MODIS/albedo_myd10.csv'

# AWS CEN reinstallation date
CEN_REINSTALLATION_DATE = '2017-07-25'

# Initialize the Earth Engine module.
Map = geemap.Map()
Map
# Create output CSV with header
with open(mod10_output_path, 'w') as f:
    f.write('aws,time,albedo_mod10\n')
with open(myd10_output_path, 'w') as f:
    f.write('aws,time,albedo_myd10\n')
# %%
# https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api-guiattard by https://github.com/guiattard
def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]]#.dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df

def extract_modis_data(collection_name, band_name, output_col_name, point, aws_name, year, start_date, end_date):
    """Extract MODIS data for a given date range and point."""
    collection = (ee.ImageCollection(collection_name)
                  .filterDate(start_date, end_date)
                  .select(band_name))
    
    # Check if collection is empty
    size = collection.size().getInfo()
    if size > 0:
        # Get the data for the point
        point_data = collection.getRegion(point, 500, crs='EPSG:3413').getInfo()
        df = ee_array_to_df(point_data, [band_name])
        df['aws'] = aws_name
        
        # Rename columns and ensure order of columns matches output csv
        df.rename(columns={band_name: output_col_name}, inplace=True)
        df = df[['aws', 'time', output_col_name]]
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df[output_col_name] = df[output_col_name] / 100
        
        return df, len(df)
    else:
        return None, 0

#%%
for i in range(len(df_aws)):
    lat = df_aws.loc[i, 'lat']
    lon = df_aws.loc[i, 'lon']
    aws_name = df_aws.loc[i, 'aws']
    year = df_aws.loc[i, 'year']

    # Special handling for 2017 due to AWS CEN reinstallation
    if year == 2017:
        # Get 2016 coordinates for early 2017 data (before 2017-07-25)
        df_2016 = df_aws[(df_aws['aws'] == aws_name) & (df_aws['year'] == 2016)]
        
        if not df_2016.empty:
            lat_2016 = df_2016.iloc[0]['lat']
            lon_2016 = df_2016.iloc[0]['lon']
            point_2016 = ee.Geometry.Point([lon_2016, lat_2016])
            Map.addLayer(point_2016, {'color': 'orange'}, f'AWS: {aws_name}_2016_coords')
            
            # Extract data from 2017-01-01 to 2017-07-24 using 2016 coordinates
            df_mod_early, mod_count_early = extract_modis_data(
                "MODIS/061/MOD10A1", 'Snow_Albedo_Daily_Tile', 'albedo_mod10',
                point_2016, aws_name, year, '2017-01-01', '2017-07-25'
            )
            if df_mod_early is not None:
                df_mod_early.to_csv(mod10_output_path, mode='a', header=False, index=False)
                print(f"Processed MOD10 for {aws_name} 2017-01-01 to 2017-07-24 (using 2016 coords): {mod_count_early} records")
            else:
                print(f"Skipping MOD10 for {aws_name} 2017-01-01 to 2017-07-24: collection is empty")
            
            df_myd_early, myd_count_early = extract_modis_data(
                "MODIS/061/MYD10A1", 'Snow_Albedo_Daily_Tile', 'albedo_myd10',
                point_2016, aws_name, year, '2017-01-01', '2017-07-25'
            )
            if df_myd_early is not None:
                df_myd_early.to_csv(myd10_output_path, mode='a', header=False, index=False)
                print(f"Processed MYD10 for {aws_name} 2017-01-01 to 2017-07-24 (using 2016 coords): {myd_count_early} records")
            else:
                print(f"Skipping MYD10 for {aws_name} 2017-01-01 to 2017-07-24: collection is empty")
        else:
            print(f"Warning: No 2016 data found for {aws_name}, skipping early 2017 extraction")
        
        # Extract data from 2017-07-25 onward using 2017 coordinates
        point_2017 = ee.Geometry.Point([lon, lat])
        Map.addLayer(point_2017, {'color': 'red'}, f'AWS: {aws_name}_2017_coords')
        
        df_mod_late, mod_count_late = extract_modis_data(
            "MODIS/061/MOD10A1", 'Snow_Albedo_Daily_Tile', 'albedo_mod10',
            point_2017, aws_name, year, '2017-07-25', '2018-01-01'
        )
        if df_mod_late is not None:
            df_mod_late.to_csv(mod10_output_path, mode='a', header=False, index=False)
            print(f"Processed MOD10 for {aws_name} 2017-07-25 to 2017-12-31 (using 2017 coords): {mod_count_late} records")
        else:
            print(f"Skipping MOD10 for {aws_name} 2017-07-25 to 2017-12-31: collection is empty")
        
        df_myd_late, myd_count_late = extract_modis_data(
            "MODIS/061/MYD10A1", 'Snow_Albedo_Daily_Tile', 'albedo_myd10',
            point_2017, aws_name, year, '2017-07-25', '2018-01-01'
        )
        if df_myd_late is not None:
            df_myd_late.to_csv(myd10_output_path, mode='a', header=False, index=False)
            print(f"Processed MYD10 for {aws_name} 2017-07-25 to 2017-12-31 (using 2017 coords): {myd_count_late} records")
        else:
            print(f"Skipping MYD10 for {aws_name} 2017-07-25 to 2017-12-31: collection is empty")
        
        print(f"Completed AWS site {i+1}/{len(df_aws)}: {aws_name} for year {year}")
    
    else:
        # Standard processing for non-2017 years
        point = ee.Geometry.Point([lon, lat])
        Map.addLayer(point, {'color': 'red'}, f'AWS: {aws_name}_{year}')

        mod10_collection = (ee.ImageCollection("MODIS/061/MOD10A1")
                            .filterDate(f'{year}-01-01', f'{year+1}-01-01')
                            .select('Snow_Albedo_Daily_Tile')) 
        myd10_collection = (ee.ImageCollection("MODIS/061/MYD10A1")
                            .filterDate(f'{year}-01-01', f'{year+1}-01-01')
                            .select('Snow_Albedo_Daily_Tile'))

        # Check if MOD10 collection is empty
        mod10_size = mod10_collection.size().getInfo()
        if mod10_size > 0:
            # Get the data for the point
            point_data = mod10_collection.getRegion(point, 500, crs='EPSG:3413').getInfo()
            df_mod = ee_array_to_df(point_data, ['Snow_Albedo_Daily_Tile'])
            df_mod['aws'] = aws_name
            
            # rename columns and ensure order of columns matches output csv
            df_mod.rename(columns={'Snow_Albedo_Daily_Tile': 'albedo_mod10'}, inplace=True)
            df_mod = df_mod[['aws', 'time', 'albedo_mod10']]
            df_mod['time'] = pd.to_datetime(df_mod['time'], unit='ms')
            df_mod['albedo_mod10'] = df_mod['albedo_mod10'] / 100
            
            # Append to CSV
            df_mod.to_csv(mod10_output_path, mode='a', header=False, index=False)
            print(f"Processed MOD10 for {aws_name} year {year}: {len(df_mod)} records")
        else:
            print(f"Skipping MOD10 for {aws_name} year {year}: collection is empty")

        # Check if MYD10 collection is empty
        myd10_size = myd10_collection.size().getInfo()
        if myd10_size > 0:
            point_data = myd10_collection.getRegion(point, 500, crs='EPSG:3413').getInfo()
            df_myd = ee_array_to_df(point_data, ['Snow_Albedo_Daily_Tile'])
            df_myd['aws'] = aws_name
            
            df_myd.rename(columns={'Snow_Albedo_Daily_Tile': 'albedo_myd10'}, inplace=True)
            df_myd = df_myd[['aws', 'time', 'albedo_myd10']]
            df_myd['time'] = pd.to_datetime(df_myd['time'], unit='ms')
            df_myd['albedo_myd10'] = df_myd['albedo_myd10'] / 100
            
            # Append to CSV
            df_myd.to_csv(myd10_output_path, mode='a', header=False, index=False)
            print(f"Processed MYD10 for {aws_name} year {year}: {len(df_myd)} records")
        else:
            print(f"Skipping MYD10 for {aws_name} year {year}: collection is empty")

        print(f"Completed AWS site {i+1}/{len(df_aws)}: {aws_name} for year {year}")