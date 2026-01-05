#%%
import geemap
import ee
import pandas as pd
# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)
mod10_output_path = '/data_3/shunan_2/AU/hsa500m/MODIS/albedo_mod10.csv'
myd10_output_path = '/data_3/shunan_2/AU/hsa500m/MODIS/albedo_myd10.csv'

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
#%%
for i in range(len(df_aws)):
    lat = df_aws.loc[i, 'lat']
    lon = df_aws.loc[i, 'lon']
    aws_name = df_aws.loc[i, 'aws']
    year = df_aws.loc[i, 'year']

    point = ee.Geometry.Point([lon, lat])
    Map.addLayer(point, {'color': 'red'}, f'AWS: {aws_name}_{year}')

    mod10_collection = (ee.ImageCollection("MODIS/061/MOD10A1")
                        .filterDate(f'{year}-01-01', f'{year}-12-31')
                        .select('Snow_Albedo_Daily_Tile')) 
    myd10_collection = (ee.ImageCollection("MODIS/061/MYD10A1")
                        .filterDate(f'{year}-01-01', f'{year}-12-31')
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
        
        # Append to CSV
        df_myd.to_csv(myd10_output_path, mode='a', header=False, index=False)
        print(f"Processed MYD10 for {aws_name} year {year}: {len(df_myd)} records")
    else:
        print(f"Skipping MYD10 for {aws_name} year {year}: collection is empty")

    print(f"Completed AWS site {i+1}/{len(df_aws)}: {aws_name} for year {year}")