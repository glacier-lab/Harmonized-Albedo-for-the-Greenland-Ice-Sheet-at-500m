"""
Build daily mosaics of GCOM-C albedo images.

This script processes GCOM-C albedo data in NetCDF format to create daily 
orthomosaics that match the extent of a Greenland ice sheet mask (EPSG:3413).
The output is resampled to 500m resolution and saved as daily GeoTIFF files.

Author: Shunan Feng (shunan.feng@envs.au.dk)
"""
#%%
import os
import glob
import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling
from pyproj import Transformer, CRS
import pandas as pd
from datetime import datetime
import h5py
import re
from rasterio.plot import show
from scipy.interpolate import interpn, griddata
#%%
def read_mask(mask_path):
    """
    Read the Greenland ice sheet mask from a TIF file.
    
    Parameters:
    -----------
    mask_path : str
        Path to the mask TIF file (should be in EPSG:3413)
    
    Returns:
    --------
    tuple: (mask_array, transform, crs, bounds, shape)
    """
    with rio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        shape = mask.shape

    # Crop to valid extent by removing rows and columns with just no data
    valid_rows = np.where(np.any(mask != src.nodata, axis=1))[0]
    valid_cols = np.where(np.any(mask != src.nodata, axis=0))[0]
    mask = mask[valid_rows[0]:valid_rows[-1]+1, valid_cols[0]:valid_cols[-1]+1]
    transform = from_bounds(
        bounds.left + valid_cols[0] * transform.a,
        bounds.bottom + valid_rows[0] * transform.e,
        bounds.left + (valid_cols[-1]+1) * transform.a,
        bounds.bottom + (valid_rows[-1]+1) * transform.e,
        mask.shape[1],  mask.shape[0]
    )
    shape = mask.shape
        
    print(f"Mask loaded:")
    print(f"  Shape: {shape}")
    print(f"  CRS: {crs}")
    print(f"  Bounds: {bounds}")
    print(f"  Resolution: {transform.a}m x {-transform.e}m")
    
    return mask, transform, crs, bounds, shape
# %%
im_mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"
immask, mask_transform, mask_crs, mask_bounds, mask_shape = read_mask(im_mask_path)
# create x, y grid for the mask
num_rows, num_cols = immask.shape
x_coords = np.linspace(mask_bounds.left + mask_transform.a / 2,
                       mask_bounds.right - mask_transform.a / 2,
                       num_cols)
y_coords = np.linspace(mask_bounds.top - mask_transform.e / 2,
                       mask_bounds.bottom + mask_transform.e / 2,
                       num_rows)
x_grid_mask, y_grid_mask = np.meshgrid(x_coords, y_coords)
# create empty mosaic array
mosaic_data = np.full(immask.shape, np.nan, dtype=np.float32)

# %%
imfolder = '/data_3/shunan_2/AU/hsa500m/GCOMC/'
imnewfolder = '/data_3/shunan_2/AU/hsa500m/GCOMC_mosaics/'
imfiles = sorted(glob.glob(os.path.join(imfolder, '**/*Q_3000.h5'), recursive=True))
imdates = [os.path.basename(f).split('_')[1] for f in imfiles]
imdates = [d[:8] for d in imdates]
imdates = pd.to_datetime(imdates, format='%Y%m%d')
df_files = pd.DataFrame({'filepath': imfiles, 'date': imdates})
unique_dates = np.unique(df_files['date'])

# %%
# %%
testfile = imfiles[3002]
with h5py.File(testfile, 'r') as f:
    print("Root Keys: %s" % list(f.keys()))
    
    # Check Image_data contents
    print("\nImage_data contents:")
    for key in f['Image_data'].keys():
        ds = f['Image_data'][key]
        print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")
        print(f"    Attributes: {list(ds.attrs.keys())}")
    
    # Check Geometry_data attributes
    print("\nGeometry_data attributes:")
    print(f"  {list(f['Geometry_data'].attrs.keys())}")
    
    # Check root attributes
    print("\nRoot attributes:")
    print(f"  {list(f.attrs.keys())}")
    
    # Check Global_attributes
    print("\nGlobal_attributes contents:")
    for key in f['Global_attributes'].keys():
        print(f"  {key}: {f['Global_attributes'][key][()]}")
    
    # Recursively print the entire structure
    def print_structure(name, obj):
        print(f"{name}: {type(obj)}")
    
    f.visititems(print_structure)
    
    # Try to access the data correctly
    if 'Geometry_data' in f:
        geom = f['Geometry_data']
        print(f"\nGeometry_data type: {type(geom)}")
        if isinstance(geom, h5py.Dataset):
            print(f"Geometry_data shape: {geom.shape}")
            print(f"Geometry_data dtype: {geom.dtype}")
        elif isinstance(geom, h5py.Group):
            print(f"Geometry_data subkeys: {list(geom.keys())}")
    qa_flag = f['/Image_data/QA_flag'][:]
    print(f"\nQA_Flag shape: {qa_flag.shape}, dtype: {qa_flag.dtype}")
    im_albedo = f['/Image_data/SALB'][:]
    print(f"Albedo shape: {im_albedo.shape}, dtype: {im_albedo.dtype}")

    # get qa_flag attributes
    qa_flag_attrs = f['/Image_data/QA_flag'].attrs
    print(f"\nQA_Flag attributes: {list(qa_flag_attrs.keys())}")
    for attr_key in qa_flag_attrs.keys():
        print(f"  {attr_key}: {qa_flag_attrs[attr_key]}")
    # get image data attributes
    im_albedo_attrs = f['/Image_data/SALB'].attrs
    print(f"\nAlbedo attributes: {list(im_albedo_attrs.keys())}")
    for attr_key in im_albedo_attrs.keys():
        print(f"  {attr_key}: {im_albedo_attrs[attr_key]}")
    slope = im_albedo_attrs['Slope']
    intercept = im_albedo_attrs['Offset']
    im_albedo = np.double(im_albedo) * slope + intercept
    # create mapx, mapy
    geometry_attrs = f['/Geometry_data'].attrs
    print(f"\nGeometry_data attributes: {list(geometry_attrs.keys())}")
    for attr_key in geometry_attrs.keys():
        print(f"  {attr_key}: {geometry_attrs[attr_key]}")
    grid_interval = geometry_attrs['Grid_interval']
    lower_left_lat = float(geometry_attrs['Lower_left_latitude'][0])
    lower_left_lon = float(geometry_attrs['Lower_left_longitude'][0])
    upper_right_lat = float(geometry_attrs['Upper_right_latitude'][0])
    upper_right_lon = float(geometry_attrs['Upper_right_longitude'][0])
    num_rows = int(geometry_attrs['Number_of_lines'][0])
    num_cols = int(geometry_attrs['Number_of_pixels'][0])
    latitudes = np.linspace(lower_left_lat, upper_right_lat, num_rows)
    longitudes = np.linspace(lower_left_lon, upper_right_lon, num_cols)
    gcom_crs = CRS.from_epsg(4326)  # WGS84
    crs_transformer = Transformer.from_crs(gcom_crs, mask_crs, always_xy=True)
    # lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    # x_grid_gcomc, y_grid_gcomc = crs_transformer.transform(lon_grid, lat_grid)
    x_gcomc, y_gcomc = crs_transformer.transform(longitudes, latitudes)
    # print(f"\nGenerated x_grid and y_grid with shapes: {x_grid_gcomc.shape}, {y_grid_gcomc.shape}")
    

#%% Create proper coordinate grids
# GCOM-C is regular in lat/lon space, but NOT regular in EPSG:3413
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
x_grid_gcomc, y_grid_gcomc = crs_transformer.transform(lon_grid, lat_grid)
print(f"Generated x_grid and y_grid with shapes: {x_grid_gcomc.shape}, {y_grid_gcomc.shape}")

#%% Interpolate albedo to the mask grid using griddata
# Interpolate from GCOM-C grid to mask grid
mosaic_data = griddata(
    points=(x_grid_gcomc.flatten(), y_grid_gcomc.flatten()),
    values=im_albedo.flatten(),
    xi=(x_grid_mask.flatten(), y_grid_mask.flatten()),
    method='linear',
    fill_value=np.nan
).reshape(x_grid_mask.shape)

print(f"Mosaic shape: {mosaic_data.shape}")
print(f"Valid mosaic pixels: {np.sum(~np.isnan(mosaic_data))}")

show(mosaic_data)

#%%
points = (y_grid_gcomc[:,0], x_grid_gcomc[0,:])  # note the order: (y, x)
mosaic_data = interpn(
    points,
    im_albedo,
    (y_grid_mask, x_grid_mask),
    method='linear',
    bounds_error=False,
    fill_value=np.nan
)#.reshape(x_grid_mask.shape)
# %%            
# f.keys()
# f.close()
#%%
# f.visit(print) 
# group = f['Geometry_data']
# # for key in group.keys():
# #     print(key)
# im = group['/Image_data'][:]
# for imdate in unique_dates:
#     daily_files = df_files[df_files['date'] == imdate]['filepath'].tolist()
#     print(f"Processing date: {imdate.strftime('%Y-%m-%d')} with {len(daily_files)} files.")
    
#     mosaic_data = np.full(immask.shape, np.nan, dtype=np.float32)
#     mosaic_count = np.zeros(immask.shape, dtype=np.int16)
    
#     for file in daily_files:
#         print(f"  Reading file: {file}")
#         with h5py.File(file, 'r') as f:
#             lat = f['/Geolocation/Latitude'][:]
#             lon = f['/Geolocation/Longitude'][:]
#             albedo = f['/Data Fields/Albedo'][:]
        
#         # Transform lat/lon to EPSG:3413
#         transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
#         x, y = transformer.transform(lon.flatten(), lat.flatten())
        
#         # Map to pixel coordinates in the mask
#         col = ((x - mask_bounds.left) / mask_transform.a).astype(int)
#         row = ((y - mask_bounds.top) / mask_transform.e).astype(int)
        
#         valid_mask = (
#             (row >= 0) & (row < immask.shape[0]) &
#             (col >= 0) & (col < immask.shape[1]) &
#             (~np.isnan(albedo.flatten()))
#         )
        
#         for r, c, a in zip(row[valid_mask], col[valid_mask], albedo.flatten()[valid_mask]):
#             if np.isnan(mosaic_data[r, c]):
#                 mosaic_data[r, c] = a
#             else:
#                 mosaic_data[r, c] += a
#             mosaic_count[r, c] += 1
    
#     # Average the mosaicked data
#     valid_pixels = mosaic_count > 0
#     mosaic_data[valid_pixels] /= mosaic_count[valid_pixels]
    
#     # Save the daily mosaic as GeoTIFF
#     output_path = f"{imnewfolder}/GCOMC_Albedo_{imdate.strftime('%Y%m%d')}.tif"
#     with rio.open(
#         output_path,
#         'w',
#         driver='GTiff',
#         height=immask.shape[0],
#         width=immask.shape[1],
#         count=1,
#         dtype=mosaic_data.dtype,
#         crs=mask_crs,
#         transform=mask_transform,
#         nodata=np.nan
#     ) as dst:
#         dst.write(mosaic_data, 1)
# %%
