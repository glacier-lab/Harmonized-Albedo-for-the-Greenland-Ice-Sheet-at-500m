#%%
import os
import glob
import numpy as np
import xarray as xr
import rasterio as rio
from pathlib import Path
from affine import Affine
from pyproj import CRS
from rasterio.warp import reproject, Resampling

#%%
def read_mask(mask_path):
    with rio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

        valid_rows = np.where(np.any(mask != src.nodata, axis=1))[0]
        valid_cols = np.where(np.any(mask != src.nodata, axis=0))[0]

        row_start, row_end = valid_rows[0], valid_rows[-1] + 1
        col_start, col_end = valid_cols[0], valid_cols[-1] + 1

        mask_cropped = mask[row_start:row_end, col_start:col_end]
        new_transform = rio.windows.transform(
            rio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start),
            transform,
        )
        mask_cropped = np.where(mask_cropped <= 0, 0, 1)

    return mask_cropped, new_transform, crs, mask_cropped.shape


# def get_xy_names(da):
#     x_candidates = ["x", "projection_x_coordinate", "longitude"]
#     y_candidates = ["y", "projection_y_coordinate", "latitude"]

#     x_name = next((name for name in x_candidates if name in da.coords), None)
#     y_name = next((name for name in y_candidates if name in da.coords), None)

#     if x_name is None or y_name is None:
#         raise ValueError(f"Could not identify x/y coordinates. Found coords: {list(da.coords)}")

#     return x_name, y_name


# def get_source_transform(ds, data_var):
#     # Prefer real projected 1D axes
#     axis_pairs = [
#         ("x", "y"),
#         ("projection_x_coordinate", "projection_y_coordinate"),
#     ]

#     for x_name, y_name in axis_pairs:
#         if x_name in ds.variables and y_name in ds.variables:
#             x = ds[x_name].values
#             y = ds[y_name].values

#             if x.ndim == 1 and y.ndim == 1:
#                 dx = float(x[1] - x[0])
#                 dy = float(y[1] - y[0])

#                 west = float(x[0] - dx / 2.0)
#                 north = float(y[0] - dy / 2.0)

#                 return Affine(dx, 0.0, west, 0.0, dy, north)

#     # Fallback: read GDAL-style GeoTransform from the grid-mapping variable
#     grid_mapping_name = ds[data_var].attrs.get("grid_mapping")
#     if grid_mapping_name and grid_mapping_name in ds.variables:
#         attrs = ds[grid_mapping_name].attrs
#         geotransform = attrs.get("GeoTransform")

#         if geotransform is not None:
#             if isinstance(geotransform, str):
#                 gt = [float(v) for v in geotransform.split()]
#             else:
#                 gt = [float(v) for v in geotransform]

#             return Affine.from_gdal(*gt)

#     raise ValueError(
#         "Could not derive source transform from 1D projected axes or GeoTransform. "
#         "Do not use 2D latitude/longitude arrays for Affine transforms."
#     )


# def get_source_crs(ds, data_var):
#     grid_mapping_name = ds[data_var].attrs.get("grid_mapping")

#     if grid_mapping_name and grid_mapping_name in ds.variables:
#         attrs = ds[grid_mapping_name].attrs

#         try:
#             return CRS.from_cf(attrs)
#         except Exception:
#             wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
#             if wkt:
#                 return CRS.from_wkt(wkt)

#     raise ValueError("Could not determine source CRS from the CARRA grid-mapping metadata.")


def get_source_geoloc(ds):
    if "longitude" not in ds or "latitude" not in ds:
        raise ValueError(
            f"Dataset must contain 2D latitude/longitude arrays. Found: {list(ds.variables)}"
        )

    lon = ds["longitude"].values
    lat = ds["latitude"].values

    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError(
            f"Expected 2D longitude/latitude arrays, got lon.ndim={lon.ndim}, lat.ndim={lat.ndim}"
        )

    # Normalize longitudes if needed
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    return lon.astype(np.float64), lat.astype(np.float64)

#%%
carra_dir = "/data_3/shunan_2/AU/hsa500m/Yukihiko"
mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"
output_dir = "/data_3/shunan_2/AU/hsa500m/CARRA_diffuse_fraction"
Path(output_dir).mkdir(parents=True, exist_ok=True)

immask, mask_transform, mask_crs, mask_shape = read_mask(mask_path)

imfiles = sorted(glob.glob(f"{carra_dir}/carra_*.nc"))

for filepath in imfiles:
    fname = Path(filepath).stem
    print(f"Processing {fname}...")

    with xr.open_dataset(filepath) as ds:
        ssrd_var = "ssrd"
        direct_var = "tidirswrf"

        # ds_daily = ds.where(ds.valid_time.dt.hour == 0, drop=True)
        ds_daily = ds.sel(valid_time=(ds.valid_time.dt.hour == 0))
        ds_daily = ds_daily.assign_coords(
            valid_time=ds_daily.valid_time - np.timedelta64(1, "D")
        )

        ssrd = ds_daily[ssrd_var]
        direct = ds_daily[direct_var]

        with np.errstate(invalid="ignore", divide="ignore"):
            f = (ssrd - direct) / ssrd

        f = f.where(ssrd > 0)
        f = f.clip(0.0, 1.0)
        f.name = "diffuse_fraction"

        lon2d, lat2d = get_source_geoloc(ds_daily)

        for t in f.valid_time.values:
            day = np.datetime_as_string(t, unit="D").replace("-", "")
            arr = f.sel(valid_time=t).values.astype(np.float32)

            reprojected = np.full(mask_shape, np.nan, dtype=np.float32)

            reproject(
                source=arr,
                destination=reprojected,
                src_geoloc_array=(lon2d, lat2d),
                src_crs=CRS.from_epsg(4326),
                dst_transform=mask_transform,
                dst_crs=mask_crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

            reprojected[immask == 0] = np.nan

            out_path = os.path.join(output_dir, f"CARRA_diffuse_fraction_{day}_500m.tif")
            with rio.open(
                out_path,
                "w",
                driver="GTiff",
                height=mask_shape[0],
                width=mask_shape[1],
                count=1,
                dtype=np.float32,
                crs=mask_crs,
                transform=mask_transform,
                nodata=np.nan,
                compress="lzw",
            ) as dst:
                dst.write(reprojected, 1)
                dst.set_band_description(1, "diffuse_fraction")

            print(f"  Saved: {os.path.basename(out_path)}")

print("Processing complete.")