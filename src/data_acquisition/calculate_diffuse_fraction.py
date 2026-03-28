#%%
import os
import glob
import numpy as np
import xarray as xr
import rasterio as rio
from pathlib import Path
from affine import Affine
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyproj import CRS
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

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
        new_transform = transform * Affine.translation(col_start, row_start)
        mask_cropped = np.where(mask_cropped <= 0, 0, 1)

    return mask_cropped, new_transform, crs, mask_cropped.shape


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
NUM_WORKERS = 6
Path(output_dir).mkdir(parents=True, exist_ok=True)

immask, mask_transform, mask_crs, mask_shape = read_mask(mask_path)

imfiles = sorted(glob.glob(f"{carra_dir}/carra_*.nc"))


def process_single_file(filepath):
    fname = Path(filepath).stem
    saved_count = 0
    errors = []

    try:
        with xr.open_dataset(filepath) as ds:
            ssrd_var = "ssrd"
            direct_var = "tidirswrf"

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

                saved_count += 1

    except Exception as e:
        errors.append(f"{fname}: {e}")

    return fname, saved_count, errors

print(f"Processing {len(imfiles)} files with {NUM_WORKERS} workers...")
total_saved = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_single_file, filepath): filepath for filepath in imfiles}

    with tqdm(total=len(futures), desc="Processing files") as pbar:
        for future in as_completed(futures):
            fname, saved_count, errors = future.result()
            total_saved += saved_count

            if errors:
                for err in errors:
                    tqdm.write(f"Error: {err}")
            else:
                tqdm.write(f"Finished {fname} ({saved_count} outputs)")

            pbar.update(1)

print(f"Processing complete. Wrote {total_saved} files.")