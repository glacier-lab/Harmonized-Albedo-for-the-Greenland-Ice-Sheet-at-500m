"""
Convert collaborator-provided CARRA albedo data to masked GeoTIFF mosaics.

This script processes custom CARRA NetCDF files (already downscaled to 500m),
reprojects them onto the PROMICE mask grid (same workflow as GCOM-C SR), applies
ice-mask filtering, and exports one GeoTIFF per input file.

Key operations:
- Reads albedo variable al from NetCDF
- Reprojects to PROMICE mask grid using rasterio.warp.reproject
- Masks non-ice areas using PROMICE-2022IceMask
- Writes compressed float32 GeoTIFF outputs
- Runs in parallel with ProcessPoolExecutor

Author: Shunan Feng (shunan.feng@envs.au.dk)
"""

# %%
import os
import re
import glob
import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import CRS, Transformer
from tqdm import tqdm
from affine import Affine
from concurrent.futures import ProcessPoolExecutor, as_completed


# %%
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


# %%
def parse_output_date_tag(file_path):
    """Parse al_ori_YYYY_dDDD filename and convert to YYYYMMDD."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    m = re.fullmatch(r"al_ori_(\d{4})_d(\d{3})", stem)
    if not m:
        raise ValueError(
            f"Unexpected filename format: {os.path.basename(file_path)}; "
            "expected al_ori_YYYY_dDDD.nc"
        )

    year = int(m.group(1))
    doy = int(m.group(2))
    if not 1 <= doy <= 366:
        raise ValueError(f"Invalid day-of-year in filename: {os.path.basename(file_path)}")

    import datetime as dt
    return (dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)).strftime("%Y%m%d")



def process_single_file(file_path, out_folder, immask, mask_transform, mask_crs, mask_shape):
    """Process one CARRA file and write one masked/reprojected GeoTIFF.

    Supports two file formats:
    - Old format (pre-2025): coords 'Y' (lat) / 'X' (lon) in WGS84, requires
      coordinate transform to EPSG:3413.
    - New format (2025+): coords 'y' / 'x' already in EPSG:3413, with an
      embedded 'spatial_ref' variable carrying the CRS and GeoTransform.
    """
    filename = os.path.basename(file_path)

    try:
        date_tag = parse_output_date_tag(file_path)
        out_path = os.path.join(out_folder, f"CARRA_Albedo_{date_tag}_500m.tif")

        with xr.open_dataset(file_path) as ds:
            data = np.squeeze(ds['al'].values) / 100.0  # Convert % -> fraction

            if 'x' in ds.coords and 'y' in ds.coords:
                # --- New 2025+ format: already in EPSG:3413 ---
                src_crs = CRS.from_wkt(ds['spatial_ref'].attrs['crs_wkt'])
                gt = [float(v) for v in ds['spatial_ref'].attrs['GeoTransform'].split()]
                # GeoTransform: [upper_left_x, x_res, 0, upper_left_y, 0, -y_res]
                src_transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
            else:
                # --- Old format (pre-2025): Y=lat, X=lon in WGS84 ---
                lat = ds['Y'].values
                lon = ds['X'].values
                crs_wgs84 = CRS.from_epsg(4326)
                src_crs = CRS.from_epsg(3413)
                transformer = Transformer.from_crs(crs_wgs84, src_crs, always_xy=True)
                x_proj, y_proj = transformer.transform(lon, lat)
                src_transform = from_bounds(
                    x_proj.min(), y_proj.min(), x_proj.max(), y_proj.max(),
                    data.shape[1], data.shape[0],
                )

        data[(data <= 0) | (data >= 1)] = np.nan

        dst = np.empty(mask_shape, dtype=np.float32)
        reproject(
            source=data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=mask_transform,
            dst_crs=mask_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        dst[immask == 0] = np.nan

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
        ) as out_ds:
            out_ds.write(dst, 1)
            out_ds.update_tags(SOURCE_FILE=filename, SOURCE_VAR="al")

        return filename, True, None

    except Exception as e:
        return filename, False, str(e)


# %%
# --- Configuration ---
im_mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"
# Set to the root folder; all subdirectories will be searched recursively.
carra_input_folder = "/data_3/shunan_2/AU/hsa500m/CARRA/GL500m"
out_folder = "/data_3/shunan_2/AU/hsa500m/CARRA_GL500m_geotiff"

NUM_WORKERS = 10
os.makedirs(out_folder, exist_ok=True)


# %%
# --- Load mask and list files ---
immask, mask_transform, mask_crs, mask_shape = read_mask(im_mask_path)
carra_files = sorted(glob.glob(os.path.join(carra_input_folder, "**/*.nc"), recursive=True))

print(f"Found {len(carra_files)} CARRA NetCDF files")
if len(carra_files) == 0:
    raise SystemExit("No CARRA files found. Check carra_input_folder path.")


# %%
# --- Parallel processing loop ---
print(f"Processing {len(carra_files)} files with {NUM_WORKERS} workers...")

processed = 0
failed = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(
            process_single_file,
            file_path,
            out_folder,
            immask,
            mask_transform,
            mask_crs,
            mask_shape,
        ): file_path
        for file_path in carra_files
    }

    with tqdm(total=len(carra_files), desc="Processing Files") as pbar:
        for future in as_completed(futures):
            filename, success, msg = future.result()

            if success:
                processed += 1
            else:
                failed += 1
                tqdm.write(f"Error processing {filename}: {msg}")

            pbar.update(1)


print("\nProcessing Summary")
print("=" * 60)
print(f"Total files:             {len(carra_files):>5}")
print(f"Successfully processed:  {processed:>5}")
print(f"Failed:                  {failed:>5}")
print("=" * 60)
