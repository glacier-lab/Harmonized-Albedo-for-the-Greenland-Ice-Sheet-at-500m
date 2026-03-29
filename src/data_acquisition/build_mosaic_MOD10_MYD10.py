"""
Harmonize MOD10/MYD10 GeoTIFF albedo to PROMICE mask grid.

This script expects MOD10 and MYD10 albedo GeoTIFF files already in EPSG:3413,
but potentially with different extents and native masks. Each input file is
warped to the PROMICE ice-mask grid so both products share identical extent,
resolution, and georeferencing. Non-ice pixels are masked out.

Key operations:
- Reads MOD10/MYD10 GeoTIFF files
- Rescales uint8 albedo from 0-100 to 0-1
- Reprojects/resamples each image onto PROMICE mask grid
- Applies PROMICE ice mask (non-ice -> NaN)
- Writes one output GeoTIFF per input file
- Runs in parallel with ProcessPoolExecutor

Author: Shunan Feng (shunan.feng@envs.au.dk)
"""

# %%
import os
import glob
import numpy as np
import rasterio as rio
from affine import Affine
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.warp import reproject, Resampling


# %%
def read_mask(mask_path):
    """Read and crop mask to valid bounds, then convert to binary ice mask."""
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


def process_single_file(file_path, product_name, out_folder, immask, mask_transform, mask_crs, mask_shape):
    """Process one MOD10/MYD10 file and save masked/aligned GeoTIFF."""
    filename = os.path.basename(file_path)

    try:
        out_path = os.path.join(out_folder, filename)

        if os.path.exists(out_path):
            return filename, None, "skipped"

        with rio.open(file_path) as src:
            if src.count < 1:
                raise ValueError("Input GeoTIFF has no raster band")

            src_arr = src.read(1).astype(np.float32)
            src_transform = src.transform
            src_crs = src.crs
            src_nodata = src.nodata

        if src_crs is None:
            raise ValueError("Input CRS is missing")

        # Convert source nodata to NaN before reprojection.
        if src_nodata is not None:
            src_arr[src_arr == src_nodata] = np.nan

        # MOD10/MYD10 albedo is uint8 in 0..100; convert to physical albedo 0..1.
        src_arr = src_arr / 100.0

        # Keep only physical albedo range after scaling.
        src_arr[(src_arr <= 0) | (src_arr >= 1)] = np.nan

        dst = np.full(mask_shape, np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=mask_transform,
            dst_crs=mask_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        # Remove non-ice area.
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
            out_ds.set_band_description(1, "Snow_Albedo")
            out_ds.update_tags(SOURCE_FILE=filename, PRODUCT=product_name)

        return filename, True, None

    except Exception as e:
        return filename, False, str(e)


def process_collection(product_name, input_folder, out_folder, immask, mask_transform, mask_crs, mask_shape, num_workers):
    """Run parallel processing for one product collection (MOD10 or MYD10)."""
    files = sorted(glob.glob(os.path.join(input_folder, "**/*.tif"), recursive=True))
    print(f"[{product_name}] Found {len(files)} input files")

    if len(files) == 0:
        print(f"[{product_name}] No files found, skipping.")
        return

    processed = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                file_path,
                product_name,
                out_folder,
                immask,
                mask_transform,
                mask_crs,
                mask_shape,
            ): file_path
            for file_path in files
        }

        with tqdm(total=len(files), desc=f"{product_name}") as pbar:
            for future in as_completed(futures):
                filename, success, msg = future.result()

                if success is None:
                    skipped += 1
                elif success:
                    processed += 1
                else:
                    failed += 1
                    tqdm.write(f"[{product_name}] Error processing {filename}: {msg}")

                pbar.update(1)

    print(f"\n[{product_name}] Summary")
    print("=" * 60)
    print(f"Total files:             {len(files):>5}")
    print(f"Successfully processed:  {processed:>5}")
    print(f"Skipped (existing):      {skipped:>5}")
    print(f"Failed:                  {failed:>5}")
    print("=" * 60)


# %%
# --- Configuration ---
im_mask_path = "/data_3/shunan_2/AU/hsa500m/PROMICE-2022IceMask.tif"

# Input folders with daily MOD10/MYD10 albedo GeoTIFF files.
mod10_input_folder = "/data_3/shunan_2/AU/hsa500m/MODIS/MOD10"
myd10_input_folder = "/data_3/shunan_2/AU/hsa500m/MODIS/MYD10"

# Output folders (same extent/resolution as PROMICE mask, non-ice masked).
mod10_out_folder = "/data_3/shunan_2/AU/hsa500m/MOD10A1_cropped"
myd10_out_folder = "/data_3/shunan_2/AU/hsa500m/MYD10A1_cropped"

NUM_WORKERS = 10

os.makedirs(mod10_out_folder, exist_ok=True)
os.makedirs(myd10_out_folder, exist_ok=True)


# %%
if __name__ == "__main__":
    immask, mask_transform, mask_crs, mask_shape = read_mask(im_mask_path)

    process_collection(
        product_name="MOD10A1",
        input_folder=mod10_input_folder,
        out_folder=mod10_out_folder,
        immask=immask,
        mask_transform=mask_transform,
        mask_crs=mask_crs,
        mask_shape=mask_shape,
        num_workers=NUM_WORKERS,
    )

    process_collection(
        product_name="MYD10A1",
        input_folder=myd10_input_folder,
        out_folder=myd10_out_folder,
        immask=immask,
        mask_transform=mask_transform,
        mask_crs=mask_crs,
        mask_shape=mask_shape,
        num_workers=NUM_WORKERS,
    )
