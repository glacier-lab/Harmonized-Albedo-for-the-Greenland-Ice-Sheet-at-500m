"""
Build daily mosaics of MCD43A3 albedo images over the Arctic or Antarctica.

This script processes MCD43A3 albedo data in HDF4 format and creates daily
mosaics on the selected ice mask grid: EPSG:3413 or EPSG:3031.

Key operations:
- Reads MCD43A3 albedo from HDF4 files via GDAL subdataset interface
- QA masking is disabled, as in the original workflow
- Applies scale factor calibration
- Reprojects from sinusoidal to polar stereographic projection
- Handles overlapping pixels by averaging
- Masks to the selected polar ice extent
- Exports daily composites as GeoTIFF files
- Saves both BSA (band1) and WSA (band2) shortwave albedo in separate bands of the same GeoTIFF
- Saves albedo multiplied by 10000 as uint16 (scale 0.0001, nodata 65535)
- Uses the output root as supplied

Shunan Feng (shunan.feng@envs.au.dk)
"""
#%%
import argparse
import os
import glob
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import reproject, Resampling
from pyproj import CRS
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from build_mosaic_VIIRS_SR import OUTPUT_NODATA, REGIONS, read_mask

# %%
def read_hdf4_subdataset(hdf_path, subdataset_name):
    """
    Read a subdataset from an HDF4 EOS file.
    Returns (data array, scale_factor, add_offset, nodata, transform, crs_wkt).
    """
    subdataset_path = f'HDF4_EOS:EOS_GRID:"{hdf_path}":MOD_Grid_BRDF:{subdataset_name}'
    with rio.open(subdataset_path) as src:
        data = src.read(1).astype(np.float32)
        scale_factor = src.scales[0] if src.scales and src.scales[0] is not None else 0.001
        add_offset = src.offsets[0] if src.offsets and src.offsets[0] is not None else 0.0
        nodata = src.nodata
        transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs is not None else None

    return data, scale_factor, add_offset, nodata, transform, crs_wkt


def apply_qa_mask(albedo_data, qa_flag):
    """
    Apply QA masking to MCD43A3 albedo data.
    
    MCD43A3 QA encoding (BRDF_Albedo_Band_Mandatory_Quality_shortwave):
      Mandatory QA  0 = processed, good quality (full BRDF inversions)
                    1 = processed, see other QA (magnitude BRDF inversions)
                    2 = processed, good quality (full BRDF inversions, only Band 6 is fill value due to non-functional or noisy detectors)
                    3 = processed, see other QA (magnitude BRDF inversions, only Band 6 is fill value due to non-functional or noisy detectors)
                    4 = processed, good quality (full BRDF inversions, only Band 5 is fill value due to non-functional or noisy detectors)
                    5 = processed, see other QA (magnitude BRDF inversions, only Band 5 is fill value due to non-functional or noisy detectors)
                    6 = processed, good quality (full BRDF inversions, both Band5 and Band 6 are fill value due to non-functional or noisy detectors)
                    7 = processed, see other QA (magnitude BRDF inversions, both Band 5 and Band 6 are fill value due to non-functional or noisy detectors)
    
    Keep full BRDF inversion classes, including detector-caveat classes:
    QA in {0, 2, 4, 6}.
    """
    albedo_masked = albedo_data.copy()
    valid_qa = np.isin(qa_flag, [0, 2, 4, 6])
    albedo_masked[~valid_qa] = np.nan
    return albedo_masked


# Subdataset names
BSA_NAME = 'Albedo_BSA_shortwave'
WSA_NAME = 'Albedo_WSA_shortwave'
QA_NAME  = 'BRDF_Albedo_Band_Mandatory_Quality_shortwave'

SCALE_FACTOR = 0.001  # MCD43A3 shortwave albedo scale factor
FILL_VALUE = 32767    # MCD43A3 fill value for 16-bit integer albedo
def process_single_date(date, daily_files, out_folder, immask, mask_transform, mask_crs, mask_shape):
    """Process all files for one date and write one output mosaic."""
    try:
        # Initialize empty daily mosaics for both BSA and WSA albedo
        daily_mosaic_bsa = np.zeros(mask_shape, dtype=np.float32)
        daily_mosaic_wsa = np.zeros(mask_shape, dtype=np.float32)
        daily_count_bsa  = np.zeros(mask_shape, dtype=np.float32)
        daily_count_wsa  = np.zeros(mask_shape, dtype=np.float32)

        for f_path in daily_files:
            try:
                # 1. Read BSA, WSA, and QA subdatasets
                bsa_raw, bsa_scale, bsa_offset, _, src_transform, crs_wkt = read_hdf4_subdataset(f_path, BSA_NAME)
                wsa_raw, wsa_scale, wsa_offset, _, _, _ = read_hdf4_subdataset(f_path, WSA_NAME)
                qa_raw,  _,         _,          _, _, _ = read_hdf4_subdataset(f_path, QA_NAME)

                # 2. Mask fill values before scaling
                bsa_raw[bsa_raw == FILL_VALUE] = np.nan
                wsa_raw[wsa_raw == FILL_VALUE] = np.nan

                # 3. Apply scale factor (MCD43A3 uses scale only, offset=0)
                if bsa_scale != SCALE_FACTOR or wsa_scale != SCALE_FACTOR:
                    tqdm.write(f"Warning: Scale factor mismatch in {os.path.basename(f_path)}. But use default values.")
                if bsa_offset != 0.0 or wsa_offset != 0.0:
                    tqdm.write(f"Warning: Non-zero offset in {os.path.basename(f_path)}. But use default values.")
                bsa_albedo = bsa_raw * SCALE_FACTOR
                wsa_albedo = wsa_raw * SCALE_FACTOR

                # 4. Apply QA mask
                # bsa_albedo = apply_qa_mask(bsa_albedo, qa_raw)
                # wsa_albedo = apply_qa_mask(wsa_albedo, qa_raw)

                # 5. Physical range check
                bsa_albedo[(bsa_albedo <= 0) | (bsa_albedo >= 1)] = np.nan
                wsa_albedo[(wsa_albedo <= 0) | (wsa_albedo >= 1)] = np.nan

                # 6. Get source CRS and transform from GDAL
                src_crs = CRS.from_wkt(crs_wkt) if crs_wkt is not None else None

                # 7. Reproject both albedo types to the mask grid
                temp_bsa = np.full(mask_shape, np.nan, dtype=np.float32)
                temp_wsa = np.full(mask_shape, np.nan, dtype=np.float32)

                reproject(
                    source=bsa_albedo,
                    destination=temp_bsa,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=mask_transform,
                    dst_crs=mask_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan
                )
                reproject(
                    source=wsa_albedo,
                    destination=temp_wsa,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=mask_transform,
                    dst_crs=mask_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan
                )

                # 8. Accumulate valid pixels for daily averaging
                valid_bsa = ~np.isnan(temp_bsa)
                valid_wsa = ~np.isnan(temp_wsa)

                daily_mosaic_bsa[valid_bsa] += temp_bsa[valid_bsa]
                daily_mosaic_wsa[valid_wsa] += temp_wsa[valid_wsa]
                daily_count_bsa[valid_bsa]  += 1
                daily_count_wsa[valid_wsa]  += 1

            except Exception as e:
                return date, False, f"Error processing {os.path.basename(f_path)}: {type(e).__name__}: {e}"

        # Calculate mean for overlapping tiles
        with np.errstate(divide='ignore', invalid='ignore'):
            final_bsa = np.divide(daily_mosaic_bsa, daily_count_bsa,
                                  where=daily_count_bsa > 0,
                                  out=np.full_like(daily_mosaic_bsa, np.nan))
            final_wsa = np.divide(daily_mosaic_wsa, daily_count_wsa,
                                  where=daily_count_wsa > 0,
                                  out=np.full_like(daily_mosaic_wsa, np.nan))

        # Apply selected ice mask
        final_bsa[immask == 0] = np.nan
        final_wsa[immask == 0] = np.nan

        # Save as GeoTIFF with 2 bands
        out_path = os.path.join(out_folder, f"MCD43A3_Albedo_{date}_500m.tif")
        with rio.open(
            out_path, 'w',
            driver='GTiff',
            height=mask_shape[0],
            width=mask_shape[1],
            count=2,
            dtype=np.uint16,
            crs=mask_crs,
            transform=mask_transform,
            nodata=OUTPUT_NODATA,
            compress='lzw'
        ) as dst:
            for band_idx, data in enumerate((final_bsa, final_wsa), 1):
                valid = np.isfinite(data)
                output = np.full(mask_shape, OUTPUT_NODATA, dtype=np.uint16)
                output[valid] = np.rint(data[valid] * 10000).astype(np.uint16)
                dst.write(output, band_idx)
            dst.scales = (0.0001, 0.0001)
            dst.set_band_description(1, 'BSA_shortwave')
            dst.set_band_description(2, 'WSA_shortwave')

        return date, True, None

    except Exception as e:
        return date, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Build polar MCD43A3 albedo mosaics.")
    parser.add_argument("--region", required=True, choices=list(REGIONS))
    parser.add_argument("--input-root", default="/data_3/shunan_2/AU/hsa500m/MODIS/MCD43A3_061",
                        help="Folder containing MCD43A3 HDF files.")
    parser.add_argument("--output-root", default="/data_3/shunan_2/AU/hsa500m/MCD43A3_061_mosaics",
                        help="Output root for mosaics.")
    parser.add_argument("--mask-path", default=None, help="Override the region's ice mask.")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-dates", type=int, default=None)
    args = parser.parse_args()

    region = REGIONS[args.region]
    immask, mask_transform, mask_crs, mask_shape = read_mask(
        args.mask_path or region["mask_path"], region["epsg"], region["ice_values"]
    )
    imfiles = sorted(glob.glob(os.path.join(args.input_root, '**/*.hdf'), recursive=True))
    if not imfiles:
        print(f"[WARN] No MCD43A3 files found in {args.input_root}")
        return
    imdates = [os.path.basename(f).split('.')[1][1:8] for f in imfiles]
    imdates = pd.to_datetime(imdates, format='%Y%j').strftime('%Y%m%d')
    df_files = pd.DataFrame({'filepath': imfiles, 'date': imdates})
    unique_dates = np.unique(df_files['date'])[:args.max_dates]
    out_folder = args.output_root
    os.makedirs(out_folder, exist_ok=True)
    print(f"Processing {len(unique_dates)} dates with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_date,
                date,
                df_files[df_files['date'] == date]['filepath'].tolist(),
                out_folder, immask, mask_transform, mask_crs, mask_shape
            ): date
            for date in unique_dates
        }

        with tqdm(total=len(unique_dates), desc="Processing Days") as pbar:
            for future in as_completed(futures):
                date, success, error_msg = future.result()
                if not success:
                    tqdm.write(f"Error processing {date}: {error_msg}")
                pbar.update(1)

    print("Processing Complete.")


if __name__ == '__main__':
    main()

#%%
# Test code (uncomment to debug)
# testfile = imfiles[0]
# with h5py.File(testfile, 'r') as f:
#     inspect_h5_structure(testfile)
# %%
# testfile = imfiles[11]
# ds = gdal.Open(testfile)
# print(ds.GetSubDatasets())
#     im_metadata = f['HDFEOS INFORMATION']['StructMetadata.0'][()].decode('utf-8')
#     # Parse metadata string into key-value pairs
#     bs_albedo = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['Albedo_BSA_shortwave']
#     ws_albedo = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['Albedo_WSA_shortwave']
#     qa_flag = f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields']['BRDF_Albedo_Band_Mandatory_Quality_shortwave']
#     im_metadata = f['HDFEOS INFORMATION']['StructMetadata.0'][()].decode('utf-8')
    
#     # Parse metadata string into key-value pairs
#     metadata_dict = {}
#     for line in im_metadata.split('\n'):
#         line = line.strip()
#         if '=' in line:
#             key, value = line.split('=', 1)
#             metadata_dict[key.strip()] = value.strip().strip('"').replace('(', '').replace(')', '')
#     # Convert to DataFrame
#     df_metadata = pd.DataFrame([metadata_dict])

#     # Apply scale and offset
#     slope_bs = bs_albedo.attrs['scale_factor']
#     offset_bs = bs_albedo.attrs['add_offset']
#     bs_albedo = bs_albedo[:].astype(np.float32) * slope_bs + offset_bs

#     slope_ws = ws_albedo.attrs['scale_factor']
#     offset_ws = ws_albedo.attrs['add_offset']
#     ws_albedo = ws_albedo[:].astype(np.float32) * slope_ws + offset_ws

#     # 2. Apply QA Mask (IMPORTANT: do this BEFORE value range checks)
#     bs_albedo_masked = apply_qa_mask(bs_albedo, qa_flag[:])
#     ws_albedo_masked = apply_qa_mask(ws_albedo, qa_flag[:])
    
#     # 3. Additional value range checks (after QA masking)
#     bs_albedo_masked[bs_albedo_masked <= 0] = np.nan
#     bs_albedo_masked[bs_albedo_masked >= 1] = np.nan
#     ws_albedo_masked[ws_albedo_masked <= 0] = np.nan
#     ws_albedo_masked[ws_albedo_masked >= 1] = np.nan
    
#     # 4. Get Geolocation Attributes
#     rows, cols = int(df_metadata.YDim[0]), int(df_metadata.XDim[0])
    
#     # 5. Define Source Transform (Sinusoidal)
#     src_transform = get_eqa_transform(df_metadata, cols, rows)

    
#     print(list(f['HDFEOS'])) 
#     print(list(f['HDFEOS']['GRIDS'])) 
#     print(list(f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF'])) 
#     print(list(f['HDFEOS']['GRIDS']['VIIRS_Grid_BRDF']['Data Fields'])) 
#     print(list(f['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'])) # is empty
#     SouthBoundingCoord = f.attrs['SouthBoundingCoord']
# # %%

# %%
