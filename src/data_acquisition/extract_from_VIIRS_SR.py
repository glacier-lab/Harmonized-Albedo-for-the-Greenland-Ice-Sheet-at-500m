#%%
"""
Extract VIIRS SR (multi-band) values from daily mosaic GeoTIFF files at AWS locations.
Handles CEN AWS reinstallation on 2017-07-25.

This follows the same workflow as extract_from_GCOMC_SR.py:
- discover daily mosaics
- parse date from filename
- read BAND_NAMES from GeoTIFF tag
- nearest-neighbor extraction at AWS points
- append daily records to CSV
"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
from pyproj import Transformer, CRS

# %%
aws_path = '/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv'
df_aws = pd.read_csv(aws_path)

# Input/output paths for VIIRS SR mosaics
# Expected structure:
# /data_3/shunan_2/AU/hsa500m/VIIRS_SR_mosaics/{VNP09GA|VJ109GA|VJ209GA}/VIIRS_SR_{PRODUCT}_{YYYYMMDD}_500m.tif
viirs_sr_root = '/data_3/shunan_2/AU/hsa500m/VIIRS_SR_mosaics'
csv_output_dir = '/data_3/shunan_2/AU/hsa500m/VIIRS_SR_mosaics'

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

PRODUCTS = ['VNP09GA', 'VJ109GA', 'VJ209GA']

# %%
# Find VIIRS SR mosaic files per product/sensor.
files_by_product = {
    product: sorted(
        glob.glob(
            os.path.join(viirs_sr_root, '**', f'VIIRS_SR_{product}_*_500m.tif'),
            recursive=True,
        )
    )
    for product in PRODUCTS
}

total_input_files = sum(len(v) for v in files_by_product.values())

if total_input_files == 0:
    print(f"No VIIRS SR files found in {viirs_sr_root}, exiting.")
else:
    total_records = 0
    skipped_files = 0
    os.makedirs(csv_output_dir, exist_ok=True)
    # Track per-product output state
    header_written_by_product = {}
    csv_output_by_product = {}

    for product in PRODUCTS:
        product_files = files_by_product.get(product, [])
        print(f"\n[{product}] Found {len(product_files)} files")

        if len(product_files) == 0:
            continue

        for i, file_path in enumerate(product_files):
            imname = os.path.basename(file_path)

            try:
                # Expected filename format: VIIRS_SR_{PRODUCT}_{YYYYMMDD}_500m.tif
                parts = imname.split('_')
                if len(parts) < 5:
                    raise ValueError(f"Unexpected filename format: {imname}")

                date_str = parts[3]
                imtime = pd.to_datetime(date_str, format='%Y%m%d')
                year = imtime.year

                print(f"[{product}] Processing {i+1}/{len(product_files)}: {imname}")

                # Read band names from GeoTIFF tag written by build_mosaic_VIIRS_SR.py
                with rio.open(file_path) as src:
                    band_names_tag = src.tags().get('BAND_NAMES', '')

                band_names = [b.strip() for b in band_names_tag.split(',') if b.strip()]

                # Open GeoTIFF with xarray/rasterio
                ds = xr.open_dataarray(file_path, engine='rasterio')

                # Normalize band naming if missing or mismatched
                n_bands = int(ds.sizes.get('band', 1))
                if len(band_names) != n_bands:
                    band_names = [f'band_{k}' for k in range(1, n_bands + 1)]

                # Create one output CSV per product/sensor.
                if product not in csv_output_by_product:
                    csv_output_by_product[product] = os.path.join(
                        csv_output_dir, f'viirs_sr_values_{product}.csv'
                    )

                csv_output_path = csv_output_by_product[product]

                # Write CSV header once per product, after band names are known.
                if not header_written_by_product.get(product, False):
                    out_cols = ['aws', 'time'] + band_names
                    with open(csv_output_path, 'w') as f:
                        f.write(','.join(out_cols) + '\n')
                    header_written_by_product[product] = True

                # Get AWS data for this year
                df_aws_year = df_aws[df_aws['year'] == year]

                if len(df_aws_year) == 0:
                    print(f"  No AWS data for year {year}, skipping")
                    ds.close()
                    continue

                sr_records = []

                for _, row in df_aws_year.iterrows():
                    aws_name = row['aws']

                    # Special handling only for CEN station in 2017 before reinstallation
                    if (
                        year == 2017
                        and aws_name == CEN_NAME
                        and imtime < pd.Timestamp(CEN_REINSTALLATION_DATE)
                        and cen_coords_2016 is not None
                    ):
                        lat, lon = cen_coords_2016
                    else:
                        lat = row['lat']
                        lon = row['lon']

                    # Transform from WGS84 to EPSG:3413
                    x_proj, y_proj = transformer.transform(lon, lat)

                    try:
                        # Nearest neighbor extraction at AWS location
                        sample = ds.sel(x=x_proj, y=y_proj, method='nearest')
                        values = np.atleast_1d(np.asarray(sample.values, dtype=np.float32))

                        rec = {
                            'aws': aws_name,
                            'time': imtime,
                        }
                        for bname, bval in zip(band_names, values):
                            rec[bname] = bval

                        sr_records.append(rec)

                    except Exception as e:
                        print(f"  Failed to extract value for AWS {aws_name}: {e}")

                if sr_records:
                    df_sr = pd.DataFrame(sr_records)
                    df_sr.to_csv(csv_output_path, mode='a', header=False, index=False)
                    total_records += len(df_sr)
                    print(f"  Extracted {len(df_sr)} records -> {os.path.basename(csv_output_path)}")

                ds.close()

            except Exception as e:
                print(f"  Error processing {imname}: {type(e).__name__}: {e}")
                skipped_files += 1
                continue

    print(f"\n{'='*60}")
    print(f"Done. Wrote {total_records} rows across {len(csv_output_by_product)} sensor CSV files")
    for product, path in sorted(csv_output_by_product.items()):
        print(f"  {product}: {path}")
    print(f"Skipped {skipped_files} files due to errors")
    print(f"{'='*60}")
