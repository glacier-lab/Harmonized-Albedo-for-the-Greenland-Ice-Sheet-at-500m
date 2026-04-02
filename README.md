# Harmonized Satellite Albedo for the Greenland Ice Sheet at 500m (HSA500m)


## Overview

This repository contains scripts to produce a harmonized, gap-filled albedo product for the Greenland Ice Sheet (GrIS) at 500m spatial resolution. The product combines multiple satellite-based albedo datasets (MODIS: MOD10A1, MYD10A1, MCD43A3, VIIRS: VJ143MA3, VNP43MA3, SICE, AND GCOM-C) with CARRA reanalysis data to create a seamless daily albedo time series. 


Note: This repository is part of a manuscript that is currently under preparation. The scripts and data products are provided here for transparency and reproducibility. We welcome feedback and suggestions for improvement.

## Data Product Structure

The final product consists of **2-band GeoTIFF files** with the following specifications:

### File Format
- **Filename**: `hsa500m_gapfilled_YYYYMMDD.tif`
- **Spatial Resolution**: 500m
- **Projection**: EPSG:3413 (NSIDC Polar Stereographic North)
- **Data Type**: Float32 (0.0–1.0 range)
- **Compression**: LZW with predictor=3

### Band 1: Harmonized Satellite Albedo (hsa500m_gapfilled)
- **Content**: Gap-filled surface albedo values
- **Range**: 0.0–1.0 
- **Sources**:
  - Satellite-derived albedo (when available)
  - Calibrated CARRA reanalysis (gap fill)
- **NoData**: NaN (non-ice areas)

### Band 2: QA (Scenario Map) 
- **Content**: Data source and quality indicator for each pixel
- **Values**:
  - **>0**: Scenario ID (satellite-calibrated pixels). Each unique scenario ID represents a specific combination of satellite sensors used for that pixel.
  - **0**: Filled with calibrated CARRA and pre-calibration CARRA albedo < 0.83
  - **−1**: Filled with calibrated CARRA and pre-calibration CARRA albedo ≥ 0.83 (high-albedo cap flag)
  - **NaN**: No valid source data available
  - **Note**: Scenario IDs are defined in `calibration_coefficients.csv` and correspond to specific sensor combinations and calibration coefficients. CARRA albedo data have a data cap so we can flag pixels where the original CARRA value exceeds the cap, indicating a higher uncertainty in the gap-filled value.

## Processing Workflow

### [src/data_acquisition](src/data_acquisition) - Data acquisition scripts for downloading and preprocessing satellite albedo products and AWS data.

Following scripts are provided for data acquisition. Users can run these scripts to download the necessary datasets or obtain the data through alternative means if preferred.
- [src/data_acquisition/MODIS_downloader.js](src/data_acquisition/MODIS_downloader.js): **MOD10A1** and **MYD10A1** can be downloaded from Google Earth Engine to user's Goole Drive.
- [https://nasa.github.io/earthdata-download/](https://nasa.github.io/earthdata-download/): **MCD43A3**, **VJ143MA3**, and **VNP43MA3** datasets are available on NASA earthdata portal and can be batch downloaded with the officalt tool.
- [src/data_acquisition/SICE_downloader_single.py](src/data_acquisition/SICE_downloader_single.py): **SICE** can be downloaded from the GEUS Thredds using the script. It is set to download single file at a time.
- [src/data_acquisition/utokyo_downloader_single.py](src/data_acquisition/utokyo_downloader_single.py): Downscaled **CARRA** albedo data are processed at University of Tokyo. Users should request the corresponding author for creditionals to access the server.

Next part of data preprocessing workflow is to build mosaic of each satellite products and CARRA, and reproject them to the same CRS (EPSG:3413), spatial resoluiton (500m) and extent. All of the preprocessed satellite products and CARRA mosaics are masked by the PROMICE-2022 Ice Mask.
- [`src/data_acquisition/build_mosaic_MOD10_MYD10.py`](src/data_acquisition/build_mosaic_MOD10_MYD10.py)
- [`src/data_acquisition/build_mosaic_MCD43A3.py`](src/data_acquisition/build_mosaic_MCD43A3.py)
- [`src/data_acquisition/build_mosaic_VIIRS.py`](src/data_acquisition/build_mosaic_VIIRS.py)
- [`src/data_acquisition/rebuild_SICE.py`](src/data_acquisition/rebuild_SICE.py)
- [`src/data_acquisition/build_mosaic_CARRA_MRI.py`](src/data_acquisition/build_mosaic_CARRA_MRI.py)

Unlike other albedo prducts, GCOM-C albedo is derived from surface reflectance products and requires an additional step to convert to albedo. The script for building GCOM-C mosaic also includes the conversion process.
- [`src/data_acquisition/build_mosaic_GCOMC_SR.py`](src/data_acquisition/build_mosaic_GCOMC_SR.py)
- [`src/data_acquisition/extract_from_GCOMC_SR.py`](src/data_acquisition/extract_from_GCOMC_SR.py): Extract SR values from PROMICE/GC-Net AWS statiions.
- [`src/data_acquisition/narrow2broadband_gcomcSR.py`](src/data_acquisition/narrow2broadband_gcomcSR.py): Develop and test the narrow-to-broadband conversion for GCOM-C SR products using PROMICE/GC-Net AWS data.
- [`src/data_acquisition/apply_n2b_gcomcsr.py`](src/data_acquisition/apply_n2b_gcomcsr.py): Apply the narrow-to-broadband conversion to the GCOM-C SR mosaic to produce the GCOM-C albedo mosaic.

### [src/analysis](src/analysis) - Main analysis scripts for point-to-pixel extraction, calibration coefficient calculation, and gapfilled product assembly.

-[`src/analysis/extract_point2pix.py`](src/analysis/extract_point2pix.py): Extracts satellite and CARRA albedo values at PROMICE/GC-Net AWS station locations, creating point-to-pixel datasets for calibration.
-[`src/analysis/calculate_calibration_coef.py`](src/analysis/calculate_calibration_coef.py): Calculates calibration coefficients for each unique sensor combination scenario using the point-to-pixel datasets. This is a critical step.

**Process**:
1. Generates all valid sensor combinations (accounts for orbital drift)
2. Splits data into 70% training / 30% test sets
3. Calculates validation metrics (R², RMSE, MAE, bias)
4. Creates calibration plots for visual inspection
5. Produces `calibration_coefficients.csv` with:
   - Scenario IDs (unique sensor combinations)
   - Calibration coefficients (slope, intercept)
   - Validation metrics for both training and test datasets

**MODIS Orbital Drift Handling**:
- MOD10A1: Pre-2020 data combined; post-2020 treated annually
- MYD10A1: Pre-2021 data combined; post-2021 treated annually
- MCD43A3: Follows MOD10A1 timeline (drift starts 2020)

-[`src/analysis/build_carra_hsa_comparison.py`](src/analysis/build_carra_hsa_comparison.py): Once the calibration coefficients are calculated, this script applies the calibration the satllite albedo and produces the daily albedo mosatic. It then compares the HSA500m (with gaps) with the paired daily CARRA albedo. Paired pixels are converted into one-dimensional arrays, split into training (70%) and testing (30%) data, and saved as hdf5 files. 

-[`src/analysis/carra_calibration_coef.py`](src/analysis/carra_calibration_coef.py): This script uses the paired HSA500m-CARRA data to calculate the calibration coefficients for CARRA. 

-[`src/analysis/build_hsa500m_gapfilled.py`](src/analysis/build_hsa500m_gapfilled.py): Builds final 2-band gapfilled GeoTIFFs for each day.    

**Input**:
- Daily CARRA GeoTIFFs (base grid reference)
- Same-day satellite GeoTIFFs
- Calibration coefficients (`calibration_coefficients.csv`)

**Process**:
1. **Collect satellite data**: Gathers same-day files from all available sensors
2. **Compute satellite average**: Averages multi-sensor albedos at each pixel
3. **Match calibration**: Identifies applicable scenario based on available sensors
4. **Apply calibration**: Transforms satellite albedo using scenario-specific coefficients
5. **Fill gaps**: Uses calibrated CARRA to fill missing satellite data
6. **Mark scenario**: Assigns appropriate scenario flag based on data source and CARRA magnitude

