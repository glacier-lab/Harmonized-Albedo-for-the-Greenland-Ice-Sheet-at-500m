For Antarctica:

VIIRS Surface Reflectance (SR) data can be downloaded from NASA Earthdata using the following command:

python src/data_acquisition/nasa_earth_data_downloader.py   --short-name VJ209GA   --version 002   --start-date 2023-01-01   --end-date 2026-12-31   --bbox -180.0 -90.0 180.0 -60.0   --output-dir /data_3/shunan_2/AU/hsa500m/Antarctica/VIIRS_SR/VJ209GA

python src/data_acquisition/nasa_earth_data_downloader.py   --short-name VJ109GA   --version 002   --start-date 2018-01-01   --end-date 2026-12-31   --bbox -180.0 -90.0 180.0 -60.0   --output-dir /data_3/shunan_2/AU/hsa500m/Antarctica/VIIRS_SR/VJ109GA


VIIRS SR data preprocessing can be performed using the following command:
python src/polar/build_mosaic_VIIRS_SR.py \
  --region antarctica \
  --input-root /data_3/shunan_2/AU/hsa500m/Antarctica/VIIRS_SR \
  --output-root /data_3/shunan_2/AU/hsa500m/Antarctica/VIIRS_SR_mosaics \
  --products VJ209GA

python src/polar/apply_n2b_viirsSR.py \
  --region antarctica \
  --input-root /data_3/shunan_2/AU/hsa500m/Antarctica/VIIRS_SR_mosaics \
  --output-root /data_3/shunan_2/AU/hsa500m/Antarctica/VIIRS_SR_albedo \
  --workers 10