"""
Build per-day calibrated albedo arrays against CARRA and export to HDF5.

Workflow:
1. Iterate through CARRA GeoTIFF files (base grid).
2. Find same-day satellite GeoTIFFs across configured sensors.
3. Build a multi-band satellite stack (one band per sensor) on CARRA grid.
4. Compute per-pixel satellite average albedo.
5. Match sensor availability to calibration scenario from calibration_coefficients.csv.
6. Calibrate satellite average albedo with matched coefficients.
7. Flatten CARRA, calibrated albedo, scenario into 3 columns.
8. Split into train/test (70/30) and save as separate HDF5 files.
9. Remove rows where scenario == 0 and save one HDF5 per day.

Output HDF5 files are Vaex-compatible and saved daily.
"""

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import vaex


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CARRA_DIR = "/data_3/shunan_2/AU/hsa500m/CARRA_GL500m_geotiff"
CALIBRATION_CSV = "/data_3/shunan_2/AU/hsa500m/calibration/calibration_coefficients.csv"

OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m"
OUTPUT_HDF5_DIR = os.path.join(OUTPUT_DIR, "carra_hsa_comparison_hdf5")
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_HDF5_DIR, "training")
OUTPUT_TEST_DIR = os.path.join(OUTPUT_HDF5_DIR, "testing")

OVERWRITE_HDF5_AT_START = True
NUM_WORKERS = 8

SENSOR_CONFIGS = {
    "MOD10A1": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/MOD10A1_cropped",
        "glob": "*.tif",
        "date_regex": r"(\d{4}-\d{2}-\d{2})",
        "date_fmt": "%Y-%m-%d",
        "scenario_key": "mod10",
        "dynamic_by_year": True,
        "drift_start_year": 2020,
    },
    "MYD10A1": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/MYD10A1_cropped",
        "glob": "*.tif",
        "date_regex": r"(\d{4}-\d{2}-\d{2})",
        "date_fmt": "%Y-%m-%d",
        "scenario_key": "myd10",
        "dynamic_by_year": True,
        "drift_start_year": 2021,
    },
    "MCD43A3_BLUESKY": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/MCD43A3_061_bluesky",
        "glob": "MCD43A3_BlueskyAlbedo_*_500m.tif",
        "date_regex": r"MCD43A3_BlueskyAlbedo_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "mcd43a3_bluesky",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "VIIRS_VJ143MA3_BLUESKY": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VJ143MA3",
        "glob": "VJ143MA3_BlueskyAlbedo_*_500m.tif",
        "date_regex": r"VJ143MA3_BlueskyAlbedo_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "viirs_vj143ma3_bluesky",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "VIIRS_VNP43MA3_BLUESKY": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/VIIRS_bluesky/VNP43MA3",
        "glob": "VNP43MA3_BlueskyAlbedo_*_500m.tif",
        "date_regex": r"VNP43MA3_BlueskyAlbedo_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "viirs_vnp43ma3_bluesky",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "GCOMC_SR_ALBEDO": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/GCOMC_SR_albedo",
        "glob": "GCOMC_SRalbedo_*_500m.tif",
        "date_regex": r"GCOMC_SRalbedo_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "gcomc",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "SICE_REBUILD": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/SICE_rebuild",
        "glob": "SICE_Albedo_*_500m.tif",
        "date_regex": r"SICE_Albedo_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "sice",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
}


COEFF_DF = None
SCENARIO_COLS = None
SENSOR_FILE_INDEX = None


def parse_date_from_name(filename: str, date_regex: str, date_fmt: str) -> Optional[pd.Timestamp]:
    match = re.search(date_regex, filename)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format=date_fmt)


def build_file_index(config: Dict) -> Dict[pd.Timestamp, str]:
    files = sorted(glob.glob(os.path.join(config["input_dir"], config["glob"])))
    idx: Dict[pd.Timestamp, str] = {}

    for fp in files:
        ts = parse_date_from_name(os.path.basename(fp), config["date_regex"], config["date_fmt"])
        if ts is None:
            continue
        idx[ts.normalize()] = fp

    return idx


def read_band_on_base_grid(
    file_path: str,
    base_shape: Tuple[int, int],
    base_transform,
    base_crs,
) -> np.ndarray:
    with rio.open(file_path) as src:
        same_grid = (
            src.crs == base_crs
            and src.transform == base_transform
            and src.height == base_shape[0]
            and src.width == base_shape[1]
        )
        if not same_grid:
            raise ValueError(
                "Satellite raster grid does not match CARRA grid. "
                "Expected pre-reprojected/masked orthomosaic inputs."
            )

        src_data = src.read(1).astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            src_data[src_data == src.nodata] = np.nan

    return src_data


def load_calibration_table(path: str) -> Tuple[pd.DataFrame, list]:
    df = pd.read_csv(path)

    known_non_sensor_cols = {
        "scenario_id",
        "scenario",
        "sensors",
        "n_sensors",
        "n_train",
        "n_test",
        "n_total",
        "train_r_squared",
        "train_rmse",
        "train_mae",
        "train_bias",
        "test_calib_r_squared",
        "test_calib_rmse",
        "test_calib_mae",
        "test_calib_bias",
        "train_slope",
        "train_intercept",
        "test_calib_slope",
        "test_calib_intercept",
        "slope",
        "intercept",
    }

    scenario_cols = [c for c in df.columns if c not in known_non_sensor_cols]
    return df, scenario_cols


def scenario_key_for_sensor(sensor_name: str, date: pd.Timestamp) -> str:
    cfg = SENSOR_CONFIGS[sensor_name]
    base_key = cfg["scenario_key"]

    if not cfg["dynamic_by_year"]:
        return base_key

    if date.year < cfg["drift_start_year"]:
        return base_key

    return f"{base_key}_{date.year}"


def build_scenario_indicator(date: pd.Timestamp, available_sensors: Set[str], scenario_cols: list) -> Dict[str, int]:
    indicator = {c: 0 for c in scenario_cols}

    for sensor_name in available_sensors:
        scenario_key = scenario_key_for_sensor(sensor_name, date)
        if scenario_key in indicator:
            indicator[scenario_key] = 1

    return indicator


def match_calibration_row(coeff_df: pd.DataFrame, scenario_cols: list, indicator: Dict[str, int]) -> Optional[pd.Series]:
    if len(scenario_cols) == 0:
        return None

    mask = np.ones(len(coeff_df), dtype=bool)
    for col in scenario_cols:
        mask &= coeff_df[col].fillna(0).astype(int).to_numpy() == int(indicator.get(col, 0))

    matched = coeff_df.loc[mask]
    if matched.empty:
        return None

    if "test_calib_r_squared" in matched.columns:
        matched = matched.sort_values("test_calib_r_squared", ascending=False)

    return matched.iloc[0]


def get_training_coefficients(row: pd.Series) -> Tuple[float, float]:
    if "train_slope" in row.index and "train_intercept" in row.index:
        return float(row["train_slope"]), float(row["train_intercept"])
    return float(row["slope"]), float(row["intercept"])


def parse_carra_date(file_name: str) -> Optional[pd.Timestamp]:
    m = re.search(r"CARRA_Albedo_(\d{8})_500m\.tif", file_name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def save_daily_hdf5(df_chunk: pd.DataFrame, h5_path: str) -> None:
    if df_chunk.empty:
        return

    df_new = vaex.from_pandas(df_chunk, copy_index=False)
    df_new.export_hdf5(h5_path, mode="w")


def init_worker(calibration_csv: str) -> None:
    global COEFF_DF, SCENARIO_COLS, SENSOR_FILE_INDEX
    COEFF_DF, SCENARIO_COLS = load_calibration_table(calibration_csv)
    SENSOR_FILE_INDEX = {name: build_file_index(cfg) for name, cfg in SENSOR_CONFIGS.items()}


def process_single_carra_file(carra_fp: str, train_dir: str, test_dir: str):
    if COEFF_DF is None or SCENARIO_COLS is None or SENSOR_FILE_INDEX is None:
        return False, "Worker not initialized"

    coeff_df = COEFF_DF
    scenario_cols = SCENARIO_COLS
    sensor_file_index = SENSOR_FILE_INDEX

    carra_name = os.path.basename(carra_fp)
    day = parse_carra_date(carra_name)
    if day is None:
        return False, f"Skip {carra_name}: cannot parse date"

    day = day.normalize()
    date_str = day.strftime('%Y%m%d')
    train_h5 = os.path.join(train_dir, f"hsa500m_{date_str}.h5")
    test_h5 = os.path.join(test_dir, f"hsa500m_{date_str}.h5")

    with rio.open(carra_fp) as carra_src:
        carra = carra_src.read(1).astype(np.float32)
        base_shape = (carra_src.height, carra_src.width)
        base_transform = carra_src.transform
        base_crs = carra_src.crs
        carra_nodata = carra_src.nodata

    if carra_nodata is not None and np.isfinite(carra_nodata):
        carra[carra == carra_nodata] = np.nan
    carra[(carra <= 0) | (carra >= 1)] = np.nan

    sat_bands = []
    available_sensors: Set[str] = set()

    for sensor_name in SENSOR_CONFIGS:
        sensor_fp = sensor_file_index[sensor_name].get(day)
        if sensor_fp is None:
            sat_bands.append(np.full(base_shape, np.nan, dtype=np.float32))
            continue

        try:
            band = read_band_on_base_grid(sensor_fp, base_shape, base_transform, base_crs)
            band[(band <= 0) | (band >= 1)] = np.nan
            sat_bands.append(band)
            if np.isfinite(band).any():
                available_sensors.add(sensor_name)
        except Exception:
            sat_bands.append(np.full(base_shape, np.nan, dtype=np.float32))

    sat_stack = np.stack(sat_bands, axis=0)
    valid_counts = np.sum(np.isfinite(sat_stack), axis=0)
    sat_sum = np.nansum(sat_stack, axis=0)
    sat_avg = np.full(base_shape, np.nan, dtype=np.float32)
    valid_pixels = valid_counts > 0
    sat_avg[valid_pixels] = (sat_sum[valid_pixels] / valid_counts[valid_pixels]).astype(np.float32)

    indicator = build_scenario_indicator(day, available_sensors, scenario_cols)
    calib_row = match_calibration_row(coeff_df, scenario_cols, indicator)

    if calib_row is None or not np.isfinite(sat_avg).any():
        scenario_id = 0
        calibrated = np.full(base_shape, np.nan, dtype=np.float32)
    else:
        slope, intercept = get_training_coefficients(calib_row)
        calibrated = (slope * sat_avg + intercept).astype(np.float32)
        calibrated = np.clip(calibrated, 0, 1)
        scenario_id = int(calib_row["scenario_id"])

    scenario_img = np.full(base_shape, scenario_id, dtype=np.uint16)
    scenario_img[np.isnan(sat_avg)] = 0

    carra_flat = carra.ravel()
    calibrated_flat = calibrated.ravel()
    scenario_flat = scenario_img.ravel().astype(np.int32)

    valid = (
        (scenario_flat != 0)
        & np.isfinite(carra_flat)
        & np.isfinite(calibrated_flat)
    )

    df_chunk = pd.DataFrame(
        {
            "carra": carra_flat[valid],
            "hsa500m": calibrated_flat[valid],
            "scenario": scenario_flat[valid],
        }
    )

    df_train, df_test = train_test_split(df_chunk, test_size=0.3, random_state=42)
    save_daily_hdf5(df_train, train_h5)
    save_daily_hdf5(df_test, test_h5)
    return True, f"{day.strftime('%Y-%m-%d')} -> train={len(df_train)}, test={len(df_test)}"


def main() -> None:
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

    if OVERWRITE_HDF5_AT_START:
        for old_file in glob.glob(os.path.join(OUTPUT_TRAIN_DIR, "hsa500m_*.h5")):
            os.remove(old_file)
        for old_file in glob.glob(os.path.join(OUTPUT_TEST_DIR, "hsa500m_*.h5")):
            os.remove(old_file)

    coeff_df, _ = load_calibration_table(CALIBRATION_CSV)

    carra_files = sorted(glob.glob(os.path.join(CARRA_DIR, "CARRA_Albedo_*_500m.tif")))
    if len(carra_files) == 0:
        raise FileNotFoundError(f"No CARRA files found in {CARRA_DIR}")

    print(f"CARRA files: {len(carra_files)}")
    print(f"Calibration rows: {len(coeff_df)}")
    print(f"Training HDF5 dir: {OUTPUT_TRAIN_DIR}")
    print(f"Testing HDF5 dir:  {OUTPUT_TEST_DIR}")
    print(f"Workers: {NUM_WORKERS}")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker,
        initargs=(CALIBRATION_CSV,),
    ) as executor:
        futures = [executor.submit(process_single_carra_file, carra_fp, OUTPUT_TRAIN_DIR, OUTPUT_TEST_DIR) for carra_fp in carra_files]

        success = 0
        failed = 0
        with tqdm(total=len(futures), desc="Processing CARRA days", unit="day") as pbar:
            for future in as_completed(futures):
                try:
                    ok, msg = future.result()
                except Exception as exc:
                    ok, msg = False, f"Worker failed: {type(exc).__name__}: {exc}"
                if ok:
                    success += 1
                else:
                    failed += 1
                    tqdm.write(msg)
                pbar.update(1)

    print("Done.")
    print(f"Succeeded: {success}, Failed: {failed}")
    print(f"Training HDF5 files: {OUTPUT_TRAIN_DIR}")
    print(f"Testing HDF5 files:  {OUTPUT_TEST_DIR}")


if __name__ == "__main__":
    main()
