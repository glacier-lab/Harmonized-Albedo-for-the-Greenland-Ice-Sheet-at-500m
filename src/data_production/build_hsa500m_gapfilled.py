"""
Build daily gapfilled HSA500m GeoTIFFs.

Workflow:
1. Iterate over CARRA daily GeoTIFF files (base grid/reference date).
2. Collect same-day satellite files from configured sensors.
3. Build per-pixel satellite average albedo on CARRA grid.
4. Match available sensors to a row in calibration_coefficients.csv.
5. Apply matched satellite calibration to build HSA estimate from satellite data.
6. Fill all satellite gaps with calibrated CARRA where CARRA is valid.
7. Mark filled pixels as scenario 0 when pre-calibration CARRA < CARRA_CAP, else -1.
8. Save 2-band GeoTIFF:
   - Band 1: gapfilled HSA500m
   - Band 2: scenario map

Scenario map conventions:
- >0: scenario_id from calibration_coefficients.csv (satellite-calibrated pixels)
-  0: filled with calibrated CARRA and pre-calibration CARRA < CARRA_CAP
- -1: filled with calibrated CARRA and pre-calibration CARRA >= CARRA_CAP
- NaN: no valid source data
"""

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CARRA_DIR = "/data_3/shunan_2/AU/hsa500m/CARRA_GL500m_geotiff"
CALIBRATION_CSV = "/data_3/shunan_2/AU/hsa500m/calibration/calibration_coefficients.csv"

OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
OVERWRITE_EXISTING_FILES = True
NUM_WORKERS = 10
ASSUME_PREALIGNED_GRID = True

# User-adjustable CARRA calibration coefficients for gap filling.
CARRA_CALIB_SLOPE = 1.0305
CARRA_CALIB_INTERCEPT = -0.0578
CARRA_CAP = 0.83

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
        "dynamic_by_year": True,
        "drift_start_year": 2020,
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
        "date_regex": r"GCOMC_SRalbedo_(\d{8})_500m.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "gcomc",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "SICE_REBUILD": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/SICE_rebuild",
        "glob": "SICE_Albedo_*_500m.tif",
        "date_regex": r"SICE_Albedo_(\d{8})_500m.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "sice",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "VIIRS_SR_ALBEDO_VNP09GA": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/VIIRS_SR_albedo/VNP09GA",
        "glob": "VIIRS_SRalbedo_VNP09GA_*_500m.tif",
        "date_regex": r"VIIRS_SRalbedo_VNP09GA_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "viirs_sr_vnp09ga",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "VIIRS_SR_ALBEDO_VJ109GA": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/VIIRS_SR_albedo/VJ109GA",
        "glob": "VIIRS_SRalbedo_VJ109GA_*_500m.tif",
        "date_regex": r"VIIRS_SRalbedo_VJ109GA_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "viirs_sr_vj109ga",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
    "VIIRS_SR_ALBEDO_VJ209GA": {
        "input_dir": "/data_3/shunan_2/AU/hsa500m/VIIRS_SR_albedo/VJ209GA",
        "glob": "VIIRS_SRalbedo_VJ209GA_*_500m.tif",
        "date_regex": r"VIIRS_SRalbedo_VJ209GA_(\d{8})_500m\.tif",
        "date_fmt": "%Y%m%d",
        "scenario_key": "viirs_sr_vj209ga",
        "dynamic_by_year": False,
        "drift_start_year": None,
    },
}

DAILY_SENSOR_NAMES = [
    "MOD10A1",
    "MYD10A1",
    "GCOMC_SR_ALBEDO",
    "SICE_REBUILD",
    "VIIRS_SR_ALBEDO_VNP09GA",
    "VIIRS_SR_ALBEDO_VJ109GA",
    "VIIRS_SR_ALBEDO_VJ209GA",
]

FALLBACK_16DAY_SENSOR_NAMES = [
    "MCD43A3_BLUESKY",
    "VIIRS_VJ143MA3_BLUESKY",
    "VIIRS_VNP43MA3_BLUESKY",
]


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
        if src.height != base_shape[0] or src.width != base_shape[1]:
            raise ValueError(
                "Satellite raster shape does not match CARRA shape. "
                "Expected pre-reprojected/masked orthomosaic inputs."
            )

        if not ASSUME_PREALIGNED_GRID:
            if src.crs != base_crs or src.transform != base_transform:
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
        "scenario_family",
        "workflow",
        "sensors",
        "n_sensors",
        "daily_sensor_groups",
        "fallback_sensor_groups",
        "daily_sensors",
        "fallback_sensors",
        "n_train",
        "n_test",
        "n_total",
        "n_daily_used",
        "n_fallback_used",
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
    return match_calibration_row_by_family(coeff_df, scenario_cols, indicator, scenario_family=None)


def match_calibration_row_by_family(
    coeff_df: pd.DataFrame,
    scenario_cols: list,
    indicator: Dict[str, int],
    scenario_family: Optional[str],
    active_sensor_keys: Optional[Set[str]] = None,
) -> Optional[pd.Series]:
    # Prefer string/group matching whenever active sensor keys are provided.
    # This avoids mismatches when legacy one-hot columns are still present.
    if active_sensor_keys is not None:
        # New calibration schema stores active sensors as comma-separated strings.
        matched = coeff_df.copy()
        if scenario_family is not None and "scenario_family" in matched.columns:
            matched = matched[matched["scenario_family"].astype(str) == scenario_family]

        if matched.empty:
            return None

        sensor_col = "daily_sensor_groups" if scenario_family == "daily" else "fallback_sensor_groups"
        if sensor_col not in matched.columns:
            return None

        def parse_sensor_set(v: object) -> Set[str]:
            if v is None:
                return set()
            if isinstance(v, float) and np.isnan(v):
                return set()
            return {s.strip() for s in str(v).split(",") if s.strip()}

        sets = matched[sensor_col].apply(parse_sensor_set)
        exact = matched[sets == set(active_sensor_keys)]
        if exact.empty:
            return None

        if "test_calib_r_squared" in exact.columns:
            exact = exact.sort_values("test_calib_r_squared", ascending=False)
        return exact.iloc[0]

    mask = np.ones(len(coeff_df), dtype=bool)
    if scenario_family is not None and "scenario_family" in coeff_df.columns:
        mask &= coeff_df["scenario_family"].astype(str).to_numpy() == scenario_family

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


def init_worker(calibration_csv: str) -> None:
    global COEFF_DF, SCENARIO_COLS, SENSOR_FILE_INDEX
    COEFF_DF, SCENARIO_COLS = load_calibration_table(calibration_csv)
    SENSOR_FILE_INDEX = {name: build_file_index(cfg) for name, cfg in SENSOR_CONFIGS.items()}


def calibrate_carra(carra: np.ndarray) -> np.ndarray:
    calibrated = (CARRA_CALIB_SLOPE * carra + CARRA_CALIB_INTERCEPT).astype(np.float32)
    return np.clip(calibrated, 0, 1)


def process_single_carra_file(carra_fp: str, output_dir: str):
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
    date_str = day.strftime("%Y%m%d")
    output_tif = os.path.join(output_dir, f"hsa500m_gapfilled_{date_str}.tif")

    if os.path.exists(output_tif) and not OVERWRITE_EXISTING_FILES:
        return True, f"{day.strftime('%Y-%m-%d')} -> skipped (exists)"

    with rio.open(carra_fp) as carra_src:
        carra = carra_src.read(1).astype(np.float32)
        profile = carra_src.profile.copy()
        base_shape = (carra_src.height, carra_src.width)
        base_transform = carra_src.transform
        base_crs = carra_src.crs
        carra_nodata = carra_src.nodata

    if carra_nodata is not None and np.isfinite(carra_nodata):
        carra[carra == carra_nodata] = np.nan
    carra[(carra <= 0) | (carra >= 1)] = np.nan

    sat_by_sensor: Dict[str, np.ndarray] = {}
    for sensor_name in SENSOR_CONFIGS:
        sensor_fp = sensor_file_index[sensor_name].get(day)
        if sensor_fp is None:
            sat_by_sensor[sensor_name] = np.full(base_shape, np.nan, dtype=np.float32)
            continue

        try:
            band = read_band_on_base_grid(sensor_fp, base_shape, base_transform, base_crs)
            band[(band <= 0) | (band >= 1)] = np.nan
            sat_by_sensor[sensor_name] = band
        except Exception:
            sat_by_sensor[sensor_name] = np.full(base_shape, np.nan, dtype=np.float32)

    daily_bands = [sat_by_sensor[name] for name in DAILY_SENSOR_NAMES]
    fallback_bands = [sat_by_sensor[name] for name in FALLBACK_16DAY_SENSOR_NAMES]

    daily_stack = np.stack(daily_bands, axis=0)
    daily_counts = np.sum(np.isfinite(daily_stack), axis=0)
    daily_sum = np.nansum(daily_stack, axis=0)
    daily_avg = np.full(base_shape, np.nan, dtype=np.float32)
    daily_mask = daily_counts > 0
    daily_avg[daily_mask] = (daily_sum[daily_mask] / daily_counts[daily_mask]).astype(np.float32)

    fallback_stack = np.stack(fallback_bands, axis=0)
    fallback_counts = np.sum(np.isfinite(fallback_stack), axis=0)
    fallback_sum = np.nansum(fallback_stack, axis=0)
    fallback_avg = np.full(base_shape, np.nan, dtype=np.float32)
    fallback_mask = fallback_counts > 0
    fallback_avg[fallback_mask] = (fallback_sum[fallback_mask] / fallback_counts[fallback_mask]).astype(np.float32)

    # Production rule: use daily where available; otherwise fallback to 16-day.
    sat_avg = np.full(base_shape, np.nan, dtype=np.float32)
    sat_avg[daily_mask] = daily_avg[daily_mask]
    sat_avg[(~daily_mask) & fallback_mask] = fallback_avg[(~daily_mask) & fallback_mask]

    hsa = np.full(base_shape, np.nan, dtype=np.float32)
    scenario = np.full(base_shape, np.nan, dtype=np.float32)

    # Daily scenarios (priority branch)
    daily_code = np.zeros(base_shape, dtype=np.uint16)
    for i, band in enumerate(daily_bands):
        daily_code |= (np.isfinite(band).astype(np.uint16) << i)

    for code in np.unique(daily_code[daily_mask]):
        if code == 0:
            continue

        indicator = {c: 0 for c in scenario_cols}
        active_keys: Set[str] = set()
        for i, sensor_name in enumerate(DAILY_SENSOR_NAMES):
            if (code >> i) & 1:
                scenario_key = SENSOR_CONFIGS[sensor_name]["scenario_key"]
                active_keys.add(scenario_key)
                if scenario_key in indicator:
                    indicator[scenario_key] = 1

        calib_row = match_calibration_row_by_family(
            coeff_df,
            scenario_cols,
            indicator,
            scenario_family="daily",
            active_sensor_keys=active_keys,
        )
        if calib_row is None:
            continue

        code_mask = daily_mask & (daily_code == code)
        if not np.any(code_mask):
            continue

        slope, intercept = get_training_coefficients(calib_row)
        sat_calibrated = np.clip(slope * sat_avg[code_mask] + intercept, 0, 1).astype(np.float32)
        hsa[code_mask] = sat_calibrated
        scenario[code_mask] = float(int(calib_row["scenario_id"]))

    # 16-day fallback scenarios (only where daily unavailable)
    fallback_only_mask = (~daily_mask) & fallback_mask
    fallback_code = np.zeros(base_shape, dtype=np.uint16)
    for i, band in enumerate(fallback_bands):
        fallback_code |= (np.isfinite(band).astype(np.uint16) << i)

    for code in np.unique(fallback_code[fallback_only_mask]):
        if code == 0:
            continue

        indicator = {c: 0 for c in scenario_cols}
        active_keys: Set[str] = set()
        for i, sensor_name in enumerate(FALLBACK_16DAY_SENSOR_NAMES):
            if (code >> i) & 1:
                scenario_key = SENSOR_CONFIGS[sensor_name]["scenario_key"]
                active_keys.add(scenario_key)
                if scenario_key in indicator:
                    indicator[scenario_key] = 1

        calib_row = match_calibration_row_by_family(
            coeff_df,
            scenario_cols,
            indicator,
            scenario_family="16day",
            active_sensor_keys=active_keys,
        )
        if calib_row is None:
            continue

        code_mask = fallback_only_mask & (fallback_code == code)
        if not np.any(code_mask):
            continue

        slope, intercept = get_training_coefficients(calib_row)
        sat_calibrated = np.clip(slope * sat_avg[code_mask] + intercept, 0, 1).astype(np.float32)
        hsa[code_mask] = sat_calibrated
        scenario[code_mask] = float(int(calib_row["scenario_id"]))

    gap_mask = ~np.isfinite(hsa)
    carra_valid = np.isfinite(carra)

    fill_with_carra = gap_mask & carra_valid
    if np.any(fill_with_carra):
        carra_calibrated = calibrate_carra(carra)
        hsa[fill_with_carra] = carra_calibrated[fill_with_carra]

        below_cap = fill_with_carra & (carra < CARRA_CAP)
        above_or_equal_cap = fill_with_carra & (carra >= CARRA_CAP)
        scenario[below_cap] = 0.0
        scenario[above_or_equal_cap] = -1.0

    profile.update(
        dtype="float32",
        count=2,
        nodata=np.nan,
        compress="LZW",
        predictor=3,
    )

    with rio.open(output_tif, "w", **profile) as dst:
        dst.write(hsa.astype(np.float32), 1)
        dst.set_band_description(1, "hsa500m_gapfilled")
        dst.write(scenario.astype(np.float32), 2)
        dst.set_band_description(2, "scenario")

    n_sat = int(np.sum((scenario > 0) & np.isfinite(hsa)))
    n_carra_cal = int(np.sum(scenario == 0))
    n_carra_raw = int(np.sum(scenario == -1))
    n_nodata = int(np.sum(~np.isfinite(hsa)))
    msg = (
        f"{day.strftime('%Y-%m-%d')} -> sat={n_sat}, carra_cal={n_carra_cal}, "
        f"carra_cal_capflag={n_carra_raw}, nodata={n_nodata}"
    )
    return True, msg


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    coeff_df, _ = load_calibration_table(CALIBRATION_CSV)

    carra_files = sorted(glob.glob(os.path.join(CARRA_DIR, "CARRA_Albedo_*_500m.tif")))
    if len(carra_files) == 0:
        raise FileNotFoundError(f"No CARRA files found in {CARRA_DIR}")

    print(f"CARRA files: {len(carra_files)}")
    print(f"Calibration rows: {len(coeff_df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"CARRA calibration slope/intercept: {CARRA_CALIB_SLOPE}, {CARRA_CALIB_INTERCEPT}")
    print(f"CARRA cap for scenario flag fallback: {CARRA_CAP}")
    print(f"Assume pre-aligned grid: {ASSUME_PREALIGNED_GRID}")
    print(f"Overwrite existing files: {OVERWRITE_EXISTING_FILES}")
    print(f"Workers: {NUM_WORKERS}")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker,
        initargs=(CALIBRATION_CSV,),
    ) as executor:
        futures = [executor.submit(process_single_carra_file, carra_fp, OUTPUT_DIR) for carra_fp in carra_files]

        success = 0
        failed = 0
        with tqdm(total=len(futures), desc="Building gapfilled HSA500m", unit="day") as pbar:
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
    print(f"Output GeoTIFFs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
