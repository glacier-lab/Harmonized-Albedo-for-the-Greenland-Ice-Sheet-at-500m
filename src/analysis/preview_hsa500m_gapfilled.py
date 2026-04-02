"""
Examine daily HSA500m gapfilled GeoTIFF outputs.

Outputs:
1. Daily map figures with two panels:
   - Panel A: gapfilled albedo map (band 1)
   - Panel B: simplified sensor-availability map from scenario band (band 2)
2. Daily scenario pixel statistics CSV in wide format
    (one row per day, one column per simplified scenario group,
    while keeping CARRA 0 and -1 separate).

Notes:
- Orbital-drift scenario variants (e.g., mod10_2021, myd10_2023, mcd43a3_bluesky_2022)
  are collapsed to their base sensors for both map display and statistics.
- Scenario values are interpreted as:
  -1: CARRA fill where original CARRA >= cap
   0: CARRA fill where original CARRA < cap
  >0: satellite-calibrated scenario_id from calibration_coefficients.csv

Shunan Feng (shunan.feng@envs.au.dk)
"""
#%%
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib import colors
from matplotlib.patches import Patch
from rasterio.plot import show
from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns
sns.set_theme(font_scale=1.5)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HSA_GAPFILLED_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
CALIBRATION_CSV = "/data_3/shunan_2/AU/hsa500m/calibration/calibration_coefficients.csv"

OUTPUT_BASE_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_preview"
OUTPUT_FIG_DIR = os.path.join(OUTPUT_BASE_DIR, "daily_maps")
OUTPUT_CSV = os.path.join(OUTPUT_BASE_DIR, "daily_scenario_pixel_counts_simplified.csv")

DATE_REGEX = r"hsa500m_gapfilled_(\d{8})\.tif"

# Optional processing controls
DATE_START = None  # e.g., "2020-01-01"
DATE_END = None    # e.g., "2024-12-31"
MAX_FILES = None   # e.g., 30
NUM_WORKERS = 8


# Custom color palette provided by user.
BLUE_FLUORITE = [
    '#291b32', '#2a1b34', '#2b1b34', '#2d1c36', '#2f1c38', '#301c39', '#301d3a', '#321d3b', '#331d3d', '#351d3f',
    '#351e40', '#371e41', '#381e43', '#3a1e45', '#3b1f45', '#3c1f46', '#3e1f48', '#3f1f4a', '#401f4c', '#42204d',
    '#43204e', '#44204f', '#462051', '#472052', '#482054', '#4a2056', '#4a2157', '#4c2158', '#4e215a', '#4f215b',
    '#50215d', '#52215e', '#532160', '#552162', '#552263', '#562264', '#582265', '#592267', '#5b2268', '#5c226b',
    '#5e226c', '#5f226e', '#60226f', '#622271', '#632272', '#642274', '#662276', '#672277', '#692278', '#6a227a',
    '#6c227b', '#6e227d', '#6e237e', '#6f247f', '#702480', '#712581', '#722681', '#732683', '#742783', '#752884',
    '#762985', '#772987', '#792a87', '#792b88', '#7a2c89', '#7b2c8a', '#7c2d8a', '#7d2d8c', '#7e2e8d', '#7f2f8d',
    '#80308e', '#813190', '#823191', '#833292', '#843292', '#863393', '#863494', '#873595', '#893596', '#8a3697',
    '#8b3798', '#8b3899', '#8c389a', '#8e399b', '#8e3a9c', '#8f3b9c', '#8f3d9d', '#8f3e9e', '#903f9e', '#90419e',
    '#90439f', '#9044a0', '#9046a0', '#9047a1', '#9049a1', '#914aa2', '#914ca2', '#914ca3', '#914ea3', '#9150a4',
    '#9151a5', '#9153a5', '#9154a6', '#9156a6', '#9157a7', '#9258a7', '#9259a8', '#925aa8', '#925ba9', '#925da9',
    '#925faa', '#9260ab', '#9260ab', '#9263ac', '#9264ac', '#9265ad', '#9266ae', '#9268ae', '#9269ae', '#926aaf',
    '#926bb0', '#926cb0', '#926eb1', '#926fb1', '#9270b2', '#9271b2', '#9273b3', '#9274b3', '#9275b4', '#9277b5',
    '#9277b5', '#9278b6', '#927ab6', '#927bb7', '#927cb7', '#927eb8', '#927fb8', '#9280b9', '#9281ba', '#9282ba',
    '#9284bb', '#9285bb', '#9285bc', '#9187bc', '#9188bd', '#918abd', '#918bbe', '#918cbf', '#918dbf', '#918ec0',
    '#918fc0', '#9191c1', '#9092c2', '#9094c2', '#9094c2', '#9095c3', '#9096c3', '#8f99c4', '#8f9ac5', '#8f9ac5',
    '#8f9bc6', '#8f9cc6', '#8f9dc7', '#8e9fc8', '#8ea0c8', '#8ea2c9', '#8ea3c9', '#8da5ca', '#8da5ca', '#8da6cb',
    '#8da7cb', '#8ca9cc', '#8caacc', '#8caccd', '#8bacce', '#8badce', '#8baecf', '#8ab0d0', '#8ab2d0', '#8ab2d1',
    '#8ab4d1', '#89b4d1', '#89b5d2', '#89b7d2', '#88b8d3', '#88bad4', '#87bad4', '#87bbd5', '#86bdd6', '#86bed6',
    '#86c0d7', '#85c0d7', '#85c1d8', '#84c3d8', '#84c4d9', '#83c5d9', '#83c6da', '#82c8da', '#82c8db', '#81cadc',
    '#81cbdc', '#80ccdd', '#81cddd', '#84cfdd', '#85cfdd', '#87d0dd', '#8ad0de', '#8dd1de', '#8fd2de', '#90d2de',
    '#92d4de', '#95d5de', '#97d5de', '#98d6de', '#9bd7de', '#9dd7df', '#a0d8df', '#a1d9df', '#a2dadf', '#a5dadf',
    '#a7dbdf', '#aadcdf', '#abdddf', '#acdde0', '#afdfe0', '#b1dfe0', '#b3e0e0', '#b4e1e0', '#b7e2e0', '#bae2e1',
    '#bae3e1', '#bee3e2', '#c0e4e3', '#c1e5e3', '#c4e6e3', '#c6e6e4', '#c8e7e4', '#cbe7e5', '#cde8e5', '#cee9e6',
    '#d2e9e7', '#d3eae7', '#d5eae7', '#d8ebe8', '#d9ece8', '#dcece9', '#deedea', '#dfeeea', '#e2eeea', '#e5efeb',
    '#e6f0eb', '#e9f0ec', '#ebf1ed', '#ecf2ed', '#eff3ee', '#f1f3ee'
]


# Worker-process globals (populated by init_worker).
_SCENARIO_ID_TO_LABEL: Optional[Dict[int, str]] = None
_LABEL_TO_CODE: Optional[Dict[str, int]] = None
_CODE_TO_LABEL: Optional[Dict[int, str]] = None


def init_worker(calibration_csv: str) -> None:
    global _SCENARIO_ID_TO_LABEL, _LABEL_TO_CODE, _CODE_TO_LABEL
    matplotlib.use("Agg")
    sid_to_label = load_scenario_id_to_label(calibration_csv)
    lbl_to_code, code_to_lbl = build_sensor_combo_codes(sid_to_label)
    _SCENARIO_ID_TO_LABEL = sid_to_label
    _LABEL_TO_CODE = lbl_to_code
    _CODE_TO_LABEL = code_to_lbl


def process_single_file(
    tif_path: str, output_fig_dir: str
) -> Tuple[bool, str, Optional[Dict]]:
    if _SCENARIO_ID_TO_LABEL is None or _LABEL_TO_CODE is None or _CODE_TO_LABEL is None:
        return False, "Worker not initialized", None

    file_name = os.path.basename(tif_path)
    ts = parse_date_from_gapfilled_name(file_name)
    if ts is None:
        return False, f"Cannot parse date from {file_name}", None

    date_str = ts.strftime("%Y-%m-%d")
    date_compact = ts.strftime("%Y%m%d")

    with rio.open(tif_path) as src:
        albedo = src.read(1).astype(np.float32)
        scenario = src.read(2).astype(np.float32)
        transform = src.transform
        if src.nodata is not None and np.isfinite(src.nodata):
            albedo[albedo == src.nodata] = np.nan
            scenario[scenario == src.nodata] = np.nan

    sensor_code_map = scenario_to_sensor_code(scenario, _SCENARIO_ID_TO_LABEL, _LABEL_TO_CODE)

    valid_sensor_codes = sensor_code_map[np.isfinite(sensor_code_map)].astype(int)
    daily_row: Dict[str, object] = {"date": date_str}
    if valid_sensor_codes.size > 0:
        uniq, counts = np.unique(valid_sensor_codes, return_counts=True)
        for code, count in zip(uniq, counts):
            group_name = _CODE_TO_LABEL.get(int(code), f"code_{int(code)}")
            daily_row[group_name] = int(count)

    out_png = os.path.join(output_fig_dir, f"hsa500m_gapfilled_map_{date_compact}.png")
    plot_daily_maps(
        tif_path=tif_path,
        date_str=date_str,
        albedo=albedo,
        sensor_code_map=sensor_code_map,
        transform=transform,
        code_to_label=_CODE_TO_LABEL,
        out_png=out_png,
    )
    return True, date_str, daily_row


def parse_date_from_gapfilled_name(file_name: str) -> Optional[pd.Timestamp]:
    m = re.search(DATE_REGEX, file_name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def simplify_sensor_name(sensor: str) -> str:
    if sensor.startswith("mod10_"):
        return "mod10a1"
    if sensor.startswith("myd10_"):
        return "myd10a1"
    if sensor.startswith("mcd43a3_bluesky_"):
        return "mcd43a3"
    if sensor == "mcd43a3_bluesky":
        return "mcd43a3"
    if sensor == "viirs_vj143ma3_bluesky":
        return "vj143ma3"
    if sensor == "viirs_vnp43ma3_bluesky":
        return "vnp43ma3"
    return sensor


def load_scenario_id_to_label(calibration_csv: str) -> Dict[int, str]:
    coeff = pd.read_csv(calibration_csv)

    known_non_sensor_cols = {
        "scenario_id", "scenario", "sensors", "n_sensors", "n_train", "n_test", "n_total",
        "train_r_squared", "train_rmse", "train_mae", "train_bias",
        "test_calib_r_squared", "test_calib_rmse", "test_calib_mae", "test_calib_bias",
        "train_slope", "train_intercept", "test_calib_slope", "test_calib_intercept",
        "slope", "intercept",
    }
    sensor_cols = [c for c in coeff.columns if c not in known_non_sensor_cols]

    scenario_map: Dict[int, str] = {}
    for _, row in coeff.iterrows():
        sid = int(row["scenario_id"])
        active_sensors: List[str] = []
        for col in sensor_cols:
            if int(row[col]) == 1:
                active_sensors.append(simplify_sensor_name(col))
        active_sensors = sorted(set(active_sensors))
        scenario_map[sid] = "+".join(active_sensors) if active_sensors else "unknown"

    return scenario_map


def build_sensor_combo_codes(scenario_id_to_label: Dict[int, str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({v for v in scenario_id_to_label.values() if v not in {"unknown"}})

    # Reserve negative/zero for CARRA classes.
    label_to_code: Dict[str, int] = {
        "carra_capflag": -1,
        "carra_calibrated": 0,
    }
    next_code = 1
    for label in labels:
        label_to_code[label] = next_code
        next_code += 1

    code_to_label = {v: k for k, v in label_to_code.items()}
    return label_to_code, code_to_label


def scenario_to_sensor_code(
    scenario: np.ndarray,
    scenario_id_to_label: Dict[int, str],
    label_to_code: Dict[str, int],
) -> np.ndarray:
    code_map = np.full(scenario.shape, np.nan, dtype=np.float32)

    # Keep CARRA classes separate.
    code_map[scenario == -1] = -1
    code_map[scenario == 0] = 0

    valid_ids = np.unique(scenario[np.isfinite(scenario)])
    valid_ids = valid_ids[valid_ids > 0]
    for sid in valid_ids.astype(int):
        label = scenario_id_to_label.get(sid, "unknown")
        if label == "unknown":
            continue
        code = label_to_code[label]
        code_map[scenario == sid] = float(code)

    return code_map


def make_sensor_cmap(codes_present: List[int]) -> Tuple[colors.ListedColormap, colors.BoundaryNorm, List[str], List[int]]:
    # Deterministic colors for CARRA categories plus dynamic colors for sensor combos.
    special_colors = {
        -1: "#5e3c99",  # CARRA capflag
        0: "#b2abd2",   # CARRA calibrated (<cap)
    }

    combo_codes = [c for c in codes_present if c > 0]
    combo_palette = plt.get_cmap("tab20", max(len(combo_codes), 1))

    ordered_codes = sorted(codes_present)
    color_list: List[str] = []
    for c in ordered_codes:
        if c in special_colors:
            color_list.append(special_colors[c])
        else:
            idx = combo_codes.index(c) if c in combo_codes else 0
            color_list.append(colors.to_hex(combo_palette(idx)))

    cmap = colors.ListedColormap(color_list)
    boundaries = np.arange(len(ordered_codes) + 1) - 0.5
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, color_list, ordered_codes


def plot_daily_maps(
    tif_path: str,
    date_str: str,
    albedo: np.ndarray,
    sensor_code_map: np.ndarray,
    transform,
    code_to_label: Dict[int, str],
    out_png: str,
) -> None:
    albedo_cmap = colors.ListedColormap(BLUE_FLUORITE)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: albedo map using rasterio.plot.show
    show(albedo, transform=transform, ax=axes[0], cmap=albedo_cmap, vmin=0, vmax=1)
    axes[0].set_title(f"a) {date_str}")
    # axes[0].set_xlabel("X")
    # axes[0].set_ylabel("Y")
    axes[0].set_axis_off()

    # Scale bar: pixel size from the rasterio transform (projected metres).
    pixel_size_m = abs(transform.a)  # transform.a is x pixel width in map units
    scalebar = ScaleBar(
        dx=1.0,
        units="m",
        fixed_value=300,
        fixed_units="km",
        location="lower right",
        frameon=False,
        color="black",
    )
    axes[0].add_artist(scalebar)

    sm = plt.cm.ScalarMappable(cmap=albedo_cmap, norm=colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Albedo")

    # Panel B: simplified available-sensor map
    present = sorted({int(v) for v in np.unique(sensor_code_map[np.isfinite(sensor_code_map)])})
    if len(present) == 0:
        present = [0]
        sensor_code_map = np.where(np.isfinite(albedo), 0, np.nan).astype(np.float32)

    sensor_cmap, sensor_norm, color_list, ordered_codes = make_sensor_cmap(present)

    # Map integer codes to compact index for BoundaryNorm.
    idx_map = np.full(sensor_code_map.shape, np.nan, dtype=np.float32)
    for idx, code in enumerate(ordered_codes):
        idx_map[sensor_code_map == code] = float(idx)

    show(idx_map, transform=transform, ax=axes[1], cmap=sensor_cmap, norm=sensor_norm)
    axes[1].set_title("b) QA band")
    # axes[1].set_xlabel("X")
    # axes[1].set_ylabel("Y")
    axes[1].set_axis_off()

    legend_handles: List[Patch] = []
    for color_hex, code in zip(color_list, ordered_codes):
        label = code_to_label.get(code, f"code_{code}")
        legend_handles.append(Patch(facecolor=color_hex, edgecolor="none", label=label))
    axes[1].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                   borderaxespad=0, frameon=True, title="Sensors")

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(HSA_GAPFILLED_DIR, "hsa500m_gapfilled_*.tif")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No gapfilled files found in {HSA_GAPFILLED_DIR}")

    # Date filtering
    files = []
    start_ts = pd.to_datetime(DATE_START) if DATE_START else None
    end_ts = pd.to_datetime(DATE_END) if DATE_END else None
    for fp in all_files:
        ts = parse_date_from_gapfilled_name(os.path.basename(fp))
        if ts is None:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        files.append(fp)

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    if len(files) == 0:
        raise ValueError("No files left after date/MAX_FILES filtering")

    print(f"Input files selected: {len(files)}")
    print(f"Figure output: {OUTPUT_FIG_DIR}")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"Workers: {NUM_WORKERS}")

    stats_rows: List[Dict] = []

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker,
        initargs=(CALIBRATION_CSV,),
    ) as executor:
        futures = [executor.submit(process_single_file, fp, OUTPUT_FIG_DIR) for fp in files]

        with tqdm(total=len(futures), desc="Processing daily gapfilled maps", unit="day") as pbar:
            for future in as_completed(futures):
                try:
                    ok, msg, daily_row = future.result()
                except Exception as exc:
                    ok, msg, daily_row = False, f"Worker failed: {type(exc).__name__}: {exc}", None
                if ok and daily_row is not None:
                    stats_rows.append(daily_row)
                else:
                    tqdm.write(msg)
                pbar.update(1)

    stats_df = pd.DataFrame(stats_rows)
    if stats_df.empty:
        print("No valid scenario pixels found; writing empty CSV header.")
        stats_df = pd.DataFrame(columns=["date"])
    else:
        # Fill missing groups with 0 and keep integer counts.
        group_cols = [c for c in stats_df.columns if c != "date"]
        if group_cols:
            stats_df[group_cols] = stats_df[group_cols].fillna(0).astype(np.int64)

            # Keep CARRA groups first, then other groups alphabetically.
            ordered_group_cols = []
            for special in ["carra_capflag", "carra_calibrated"]:
                if special in group_cols:
                    ordered_group_cols.append(special)
            ordered_group_cols.extend(sorted([c for c in group_cols if c not in ordered_group_cols]))
            stats_df = stats_df[["date"] + ordered_group_cols]

    stats_df = stats_df.sort_values(["date"]).reset_index(drop=True)
    stats_df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")
    print(f"Saved figures: {OUTPUT_FIG_DIR}")
    print(f"Saved scenario stats: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
