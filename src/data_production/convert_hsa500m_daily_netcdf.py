"""
Convert daily HSA500m GeoTIFF files to NetCDF.

Reads daily `hsa500m_gapfilled_YYYYMMDD.tif` files and writes one NetCDF file
per day. Band 1 is exported as `hsa500m_gapfilled`; if Band 2 exists, it is
exported as `scenario_map`.

Output filename: hsa500m_gapfilled_YYYYMMDD.nc
"""

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import rasterio as rio
import xarray as xr
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff"
OUTPUT_DIR = "/data_3/shunan_2/AU/hsa500m/hsa500m_netcdf"
OVERWRITE_EXISTING_FILES = True
NUM_WORKERS = 10


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
AUTHOR = "Shunan Feng"
EMAIL = "shunan.feng@envs.au.dk"
AFFILIATION_1 = (
    "Department of Environmental Science, Aarhus University, "
    "Frederiksborgvej 399, DK-4000 Roskilde, Denmark"
)
AFFILIATION_2 = "Arctic Research Cluster, Aarhus University, Denmark"


def parse_date_from_name(filename: str) -> Optional[str]:
    match = re.search(r"hsa500m_gapfilled_(\d{8})\.tif$", filename)
    if not match:
        return None
    return match.group(1)


def list_daily_files(input_dir: str) -> List[str]:
    pattern = os.path.join(input_dir, "hsa500m_gapfilled_*.tif")
    return sorted(glob.glob(pattern))


def build_xy_coords(transform, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    x = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x, y


def write_netcdf_with_fallback(ds: xr.Dataset, out_path: str) -> None:
    encoding = {
        "hsa500m_gapfilled": {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": np.nan}
    }
    if "scenario_map" in ds.variables:
        encoding["scenario_map"] = {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.nan,
        }

    try:
        ds.to_netcdf(out_path, encoding=encoding)
    except Exception:
        # Fallback for environments/backends that do not support compression.
        ds.to_netcdf(out_path)


def convert_single_file(file_path: str, output_dir: str) -> Tuple[bool, str]:
    filename = os.path.basename(file_path)
    date_tag = parse_date_from_name(filename)
    if date_tag is None:
        return False, f"{filename} -> skipped (unexpected filename pattern)"

    out_path = os.path.join(output_dir, f"hsa500m_gapfilled_{date_tag}.nc")
    if os.path.exists(out_path) and not OVERWRITE_EXISTING_FILES:
        return True, f"{date_tag} -> skipped (exists)"

    try:
        with rio.open(file_path) as src:
            hsa = src.read(1).astype(np.float32)
            hsa_nodata = src.nodata

            scenario = None
            if src.count >= 2:
                scenario = src.read(2).astype(np.float32)

            if hsa_nodata is not None and np.isfinite(hsa_nodata):
                hsa[hsa == hsa_nodata] = np.nan
                if scenario is not None:
                    scenario[scenario == hsa_nodata] = np.nan

            transform = src.transform
            width = src.width
            height = src.height
            crs = src.crs
            bounds = src.bounds

        x, y = build_xy_coords(transform, width, height)

        ds = xr.Dataset(
            data_vars={
                "hsa500m_gapfilled": (("y", "x"), hsa),
            },
            coords={
                "x": ("x", x),
                "y": ("y", y),
            },
            attrs={
                "title": "Harmonized Surface Albedo (HSA500m) Daily Gapfilled",
                "summary": "Daily harmonized albedo converted from GeoTIFF to NetCDF.",
                "Conventions": "CF-1.10",
                "source_geotiff": filename,
                "history": (
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    "Converted from GeoTIFF to NetCDF"
                ),
                "author": AUTHOR,
                "email": EMAIL,
                "affiliation": AFFILIATION_1,
                "affiliation_2": AFFILIATION_2,
                "creator_name": AUTHOR,
                "creator_email": EMAIL,
                "creator_institution": f"{AFFILIATION_1}; {AFFILIATION_2}",
                "institution": f"{AFFILIATION_1}; {AFFILIATION_2}",
                "geospatial_bounds": (
                    f"left={bounds.left}, bottom={bounds.bottom}, "
                    f"right={bounds.right}, top={bounds.top}"
                ),
            },
        )

        ds["x"].attrs.update({
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
            "units": "m",
        })
        ds["y"].attrs.update({
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
            "units": "m",
        })
        ds["hsa500m_gapfilled"].attrs.update({
            "long_name": "Daily gapfilled HSA500m albedo",
            "units": "1",
            "grid_mapping": "spatial_ref",
        })

        if scenario is not None:
            ds["scenario_map"] = (("y", "x"), scenario)
            ds["scenario_map"].attrs.update({
                "long_name": "Scenario map used in gapfilling",
                "units": "1",
                "grid_mapping": "spatial_ref",
            })

        crs_wkt = crs.to_wkt() if crs is not None else ""
        ds["spatial_ref"] = xr.DataArray(0)
        ds["spatial_ref"].attrs.update({
            "spatial_ref": crs_wkt,
            "crs_wkt": crs_wkt,
            "GeoTransform": (
                f"{transform.c} {transform.a} {transform.b} "
                f"{transform.f} {transform.d} {transform.e}"
            ),
        })

        write_netcdf_with_fallback(ds, out_path)

        valid_pixels = int(np.sum(np.isfinite(hsa)))
        return True, f"{date_tag} -> valid pixels={valid_pixels}"

    except Exception as exc:
        return False, f"{date_tag} -> failed: {type(exc).__name__}: {exc}"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    daily_files = list_daily_files(INPUT_DIR)
    if not daily_files:
        raise FileNotFoundError(f"No daily HSA500m files found in {INPUT_DIR}")

    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Files to process: {len(daily_files)}")
    print(f"Overwrite existing: {OVERWRITE_EXISTING_FILES}")
    print(f"Workers: {NUM_WORKERS}")

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(convert_single_file, fp, OUTPUT_DIR): fp for fp in daily_files}

        with tqdm(total=len(futures), desc="Converting daily HSA500m", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    ok, msg = future.result()
                except Exception as exc:
                    fp = futures[future]
                    ok, msg = False, f"{os.path.basename(fp)} -> failed: {type(exc).__name__}: {exc}"

                if ok:
                    success += 1
                else:
                    failed += 1
                tqdm.write(msg)
                pbar.update(1)

    print("Done.")
    print(f"Succeeded: {success}, Failed: {failed}")
    print(f"Output NetCDFs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
