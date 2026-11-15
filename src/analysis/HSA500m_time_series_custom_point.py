"""
Extract and plot the HSA500m time series at a manually selected point.

Edit the coordinates and date range below before running. Coordinates are
latitude and longitude in WGS84; both date-range endpoints are inclusive.
"""

import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from pyproj import CRS, Transformer
from tqdm import tqdm


# Edit these settings before running.
INPUT_DIR = Path("/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff")
OUTPUT_DIR = Path("print")
LATITUDE = 72.25
LONGITUDE = -52.9
DATE_START = "2025-05-01"
DATE_END = "2025-07-01"

FILENAME = re.compile(r"hsa500m_gapfilled_(\d{8})\.tif")
ALBEDO_COLUMN = "albedo_hsa500m_gapfilled"
SCENARIO_COLUMN = "scenario_hsa500m_gapfilled"


def extract_time_series(input_dir, latitude, longitude, date_start, date_end):
    """Sample both product bands at a fixed point over the selected dates."""
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(f"Invalid latitude/longitude: {latitude}, {longitude}")
    if date_start > date_end:
        raise ValueError("DATE_START must be on or before DATE_END.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"HSA500m input directory not found: {input_dir}")

    paths = sorted(input_dir.glob("hsa500m_gapfilled_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No HSA500m daily GeoTIFFs found in {input_dir}")

    records = []
    seen_dates = set()
    for path in tqdm(paths, desc="Extracting HSA500m", unit="file"):
        match = FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Cannot parse date from filename: {path.name}")
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        if date < date_start or date > date_end:
            continue
        if date in seen_dates:
            raise ValueError(f"Duplicate HSA500m date: {date.date()}")
        seen_dates.add(date)

        with rio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")
            if src.count < 2:
                raise ValueError(f"Expected albedo and scenario bands in {path}")
            transformer = Transformer.from_crs(
                CRS.from_epsg(4326), src.crs, always_xy=True
            )
            x, y = transformer.transform(longitude, latitude)
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                raise ValueError(
                    f"Point ({latitude}, {longitude}) is outside raster {path}"
                )
            values = next(src.sample([(x, y)], indexes=[1, 2], masked=True))
            values = values.astype(np.float64).filled(np.nan)

        records.append({
            "time": date,
            ALBEDO_COLUMN: values[0],
            SCENARIO_COLUMN: values[1],
        })

    if not records:
        raise FileNotFoundError(
            f"No HSA500m daily GeoTIFFs found between "
            f"{date_start.date()} and {date_end.date()} in {input_dir}"
        )
    return pd.DataFrame.from_records(records).sort_values("time").reset_index(drop=True)


def plot_time_series(data, latitude, longitude, date_start, date_end, output_dir):
    """Plot daily albedo, leaving gaps where a daily raster is missing."""
    daily = data.set_index("time").asfreq("D")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily.index, daily[ALBEDO_COLUMN], linewidth=1)
    ax.set(
        title=f"HSA500m at ({latitude:.6f}, {longitude:.6f})",
        xlabel="Date",
        ylabel="Albedo",
        xlim=(date_start, date_end),
        ylim=(0.2, 0.95),
    )
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()

    point_suffix = f"{latitude:.4f}_{longitude:.4f}".replace("-", "m")
    date_suffix = f"{date_start:%Y%m%d}_{date_end:%Y%m%d}"
    output_stem = f"HSA500m_point_{point_suffix}_{date_suffix}_time_series"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{output_stem}.{suffix}", dpi=300)
    plt.close(fig)


def main():
    date_start = pd.to_datetime(DATE_START, format="%Y-%m-%d")
    date_end = pd.to_datetime(DATE_END, format="%Y-%m-%d")
    data = extract_time_series(
        INPUT_DIR, LATITUDE, LONGITUDE, date_start, date_end
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    point_suffix = f"{LATITUDE:.4f}_{LONGITUDE:.4f}".replace("-", "m")
    date_suffix = f"{date_start:%Y%m%d}_{date_end:%Y%m%d}"
    output_stem = f"HSA500m_point_{point_suffix}_{date_suffix}_time_series"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    data.to_csv(csv_path, index=False)
    plot_time_series(
        data, LATITUDE, LONGITUDE, date_start, date_end, OUTPUT_DIR
    )
    print(
        f"Wrote {len(data)} daily records for "
        f"({LATITUDE:.6f}, {LONGITUDE:.6f}) to {csv_path}"
    )


if __name__ == "__main__":
    main()
