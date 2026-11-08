"""
Extract and plot the full HSA500m time series at one fixed AWS location.

The station coordinates come from one year of the annual-drift table, not
from the AWS observation period. The same location is sampled for every
available daily GeoTIFF. Boxplots contain daily values from the selected
calendar months, grouped by calendar year.
"""
#%%
import calendar
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
from pyproj import CRS, Transformer
from tqdm import tqdm


# Edit these settings before running.
AWS_PATH = Path("/data_3/shunan_2/AU/hsa500m/PROMICE/aws_annual_drift.csv")
INPUT_DIR = Path("/data_3/shunan_2/AU/hsa500m/hsa500m_geotiff")
OUTPUT_DIR = Path("print")
STATION = "KAN_M"
COORDINATE_YEAR = None  # None selects the latest year in the station table.
MONTHS = (6, 7, 8)  # Use any distinct calendar months; JJA is (6, 7, 8).

FILENAME = re.compile(r"hsa500m_gapfilled_(\d{8})\.tif")
ALBEDO_COLUMN = "albedo_hsa500m_gapfilled"
SCENARIO_COLUMN = "scenario_hsa500m_gapfilled"


def station_location(aws_path, station, coordinate_year):
    """Return one fixed latitude and longitude and the year they came from."""
    stations = pd.read_csv(aws_path)
    rows = stations.loc[stations["aws"] == station].copy()
    if rows.empty:
        raise ValueError(f"Station {station!r} not found in {aws_path}")
    rows["year"] = pd.to_numeric(rows["year"], errors="raise")
    if rows["year"].isna().any():
        raise ValueError(f"Missing coordinate year for station {station!r}")
    if coordinate_year is None:
        coordinate_year = rows["year"].max()
    rows = rows.loc[rows["year"] == coordinate_year]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one coordinate record for {station!r} in "
            f"{coordinate_year}; found {len(rows)}"
        )
    lat = float(rows.iloc[0]["lat"])
    lon = float(rows.iloc[0]["lon"])
    if not np.isfinite(lat) or not np.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coordinates for {station!r} in {coordinate_year}: {lat}, {lon}")
    return lat, lon, int(coordinate_year)


def extract_time_series(input_dir, station, lat, lon):
    """Sample both product bands at the fixed point for every available day."""
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
        if date in seen_dates:
            raise ValueError(f"Duplicate HSA500m date: {date.date()}")
        seen_dates.add(date)
        with rio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")
            if src.count < 2:
                raise ValueError(f"Expected albedo and scenario bands in {path}")
            transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                raise ValueError(f"Station {station!r} is outside raster {path}")
            values = next(src.sample([(x, y)], indexes=[1, 2], masked=True))
            values = values.astype(np.float64).filled(np.nan)
        records.append({
            "aws": station,
            "time": date,
            ALBEDO_COLUMN: values[0],
            SCENARIO_COLUMN: values[1],
        })
    return pd.DataFrame.from_records(records).sort_values("time").reset_index(drop=True)


def plot_time_series(df, station, output_dir):
    """Plot all available dates, leaving gaps where a daily raster is missing."""
    daily = df.set_index("time").asfreq("D")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily.index, daily[ALBEDO_COLUMN], linewidth=1)
    ax.set(xlabel="Date", ylabel="Albedo", ylim=(0, 1)) #title=f"HSA500m at {station}", 
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"HSA500m_{station}_time_series.{suffix}", dpi=300)
    plt.close(fig)


def plot_month_boxplots(df, station, months, output_dir):
    """Plot daily albedo distributions for the selected months in each year."""
    if not months or any(not isinstance(month, int) or isinstance(month, bool) or month < 1 or month > 12 for month in months):
        raise ValueError("MONTHS must contain integers from 1 through 12.")
    if len(set(months)) != len(months):
        raise ValueError("MONTHS must not contain duplicates.")
    selected = df.loc[df["time"].dt.month.isin(months)].dropna(subset=[ALBEDO_COLUMN]).copy()
    if selected.empty:
        raise ValueError("No valid albedo values found for the selected months.")
    selected["year"] = selected["time"].dt.year
    month_label = ", ".join(calendar.month_abbr[month] for month in sorted(months))
    month_suffix = "_".join(f"{month:02d}" for month in sorted(months))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=selected, x="year", y=ALBEDO_COLUMN, ax=ax)
    ax.set(
        # title=f"HSA500m at {station}: {month_label}",
        # xlabel="Calendar year",
        ylabel="Albedo",
        ylim=(0, 1),
    )
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"HSA500m_{station}_boxplot_{month_suffix}.{suffix}", dpi=300)
    plt.close(fig)


def main():
    sns.set_theme(font_scale=1.5, style="darkgrid")
    lat, lon, year = station_location(AWS_PATH, STATION, COORDINATE_YEAR)
    print(f"Sampling {STATION} at {lat:.6f}, {lon:.6f} (coordinates from {year})")
    df = extract_time_series(INPUT_DIR, STATION, lat, lon)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"HSA500m_{STATION}_time_series.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} daily records to {csv_path}")
    plot_time_series(df, STATION, OUTPUT_DIR)
    plot_month_boxplots(df, STATION, MONTHS, OUTPUT_DIR)


if __name__ == "__main__":
    main()

# %%
