"""
Extract and plot the full HSA500m time series at fixed AWS locations.

The station coordinates come from one year of the annual-drift table, not
from the AWS observation period. The same location is sampled for every
available daily GeoTIFF. Boxplots contain daily values from the selected
calendar months, grouped by calendar year. Annual anomalies compare each
selected-month mean with the mean of the available yearly means.
"""
#%%
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
STATIONS = ("KAN_M",)  # Add station names for additional panels, e.g. ("NEM", "KAN_M").
COORDINATE_YEAR = None  # None selects the latest year in the station table.
MONTHS = (5, 6, 7, 8, 9)  # Use any distinct calendar months; JJA is (6, 7, 8).

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


def panel_label(index):
    """Return a lowercase panel label, including labels after z."""
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("a") + remainder) + label
    return label


def label_panel(ax, index, station):
    ax.text(
        0.02, 0.05, f"({panel_label(index)}) {station}",
        transform=ax.transAxes, ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )


def plot_time_series(station_data, output_dir):
    """Plot all available dates, leaving gaps where a daily raster is missing."""
    fig, axes = plt.subplots(
        len(station_data), 1, figsize=(12, 6 * len(station_data)),
        sharex=True, squeeze=False,
    )
    for index, (station, df) in enumerate(station_data.items()):
        ax = axes[index, 0]
        daily = df.set_index("time").asfreq("D")
        ax.plot(daily.index, daily[ALBEDO_COLUMN], linewidth=1)
        ax.set(ylabel="Albedo", ylim=(0.2, 0.95))
        label_panel(ax, index, station)
    axes[-1, 0].set_xlabel("Date")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    station_suffix = "_".join(station_data)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"HSA500m_{station_suffix}_time_series.{suffix}", dpi=300)
    plt.close(fig)


def select_months(station_data, months):
    """Return valid albedo observations for the selected calendar months."""
    if not months or any(not isinstance(month, int) or isinstance(month, bool) or month < 1 or month > 12 for month in months):
        raise ValueError("MONTHS must contain integers from 1 through 12.")
    if len(set(months)) != len(months):
        raise ValueError("MONTHS must not contain duplicates.")
    selected_data = {}
    for station, df in station_data.items():
        selected = df.loc[df["time"].dt.month.isin(months)].dropna(subset=[ALBEDO_COLUMN]).copy()
        if selected.empty:
            raise ValueError(f"No valid albedo values for {station} in the selected months.")
        selected["year"] = selected["time"].dt.year
        selected_data[station] = selected
    return selected_data


def plot_month_boxplots(station_data, months, output_dir):
    """Plot yearly albedo distributions and annual anomalies for each station."""
    selected_data = select_months(station_data, months)
    years = sorted({year for df in selected_data.values() for year in df["year"]})
    month_suffix = "_".join(f"{month:02d}" for month in sorted(months))
    fig, axes = plt.subplots(
        2 * len(selected_data), 1, figsize=(12, 9 * len(selected_data)),
        sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [2, 1] * len(selected_data)},
    )
    annual_records = []
    for index, (station, selected) in enumerate(selected_data.items()):
        annual = selected.groupby("year")[ALBEDO_COLUMN].agg(["mean", "count"])
        annual["anomaly"] = annual["mean"] - annual["mean"].mean()
        annual.index.name = "year"
        annual_records.append(annual.assign(aws=station).reset_index())

        box_ax = axes[2 * index, 0]
        sns.boxplot(data=selected, x="year", y=ALBEDO_COLUMN, order=years, ax=box_ax)
        box_ax.set(xlabel="", ylabel="Albedo", ylim=(0.2, 0.95))
        label_panel(box_ax, 2 * index, station)

        anomaly_ax = axes[2 * index + 1, 0]
        aligned = annual["anomaly"].reindex(years)
        positions = np.arange(len(years))
        valid = aligned.notna().to_numpy()
        values = aligned.to_numpy()[valid]
        colors = np.where(values >= 0, "#3274a1", "#c44e52")
        anomaly_ax.bar(positions[valid], values, color=colors, width=0.8)
        anomaly_ax.axhline(0, color="black", linewidth=0.8)
        anomaly_ax.set(xlabel="", ylabel="Albedo anomaly")
        label_panel(anomaly_ax, 2 * index + 1, station)
    axes[-1, 0].set_xlabel("Calendar year")
    axes[-1, 0].tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    station_suffix = "_".join(station_data)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"HSA500m_{station_suffix}_boxplot_{month_suffix}.{suffix}", dpi=300)
    plt.close(fig)
    csv_path = output_dir / f"HSA500m_{station_suffix}_annual_anomaly_{month_suffix}.csv"
    pd.concat(annual_records, ignore_index=True).to_csv(csv_path, index=False)


def main():
    sns.set_theme(font_scale=1.5, style="darkgrid")
    if not STATIONS or len(set(STATIONS)) != len(STATIONS):
        raise ValueError("STATIONS must contain at least one station, without duplicates.")
    station_data = {}
    for station in STATIONS:
        lat, lon, year = station_location(AWS_PATH, station, COORDINATE_YEAR)
        print(f"Sampling {station} at {lat:.6f}, {lon:.6f} (coordinates from {year})")
        station_data[station] = extract_time_series(INPUT_DIR, station, lat, lon)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for station, df in station_data.items():
        csv_path = OUTPUT_DIR / f"HSA500m_{station}_time_series.csv"
        df.to_csv(csv_path, index=False)
        print(f"Wrote {len(df)} daily records to {csv_path}")
    plot_time_series(station_data, OUTPUT_DIR)
    plot_month_boxplots(station_data, MONTHS, OUTPUT_DIR)


if __name__ == "__main__":
    main()

# %%
