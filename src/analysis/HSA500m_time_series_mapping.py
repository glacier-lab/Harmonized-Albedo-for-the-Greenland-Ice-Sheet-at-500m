"""
Map yearly means of selected monthly HSA500m albedo GeoTIFFs.

Each selected month has equal weight. At each pixel, missing monthly values are
excluded; a pixel with no valid selected months remains transparent. Every
selected monthly file must exist, and all input rasters must share a grid.
"""
#%%
import calendar
import math
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import seaborn as sns
from matplotlib import colors
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.plot import show
sns.set_theme(font_scale=1.5, style="white")

# -----------------------------------------------------------------------------
# Configuration: edit these values before running
# -----------------------------------------------------------------------------
INPUT_DIR = Path("/data_3/shunan_2/AU/hsa500m/hsa500m_monthly")
MONTHS = (6, 7, 8)  # Calendar months to average, e.g. June-August.
YEARS = tuple(range(2000, 2026))  # Set to None to use all available years.
COLUMNS = 5
NUM_WORKERS = 30  # Concurrent years; increase only if RAM and disk throughput allow.
OUTPUT_PATH = Path(
    "/data/shunan/github/Harmonized-Albedo-for-the-Greenland-Ice-Sheet-at-500m/"
    "print/HSA500m_time_series_mapping.png"
)  # Both PNG and PDF are saved using this filename stem.

# Custom color palette from HSA500m_mapping.py
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

MONTHLY_FILENAME = re.compile(r"hsa500m_monthly_(\d{4})(\d{2})\.tif")


def collect_year_files(input_dir, months, years=None):
    """Return all required monthly files for each selected year."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Monthly input directory not found: {input_dir}")
    if not months or len(set(months)) != len(months) or any(m < 1 or m > 12 for m in months):
        raise ValueError("Months must be distinct integers from 1 through 12.")
    if years is not None and (not years or len(set(years)) != len(years)):
        raise ValueError("Years must be a non-empty list of distinct years.")

    if years is None:
        years = sorted({
            int(match.group(1))
            for path in input_dir.iterdir()
            if (match := MONTHLY_FILENAME.fullmatch(path.name))
            and int(match.group(2)) in months
        })
    if not years:
        raise FileNotFoundError(f"No monthly files found for the selected months in {input_dir}")

    result = {}
    missing = []
    for year in sorted(years):
        paths = [input_dir / f"hsa500m_monthly_{year:04d}{month:02d}.tif" for month in sorted(months)]
        missing.extend(path for path in paths if not path.is_file())
        result[year] = paths
    if missing:
        raise FileNotFoundError("Missing selected monthly files:\n" + "\n".join(map(str, missing)))
    return result


def monthly_mean(paths, reference_grid=None):
    """Return the pixel-wise mean and (height, width, transform, CRS) grid."""
    total = count = None
    for path in paths:
        with rio.open(path) as src:
            grid = (src.height, src.width, src.transform, src.crs)
            if src.crs is None or not src.crs.is_projected:
                raise ValueError(f"Raster must have a projected CRS in metres: {path}")
            if not math.isclose(src.crs.linear_units_factor[1], 1.0):
                raise ValueError(f"Raster CRS must use metres for the scale bar: {path}")
            if reference_grid is None:
                reference_grid = grid
            elif grid != reference_grid:
                raise ValueError(f"Raster grid does not match the other monthly files: {path}")

            data = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            valid = np.isfinite(data)
            if total is None:
                total = np.zeros(data.shape, dtype=np.float64)
                count = np.zeros(data.shape, dtype=np.uint32)
            total[valid] += data[valid]
            count[valid] += 1

    if total is None or count is None:
        raise ValueError("At least one monthly file is required.")
    mean = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=mean, where=count > 0)
    if not np.any(count):
        raise ValueError(f"No valid albedo pixels in the selected files: {paths[0]}")
    return mean, reference_grid


def panel_letter(index):
    """Return spreadsheet-style lowercase panel labels, including after z."""
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("a") + remainder) + label
    return label


def plot_time_series(year_files, months, output, columns=4, num_workers=4):
    """Plot and save yearly maps in PNG and PDF with a common color scale."""
    if not year_files:
        raise ValueError("At least one year is required.")
    if columns < 1:
        raise ValueError("The number of columns must be positive.")
    if num_workers < 1:
        raise ValueError("The number of workers must be positive.")
    columns = min(columns, len(year_files))
    rows = math.ceil(len(year_files) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 5.6 * rows), squeeze=False)
    fig.subplots_adjust(left=0.03, right=0.90, top=0.92, bottom=0.05, wspace=0.08, hspace=0.15)
    # fig.suptitle("Mean albedo: " + ", ".join(calendar.month_abbr[m] for m in sorted(months)))
    cmap = colors.ListedColormap(BLUE_FLUORITE)
    reference_grid = None

    try:
        with ThreadPoolExecutor(max_workers=min(num_workers, len(year_files))) as executor:
            remaining = iter(sorted(year_files.items()))
            pending = deque()
            for _ in range(min(num_workers, len(year_files))):
                year, paths = next(remaining)
                pending.append((year, paths, executor.submit(monthly_mean, paths)))

            index = 0
            while pending:
                year, paths, future = pending.popleft()
                mean, grid = future.result()
                if reference_grid is None:
                    reference_grid = grid
                elif grid != reference_grid:
                    raise ValueError(f"Raster grid does not match the other monthly files: {paths[0]}")

                next_year = next(remaining, None)
                if next_year is not None:
                    next_label, next_paths = next_year
                    pending.append((next_label, next_paths, executor.submit(monthly_mean, next_paths)))

                ax = axes.flat[index]
                _, _, transform, _ = grid
                show(mean, transform=transform, ax=ax, cmap=cmap, vmin=0, vmax=1)
                ax.add_artist(ScaleBar(
                    dx=1.0, units="m", fixed_value=300, fixed_units="km",
                    location="lower right", frameon=False, color="black",
                ))
                ax.axis("off")
                ax.text(0.02, 0.1, f"({panel_letter(index)})", transform=ax.transAxes,
                        va="top", ha="left", color="black")
                ax.set_title(str(year))
                print(f"Prepared {year} ({index + 1}/{len(year_files)})", flush=True)
                index += 1

        for ax in list(axes.flat)[len(year_files):]:
            ax.set_visible(False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=0, vmax=1))
        colorbar = fig.colorbar(mappable, ax=list(axes.flat), orientation="vertical",
                                fraction=0.03, pad=0.02)
        colorbar.set_label("Albedo")
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def main():
    files = collect_year_files(INPUT_DIR, MONTHS, YEARS)
    plot_time_series(files, MONTHS, OUTPUT_PATH, COLUMNS, NUM_WORKERS)
    print(f"Saved {OUTPUT_PATH.with_suffix('.png')} and {OUTPUT_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

# %%
