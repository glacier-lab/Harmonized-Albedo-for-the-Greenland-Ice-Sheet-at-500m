import calendar
import os
from pathlib import Path

import cdsapi


# Configuration
output_dir = "/data_3/shunan_2/AU/hsa500m/Arctic/CARRA2"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
dataset = "reanalysis-pan-carra-means"


# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)


# Generate all years, months, and days covered by the request
years = [str(year) for year in range(2000, 2026)]
months = [f"{month:02d}" for month in range(1, 13)]

client = cdsapi.Client()

for year in years:
    for month in months:
        print(f"Downloading CARRA2 data for {year}-{month}...")

        num_days = calendar.monthrange(int(year), int(month))[1]
        days = [f"{day:02d}" for day in range(1, num_days + 1)]

        request = {
            "time_aggregation": "daily",
            "level_type": "single_levels",
            "variable": [
                "surface_solar_radiation_downwards",
                "time_integrated_surface_direct_short_wave_radiation_flux"
            ],
            "product_type": "forecast_based",
            "year": [year],
            "month": [month],
            "day": days,
            "data_format": "netcdf"
        }

        filename = f"carra2_{year}_{month}.nc"
        filepath = os.path.join(output_dir, filename)

        attempt = 1
        while True:
            try:
                # Ensure each attempt writes a fresh file.
                if os.path.exists(filepath):
                    os.remove(filepath)

                print(f"Attempt {attempt}: requesting {filename}")
                client.retrieve(dataset, request).download(filepath)
                print(f"Successfully downloaded {filename}")
                break
            except Exception as exc:
                print(f"Attempt {attempt} failed for {year}-{month}: {exc}")
                attempt += 1
# import cdsapi

# dataset = "reanalysis-pan-carra-means"
# request = {
#     "time_aggregation": "daily",
#     "level_type": "single_levels",
#     "variable": [
#         "surface_solar_radiation_downwards",
#         "time_integrated_surface_direct_short_wave_radiation_flux"
#     ],
#     "product_type": "forecast_based",
#     "year": [
#         "2000", "2001", "2002",
#         "2003", "2004", "2005",
#         "2006", "2007", "2008",
#         "2009", "2010", "2011",
#         "2012", "2013", "2014",
#         "2015", "2016", "2017",
#         "2018", "2019", "2020",
#         "2021", "2022", "2023",
#         "2024", "2025"
#     ],
#     "month": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12"
#     ],
#     "day": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12",
#         "13", "14", "15",
#         "16", "17", "18",
#         "19", "20", "21",
#         "22", "23", "24",
#         "25", "26", "27",
#         "28", "29", "30",
#         "31"
#     ],
#     "data_format": "netcdf"
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()
