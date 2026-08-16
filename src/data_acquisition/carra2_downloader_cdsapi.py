import calendar
import os
from pathlib import Path

import cdsapi


# Configuration
output_dir = "/data_3/shunan_2/AU/hsa500m/Yukihiko/CARRA2"
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
            "variable": ["total_cloud_cover"],
            "product_type": "analysis_based",
            "year": [year],
            "month": [month],
            "day": days,
            "data_format": "netcdf",
            "area": [83.7553561747887, -74.08813799635422, 58.59235996703347, -10.015872371354217],
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