#%%
import cdsapi
import os
from pathlib import Path
import calendar

#%%
# Configuration
output_dir = "/data_3/shunan_2/AU/hsa500m/Yukihiko"  
os.makedirs(output_dir, exist_ok=True)

#%%

dataset = "reanalysis-carra-single-levels"

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Generate all months from 2000 to 2025
years = [str(year) for year in range(2000, 2026)]
months = [f"{month:02d}" for month in range(1, 13)]

client = cdsapi.Client()

for year in years:
    for month in months:
        print(f"Downloading data for {year}-{month}...")
        
        # Get the number of days in this month
        num_days = calendar.monthrange(int(year), int(month))[1]
        days = [f"{day:02d}" for day in range(1, num_days + 1)]
        
        request = {
            "domain": "west_domain",
            "level_type": "surface_or_atmosphere",
            "variable": [
                "surface_solar_radiation_downwards",
                "time_integrated_surface_direct_short_wave_radiation_flux"
            ],
            "product_type": "forecast",
            "time": ["00:00"],
            "leadtime_hour": [
                "3",
                "6",
                "9",
                "12",
                "15",
                "18",
                "21",
                "24"
            ],
            "year": [year],
            "month": [month],
            "day": days,
            "data_format": "netcdf"
        }
        
        filename = f"carra_{year}_{month}.nc"
        filepath = os.path.join(output_dir, filename)
        
        try:
            client.retrieve(dataset, request).download(filepath)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {year}-{month}: {e}")