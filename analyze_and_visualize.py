#!/usr/bin/env python3
"""
ECMWF 100m WIND ANALYSIS + VISUALIZATION
Part 2 of 2
----------------------------------------
This script:
  - Loads processed 0.01° tiles
  - Extracts Khavda wind time series
  - Generates 100-m wind speed/direction maps
  - Verifies NetCDF tile structure
Run ONLY after download_and_process.py has completed.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PROCESSED_DIR = "ecmwf_wind_100m_processed"
FORECAST_EXPORT_DIR = "khavda_forecasts"
VIS_DIR = "khavda_wind_maps"

KHAVDA_LAT = 24.1132
KHAVDA_LON = 69.3669
BUFFER_DEG = 2.0

Path(FORECAST_EXPORT_DIR).mkdir(exist_ok=True)
Path(VIS_DIR).mkdir(exist_ok=True)

# ============================================================================
# TIME SERIES EXTRACTION
# ============================================================================

def extract_timeseries(run_name):
    base = Path(PROCESSED_DIR)/run_name
    forecast_dirs = sorted([d for d in base.iterdir() if d.is_dir()])

    data = []
    for fd in tqdm(forecast_dirs, desc="Extracting"):
        tile_fn = fd/"tiles"/"tile_lat20_lon60.nc"
        if not tile_fn.exists():
            continue

        ds = xr.open_dataset(tile_fn)

        p = ds.sel(latitude=KHAVDA_LAT, longitude=KHAVDA_LON, method="nearest")
        fh = int(fd.name[1:])
        data.append({
            "forecast_hour": fh,
            "valid_time": pd.to_datetime(p.time.values),
            "windspeed_100m": float(p["windspeed_100m"].values),
            "winddir_100m": float(p["winddir_100m"].values),
            "100u": float(p["100u"].values),
            "100v": float(p["100v"].values)
        })
        ds.close()

    df = pd.DataFrame(data).sort_values("forecast_hour")
    outfile = Path(FORECAST_EXPORT_DIR)/f"khavda_timeseries_{run_name}.csv"
    df.to_csv(outfile, index=False)

    print(f"Saved {outfile}")
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize(run_name, fh="f000"):
    tile_dir = Path(PROCESSED_DIR)/run_name/fh/"tiles"
    tile_fn = tile_dir/"tile_lat20_lon60.nc"
    
    if not tile_fn.exists():
        print(f"No tile for {fh}")
        return

    ds = xr.open_dataset(tile_fn)
    u = ds["100u"].values
    v = ds["100v"].values
    ws = ds["windspeed_100m"].values
    lats = ds.latitude.values
    lons = ds.longitude.values

    fig, ax = plt.subplots(figsize=(12,10), subplot_kw={"projection":ccrs.PlateCarree()})

    ax.set_extent([KHAVDA_LON-BUFFER_DEG, KHAVDA_LON+BUFFER_DEG,
                   KHAVDA_LAT-BUFFER_DEG, KHAVDA_LAT+BUFFER_DEG])

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)

    cf = ax.contourf(lons, lats, ws, 20, cmap="YlOrRd")
    plt.colorbar(cf, ax=ax, label="100m wind speed (m/s)")

    stride = max(1, len(lats)//25)
    ax.quiver(lons[::stride], lats[::stride], u[::stride,::stride], v[::stride,::stride],
              color="black", scale=200)

    ax.plot(KHAVDA_LON, KHAVDA_LAT, marker="*", color="blue", markersize=16)

    out = Path(VIS_DIR)/f"khavda_windmap_{run_name}_{fh}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

# ============================================================================
# NETCDF VERIFICATION
# ============================================================================

def verify_one_tile(run_name, fh="f000"):
    tile = Path(PROCESSED_DIR)/run_name/fh/"tiles"/"tile_lat20_lon60.nc"
    if not tile.exists():
        print("Tile missing.")
        return

    ds = xr.open_dataset(tile)
    print(ds)
    ds.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    run_name = input("Enter ECMWF run name (YYYYMMDD_HHZ): ").strip()

    print("\nExtracting Khavda time series...")
    df = extract_timeseries(run_name)

    print("\nGenerating sample visualizations...")
    for fh in ["f000","f024","f048","f072","f120","f144"]:
        visualize(run_name, fh)

    print("\nVerifying one tile...")
    verify_one_tile(run_name)

    print("\nPART 2 COMPLETE.")

if __name__ == "__main__":
    main()
