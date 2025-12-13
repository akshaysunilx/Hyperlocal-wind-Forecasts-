#!/usr/bin/env python3
"""
ECMWF 100m WIND DOWNLOADING + HIGH-RES PROCESSING PIPELINE
Part 1 of 2
------------------------------------------------------------
This script:
 - Detects latest ECMWF cycle
 - Downloads 3-hourly forecasts (100u, 100v)
 - Verifies GRIB2 integrity
 - Interpolates from 0.25° → 0.01°
 - Writes NetCDF tiles for downstream analysis
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import json, time, warnings
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================

BASE_OUTPUT_DIR = "ecmwf_wind_100m_data"
PROCESSED_DIR = "ecmwf_wind_100m_processed"

KHAVDA_LAT = 24.1132
KHAVDA_LON = 69.3669

TARGET_RES = 0.01
TILE_SIZE_DEG = 10
DOWNLOAD_DELAY = 0.5

# Create all folders
for d in [BASE_OUTPUT_DIR, PROCESSED_DIR]:
    Path(d).mkdir(exist_ok=True, parents=True)

# =====================================================================
# ECMWF RUN DETECTION
# =====================================================================

def get_latest_ecmwf_run():
    now = datetime.utcnow()
    available = now - timedelta(hours=8)
    hour = available.hour

    if hour < 6:
        cycle = 18
        date = (available - timedelta(days=1)).strftime("%Y%m%d")
    elif hour < 12:
        cycle = 0
        date = available.strftime("%Y%m%d")
    elif hour < 18:
        cycle = 6
        date = available.strftime("%Y%m%d")
    else:
        cycle = 12
        date = available.strftime("%Y%m%d")
    return date, cycle

# =====================================================================
# DOWNLOAD ECMWF DATA
# =====================================================================

def download_ecmwf_100m_winds():
    try:
        from ecmwf.opendata import Client
    except:
        print("Install ecmwf-opendata: pip install ecmwf-opendata")
        return None

    date_str, cycle = get_latest_ecmwf_run()
    run_name = f"{date_str}_{cycle:02d}Z"
    run_dir = Path(BASE_OUTPUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading ECMWF run {run_name}")

    client = Client(source="ecmwf")
    forecast_hours = list(range(0,145,3))
    file_map = {}

    for fh in tqdm(forecast_hours, desc="Downloading"):
        fh_dir = run_dir / f"f{fh:03d}"
        fh_dir.mkdir(exist_ok=True)

        for var in ["100u","100v"]:
            grib_file = fh_dir / f"{var}_{run_name}_f{fh:03d}.grib2"
            key = f"{var}_f{fh:03d}"

            if grib_file.exists() and grib_file.stat().st_size > 10000:
                file_map[key] = str(grib_file.relative_to(run_dir))
                continue

            try:
                client.retrieve(
                    date=int(date_str),
                    time=cycle,
                    step=fh,
                    type="fc",
                    param=var,
                    target=str(grib_file)
                )
                time.sleep(DOWNLOAD_DELAY)
            except Exception as e:
                print(f"FAILED: {key} {e}")
                continue

            if grib_file.exists() and grib_file.stat().st_size > 10000:
                file_map[key] = str(grib_file.relative_to(run_dir))

    metadata = {
        "run_date": date_str,
        "cycle": cycle,
        "files": file_map,
        "forecast_hours": forecast_hours
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Download complete.")
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "forecast_hours": forecast_hours
    }

# =====================================================================
# VERIFY GRIB2
# =====================================================================

def verify_downloaded_files(run_dir):
    metadata = json.load(open(run_dir/"metadata.json"))
    ok = 0

    for key, rel in metadata["files"].items():
        fn = run_dir / rel
        try:
            ds = xr.open_dataset(fn, engine="cfgrib")
            if len(ds.data_vars) > 0:
                ok += 1
            ds.close()
        except:
            pass

    if ok == 0:
        print("ERROR: no valid GRIB files")
        return False

    print(f"Verified: {ok}/{len(metadata['files'])}")
    return True

# =====================================================================
# HIGH RES INTERPOLATION
# =====================================================================

def interpolate_tile(src_lat, src_lon, src_data, lat0, lon0):
    tgt_lat = np.arange(lat0, lat0+TILE_SIZE_DEG, TARGET_RES)
    tgt_lon = np.arange(lon0, lon0+TILE_SIZE_DEG, TARGET_RES)

    interp = RectBivariateSpline(src_lat, src_lon, src_data, kx=3, ky=3)
    data_out = interp(tgt_lat, tgt_lon)
    return tgt_lat, tgt_lon, data_out

def process_one_hour(grib_u, grib_v, output_dir, run_name, fh):
    ds_u = xr.open_dataset(grib_u, engine="cfgrib")
    ds_v = xr.open_dataset(grib_v, engine="cfgrib")

    lat = ds_u.latitude.values
    lon = ds_u.longitude.values
    u = ds_u[list(ds_u.data_vars)[0]].values
    v = ds_v[list(ds_v.data_vars)[0]].values
    valid_time = ds_u.time.values

    if lon.max()>180:
        lon = np.where(lon>180, lon-360, lon)
        idx = np.argsort(lon)
        lon = lon[idx]
        u = u[:,idx]
        v = v[:,idx]

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[::-1,:]
        v = v[::-1,:]

    lat0, lon0 = 20, 60
    tile_fn = output_dir / f"tile_lat{lat0}_lon{lon0}.nc"

    lat_new, lon_new, u_new = interpolate_tile(lat,lon,u,lat0,lon0)
    _, _, v_new = interpolate_tile(lat,lon,v,lat0,lon0)

    ws = np.sqrt(u_new*u_new + v_new*v_new)
    wd = (270 - np.degrees(np.arctan2(v_new,u_new))) % 360

    out = xr.Dataset(
        {
            "100u": (["latitude","longitude"], u_new.astype("float32")),
            "100v": (["latitude","longitude"], v_new.astype("float32")),
            "windspeed_100m": (["latitude","longitude"], ws.astype("float32")),
            "winddir_100m": (["latitude","longitude"], wd.astype("float32"))
        },
        coords={
            "latitude": lat_new,
            "longitude": lon_new,
            "time": valid_time
        }
    )
    out.to_netcdf(tile_fn, encoding={v:{"zlib":True,"complevel":4} for v in out.data_vars})

    return True

def process_all_forecast_hours(run_info):
    run_dir = run_info["run_dir"]
    run_name = run_info["run_name"]

    for fh in tqdm(run_info["forecast_hours"], desc="Processing hours"):
        
        fh_dir = run_dir / f"f{fh:03d}"
        files = list(fh_dir.glob("*.grib2"))
        if len(files)<2:
            continue

        output_dir = Path(PROCESSED_DIR)/run_name/f"f{fh:03d}"/"tiles"
        output_dir.mkdir(parents=True,exist_ok=True)

        ufile = [f for f in files if "100u" in f.name][0]
        vfile = [f for f in files if "100v" in f.name][0]

        process_one_hour(ufile, vfile, output_dir, run_name, fh)

    print("All forecast hours processed.")

# =====================================================================
# MAIN
# =====================================================================

def main():
    info = download_ecmwf_100m_winds()
    if info is None:
        return
    if not verify_downloaded_files(info["run_dir"]):
        return
    process_all_forecast_hours(info)
    print("\nPART 1 COMPLETE — Run CODE 2 to extract time series + visualize.\n")

if __name__ == "__main__":
    main()
