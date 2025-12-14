#!/usr/bin/env python3
"""
Integrated 10m Wind Forecast (Notebook-Safe, Debuggable, Robust)

Computes 10 m sustained wind speed (m/s):
  ws = sqrt(u10^2 + v10^2)

Models
- GFS 0.25  (NOAA NOMADS GRIB2 filter; subregion download per step)
- ECMWF IFS Open Data 0.25 (ecmwf-opendata; robust cycle selection + 404-safe)
- ICON Global (DWD Open Data; U/V .bz2 pairs; UNSTRUCTURED grid handled via CLAT/CLON)

Target region (box mean)
  LAT: 23.375–24.625
  LON: 68.625–69.875

Output
- 15-minute timestamps (UTC index + IST column), OR native-only timestamps
- One CSV containing all models (some columns may be empty if a model is unavailable)

Recommended installs (Colab)
  pip install -q requests numpy pandas beautifulsoup4 ecmwf-opendata eccodes
"""

from __future__ import annotations

import argparse
import bz2
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ecmwf.opendata import Client

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetBox:
    lat_min: float = 23.375
    lat_max: float = 24.625
    lon_min: float = 68.625
    lon_max: float = 69.875

TARGET = TargetBox()
IST_OFFSET = timedelta(hours=5, minutes=30)

BASE_DIR = Path("wind_forecast_data")
RAW_DIR = BASE_DIR / "raw"
CSV_DIR = BASE_DIR / "csv"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# NOAA NOMADS (GFS 0.25deg)
GFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# DWD ICON
ICON_BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"

# Networking defaults
REQ_TIMEOUT = 60
REQ_RETRIES = 6
REQ_BACKOFF_S = 2.0
DEFAULT_UA = "wind-forecast-script/4.2"

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------

LOGGER = logging.getLogger("wind_forecast")


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    LOGGER.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    LOGGER.handlers[:] = [handler]
    LOGGER.propagate = False


# ------------------------------------------------------------------------------
# TIME HELPERS
# ------------------------------------------------------------------------------

def utcnow_naive() -> datetime:
    """Return naive UTC datetime (safe for pandas naive indexes)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ------------------------------------------------------------------------------
# HTTP helpers (handles 429/503 with backoff)
# ------------------------------------------------------------------------------

def _request(method: str, url: str, *, timeout: int, stream: bool) -> requests.Response:
    headers = {"User-Agent": DEFAULT_UA}
    return requests.request(method, url, timeout=timeout, stream=stream, headers=headers)


def _request_with_retries(
    method: str,
    url: str,
    *,
    timeout: int = REQ_TIMEOUT,
    stream: bool = False
) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(REQ_RETRIES):
        try:
            r = _request(method, url, timeout=timeout, stream=stream)

            if r.status_code in (429, 503):
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = int(retry_after)
                else:
                    sleep_s = REQ_BACKOFF_S * (2 ** attempt)
                LOGGER.warning(f"HTTP {r.status_code} for {url}. Sleeping {sleep_s:.1f}s then retrying.")
                time.sleep(sleep_s)
                continue

            return r
        except Exception as e:
            last_err = e
            sleep_s = REQ_BACKOFF_S * (2 ** attempt)
            LOGGER.warning(
                f"Request failed ({method} {url}) attempt={attempt+1}/{REQ_RETRIES}: {e}. "
                f"Sleeping {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"HTTP request failed after {REQ_RETRIES} attempts: {method} {url}") from last_err


def _looks_like_grib(path: Path) -> bool:
    """GRIB files begin with b'GRIB'. NOMADS sometimes returns HTML with 200 OK."""
    try:
        if not path.exists() or path.stat().st_size < 16:
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"GRIB"
    except Exception:
        return False


def _download_to_file(url: str, path: Path, *, debug: bool = False) -> bool:
    """
    Download URL to path.
    Returns:
      True  if downloaded (or already valid)
      False if 404 OR content is not GRIB (common NOMADS issue)
    """
    if path.exists() and path.stat().st_size > 0 and _looks_like_grib(path):
        if debug:
            LOGGER.debug(f"Download skip (already valid GRIB): {path}")
        return True

    r = _request_with_retries("GET", url, timeout=REQ_TIMEOUT, stream=True)
    if r.status_code == 404:
        if debug:
            LOGGER.debug(f"404 Not Found: {url}")
        return False
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code} for {url}", response=r)

    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(path)

    if not _looks_like_grib(path):
        try:
            with open(path, "rb") as f:
                sample = f.read(200)
            LOGGER.warning(f"Non-GRIB response from {url}. Deleting. Head={sample[:80]!r}")
        except Exception:
            LOGGER.warning(f"Non-GRIB response from {url}. Deleting.")
        path.unlink(missing_ok=True)
        return False

    return True


def _download_and_decompress_bz2(url: str, out_path: Path, *, debug: bool = False) -> bool:
    """
    Download .bz2 and decompress to out_path.
    Returns False if 404 OR decompressed content is not GRIB.
    """
    if out_path.exists() and out_path.stat().st_size > 0 and _looks_like_grib(out_path):
        if debug:
            LOGGER.debug(f"Decompress skip (already valid GRIB): {out_path}")
        return True

    r = _request_with_retries("GET", url, timeout=120, stream=True)
    if r.status_code == 404:
        if debug:
            LOGGER.debug(f"404 Not Found: {url}")
        return False
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code} for {url}", response=r)

    bz2_path = out_path.with_suffix(out_path.suffix + ".bz2")
    tmp = bz2_path.with_suffix(bz2_path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(bz2_path)

    out_part = out_path.with_suffix(out_path.suffix + ".part")
    with bz2.open(bz2_path, "rb") as src, open(out_part, "wb") as dst:
        shutil.copyfileobj(src, dst)
    out_part.replace(out_path)
    bz2_path.unlink(missing_ok=True)

    if not _looks_like_grib(out_path):
        LOGGER.warning(f"Decompressed file is not GRIB. Deleting: {out_path}")
        out_path.unlink(missing_ok=True)
        return False

    return True


# ------------------------------------------------------------------------------
# MATH + REGION
# ------------------------------------------------------------------------------

def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u**2 + v**2)


def _to_degrees_if_radians(arr: np.ndarray) -> np.ndarray:
    mx = np.nanmax(np.abs(arr)) if arr.size else np.nan
    if np.isfinite(mx) and mx <= 7.0:  # radians threshold
        return arr * (180.0 / np.pi)
    return arr


def _normalize_lon(lon: np.ndarray) -> np.ndarray:
    lon = _to_degrees_if_radians(lon)
    # Normalize [-180..180] to [0..360) where needed
    if np.nanmin(lon) < 0 and np.nanmax(lon) <= 180:
        lon = (lon + 360) % 360
    return lon


# ------------------------------------------------------------------------------
# GRIB: eccodes helpers
# ------------------------------------------------------------------------------

def _require_eccodes() -> None:
    try:
        import eccodes  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "This script needs the Python 'eccodes' package for GRIB parsing.\n"
            "Install:\n  pip install -q eccodes\n"
        ) from e


def _eccodes_get_latlon(gid, *, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prefer latitudes/longitudes arrays. Fallback to codes_grib_get_data if needed.
    """
    import eccodes as ec

    try:
        lat = np.array(ec.codes_get_array(gid, "latitudes"), dtype=float)
        lon = np.array(ec.codes_get_array(gid, "longitudes"), dtype=float)
        if lat.size and lon.size:
            return lat, lon
    except Exception as e:
        if debug:
            LOGGER.debug(f"latitudes/longitudes not available: {e}")

    # Fallback: slower, may be unsupported for some grid types
    if not hasattr(ec, "codes_grib_get_data"):
        raise RuntimeError("eccodes missing codes_grib_get_data(); cannot derive lat/lon")

    data = ec.codes_grib_get_data(gid)  # iterable of dicts {'lat','lon','value'}
    lats: List[float] = []
    lons: List[float] = []
    for d in data:
        lats.append(d["lat"])
        lons.append(d["lon"])
    lat = np.array(lats, dtype=float)
    lon = np.array(lons, dtype=float)
    return lat, lon


def _eccodes_try_reshape(vals: np.ndarray, lat: np.ndarray, lon: np.ndarray, gid, *, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If Ni/Nj are present and match, reshape vals/lat/lon into (Nj, Ni).
    """
    import eccodes as ec
    try:
        if ec.codes_is_defined(gid, "Ni") and ec.codes_is_defined(gid, "Nj"):
            ni = int(ec.codes_get(gid, "Ni"))
            nj = int(ec.codes_get(gid, "Nj"))
            if ni * nj == vals.size == lat.size == lon.size:
                return vals.reshape(nj, ni), lat.reshape(nj, ni), lon.reshape(nj, ni)
    except Exception as e:
        if debug:
            LOGGER.debug(f"reshape skip: {e}")
    return vals, lat, lon


def _harmonize_uv_shapes(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray, *, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensures u and v have identical shapes.
    Fixes cases where U was reshaped (Nj,Ni) but V remained 1D.
    """
    if u.ndim == 2 and v.ndim == 1 and v.size == u.size:
        v = v.reshape(u.shape)
        if debug:
            LOGGER.debug(f"Harmonized: reshaped V to {v.shape}")
    if v.ndim == 2 and u.ndim == 1 and u.size == v.size:
        u = u.reshape(v.shape)
        if debug:
            LOGGER.debug(f"Harmonized: reshaped U to {u.shape}")

    if u.ndim == 2 and lat.ndim == 1 and lat.size == u.size:
        lat = lat.reshape(u.shape)
    if u.ndim == 2 and lon.ndim == 1 and lon.size == u.size:
        lon = lon.reshape(u.shape)

    if u.shape != v.shape:
        raise RuntimeError(f"u/v shape mismatch after harmonize: u{u.shape} v{v.shape}")
    return u, v, lat, lon


def _mask_from_latlon(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = _to_degrees_if_radians(lat)
    lon = _normalize_lon(lon)
    return (
        (lat >= TARGET.lat_min) & (lat <= TARGET.lat_max) &
        (lon >= TARGET.lon_min) & (lon <= TARGET.lon_max)
    )


def _eccodes_read_first_message_values(fp: Path) -> np.ndarray:
    import eccodes as ec
    with open(fp, "rb") as f:
        gid = ec.codes_grib_new_from_file(f)
        if gid is None:
            raise RuntimeError(f"Empty GRIB file: {fp}")
        try:
            return np.array(ec.codes_get_values(gid), dtype=float)
        finally:
            ec.codes_release(gid)


# ------------------------------------------------------------------------------
# GFS
# ------------------------------------------------------------------------------

def _gfs_probe_url(date_yyyymmdd: str, cycle_hh: str) -> str:
    params = (
        f"dir=%2Fgfs.{date_yyyymmdd}%2F{cycle_hh}%2Fatmos"
        f"&file=gfs.t{cycle_hh}z.pgrb2.0p25.f000"
        f"&var_UGRD=on&var_VGRD=on"
        f"&lev_10_m_above_ground=on"
        f"&subregion="
        f"&leftlon={TARGET.lon_min}&rightlon={TARGET.lon_max}"
        f"&toplat={TARGET.lat_max}&bottomlat={TARGET.lat_min}"
    )
    return f"{GFS_FILTER}?{params}"


def pick_latest_available_gfs_cycle(now_utc: datetime, *, debug: bool = False) -> datetime:
    dates = [now_utc.strftime("%Y%m%d"), (now_utc - timedelta(days=1)).strftime("%Y%m%d")]
    cycles = ["18", "12", "06", "00"]

    for d in dates:
        for c in cycles:
            if d == dates[0] and int(c) > now_utc.hour:
                continue
            url = _gfs_probe_url(d, c)
            r = _request_with_retries("GET", url, timeout=30, stream=False)
            if debug:
                LOGGER.debug(f"GFS probe {d} {c}Z -> {r.status_code} len={len(r.content)}")
            if r.status_code == 200 and len(r.content) > 0:
                init = datetime.strptime(d + c, "%Y%m%d%H")
                LOGGER.info(f"Selected GFS cycle: {init:%Y-%m-%d %HZ} (GRIB probe OK)")
                return init

    raise RuntimeError("Could not find an available GFS cycle via probing.")


def build_gfs_url(init_time: datetime, fh: int) -> str:
    date_yyyymmdd = init_time.strftime("%Y%m%d")
    cycle_hh = init_time.strftime("%H")
    params = (
        f"dir=%2Fgfs.{date_yyyymmdd}%2F{cycle_hh}%2Fatmos"
        f"&file=gfs.t{cycle_hh}z.pgrb2.0p25.f{fh:03d}"
        f"&var_UGRD=on&var_VGRD=on"
        f"&lev_10_m_above_ground=on"
        f"&subregion="
        f"&leftlon={TARGET.lon_min}&rightlon={TARGET.lon_max}"
        f"&toplat={TARGET.lat_max}&bottomlat={TARGET.lat_min}"
    )
    return f"{GFS_FILTER}?{params}"


def gfs_forecast_hours(hours: int, stride: int) -> List[int]:
    h = int(hours)
    s = max(1, int(stride))
    return list(range(0, h + 1, s))


def download_gfs(
    init_time: datetime,
    hours: int,
    stride: int,
    *,
    debug: bool,
    max_files: int,
    sleep_s: float
) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    gfs_dir = RAW_DIR / f"gfs_{init_time:%Y%m%d%H}Z"
    gfs_dir.mkdir(parents=True, exist_ok=True)

    fhs = gfs_forecast_hours(hours, stride)
    LOGGER.info(f"GFS planned forecast hours: {len(fhs)} (stride={stride})")

    for fh in fhs:
        if len(out) >= max_files:
            LOGGER.warning(f"GFS max_files reached ({max_files}). Stopping downloads.")
            break

        url = build_gfs_url(init_time, fh)
        path = gfs_dir / f"gfs_f{fh:03d}.grib2"

        ok = _download_to_file(url, path, debug=debug)
        if not ok:
            LOGGER.info(f"GFS missing/invalid fh={fh:03d}. Skipping.")
            continue

        out.append((fh, path))
        time.sleep(max(0.0, float(sleep_s)))

    LOGGER.info(f"GFS downloaded valid GRIB files: {len(out)}")
    return out


def gfs_process_files(files: List[Tuple[int, Path]], init_time: datetime, *, debug: bool) -> pd.Series:
    _require_eccodes()
    import eccodes as ec

    vals: List[float] = []
    idx: List[pd.Timestamp] = []
    region_mask: Optional[np.ndarray] = None
    processed = 0

    def extract_uv_from_file(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = None
        v = None
        lat = None
        lon = None

        with open(fp, "rb") as f:
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                try:
                    sn = str(ec.codes_get(gid, "shortName")).lower()
                    tol = str(ec.codes_get(gid, "typeOfLevel")) if ec.codes_is_defined(gid, "typeOfLevel") else ""
                    lvl = int(ec.codes_get(gid, "level")) if ec.codes_is_defined(gid, "level") else -1

                    # Accept 10m AGL
                    if not ((tol == "heightAboveGround" and lvl == 10) or lvl == 10):
                        continue

                    if sn in {"10u", "u10", "ugrd"}:
                        u = np.array(ec.codes_get_values(gid), dtype=float)
                        lat_, lon_ = _eccodes_get_latlon(gid, debug=debug)
                        u, lat_, lon_ = _eccodes_try_reshape(u, lat_, lon_, gid, debug=debug)
                        lat, lon = lat_, lon_
                    elif sn in {"10v", "v10", "vgrd"}:
                        v = np.array(ec.codes_get_values(gid), dtype=float)
                finally:
                    ec.codes_release(gid)

        if u is None or v is None or lat is None or lon is None:
            raise RuntimeError("Missing u/v/lat/lon in GFS file")

        # Critical fix: ensure U/V/lat/lon shapes match
        u, v, lat, lon = _harmonize_uv_shapes(u, v, lat, lon, debug=debug)
        return u, v, lat, lon

    for fh, fp in files:
        try:
            u, v, lat, lon = extract_uv_from_file(fp)

            if region_mask is None:
                region_mask = _mask_from_latlon(lat, lon)
                LOGGER.info(f"GFS region mask points in box: {int(region_mask.sum())} / {region_mask.size}")

            ws = wind_speed(u, v)
            mval = float(np.nanmean(ws[region_mask])) if np.any(region_mask) else float("nan")

            idx.append(pd.Timestamp(init_time + timedelta(hours=int(fh))))
            vals.append(mval)
            processed += 1

            if debug:
                LOGGER.debug(f"GFS fh={fh:03d} mean={mval}")
        except Exception as e:
            LOGGER.warning(f"GFS processing failed fh={fh:03d}: {e}")

    s = pd.Series(vals, index=pd.to_datetime(idx), name="GFS_10m_ms").sort_index()
    LOGGER.info(f"GFS processed points: {processed}")
    return s


# ------------------------------------------------------------------------------
# ECMWF (robust: latest() + candidate cycles; 404-safe)
# ------------------------------------------------------------------------------

def ecmwf_steps_oper(horizon_h: int) -> List[int]:
    h = int(horizon_h)
    steps = list(range(0, min(h, 144) + 1, 3))
    if h > 144:
        steps += list(range(150, h + 1, 6))
    return sorted(set(steps))


def ecmwf_steps_scda(horizon_h: int) -> List[int]:
    h = int(horizon_h)
    steps = list(range(0, min(h, 90) + 1, 1))
    if h > 90:
        steps += list(range(93, min(h, 144) + 1, 3))
    return sorted(set(steps))


def _ecmwf_latest_safe(client: Client, stream: str, *, debug: bool = False) -> Optional[datetime]:
    for step_try in [0, 3, 6]:
        try:
            dt = client.latest(type="fc", stream=stream, step=step_try, param="10u")
            if isinstance(dt, datetime):
                return dt
            if hasattr(dt, "datetime"):
                return dt.datetime
        except Exception as e:
            if debug:
                LOGGER.debug(f"ECMWF latest failed stream={stream} step={step_try}: {e}")
    return None


def _ecmwf_candidate_inits(now_utc: datetime, days_back: int = 3) -> List[datetime]:
    cands: List[datetime] = []
    for dd in range(0, days_back + 1):
        day = now_utc - timedelta(days=dd)
        for hh in ["18", "12", "06", "00"]:
            init = datetime.strptime(day.strftime("%Y%m%d") + hh, "%Y%m%d%H")
            if init <= now_utc:
                cands.append(init)
    return sorted(set(cands), reverse=True)


def _retrieve_ecmwf_uv10(
    client: Client,
    init_time: datetime,
    stream: str,
    steps: List[int],
    out_path: Path,
    *,
    debug: bool = False
) -> Optional[Path]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    try:
        # Use strings for date/time for maximum compatibility
        client.retrieve(
            date=init_time.strftime("%Y-%m-%d"),
            time=init_time.strftime("%H"),
            stream=stream,
            type="fc",
            step=steps,
            param=["10u", "10v"],
            target=str(out_path),
        )
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        return None
    except requests.HTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:
            if debug:
                LOGGER.debug(f"ECMWF 404 stream={stream} init={init_time:%Y-%m-%d %HZ}")
            return None
        raise
    except Exception as e:
        if debug:
            LOGGER.debug(f"ECMWF retrieve failed stream={stream} init={init_time:%Y-%m-%d %HZ}: {e}")
        return None


def process_ecmwf_multi_step_grib(path: Path, *, debug: bool = False) -> pd.Series:
    _require_eccodes()
    import eccodes as ec

    LOGGER.info(f"Processing ECMWF GRIB: {path}")

    rec: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
    box_idx: Optional[np.ndarray] = None
    box_count_logged = False

    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                sn = str(ec.codes_get(gid, "shortName")).lower()
                tol = str(ec.codes_get(gid, "typeOfLevel")) if ec.codes_is_defined(gid, "typeOfLevel") else ""
                lvl = int(ec.codes_get(gid, "level")) if ec.codes_is_defined(gid, "level") else -1

                if tol != "heightAboveGround" or lvl != 10:
                    continue
                if sn not in {"10u", "10v", "u10", "v10"}:
                    continue

                vdate = int(ec.codes_get(gid, "validityDate")) if ec.codes_is_defined(gid, "validityDate") else int(ec.codes_get(gid, "dataDate"))
                vtime = int(ec.codes_get(gid, "validityTime")) if ec.codes_is_defined(gid, "validityTime") else int(ec.codes_get(gid, "dataTime"))
                vt = pd.Timestamp(datetime.strptime(f"{vdate}{vtime:04d}", "%Y%m%d%H%M"))

                vals = np.array(ec.codes_get_values(gid), dtype=float)

                if box_idx is None and ("u" in sn):
                    lat_arr, lon_arr = _eccodes_get_latlon(gid, debug=debug)
                    lat_arr = _to_degrees_if_radians(lat_arr)
                    lon_arr = _normalize_lon(lon_arr)
                    mask = (
                        (lat_arr >= TARGET.lat_min) & (lat_arr <= TARGET.lat_max) &
                        (lon_arr >= TARGET.lon_min) & (lon_arr <= TARGET.lon_max)
                    )
                    box_idx = np.where(mask)[0]
                    if not box_count_logged:
                        LOGGER.info(f"ECMWF region mask points in box: {box_idx.size} / {lat_arr.size}")
                        box_count_logged = True

                comp = "u" if "u" in sn else "v"
                rec.setdefault(vt, {})[comp] = vals
            finally:
                ec.codes_release(gid)

    if box_idx is None:
        raise RuntimeError("ECMWF: could not build region index (no U field found)")

    out_idx: List[pd.Timestamp] = []
    out_vals: List[float] = []
    pending = 0

    for vt in sorted(rec.keys()):
        r = rec[vt]
        if "u" not in r or "v" not in r:
            pending += 1
            continue
        u = r["u"][box_idx]
        v = r["v"][box_idx]
        ws = wind_speed(u, v)
        out_idx.append(vt)
        out_vals.append(float(np.nanmean(ws)))

    s = pd.Series(out_vals, index=pd.to_datetime(out_idx), name="ECMWF_10m_ms").sort_index()
    if not s.index.is_unique:
        s = s.groupby(level=0).mean().sort_index()

    LOGGER.info(f"ECMWF processed points: {len(s)} (pending leftover={pending})")
    return s


def build_ecmwf_series(source: str, horizon_h: int, *, debug: bool = False) -> Tuple[pd.Series, datetime, str]:
    """
    Robust ECMWF selection:
      1) Try client.latest() for oper/scda (if available)
      2) Then walk back through recent cycles (last ~3 days)
      3) For each init: try oper first, then scda (stream-specific steps)
    """
    client = Client(
        source=source,
        model="ifs",
        resol="0p25",
        preserve_request_order=False,
        infer_stream_keyword=True,
    )

    now_utc = utcnow_naive()
    latest_oper = _ecmwf_latest_safe(client, "oper", debug=debug)
    latest_scda = _ecmwf_latest_safe(client, "scda", debug=debug)

    candidate_inits = _ecmwf_candidate_inits(now_utc, days_back=3)
    # Seed list with latest() results (if present) at the front
    seeded: List[datetime] = []
    for dt in [latest_oper, latest_scda]:
        if dt is not None:
            seeded.append(dt.replace(minute=0, second=0, microsecond=0))
    for dt in candidate_inits:
        seeded.append(dt)
    candidates = []
    seen = set()
    for dt in seeded:
        key = dt.strftime("%Y%m%d%H")
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dt)

    last_err: Optional[str] = None

    for init in candidates:
        # Try OPER first
        for stream, steps in [
            ("oper", ecmwf_steps_oper(horizon_h)),
            ("scda", ecmwf_steps_scda(horizon_h)),
        ]:
            ecmwf_dir = RAW_DIR / f"ecmwf_{init:%Y%m%d%H}Z_{stream}"
            ecmwf_dir.mkdir(parents=True, exist_ok=True)
            out_path = ecmwf_dir / f"ecmwf_{stream}_10u10v.grib2"

            try:
                got = _retrieve_ecmwf_uv10(client, init, stream, steps, out_path, debug=debug)
                if not got or not got.exists() or got.stat().st_size == 0:
                    last_err = f"ECMWF retrieve empty stream={stream} init={init:%Y-%m-%d %HZ}"
                    continue

                s = process_ecmwf_multi_step_grib(got, debug=debug)
                s.name = "ECMWF_10m_ms"
                LOGGER.info(f"Selected ECMWF cycle: {init:%Y-%m-%d %HZ} ({stream.upper()})")
                return s.sort_index(), init, stream
            except Exception as e:
                last_err = f"ECMWF failed stream={stream} init={init:%Y-%m-%d %HZ}: {e}"
                if debug:
                    LOGGER.debug(last_err)

    raise RuntimeError(last_err or "ECMWF: no available cycle found in candidate window")


# ------------------------------------------------------------------------------
# ICON (unstructured grid: compute bbox indices using CLAT/CLON)
# ------------------------------------------------------------------------------

def _list_icon_dir(url: str, *, debug: bool = False) -> List[str]:
    r = _request_with_retries("GET", url, timeout=60, stream=False)
    if debug:
        LOGGER.debug(f"ICON list {url} -> {r.status_code}")
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]
    return sorted(hrefs)


def pick_latest_available_icon_cycle(now_utc: datetime, *, debug: bool = False) -> datetime:
    dates = [now_utc.strftime("%Y%m%d"), (now_utc - timedelta(days=1)).strftime("%Y%m%d")]
    cycles = ["18", "12", "06", "00"]

    for d in dates:
        for c in cycles:
            if d == dates[0] and int(c) > now_utc.hour:
                continue

            run_stamp = f"{d}{c}"
            u_files = _list_icon_dir(f"{ICON_BASE}/{c}/u_10m/", debug=debug)
            v_files = _list_icon_dir(f"{ICON_BASE}/{c}/v_10m/", debug=debug)

            has_u = any((run_stamp in fn and "U_10M" in fn and fn.endswith(".grib2.bz2")) for fn in u_files)
            has_v = any((run_stamp in fn and "V_10M" in fn and fn.endswith(".grib2.bz2")) for fn in v_files)
            if has_u and has_v:
                init = datetime.strptime(d + c, "%Y%m%d%H")
                LOGGER.info(f"Selected ICON cycle: {init:%Y-%m-%d %HZ}")
                return init

    raise RuntimeError("Could not find an available ICON cycle via directory probing.")


def _icon_extract_fh(fn: str, token: str) -> Optional[int]:
    # Typical:
    # icon_global_icosahedral_single-level_2025121406_000_U_10M.grib2.bz2
    m = re.search(rf"_(\d{{3}})_{re.escape(token)}\b", fn)
    if m:
        return int(m.group(1))
    m = re.search(rf"_(\d{{3}})_{re.escape(token)}\.", fn)
    if m:
        return int(m.group(1))
    m = re.search(rf"(\d{{3}}).*{re.escape(token)}", fn)
    if m:
        return int(m.group(1))
    return None


def _build_icon_maps(init_time: datetime, cycle_hh: str, *, debug: bool = False) -> Tuple[Dict[int, str], Dict[int, str]]:
    run_stamp = f"{init_time:%Y%m%d}{cycle_hh}"
    u_files = _list_icon_dir(f"{ICON_BASE}/{cycle_hh}/u_10m/", debug=debug)
    v_files = _list_icon_dir(f"{ICON_BASE}/{cycle_hh}/v_10m/", debug=debug)

    def to_map(files: List[str], token: str) -> Dict[int, str]:
        m: Dict[int, str] = {}
        for fn in files:
            if run_stamp not in fn or token not in fn or not fn.endswith(".grib2.bz2"):
                continue
            fh = _icon_extract_fh(fn, token)
            if fh is not None:
                m[fh] = fn
        return m

    u_map = to_map(u_files, "U_10M")
    v_map = to_map(v_files, "V_10M")

    if debug:
        LOGGER.debug(f"ICON map sizes: u={len(u_map)} v={len(v_map)} run_stamp={run_stamp}")

    return u_map, v_map


def _pick_icon_grid_files(cycle_hh: str, init_time: datetime, *, debug: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick CLAT/CLON files. Prefer ones matching the run stamp; else newest in listing.
    """
    run_stamp = f"{init_time:%Y%m%d}{cycle_hh}"

    clat_list = _list_icon_dir(f"{ICON_BASE}/{cycle_hh}/clat/", debug=debug)
    clon_list = _list_icon_dir(f"{ICON_BASE}/{cycle_hh}/clon/", debug=debug)

    def pick(lst: List[str], token: str) -> Optional[str]:
        cand = [x for x in lst if (token in x.upper() and x.endswith(".grib2.bz2"))]
        if not cand:
            return None
        same = [x for x in cand if run_stamp in x]
        return sorted(same)[-1] if same else sorted(cand)[-1]

    return pick(clat_list, "CLAT"), pick(clon_list, "CLON")


def download_icon(
    init_time: datetime,
    horizon_h: int,
    stride: int,
    *,
    debug: bool,
    sleep_s: float
) -> Tuple[List[Tuple[int, Tuple[Path, Path]]], Optional[Path], Optional[Path]]:
    """
    Returns:
      (pairs, clat_path, clon_path)
    """
    out: List[Tuple[int, Tuple[Path, Path]]] = []
    cycle_hh = init_time.strftime("%H")
    icon_dir = RAW_DIR / f"icon_{init_time:%Y%m%d%H}Z"
    icon_dir.mkdir(parents=True, exist_ok=True)

    # Download CLAT/CLON once
    clat_fn, clon_fn = _pick_icon_grid_files(cycle_hh, init_time, debug=debug)
    clat_path = clon_path = None
    if clat_fn and clon_fn:
        clat_url = f"{ICON_BASE}/{cycle_hh}/clat/{clat_fn}"
        clon_url = f"{ICON_BASE}/{cycle_hh}/clon/{clon_fn}"
        clat_out = icon_dir / "icon_clat.grib2"
        clon_out = icon_dir / "icon_clon.grib2"
        ok1 = _download_and_decompress_bz2(clat_url, clat_out, debug=debug)
        ok2 = _download_and_decompress_bz2(clon_url, clon_out, debug=debug)
        if ok1 and ok2:
            clat_path, clon_path = clat_out, clon_out
            LOGGER.info(f"ICON grid downloaded: CLAT={clat_out.name} CLON={clon_out.name}")
        else:
            LOGGER.warning("ICON grid (CLAT/CLON) unavailable; ICON processing may fail.")
    else:
        LOGGER.warning("ICON CLAT/CLON not found in directory listing; ICON processing may fail.")

    # Download U/V pairs
    u_map, v_map = _build_icon_maps(init_time, cycle_hh, debug=debug)
    if not u_map or not v_map:
        LOGGER.warning("ICON maps empty; ICON may be unavailable for this cycle.")
        return out, clat_path, clon_path

    stride = max(1, int(stride))
    for fh in range(0, int(horizon_h) + 1, stride):
        if fh not in u_map or fh not in v_map:
            continue

        u_url = f"{ICON_BASE}/{cycle_hh}/u_10m/{u_map[fh]}"
        v_url = f"{ICON_BASE}/{cycle_hh}/v_10m/{v_map[fh]}"

        u_out = icon_dir / f"icon_u10_f{fh:03d}.grib2"
        v_out = icon_dir / f"icon_v10_f{fh:03d}.grib2"

        ok_u = _download_and_decompress_bz2(u_url, u_out, debug=debug)
        ok_v = _download_and_decompress_bz2(v_url, v_out, debug=debug)
        if ok_u and ok_v:
            out.append((fh, (u_out, v_out)))

        time.sleep(max(0.0, float(sleep_s)))

    LOGGER.info(f"ICON downloaded pairs: {len(out)}")
    return out, clat_path, clon_path


def _icon_build_box_indices_from_clat_clon(clat_path: Path, clon_path: Path) -> np.ndarray:
    clat = _eccodes_read_first_message_values(clat_path)
    clon = _eccodes_read_first_message_values(clon_path)

    if clat.size != clon.size:
        raise RuntimeError(f"ICON CLAT/CLON size mismatch: {clat.size} vs {clon.size}")

    lat = _to_degrees_if_radians(clat)
    lon = _normalize_lon(clon)

    mask = (
        (lat >= TARGET.lat_min) & (lat <= TARGET.lat_max) &
        (lon >= TARGET.lon_min) & (lon <= TARGET.lon_max)
    )
    idx = np.where(mask)[0].astype(int)

    LOGGER.info(f"ICON region mask points in box: {idx.size} / {lat.size}")
    return idx


def icon_process_pairs(
    pairs: List[Tuple[int, Tuple[Path, Path]]],
    init_time: datetime,
    clat_path: Optional[Path],
    clon_path: Optional[Path],
    *,
    debug: bool
) -> pd.Series:
    _require_eccodes()

    if not pairs:
        LOGGER.info("ICON processed points: 0")
        return pd.Series(dtype=float, name="ICON_10m_ms")

    if clat_path is None or clon_path is None or not clat_path.exists() or not clon_path.exists():
        LOGGER.warning("ICON cannot be processed: CLAT/CLON grid files missing.")
        return pd.Series(dtype=float, name="ICON_10m_ms")

    try:
        box_idx = _icon_build_box_indices_from_clat_clon(clat_path, clon_path)
        if box_idx.size == 0:
            LOGGER.warning("ICON: no points found inside target box (check bbox/grid).")
            return pd.Series(dtype=float, name="ICON_10m_ms")
    except Exception as e:
        LOGGER.warning(f"ICON: failed to build box indices from CLAT/CLON: {e}")
        return pd.Series(dtype=float, name="ICON_10m_ms")

    vals: List[float] = []
    idx: List[pd.Timestamp] = []
    processed = 0

    for fh, (u_fp, v_fp) in pairs:
        try:
            u_vals = _eccodes_read_first_message_values(u_fp)
            v_vals = _eccodes_read_first_message_values(v_fp)

            if u_vals.size != v_vals.size:
                raise RuntimeError(f"ICON u/v size mismatch: u={u_vals.size} v={v_vals.size}")
            if u_vals.size <= int(np.max(box_idx)):
                raise RuntimeError("ICON values array smaller than max box index (grid mismatch).")

            ws = wind_speed(u_vals[box_idx], v_vals[box_idx])
            mval = float(np.nanmean(ws))

            idx.append(pd.Timestamp(init_time + timedelta(hours=int(fh))))
            vals.append(mval)
            processed += 1

            if debug:
                LOGGER.debug(f"ICON fh={fh:03d} mean={mval} box_n={box_idx.size}")
        except Exception as e:
            LOGGER.warning(f"ICON processing failed fh={fh:03d}: {e}")

    s = pd.Series(vals, index=pd.to_datetime(idx), name="ICON_10m_ms").sort_index()
    LOGGER.info(f"ICON processed points: {processed}")
    return s


# ------------------------------------------------------------------------------
# SERIES CLEANUP + TIMELINES
# ------------------------------------------------------------------------------

def dedup_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    s = s.sort_index()
    if not s.index.is_unique:
        s = s.groupby(level=0).mean().sort_index()
    return s


def build_dataframe_native(gfs: pd.Series, ecmwf: pd.Series, icon: pd.Series) -> pd.DataFrame:
    all_idx = pd.Index([])
    for s in [gfs, ecmwf, icon]:
        if s is not None and not s.empty:
            all_idx = all_idx.union(pd.to_datetime(s.index))
    all_idx = pd.to_datetime(all_idx).sort_values()

    df = pd.DataFrame(index=all_idx)
    df.index.name = "Timestamp_UTC"
    df.insert(0, "Timestamp_IST", df.index + IST_OFFSET)

    for s in [gfs, ecmwf, icon]:
        if s is None or s.empty:
            continue
        s = dedup_series(s)
        df[s.name] = s.reindex(df.index)

    return df


def build_15min_dataframe(
    gfs: pd.Series,
    ecmwf: pd.Series,
    icon: pd.Series,
    gfs_init: datetime,
    ecmwf_init: datetime,
    icon_init: datetime,
    horizon_h: int,
    *,
    align: str = "latest",          # latest|earliest
    no_interpolate: bool = False,
    ecmwf_stream: str = ""
) -> pd.DataFrame:
    series_list = [("GFS_10m_ms", gfs), ("ECMWF_10m_ms", ecmwf), ("ICON_10m_ms", icon)]
    nonempty = [s for _n, s in series_list if s is not None and not s.empty]

    if not nonempty:
        start_utc = utcnow_naive()
    else:
        mins = [pd.to_datetime(s.index.min()).to_pydatetime() for s in nonempty]
        start_utc = max(mins) if align.lower() == "latest" else min(mins)

    end_utc = start_utc + timedelta(hours=int(horizon_h))
    idx_15 = pd.date_range(start=start_utc, end=end_utc, freq="15min")

    df = pd.DataFrame(index=idx_15)
    df.index.name = "Timestamp_UTC"
    df.insert(0, "Timestamp_IST", df.index + IST_OFFSET)

    for sname, s in series_list:
        if s is None or s.empty:
            continue
        s = dedup_series(s)
        df[sname] = s.reindex(df.index)
        if not no_interpolate:
            df[sname] = df[sname].interpolate(method="time")

    df["GFS_Init_UTC"] = gfs_init.strftime("%Y-%m-%d %H:%M:%S")
    df["ECMWF_Init_UTC"] = ecmwf_init.strftime("%Y-%m-%d %H:%M:%S")
    df["ICON_Init_UTC"] = icon_init.strftime("%Y-%m-%d %H:%M:%S")
    df["ECMWF_Stream"] = ecmwf_stream or ""

    return df


# ------------------------------------------------------------------------------
# NOTEBOOK-SAFE ARGS
# ------------------------------------------------------------------------------

def parse_args_notebook_safe() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--hours", type=int, default=200)
    p.add_argument("--gfs-stride", type=int, default=3)
    p.add_argument("--icon-stride", type=int, default=3)
    p.add_argument("--ecmwf-source", type=str, default="aws", choices=["ecmwf", "aws", "azure", "google"])

    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--max-gfs-files", type=int, default=9999)
    p.add_argument("--sleep", type=float, default=0.15, help="Sleep between HTTP requests.")

    # Output controls
    p.add_argument("--native-only", action="store_true", default=False, help="Write native model timestamps (no 15-min grid).")
    p.add_argument("--no-interpolate", action="store_true", default=False, help="Do not interpolate on 15-min grid.")
    p.add_argument("--align", type=str, default="latest", choices=["earliest", "latest"], help="Start alignment for 15-min grid.")
    p.add_argument("--out", type=str, default="", help="Output CSV path (optional).")

    args, _unknown = p.parse_known_args()
    return args


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main() -> Path:
    args = parse_args_notebook_safe()
    setup_logging(args.debug)

    LOGGER.info(f"Args: {args}")

    now_utc = utcnow_naive()

    # -------------------------
    # GFS
    # -------------------------
    gfs_init = pick_latest_available_gfs_cycle(now_utc, debug=args.debug)
    gfs_files = download_gfs(
        gfs_init,
        hours=args.hours,
        stride=args.gfs_stride,
        debug=args.debug,
        max_files=args.max_gfs_files,
        sleep_s=args.sleep,
    )
    gfs_s = gfs_process_files(gfs_files, gfs_init, debug=args.debug)

    # -------------------------
    # ECMWF
    # -------------------------
    try:
        ecmwf_s, ecmwf_init, ecmwf_stream = build_ecmwf_series(args.ecmwf_source, args.hours, debug=args.debug)
    except Exception as e:
        LOGGER.warning(f"ECMWF failed entirely: {e}")
        ecmwf_s = pd.Series(dtype=float, name="ECMWF_10m_ms")
        ecmwf_init = now_utc
        ecmwf_stream = ""

    # -------------------------
    # ICON
    # -------------------------
    try:
        icon_init = pick_latest_available_icon_cycle(now_utc, debug=args.debug)
        icon_pairs, clat_path, clon_path = download_icon(
            icon_init,
            horizon_h=args.hours,
            stride=args.icon_stride,
            debug=args.debug,
            sleep_s=args.sleep,
        )
        icon_s = icon_process_pairs(icon_pairs, icon_init, clat_path, clon_path, debug=args.debug)
    except Exception as e:
        LOGGER.warning(f"ICON failed entirely: {e}")
        icon_s = pd.Series(dtype=float, name="ICON_10m_ms")
        icon_init = now_utc

    if (gfs_s is None or gfs_s.empty) and (ecmwf_s is None or ecmwf_s.empty) and (icon_s is None or icon_s.empty):
        raise RuntimeError(
            "No model data could be processed.\n"
            "Most common fix in Colab: install eccodes:\n"
            "  pip install -q eccodes\n"
        )

    # -------------------------
    # OUTPUT DF
    # -------------------------
    if args.native_only:
        df = build_dataframe_native(gfs_s, ecmwf_s, icon_s)
        df["GFS_Init_UTC"] = gfs_init.strftime("%Y-%m-%d %H:%M:%S")
        df["ECMWF_Init_UTC"] = ecmwf_init.strftime("%Y-%m-%d %H:%M:%S")
        df["ICON_Init_UTC"] = icon_init.strftime("%Y-%m-%d %H:%M:%S")
        df["ECMWF_Stream"] = ecmwf_stream
    else:
        df = build_15min_dataframe(
            gfs=gfs_s,
            ecmwf=ecmwf_s,
            icon=icon_s,
            gfs_init=gfs_init,
            ecmwf_init=ecmwf_init,
            icon_init=icon_init,
            horizon_h=args.hours,
            align=args.align,
            no_interpolate=args.no_interpolate,
            ecmwf_stream=ecmwf_stream,
        )

    if args.out.strip():
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = CSV_DIR / f"wind_10m_forecast_{now_utc:%Y%m%d_%H%M}UTC.csv"

    df.to_csv(out, index=True)
    LOGGER.info(f"Saved CSV: {out}")
    return out


if __name__ == "__main__":
    # Notebook-safe (no SystemExit spam)
    main()
