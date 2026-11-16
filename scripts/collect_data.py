"""
collect_data.py - Data collection for Urban Air Quality Intelligence & Health Risk Prediction

Sources
  1. OpenAQ v3   : hourly pollutant readings from ground stations (incl. CPCB stations)
  2. Open-Meteo  : historical weather (ERA5 archive)
  3. Open-Meteo  : CAMS modelled air quality (used as a feature / gap-filler)

Setup
  pip install -r requirements.txt
  Get a free OpenAQ API key: https://explore.openaq.org  (sign up -> Settings -> API key)
  export OPENAQ_API_KEY="your_key"          # Linux/Mac
  set OPENAQ_API_KEY=your_key               # Windows cmd
  # Colab: os.environ["OPENAQ_API_KEY"] = "your_key"

Usage
  python collect_data.py --start 2024-01-01 --end 2026-10-01
  python collect_data.py --start 2025-10-01 --end 2026-01-31 --cities Ghaziabad Noida
  python collect_data.py --skip-openaq        # only weather + CAMS (no key needed)

Outputs (data/raw/ and data/processed/)
  openaq_hourly.csv, weather_hourly.csv, cams_hourly.csv, merged_hourly.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CITIES = {
    "Ghaziabad": (28.6692, 77.4538),
    "Noida": (28.5355, 77.3910),
    "Delhi": (28.6139, 77.2090),
    "Gurugram": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178),
}

POLLUTANTS = {"pm25", "pm10"}

OPENAQ_BASE = "https://api.openaq.org/v3"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CAMS_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]
CAMS_VARS = ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide"]

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
TAG = ""


# --------------------------------------------------------------------------- #
# HTTP helper with retry / backoff
# --------------------------------------------------------------------------- #
def get_json(url, params=None, headers=None, retries=5, pause=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            if r.status_code == 429:  # rate limited
                wait = 10 * (attempt + 1)
                print(f"    rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"    request failed ({e}); retry in {wait}s")
            time.sleep(wait)
    print(f"    giving up on {url}")
    return None


def date_chunks(start, end, days):
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    while s <= e:
        c_end = min(s + timedelta(days=days - 1), e)
        yield s.date().isoformat(), c_end.date().isoformat()
        s = c_end + timedelta(days=1)


# --------------------------------------------------------------------------- #
# 1. OpenAQ
# --------------------------------------------------------------------------- #
def find_sensors(city, lat, lon, headers, radius=12000):
    """Return list of dicts: location_id, location_name, lat, lon, sensor_id, parameter."""
    out = []
    page = 1
    while True:
        data = get_json(
            f"{OPENAQ_BASE}/locations",
            params={"coordinates": f"{lat},{lon}", "radius": radius, "limit": 100, "page": page},
            headers=headers,
        )
        if not data or not data.get("results"):
            break
        for loc in data["results"]:
            coords = loc.get("coordinates") or {}
            for s in loc.get("sensors", []):
                pname = (s.get("parameter") or {}).get("name", "").lower()
                if pname in POLLUTANTS:
                    out.append(
                        {
                            "city": city,
                            "location_id": loc["id"],
                            "location_name": loc.get("name"),
                            "lat": coords.get("latitude"),
                            "lon": coords.get("longitude"),
                            "sensor_id": s["id"],
                            "parameter": pname,
                        }
                    )
        if len(data["results"]) < 100:
            break
        page += 1
    return out


def fetch_sensor_hours(sensor_id, start, end, headers):
    rows = []
    for c_start, c_end in date_chunks(start, end, 90):
        page = 1
        while True:
            data = get_json(
                f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                params={
                    "datetime_from": f"{c_start}T00:00:00Z",
                    "datetime_to": f"{c_end}T23:59:59Z",
                    "limit": 1000,
                    "page": page,
                },
                headers=headers,
            )
            if not data or not data.get("results"):
                break
            for r in data["results"]:
                ts = (((r.get("period") or {}).get("datetimeFrom")) or {}).get("utc")
                rows.append({"timestamp": ts, "value": r.get("value")})
            if len(data["results"]) < 1000:
                break
            page += 1
    return rows


def collect_openaq(cities, start, end):
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY not set. Get a free key at https://explore.openaq.org")
    headers = {"X-API-Key": key}

    all_rows = []
    for city in cities:
        lat, lon = CITIES[city]
        sensors = find_sensors(city, lat, lon, headers)
        print(f"[OpenAQ] {city}: {len(sensors)} sensors found")
        for i, s in enumerate(sensors, 1):
            print(f"  ({i}/{len(sensors)}) {s['location_name']} - {s['parameter']}")
            for row in fetch_sensor_hours(s["sensor_id"], start, end, headers):
                all_rows.append({**s, **row})

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[OpenAQ] no data returned")
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "value"])
    df = df[df["value"] >= 0]  # drop sentinel negative values
    df.to_csv(RAW_DIR / f"openaq_hourly{TAG}.csv", index=False)
    print(f"[OpenAQ] saved {len(df):,} rows")
    return df


# --------------------------------------------------------------------------- #
# 2. Open-Meteo weather (ERA5) and 3. CAMS air quality
# --------------------------------------------------------------------------- #
def collect_open_meteo(url, variables, cities, start, end, label, chunk_days):
    frames = []
    for city in cities:
        lat, lon = CITIES[city]
        print(f"[{label}] {city}")
        for c_start, c_end in date_chunks(start, end, chunk_days):
            data = get_json(
                url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": c_start,
                    "end_date": c_end,
                    "hourly": ",".join(variables),
                    "timezone": "UTC",
                },
                pause=0.5,
            )
            if not data or "hourly" not in data:
                continue
            d = pd.DataFrame(data["hourly"])
            d["city"] = city
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    return df


# --------------------------------------------------------------------------- #
# 4. Merge
# --------------------------------------------------------------------------- #
def build_merged(aq, weather, cams):
    if aq.empty:
        print("[merge] no OpenAQ data; skipping merged file")
        return
    aq = aq.copy()
    aq["timestamp"] = aq["timestamp"].dt.floor("h")
    wide = (
        aq.pivot_table(
            index=["city", "location_id", "location_name", "lat", "lon", "timestamp"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None

    if not weather.empty:
        wide = wide.merge(weather, on=["city", "timestamp"], how="left")
    if not cams.empty:
        cams = cams.rename(columns={c: f"cams_{c}" for c in CAMS_VARS if c in cams.columns})
        wide = wide.merge(cams, on=["city", "timestamp"], how="left")

    wide = wide.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    wide.to_csv(PROC_DIR / f"merged_hourly{TAG}.csv", index=False)
    print(f"[merge] saved {len(wide):,} rows x {wide.shape[1]} cols -> merged_hourly.csv")


# --------------------------------------------------------------------------- #
def main():
    global TAG
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--cities", nargs="+", default=list(CITIES), choices=list(CITIES))
    p.add_argument("--skip-openaq", action="store_true")
    p.add_argument("--tag", default="", help="suffix added to output filenames")
    args = p.parse_args()
    TAG = args.tag

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    weather = collect_open_meteo(WEATHER_URL, WEATHER_VARS, args.cities, args.start, args.end, "Weather", 365)
    if not weather.empty:
        weather.to_csv(RAW_DIR / f"weather_hourly{TAG}.csv", index=False)
        print(f"[Weather] saved {len(weather):,} rows")

    cams = collect_open_meteo(CAMS_URL, CAMS_VARS, args.cities, args.start, args.end, "CAMS", 90)
    if not cams.empty:
        cams.to_csv(RAW_DIR / f"cams_hourly{TAG}.csv", index=False)
        print(f"[CAMS] saved {len(cams):,} rows")

    if args.skip_openaq:
        return
    aq = collect_openaq(args.cities, args.start, args.end)
    build_merged(aq, weather, cams)


if __name__ == "__main__":
    main()
