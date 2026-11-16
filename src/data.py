"""Live weather + CAMS air-quality forecasts from Open-Meteo (free, no API key)."""
import time

import pandas as pd
import requests

CITIES = {
    'Ghaziabad': (28.6692, 77.4538), 'Noida': (28.5355, 77.3910), 'Delhi': (28.6139, 77.2090),
    'Gurugram': (28.4595, 77.0266), 'Faridabad': (28.4089, 77.3178),
}
AQ_URL = 'https://air-quality-api.open-meteo.com/v1/air-quality'
WX_URL = 'https://api.open-meteo.com/v1/forecast'
AQ_VARS = ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone', 'carbon_monoxide', 'sulphur_dioxide']
WX_VARS = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'surface_pressure',
           'precipitation', 'wind_speed_10m', 'wind_direction_10m', 'cloud_cover']


def _get(url, params, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f'Could not reach {url}')


def fetch_city(city, past_days=3, forecast_days=4):
    lat, lon = CITIES[city]
    base = {'latitude': lat, 'longitude': lon, 'timezone': 'UTC',
            'past_days': past_days, 'forecast_days': forecast_days}
    aq = pd.DataFrame(_get(AQ_URL, {**base, 'hourly': ','.join(AQ_VARS)})['hourly'])
    wx = pd.DataFrame(_get(WX_URL, {**base, 'hourly': ','.join(WX_VARS)})['hourly'])
    aq = aq.rename(columns={c: f'cams_{c}' for c in AQ_VARS})
    df = aq.merge(wx, on='time')
    df.index = pd.to_datetime(df.pop('time'), utc=True)
    df['city'] = city
    return df
