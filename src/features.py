"""Feature construction shared by training and the app (only inputs known at forecast time)."""
import numpy as np
import pandas as pd

BASE = ['cams_pm2_5', 'cams_pm10', 'cams_nitrogen_dioxide', 'cams_ozone', 'cams_carbon_monoxide',
        'cams_sulphur_dioxide', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'surface_pressure', 'precipitation', 'wind_speed_10m', 'cloud_cover']
FEATURES = BASE + ['wind_sin', 'wind_cos', 'hour', 'month', 'dayofweek', 'city_code']
LABELS = {
    'cams_pm2_5': 'modelled CAMS PM2.5', 'cams_pm10': 'modelled CAMS PM10',
    'cams_nitrogen_dioxide': 'modelled NO2', 'cams_ozone': 'modelled ozone',
    'cams_carbon_monoxide': 'modelled CO', 'cams_sulphur_dioxide': 'modelled SO2',
    'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity', 'dew_point_2m': 'dew point',
    'surface_pressure': 'surface pressure', 'precipitation': 'rainfall', 'wind_speed_10m': 'wind speed',
    'cloud_cover': 'cloud cover', 'wind_sin': 'wind direction', 'wind_cos': 'wind direction',
    'hour': 'time of day', 'month': 'time of year', 'dayofweek': 'day of week', 'city_code': 'location',
}


def make_features(df, city_codes):
    """df: UTC DatetimeIndex with BASE columns, wind_direction_10m and a 'city' column."""
    X = df[BASE].copy()
    rad = np.deg2rad(df['wind_direction_10m'])
    X['wind_sin'], X['wind_cos'] = np.sin(rad), np.cos(rad)
    local = df.index.tz_convert('Asia/Kolkata')
    X['hour'], X['month'], X['dayofweek'] = local.hour, local.month, local.dayofweek
    X['city_code'] = df['city'].map(city_codes)
    return X[FEATURES].astype(float)
