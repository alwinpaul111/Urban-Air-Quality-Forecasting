"""Latest ground-station PM readings from OpenAQ (needs a free API key stored as a secret)."""
import os
import statistics

import pandas as pd
import requests

BASE = 'https://api.openaq.org/v3'
DEFAULT_IDS = {'pm10': 1, 'pm25': 2}


def _key():
    k = os.environ.get('OPENAQ_API_KEY')
    if not k:
        try:
            import streamlit as st
            k = st.secrets.get('OPENAQ_API_KEY')
        except Exception:
            k = None
    return k


def _param_ids(headers):
    ids = dict(DEFAULT_IDS)
    try:
        r = requests.get(f'{BASE}/parameters', headers=headers, params={'limit': 200}, timeout=20)
        r.raise_for_status()
        for p in r.json().get('results', []):
            if p.get('name') in ids and p.get('id'):
                ids[p['name']] = p['id']
    except Exception:
        pass
    return ids


def latest_pm(lat, lon, radius=15000, max_age_hours=6):
    """Median of the latest PM2.5 / PM10 readings from stations near (lat, lon), or None."""
    key = _key()
    if not key:
        return None
    headers = {'X-API-Key': key}
    now = pd.Timestamp.now(tz='UTC')
    out, stamps = {}, []
    try:
        ids = _param_ids(headers)
        for name in ('pm25', 'pm10'):
            r = requests.get(f'{BASE}/parameters/{ids[name]}/latest', headers=headers, timeout=30,
                             params={'coordinates': f'{lat},{lon}', 'radius': radius, 'limit': 100})
            r.raise_for_status()
            vals = []
            for x in r.json().get('results', []):
                v, t = x.get('value'), (x.get('datetime') or {}).get('utc')
                if v is None or not t or v < 0 or v > 1500:
                    continue
                ts = pd.Timestamp(t)
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                if (now - ts) > pd.Timedelta(hours=max_age_hours):
                    continue
                vals.append(float(v))
                stamps.append(ts)
            if vals:
                out[name] = statistics.median(vals)
                out[f'{name}_n'] = len(vals)
    except Exception:
        return None
    if not out:
        return None
    out['time'] = max(stamps)
    return out
