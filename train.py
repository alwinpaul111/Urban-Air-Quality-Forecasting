"""Train LightGBM models that correct CAMS forecasts towards ground-station PM readings.

Usage:  python train.py --data data/merged_hourly_all.csv
Output: models/*.txt (LightGBM boosters) and models/meta.json (features, metrics)
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import CITIES
from src.features import BASE, FEATURES, make_features

TARGETS = {'pm25': 'cams_pm2_5', 'pm10': 'cams_pm10'}
PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=42)


def fit(Xtr, ytr, Xva, yva, **extra):
    dtr, dva = lgb.Dataset(Xtr, ytr), lgb.Dataset(Xva, yva)
    return lgb.train({**PARAMS, **extra}, dtr, 1500, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def scores(y, p):
    err = y - p
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {'mae': round(float(np.abs(err).mean()), 2), 'rmse': round(float(np.sqrt((err ** 2).mean())), 2),
            'r2': round(1 - ss_res / ss_tot, 3) if ss_tot else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='models')
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    need = ['pm25', 'pm10', 'wind_direction_10m'] + BASE
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing columns in {a.data}: {missing}')
    g = df.groupby(['city', 'timestamp'])[need].mean().reset_index().set_index('timestamp').sort_index()
    city_codes = {c: i for i, c in enumerate(CITIES)}
    g = g[g['city'].isin(city_codes)]
    X = make_features(g, city_codes)

    ts = g.index.sort_values()
    cut1, cut2 = ts[int(0.70 * len(ts))], ts[int(0.85 * len(ts))]   # time-based split
    tr, va, te = g.index < cut1, (g.index >= cut1) & (g.index < cut2), g.index >= cut2

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    meta = {'features': FEATURES, 'city_codes': city_codes, 'splits': {'train_end': str(cut1), 'val_end': str(cut2)},
            'metrics': {}}
    for pol, cams_col in TARGETS.items():
        y = g[pol]
        ok = y.notna()
        parts = {k: (X[m & ok], y[m & ok]) for k, m in (('tr', tr), ('va', va), ('te', te))}
        if len(parts['tr'][0]) < 500 or len(parts['te'][0]) < 100:
            raise SystemExit(f'Not enough {pol} rows to train (train={len(parts["tr"][0])}, test={len(parts["te"][0])})')
        models = {'mean': fit(*parts['tr'], *parts['va'], objective='regression'),
                  'lo': fit(*parts['tr'], *parts['va'], objective='quantile', alpha=0.1),
                  'hi': fit(*parts['tr'], *parts['va'], objective='quantile', alpha=0.9)}
        for k, m in models.items():
            m.save_model(str(out / f'{pol}_{k}.txt'))
        Xte, yte = parts['te']
        pred = np.clip(models['mean'].predict(Xte), 0, None)
        lo, hi = models['lo'].predict(Xte), models['hi'].predict(Xte)
        base = Xte[cams_col].fillna(yte.mean())
        meta['metrics'][pol] = {'model': scores(yte.values, pred), 'raw_cams_baseline': scores(yte.values, base.values),
                                'interval_80_coverage': round(float(((yte.values >= lo) & (yte.values <= hi)).mean()), 3),
                                'test_rows': int(len(yte))}
        print(pol, json.dumps(meta['metrics'][pol], indent=2))
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('Saved models to', out)


if __name__ == '__main__':
    main()
