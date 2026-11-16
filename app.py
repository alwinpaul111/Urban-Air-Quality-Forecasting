"""Urban Air Quality Intelligence & Health Risk Prediction - Streamlit dashboard."""
import json
from pathlib import Path

import altair as alt
import lightgbm as lgb
import pandas as pd
import streamlit as st

from src.aqi import ADVICE, CAT_COLORS, GROUPS, RISK_COLORS, aqi, category, risk
from src.data import CITIES, fetch_city
from src.features import FEATURES, LABELS, make_features

st.set_page_config(page_title='NCR Air Quality Intelligence', page_icon='🌫️', layout='wide')
TZ = 'Asia/Kolkata'


@st.cache_resource
def load_models():
    meta_path = Path('models/meta.json')
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    boosters = {f'{p}_{k}': lgb.Booster(model_file=f'models/{p}_{k}.txt')
                for p in ('pm25', 'pm10') for k in ('mean', 'lo', 'hi')}
    return meta, boosters


@st.cache_data(ttl=3600, show_spinner='Fetching latest forecasts...')
def get_city(city):
    return fetch_city(city)


def predict(df, bundle):
    """Return hourly PM estimates (+ interval and SHAP-style contributions when models exist)."""
    out = pd.DataFrame({'pm25': df['cams_pm2_5'], 'pm10': df['cams_pm10']}, index=df.index)
    contrib = None
    if bundle:
        meta, b = bundle
        X = make_features(df, meta['city_codes'])
        for p in ('pm25', 'pm10'):
            mean = b[f'{p}_mean'].predict(X).clip(0)
            out[p] = mean
            out[f'{p}_lo'] = pd.Series(b[f'{p}_lo'].predict(X), index=df.index).clip(0).combine(pd.Series(mean, index=df.index), min)
            out[f'{p}_hi'] = pd.Series(b[f'{p}_hi'].predict(X), index=df.index).clip(0).combine(pd.Series(mean, index=df.index), max)
        c = b['pm25_mean'].predict(X, pred_contrib=True)[:, :-1]
        contrib = pd.DataFrame(c, columns=FEATURES, index=df.index)
    return out, contrib


def daily(out):
    local = out.index.tz_convert(TZ)
    g = out.groupby(local.date)
    d = g.mean()
    d = d[g.size() >= 20].head(4)
    d['aqi'] = [aqi(r.pm25, r.pm10) for r in d.itertuples()]
    if 'pm25_lo' in d:
        d['aqi_lo'] = [aqi(r.pm25_lo, r.pm10_lo) for r in d.itertuples()]
        d['aqi_hi'] = [aqi(r.pm25_hi, r.pm10_hi) for r in d.itertuples()]
    d['cat'] = d['aqi'].map(category)
    return d


bundle = load_models()
st.title('🌫️ Urban Air Quality Intelligence & Health Risk Prediction')
st.caption('Delhi-NCR | 4-day outlook turned into health risk for different groups | SDG 3, 11, 13')
if bundle:
    st.success('Mode: ML forecast (LightGBM models correcting CAMS forecasts towards ground-station readings)')
else:
    st.info('Mode: CAMS modelled outlook. Train and commit the ML models (see README) to enable ML forecasts, '
            'prediction intervals and explanations.')

with st.sidebar:
    city = st.selectbox('City', list(CITIES))
    group = st.selectbox('Who is this for?', GROUPS)
    if st.button('Refresh data'):
        st.cache_data.clear()
        st.rerun()

try:
    df = get_city(city)
    out, contrib = predict(df, bundle)
    d = daily(out)
except Exception as e:  # network or API failure
    st.error(f'Could not load forecast data: {e}')
    st.stop()

tab1, tab2, tab3 = st.tabs(['Forecast & risk', 'Region overview', 'About & method'])

with tab1:
    st.subheader(f'{city}: next days at a glance')
    cols = st.columns(len(d))
    for col, (day, r) in zip(cols, d.iterrows()):
        rk = risk(r['cat'], group)
        col.markdown(
            f"<div style='border-radius:10px;padding:14px;background:{CAT_COLORS[r['cat']]}22;"
            f"border-left:6px solid {CAT_COLORS[r['cat']]}'><b>{pd.Timestamp(day):%a %d %b}</b><br>"
            f"<span style='font-size:30px;font-weight:700'>{r['aqi']:.0f}</span> AQI<br>{r['cat']}<br>"
            f"<span style='color:{RISK_COLORS[rk]};font-weight:700'>Risk: {rk}</span></div>",
            unsafe_allow_html=True)

    days = list(d.index)
    sel = st.selectbox('Day for details', days, format_func=lambda x: f'{pd.Timestamp(x):%A, %d %b}')
    row = d.loc[sel]
    rk = risk(row['cat'], group)
    st.markdown(f"### Advice for **{group.lower()}**: "
                f"<span style='color:{RISK_COLORS[rk]}'>{rk} risk</span>", unsafe_allow_html=True)
    st.write(ADVICE[rk])
    if 'aqi_lo' in d:
        st.caption(f"AQI range for this day (80% interval): {row['aqi_lo']:.0f} to {row['aqi_hi']:.0f}")

    st.markdown('**All groups on this day**')
    st.dataframe(pd.DataFrame({'Group': GROUPS, 'Risk': [risk(row['cat'], g) for g in GROUPS]}),
                 hide_index=True, width='stretch')

    st.markdown('**Hourly PM2.5 (µg/m³)**')
    cd = out.reset_index().rename(columns={out.index.name or 'index': 'time'})
    cd['time'] = cd['time'].dt.tz_convert(TZ).dt.tz_localize(None)
    base = alt.Chart(cd).encode(x=alt.X('time:T', title='Local time (IST)'))
    layers = [base.mark_line(color='#13A89E').encode(y=alt.Y('pm25:Q', title='PM2.5'))]
    if 'pm25_lo' in cd:
        layers.insert(0, base.mark_area(opacity=0.2, color='#13A89E').encode(y='pm25_lo:Q', y2='pm25_hi:Q'))
    refs = pd.DataFrame({'y': [15, 60]})
    layers.append(alt.Chart(refs).mark_rule(strokeDash=[4, 4], color='#E76F51').encode(y='y:Q'))
    st.altair_chart(alt.layer(*layers).properties(height=300), width='stretch')
    st.caption('Dashed lines: WHO 24-hour guideline (15) and India 24-hour standard (60).')

    st.markdown('**Why this forecast?**')
    if contrib is not None:
        mask = out.index.tz_convert(TZ).date == sel
        c = contrib[mask].mean().groupby(lambda f: LABELS[f]).sum().sort_values(key=abs, ascending=False).head(5)
        for name, v in c.head(3).items():
            st.write(f"- {name.capitalize()} {'raises' if v > 0 else 'lowers'} expected PM2.5 by about {abs(v):.0f} µg/m³")
        st.bar_chart(c)
    else:
        st.write('Explanations appear once the ML models are trained and added to the repo.')

with tab2:
    rows = []
    for c in CITIES:
        try:
            dd = daily(predict(get_city(c), bundle)[0])
            rows.append({'City': c, 'lat': CITIES[c][0], 'lon': CITIES[c][1],
                         'Today AQI': round(dd['aqi'].iloc[0]), 'Today': dd['cat'].iloc[0],
                         'Tomorrow AQI': round(dd['aqi'].iloc[1]), 'Tomorrow': dd['cat'].iloc[1]})
        except Exception:
            pass
    if rows:
        reg = pd.DataFrame(rows)
        st.map(reg, latitude='lat', longitude='lon')
        st.dataframe(reg.drop(columns=['lat', 'lon']), hide_index=True, width='stretch')

with tab3:
    st.markdown(
        '**Data:** hourly weather and CAMS modelled air quality from Open-Meteo (free, no key).  \n'
        '**Models:** LightGBM models trained on OpenAQ station readings learn how ground values differ from CAMS '
        'forecasts given weather and calendar features; quantile models give an 80% interval; per-feature '
        'contributions come from LightGBM (SHAP-style) values.  \n'
        '**AQI:** CPCB breakpoints for PM2.5 and PM10 (24-hour averages); the overall AQI is the higher sub-index.  \n'
        '**Risk levels:** our own mapping from AQI category to risk for three sensitivity profiles.')
    if bundle:
        st.markdown('**Held-out test results (time-based split)**')
        rows = [{'Pollutant': p, 'Model MAE': m['model']['mae'], 'CAMS MAE': m['raw_cams_baseline']['mae'],
                 'Model R2': m['model']['r2'], 'CAMS R2': m['raw_cams_baseline']['r2'],
                 '80% interval coverage': m['interval_80_coverage']} for p, m in bundle[0]['metrics'].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.warning('General guidance only, not medical advice. Forecasts can be wrong, especially during extreme events.')
