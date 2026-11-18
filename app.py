"""Urban Air Quality Intelligence & Health Risk Prediction - Streamlit dashboard."""
import json
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src import aqi as A
from src.data import CITIES, fetch_city
from src.features import FEATURES, LABELS, make_features
from src.live import latest_pm

st.set_page_config(page_title='NCR Air Quality Intelligence', page_icon='🌫️', layout='wide')
TZ = 'Asia/Kolkata'
CARD, MUTED = '#25282E', '#9AA5B1'
st.markdown("""<style>
.hero{border-radius:20px;padding:22px 26px;color:#fff}
.hero .lbl{font-size:14px;opacity:.9}.hero .big{font-size:76px;font-weight:800;line-height:1}
.hero .pill{display:inline-block;background:rgba(0,0,0,.28);padding:6px 16px;border-radius:12px;font-size:20px;font-weight:700;margin-left:10px}
.hero .pm{font-size:17px;margin-top:10px}.hero .src{font-size:12px;opacity:.85;margin-top:8px}
.scale{position:relative;height:8px;border-radius:5px;margin:26px 0 6px}.marker{position:absolute;top:-5px;width:16px;height:16px;border-radius:50%;background:#fff;border:3px solid #222;transform:translateX(-50%)}
.ticks{display:flex;font-size:11px;opacity:.9}.ticks div{flex:1;text-align:center}
.wx{background:#25282E;border-radius:20px;padding:22px;height:100%;color:#E8EAED}.wx .t{font-size:38px;font-weight:700}
.wx .row{display:flex;justify-content:space-between;margin-top:16px;font-size:13px;color:#9AA5B1}.wx b{color:#E8EAED;font-size:16px}
.pcard{background:#25282E;border-radius:14px;padding:16px 18px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
.pcard .n{font-size:15px;font-weight:600;max-width:60%}.pcard .tag{font-size:11px;color:#9AA5B1}.pcard .v{font-size:28px;font-weight:700;text-align:right}.pcard .u{font-size:13px;color:#9AA5B1}
.dcard{border-radius:14px;padding:14px;background:#25282E}.dcard .d{font-weight:600}.dcard .a{font-size:32px;font-weight:800}
.hbox{border-radius:16px;overflow:hidden;display:flex;background:#25282E}.hleft{padding:26px;min-width:240px;color:#fff;display:flex;flex-direction:column;justify-content:flex-end}
.hright{padding:24px 28px;font-size:14px}.hright h4{margin:14px 0 6px}.hright li{margin-bottom:6px}
.pcard,.dcard,.hbox,.hright,.wx{color:#E8EAED}.hright h4{color:#E8EAED}
.chip{display:inline-block;color:#fff;padding:2px 9px;border-radius:10px;font-size:12px;margin-right:6px}
</style>""", unsafe_allow_html=True)

CONDITIONS = {
    'Asthma': (2, ['Keep your prescribed inhaler or medication handy and take it as directed by your healthcare provider.',
                   'Keep windows closed during high-pollution hours and use an air purifier indoors if you have one.',
                   'Wear a well-fitted N95 mask if you must go outside.'],
               ['Do hard exercise outdoors.', 'Smoke or stay around smoke.', 'Ignore worsening wheezing or breathlessness: seek medical care.']),
    'Heart issues': (2, ['Limit time outdoors and avoid busy roads.', 'Take prescribed medicines as directed.',
                         'Rest and get medical help if you feel chest discomfort or unusual breathlessness.'],
                     ['Do heavy exertion outdoors.', 'Ignore new or worsening symptoms.']),
    'Allergies': (1, ['Keep windows closed in peak pollution hours.', 'Shower and change clothes after being outdoors.',
                      'Use clean, filtered air indoors where possible.'],
                  ['Leave windows open when air is bad.', 'Rub irritated eyes.']),
    'Sinus': (1, ['Stay hydrated and keep indoor air clean.', 'Follow any nasal-care routine your doctor has advised.',
                  'Wear a mask in dusty or smoky places.'],
              ['Spend long periods in smoky or dusty areas.', 'Use strong sprays or fragrances indoors.']),
    'Cold / Flu': (1, ['Rest, drink fluids and keep indoor air clean.', 'Wear a mask outdoors on bad air days.',
                       'See a doctor if fever or breathing trouble persists.'],
                   ['Exercise strenuously outdoors while unwell.', 'Stay in crowded, poorly ventilated places on bad air days.']),
    'Chronic (COPD)': (2, ['Follow your action plan and take prescribed medication as directed.',
                           'Stay indoors with clean air on bad days; wear an N95 mask if you must go out.',
                           'Seek care early if your breathing gets worse.'],
                       ['Do exertion outdoors.', 'Smoke or stay around smoke.', 'Delay getting help if breathing worsens.']),
}
UG_TO_PPB = {'co': 1 / 1.146, 'so2': 1 / 2.62, 'no2': 1 / 1.882, 'o3': 1 / 1.963}


@st.cache_resource
def load_models():
    meta_path = Path('models/meta.json')
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    boosters = {f'{p}_{k}': lgb.Booster(model_file=f'models/{p}_{k}.txt') for p in ('pm25', 'pm10') for k in ('mean', 'lo', 'hi')}
    return meta, boosters


@st.cache_data(ttl=3600, show_spinner='Fetching latest forecasts...')
def get_city(city):
    return fetch_city(city)


@st.cache_data(ttl=600, show_spinner=False)
def get_live(city):
    return latest_pm(*CITIES[city])


def predict(df, bundle):
    out = pd.DataFrame({'pm25': df['cams_pm2_5'], 'pm10': df['cams_pm10']}, index=df.index)
    contrib = None
    if bundle:
        meta, b = bundle
        X = make_features(df, meta['city_codes'])
        for p in ('pm25', 'pm10'):
            mean = np.clip(b[f'{p}_mean'].predict(X), 0, None)
            out[p] = mean
            out[f'{p}_lo'] = np.minimum(np.clip(b[f'{p}_lo'].predict(X), 0, None), mean)
            out[f'{p}_hi'] = np.maximum(np.clip(b[f'{p}_hi'].predict(X), 0, None), mean)
        contrib = pd.DataFrame(b['pm25_mean'].predict(X, pred_contrib=True)[:, :-1], columns=FEATURES, index=df.index)
    return out, contrib


def anchor(out, live, now):
    """Scale forecasts by the latest ground/model ratio; the correction fades over ~36 hours."""
    out = out.copy()
    i = out.index.get_indexer([now], method='nearest')[0]
    h = ((out.index - now).total_seconds() / 3600).to_numpy()
    w = np.where(h >= 0, np.exp(-np.clip(h, 0, None) / 36), 0.0)
    for p in ('pm25', 'pm10'):
        if p in live and out[p].iloc[i] > 1:
            f = float(np.clip(live[p] / out[p].iloc[i], 0.25, 4))
            for c in (p, f'{p}_lo', f'{p}_hi'):
                if c in out:
                    out[c] = out[c] * (1 + (f - 1) * w)
    return out


def daily(out, std):
    local = out.index.tz_convert(TZ)
    g = out.groupby(local.date)
    d = g.mean()
    d = d[(g.size() >= 20) & (d.index >= pd.Timestamp.now(tz=TZ).date())].head(4)
    d['aqi'] = [A.aqi(r.pm25, r.pm10, std) for r in d.itertuples()]
    if 'pm25_lo' in d:
        d['aqi_lo'] = [A.aqi(r.pm25_lo, r.pm10_lo, std) for r in d.itertuples()]
        d['aqi_hi'] = [A.aqi(r.pm25_hi, r.pm10_hi, std) for r in d.itertuples()]
    d['cat'] = [A.category(a, std) for a in d['aqi']]
    return d


bundle = load_models()
with st.sidebar:
    st.header('Settings')
    city = st.selectbox('City', list(CITIES))
    std = st.selectbox('AQI standard', A.STANDARDS, help='aqi.in shows US EPA AQI by default; India uses the CPCB scale.')
    group = st.selectbox('Forecast cards are for', A.GROUPS)
    if st.button('Refresh data'):
        st.cache_data.clear()
        st.rerun()
    st.caption('General guidance only, not medical advice.')

st.title(f'{city} Air Quality Index (AQI)')
st.caption('Urban Air Quality Intelligence & Health Risk Prediction | Delhi-NCR | SDG 3, 11, 13')

try:
    df = get_city(city)
    out, contrib = predict(df, bundle)
except Exception as e:
    st.error(f'Could not load forecast data: {e}')
    st.stop()

now = pd.Timestamp.now(tz='UTC').floor('h')
live = get_live(city)
if live:
    out = anchor(out, live, now)
ci = out.index.get_indexer([now], method='nearest')[0]
cur_df, cur = df.iloc[ci], out.iloc[ci]
pm25_now = live['pm25'] if live and 'pm25' in live else float(cur['pm25'])
pm10_now = live['pm10'] if live and 'pm10' in live else float(cur['pm10'])
a_now = A.aqi(pm25_now, pm10_now, std)
col_now, cat_now = A.color(a_now, std), A.category(a_now, std)
if live:
    src = f"Ground stations (OpenAQ): median of {max(live.get('pm25_n', 0), live.get('pm10_n', 0))} nearby stations, latest reading {live['time'].tz_convert(TZ):%d %b %H:%M} IST"
else:
    src = 'Estimated now from the ML-corrected CAMS model. Add an OpenAQ key as a secret for live ground readings.'

tab1, tab2, tab3 = st.tabs(['City dashboard', 'Region overview', 'About & method'])

with tab1:
    c1, c2 = st.columns([3, 1.4])
    ticks = ''.join(f'<div>{s}</div>' for s in A.SHORT[std])
    grad = ','.join(f'{c} {i * 100 / 6:.1f}% {(i + 1) * 100 / 6:.1f}%' for i, c in enumerate(A.COLORS[std]))
    c1.markdown(
        f"<div class='hero' style='background:linear-gradient(135deg,{col_now},{col_now}99)'>"
        f"<div class='lbl'>● {'LIVE' if live else 'ESTIMATED'} AQI ({'US' if std == A.STD_US else 'India'})</div>"
        f"<div><span class='big'>{a_now:.0f}</span><span class='pill'>{cat_now}</span></div>"
        f"<div class='pm'>PM2.5: <b>{pm25_now:.0f}</b> µg/m³ &nbsp;&nbsp; PM10: <b>{pm10_now:.0f}</b> µg/m³</div>"
        f"<div class='scale' style='background:linear-gradient(90deg,{grad})'><div class='marker' style='left:{A.scale_pos(a_now, std) * 100:.1f}%'></div></div>"
        f"<div class='ticks'>{ticks}</div><div class='src'>{src}</div></div>", unsafe_allow_html=True)
    c2.markdown(
        f"<div class='wx'><div class='t'>{cur_df['temperature_2m']:.0f}°C</div><div style='color:{MUTED}'>Current weather (model)</div>"
        f"<div class='row'><div>Humidity<br><b>{cur_df['relative_humidity_2m']:.0f}%</b></div>"
        f"<div>Wind<br><b>{cur_df['wind_speed_10m']:.1f} km/h</b></div></div></div>", unsafe_allow_html=True)

    st.subheader('Major air pollutants')
    def pcard(name, val, unit, tag, c):
        return (f"<div class='pcard' style='border-left:6px solid {c}'><div><div class='n'>{name}</div><div class='tag'>{tag}</div></div>"
                f"<div><div class='v'>{val:.0f}</div><div class='u'>{unit}</div></div></div>")
    sub25 = A.color(A.sub_index(pm25_now, A.BREAKS[std]['pm25']), std)
    sub10 = A.color(A.sub_index(pm10_now, A.BREAKS[std]['pm10']), std)
    tag_pm = 'ground stations' if live else 'estimated'
    gases = [('Carbon Monoxide (CO)', 'cams_carbon_monoxide', 'co'), ('Sulfur Dioxide (SO₂)', 'cams_sulphur_dioxide', 'so2'),
             ('Nitrogen Dioxide (NO₂)', 'cams_nitrogen_dioxide', 'no2'), ('Ozone (O₃)', 'cams_ozone', 'o3')]
    cards = [pcard('Particulate Matter (PM2.5)', pm25_now, 'µg/m³', tag_pm, sub25), pcard('Particulate Matter (PM10)', pm10_now, 'µg/m³', tag_pm, sub10)]
    cards += [pcard(n, cur_df[k] * UG_TO_PPB[g], 'ppb', 'modelled (CAMS)', '#4DA3FF') for n, k, g in gases]
    pc = st.columns(3)
    for i, html in enumerate(cards):
        pc[i % 3].markdown(html, unsafe_allow_html=True)
    st.caption('PM border colour shows the pollutant level on the selected AQI scale. Gas values are modelled and shown in ppb.')

    d = daily(out, std)
    st.subheader('4-day forecast')
    cols = st.columns(len(d))
    for cc, (day, r) in zip(cols, d.iterrows()):
        rk = A.risk(r['cat'], group, std)
        c = A.color(r['aqi'], std)
        cc.markdown(f"<div class='dcard' style='border-top:5px solid {c}'><div class='d'>{pd.Timestamp(day):%a %d %b}</div>"
                    f"<div class='a' style='color:{c}'>{r['aqi']:.0f}</div><div>{r['cat']}</div>"
                    f"<div style='color:{A.RISK_COLORS[rk]};font-weight:700'>Risk: {rk}</div></div>", unsafe_allow_html=True)
    sel = st.selectbox('Day for details', list(d.index), format_func=lambda x: f'{pd.Timestamp(x):%A, %d %b}')
    row = d.loc[sel]
    rk = A.risk(row['cat'], group, std)
    st.markdown(f"**Advice for {group.lower()}:** <span style='color:{A.RISK_COLORS[rk]};font-weight:700'>{rk} risk</span>", unsafe_allow_html=True)
    st.write(A.ADVICE[rk])
    if 'aqi_lo' in d:
        st.caption(f"AQI range for this day (80% interval): {row['aqi_lo']:.0f} to {row['aqi_hi']:.0f}")
    if live:
        st.caption('Forecast is anchored to the latest ground reading; this correction fades over about 36 hours.')

    cd = out.reset_index().rename(columns={out.index.name or 'index': 'time'})
    cd['time'] = cd['time'].dt.tz_convert(TZ).dt.tz_localize(None)
    base = alt.Chart(cd).encode(x=alt.X('time:T', title='Local time (IST)'))
    layers = [base.mark_line(color='#13A89E').encode(y=alt.Y('pm25:Q', title='PM2.5 (µg/m³)'))]
    if 'pm25_lo' in cd:
        layers.insert(0, base.mark_area(opacity=0.2, color='#13A89E').encode(y='pm25_lo:Q', y2='pm25_hi:Q'))
    layers.append(alt.Chart(pd.DataFrame({'y': [15, 60]})).mark_rule(strokeDash=[4, 4], color='#E76F51').encode(y='y:Q'))
    st.altair_chart(alt.layer(*layers).properties(height=280), width='stretch')
    st.caption('Hourly PM2.5. Dashed lines: WHO 24-hour guideline (15) and India 24-hour standard (60).')

    st.markdown('**Why this forecast?**')
    if contrib is not None:
        mask = out.index.tz_convert(TZ).date == sel
        cs = contrib[mask].mean().groupby(lambda f: LABELS[f]).sum().sort_values(key=abs, ascending=False).head(5)
        for name, v in cs.head(3).items():
            st.write(f"- {name.capitalize()} {'raises' if v > 0 else 'lowers'} expected PM2.5 by about {abs(v):.0f} µg/m³")
    else:
        st.write('Explanations appear once the ML models are trained and added to the repo.')

    st.subheader('Prevent health problems: understand your risks')
    cond = st.radio('Condition', list(CONDITIONS), horizontal=True, label_visibility='collapsed')
    gi, dos, donts = CONDITIONS[cond]
    hr = A.risk(cat_now, A.GROUPS[gi], std)
    hc = A.RISK_COLORS[hr]
    lo_r = {'Low': 'Low chances', 'Moderate': 'Some chance', 'High': 'High chances', 'Very High': 'Very high chances'}.get(hr, '')
    st.markdown(
        f"<div class='hbox'><div class='hleft' style='background:{hc}'><div style='font-size:13px;opacity:.9'>Today's air: {cat_now} (AQI {a_now:.0f})</div>"
        f"<div style='font-size:20px;font-weight:800'>{lo_r} of {cond.lower()} symptoms</div></div>"
        f"<div class='hright'><b style='font-size:17px'>{cond}</b><br>Risk of {cond.lower()} symptoms is "
        f"<b style='color:{hc}'>{hr}</b> at the current AQI of {a_now:.0f} ({cat_now}).<h4>Do's</h4><ul>"
        + ''.join(f'<li>✅ {t}</li>' for t in dos) + "</ul><h4>Don'ts</h4><ul>" + ''.join(f'<li>❌ {t}</li>' for t in donts) + '</ul></div></div>',
        unsafe_allow_html=True)
    st.caption('Disclaimer: these are precautionary suggestions based on air quality, not medical advice. Please consult a healthcare provider.')

with tab2:
    st.caption('City-level forecast, no ground-station anchoring.')
    when = st.radio('Show', ['Today', 'Tomorrow'], horizontal=True)
    rows = []
    for c in CITIES:
        try:
            dd = daily(predict(get_city(c), bundle)[0], std)
            i = 0 if when == 'Today' else 1
            a = dd['aqi'].iloc[i]
            rows.append({'City': c, 'lat': CITIES[c][0], 'lon': CITIES[c][1], 'AQI': round(a), 'Category': dd['cat'].iloc[i],
                         'color': [int(A.color(a, std)[j:j + 2], 16) for j in (1, 3, 5)] + [190]})
        except Exception:
            pass
    if rows:
        reg = pd.DataFrame(rows)
        reg['label'] = reg['City'] + ' ' + reg['AQI'].astype(str)
        layers = [pdk.Layer('ScatterplotLayer', reg, get_position='[lon, lat]', get_radius=7000, get_fill_color='color', pickable=True),
                  pdk.Layer('TextLayer', reg, get_position='[lon, lat]', get_text='label', get_size=15, get_color=[20, 20, 20, 255])]
        st.pydeck_chart(pdk.Deck(layers=layers, map_style='light', initial_view_state=pdk.ViewState(latitude=28.55, longitude=77.3, zoom=8.6),
                                 tooltip={'text': '{City}\nAQI {AQI} ({Category})'}))
        st.markdown(' '.join(f"<span class='chip' style='background:{c}'>{n}</span>" for n, c in zip(A.CATS[std], A.COLORS[std])), unsafe_allow_html=True)
        st.dataframe(reg[['City', 'AQI', 'Category']], hide_index=True, width='stretch')

with tab3:
    st.markdown(
        '**Live values:** when an OpenAQ key is set as a secret, the hero card and PM cards use the median of the latest ground-station readings within 15 km. '
        'Otherwise they are estimates from the ML model.  \n'
        '**Forecast:** LightGBM models trained on OpenAQ station readings correct CAMS forecasts using weather and calendar features; quantile models give an 80% interval; '
        'when live readings exist, forecasts are scaled by the latest ground/model ratio, fading over about 36 hours.  \n'
        '**AQI:** computed from PM2.5 and PM10 only, on either the US EPA scale (as on aqi.in) or the CPCB India scale, so numbers differ between the two.  \n'
        '**Health risk:** our own mapping from AQI category to risk for three sensitivity profiles.')
    if bundle:
        m = bundle[0]['metrics']
        st.dataframe(pd.DataFrame([{'Pollutant': p, 'Model MAE': v['model']['mae'], 'CAMS MAE': v['raw_cams_baseline']['mae'], 'Model R2': v['model']['r2'],
                                    '80% interval coverage': v['interval_80_coverage']} for p, v in m.items()]), hide_index=True, width='stretch')
    st.warning('General guidance only, not medical advice. Forecasts can be wrong, especially during extreme events.')
