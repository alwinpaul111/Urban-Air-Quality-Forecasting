# Urban Air Quality Intelligence & Health Risk Prediction

Streamlit dashboard for Delhi-NCR (Ghaziabad, Noida, Delhi, Gurugram, Faridabad) that turns a 4-day air-quality outlook into AQI categories and **health-risk levels for three sensitivity groups**, with prediction intervals and feature-level explanations. 

## How it works
- `src/data.py`: live weather + CAMS air-quality forecasts from Open-Meteo (no API key).
- `train.py`: LightGBM models learn to correct CAMS forecasts towards OpenAQ ground-station readings (mean + 10th/90th percentile models, time-based split).
- `src/aqi.py`: CPCB AQI (PM2.5/PM10) and the AQI-to-risk mapping.
- `app.py`: dashboard. **Without trained models it shows the CAMS modelled outlook; with models in `models/` it switches to the ML forecast with intervals and explanations.**

## 1. Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. Train the ML models (Google Colab)
```python
from google.colab import drive, userdata
import os
drive.mount('/content/drive')
%cd /content/drive/MyDrive/aqi-capstone       # folder holding this repo
os.environ['OPENAQ_API_KEY'] = userdata.get('OPENAQ_API_KEY')
!pip install -q -r requirements.txt
!python scripts/collect_data.py --start 2024-01-01 --end 2026-10-31 --cities Ghaziabad --tag _ghaziabad
# repeat for Noida, Delhi, Gurugram, Faridabad, then:
import glob, pandas as pd
fs = [f for f in glob.glob('data/processed/merged_hourly_*.csv') if '_all' not in f]
pd.concat(pd.read_csv(f) for f in fs).to_csv('data/merged_hourly_all.csv', index=False)
!python train.py --data data/merged_hourly_all.csv
```
Training prints test metrics against the raw CAMS baseline and writes `models/*.txt` and `models/meta.json`. **Commit the `models/` folder** so the deployed app uses it. Put the real numbers in your report (Chapter 5).

## 3. Push to GitHub
```bash
git init
git add .
git commit -m "Air quality intelligence dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
(Create the empty repo on github.com first. `data/` is git-ignored.)

## 4. Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, pick your repo, branch `main`, main file `app.py`.
3. Optionally set a custom subdomain, then **Deploy**. The first build takes a few minutes.
4. Copy the `https://<name>.streamlit.app` link into your report and presentation.

No secrets are needed: the live app only calls Open-Meteo, which is keyless. The OpenAQ key is used only when collecting training data.

## Tests
```bash
pip install pytest && python -m pytest tests
```

## Limits
Station data is uneven, CAMS/ERA5 are coarse, and forecasts are weakest in extreme episodes. Health advice is general guidance, not medical advice.
