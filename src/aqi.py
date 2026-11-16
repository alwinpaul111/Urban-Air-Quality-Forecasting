"""CPCB-style AQI (particulate matter only) and the project's health-risk mapping."""
import math

# (conc_low, conc_high, aqi_low, aqi_high), 24-hour averages in ug/m3
PM25 = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (250, 500, 401, 500)]
PM10 = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (430, 600, 401, 500)]

CAT_COLORS = {'Good': '#2E9E5B', 'Satisfactory': '#8CC152', 'Moderate': '#F2C230',
              'Poor': '#F28C28', 'Very Poor': '#E04B3A', 'Severe': '#8E1B2E',
              'Unavailable': '#9AA5B1'}
RISK_COLORS = {'Low': '#2E9E5B', 'Moderate': '#F2C230', 'High': '#F28C28', 'Very High': '#C0392B',
               'Unavailable': '#9AA5B1'}
GROUPS = ['Healthy adults', 'Children and older adults', 'Respiratory or cardiac conditions']

# AQI category -> risk level for each group (design table 3.3 in the report)
_RISK = {
    'Good': ('Low', 'Low', 'Low'),
    'Satisfactory': ('Low', 'Low', 'Moderate'),
    'Moderate': ('Low', 'Moderate', 'High'),
    'Poor': ('Moderate', 'High', 'Very High'),
    'Very Poor': ('High', 'Very High', 'Very High'),
    'Severe': ('Very High', 'Very High', 'Very High'),
}
ADVICE = {
    'Low': 'Air quality is fine for normal outdoor activity.',
    'Moderate': 'Fine for most activities. Shorten hard outdoor exercise if you notice any irritation.',
    'High': 'Cut down on long outdoor exertion, keep windows closed in peak hours, and wear a well-fitted N95 mask outdoors.',
    'Very High': 'Stay indoors where possible, skip outdoor exercise, keep any prescribed medication handy, and see a doctor if symptoms get worse.',
    'Unavailable': 'No forecast available.',
}


def sub_index(conc, table):
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return float('nan')
    conc = max(0.0, float(conc))
    for lo, hi, ilo, ihi in table:
        if conc <= hi:
            return min(500.0, max(float(ilo), ilo + (ihi - ilo) * (conc - lo) / (hi - lo)))
    return 500.0


def aqi(pm25=None, pm10=None):
    """AQI = highest sub-index of the particulates available (needs at least one)."""
    vals = [v for v in (sub_index(pm25, PM25), sub_index(pm10, PM10)) if not math.isnan(v)]
    return max(vals) if vals else float('nan')


def category(a):
    if a is None or math.isnan(a):
        return 'Unavailable'
    for limit, name in [(50, 'Good'), (100, 'Satisfactory'), (200, 'Moderate'),
                        (300, 'Poor'), (400, 'Very Poor')]:
        if a <= limit:
            return name
    return 'Severe'


def risk(cat, group):
    if cat not in _RISK:
        return 'Unavailable'
    return _RISK[cat][GROUPS.index(group)]
