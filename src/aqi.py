"""AQI calculation (CPCB India and US EPA, particulate matter only) and the health-risk mapping."""
import math

STD_CPCB, STD_US = 'CPCB (India)', 'US EPA'
STANDARDS = [STD_US, STD_CPCB]

# (conc_low, conc_high, aqi_low, aqi_high) in ug/m3
PM25 = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (250, 500, 401, 500)]
PM10 = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (430, 600, 401, 500)]
US_PM25 = [(0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
           (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]
US_PM10 = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
           (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)]
BREAKS = {STD_CPCB: {'pm25': PM25, 'pm10': PM10}, STD_US: {'pm25': US_PM25, 'pm10': US_PM10}}

CATS = {STD_CPCB: ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'],
        STD_US: ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']}
SHORT = {STD_CPCB: ['Good', 'Satisf.', 'Moderate', 'Poor', 'V. Poor', 'Severe'],
         STD_US: ['Good', 'Moderate', 'Sensitive', 'Unhealthy', 'V. Unhealthy', 'Hazardous']}
LIMITS = {STD_CPCB: [50, 100, 200, 300, 400], STD_US: [50, 100, 150, 200, 300]}
COLORS = {STD_CPCB: ['#2E9E5B', '#8CC152', '#F2C230', '#F28C28', '#E04B3A', '#8E1B2E'],
          STD_US: ['#2E9E5B', '#F2C230', '#F28C28', '#E04B3A', '#8E44AD', '#8E1B2E']}
GREY = '#9AA5B1'

GROUPS = ['Healthy adults', 'Children and older adults', 'Respiratory or cardiac conditions']
RISK_COLORS = {'Low': '#2E9E5B', 'Moderate': '#F2C230', 'High': '#F28C28', 'Very High': '#E04B3A', 'Unavailable': GREY}
_RISK = [('Low', 'Low', 'Low'), ('Low', 'Low', 'Moderate'), ('Low', 'Moderate', 'High'),
         ('Moderate', 'High', 'Very High'), ('High', 'Very High', 'Very High'),
         ('Very High', 'Very High', 'Very High')]
ADVICE = {
    'Low': 'Air quality is fine for normal outdoor activity.',
    'Moderate': 'Fine for most activities. Shorten hard outdoor exercise if you notice any irritation.',
    'High': 'Cut down on long outdoor exertion, keep windows closed in peak hours, and wear a well-fitted N95 mask outdoors.',
    'Very High': 'Stay indoors where possible, skip outdoor exercise, keep any prescribed medication handy, and see a doctor if symptoms get worse.',
    'Unavailable': 'No forecast available.',
}


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def sub_index(conc, table):
    if _nan(conc):
        return float('nan')
    conc = max(0.0, float(conc))
    for lo, hi, ilo, ihi in table:
        if conc <= hi:
            return min(500.0, max(float(ilo), ilo + (ihi - ilo) * (conc - lo) / (hi - lo)))
    return 500.0


def aqi(pm25=None, pm10=None, std=STD_CPCB):
    """AQI = highest sub-index of the particulates available (needs at least one)."""
    b = BREAKS[std]
    vals = [v for v in (sub_index(pm25, b['pm25']), sub_index(pm10, b['pm10'])) if not math.isnan(v)]
    return max(vals) if vals else float('nan')


def level(a, std=STD_CPCB):
    if _nan(a):
        return -1
    for i, t in enumerate(LIMITS[std]):
        if a <= t:
            return i
    return 5


def category(a, std=STD_CPCB):
    i = level(a, std)
    return 'Unavailable' if i < 0 else CATS[std][i]


def color(a, std=STD_CPCB):
    i = level(a, std)
    return GREY if i < 0 else COLORS[std][i]


def scale_pos(a, std=STD_CPCB):
    """Marker position (0-1) on a six-band scale bar."""
    i = level(a, std)
    if i < 0:
        return 0.0
    bounds = [0] + LIMITS[std] + [LIMITS[std][-1] * 1.5]
    frac = min(1.0, max(0.0, (a - bounds[i]) / (bounds[i + 1] - bounds[i])))
    return (i + frac) / 6


def risk(cat, group, std=STD_CPCB):
    if cat not in CATS[std] or group not in GROUPS:
        return 'Unavailable'
    return _RISK[CATS[std].index(cat)][GROUPS.index(group)]
