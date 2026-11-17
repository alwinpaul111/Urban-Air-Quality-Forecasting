from src.aqi import PM25, STD_CPCB, STD_US, aqi, category, risk, sub_index


def test_sub_index_bands():
    assert sub_index(30, PM25) == 50
    assert 51 <= sub_index(30.5, PM25) <= 52
    assert round(sub_index(45, PM25)) == 75


def test_aqi_takes_max_and_handles_missing():
    assert aqi(pm25=45, pm10=300) > aqi(pm25=45)
    assert aqi(pm25=None, pm10=None) != aqi(pm25=None, pm10=None)


def test_categories_and_risk():
    assert category(35) == 'Good' and category(450) == 'Severe'
    assert risk('Poor', 'Respiratory or cardiac conditions') == 'Very High'
    assert risk('Good', 'Healthy adults') == 'Low'


def test_us_aqi_matches_reference():
    assert round(aqi(pm25=86, pm10=150, std=STD_US)) in (172, 173)   # aqi.in showed 173 for PM2.5 86
    assert category(173, STD_US) == 'Unhealthy'
    assert risk('Unhealthy', 'Healthy adults', STD_US) == 'Moderate'
    assert aqi(pm25=86, pm10=150, std=STD_CPCB) > 150
