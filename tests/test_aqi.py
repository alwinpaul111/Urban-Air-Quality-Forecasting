from src.aqi import aqi, category, risk, sub_index, PM25


def test_sub_index_bands():
    assert sub_index(30, PM25) == 50
    assert 51 <= sub_index(30.5, PM25) <= 52          # gap between bands handled
    assert round(sub_index(45, PM25)) == 75


def test_aqi_takes_max_and_handles_missing():
    assert aqi(pm25=45, pm10=300) > aqi(pm25=45)
    assert aqi(pm25=None, pm10=None) != aqi(pm25=None, pm10=None)  # NaN


def test_categories_and_risk():
    assert category(35) == 'Good' and category(450) == 'Severe'
    assert risk('Poor', 'Respiratory or cardiac conditions') == 'Very High'
    assert risk('Good', 'Healthy adults') == 'Low'
