"""Tests for sensor-family thermal features."""

import pandas as pd
import pytest

from sardinia_fire.features.sensor_thermal import (
    extract_sensor_thermal_features,
)


def make_data():
    return pd.DataFrame({
        "cluster_id": [0, 0, 0, 1],
        "detection_id": ["a", "b", "c", "d"],
        "acq_datetime_utc": pd.to_datetime([
            "2024-07-20T10:00:00Z",
            "2024-07-20T10:05:00Z",
            "2024-07-20T10:10:00Z",
            "2024-07-20T11:00:00Z",
        ], utc=True),
        "source_product": [
            "VIIRS_NOAA20_SP",
            "VIIRS_NOAA20_SP",
            "MODIS_SP",
            "VIIRS_SNPP_SP",
        ],
        "satellite": ["N20", "N20", "Terra", "NPP"],
        "frp_mw": [10.0, 30.0, 100.0, 20.0],
    })


def test_sensor_separation():
    result = extract_sensor_thermal_features(make_data())
    event = result.loc[result["cluster_id"] == 0].iloc[0]

    assert event["viirs_detections"] == 2
    assert event["modis_detections"] == 1
    assert event["viirs_groups"] == 1
    assert event["modis_groups"] == 1
    assert event["viirs_frp_median_mw"] == pytest.approx(20)
    assert event["modis_frp_median_mw"] == pytest.approx(100)
    assert event["viirs_group_peak_mean_mw"] == pytest.approx(30)


def test_missing_family_is_nan():
    result = extract_sensor_thermal_features(make_data())
    event = result.loc[result["cluster_id"] == 1].iloc[0]

    assert event["modis_detections"] == 0
    assert event["modis_groups"] == 0
    assert pd.isna(event["modis_frp_p90_mw"])


def test_unknown_source():
    data = make_data()
    data.loc[0, "source_product"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unknown"):
        extract_sensor_thermal_features(data)


def test_invalid_interval():
    with pytest.raises(ValueError):
        extract_sensor_thermal_features(make_data(), group_minutes=0)


def test_duplicate_detection():
    data = make_data()
    data.loc[1, "detection_id"] = "a"

    with pytest.raises(ValueError, match="Duplicate"):
        extract_sensor_thermal_features(data)
