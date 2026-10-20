"""Tests for thermal temporal evolution features."""

import pandas as pd
import pytest

from sardinia_fire.features.temporal import extract_temporal_features


def make_data():
    return pd.DataFrame({
        "cluster_id": [0, 0, 0, 0, 1, -1],
        "detection_id": ["a", "b", "c", "d", "e", "noise"],
        "acq_datetime_utc": pd.to_datetime([
            "2024-07-20T10:00:00Z",
            "2024-07-20T10:05:00Z",
            "2024-07-20T12:00:00Z",
            "2024-07-20T11:00:00Z",
            "2024-07-21T10:00:00Z",
            "2024-07-20T10:00:00Z",
        ], utc=True),
        "source_product": [
            "VIIRS_NOAA20_SP", "VIIRS_NOAA20_SP",
            "VIIRS_NOAA20_SP", "MODIS_SP",
            "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP",
        ],
        "satellite": ["N20", "N20", "N20", "Terra", "NPP", "N20"],
        "frp_mw": [10.0, 30.0, 50.0, 100.0, 20.0, 999.0],
    })


def test_viirs_thermal_change():
    result = extract_temporal_features(make_data())
    event = result.loc[result["cluster_id"] == 0].iloc[0]

    assert event["viirs_frp_first_last_change_mw"] == pytest.approx(20)
    assert event["viirs_frp_group_range_mw"] == pytest.approx(20)
    assert event["viirs_median_revisit_h"] == pytest.approx(2)


def test_single_modis_group_has_no_trend():
    result = extract_temporal_features(make_data())
    event = result.loc[result["cluster_id"] == 0].iloc[0]

    assert pd.isna(event["modis_frp_first_last_change_mw"])
    assert pd.isna(event["modis_median_revisit_h"])


def test_noise_is_excluded():
    result = extract_temporal_features(make_data())

    assert len(result) == 2
    assert -1 not in result["cluster_id"].tolist()


def test_unknown_source_rejected():
    data = make_data()
    data.loc[0, "source_product"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unknown"):
        extract_temporal_features(data)


def test_invalid_interval():
    with pytest.raises(ValueError):
        extract_temporal_features(make_data(), group_minutes=0)


def test_duplicate_ids():
    data = make_data()
    data.loc[1, "detection_id"] = "a"

    with pytest.raises(ValueError, match="Duplicate"):
        extract_temporal_features(data)
