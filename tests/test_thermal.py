"""Tests for temporal grouping of FIRMS thermal observations."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sardinia_fire.features.thermal import extract_thermal_features


def make_data():
    return gpd.GeoDataFrame(
        {
            "cluster_id": [0, 0, 0, 0, 0, -1],
            "detection_id": ["a", "b", "c", "d", "e", "noise"],
            "acq_datetime_utc": pd.to_datetime(
                [
                    "2024-07-20T10:01:00Z",
                    "2024-07-20T10:05:00Z",
                    "2024-07-20T10:40:00Z",
                    "2024-07-20T10:10:00Z",
                    "2024-07-21T10:10:00Z",
                    "2024-07-20T10:00:00Z",
                ],
                utc=True,
            ),
            "source_product": [
                "VIIRS", "VIIRS", "VIIRS", "MODIS", "VIIRS", "VIIRS"
            ],
            "satellite": ["N20", "N20", "N20", "Terra", "N20", "N20"],
            "frp_mw": [10, 30, 20, 50, 40, 999],
        },
        geometry=[Point(500000 + i, 4400000) for i in range(6)],
        crs="EPSG:32632",
    )


def test_group_counts():
    event = extract_thermal_features(make_data()).iloc[0]

    assert event["observation_groups"] == 4
    assert event["active_days"] == 2
    assert event["detections_per_group_mean"] == pytest.approx(1.25)


def test_group_frp_statistics():
    event = extract_thermal_features(make_data()).iloc[0]

    assert event["frp_group_peak_max_mw"] == pytest.approx(50)
    assert event["frp_group_peak_mean_mw"] == pytest.approx(35)
    assert event["frp_group_peak_p90_mw"] == pytest.approx(47)


def test_noise_excluded():
    event = extract_thermal_features(make_data()).iloc[0]

    assert event["frp_group_peak_max_mw"] != 999


def test_invalid_group_interval():
    with pytest.raises(ValueError):
        extract_thermal_features(make_data(), group_minutes=0)


def test_duplicate_ids():
    points = make_data()
    points.loc[1, "detection_id"] = "a"

    with pytest.raises(ValueError, match="Duplicate"):
        extract_thermal_features(points)


def test_noise_only():
    points = make_data()
    points["cluster_id"] = -1

    assert extract_thermal_features(points).empty
