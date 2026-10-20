"""Tests for event-level FIRMS feature extraction."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sardinia_fire.features.event_features import extract_event_features


def make_detections(records):
    """Records: ID, cluster, x, y, hour, FRP, product, satellite."""
    return gpd.GeoDataFrame(
        {
            "detection_id": [r[0] for r in records],
            "cluster_id": [r[1] for r in records],
            "acq_datetime_utc": [
                pd.Timestamp("2024-07-20T00:00:00Z")
                + pd.Timedelta(hours=r[4])
                for r in records
            ],
            "frp_mw": [r[5] for r in records],
            "source_product": [r[6] for r in records],
            "satellite": [r[7] for r in records],
        },
        geometry=[
            Point(r[2], r[3]) for r in records
        ],
        crs="EPSG:32632",
    )


def test_thermal_features():
    points = make_detections([
        ("a", 0, 500000, 4400000, 0, 10, "VIIRS", "N20"),
        ("b", 0, 500100, 4400000, 1, 20, "VIIRS", "N20"),
        ("c", 0, 500200, 4400000, 2, 30, "MODIS", "Terra"),
    ])

    result = extract_event_features(points)
    event = result.iloc[0]

    assert event["detections"] == 3
    assert event["frp_max_mw"] == pytest.approx(30)
    assert event["frp_mean_mw"] == pytest.approx(20)
    assert event["frp_median_mw"] == pytest.approx(20)
    assert event["frp_p90_mw"] == pytest.approx(28)


def test_temporal_and_spatial_features():
    points = make_detections([
        ("a", 4, 500000, 4400000, 0, 10, "VIIRS", "N20"),
        ("b", 4, 503000, 4404000, 5, 15, "VIIRS", "N20"),
    ])

    event = extract_event_features(points).iloc[0]

    assert event["cluster_id"] == 4
    assert event["duration_h"] == pytest.approx(5)
    assert event["bbox_diagonal_km"] == pytest.approx(5)


def test_multisensor_counts():
    points = make_detections([
        ("a", 0, 500000, 4400000, 0, 10, "NOAA20", "N20"),
        ("b", 0, 500100, 4400000, 1, 20, "NOAA21", "N21"),
        ("c", 0, 500200, 4400000, 2, 30, "MODIS", "Terra"),
    ])

    event = extract_event_features(points).iloc[0]

    assert event["products"] == 3
    assert event["satellites"] == 3


def test_noise_is_excluded():
    points = make_detections([
        ("a", 0, 500000, 4400000, 0, 10, "VIIRS", "N20"),
        ("b", -1, 500100, 4400000, 1, 100, "VIIRS", "N20"),
    ])

    result = extract_event_features(points)

    assert len(result) == 1
    assert result.iloc[0]["detections"] == 1
    assert result.iloc[0]["frp_max_mw"] == pytest.approx(10)


def test_missing_frp():
    points = make_detections([
        ("a", 0, 500000, 4400000, 0, None, "VIIRS", "N20"),
        ("b", 0, 500100, 4400000, 1, 20, "VIIRS", "N20"),
    ])

    event = extract_event_features(points).iloc[0]

    assert event["detections"] == 2
    assert event["frp_mean_mw"] == pytest.approx(20)
    assert event["frp_p90_mw"] == pytest.approx(20)


def test_empty_after_noise_filter():
    points = make_detections([
        ("a", -1, 500000, 4400000, 0, 10, "VIIRS", "N20"),
    ])

    result = extract_event_features(points)

    assert result.empty
    assert "cluster_id" in result.columns


def test_missing_required_column():
    points = make_detections([
        ("a", 0, 500000, 4400000, 0, 10, "VIIRS", "N20"),
    ]).drop(columns="frp_mw")

    with pytest.raises(ValueError, match="Missing columns"):
        extract_event_features(points)
