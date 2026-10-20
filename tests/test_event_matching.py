"""Unit tests for event-to-reference candidate matching."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sardinia_fire.evaluation.event_matching import match_events

CRS = "EPSG:32632"


def make_points(records):
    """Records: detection_id, cluster_id, x, y, hour."""
    return gpd.GeoDataFrame(
        {
            "detection_id": [r[0] for r in records],
            "cluster_id": [r[1] for r in records],
            "acq_datetime_utc": [
                pd.Timestamp("2024-07-20T00:00:00Z")
                + pd.Timedelta(hours=r[4])
                for r in records
            ],
        },
        geometry=[Point(r[2], r[3]) for r in records],
        crs=CRS,
    )


def make_fires(records):
    """Records: fire_id, xmin, ymin, xmax, ymax, day."""
    return gpd.GeoDataFrame(
        {
            "fire_id": [r[0] for r in records],
            "fire_date": [
                pd.Timestamp(f"2024-07-{r[5]:02d}")
                for r in records
            ],
        },
        geometry=[
            box(r[1], r[2], r[3], r[4])
            for r in records
        ],
        crs=CRS,
    )


def test_correct_event_association():
    points = make_points([
        ("d1", 0, 100, 100, 0),
        ("d2", 0, 150, 150, 1),
    ])
    fires = make_fires([
        ("F1", 0, 0, 500, 500, 20),
    ])

    result = match_events(points, fires)

    assert len(result) == 1
    assert result.iloc[0]["cluster_id"] == 0
    assert result.iloc[0]["fire_id"] == "F1"
    assert result.iloc[0]["matching_detections"] == 2


def test_spatial_separation():
    points = make_points([
        ("d1", 0, 100, 100, 0),
    ])
    fires = make_fires([
        ("F1", 5000, 5000, 5500, 5500, 20),
    ])

    result = match_events(
        points, fires, spatial_tolerance_m=500
    )

    assert result.empty


def test_temporal_separation():
    points = make_points([
        ("d1", 0, 100, 100, 0),
    ])
    fires = make_fires([
        ("F1", 0, 0, 500, 500, 10),
    ])

    result = match_events(
        points, fires, temporal_tolerance_days=3
    )

    assert result.empty


def test_merging_candidates():
    points = make_points([
        ("d1", 0, 100, 100, 0),
        ("d2", 0, 5100, 100, 1),
    ])
    fires = make_fires([
        ("F1", 0, 0, 500, 500, 20),
        ("F2", 5000, 0, 5500, 500, 20),
    ])

    result = match_events(
        points, fires, spatial_tolerance_m=100
    )

    assert len(result) == 2
    assert set(result["fire_id"]) == {"F1", "F2"}
    assert result["cluster_id"].nunique() == 1


def test_splitting_candidates():
    points = make_points([
        ("d1", 0, 100, 100, 0),
        ("d2", 1, 200, 200, 1),
    ])
    fires = make_fires([
        ("F1", 0, 0, 500, 500, 20),
    ])

    result = match_events(points, fires)

    assert len(result) == 2
    assert result["cluster_id"].nunique() == 2
    assert result["fire_id"].nunique() == 1


def test_noise_is_excluded():
    points = make_points([
        ("d1", -1, 100, 100, 0),
        ("d2", 0, 150, 150, 1),
    ])
    fires = make_fires([
        ("F1", 0, 0, 500, 500, 20),
    ])

    result = match_events(points, fires)

    assert len(result) == 1
    assert result.iloc[0]["matching_detections"] == 1


def test_minimum_matching_detections():
    points = make_points([
        ("d1", 0, 100, 100, 0),
        ("d2", 0, 150, 150, 1),
    ])
    fires = make_fires([
        ("F1", 0, 0, 500, 500, 20),
    ])

    result = match_events(
        points, fires, min_matching_detections=3
    )

    assert result.empty


def test_invalid_tolerances():
    points = make_points([
        ("d1", 0, 100, 100, 0),
    ])
    fires = make_fires([
        ("F1", 0, 0, 500, 500, 20),
    ])

    with pytest.raises(ValueError):
        match_events(points, fires, spatial_tolerance_m=-1)

    with pytest.raises(ValueError):
        match_events(points, fires, temporal_tolerance_days=-1)

    with pytest.raises(ValueError):
        match_events(points, fires, min_matching_detections=0)
