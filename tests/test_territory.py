"""Tests for territorial filtering of FIRMS detections."""

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from sardinia_fire.preprocessing.territory import (
    classify_territorial_detections,
)


def make_data():
    detections = gpd.GeoDataFrame(
        {"detection_id": ["inside", "coast", "outside"]},
        geometry=[
            Point(500, 500),
            Point(1200, 500),
            Point(2000, 500),
        ],
        crs="EPSG:32632",
    )

    boundary = gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[box(0, 0, 1000, 1000)],
        crs="EPSG:32632",
    )

    return detections, boundary


def test_coastal_classification():
    result = classify_territorial_detections(*make_data())

    assert result["inside_territory"].tolist() == [
        True, False, False
    ]

    assert result["within_study_area"].tolist() == [
        True, True, False
    ]

    assert result["distance_to_territory_m"].tolist() == [
        0, 200, 1000
    ]


def test_strict_filter():
    result = classify_territorial_detections(
        *make_data(), coastal_buffer_m=0
    )

    assert result["within_study_area"].tolist() == [
        True, False, False
    ]


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        classify_territorial_detections(
            *make_data(), coastal_buffer_m=-1
        )


def test_original_unchanged():
    detections, boundary = make_data()
    classify_territorial_detections(detections, boundary)

    assert "within_study_area" not in detections.columns


def test_empty_detections():
    detections, boundary = make_data()
    result = classify_territorial_detections(
        detections.iloc[:0].copy(), boundary
    )

    assert result.empty
    assert "within_study_area" in result.columns


def test_missing_crs():
    detections, boundary = make_data()
    boundary = boundary.set_crs(None, allow_override=True)

    with pytest.raises(ValueError, match="Missing CRS"):
        classify_territorial_detections(detections, boundary)
