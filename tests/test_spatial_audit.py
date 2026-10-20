"""Tests for the FIRMS-CFVA spatial label audit."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sardinia_fire.evaluation.spatial_audit import audit_reference_labels


def make_data():
    dataset = pd.DataFrame({
        "cluster_id": [0, 1],
        "reference_fire_id": ["F1", None],
        "label_status": ["provisional_one_to_one", "unlabeled"],
    })

    detections = gpd.GeoDataFrame(
        {
            "cluster_id": [0, 0, 0, 1],
            "acq_datetime_utc": pd.to_datetime([
                "2024-07-20T10:00:00Z",
                "2024-07-20T11:00:00Z",
                "2024-07-20T12:00:00Z",
                "2024-07-20T10:00:00Z",
            ], utc=True),
        },
        geometry=[
            Point(500, 500),
            Point(600, 600),
            Point(1200, 500),
            Point(5000, 5000),
        ],
        crs="EPSG:32632",
    )

    reference = gpd.GeoDataFrame(
        {
            "fire_id": ["F1"],
            "fire_date": ["2024-07-20"],
        },
        geometry=[box(0, 0, 1000, 1000)],
        crs="EPSG:32632",
    )

    return dataset, detections, reference


def test_spatial_counts():
    result = audit_reference_labels(*make_data())

    assert len(result) == 1
    assert result.iloc[0]["supported"] == 3
    assert result.iloc[0]["inside"] == 2
    assert result.iloc[0]["quality_category"] == "A_high_spatial_evidence"


def test_unlabeled_events_excluded():
    result = audit_reference_labels(*make_data())
    assert result["cluster_id"].tolist() == [0]


def test_temporal_tolerance():
    dataset, detections, reference = make_data()
    detections.loc[2, "acq_datetime_utc"] = pd.Timestamp(
        "2024-07-25T12:00:00Z"
    )

    result = audit_reference_labels(
        dataset, detections, reference
    )

    assert result.iloc[0]["supported"] == 2
    assert result.iloc[0]["inside"] == 2


def test_missing_fire():
    dataset, detections, reference = make_data()
    dataset.loc[0, "reference_fire_id"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Missing CFVA fire"):
        audit_reference_labels(dataset, detections, reference)


def test_negative_tolerance():
    with pytest.raises(ValueError):
        audit_reference_labels(*make_data(), spatial_tolerance_m=-1)


def test_no_labeled_events():
    dataset, detections, reference = make_data()
    dataset["label_status"] = "unlabeled"

    result = audit_reference_labels(
        dataset, detections, reference
    )

    assert result.empty
