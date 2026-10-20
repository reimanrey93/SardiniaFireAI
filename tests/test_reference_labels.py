"""Tests for conservative CFVA reference-area labeling."""

import pandas as pd
import pytest

from sardinia_fire.evaluation.reference_labels import build_reference_labels


def reference():
    return pd.DataFrame({
        "fire_id": ["F1", "F2"],
        "benchmark_area_ha": [120.0, 35.0],
        "area_review_required": [False, False],
        "temporal_reference_valid": [True, True],
        "spatial_reference_valid": [True, True],
    })


def events():
    return pd.DataFrame({"cluster_id": [0, 1]})


def test_unique_pair_gets_area():
    pairs = pd.DataFrame({
        "cluster_id": [0],
        "fire_id": ["F1"],
        "matching_detections": [5],
    })
    result = build_reference_labels(events(), pairs, reference())

    assert result.loc[0, "reference_area_ha"] == 120
    assert result.loc[0, "label_status"] == "provisional_one_to_one"
    assert result.loc[1, "label_status"] == "unlabeled"


def test_merging_is_not_labeled():
    pairs = pd.DataFrame({
        "cluster_id": [0, 0],
        "fire_id": ["F1", "F2"],
        "matching_detections": [5, 4],
    })
    result = build_reference_labels(events(), pairs, reference())

    assert result["reference_area_ha"].isna().all()


def test_splitting_is_not_labeled():
    pairs = pd.DataFrame({
        "cluster_id": [0, 1],
        "fire_id": ["F1", "F1"],
        "matching_detections": [5, 4],
    })
    result = build_reference_labels(events(), pairs, reference())

    assert result["reference_area_ha"].isna().all()


def test_review_required_is_excluded():
    pairs = pd.DataFrame({
        "cluster_id": [0],
        "fire_id": ["F1"],
        "matching_detections": [5],
    })
    fires = reference()
    fires.loc[0, "area_review_required"] = True

    result = build_reference_labels(events(), pairs, fires)
    assert result["reference_area_ha"].isna().all()


def test_minimum_support():
    pairs = pd.DataFrame({
        "cluster_id": [0],
        "fire_id": ["F1"],
        "matching_detections": [1],
    })
    result = build_reference_labels(events(), pairs, reference())

    assert result["reference_area_ha"].isna().all()


def test_duplicate_event_ids():
    pairs = pd.DataFrame({
        "cluster_id": [0],
        "fire_id": ["F1"],
        "matching_detections": [5],
    })
    duplicated = pd.DataFrame({"cluster_id": [0, 0]})

    with pytest.raises(ValueError, match="Duplicate event"):
        build_reference_labels(duplicated, pairs, reference())
