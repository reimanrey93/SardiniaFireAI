"""Tests for spatial evidence categories of CFVA labels."""

import pandas as pd
import pytest

from sardinia_fire.evaluation.label_quality import classify_label_quality


def test_high_spatial_evidence():
    data = pd.DataFrame({
        "cluster_id": [13],
        "supported": [183],
        "inside": [118],
    })

    result = classify_label_quality(data)

    assert result.iloc[0]["quality_category"] == "A_high_spatial_evidence"
    assert result.iloc[0]["inside_fraction"] == pytest.approx(118 / 183)


def test_partial_spatial_evidence():
    data = pd.DataFrame({
        "cluster_id": [4],
        "supported": [16],
        "inside": [6],
    })

    result = classify_label_quality(data)

    assert result.iloc[0]["quality_category"] == "B_partial"


def test_buffer_only():
    data = pd.DataFrame({
        "cluster_id": [32],
        "supported": [5],
        "inside": [0],
    })

    result = classify_label_quality(data)

    assert result.iloc[0]["quality_category"] == "C_buffer_only"


def test_exact_boundary():
    data = pd.DataFrame({
        "cluster_id": [1],
        "supported": [4],
        "inside": [2],
    })

    result = classify_label_quality(data)

    assert result.iloc[0]["quality_category"] == "A_high_spatial_evidence"


def test_invalid_counts():
    data = pd.DataFrame({
        "cluster_id": [1],
        "supported": [3],
        "inside": [4],
    })

    with pytest.raises(ValueError):
        classify_label_quality(data)


def test_zero_support():
    data = pd.DataFrame({
        "cluster_id": [1],
        "supported": [0],
        "inside": [0],
    })

    with pytest.raises(ValueError):
        classify_label_quality(data)


def test_duplicate_clusters():
    data = pd.DataFrame({
        "cluster_id": [1, 1],
        "supported": [3, 4],
        "inside": [1, 2],
    })

    with pytest.raises(ValueError, match="Duplicate"):
        classify_label_quality(data)


def test_original_data_unchanged():
    data = pd.DataFrame({
        "cluster_id": [1],
        "supported": [5],
        "inside": [3],
    })

    classify_label_quality(data)

    assert "quality_category" not in data.columns
