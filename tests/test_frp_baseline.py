"""Tests for FRP-only event ranking."""

import pandas as pd
import pytest

from sardinia_fire.ranking.frp_baseline import rank_events_by_frp


def make_events():
    return pd.DataFrame({
        "cluster_id": [10, 11, 12],
        "frp_max_mw": [100.0, 300.0, 200.0],
        "frp_p90_mw": [80.0, 20.0, 50.0],
        "frp_group_peak_p90_mw": [90.0, 40.0, 60.0],
    })


def test_rank_by_maximum_frp():
    result = rank_events_by_frp(make_events(), "frp_max_mw")
    assert result["cluster_id"].tolist() == [11, 12, 10]


def test_rank_by_p90():
    result = rank_events_by_frp(make_events(), "frp_p90_mw")
    assert result["cluster_id"].tolist() == [10, 12, 11]


def test_ranks_are_consecutive():
    result = rank_events_by_frp(make_events())
    assert result["rank"].tolist() == [1, 2, 3]


def test_original_dataframe_unchanged():
    original = make_events()
    rank_events_by_frp(original)
    assert "ranking_score" not in original.columns


def test_invalid_metric():
    with pytest.raises(ValueError):
        rank_events_by_frp(make_events(), "detections")


def test_duplicate_cluster_ids():
    events = make_events()
    events.loc[1, "cluster_id"] = 10
    with pytest.raises(ValueError):
        rank_events_by_frp(events)


def test_missing_frp_sorted_last():
    events = make_events()
    events.loc[0, "frp_p90_mw"] = None

    result = rank_events_by_frp(events)
    assert result.iloc[-1]["cluster_id"] == 10
