"""Tests for chronological ST-DBSCAN event tracking."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sardinia_fire.clustering.incremental import IncrementalSTDBSCAN
from sardinia_fire.clustering.st_dbscan_sparse import st_dbscan_sparse


def make_batch(xs, ids, hour=0):
    return gpd.GeoDataFrame(
        {
            "detection_id": ids,
            "acq_datetime_utc": pd.to_datetime(
                ["2024-07-01T00:00:00Z"] * len(ids),
                utc=True,
            ) + pd.Timedelta(hours=hour),
        },
        geometry=[Point(x, 0) for x in xs],
        crs="EPSG:32632",
    )


def partition(df, column):
    groups = frozenset(
        frozenset(group["detection_id"])
        for _, group in df.loc[df[column].notna()]
        .groupby(column)
    )
    noise = frozenset(
        df.loc[df[column].isna(), "detection_id"]
    )
    return groups, noise


def test_noise_becomes_event():
    tracker = IncrementalSTDBSCAN()

    first = tracker.update(make_batch([0, 100], ["a", "b"]))

    assert first["operational_event_id"].isna().all()

    second = tracker.update(make_batch([200], ["c"], hour=1))

    assert second["operational_event_id"].nunique() == 1
    assert second["operational_event_id"].notna().all()


def test_event_id_persists_when_growing():
    tracker = IncrementalSTDBSCAN()

    first = tracker.update(
        make_batch([0, 100, 200], ["a", "b", "c"])
    )

    event_id = int(first["operational_event_id"].iloc[0])

    second = tracker.update(
        make_batch([300], ["d"], hour=1)
    )

    assert second["operational_event_id"].nunique() == 1
    assert int(second["operational_event_id"].iloc[0]) == event_id


def test_separate_events_can_merge():
    tracker = IncrementalSTDBSCAN()

    initial = tracker.update(
        make_batch(
            [0, 100, 200, 2800, 2900, 3000],
            ["a", "b", "c", "d", "e", "f"],
        )
    )

    assert initial["operational_event_id"].nunique() == 2

    updated = tracker.update(
        make_batch([1600], ["bridge"], hour=1)
    )

    assert updated["operational_event_id"].nunique() == 1

    history = tracker.get_history()

    assert (history["event_type"] == "merged").any()


def test_out_of_order_update_rejected():
    tracker = IncrementalSTDBSCAN()

    tracker.update(
        make_batch([0, 100, 200], ["a", "b", "c"], hour=2)
    )

    with pytest.raises(ValueError, match="Out-of-order"):
        tracker.update(
            make_batch([300], ["d"], hour=1)
        )


def test_duplicate_detection_rejected():
    tracker = IncrementalSTDBSCAN()

    tracker.update(
        make_batch([0, 100, 200], ["a", "b", "c"])
    )

    with pytest.raises(ValueError, match="Previously observed"):
        tracker.update(
            make_batch([300], ["a"], hour=1)
        )


def test_final_partition_matches_batch_clustering():
    tracker = IncrementalSTDBSCAN()

    tracker.update(make_batch([0, 100], ["a", "b"]))
    final = tracker.update(
        make_batch([200, 5000, 5100, 5200],
                   ["c", "d", "e", "f"], hour=1)
    )

    reference = st_dbscan_sparse(tracker.observed)

    actual = partition(final, "operational_event_id")

    ref = reference.copy()
    ref["reference_event_id"] = (
        ref["cluster_id"].where(ref["cluster_id"] >= 0)
        .astype("Int64")
    )

    expected = partition(ref, "reference_event_id")

    assert actual == expected


def test_no_future_observation_used():
    tracker = IncrementalSTDBSCAN()

    first = tracker.update(make_batch([0, 100], ["a", "b"]))

    assert len(first) == 2
    assert first["operational_event_id"].isna().all()
    assert "future" not in set(first["detection_id"])

    second = tracker.update(
        make_batch([200], ["future"], hour=6)
    )

    assert len(second) == 3
    assert second["operational_event_id"].notna().all()
