"""Regression against the July 2024 ST-DBSCAN reference."""

from pathlib import Path

import geopandas as gpd
import pytest

from sardinia_fire.clustering.st_dbscan import st_dbscan
from sardinia_fire.clustering.st_dbscan_sparse import st_dbscan_sparse

DATA_PATH = Path(
    "data/processed/clustering/st_dbscan_2024_07.parquet"
)


def partition(frame):
    clusters = frozenset(
        frozenset(group["detection_id"])
        for _, group in frame.loc[
            frame["cluster_id"] >= 0
        ].groupby("cluster_id")
    )
    noise = frozenset(
        frame.loc[frame["cluster_id"] == -1, "detection_id"]
    )
    return clusters, noise


def test_july_2024_reference_partition():
    if not DATA_PATH.exists():
        pytest.skip("Local July 2024 data unavailable")

    original = gpd.read_parquet(DATA_PATH)
    assert len(original) == 623
    assert original["detection_id"].is_unique

    detections = original.drop(columns="cluster_id")

    sparse = st_dbscan_sparse(
        detections,
        eps_space_m=1500,
        eps_time_h=24,
        min_samples=3,
    )

    dense = st_dbscan(
        detections,
        eps_space_m=1500,
        eps_time_h=24,
        min_samples=3,
    )

    assert partition(sparse) == partition(dense)
    assert partition(sparse) == partition(original)
    assert sparse.loc[
        sparse["cluster_id"] >= 0, "cluster_id"
    ].nunique() == 35
    assert (sparse["cluster_id"] == -1).sum() == 86
