"""Regression tests for sparse ST-DBSCAN."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sardinia_fire.clustering.st_dbscan import st_dbscan
from sardinia_fire.clustering.st_dbscan_sparse import st_dbscan_sparse


def make_data(coordinates, hours):
    times = pd.Timestamp("2024-07-20T00:00:00Z") + pd.to_timedelta(
        hours, unit="h"
    )

    return gpd.GeoDataFrame(
        {
            "detection_id": [f"d{i}" for i in range(len(coordinates))],
            "acq_datetime_utc": times,
        },
        geometry=[Point(x, y) for x, y in coordinates],
        crs="EPSG:32632",
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


@pytest.mark.parametrize(
    ("coordinates", "hours", "space", "time", "minimum"),
    [
        (
            [(0, 0), (100, 0), (200, 0), (8000, 0)],
            [0, 1, 2, 1], 500, 3, 3,
        ),
        (
            [(0, 0), (3000, 0), (6000, 0)],
            [0, 0, 0], 500, 3, 2,
        ),
        (
            [(0, 0), (100, 0), (200, 0)],
            [0, 12, 24], 500, 3, 2,
        ),
        (
            [(0, 0), (400, 0), (800, 0), (1200, 0)],
            [0, 1, 2, 3], 500, 2, 2,
        ),
        (
            [
                (0, 0), (100, 0), (200, 0),
                (5000, 0), (5100, 0), (5200, 0),
                (15000, 0),
            ],
            [0, 1, 2, 0, 1, 2, 0], 500, 3, 3,
        ),
    ],
)
def test_sparse_matches_dense(
    coordinates, hours, space, time, minimum
):
    data = make_data(coordinates, hours)

    dense = st_dbscan(data, space, time, minimum)
    sparse = st_dbscan_sparse(data, space, time, minimum)

    assert partition(dense) == partition(sparse)


def test_empty_input():
    data = make_data([], [])
    result = st_dbscan_sparse(data)

    assert result.empty
    assert "cluster_id" in result.columns


def test_invalid_parameters():
    data = make_data([(0, 0)], [0])

    with pytest.raises(ValueError):
        st_dbscan_sparse(data, eps_space_m=0)

    with pytest.raises(ValueError):
        st_dbscan_sparse(data, eps_time_h=0)

    with pytest.raises(ValueError):
        st_dbscan_sparse(data, min_samples=0)
