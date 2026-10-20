"""Unit tests for the spatiotemporal DBSCAN baseline."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sardinia_fire.clustering.st_dbscan import st_dbscan


def make_detections(coordinates, hours):
    """Create synthetic detections in a metric CRS."""
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


def test_nearby_observations_form_cluster():
    points = make_detections(
        [(0, 0), (100, 100), (200, 100)],
        [0, 1, 2],
    )

    result = st_dbscan(
        points, eps_space_m=500, eps_time_h=3, min_samples=3
    )

    assert result["cluster_id"].nunique() == 1
    assert (result["cluster_id"] >= 0).all()


def test_spatially_distant_observations_remain_noise():
    points = make_detections(
        [(0, 0), (3000, 0), (6000, 0)],
        [0, 0, 0],
    )

    result = st_dbscan(
        points, eps_space_m=500, eps_time_h=3, min_samples=2
    )

    assert (result["cluster_id"] == -1).all()


def test_temporally_distant_observations_remain_noise():
    points = make_detections(
        [(0, 0), (100, 0), (200, 0)],
        [0, 12, 24],
    )

    result = st_dbscan(
        points, eps_space_m=500, eps_time_h=3, min_samples=2
    )

    assert (result["cluster_id"] == -1).all()


def test_noise_and_cluster_coexist():
    points = make_detections(
        [(0, 0), (100, 0), (200, 0), (8000, 8000)],
        [0, 1, 2, 1],
    )

    result = st_dbscan(
        points, eps_space_m=500, eps_time_h=3, min_samples=3
    )

    assert (result["cluster_id"] >= 0).sum() == 3
    assert (result["cluster_id"] == -1).sum() == 1


def test_input_order_preserves_cluster_partition():
    points = make_detections(
        [
            (0, 0), (100, 0), (200, 0),
            (5000, 0), (5100, 0), (5200, 0),
            (15000, 0),
        ],
        [0, 1, 2, 0, 1, 2, 0],
    )

    first = st_dbscan(
        points, eps_space_m=500, eps_time_h=3, min_samples=3
    )

    shuffled = points.sample(frac=1, random_state=42)
    second = st_dbscan(
        shuffled, eps_space_m=500, eps_time_h=3, min_samples=3
    )

    def partition(result):
        clusters = []
        for _, group in result[result["cluster_id"] >= 0].groupby(
            "cluster_id"
        ):
            clusters.append(frozenset(group["detection_id"]))

        noise = frozenset(
            result.loc[result["cluster_id"] == -1, "detection_id"]
        )

        return set(clusters), noise

    assert partition(first) == partition(second)


def test_invalid_parameters():
    points = make_detections([(0, 0)], [0])

    with pytest.raises(ValueError):
        st_dbscan(points, eps_space_m=0)

    with pytest.raises(ValueError):
        st_dbscan(points, eps_time_h=-1)

    with pytest.raises(ValueError):
        st_dbscan(points, min_samples=0)


def test_empty_input():
    points = make_detections([], [])

    result = st_dbscan(points)

    assert result.empty
    assert "cluster_id" in result.columns
