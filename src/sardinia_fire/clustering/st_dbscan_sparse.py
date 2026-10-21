"""Memory-efficient spatiotemporal DBSCAN using cKDTree."""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def st_dbscan_sparse(
    detections: gpd.GeoDataFrame,
    eps_space_m: float = 1500,
    eps_time_h: float = 24,
    min_samples: int = 3,
) -> gpd.GeoDataFrame:
    """Cluster detections using sparse joint space-time neighborhoods."""

    if eps_space_m <= 0 or eps_time_h <= 0 or min_samples < 1:
        raise ValueError("Invalid ST-DBSCAN parameters")

    required = {"acq_datetime_utc", "geometry"}
    if not required.issubset(detections.columns):
        raise ValueError("Missing required detection columns")

    if detections.crs is None:
        raise ValueError("Input CRS is missing")

    result = detections.copy().reset_index(drop=True)

    if result.empty:
        result["cluster_id"] = pd.Series(dtype="int64")
        return result

    if result.geometry.isna().any() or result.geometry.is_empty.any():
        raise ValueError("Missing or empty geometries")

    if not result.geometry.geom_type.eq("Point").all():
        raise ValueError("ST-DBSCAN requires point geometries")

    timestamps = pd.to_datetime(
        result["acq_datetime_utc"],
        utc=True,
        errors="raise",
    )

    if timestamps.isna().any():
        raise ValueError("Missing acquisition timestamps")

    metric = result.to_crs("EPSG:32632")

    xy = np.column_stack((
        metric.geometry.x.to_numpy(),
        metric.geometry.y.to_numpy(),
    ))

    times = timestamps.astype("int64").to_numpy()
    tree = cKDTree(xy)

    n = len(result)
    indptr = [0]
    indices = []

    for i in range(n):
        candidates = np.asarray(
            tree.query_ball_point(xy[i], r=eps_space_m),
            dtype=np.int64,
        )

        candidates.sort()

        time_hours = (
            np.abs(times[candidates] - times[i]) / 3.6e12
        )

        valid = candidates[time_hours <= eps_time_h]

        indices.extend(valid.tolist())
        indptr.append(len(indices))

    graph = csr_matrix(
        (
            np.ones(len(indices), dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(n, n),
    )

    labels = DBSCAN(
        eps=1.0,
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(graph)

    result["cluster_id"] = labels

    return result
