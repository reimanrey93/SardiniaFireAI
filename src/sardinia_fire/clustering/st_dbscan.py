"""Spatiotemporal DBSCAN for satellite active-fire detections."""

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def st_dbscan(
    detections: gpd.GeoDataFrame,
    eps_space_m: float = 1500,
    eps_time_h: float = 24,
    min_samples: int = 3,
) -> gpd.GeoDataFrame:
    """Cluster detections using joint spatial and temporal neighborhoods."""

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

    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]

    spatial_neighbors = (
        dx**2 + dy**2 <= eps_space_m**2
    )

    time_hours = (
        np.abs(times[:, None] - times[None, :]) / 3.6e12
    )

    temporal_neighbors = time_hours <= eps_time_h

    neighbors = spatial_neighbors & temporal_neighbors

    distances = np.where(neighbors, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)

    labels = DBSCAN(
        eps=1.0,
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(distances)

    result["cluster_id"] = labels

    return result
