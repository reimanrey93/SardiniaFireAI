"""Territorial filtering for active-fire detections."""

import geopandas as gpd
import numpy as np


def classify_territorial_detections(
    detections: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    coastal_buffer_m: float = 500,
    metric_crs: str = "EPSG:32632",
) -> gpd.GeoDataFrame:
    """Classify FIRMS points relative to the Sardinia boundary."""

    if not np.isfinite(coastal_buffer_m) or coastal_buffer_m < 0:
        raise ValueError("Invalid coastal tolerance")

    if detections.crs is None or boundary.crs is None:
        raise ValueError("Missing CRS")

    if boundary.empty or boundary.geometry.isna().any():
        raise ValueError("Invalid territory boundary")

    if not boundary.geometry.is_valid.all():
        raise ValueError("Invalid territory geometry")

    result = detections.copy()

    if result.geometry.isna().any() or result.geometry.is_empty.any():
        raise ValueError("Invalid detection geometry")

    if not result.geometry.geom_type.eq("Point").all():
        raise ValueError("Expected point detections")

    territory = boundary.to_crs(metric_crs).geometry.union_all()
    points = result.to_crs(metric_crs)

    distance = points.geometry.distance(territory)
    strictly_inside = points.geometry.intersects(territory)

    result["inside_territory"] = strictly_inside.to_numpy()
    result["distance_to_territory_m"] = distance.to_numpy()
    result["within_study_area"] = (
        distance <= coastal_buffer_m
    ).to_numpy()

    return result
