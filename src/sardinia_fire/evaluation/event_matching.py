"""Candidate associations between detected events and CFVA fires."""

import geopandas as gpd
import pandas as pd


def match_events(
    detections: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    spatial_tolerance_m: float = 1000,
    temporal_tolerance_days: int = 3,
    metric_crs: str = "EPSG:32632",
    min_matching_detections: int = 1,
) -> pd.DataFrame:
    """Find candidate cluster-fire associations using member detections."""

    if spatial_tolerance_m < 0 or temporal_tolerance_days < 0:
        raise ValueError("Tolerances must be nonnegative")

    if min_matching_detections < 1:
        raise ValueError("min_matching_detections must be >= 1")

    required_points = {
        "cluster_id", "detection_id", "acq_datetime_utc"
    }
    required_fires = {"fire_id", "fire_date"}

    if not required_points.issubset(detections.columns):
        raise ValueError("Missing detection columns")

    if not required_fires.issubset(reference.columns):
        raise ValueError("Missing reference columns")

    if detections.crs is None or reference.crs is None:
        raise ValueError("Missing CRS")

    columns = [
        "cluster_id",
        "fire_id",
        "matching_detections",
        "min_abs_delta_days",
    ]

    points = detections.loc[
        detections["cluster_id"] >= 0
    ].copy()

    if points.empty or reference.empty:
        return pd.DataFrame(columns=columns)

    points = points.to_crs(metric_crs)
    fires = reference.to_crs(metric_crs)

    buffers = fires[["fire_id", "fire_date", "geometry"]].copy()

    if spatial_tolerance_m > 0:
        buffers["geometry"] = buffers.geometry.buffer(
            spatial_tolerance_m
        )

    candidates = gpd.sjoin(
        points,
        buffers,
        how="inner",
        predicate="intersects",
    ).reset_index(drop=True)

    if candidates.empty:
        return pd.DataFrame(columns=columns)

    observation_dates = pd.to_datetime(
        candidates["acq_datetime_utc"],
        utc=True,
    ).dt.tz_localize(None).dt.normalize()

    reference_dates = pd.to_datetime(
        candidates["fire_date"]
    ).dt.normalize()

    candidates["abs_delta_days"] = (
        observation_dates - reference_dates
    ).dt.days.abs()

    candidates = candidates.loc[
        candidates["abs_delta_days"] <= temporal_tolerance_days
    ].copy()

    if candidates.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        candidates.groupby(["cluster_id", "fire_id"], as_index=False)
        .agg(
            matching_detections=("detection_id", "nunique"),
            min_abs_delta_days=("abs_delta_days", "min"),
        )
    )

    summary = summary.loc[
        summary["matching_detections"] >= min_matching_detections
    ]

    return summary.sort_values(
        ["cluster_id", "matching_detections", "fire_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
