"""Spatial audit of provisional FIRMS-CFVA event labels."""

import geopandas as gpd
import pandas as pd

from sardinia_fire.evaluation.label_quality import classify_label_quality


def audit_reference_labels(
    dataset: pd.DataFrame,
    detections: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    spatial_tolerance_m: float = 1000,
    temporal_tolerance_days: int = 3,
    metric_crs: str = "EPSG:32632",
) -> pd.DataFrame:
    """Audit labeled events against their CFVA perimeters."""

    if spatial_tolerance_m < 0 or temporal_tolerance_days < 0:
        raise ValueError("Tolerances cannot be negative")

    required_events = {
        "cluster_id", "reference_fire_id", "label_status"
    }
    required_points = {"cluster_id", "acq_datetime_utc"}
    required_fires = {"fire_id", "fire_date"}

    for name, frame, required in (
        ("dataset", dataset, required_events),
        ("detections", detections, required_points),
        ("reference", reference, required_fires),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing {sorted(missing)}")

    if dataset["cluster_id"].duplicated().any():
        raise ValueError("Duplicate cluster IDs in dataset")

    if reference["fire_id"].duplicated().any():
        raise ValueError("Duplicate CFVA fire IDs")

    if detections.crs is None or reference.crs is None:
        raise ValueError("Missing spatial reference system")

    labeled = dataset.loc[
        dataset["label_status"] == "provisional_one_to_one"
    ].copy()

    columns = [
        "cluster_id", "supported", "inside",
        "median_distance_m", "min_delta_days",
        "inside_fraction", "quality_category",
    ]

    if labeled.empty:
        return pd.DataFrame(columns=columns)

    points = detections.to_crs(metric_crs)
    fires = reference.to_crs(metric_crs).set_index("fire_id")

    records = []

    for event in labeled.itertuples(index=False):
        if event.reference_fire_id not in fires.index:
            raise ValueError(
                f"Missing CFVA fire: {event.reference_fire_id}"
            )

        group = points.loc[
            points["cluster_id"] == event.cluster_id
        ]

        if group.empty:
            raise ValueError(
                f"No FIRMS detections for cluster {event.cluster_id}"
            )

        fire = fires.loc[event.reference_fire_id]

        timestamps = pd.to_datetime(
            group["acq_datetime_utc"],
            utc=True,
            errors="raise",
        )

        if timestamps.isna().any():
            raise ValueError("Missing acquisition timestamps")

        fire_date = pd.to_datetime(
            fire["fire_date"], errors="raise"
        )

        if pd.isna(fire_date):
            raise ValueError("Missing CFVA fire date")

        dates = timestamps.dt.tz_localize(None).dt.normalize()
        fire_date = pd.Timestamp(fire_date).normalize()

        delta = (dates - fire_date).dt.days.abs()

        distances = group.geometry.distance(fire.geometry)

        supported = (
            distances.le(spatial_tolerance_m)
            & delta.le(temporal_tolerance_days)
        )

        inside = supported & distances.le(1e-6)

        if not supported.any():
            raise ValueError(
                f"No compatible detections for cluster {event.cluster_id}"
            )

        records.append({
            "cluster_id": event.cluster_id,
            "supported": int(supported.sum()),
            "inside": int(inside.sum()),
            "median_distance_m": float(
                distances.loc[supported].median()
            ),
            "min_delta_days": int(delta.loc[supported].min()),
        })

    audit = classify_label_quality(pd.DataFrame(records))

    return audit[columns].sort_values(
        "cluster_id"
    ).reset_index(drop=True)
