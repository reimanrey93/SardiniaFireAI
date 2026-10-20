"""Thermal and revisit-related event features from FIRMS."""

import geopandas as gpd
import pandas as pd


def extract_thermal_features(
    detections: gpd.GeoDataFrame,
    group_minutes: int = 30,
) -> pd.DataFrame:
    """Summarize FRP using approximate observation groups."""

    required = {
        "cluster_id",
        "detection_id",
        "acq_datetime_utc",
        "source_product",
        "satellite",
        "frp_mw",
    }

    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if group_minutes <= 0:
        raise ValueError("group_minutes must be positive")

    points = detections.loc[
        detections["cluster_id"] >= 0
    ].copy()

    columns = [
        "cluster_id",
        "observation_groups",
        "active_days",
        "frp_group_peak_max_mw",
        "frp_group_peak_mean_mw",
        "frp_group_peak_p90_mw",
        "detections_per_group_mean",
    ]

    if points.empty:
        return pd.DataFrame(columns=columns)

    if points["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection IDs")

    points["time_utc"] = pd.to_datetime(
        points["acq_datetime_utc"],
        utc=True,
        errors="raise",
    )

    if points["time_utc"].isna().any():
        raise ValueError("Missing acquisition timestamps")

    points["frp_numeric"] = pd.to_numeric(
        points["frp_mw"],
        errors="coerce",
    )

    points["time_bin"] = points["time_utc"].dt.floor(
        f"{group_minutes}min"
    )

    records = []

    for cluster_id, group in points.groupby("cluster_id"):
        observations = (
            group.groupby(
                ["source_product", "satellite", "time_bin"],
                dropna=False,
            )
            .agg(
                group_detections=("detection_id", "nunique"),
                group_frp_peak_mw=("frp_numeric", "max"),
            )
            .reset_index()
        )

        peaks = observations["group_frp_peak_mw"].dropna()

        records.append({
            "cluster_id": int(cluster_id),
            "observation_groups": len(observations),
            "active_days": group["time_utc"].dt.normalize().nunique(),
            "frp_group_peak_max_mw": peaks.max(),
            "frp_group_peak_mean_mw": peaks.mean(),
            "frp_group_peak_p90_mw": peaks.quantile(0.90),
            "detections_per_group_mean": (
                observations["group_detections"].mean()
            ),
        })

    return pd.DataFrame(records, columns=columns).sort_values(
        "cluster_id"
    ).reset_index(drop=True)
