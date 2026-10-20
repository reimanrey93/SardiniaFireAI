"""Temporal evolution features for VIIRS and MODIS fire observations."""

import numpy as np
import pandas as pd


def extract_temporal_features(detections, group_minutes=30):
    """Summarize observed thermal evolution by sensor family."""

    required = {
        "cluster_id", "detection_id", "acq_datetime_utc",
        "source_product", "satellite", "frp_mw",
    }
    missing = required - set(detections.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if group_minutes <= 0:
        raise ValueError("group_minutes must be positive")

    points = detections.loc[detections["cluster_id"] >= 0].copy()

    columns = ["cluster_id"]
    for family in ("viirs", "modis"):
        columns.extend([
            f"{family}_frp_first_last_change_mw",
            f"{family}_frp_group_range_mw",
            f"{family}_median_revisit_h",
        ])

    if points.empty:
        return pd.DataFrame(columns=columns)

    if points["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection IDs")

    sources = points["source_product"].astype("string")
    is_viirs = sources.str.startswith("VIIRS_").fillna(False)
    is_modis = sources.eq("MODIS_SP").fillna(False)

    if not (is_viirs | is_modis).all():
        raise ValueError("Unknown FIRMS source product")

    points["family"] = np.where(is_viirs, "viirs", "modis")
    points["time_utc"] = pd.to_datetime(
        points["acq_datetime_utc"], utc=True
    )

    if points["time_utc"].isna().any():
        raise ValueError("Missing acquisition timestamps")

    points["frp_numeric"] = pd.to_numeric(
        points["frp_mw"], errors="coerce"
    )
    points["time_bin"] = points["time_utc"].dt.floor(
        f"{group_minutes}min"
    )

    records = []

    for cluster_id, event in points.groupby("cluster_id"):
        record = {"cluster_id": int(cluster_id)}

        for family in ("viirs", "modis"):
            group = event.loc[event["family"] == family]

            observations = (
                group.groupby(
                    ["source_product", "satellite", "time_bin"],
                    dropna=False,
                )
                .agg(
                    peak_frp=("frp_numeric", "max"),
                    first_time=("time_utc", "min"),
                )
                .reset_index()
                .sort_values("first_time")
            )

            valid = observations.dropna(subset=["peak_frp"])
            peaks = valid["peak_frp"]

            change = float("nan")
            amplitude = float("nan")
            revisit = float("nan")

            if len(valid) >= 2:
                change = float(peaks.iloc[-1] - peaks.iloc[0])
                amplitude = float(peaks.max() - peaks.min())

            unique_times = (
                observations["first_time"]
                .dropna()
                .drop_duplicates()
                .sort_values()
            )

            if len(unique_times) >= 2:
                intervals = (
                    unique_times.diff()
                    .dropna()
                    .dt.total_seconds() / 3600
                )
                revisit = float(intervals.median())

            record[f"{family}_frp_first_last_change_mw"] = change
            record[f"{family}_frp_group_range_mw"] = amplitude
            record[f"{family}_median_revisit_h"] = revisit

        records.append(record)

    return (
        pd.DataFrame(records, columns=columns)
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
