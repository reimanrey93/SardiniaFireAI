"""Sensor-family thermal features for FIRMS events."""

import pandas as pd


def extract_sensor_thermal_features(detections, group_minutes=30):
    """Calculate VIIRS and MODIS features separately."""

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

    if points.empty:
        return pd.DataFrame(columns=["cluster_id"])

    if points["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection IDs")

    sources = points["source_product"].astype("string")

    is_viirs = sources.str.startswith("VIIRS_").fillna(False)
    is_modis = sources.eq("MODIS_SP").fillna(False)

    if not (is_viirs | is_modis).all():
        raise ValueError("Unknown FIRMS source product")

    points["family"] = "VIIRS"
    points.loc[is_modis, "family"] = "MODIS"

    points["time_utc"] = pd.to_datetime(
        points["acq_datetime_utc"], utc=True, errors="raise"
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

        for family in ("VIIRS", "MODIS"):
            group = event.loc[event["family"] == family]
            prefix = family.lower()

            record[f"{prefix}_detections"] = len(group)

            if group.empty:
                record[f"{prefix}_groups"] = 0
                record[f"{prefix}_frp_median_mw"] = float("nan")
                record[f"{prefix}_frp_p90_mw"] = float("nan")
                record[f"{prefix}_group_peak_mean_mw"] = float("nan")
                continue

            grouped = group.groupby(
                ["source_product", "satellite", "time_bin"],
                dropna=False,
            )["frp_numeric"].max()

            record[f"{prefix}_groups"] = len(grouped)
            record[f"{prefix}_frp_median_mw"] = (
                group["frp_numeric"].median()
            )
            record[f"{prefix}_frp_p90_mw"] = (
                group["frp_numeric"].quantile(0.90)
            )
            record[f"{prefix}_group_peak_mean_mw"] = grouped.mean()

        records.append(record)

    return pd.DataFrame(records).sort_values(
        "cluster_id"
    ).reset_index(drop=True)
