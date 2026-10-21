"""Early-observation features for retrospectively identified fire events."""

import geopandas as gpd
import numpy as np
import pandas as pd


def extract_early_features(
    detections: gpd.GeoDataFrame,
    horizons: tuple[int, ...] = (0, 6, 12, 24),
) -> pd.DataFrame:
    """Extract features available by each horizon from first detection."""

    required = {
        "cluster_id",
        "detection_id",
        "acq_datetime_utc",
        "source_product",
        "frp_mw",
    }

    if not required.issubset(detections.columns):
        raise ValueError("Missing required detection columns")

    if not horizons or any(h < 0 for h in horizons):
        raise ValueError("Invalid horizons")

    points = detections.loc[
        detections["cluster_id"] >= 0
    ].copy()

    if points.empty:
        raise ValueError("No clustered detections")

    if points["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection IDs")

    points["acq_datetime_utc"] = pd.to_datetime(
        points["acq_datetime_utc"], utc=True
    )

    if points["acq_datetime_utc"].isna().any():
        raise ValueError("Invalid observation timestamps")

    points["frp_mw"] = pd.to_numeric(
        points["frp_mw"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    records = []

    for cluster_id, event in points.groupby("cluster_id"):
        event = event.sort_values(
            ["acq_datetime_utc", "detection_id"]
        )

        first = event["acq_datetime_utc"].min()

        elapsed = (
            event["acq_datetime_utc"] - first
        ).dt.total_seconds() / 3600

        for horizon in horizons:
            early = event.loc[elapsed <= horizon]
            frp = early["frp_mw"].dropna()

            temporal = (
                early.groupby("acq_datetime_utc")["frp_mw"]
                .max()
                .dropna()
                .sort_index()
            )

            change = np.nan

            if len(temporal) >= 2:
                change = float(
                    temporal.iloc[-1] - temporal.iloc[0]
                )

            records.append({
                "cluster_id": cluster_id,
                "horizon_h": horizon,
                "first_detection_utc": first,
                "early_detections": len(early),
                "early_timestamps": (
                    early["acq_datetime_utc"].nunique()
                ),
                "early_products": (
                    early["source_product"].nunique()
                ),
                "early_duration_h": (
                    early["acq_datetime_utc"].max() - first
                ).total_seconds() / 3600,
                "early_frp_max_mw": (
                    frp.max() if len(frp) else np.nan
                ),
                "early_frp_mean_mw": (
                    frp.mean() if len(frp) else np.nan
                ),
                "early_frp_p90_mw": (
                    frp.quantile(0.9) if len(frp) else np.nan
                ),
                "early_frp_first_last_change_mw": change,
            })

    return pd.DataFrame(records).sort_values(
        ["cluster_id", "horizon_h"]
    ).reset_index(drop=True)
