"""Event-level features from multisensor FIRMS detections."""

import geopandas as gpd
import pandas as pd


def extract_event_features(
    detections: gpd.GeoDataFrame,
    metric_crs: str = "EPSG:32632",
) -> pd.DataFrame:
    """Extract thermal, temporal, and spatial features for each cluster."""

    required = {
        "cluster_id",
        "detection_id",
        "acq_datetime_utc",
        "frp_mw",
        "source_product",
        "satellite",
    }

    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if detections.crs is None:
        raise ValueError("Missing coordinate reference system")

    points = detections.loc[detections["cluster_id"] >= 0].copy()

    if points.empty:
        return pd.DataFrame(columns=[
            "cluster_id", "detections", "duration_h", "frp_max_mw"
        ])

    points = points.to_crs(metric_crs)

    points["acq_datetime_utc"] = pd.to_datetime(
        points["acq_datetime_utc"], utc=True
    )

    points["frp_mw"] = pd.to_numeric(
        points["frp_mw"], errors="coerce"
    )

    records = []

    for cluster_id, group in points.groupby("cluster_id"):
        times = group["acq_datetime_utc"]
        frp = group["frp_mw"].dropna()

        xmin, ymin, xmax, ymax = group.total_bounds

        records.append({
            "cluster_id": int(cluster_id),
            "detections": len(group),
            "satellites": group["satellite"].nunique(),
            "products": group["source_product"].nunique(),
            "first_utc": times.min(),
            "last_utc": times.max(),
            "duration_h": (
                times.max() - times.min()
            ).total_seconds() / 3600,
            "frp_max_mw": frp.max() if not frp.empty else float("nan"),
            "frp_mean_mw": frp.mean() if not frp.empty else float("nan"),
            "frp_median_mw": frp.median() if not frp.empty else float("nan"),
            "frp_p90_mw": (
                frp.quantile(0.90) if not frp.empty else float("nan")
            ),
            "bbox_diagonal_km": (
                ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
            ) / 1000,
        })

    return (
        pd.DataFrame(records)
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
