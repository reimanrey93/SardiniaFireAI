"""FRP-only ranking baselines for detected fire events."""

import numpy as np
import pandas as pd

FRP_METRICS = (
    "frp_max_mw",
    "frp_p90_mw",
    "frp_group_peak_p90_mw",
)


def rank_events_by_frp(
    events: pd.DataFrame,
    metric: str = "frp_p90_mw",
) -> pd.DataFrame:
    """Rank events by observed FRP, highest first."""

    if metric not in FRP_METRICS:
        raise ValueError(f"Unsupported FRP metric: {metric}")

    if "cluster_id" not in events.columns or metric not in events.columns:
        raise ValueError("Missing required event columns")

    if events["cluster_id"].duplicated().any():
        raise ValueError("Duplicate cluster IDs")

    result = events.copy()
    result["ranking_score"] = pd.to_numeric(
        result[metric], errors="coerce"
    )

    valid = result["ranking_score"].dropna()

    if not np.isfinite(valid.to_numpy()).all():
        raise ValueError("Non-finite FRP scores")

    result = result.sort_values(
        ["ranking_score", "cluster_id"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    result["rank"] = range(1, len(result) + 1)
    result["ranking_metric"] = metric

    return result
