"""FRP-based preliminary alert selection."""

import numpy as np
import pandas as pd


def select_preliminary_alerts(
    detections: pd.DataFrame,
    policy: str = "all",
    global_threshold_mw: float = 3.0,
    product_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Select preliminary alerts from currently unclustered detections."""

    required = {
        "detection_id",
        "source_product",
        "frp_mw",
    }

    if not required.issubset(detections.columns):
        raise ValueError("Missing required detection columns")

    if detections["detection_id"].isna().any():
        raise ValueError("Missing detection IDs")

    if detections["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection IDs")

    result = detections.copy()

    frp = pd.to_numeric(
        result["frp_mw"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    if policy == "all":
        selected = pd.Series(True, index=result.index)

    elif policy == "global":
        if (
            not np.isfinite(global_threshold_mw)
            or global_threshold_mw < 0
        ):
            raise ValueError("Invalid global FRP threshold")

        selected = frp.ge(global_threshold_mw)

    elif policy == "product":
        if not product_thresholds:
            raise ValueError("Missing product thresholds")

        thresholds = result["source_product"].map(
            product_thresholds
        )

        if thresholds.isna().any():
            raise ValueError("Missing threshold for a product")

        thresholds = pd.to_numeric(
            thresholds, errors="coerce"
        )

        if (
            thresholds.isna().any()
            or not np.isfinite(thresholds).all()
            or thresholds.lt(0).any()
        ):
            raise ValueError("Invalid product thresholds")

        selected = frp.ge(thresholds)

    else:
        raise ValueError(f"Unknown alert policy: {policy}")

    result["preliminary_alert"] = selected.to_numpy()

    return result
