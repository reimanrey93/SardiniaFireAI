"""Quality categories for provisional FIRMS-CFVA area labels."""

import pandas as pd


def classify_label_quality(audit: pd.DataFrame) -> pd.DataFrame:
    """Classify spatial evidence without changing reference labels."""

    required = {"cluster_id", "supported", "inside"}
    missing = required - set(audit.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = audit.copy()

    if result["cluster_id"].duplicated().any():
        raise ValueError("Duplicate cluster IDs")

    supported = pd.to_numeric(result["supported"], errors="raise")
    inside = pd.to_numeric(result["inside"], errors="raise")

    if (
        supported.isna().any()
        or inside.isna().any()
        or (supported <= 0).any()
        or (inside < 0).any()
        or (inside > supported).any()
    ):
        raise ValueError("Invalid spatial support counts")

    if (
        (supported % 1 != 0).any()
        or (inside % 1 != 0).any()
    ):
        raise ValueError("Support counts must be integers")

    result["inside_fraction"] = inside / supported

    result["quality_category"] = "C_buffer_only"
    result.loc[inside > 0, "quality_category"] = "B_partial"
    result.loc[
        result["inside_fraction"] >= 0.5,
        "quality_category",
    ] = "A_high_spatial_evidence"

    return result
