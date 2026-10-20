"""Conservative CFVA area labels for candidate fire events."""

import numpy as np
import pandas as pd


def build_reference_labels(
    events: pd.DataFrame,
    associations: pd.DataFrame,
    reference: pd.DataFrame,
    min_matching_detections: int = 2,
) -> pd.DataFrame:
    """Attach CFVA area only to unambiguous one-to-one candidates."""

    if min_matching_detections < 1:
        raise ValueError("Minimum matching detections must be positive")

    event_cols = {"cluster_id"}
    pair_cols = {"cluster_id", "fire_id", "matching_detections"}
    fire_cols = {
        "fire_id",
        "benchmark_area_ha",
        "area_review_required",
        "temporal_reference_valid",
        "spatial_reference_valid",
    }

    for name, frame, required in (
        ("events", events, event_cols),
        ("associations", associations, pair_cols),
        ("reference", reference, fire_cols),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing {sorted(missing)}")

    if events["cluster_id"].duplicated().any():
        raise ValueError("Duplicate event IDs")

    if reference["fire_id"].duplicated().any():
        raise ValueError("Duplicate reference fire IDs")

    if associations.duplicated(["cluster_id", "fire_id"]).any():
        raise ValueError("Duplicate candidate pairs")

    counts_event = associations.groupby("cluster_id")["fire_id"].nunique()
    counts_fire = associations.groupby("fire_id")["cluster_id"].nunique()

    pairs = associations.copy()
    pairs["event_candidates"] = pairs["cluster_id"].map(counts_event)
    pairs["fire_candidates"] = pairs["fire_id"].map(counts_fire)

    pairs = pairs.loc[
        (pairs["event_candidates"] == 1)
        & (pairs["fire_candidates"] == 1)
        & (pairs["matching_detections"] >= min_matching_detections)
    ].copy()

    columns = list(fire_cols)
    selected = pairs.merge(
        reference[columns],
        on="fire_id",
        how="left",
        validate="many_to_one",
    )

    area = pd.to_numeric(
        selected["benchmark_area_ha"], errors="coerce"
    )

    good = (
        selected["temporal_reference_valid"].eq(True)
        & selected["spatial_reference_valid"].eq(True)
        & selected["area_review_required"].eq(False)
        & np.isfinite(area)
        & area.gt(0)
    )

    selected = selected.loc[good].copy()
    selected["reference_area_ha"] = area.loc[good].to_numpy()

    selected = selected[
        ["cluster_id", "fire_id", "reference_area_ha"]
    ].rename(columns={"fire_id": "reference_fire_id"})

    result = events.merge(
        selected,
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )

    result["label_status"] = np.where(
        result["reference_fire_id"].notna(),
        "provisional_one_to_one",
        "unlabeled",
    )

    return result
