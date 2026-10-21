"""Replay chronological FIRMS detections and preliminary alert policies."""

from pathlib import Path
from time import perf_counter

import geopandas as gpd
import pandas as pd

from sardinia_fire.clustering.incremental import IncrementalSTDBSCAN
from sardinia_fire.ranking.alert_lifecycle import PreliminaryAlertTracker
from sardinia_fire.utils.config import load_yaml


def main():
    path = Path(
        "data/processed/historical/firms_sardinia_2024_06_09.parquet"
    )
    output_dir = Path("data/processed/operational/alerts")
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = gpd.read_parquet(path)
    detections["acq_datetime_utc"] = pd.to_datetime(
        detections["acq_datetime_utc"], utc=True
    )
    detections = detections.sort_values(
        ["acq_datetime_utc", "detection_id"]
    ).reset_index(drop=True)

    reference = gpd.read_parquet(
        "data/processed/operational/operational_detections_2024_06_09.parquet"
    )

    noise = reference.loc[
        reference["cluster_id"] == -1
    ].copy()

    product_p50 = (
        noise.groupby("source_product")["frp_mw"]
        .median()
        .to_dict()
    )

    trackers = {
        "all": PreliminaryAlertTracker(policy="all"),
        "global_3mw": PreliminaryAlertTracker(
            policy="global", global_threshold_mw=3.0
        ),
        "product_p50": PreliminaryAlertTracker(
            policy="product",
            product_thresholds=product_p50,
        ),
    }

    params = load_yaml(
        "configs/clustering/st_dbscan.yaml"
    )["clustering"]

    clustering = IncrementalSTDBSCAN(
        eps_space_m=params["eps_space_m"],
        eps_time_h=params["eps_time_h"],
        min_samples=params["min_samples"],
    )

    output_paths = {
        name: output_dir / f"alert_transitions_{name}_2024_06_09.csv"
        for name in trackers
    }

    for output in output_paths.values():
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")

    start = perf_counter()
    updates = 0

    for timestamp, batch in detections.groupby(
        "acq_datetime_utc", sort=True
    ):
        snapshot = clustering.update(batch)
        updates += 1

        for tracker in trackers.values():
            tracker.update(snapshot, timestamp)

    elapsed = perf_counter() - start

    final = clustering.snapshot

    assert len(final) == 1385
    assert final["operational_event_id"].nunique() == 121
    assert final["operational_event_id"].isna().sum() == 287

    reference_partition = frozenset(
        frozenset(group["detection_id"])
        for _, group in reference.loc[
            reference["cluster_id"] >= 0
        ].groupby("cluster_id")
    )

    final_partition = frozenset(
        frozenset(group["detection_id"])
        for _, group in final.loc[
            final["operational_event_id"].notna()
        ].groupby("operational_event_id")
    )

    assert final_partition == reference_partition

    print("\nHISTORICAL ALERT LIFECYCLE")
    print("Detections:", len(final))
    print("Updates:", updates)
    print("Final clusters:", final["operational_event_id"].nunique())
    print("Elapsed seconds:", round(elapsed, 2))

    print("\nPOLICY COMPARISON")

    records = []

    for name, tracker in trackers.items():
        states = tracker.get_states()
        transitions = tracker.get_transitions()

        counts = transitions["transition"].value_counts()

        created = int(counts.get("alert_created", 0))
        promoted = int(counts.get("alert_confirmed", 0))
        direct = int(counts.get("confirmed_directly", 0))

        unresolved = int(
            states["state"].eq("preliminary").sum()
        )
        unselected = int(
            states["state"].eq("unselected").sum()
        )
        confirmed = int(
            states["state"].eq("confirmed").sum()
        )

        assert len(states) == 1385
        assert confirmed == 1098
        assert unresolved + unselected == 287
        assert created == promoted + unresolved
        assert promoted + direct == confirmed

        records.append({
            "policy": name,
            "alerts_created": created,
            "alerts_confirmed": promoted,
            "confirmed_directly": direct,
            "preliminary_at_end": unresolved,
            "unselected_at_end": unselected,
            "confirmed_detections_at_end": confirmed,
        })

        transitions.to_csv(
            output_paths[name], index=False
        )

    summary = pd.DataFrame(records)
    print(summary.to_string(index=False))

    print("\nPRODUCT P50 THRESHOLDS (MW)")
    print(pd.Series(product_p50).round(2).to_string())

    print("\nSaved transition logs to:", output_dir)
    print("Historical alert lifecycle validation passed.")


if __name__ == "__main__":
    main()
