"""Replay FIRMS observations chronologically and save operational events."""

from pathlib import Path
from time import perf_counter

import geopandas as gpd
import pandas as pd

from sardinia_fire.clustering.incremental import IncrementalSTDBSCAN
from sardinia_fire.utils.config import load_yaml


def main():
    params = load_yaml(
        "configs/clustering/st_dbscan.yaml"
    )["clustering"]

    input_path = Path(
        "data/processed/historical/firms_sardinia_2024_06_09.parquet"
    )
    reference_path = Path(
        "data/processed/clustering/st_dbscan_2024_06_09.parquet"
    )

    output_dir = Path("data/processed/operational")

    event_path = output_dir / "operational_events_2024_06_09.parquet"
    detection_path = output_dir / "operational_detections_2024_06_09.parquet"
    history_path = output_dir / "operational_transitions_2024_06_09.csv"

    for path in (event_path, detection_path, history_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    detections = gpd.read_parquet(input_path)
    reference = gpd.read_parquet(reference_path)

    detections["acq_datetime_utc"] = pd.to_datetime(
        detections["acq_datetime_utc"], utc=True
    )

    detections = detections.sort_values(
        ["acq_datetime_utc", "detection_id"]
    ).reset_index(drop=True)

    tracker = IncrementalSTDBSCAN(
        eps_space_m=params["eps_space_m"],
        eps_time_h=params["eps_time_h"],
        min_samples=params["min_samples"],
    )

    start = perf_counter()
    updates = 0

    for _, batch in detections.groupby(
        "acq_datetime_utc", sort=True
    ):
        tracker.update(batch)
        updates += 1

    elapsed = perf_counter() - start
    final = tracker.snapshot.copy()
    history = tracker.get_history()

    def get_partition(frame, column, noise_value=None):
        valid = (
            frame[column].notna()
            if noise_value is None
            else frame[column] != noise_value
        )

        clusters = frozenset(
            frozenset(group["detection_id"])
            for _, group in frame.loc[valid].groupby(column)
        )

        noise = frozenset(frame.loc[~valid, "detection_id"])
        return clusters, noise

    assert get_partition(
        final, "operational_event_id"
    ) == get_partition(
        reference, "cluster_id", noise_value=-1
    )

    created = history.loc[
        history["event_type"] == "created",
        ["event_id", "timestamp_utc"],
    ].rename(
        columns={
            "event_id": "operational_event_id",
            "timestamp_utc": "identified_utc",
        }
    )

    assert created["operational_event_id"].is_unique

    events = (
        final.loc[final["operational_event_id"].notna()]
        .groupby("operational_event_id")
        .agg(
            first_detection_utc=("acq_datetime_utc", "min"),
            last_detection_utc=("acq_datetime_utc", "max"),
            detections=("detection_id", "size"),
            frp_max_mw=("frp_mw", "max"),
        )
        .reset_index()
        .merge(
            created,
            on="operational_event_id",
            how="left",
            validate="one_to_one",
        )
    )

    events["identification_delay_h"] = (
        pd.to_datetime(events["identified_utc"], utc=True)
        - events["first_detection_utc"]
    ).dt.total_seconds() / 3600

    assert len(events) == 121
    assert events["identified_utc"].notna().all()
    assert events["identification_delay_h"].ge(0).all()

    output_dir.mkdir(parents=True, exist_ok=True)

    events.to_parquet(event_path, index=False)
    final.to_parquet(detection_path, index=False)
    history.to_csv(history_path, index=False)

    print("\nOPERATIONAL CATALOG")
    print("Updates:", updates)
    print("Observations:", len(final))
    print("Events:", len(events))
    print("Noise:", int(final["operational_event_id"].isna().sum()))
    print("Elapsed seconds:", round(elapsed, 2))

    print("\nIDENTIFICATION DELAY")
    print(
        events["identification_delay_h"]
        .describe(percentiles=[0.5, 0.9, 0.95])
        .round(3)
        .to_string()
    )

    print("\nTRANSITIONS")
    print(history["event_type"].value_counts().to_string())

    print("\nSaved:", event_path)
    print("Saved:", detection_path)
    print("Saved:", history_path)

    print("\nOperational catalog validation passed.")


if __name__ == "__main__":
    main()
