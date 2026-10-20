"""Evaluate clustered fire detections against CFVA reference events."""

from pathlib import Path

import geopandas as gpd

from sardinia_fire.evaluation.event_matching import match_events
from sardinia_fire.utils.config import load_yaml


def main():
    config = load_yaml("configs/evaluation/event_matching.yaml")
    data = config["data"]
    params = config["evaluation"]

    detections = gpd.read_parquet(data["clusters"])
    reference = gpd.read_parquet(data["reference"])

    reference = reference.loc[
        (reference["year"] == params["reference_year"])
        & reference["temporal_reference_valid"]
        & reference["fire_date"].between(
            params["start_date"],
            params["end_date"],
        )
    ].copy()

    associations = match_events(
        detections=detections,
        reference=reference,
        spatial_tolerance_m=params["spatial_tolerance_m"],
        temporal_tolerance_days=params["temporal_tolerance_days"],
        metric_crs=params["metric_crs"],
        min_matching_detections=params["min_matching_detections"],
    )

    output = Path(data["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    associations.to_csv(output, index=False)

    n_clusters = detections.loc[
        detections["cluster_id"] >= 0, "cluster_id"
    ].nunique()

    matched_clusters = associations["cluster_id"].nunique()
    matched_fires = associations["fire_id"].nunique()

    merging = (
        associations.groupby("cluster_id")["fire_id"].nunique().gt(1).sum()
        if not associations.empty else 0
    )

    splitting = (
        associations.groupby("fire_id")["cluster_id"].nunique().gt(1).sum()
        if not associations.empty else 0
    )

    print("\nEVENT MATCHING - ST-DBSCAN VS CFVA")
    print(f"Reference fires: {len(reference)}")
    print(f"Detected clusters: {n_clusters}")
    print(f"Candidate cluster-fire pairs: {len(associations)}")
    print(f"Clusters with candidates: {matched_clusters}")
    print(f"Reference fires with candidates: {matched_fires}")
    print(f"Clusters without candidates: {n_clusters - matched_clusters}")
    print(f"Reference fires without candidates: {len(reference) - matched_fires}")
    print(f"Clusters with multiple candidate fires: {merging}")
    print(f"Fires with multiple candidate clusters: {splitting}")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
