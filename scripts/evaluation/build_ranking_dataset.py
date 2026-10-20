"""Build a provisional CFVA-labeled event ranking dataset."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from sardinia_fire.evaluation.reference_labels import build_reference_labels
from sardinia_fire.evaluation.spatial_audit import audit_reference_labels
from sardinia_fire.utils.config import load_yaml


def main():
    config = load_yaml("configs/evaluation/event_matching.yaml")
    params = config["evaluation"]
    data = config["data"]

    features_path = Path(
        "data/processed/features/st_dbscan_events_2024_07.parquet"
    )
    associations_path = Path(data["output"])
    reference_path = Path(data["reference"])
    output_path = Path(
        "data/processed/evaluation/ranking_dataset_2024_07.parquet"
    )

    features = pd.read_parquet(features_path)
    associations = pd.read_csv(associations_path)
    reference = gpd.read_parquet(reference_path)

    reference = reference.loc[
        (reference["year"] == params["reference_year"])
        & reference["fire_date"].between(
            params["start_date"],
            params["end_date"],
        )
    ].copy()

    if reference["fire_id"].duplicated().any():
        raise ValueError("Duplicate CFVA reference fire IDs")

    dataset = build_reference_labels(
        events=features,
        associations=associations,
        reference=reference,
        min_matching_detections=2,
    )

    detections = gpd.read_parquet(
        'data/processed/clustering/st_dbscan_2024_07.parquet'
    )

    audit = audit_reference_labels(
        dataset=dataset,
        detections=detections,
        reference=reference,
        spatial_tolerance_m=params['spatial_tolerance_m'],
        temporal_tolerance_days=params['temporal_tolerance_days'],
    )

    dataset = dataset.merge(
        audit,
        on='cluster_id',
        how='left',
        validate='one_to_one',
    )

    if len(audit) != dataset['reference_area_ha'].notna().sum():
        raise ValueError('Incomplete spatial label audit')

    labeled = dataset.loc[
        dataset["label_status"] == "provisional_one_to_one"
    ].copy()

    if len(dataset) != len(features):
        raise ValueError("Event count changed during labeling")

    if dataset["cluster_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in output")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    print("\nCFVA RANKING DATASET")
    print(f"Total events: {len(dataset)}")
    print(f"Labeled events: {len(labeled)}")
    print(f"Unlabeled events: {len(dataset) - len(labeled)}")

    print("\nSPATIAL EVIDENCE CATEGORIES")
    print(
        dataset["quality_category"]
        .value_counts(dropna=False)
        .to_string()
    )

    if not labeled.empty:
        print("\nREFERENCE AREA DISTRIBUTION (ha)")
        print(labeled["reference_area_ha"].describe().round(2))

        print("\nLABELED EVENTS")
        print(
            labeled[
                [
                    "cluster_id",
                    "reference_fire_id",
                    "reference_area_ha",
                    "detections",
                    "frp_max_mw",
                    "viirs_frp_p90_mw",
                    "modis_frp_p90_mw",
                ]
            ]
            .sort_values("reference_area_ha", ascending=False)
            .round(2)
            .to_string(index=False)
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
