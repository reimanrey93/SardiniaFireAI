"""Build the multisensor event feature catalog."""

from pathlib import Path

import geopandas as gpd

from sardinia_fire.features.event_features import extract_event_features
from sardinia_fire.features.sensor_thermal import (
    extract_sensor_thermal_features,
)
from sardinia_fire.features.temporal import extract_temporal_features
from sardinia_fire.features.thermal import extract_thermal_features
from sardinia_fire.utils.config import load_yaml


def main():
    config = load_yaml("configs/features/event_features.yaml")
    clustering_config = load_yaml(
        config["data"]["clustering_config"]
    )

    input_path = Path(clustering_config["data"]["output_path"])
    output_path = Path(config["data"]["output_path"])

    params = config["features"]
    group_minutes = params["observation_group_minutes"]

    detections = gpd.read_parquet(input_path)

    basic = extract_event_features(
        detections,
        metric_crs=params["metric_crs"],
    )

    thermal = extract_thermal_features(
        detections,
        group_minutes=group_minutes,
    )

    sensor = extract_sensor_thermal_features(
        detections,
        group_minutes=group_minutes,
    )

    temporal = extract_temporal_features(
        detections,
        group_minutes=group_minutes,
    )

    for name, frame in (
        ("basic", basic),
        ("thermal", thermal),
        ("sensor", sensor),
        ("temporal", temporal),
    ):
        if frame["cluster_id"].duplicated().any():
            raise ValueError(f"Duplicate cluster IDs in {name}")

    features = basic.merge(
        thermal,
        on="cluster_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not features["_merge"].eq("both").all():
        raise ValueError("Missing general thermal features")

    features = features.drop(columns="_merge")

    features = features.merge(
        sensor,
        on="cluster_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not features["_merge"].eq("both").all():
        raise ValueError("Missing sensor-family features")

    features = features.drop(columns="_merge")

    if not (
        features["viirs_detections"]
        + features["modis_detections"]
    ).eq(features["detections"]).all():
        raise ValueError("Sensor detection totals are inconsistent")

    if not (
        features["viirs_groups"]
        + features["modis_groups"]
    ).eq(features["observation_groups"]).all():
        raise ValueError("Sensor observation groups are inconsistent")

    features = features.merge(
        temporal,
        on="cluster_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not features["_merge"].eq("both").all():
        raise ValueError("Missing temporal features")

    features = features.drop(columns="_merge")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print("\nMULTISENSOR FEATURE CATALOG")
    print(f"Events: {len(features)}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"Observation group interval: {group_minutes} minutes")
    print(f"Observation groups: {features['observation_groups'].sum()}")

    columns = [
        "cluster_id",
        "detections",
        "viirs_detections",
        "modis_detections",
        "viirs_groups",
        "modis_groups",
        "viirs_frp_p90_mw",
        "modis_frp_p90_mw",
    ]

    print("\nTEN LARGEST EVENTS")
    print(
        features.sort_values("detections", ascending=False)
        [columns]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
