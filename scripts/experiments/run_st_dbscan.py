"""Run the ST-DBSCAN experiment from its YAML configuration."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from sardinia_fire.clustering.st_dbscan import st_dbscan
from sardinia_fire.utils.config import load_yaml


def main():
    config = load_yaml("configs/clustering/st_dbscan.yaml")
    experiment = config["experiment"]
    data = config["data"]
    params = config["clustering"]

    input_dir = Path(data["input_dir"])
    output_path = Path(data["output_path"])
    frames = []

    for source in data["sources"]:
        filename = (
            f"{source}_{experiment['year']}_"
            f"{experiment['month']:02d}.parquet"
        )
        observations = gpd.read_parquet(input_dir / filename)

        for column in ("confidence", "confidence_raw"):
            if column in observations.columns:
                observations[column] = observations[column].astype("string")

        print(f"{source}: {len(observations)} observations")
        frames.append(observations)

    if not frames:
        raise ValueError("No FIRMS sources configured")

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )

    if combined["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection identifiers")

    if params.get("metric_crs", "EPSG:32632") != "EPSG:32632":
        raise ValueError(
            "Current ST-DBSCAN implementation requires EPSG:32632"
        )

    result = st_dbscan(
        combined,
        eps_space_m=params["eps_space_m"],
        eps_time_h=params["eps_time_h"],
        min_samples=params["min_samples"],
    )

    clustered = result["cluster_id"] >= 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    print("\nST-DBSCAN RESULTS")
    print(f"Experiment: {experiment['name']}")
    print(f"Observations: {len(result)}")
    print(f"Clusters: {result.loc[clustered, 'cluster_id'].nunique()}")
    print(f"Clustered observations: {int(clustered.sum())}")
    print(f"Noise: {int((~clustered).sum())}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
