"""Run the first multisensor ST-DBSCAN experiment."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from sardinia_fire.clustering.st_dbscan import st_dbscan

SOURCES = (
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "MODIS_SP",
)

INPUT_DIR = Path("data/processed/firms")
OUTPUT = Path("data/processed/clustering/st_dbscan_2024_07.parquet")


def main():
    frames = []

    for source in SOURCES:
        path = INPUT_DIR / f"{source}_2024_07.parquet"
        gdf = gpd.read_parquet(path)
        print(f"{source}: {len(gdf)} observations")
        gdf["confidence"] = gdf["confidence"].astype("string")
        gdf["confidence_raw"] = gdf["confidence_raw"].astype("string")

        frames.append(gdf)

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    if combined["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection IDs across products")

    result = st_dbscan(
        combined,
        eps_space_m=1500,
        eps_time_h=24,
        min_samples=3,
    )

    clustered = result["cluster_id"] >= 0

    print("\nST-DBSCAN JULY 2024")
    print(f"Observations: {len(result)}")
    print(f"Clusters: {result.loc[clustered, 'cluster_id'].nunique()}")
    print(f"Clustered observations: {int(clustered.sum())}")
    print(f"Noise observations: {int((~clustered).sum())}")

    print("\nLargest clusters:")
    print(
        result.loc[clustered, "cluster_id"]
        .value_counts()
        .head(10)
        .to_string()
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
