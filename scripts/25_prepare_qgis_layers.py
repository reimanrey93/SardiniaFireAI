"""Prepare ST-DBSCAN and CFVA layers for QGIS inspection."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

CLUSTERS = Path("data/processed/clustering/st_dbscan_2024_07.parquet")
CFVA = Path("data/processed/cfva/cfva_benchmark_reference.parquet")
SUMMARY = Path("outputs/tables/st_dbscan_clusters_2024_07.csv")
OUT = Path("outputs/qgis")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    points = gpd.read_parquet(CLUSTERS)
    summary = pd.read_csv(SUMMARY)

    if points["cluster_id"].ge(0).sum() != 537:
        raise ValueError("Unexpected number of clustered observations")

    centroids = gpd.GeoDataFrame(
        summary.drop(columns=["centroid_x_m", "centroid_y_m"]),
        geometry=gpd.points_from_xy(
            summary["centroid_x_m"],
            summary["centroid_y_m"],
        ),
        crs="EPSG:32632",
    ).to_crs("EPSG:4326")

    centroids_path = OUT / "st_dbscan_cluster_centroids.parquet"
    centroids.to_parquet(centroids_path, index=False)

    cluster13 = points.loc[points["cluster_id"] == 13].copy()
    cluster13_path = OUT / "st_dbscan_cluster13_points.parquet"
    cluster13.to_parquet(cluster13_path, index=False)

    official = gpd.read_parquet(CFVA)
    official = official.loc[
        (official["year"] == 2024)
        & official["temporal_reference_valid"]
        & official["fire_date"].between("2024-07-01", "2024-07-31")
    ].copy()

    official_path = OUT / "cfva_july2024.parquet"
    official.to_crs("EPSG:4326").to_parquet(
        official_path, index=False
    )

    print("\nQGIS LAYERS")
    print(f"Cluster centroids: {len(centroids)}")
    print(f"Cluster 13 points: {len(cluster13)}")
    print(f"CFVA reference polygons: {len(official)}")

    print("\nSAVED FILES")
    print(centroids_path)
    print(cluster13_path)
    print(official_path)


if __name__ == "__main__":
    main()
