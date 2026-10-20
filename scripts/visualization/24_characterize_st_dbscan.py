"""Summarize ST-DBSCAN clusters from multisensor FIRMS data."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

INPUT = Path("data/processed/clustering/st_dbscan_2024_07.parquet")
OUTPUT = Path("outputs/tables/st_dbscan_clusters_2024_07.csv")


def main():
    gdf = gpd.read_parquet(INPUT)

    if gdf["detection_id"].duplicated().any():
        raise ValueError("Duplicate detection identifiers")

    clustered = gdf.loc[gdf["cluster_id"] >= 0].copy()
    metric = clustered.to_crs("EPSG:32632")

    records = []

    for cluster_id, group in metric.groupby("cluster_id"):
        times = pd.to_datetime(group["acq_datetime_utc"], utc=True)
        frp = pd.to_numeric(group["frp_mw"], errors="coerce")

        xmin, ymin, xmax, ymax = group.total_bounds
        bbox_diagonal_km = (
            ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
        ) / 1000

        first = times.min()
        last = times.max()

        records.append({
            "cluster_id": int(cluster_id),
            "detections": len(group),
            "satellites": group["satellite"].nunique(),
            "products": group["source_product"].nunique(),
            "first_utc": first,
            "last_utc": last,
            "duration_h": round(
                (last - first).total_seconds() / 3600, 2
            ),
            "bbox_diagonal_km": round(bbox_diagonal_km, 3),
            "frp_max_mw": frp.max(),
            "frp_sum_mw": frp.sum(),
            "centroid_x_m": group.geometry.x.mean(),
            "centroid_y_m": group.geometry.y.mean(),
        })

    summary = pd.DataFrame(records).sort_values(
        "detections", ascending=False
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print("\nST-DBSCAN CLUSTER CHARACTERIZATION")
    print(f"Clusters: {len(summary)}")
    print(f"Clustered detections: {len(clustered)}")
    print(f"Noise detections: {(gdf['cluster_id'] == -1).sum()}")

    print("\nTEN LARGEST CLUSTERS")
    print(summary.head(10).to_string(index=False))

    print("\nCLUSTER 13 — SATELLITE COMPOSITION")
    largest = gdf.loc[gdf["cluster_id"] == 13]
    print(largest["source_product"].value_counts().to_string())

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
