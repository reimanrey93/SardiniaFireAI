"""Exploratory spatial-temporal comparison of FIRMS and CFVA."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

FIRMS_DIR = Path("data/processed/firms")
CFVA_PATH = Path("data/processed/cfva/cfva_benchmark_reference.parquet")
OUTPUT = Path("outputs/tables/firms_cfva_july2024_comparison.csv")

SOURCES = ["VIIRS_NOAA20_SP", "MODIS_SP"]
DISTANCE_M = 1000
TIME_WINDOW_DAYS = 3


def main():
    cfva = gpd.read_parquet(CFVA_PATH)

    cfva = cfva.loc[
        (cfva["year"] == 2024)
        & cfva["temporal_reference_valid"]
        & cfva["fire_date"].between("2024-07-01", "2024-07-31")
    ].copy()

    print(f"CFVA July 2024 perimeters: {len(cfva)}")

    buffers = cfva[["fire_id", "fire_date", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(DISTANCE_M)

    summaries = []

    for source in SOURCES:
        path = FIRMS_DIR / f"{source}_2024_07.parquet"
        points = gpd.read_parquet(path).to_crs(cfva.crs)

        candidates = gpd.sjoin(
            points,
            buffers,
            how="inner",
            predicate="intersects",
        )

        if not candidates.empty:
            delta = (
                candidates["acq_datetime_utc"].dt.tz_localize(None).dt.normalize()
                - candidates["fire_date"].dt.normalize()
            ).dt.days.abs()

            matched = candidates.loc[delta <= TIME_WINDOW_DAYS].copy()
        else:
            matched = candidates.copy()

        matched_ids = matched["detection_id"].nunique()
        matched_fires = matched["fire_id"].nunique()

        summaries.append({
            "source": source,
            "total_detections": len(points),
            "matched_detections": matched_ids,
            "matched_cfva_fires": matched_fires,
            "spatial_candidates": candidates["detection_id"].nunique(),
        })

        print(f"\n{source}")
        print(f"Detections: {len(points)}")
        print(
            "Spatial candidates:",
            candidates["detection_id"].nunique(),
        )
        print(f"Spatial-temporal matches: {matched_ids}")
        print(f"Matched CFVA fire IDs: {matched_fires}")

    summary = pd.DataFrame(summaries)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print("\nCOMPARISON SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
