"""Audit ambiguous FIRMS-CFVA associations."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("data/processed")
OUTPUT = Path("outputs/tables/firms_cfva_ambiguity_2024_07.csv")

SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "MODIS_SP",
]


def main():
    cfva = gpd.read_parquet(
        ROOT / "cfva/cfva_benchmark_reference.parquet"
    )

    cfva = cfva.loc[
        (cfva["year"] == 2024)
        & cfva["temporal_reference_valid"]
        & cfva["fire_date"].between("2024-07-01", "2024-07-31")
    ].copy()

    polygons = cfva[["fire_id", "fire_date", "geometry"]].copy()
    polygons["geometry"] = polygons.geometry.buffer(1000)

    results = []

    for source in SOURCES:
        points = gpd.read_parquet(
            ROOT / "firms" / f"{source}_2024_07.parquet"
        ).to_crs(cfva.crs)

        matches = gpd.sjoin(
            points,
            polygons,
            how="inner",
            predicate="intersects",
        )

        if matches.empty:
            print(f"{source}: no matches")
            continue

        dates = (
            matches["acq_datetime_utc"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .dt.normalize()
        )

        delta = (dates - matches["fire_date"].dt.normalize()).dt.days

        matches["delta_days"] = delta.to_numpy()
        matches = matches.loc[
            matches["delta_days"].abs() <= 3
        ].copy()

        counts = matches.groupby("detection_id")["fire_id"].nunique()

        print(f"\n{source}")
        print(f"Matched detections: {len(counts)}")
        print(f"Single-fire associations: {(counts == 1).sum()}")
        print(f"Ambiguous detections: {(counts > 1).sum()}")
        print(f"Maximum candidate fires: {counts.max()}")

        matches["candidate_count"] = (
            matches["detection_id"].map(counts)
        )
        matches["source_product"] = source

        results.append(
            matches[
                [
                    "source_product",
                    "detection_id",
                    "fire_id",
                    "delta_days",
                    "candidate_count",
                ]
            ].copy()
        )

    if not results:
        raise RuntimeError("No associations found")

    combined = pd.concat(results, ignore_index=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)

    print("\nOVERALL")
    print(f"Candidate detection-fire pairs: {len(combined)}")
    print(
        "Ambiguous detections:",
        combined.loc[
            combined["candidate_count"] > 1,
            "detection_id",
        ].nunique(),
    )
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
