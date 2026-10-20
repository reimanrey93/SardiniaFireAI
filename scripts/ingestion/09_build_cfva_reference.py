"""Build a quality-controlled CFVA benchmark reference."""

from pathlib import Path

import geopandas as gpd

INPUT = Path("data/processed/cfva/cfva_2019_2025.parquet")
OUTPUT = Path("data/processed/cfva/cfva_benchmark_reference.parquet")
REPORT = Path("outputs/tables/cfva_area_quality_flags.csv")


def main():
    gdf = gpd.read_parquet(INPUT)

    fields = [
        "fire_id",
        "source_fire_id",
        "wfs_feature_id",
        "year",
        "fire_date",
        "fire_date_raw",
        "date_quality",
        "temporal_reference_valid",
        "geometry_repaired",
        "official_area_ha",
        "calculated_area_ha",
        "area_difference_ha",
        "area_difference_pct",
        "geometry",
    ]

    reference = gdf[fields].copy()

    reference["area_review_required"] = (
        (reference["area_difference_ha"].abs() > 0.05)
        & (reference["area_difference_pct"] > 1.0)
    ).fillna(True)

    reference["benchmark_area_ha"] = reference["calculated_area_ha"]

    reference["spatial_reference_valid"] = (
        reference.geometry.notna()
        & ~reference.geometry.is_empty
        & reference.geometry.is_valid
    )

    if reference["fire_id"].duplicated().any():
        raise ValueError("Duplicate benchmark fire IDs")

    if not reference["spatial_reference_valid"].all():
        raise ValueError("Invalid spatial references")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    reference.to_parquet(OUTPUT, index=False)

    flagged = reference.loc[
        reference["area_review_required"],
        [
            "fire_id",
            "year",
            "official_area_ha",
            "calculated_area_ha",
            "area_difference_ha",
            "area_difference_pct",
        ],
    ].copy()

    flagged.to_csv(REPORT, index=False)

    print("\nCFVA BENCHMARK REFERENCE")
    print(f"Total records: {len(reference)}")
    print(
        "Valid spatial references:",
        int(reference["spatial_reference_valid"].sum()),
    )
    print(
        "Valid temporal references:",
        int(reference["temporal_reference_valid"].sum()),
    )
    print(
        "Area review required:",
        int(reference["area_review_required"].sum()),
    )

    print("\nAREA QUALITY FLAGS")
    print(flagged.to_string(index=False))

    print(f"\nSaved: {OUTPUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
