"""Sensitivity analysis of FIRMS-CFVA spatial-temporal associations."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("data/processed")
OUTPUT = Path("outputs/tables/firms_cfva_sensitivity_2024_07.csv")

SOURCES = (
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "MODIS_SP",
)

DISTANCES_M = (0, 375, 500, 1000, 2000)
WINDOWS_DAYS = (0, 1, 3, 7)


def main():
    cfva = gpd.read_parquet(
        ROOT / "cfva/cfva_benchmark_reference.parquet"
    )

    cfva = cfva.loc[
        (cfva["year"] == 2024)
        & cfva["temporal_reference_valid"]
        & cfva["fire_date"].between(
            "2024-07-01", "2024-07-31"
        )
    ].copy()

    fires = set(cfva["fire_id"])
    points_by_source = {}

    for source in SOURCES:
        path = ROOT / "firms" / f"{source}_2024_07.parquet"
        points_by_source[source] = gpd.read_parquet(path).to_crs(
            cfva.crs
        )

    records = []

    for distance in DISTANCES_M:
        polygons = cfva[
            ["fire_id", "fire_date", "geometry"]
        ].copy()

        if distance > 0:
            polygons["geometry"] = polygons.geometry.buffer(distance)

        candidate_pairs = {}

        for source, points in points_by_source.items():
            matches = gpd.sjoin(
                points,
                polygons,
                how="inner",
                predicate="intersects",
            )

            if not matches.empty:
                observation_date = (
                    matches["acq_datetime_utc"]
                    .dt.tz_convert("UTC")
                    .dt.tz_localize(None)
                    .dt.normalize()
                )
                reference_date = matches["fire_date"].dt.normalize()
                matches = matches.copy()
                matches["delta_days"] = (
                    observation_date - reference_date
                ).dt.days.abs()

            candidate_pairs[source] = matches

        for window in WINDOWS_DAYS:
            associated = {}

            for source, pairs in candidate_pairs.items():
                if pairs.empty:
                    associated[source] = set()
                else:
                    associated[source] = set(
                        pairs.loc[
                            pairs["delta_days"] <= window,
                            "fire_id",
                        ]
                    )

            viirs = set().union(
                associated["VIIRS_SNPP_SP"],
                associated["VIIRS_NOAA20_SP"],
                associated["VIIRS_NOAA21_SP"],
            )

            modis = associated["MODIS_SP"]
            combined = viirs | modis

            records.append({
                "distance_m": distance,
                "window_days": window,
                "cfva_fires": len(fires),
                "viirs_associated": len(viirs),
                "modis_associated": len(modis),
                "combined_associated": len(combined),
                "unassociated": len(fires - combined),
                "association_pct": round(
                    100 * len(combined) / len(fires), 2
                ),
            })

    result = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("\nFIRMS-CFVA SENSITIVITY ANALYSIS")
    print(result.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
