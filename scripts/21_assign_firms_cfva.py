"""Rank FIRMS-CFVA candidate associations without discarding ambiguity."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("data/processed")
OUT = Path("outputs/tables")

SOURCES = (
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "MODIS_SP",
)

BUFFER_M = 1000
WINDOW_DAYS = 3


def main():
    cfva = gpd.read_parquet(
        ROOT / "cfva/cfva_benchmark_reference.parquet"
    )

    cfva = cfva.loc[
        (cfva["year"] == 2024)
        & cfva["temporal_reference_valid"]
        & cfva["fire_date"].between("2024-07-01", "2024-07-31")
    ].copy()

    cfva = cfva[["fire_id", "fire_date", "geometry"]]
    cfva = cfva.reset_index(drop=True)

    buffers = cfva.copy()
    buffers["geometry"] = buffers.geometry.buffer(BUFFER_M)

    all_candidates = []

    for source in SOURCES:
        points = gpd.read_parquet(
            ROOT / "firms" / f"{source}_2024_07.parquet"
        ).to_crs(cfva.crs)

        points = points.reset_index(drop=True)

        matches = gpd.sjoin(
            points,
            buffers,
            how="inner",
            predicate="intersects",
        ).reset_index(drop=True)

        if matches.empty:
            continue

        observation_dates = (
            matches["acq_datetime_utc"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .dt.normalize()
        )

        matches["delta_days"] = (
            observation_dates - matches["fire_date"].dt.normalize()
        ).dt.days

        matches = matches.loc[
            matches["delta_days"].abs() <= WINDOW_DAYS
        ].copy()

        if matches.empty:
            continue

        original_polygons = cfva.geometry.iloc[
            matches["index_right"].to_numpy()
        ].reset_index(drop=True)

        points_geometry = matches.geometry.reset_index(drop=True)

        matches["distance_m"] = points_geometry.distance(
            original_polygons
        ).to_numpy()

        matches["source_product"] = source

        all_candidates.append(matches[
            [
                "source_product",
                "detection_id",
                "fire_id",
                "delta_days",
                "distance_m",
            ]
        ].copy())

    if not all_candidates:
        raise RuntimeError("No candidate associations found")

    candidates = pd.concat(all_candidates, ignore_index=True)

    candidates = candidates.sort_values(
        ["detection_id", "distance_m", "delta_days", "fire_id"],
        key=None,
        kind="mergesort",
    )

    candidates["abs_delta_days"] = candidates["delta_days"].abs()

    candidates = candidates.sort_values(
        [
            "detection_id",
            "distance_m",
            "abs_delta_days",
            "fire_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    candidates["candidate_count"] = (
        candidates.groupby("detection_id")["fire_id"]
        .transform("nunique")
    )

    candidates["candidate_rank"] = (
        candidates.groupby("detection_id").cumcount() + 1
    )

    selected = candidates.loc[
        candidates["candidate_rank"] == 1
    ].copy()

    selected["ambiguous"] = selected["candidate_count"] > 1

    OUT.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(
        OUT / "firms_cfva_all_candidates_2024_07.csv",
        index=False,
    )

    selected.to_csv(
        OUT / "firms_cfva_provisional_assignment_2024_07.csv",
        index=False,
    )

    print("\nFIRMS-CFVA PROVISIONAL ASSIGNMENT")
    print(f"Candidate pairs: {len(candidates)}")
    print(f"Assigned detections: {len(selected)}")
    print(f"Ambiguous detections: {int(selected['ambiguous'].sum())}")
    print(f"Distinct selected CFVA fires: {selected['fire_id'].nunique()}")

    print("\nASSIGNMENTS BY PRODUCT")
    print(
        selected.groupby("source_product")
        .agg(
            detections=("detection_id", "count"),
            ambiguous=("ambiguous", "sum"),
            assigned_fires=("fire_id", "nunique"),
        )
        .to_string()
    )

    print("\nFiles saved to outputs/tables/")


if __name__ == "__main__":
    main()
