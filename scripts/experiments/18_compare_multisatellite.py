"""Compare four FIRMS products against CFVA perimeters."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("data/processed")
OUT = Path("outputs/tables")

SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "MODIS_SP",
]

BUFFER_M = 1000
WINDOW_DAYS = 3


def match_product(source, cfva, buffers):
    path = ROOT / "firms" / f"{source}_2024_07.parquet"
    points = gpd.read_parquet(path)

    if points["detection_id"].duplicated().any():
        raise ValueError(f"Duplicated detection IDs: {source}")

    points = points.to_crs(cfva.crs)

    candidates = gpd.sjoin(
        points,
        buffers,
        how="inner",
        predicate="intersects",
    )

    if candidates.empty:
        return set(), 0, len(points)

    detection_dates = (
        candidates["acq_datetime_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    fire_dates = candidates["fire_date"].dt.normalize()

    delta = (detection_dates - fire_dates).dt.days.abs()
    matched = candidates.loc[delta <= WINDOW_DAYS]

    return (
        set(matched["fire_id"]),
        matched["detection_id"].nunique(),
        len(points),
    )


def main():
    cfva = gpd.read_parquet(
        ROOT / "cfva" / "cfva_benchmark_reference.parquet"
    )

    cfva = cfva.loc[
        (cfva["year"] == 2024)
        & cfva["temporal_reference_valid"]
        & cfva["fire_date"].between("2024-07-01", "2024-07-31")
    ].copy()

    buffers = cfva[["fire_id", "fire_date", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(BUFFER_M)

    associations = {}
    summary = []

    for source in SOURCES:
        fire_ids, matched_detections, total_detections = match_product(
            source, cfva, buffers
        )

        associations[source] = fire_ids

        summary.append({
            "source": source,
            "detections": total_detections,
            "matched_detections": matched_detections,
            "associated_fires": len(fire_ids),
        })

    all_fires = set(cfva["fire_id"])

    viirs_union = set().union(
        associations["VIIRS_SNPP_SP"],
        associations["VIIRS_NOAA20_SP"],
        associations["VIIRS_NOAA21_SP"],
    )

    modis = associations["MODIS_SP"]
    combined = viirs_union | modis

    print("\nMULTISATELLITE FIRMS-CFVA COMPARISON")
    print(f"CFVA fires in July: {len(all_fires)}")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\nCOMBINED COVERAGE")
    print(f"VIIRS union: {len(viirs_union)}")
    print(f"MODIS: {len(modis)}")
    print(f"VIIRS + MODIS: {len(combined)}")
    print(f"VIIRS only: {len(viirs_union - modis)}")
    print(f"MODIS only: {len(modis - viirs_union)}")
    print(f"Both sensor families: {len(viirs_union & modis)}")
    print(f"No association: {len(all_fires - combined)}")

    records = []
    for fire_id in sorted(all_fires):
        row = {"fire_id": fire_id}
        for source in SOURCES:
            row[source] = fire_id in associations[source]
        row["any_viirs"] = fire_id in viirs_union
        row["any_sensor"] = fire_id in combined
        records.append(row)

    OUT.mkdir(parents=True, exist_ok=True)

    output = OUT / "cfva_multisatellite_coverage_2024_07.csv"
    pd.DataFrame(records).to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
