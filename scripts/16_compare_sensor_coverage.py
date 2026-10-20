"""Compare VIIRS and MODIS associations with CFVA fires."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("data/processed")
OUT = Path("outputs/tables")
SOURCES = ["VIIRS_NOAA20_SP", "MODIS_SP"]

def associated_fires(source, cfva):
    points = gpd.read_parquet(
        ROOT / "firms" / f"{source}_2024_07.parquet"
    ).to_crs(cfva.crs)

    buffered = cfva[["fire_id", "fire_date", "geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(1000)

    matches = gpd.sjoin(
        points,
        buffered,
        how="inner",
        predicate="intersects",
    )

    if matches.empty:
        return set()

    delta = (
        matches["acq_datetime_utc"].dt.tz_localize(None).dt.normalize()
        - matches["fire_date"].dt.normalize()
    ).dt.days.abs()

    return set(matches.loc[delta <= 3, "fire_id"])


def main():
    cfva = gpd.read_parquet(
        ROOT / "cfva" / "cfva_benchmark_reference.parquet"
    )

    cfva = cfva.loc[
        (cfva["year"] == 2024)
        & cfva["temporal_reference_valid"]
        & cfva["fire_date"].between("2024-07-01", "2024-07-31")
    ].copy()

    viirs = associated_fires(SOURCES[0], cfva)
    modis = associated_fires(SOURCES[1], cfva)

    both = viirs & modis
    only_viirs = viirs - modis
    only_modis = modis - viirs
    neither = set(cfva["fire_id"]) - (viirs | modis)

    counts = {
        "Both sensors": len(both),
        "VIIRS only": len(only_viirs),
        "MODIS only": len(only_modis),
        "Neither sensor": len(neither),
    }

    print("\nCFVA JULY 2024 — SENSOR ASSOCIATION")
    print(f"CFVA perimeters: {len(cfva)}")
    for label, value in counts.items():
        print(f"{label}: {value}")
    print(f"Associated by at least one sensor: {len(viirs | modis)}")

    if sum(counts.values()) != len(cfva):
        raise ValueError("Association totals are inconsistent")

    records = []
    for fire_id in sorted(set(cfva["fire_id"])):
        records.append({
            "fire_id": fire_id,
            "viirs_associated": fire_id in viirs,
            "modis_associated": fire_id in modis,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "cfva_firms_sensor_coverage_2024_07.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
