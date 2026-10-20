"""Audit CFVA wildfire perimeter geometries."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import from_wkb

INPUT = Path("data/raw/cfva/cfva_perimeters_2024.parquet")
OUTPUT = Path("outputs/tables/cfva_geometry_audit_2024.csv")


def main():
    gdf = gpd.read_parquet(INPUT)

    original = gpd.GeoSeries(
        from_wkb(gdf["geometry_original"].apply(bytes.fromhex).tolist()),
        crs=gdf.crs,
    )

    audit = pd.DataFrame({
        "cod_incend": gdf["cod_incend"],
        "geometry_repaired": gdf["geometry_repaired"],
        "original_area_m2": original.area,
        "repaired_area_m2": gdf.geometry.area,
        "official_area_m2": pd.to_numeric(gdf["totale_mq"]),
    })

    audit["area_difference_m2"] = (
        audit["repaired_area_m2"] - audit["original_area_m2"]
    )

    audit["relative_difference_pct"] = (
        100 * audit["area_difference_m2"]
        / audit["original_area_m2"].replace(0, float("nan"))
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT, index=False)

    print(f"Total records: {len(audit)}")
    print(f"Repaired records: {audit['geometry_repaired'].sum()}")

    print("\nRepaired geometry audit:")
    print(
        audit.loc[audit["geometry_repaired"]].to_string(index=False)
    )

    print(f"\nAudit saved: {OUTPUT}")


if __name__ == "__main__":
    main()
