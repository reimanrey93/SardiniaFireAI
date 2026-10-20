"""Download and summarize historical CFVA wildfire perimeters."""

from pathlib import Path

import pandas as pd

from sardinia_fire.ingestion.cfva import download_cfva, save_cfva

YEARS = range(2019, 2026)
SUMMARY_PATH = Path("outputs/tables/cfva_inventory_2019_2025.csv")

AREA_FIELDS = [
    "totale_ha",
    "ha_totale",
    "ha_totali",
]


def main():
    records = []

    for year in YEARS:
        print(f"\n{'=' * 60}")
        print(f"Processing CFVA {year}")
        print("=" * 60)

        try:
            gdf = download_cfva(year)

            area_field = next(
                (field for field in AREA_FIELDS if field in gdf.columns),
                None,
            )

            if area_field:
                total_area = pd.to_numeric(
                    gdf[area_field],
                    errors="coerce",
                ).sum()
            else:
                total_area = gdf.geometry.area.sum() / 10000

            output = save_cfva(gdf, year)

            records.append({
                "year": year,
                "records": len(gdf),
                "unique_fires": gdf["fire_id"].nunique(),
                "repaired_geometries": int(
                    gdf["geometry_repaired"].sum()
                ),
                "first_date": gdf["fire_date"].min().date(),
                "last_date": gdf["fire_date"].max().date(),
                "total_area_ha": round(float(total_area), 3),
                "area_source": area_field or "geometry",
                "status": "success",
            })

            print(f"Saved: {output}")

        except Exception as error:
            print(f"FAILED {year}: {error}")
            records.append({
                "year": year,
                "status": "failed",
                "error": str(error),
            })

    summary = pd.DataFrame(records)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nCFVA INVENTORY SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved: {SUMMARY_PATH}")

    if (summary["status"] != "success").any():
        raise RuntimeError("Some annual downloads failed")


if __name__ == "__main__":
    main()
