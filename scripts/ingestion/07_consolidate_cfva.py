"""Consolidate and audit historical CFVA datasets."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

YEARS = range(2019, 2026)
INPUT_DIR = Path("data/raw/cfva")
OUTPUT_DIR = Path("data/processed/cfva")
REPORT_DIR = Path("outputs/tables")

AREA_FIELDS = ("totale_ha", "ha_totale", "ha_totali")


def main():
    datasets = []
    summary = []
    anomalies = []

    for year in YEARS:
        path = INPUT_DIR / f"cfva_perimeters_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)

        gdf = gpd.read_parquet(path)

        if gdf.crs is None or gdf.crs.to_epsg() != 7791:
            raise ValueError(f"Unexpected CRS: {year}")

        if gdf["fire_id"].isna().any():
            raise ValueError(f"Missing fire IDs: {year}")

        if gdf["fire_id"].duplicated().any():
            raise ValueError(f"Duplicate fire IDs: {year}")

        if not gdf.geometry.is_valid.all():
            raise ValueError(f"Invalid geometries: {year}")

        if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
            raise ValueError(f"Missing or empty geometries: {year}")

        field = next(
            (name for name in AREA_FIELDS if name in gdf.columns),
            None,
        )

        if field is None:
            raise ValueError(f"Official area column missing: {year}")

        gdf["official_area_ha"] = pd.to_numeric(
            gdf[field], errors="coerce"
        )
        gdf["calculated_area_ha"] = gdf.geometry.area / 10000
        gdf["area_difference_ha"] = (
            gdf["calculated_area_ha"] - gdf["official_area_ha"]
        )

        gdf["area_difference_pct"] = (
            100 * gdf["area_difference_ha"].abs()
            / gdf["official_area_ha"].where(
                gdf["official_area_ha"] > 0
            )
        )

        if "temporal_reference_valid" not in gdf.columns:
            print(f"{year}: reconstructing temporal quality columns")

            gdf["fire_date_raw"] = gdf["fire_date"].dt.strftime(
                "%Y-%m-%d"
            )
            gdf["date_quality"] = "valid"
            gdf["date_correction_reason"] = "none"
            gdf["date_corrected"] = False

            gdf["temporal_reference_valid"] = (
                gdf["fire_date"].notna()
                & gdf["fire_date"].dt.year.eq(year)
            )

        temporal_valid = (
            gdf["temporal_reference_valid"].fillna(False).astype(bool)
        )

        summary.append({
            "year": year,
            "records": len(gdf),
            "valid_temporal": int(temporal_valid.sum()),
            "unverified_temporal": int((~temporal_valid).sum()),
            "repaired_geometries": int(gdf["geometry_repaired"].sum()),
            "official_area_ha": gdf["official_area_ha"].sum(),
            "calculated_area_ha": gdf["calculated_area_ha"].sum(),
            "missing_official_area": int(
                gdf["official_area_ha"].isna().sum()
            ),
            "area_difference_gt_1pct": int(
                (gdf["area_difference_pct"] > 1).sum()
            ),
        })

        anomalies.append(
            gdf.loc[
                ~temporal_valid,
                [
                    "fire_id",
                    "year",
                    "fire_date_raw",
                    "date_quality",
                    "date_correction_reason",
                ],
            ].copy()
        )

        if "anno" in gdf.columns:
            gdf["anno"] = pd.to_numeric(
                gdf["anno"], errors="coerce"
            ).astype("Int64")

        datasets.append(gdf)

    combined = gpd.GeoDataFrame(
        pd.concat(datasets, ignore_index=True),
        geometry="geometry",
        crs="EPSG:7791",
    )

    if combined["fire_id"].duplicated().any():
        raise ValueError("Duplicate IDs across years")

    summary_df = pd.DataFrame(summary)
    nonempty_anomalies = [
        df for df in anomalies if not df.empty
    ]

    if nonempty_anomalies:
        anomalies_df = pd.concat(
            nonempty_anomalies, ignore_index=True
        )
    else:
        anomalies_df = pd.DataFrame(columns=[
            "fire_id",
            "year",
            "fire_date_raw",
            "date_quality",
            "date_correction_reason",
        ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / "cfva_2019_2025.parquet"
    combined.to_parquet(output, index=False)

    summary_df.to_csv(
        REPORT_DIR / "cfva_inventory_2019_2025.csv",
        index=False,
    )
    anomalies_df.to_csv(
        REPORT_DIR / "cfva_date_anomalies.csv",
        index=False,
    )

    print("\nCFVA HISTORICAL INVENTORY")
    print(summary_df.to_string(index=False))

    print("\nDATE ANOMALIES")
    print(anomalies_df.to_string(index=False))

    print(f"\nTotal records: {len(combined)}")
    print(f"Unique fire IDs: {combined['fire_id'].nunique()}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
