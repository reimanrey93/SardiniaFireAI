"""Investigate CFVA official versus geometric area discrepancies."""

from pathlib import Path

import geopandas as gpd

INPUT = Path("data/processed/cfva/cfva_2019_2025.parquet")
OUTPUT = Path("outputs/tables/cfva_area_anomalies.csv")

AREA_FIELDS = [
    "ha_totale",
    "ha_totali",
    "totale_ha",
    "mq_totale",
    "mq_totali",
    "sup_tot_mq",
    "totale_mq",
]


def main():
    gdf = gpd.read_parquet(INPUT)
    df = gdf.drop(columns="geometry").copy()

    df["absolute_difference_ha"] = df["area_difference_ha"].abs()

    print("\n2021: LARGEST ABSOLUTE DISCREPANCIES")
    subset = df.loc[df["year"] == 2021].copy()
    top = subset.nlargest(12, "absolute_difference_ha")

    columns = [
        "fire_id",
        "source_fire_id",
        "official_area_ha",
        "calculated_area_ha",
        "area_difference_ha",
        "area_difference_pct",
    ]

    columns += [c for c in AREA_FIELDS if c in top.columns]

    print(top[columns].to_string(index=False))

    print("\n2019: LARGEST RELATIVE DISCREPANCIES")
    subset = df.loc[df["year"] == 2019].copy()
    top_2019 = subset.nlargest(12, "area_difference_pct")
    print(top_2019[columns].to_string(index=False))

    print("\nAREA STATISTICS BY YEAR")
    for year, group in df.groupby("year"):
        diff = group["absolute_difference_ha"]
        print(
            f"{year}: max_abs={diff.max():.6f} ha, "
            f"median_abs={diff.median():.6f} ha, "
            f"sum_abs={diff.sum():.6f} ha"
        )

    anomalies = df.loc[
        (df["area_difference_pct"] > 1)
        | (df["absolute_difference_ha"] > 1)
    ].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    anomalies.to_csv(OUTPUT, index=False)

    print(f"\nFlagged records: {len(anomalies)}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
