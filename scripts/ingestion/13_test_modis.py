"""Test historical MODIS FIRMS data over Sardinia."""

from pathlib import Path

from sardinia_fire.ingestion.firms import download_firms_area
from sardinia_fire.preprocessing.firms import (
    normalize_firms,
    save_firms_parquet,
)

SOURCE = "MODIS_SP"
START_DATE = "2024-07-20"
DAYS = 5

OUTPUT = Path("data/processed/firms/modis_20240720_5d.parquet")


def main():
    df = download_firms_area(
        source=SOURCE,
        date=START_DATE,
        days=DAYS,
    )

    print("\nMODIS COLUMNS")
    print(df.columns.tolist())

    print("\nSAMPLE DETECTIONS")
    print(df.head(10).to_string(index=False))

    if df.empty:
        print("No MODIS detections in this period.")
        return

    gdf = normalize_firms(df, source=SOURCE)
    save_firms_parquet(gdf, str(OUTPUT))

    print("\nNORMALIZED MODIS DATASET")
    print(f"Detections: {len(gdf)}")
    print(f"Unique IDs: {gdf['detection_id'].nunique()}")
    print(f"CRS: {gdf.crs}")
    print(f"Sensor: {gdf['sensor'].unique().tolist()}")

    print("\nConfidence:")
    print(gdf["confidence_raw"].value_counts().head(10).to_string())

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
