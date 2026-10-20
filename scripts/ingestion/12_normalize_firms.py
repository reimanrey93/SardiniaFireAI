"""Normalize the first NOAA-20 FIRMS sample."""

from pathlib import Path

import pandas as pd

from sardinia_fire.preprocessing.firms import (
    normalize_firms,
    save_firms_parquet,
)

INPUT = Path("data/raw/firms/viirs_noaa20_20240720_5d.csv")
OUTPUT = "data/processed/firms/viirs_noaa20_20240720_5d.parquet"


def main():
    df = pd.read_csv(INPUT)

    gdf = normalize_firms(df, source="VIIRS_NOAA20_SP")
    save_firms_parquet(gdf, OUTPUT)

    print("\nNORMALIZED FIRMS DATASET")
    print(f"Detections: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Sensor: {gdf['sensor'].unique().tolist()}")
    print(f"Unique timestamps: {gdf['acq_datetime_utc'].nunique()}")

    print("\nUTC acquisition timestamps:")
    print(
        gdf[
            ["acq_date", "acq_time", "acq_datetime_utc", "frp_mw"]
        ].head(10).to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
