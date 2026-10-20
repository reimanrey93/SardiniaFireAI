"""Download and normalize monthly FIRMS observations."""

from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sardinia_fire.ingestion.firms import download_firms_area
from sardinia_fire.preprocessing.firms import normalize_firms

SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA21_SP"]
START = date(2024, 7, 1)
END = date(2024, 7, 31)

RAW_DIR = Path("data/raw/firms/2024_07")
OUT_DIR = Path("data/processed/firms")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING {source}")
        print("=" * 60)

        batches = []
        current = START

        while current <= END:
            days = min(5, (END - current).days + 1)

            filename = f"{source}_{current:%Y%m%d}_{days}d.csv"
            path = RAW_DIR / filename

            if path.exists():
                print(f"Reading cached: {filename}")
                df = pd.read_csv(path)
            else:
                df = download_firms_area(
                    source=source,
                    date=current.isoformat(),
                    days=days,
                )
                df.to_csv(path, index=False)

            if not df.empty:
                normalized = normalize_firms(df, source)

                window_end = current + timedelta(days=days)
                timestamps = normalized["acq_datetime_utc"]

                if not (
                    timestamps.ge(pd.Timestamp(current, tz="UTC"))
                    & timestamps.lt(pd.Timestamp(window_end, tz="UTC"))
                ).all():
                    raise ValueError(
                        f"Out-of-window detections: {filename}"
                    )

                batches.append(normalized)

            current += timedelta(days=days)

        if not batches:
            print(f"No detections found for {source}")
            continue

        combined = gpd.GeoDataFrame(
            pd.concat(batches, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )

        duplicates = int(combined["detection_id"].duplicated().sum())

        print(f"\nRecords before deduplication: {len(combined)}")
        print(f"Duplicate IDs: {duplicates}")

        combined = combined.drop_duplicates(
            subset="detection_id",
            keep="first",
        ).copy()

        if not combined.geometry.is_valid.all():
            raise ValueError("Invalid FIRMS geometries")

        output = OUT_DIR / f"{source}_2024_07.parquet"
        combined.to_parquet(output, index=False)

        print(f"Unique detections: {len(combined)}")
        print(f"First: {combined['acq_datetime_utc'].min()}")
        print(f"Last: {combined['acq_datetime_utc'].max()}")
        print(f"Saved: {output}")

    print("\nMonthly FIRMS download completed.")


if __name__ == "__main__":
    main()
