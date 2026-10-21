"""Resumable monthly NASA FIRMS ingestion."""

from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sardinia_fire.ingestion.firms import SOURCES, download_firms_area
from sardinia_fire.ingestion.history import plan_download_windows
from sardinia_fire.preprocessing.firms import normalize_firms


def ingest_firms_month(
    year: int,
    month: int,
    sources=None,
    raw_root="data/raw/firms",
    processed_root="data/processed/firms",
    overwrite=False,
):
    """Download and normalize FIRMS observations with a block cache."""

    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    period = f"{year}_{month:02d}"

    selected = list(SOURCES if sources is None else sources)

    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Sources must be nonempty and unique")

    if any(source not in SOURCES for source in selected):
        raise ValueError("Unsupported FIRMS source")

    raw_dir = Path(raw_root) / period
    processed_dir = Path(processed_root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    windows = plan_download_windows(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    summary = []

    for source in selected:
        output = processed_dir / f"{source}_{period}.parquet"

        if output.exists() and not overwrite:
            print(f"Existing: {output}")
            summary.append((source, "existing", -1))
            continue

        print(f"\nPROCESSING {source} | {period}")
        batches = []

        for date_string, days in windows:
            date = pd.Timestamp(date_string)
            filename = f"{source}_{date:%Y%m%d}_{days}d.csv"
            path = raw_dir / filename

            if path.exists():
                raw = pd.read_csv(path)
                print(f"Cached: {filename} ({len(raw)})")
            else:
                raw = download_firms_area(
                    source=source,
                    date=date_string,
                    days=days,
                )

                temporary = path.with_suffix(".csv.tmp")
                raw.to_csv(temporary, index=False)
                temporary.replace(path)
                print(f"Downloaded: {filename} ({len(raw)})")

            if raw.empty:
                continue

            normalized = normalize_firms(raw, source)

            window_start = pd.Timestamp(date_string, tz="UTC")
            window_end = window_start + timedelta(days=days)

            timestamps = normalized["acq_datetime_utc"]

            if not (
                timestamps.ge(window_start)
                & timestamps.lt(window_end)
            ).all():
                raise ValueError(
                    f"Out-of-window observations: {filename}"
                )

            batches.append(normalized)

        if not batches:
            print(f"No detections for {source} in {period}")
            summary.append((source, "empty", 0))
            continue

        combined = gpd.GeoDataFrame(
            pd.concat(batches, ignore_index=True),
            geometry="geometry",
            crs=batches[0].crs,
        )

        before = len(combined)
        combined = combined.drop_duplicates(
            subset="detection_id",
            keep="first",
        ).copy()

        if combined["detection_id"].duplicated().any():
            raise ValueError("Duplicate detection IDs remain")

        if not combined.geometry.is_valid.all():
            raise ValueError("Invalid FIRMS geometries")

        timestamps = combined["acq_datetime_utc"]

        if not (
            timestamps.dt.year.eq(year)
            & timestamps.dt.month.eq(month)
        ).all():
            raise ValueError("Observations outside requested month")

        temporary = output.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary, index=False)
        temporary.replace(output)

        print(f"Records: {before} -> {len(combined)} unique")
        print(f"Saved: {output}")

        summary.append((source, "saved", len(combined)))

    return pd.DataFrame(
        summary,
        columns=["source", "status", "detections"],
    )
