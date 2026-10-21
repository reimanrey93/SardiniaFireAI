"""Date-window utilities for historical FIRMS ingestion."""

from datetime import timedelta

import pandas as pd


def plan_download_windows(
    start_date: str,
    end_date: str,
    max_days: int = 5,
) -> list[tuple[str, int]]:
    """Plan non-overlapping, inclusive FIRMS download windows."""

    if isinstance(max_days, bool) or not isinstance(max_days, int):
        raise ValueError("max_days must be an integer")

    if not 1 <= max_days <= 5:
        raise ValueError("max_days must be between 1 and 5")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if (
        pd.isna(start)
        or pd.isna(end)
        or start.tzinfo is not None
        or end.tzinfo is not None
        or start != start.normalize()
        or end != end.normalize()
    ):
        raise ValueError("Dates must be calendar dates without timezones")

    if start > end:
        raise ValueError("start_date must not exceed end_date")

    windows = []
    current = start

    while current <= end:
        remaining = (end - current).days + 1
        days = min(max_days, remaining)

        windows.append((current.strftime("%Y-%m-%d"), days))
        current += timedelta(days=days)

    return windows
