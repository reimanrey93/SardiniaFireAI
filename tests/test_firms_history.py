"""Tests for historical FIRMS download-window planning."""

from datetime import timedelta

import pandas as pd
import pytest

from sardinia_fire.ingestion.history import plan_download_windows


def test_five_day_windows():
    result = plan_download_windows("2024-06-01", "2024-06-12")

    assert result == [
        ("2024-06-01", 5),
        ("2024-06-06", 5),
        ("2024-06-11", 2),
    ]


def test_single_day():
    assert plan_download_windows(
        "2024-06-01", "2024-06-01"
    ) == [("2024-06-01", 1)]


def test_month_boundary():
    assert plan_download_windows(
        "2024-06-29", "2024-07-03"
    ) == [("2024-06-29", 5)]


def test_no_missing_or_repeated_dates():
    windows = plan_download_windows(
        "2024-06-01", "2024-09-30"
    )

    actual = []

    for start, days in windows:
        date = pd.Timestamp(start)
        actual.extend(
            date + timedelta(days=i)
            for i in range(days)
        )

    expected = list(pd.date_range(
        "2024-06-01", "2024-09-30", freq="D"
    ))

    assert actual == expected
    assert len(windows) == 25


@pytest.mark.parametrize(
    "start,end",
    [
        ("2024-07-02", "2024-07-01"),
        ("2024-06-01T12:00:00", "2024-06-05"),
    ],
)
def test_invalid_dates(start, end):
    with pytest.raises(ValueError):
        plan_download_windows(start, end)


@pytest.mark.parametrize("days", [0, 6, 2.5, True])
def test_invalid_window_size(days):
    with pytest.raises(ValueError):
        plan_download_windows(
            "2024-06-01", "2024-06-10", days
        )
