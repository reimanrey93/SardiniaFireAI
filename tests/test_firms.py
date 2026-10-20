"""Unit tests for FIRMS normalization."""

import pandas as pd
import pytest

from sardinia_fire.preprocessing.firms import normalize_firms


def sample_data():
    return pd.DataFrame({
        "latitude": [40.39474, 39.42023],
        "longitude": [9.00416, 8.94771],
        "acq_date": ["2024-07-20", "2024-07-20"],
        "acq_time": [145, 1128],
        "satellite": ["N20", "N20"],
        "confidence": ["n", "h"],
        "frp": [1.19, 9.98],
        "bright_ti4": [305.56, 367.0],
        "scan": [0.41, 0.42],
        "track": [0.45, 0.61],
    })


def test_utc_timestamps():
    gdf = normalize_firms(sample_data(), "VIIRS_NOAA20_SP")

    assert str(gdf.loc[0, "acq_datetime_utc"]) == (
        "2024-07-20 01:45:00+00:00"
    )
    assert str(gdf.loc[1, "acq_datetime_utc"]) == (
        "2024-07-20 11:28:00+00:00"
    )


def test_geometries():
    gdf = normalize_firms(sample_data(), "VIIRS_NOAA20_SP")

    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.geom_type.eq("Point").all()
    assert gdf.geometry.is_valid.all()


def test_ids_independent_of_row_order():
    df = sample_data()
    reversed_df = df.iloc[::-1].reset_index(drop=True)

    first = normalize_firms(df, "VIIRS_NOAA20_SP")
    second = normalize_firms(reversed_df, "VIIRS_NOAA20_SP")

    assert set(first["detection_id"]) == set(second["detection_id"])
    assert first["detection_id"].is_unique


def test_identical_detections_share_id():
    df = sample_data().iloc[[0]]
    duplicates = pd.concat([df, df], ignore_index=True)

    gdf = normalize_firms(duplicates, "VIIRS_NOAA20_SP")

    assert gdf["detection_id"].nunique() == 1


def test_invalid_latitude():
    df = sample_data()
    df.loc[0, "latitude"] = 95

    with pytest.raises(ValueError, match="latitude"):
        normalize_firms(df, "VIIRS_NOAA20_SP")


def test_missing_column():
    df = sample_data().drop(columns="frp")

    with pytest.raises(ValueError, match="Missing FIRMS columns"):
        normalize_firms(df, "VIIRS_NOAA20_SP")
