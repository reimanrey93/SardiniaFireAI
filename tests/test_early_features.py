import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sardinia_fire.features.early_features import (
    extract_early_features,
)


def make_detections():
    return gpd.GeoDataFrame(
        {
            "cluster_id": [0, 0, 0, 1],
            "detection_id": ["a", "b", "c", "d"],
            "acq_datetime_utc": pd.to_datetime(
                [
                    "2024-07-01T10:00:00Z",
                    "2024-07-01T10:00:00Z",
                    "2024-07-01T16:00:00Z",
                    "2024-07-02T10:00:00Z",
                ],
                utc=True,
            ),
            "source_product": [
                "VIIRS_SNPP_SP",
                "VIIRS_SNPP_SP",
                "VIIRS_NOAA20_SP",
                "MODIS_SP",
            ],
            "frp_mw": [10.0, 20.0, 30.0, 5.0],
        },
        geometry=[Point(i, 0) for i in range(4)],
        crs="EPSG:4326",
    )


def test_early_feature_horizons():
    result = extract_early_features(make_detections())

    assert len(result) == 8

    zero = result.loc[
        (result.cluster_id == 0)
        & (result.horizon_h == 0)
    ].iloc[0]

    six = result.loc[
        (result.cluster_id == 0)
        & (result.horizon_h == 6)
    ].iloc[0]

    assert zero.early_detections == 2
    assert zero.early_timestamps == 1
    assert zero.early_frp_max_mw == 20
    assert pd.isna(zero.early_frp_first_last_change_mw)

    assert six.early_detections == 3
    assert six.early_timestamps == 2
    assert six.early_products == 2
    assert six.early_frp_max_mw == 30
    assert six.early_frp_first_last_change_mw == 10


def test_future_observations_excluded():
    result = extract_early_features(make_detections())

    zero = result.loc[
        (result.cluster_id == 0)
        & (result.horizon_h == 0)
    ].iloc[0]

    assert zero.early_frp_max_mw == 20


def test_single_observation_event():
    result = extract_early_features(make_detections())
    one = result.loc[result.cluster_id == 1]

    assert one.early_detections.eq(1).all()
    assert one.early_duration_h.eq(0).all()
    assert one.early_frp_first_last_change_mw.isna().all()


def test_invalid_horizon():
    with pytest.raises(ValueError, match="Invalid horizons"):
        extract_early_features(
            make_detections(), horizons=(-1, 6)
        )


def test_duplicate_detection_ids():
    df = make_detections()
    df.loc[1, "detection_id"] = "a"

    with pytest.raises(ValueError, match="Duplicate detection IDs"):
        extract_early_features(df)


def test_input_unchanged():
    df = make_detections()
    original = df.copy(deep=True)

    extract_early_features(df)

    pd.testing.assert_frame_equal(df, original)
