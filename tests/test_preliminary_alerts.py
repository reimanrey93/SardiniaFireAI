import pandas as pd
import pytest

from sardinia_fire.ranking.preliminary_alerts import (
    select_preliminary_alerts,
)


@pytest.fixture
def detections():
    return pd.DataFrame({
        "detection_id": ["a", "b", "c", "d"],
        "source_product": [
            "VIIRS_SNPP_SP",
            "VIIRS_SNPP_SP",
            "MODIS_SP",
            "MODIS_SP",
        ],
        "frp_mw": [2.0, 4.0, 8.0, 20.0],
    })


def test_all_policy(detections):
    result = select_preliminary_alerts(
        detections, policy="all"
    )
    assert result["preliminary_alert"].tolist() == [
        True, True, True, True
    ]


def test_global_policy(detections):
    result = select_preliminary_alerts(
        detections,
        policy="global",
        global_threshold_mw=3.0,
    )
    assert result["preliminary_alert"].tolist() == [
        False, True, True, True
    ]


def test_product_policy(detections):
    result = select_preliminary_alerts(
        detections,
        policy="product",
        product_thresholds={
            "VIIRS_SNPP_SP": 3.0,
            "MODIS_SP": 10.0,
        },
    )
    assert result["preliminary_alert"].tolist() == [
        False, True, False, True
    ]


def test_missing_product_threshold(detections):
    with pytest.raises(ValueError, match="Missing threshold"):
        select_preliminary_alerts(
            detections,
            policy="product",
            product_thresholds={"VIIRS_SNPP_SP": 3.0},
        )


def test_invalid_policy(detections):
    with pytest.raises(ValueError, match="Unknown alert policy"):
        select_preliminary_alerts(
            detections, policy="invalid"
        )


def test_input_unchanged(detections):
    original = detections.copy(deep=True)

    select_preliminary_alerts(detections, policy="all")

    pd.testing.assert_frame_equal(
        detections, original
    )
