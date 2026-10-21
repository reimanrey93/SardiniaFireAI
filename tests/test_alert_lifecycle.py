import pandas as pd
import pytest

from sardinia_fire.ranking.alert_lifecycle import (
    PreliminaryAlertTracker,
)


def snapshot(ids, frps, events, hour=0):
    return pd.DataFrame({
        "detection_id": ids,
        "source_product": ["VIIRS_SNPP_SP"] * len(ids),
        "frp_mw": frps,
        "acq_datetime_utc": pd.to_datetime(
            ["2024-07-01T00:00:00Z"] * len(ids),
            utc=True,
        ),
        "operational_event_id": pd.array(events, dtype="Int64"),
    })


def timestamp(hour=0):
    return pd.Timestamp("2024-07-01T00:00:00Z") + pd.Timedelta(
        hours=hour
    )


def test_preliminary_alert_created():
    tracker = PreliminaryAlertTracker()

    states = tracker.update(
        snapshot(["a"], [5.0], [None]),
        timestamp(),
    )

    assert states.iloc[0]["state"] == "preliminary"
    assert (
        tracker.get_transitions().iloc[0]["transition"]
        == "alert_created"
    )


def test_preliminary_alert_becomes_confirmed():
    tracker = PreliminaryAlertTracker()

    tracker.update(
        snapshot(["a"], [5.0], [None]),
        timestamp(),
    )

    states = tracker.update(
        snapshot(["a", "b", "c"], [5.0, 2.0, 1.0], [10, 10, 10]),
        timestamp(1),
    )

    assert states["state"].eq("confirmed").all()

    transitions = tracker.get_transitions()
    assert (
        transitions["transition"] == "alert_confirmed"
    ).sum() == 1


def test_no_duplicate_alert_on_repeated_update():
    tracker = PreliminaryAlertTracker()

    data = snapshot(["a"], [5.0], [None])

    tracker.update(data, timestamp())
    tracker.update(data, timestamp(1))

    assert len(tracker.get_transitions()) == 1


def test_below_threshold_is_not_preliminary():
    tracker = PreliminaryAlertTracker()

    states = tracker.update(
        snapshot(["a"], [1.0], [None]),
        timestamp(),
    )

    assert states.iloc[0]["state"] == "unselected"
    assert tracker.get_transitions().empty


def test_future_observation_rejected():
    tracker = PreliminaryAlertTracker()

    with pytest.raises(ValueError, match="future"):
        tracker.update(
            snapshot(["a"], [5.0], [None]),
            timestamp() - pd.Timedelta(minutes=1),
        )


def test_out_of_order_alert_update():
    tracker = PreliminaryAlertTracker()

    tracker.update(
        snapshot(["a"], [5.0], [None]),
        timestamp(2),
    )

    with pytest.raises(ValueError, match="Out-of-order"):
        tracker.update(
            snapshot(["a"], [5.0], [None]),
            timestamp(1),
        )
