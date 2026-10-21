"""Lifecycle tracking for preliminary FIRMS alerts."""

import pandas as pd

from sardinia_fire.ranking.preliminary_alerts import (
    select_preliminary_alerts,
)


class PreliminaryAlertTracker:
    """Track preliminary and confirmed detections across snapshots."""

    def __init__(
        self,
        policy="global",
        global_threshold_mw=3.0,
        product_thresholds=None,
    ):
        self.policy = policy
        self.global_threshold_mw = global_threshold_mw
        self.product_thresholds = product_thresholds
        self.states = {}
        self.transitions = []

    def update(self, snapshot, timestamp_utc):
        """Update alert states using the current ST-DBSCAN snapshot."""

        timestamp = pd.Timestamp(timestamp_utc)

        if timestamp.tzinfo is None:
            raise ValueError("Update timestamp must be timezone-aware")

        timestamp = timestamp.tz_convert("UTC")

        required = {
            "detection_id",
            "source_product",
            "frp_mw",
            "acq_datetime_utc",
            "operational_event_id",
        }

        if not required.issubset(snapshot.columns):
            raise ValueError("Missing snapshot columns")

        if snapshot["detection_id"].duplicated().any():
            raise ValueError("Duplicate detection IDs")

        acquisition = pd.to_datetime(
            snapshot["acq_datetime_utc"], utc=True
        )

        if acquisition.isna().any() or acquisition.gt(timestamp).any():
            raise ValueError("Snapshot contains future observations")

        if any(
            previous["updated_utc"] > timestamp
            for previous in self.states.values()
        ):
            raise ValueError("Out-of-order alert update")

        selected = select_preliminary_alerts(
            snapshot,
            policy=self.policy,
            global_threshold_mw=self.global_threshold_mw,
            product_thresholds=self.product_thresholds,
        )

        selected = selected.copy()
        selected["_acquisition"] = acquisition.to_numpy()

        for row in selected.itertuples():
            detection_id = row.detection_id
            event_id = row.operational_event_id

            confirmed = pd.notna(event_id)
            alert_selected = bool(row.preliminary_alert)

            if confirmed:
                new_state = "confirmed"
            elif alert_selected:
                new_state = "preliminary"
            else:
                new_state = "unselected"

            old = self.states.get(detection_id)

            if old is not None and old["state"] == new_state:
                if confirmed:
                    old["operational_event_id"] = int(event_id)
                old["updated_utc"] = timestamp
                continue

            if old is not None and old["state"] == "confirmed":
                raise ValueError(
                    "Confirmed detection returned to an unconfirmed state"
                )

            if old is None:
                transition = (
                    "alert_created"
                    if new_state == "preliminary"
                    else "confirmed_directly"
                    if new_state == "confirmed"
                    else None
                )
            elif old["state"] == "preliminary" and confirmed:
                transition = "alert_confirmed"
            elif old["state"] == "unselected" and confirmed:
                transition = "confirmed_directly"
            else:
                transition = None

            if transition is not None:
                self.transitions.append({
                    "timestamp_utc": timestamp,
                    "detection_id": detection_id,
                    "transition": transition,
                    "operational_event_id": (
                        int(event_id) if confirmed else pd.NA
                    ),
                })

            self.states[detection_id] = {
                "detection_id": detection_id,
                "state": new_state,
                "operational_event_id": (
                    int(event_id) if confirmed else pd.NA
                ),
                "updated_utc": timestamp,
            }

        return self.get_states()

    def get_states(self):
        return pd.DataFrame(
            self.states.values(),
            columns=[
                "detection_id",
                "state",
                "operational_event_id",
                "updated_utc",
            ],
        )

    def get_transitions(self):
        return pd.DataFrame(
            self.transitions,
            columns=[
                "timestamp_utc",
                "detection_id",
                "transition",
                "operational_event_id",
            ],
        )
