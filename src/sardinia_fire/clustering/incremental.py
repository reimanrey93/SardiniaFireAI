"""Chronological ST-DBSCAN snapshots with persistent event identifiers."""

import geopandas as gpd
import pandas as pd

from sardinia_fire.clustering.st_dbscan_sparse import st_dbscan_sparse


class IncrementalSTDBSCAN:
    """Recompute clustering on arrived detections, tracking event identity."""

    def __init__(
        self,
        eps_space_m: float = 1500,
        eps_time_h: float = 24,
        min_samples: int = 3,
    ):
        if eps_space_m <= 0 or eps_time_h <= 0 or min_samples < 1:
            raise ValueError("Invalid ST-DBSCAN parameters")

        self.params = {
            "eps_space_m": eps_space_m,
            "eps_time_h": eps_time_h,
            "min_samples": min_samples,
        }

        self.observed = None
        self.snapshot = None
        self.next_event_id = 0
        self.history = []

    def update(self, batch: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process a chronological batch without accessing future data."""

        if batch.empty:
            raise ValueError("Empty update batch")

        if "detection_id" not in batch.columns:
            raise ValueError("Missing detection_id")

        if batch["detection_id"].isna().any():
            raise ValueError("Missing detection identifiers")

        if batch["detection_id"].duplicated().any():
            raise ValueError("Duplicate detection identifiers")

        incoming = batch.copy()

        if "acq_datetime_utc" not in incoming.columns:
            raise ValueError("Missing acquisition timestamps")

        incoming["acq_datetime_utc"] = pd.to_datetime(
            incoming["acq_datetime_utc"],
            utc=True,
            errors="raise",
        )

        if incoming["acq_datetime_utc"].isna().any():
            raise ValueError("Missing acquisition timestamps")

        if self.observed is not None:
            previous_time = self.observed["acq_datetime_utc"].max()

            if incoming["acq_datetime_utc"].min() < previous_time:
                raise ValueError("Out-of-order detection batch")

            if incoming.crs != self.observed.crs:
                incoming = incoming.to_crs(self.observed.crs)

            if incoming["detection_id"].isin(
                self.observed["detection_id"]
            ).any():
                raise ValueError("Previously observed detection ID")

            combined = gpd.GeoDataFrame(
                pd.concat(
                    [self.observed, incoming],
                    ignore_index=True,
                ),
                geometry="geometry",
                crs=self.observed.crs,
            )
        else:
            combined = incoming

        combined = combined.sort_values(
            ["acq_datetime_utc", "detection_id"],
        ).reset_index(drop=True)

        clustered = st_dbscan_sparse(combined, **self.params)

        previous = {}

        if self.snapshot is not None:
            old = self.snapshot.loc[
                self.snapshot["operational_event_id"].notna()
            ]

            previous = dict(
                zip(
                    old["detection_id"],
                    old["operational_event_id"].astype(int),
                    strict=True,
                )
            )

        members = {}

        for cluster_id, group in clustered.loc[
            clustered["cluster_id"] >= 0
        ].groupby("cluster_id"):
            members[int(cluster_id)] = set(group["detection_id"])

        overlaps = {}

        for cluster_id, ids in members.items():
            counts = {}

            for detection_id in ids:
                old_id = previous.get(detection_id)

                if old_id is not None:
                    counts[old_id] = counts.get(old_id, 0) + 1

            overlaps[cluster_id] = counts

        candidates = sorted(
            (
                (-count, old_id, cluster_id)
                for cluster_id, counts in overlaps.items()
                for old_id, count in counts.items()
            )
        )

        assignments = {}
        used_old = set()

        for _, old_id, cluster_id in candidates:
            if cluster_id not in assignments and old_id not in used_old:
                assignments[cluster_id] = old_id
                used_old.add(old_id)

        timestamp = incoming["acq_datetime_utc"].max()

        for cluster_id in sorted(members):
            if cluster_id not in assignments:
                event_id = self.next_event_id
                self.next_event_id += 1
                assignments[cluster_id] = event_id

                self.history.append({
                    "timestamp_utc": timestamp,
                    "event_type": "created",
                    "event_id": event_id,
                    "related_event_ids": "",
                })

            old_ids = sorted(overlaps[cluster_id])

            if len(old_ids) > 1:
                self.history.append({
                    "timestamp_utc": timestamp,
                    "event_type": "merged",
                    "event_id": assignments[cluster_id],
                    "related_event_ids": ",".join(
                        map(str, old_ids)
                    ),
                })

        for old_id in sorted(set(previous.values())):
            descendants = sorted(
                cid for cid, counts in overlaps.items()
                if old_id in counts
            )

            if len(descendants) > 1:
                self.history.append({
                    "timestamp_utc": timestamp,
                    "event_type": "split",
                    "event_id": old_id,
                    "related_event_ids": ",".join(
                        str(assignments[cid]) for cid in descendants
                    ),
                })

        clustered["operational_event_id"] = (
            clustered["cluster_id"]
            .map(assignments)
            .astype("Int64")
        )

        self.observed = combined.copy()
        self.snapshot = clustered.copy()

        return clustered.copy()

    def get_history(self) -> pd.DataFrame:
        """Return the chronological event-transition log."""
        return pd.DataFrame(
            self.history,
            columns=[
                "timestamp_utc",
                "event_type",
                "event_id",
                "related_event_ids",
            ],
        )
