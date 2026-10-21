"""Audit FIRMS event rasters before CNN training."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sardinia_fire.models.spatial_raster import rasterize_event

DETECTIONS = Path(
    "data/processed/clustering/st_dbscan_2024_06_09.parquet"
)
LABELS = Path(
    "data/processed/evaluation/ranking_dataset_2024_06_09.parquet"
)

labels = pd.read_parquet(LABELS)

labels = labels.loc[
    labels["label_status"].eq("provisional_one_to_one")
    & labels["reference_area_ha"].notna()
    & labels["reference_area_ha"].gt(0)
].copy()

detections = gpd.read_parquet(DETECTIONS)

assert len(labels) == 56
assert labels["cluster_id"].is_unique

if detections.crs is None:
    raise ValueError("Detection CRS is missing")

detections = detections.to_crs("EPSG:32632")

records = []
rasters = []

for row in labels.itertuples(index=False):
    event = detections.loc[
        detections["cluster_id"] == row.cluster_id
    ].copy()

    if event.empty:
        raise ValueError(
            f"No detections for cluster {row.cluster_id}"
        )

    raster = rasterize_event(
        event,
        grid_size=32,
        window_m=8000.0,
    )

    if not np.isfinite(raster).all():
        raise ValueError(
            f"Invalid values in cluster {row.cluster_id}"
        )

    raster_count = float(raster[0].sum())

    if not np.isclose(raster_count, len(event)):
        raise ValueError(
            f"Lost detections in cluster {row.cluster_id}"
        )

    records.append({
        "cluster_id": row.cluster_id,
        "detections": len(event),
        "occupied_pixels": int(
            np.count_nonzero(raster[0])
        ),
        "max_pixel_count": float(raster[0].max()),
        "viirs_channel_sum": float(raster[1].sum()),
        "modis_channel_sum": float(raster[2].sum()),
        "area_ha": float(row.reference_area_ha),
    })

    rasters.append(raster)

audit = pd.DataFrame(records)
x = np.stack(rasters)

print("\nREAL-EVENT CNN RASTER AUDIT")
print("Events:", len(audit))
print("Tensor shape:", x.shape)
print("All finite:", bool(np.isfinite(x).all()))
print(
    "Detections conserved:",
    bool(np.allclose(
        audit["detections"],
        x[:, 0].sum(axis=(1, 2)),
    )),
)

print("\nRASTER STATISTICS")
print(
    audit[
        [
            "detections",
            "occupied_pixels",
            "max_pixel_count",
            "viirs_channel_sum",
            "modis_channel_sum",
        ]
    ].describe().round(3).to_string()
)

print("\nSPATIAL SPARSITY")
print(
    "Median occupied pixels:",
    audit["occupied_pixels"].median(),
)
print(
    "Events with <=3 occupied pixels:",
    int(audit["occupied_pixels"].le(3).sum()),
)
print(
    "Events with only 1 occupied pixel:",
    int(audit["occupied_pixels"].eq(1).sum()),
)

assert x.shape == (56, 3, 32, 32)
assert np.isfinite(x).all()

print("\nCNN raster audit passed.")
