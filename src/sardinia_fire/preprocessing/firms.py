"""Normalize NASA FIRMS active-fire detections."""

import geopandas as gpd
import pandas as pd


def normalize_firms(df: pd.DataFrame, source: str) -> gpd.GeoDataFrame:
    """Normalize FIRMS observations while retaining original attributes."""
    required = {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence",
        "frp",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing FIRMS columns: {sorted(missing)}")

    result = df.copy()

    result["latitude"] = pd.to_numeric(result["latitude"], errors="raise")
    result["longitude"] = pd.to_numeric(result["longitude"], errors="raise")

    if result[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing FIRMS coordinates")

    if not result["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values")

    if not result["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values")

    times = result["acq_time"].astype("string").str.strip().str.zfill(4)

    valid_times = times.str.fullmatch(r"[0-2][0-9][0-5][0-9]", na=False)
    valid_times &= times.str[:2].astype(int).lt(24)

    if not valid_times.all():
        raise ValueError("Invalid FIRMS acquisition times")

    timestamps = (
        result["acq_date"].astype("string").str.strip()
        + " "
        + times
    )

    result["acq_datetime_utc"] = pd.to_datetime(
        timestamps,
        format="%Y-%m-%d %H%M",
        errors="raise",
        utc=True,
    )

    result["source_product"] = source
    result["frp_mw"] = pd.to_numeric(result["frp"], errors="coerce")
    result["confidence_raw"] = result["confidence"].astype("string")

    result["sensor"] = (
        "VIIRS" if source.startswith("VIIRS") else "MODIS"
    )

    from hashlib import sha256

    fingerprint_columns = [
        "satellite",
        "acq_datetime_utc",
        "latitude",
        "longitude",
        "frp_mw",
    ]

    for optional in ("bright_ti4", "bright_t31", "scan", "track"):
        if optional in result.columns:
            fingerprint_columns.append(optional)

    def fingerprint(row):
        values = [
            str(source),
            *[
                str(value) if pd.notna(value) else "<NA>"
                for value in row
            ],
        ]
        payload = "|".join(values).encode("utf-8")
        return sha256(payload).hexdigest()

    result["detection_id"] = result[
        fingerprint_columns
    ].apply(fingerprint, axis=1)

    geometry = gpd.points_from_xy(
        result["longitude"],
        result["latitude"],
    )

    gdf = gpd.GeoDataFrame(
        result,
        geometry=geometry,
        crs="EPSG:4326",
    )

    return gdf


def save_firms_parquet(
    gdf: gpd.GeoDataFrame,
    path: str,
) -> None:
    """Save normalized FIRMS observations as GeoParquet."""
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_parquet(output, index=False)
