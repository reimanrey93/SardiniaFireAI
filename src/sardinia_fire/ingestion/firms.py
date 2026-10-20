"""NASA FIRMS data availability and ingestion utilities."""

import io
import os

import pandas as pd
import requests

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"

SOURCES = (
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "MODIS_SP",
    "VIIRS_NOAA21_SP",
)


def get_map_key() -> str:
    key = os.getenv("FIRMS_MAP_KEY", "").strip()

    if not key:
        raise RuntimeError("FIRMS_MAP_KEY is not configured")

    return key


def get_data_availability() -> pd.DataFrame:
    key = get_map_key()
    url = f"{FIRMS_BASE_URL}/data_availability/csv/{key}/ALL"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        raise RuntimeError("NASA FIRMS availability request failed") from None

    try:
        df = pd.read_csv(io.StringIO(response.text))
    except (ValueError, pd.errors.ParserError):
        raise RuntimeError("Invalid NASA FIRMS CSV response") from None

    required = {"data_id", "min_date", "max_date"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected FIRMS columns: {df.columns.tolist()}"
        )

    return df


def get_selected_availability() -> pd.DataFrame:
    df = get_data_availability()
    return df.loc[df["data_id"].isin(SOURCES)].copy()
def download_firms_area(
    source: str,
    date: str,
    days: int = 5,
    bbox: tuple[float, float, float, float] = (7.8, 38.7, 10.1, 41.5),
) -> pd.DataFrame:
    """Download FIRMS hotspots inside a bounding box."""

    if source not in SOURCES:
        raise ValueError(f"Unsupported FIRMS source: {source}")

    if not 1 <= days <= 5:
        raise ValueError("days must be between 1 and 5")

    start_date = pd.Timestamp(date)
    west, south, east, north = bbox

    if not (-180 <= west < east <= 180):
        raise ValueError("Invalid longitude bounds")

    if not (-90 <= south < north <= 90):
        raise ValueError("Invalid latitude bounds")

    coordinates = f"{west},{south},{east},{north}"

    url = (
        f"{FIRMS_BASE_URL}/area/csv/"
        f"{get_map_key()}/{source}/{coordinates}/{days}/"
        f"{start_date:%Y-%m-%d}"
    )

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException:
        raise RuntimeError("FIRMS area request failed") from None

    try:
        df = pd.read_csv(io.StringIO(response.text))
    except (ValueError, pd.errors.ParserError):
        raise RuntimeError("Invalid FIRMS CSV response") from None

    required = {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence",
        "frp",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected FIRMS columns: {df.columns.tolist()}"
        )

    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing FIRMS coordinates")

    print(f"Source: {source}")
    print(f"Period: {date} ({days} days)")
    print(f"Detections: {len(df)}")

    return df
