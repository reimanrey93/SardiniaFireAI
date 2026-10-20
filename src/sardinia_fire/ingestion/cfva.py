"""CFVA wildfire perimeter ingestion for Sardinia."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

WFS_URL = "https://webgis.regione.sardegna.it/geoserver/ows"
SOURCE_CRS = "EPSG:7791"

YEAR_CONFIG = {
    2019: ("id_ince", "dt_ince", "%Y-%m-%dZ"),
    2020: ("id_ince", "data", "%Y-%m-%dZ"),
    2021: ("id_ince", "data", "%Y/%m/%d"),
    2022: ("id_ince", "dt_ince", "%Y/%m/%d"),
    2023: ("id_ince", "data", "%d/%m/%Y"),
    2024: ("cod_incend", "data", "%d/%m/%Y"),
    2025: ("cod_incend", "data", "%d/%m/%Y"),
}


def polygon_only(geom):
    """Keep polygonal components of repaired geometries."""
    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        return geom

    if isinstance(geom, GeometryCollection):
        polygons = []
        for part in geom.geoms:
            result = polygon_only(part)
            if isinstance(result, Polygon):
                polygons.append(result)
            elif isinstance(result, MultiPolygon):
                polygons.extend(result.geoms)

        if polygons:
            return MultiPolygon(polygons)

    return GeometryCollection()


def download_cfva(year: int, page_size: int = 500) -> gpd.GeoDataFrame:
    """Download, validate and normalize annual CFVA perimeters."""
    if year not in YEAR_CONFIG:
        raise ValueError(f"Unsupported year: {year}")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    sort_field, date_field, date_format = YEAR_CONFIG[year]
    layer = f"dbu:areeincendiateperim{year}"

    features = []
    matched = None

    with requests.Session() as session:
        start = 0

        while True:
            params = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": layer,
                "outputFormat": "application/json",
                "srsName": SOURCE_CRS,
                "count": page_size,
                "startIndex": start,
                "sortBy": sort_field,
            }

            response = session.get(WFS_URL, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()

            if payload.get("type") != "FeatureCollection":
                raise ValueError("Unexpected WFS response")

            crs_info = payload.get("crs", {})
            crs_name = crs_info.get("properties", {}).get("name")
            if crs_name is None:
                raise ValueError("WFS response does not declare CRS")
            if not crs_name.upper().endswith(("::7791", ":7791")):
                raise ValueError(f"Unexpected source CRS: {crs_name}")

            current_matched = payload.get("numberMatched")
            if not isinstance(current_matched, int):
                raise ValueError("Missing numeric numberMatched")

            if matched is None:
                matched = current_matched
            elif matched != current_matched:
                raise ValueError("WFS total changed during pagination")

            batch = payload.get("features", [])
            if not batch:
                break

            features.extend(batch)
            start += len(batch)

            print(f"{year}: {start}/{matched} downloaded")

            if start >= matched:
                break

    if len(features) != matched:
        raise ValueError(
            f"Incomplete download: {len(features)} of {matched}"
        )

    wfs_ids = [feature.get("id") for feature in features]
    if any(value is None for value in wfs_ids):
        raise ValueError("Missing WFS feature IDs")
    if len(set(wfs_ids)) != len(wfs_ids):
        raise ValueError("Duplicated WFS feature IDs")

    gdf = gpd.GeoDataFrame.from_features(features).set_crs(SOURCE_CRS)

    if sort_field not in gdf or date_field not in gdf:
        raise ValueError("Missing expected source columns")

    source_ids = gdf[sort_field].astype("string")
    if source_ids.isna().any() or source_ids.duplicated().any():
        raise ValueError(f"Missing or duplicate {sort_field} values")

    gdf["source_fire_id"] = source_ids
    gdf["fire_id"] = str(year) + ":" + source_ids
    gdf["wfs_feature_id"] = wfs_ids

    raw_dates = gdf[date_field].astype("string").str.strip()
    gdf["fire_date_raw"] = raw_dates.copy()

    normalized_dates = raw_dates.copy()
    reasons = pd.Series("none", index=gdf.index, dtype="string")

    if year == 2021:
        alternate = normalized_dates.str.fullmatch(
            r"\d{4}-\d{2}-\d{2}", na=False
        )
        normalized_dates.loc[alternate] = (
            normalized_dates.loc[alternate].str.replace("-", "/", regex=False)
        )
        reasons.loc[alternate] = "alternate_iso_date_format"

    if year == 2025:
        typo = normalized_dates.eq("01/102025").fillna(False)
        normalized_dates.loc[typo] = "01/10/2025"
        reasons.loc[typo] = "manual_source_date_correction"

    gdf["date_corrected"] = reasons.ne("none")
    gdf["date_correction_reason"] = reasons

    gdf["fire_date"] = pd.to_datetime(
        normalized_dates,
        format=date_format,
        errors="coerce",
    )

    invalid_mask = gdf["fire_date"].isna()
    invalid_dates = int(invalid_mask.sum())

    if invalid_dates:
        print(f"Unverified source dates: {invalid_dates}")
        print(
            gdf.loc[
                invalid_mask,
                ["source_fire_id", "fire_date_raw"],
            ].to_string(index=False)
        )
        gdf.loc[invalid_mask, "date_correction_reason"] = (
            "unparseable_source_date"
        )

    wrong_year = (
        gdf["fire_date"].notna()
        & gdf["fire_date"].dt.year.ne(year)
    )

    if wrong_year.any():
        print("Dates outside the expected year:")
        print(
            gdf.loc[
                wrong_year,
                ["source_fire_id", "fire_date_raw", "fire_date"],
            ].to_string(index=False)
        )
        gdf.loc[wrong_year, "date_correction_reason"] = (
            "year_mismatch"
        )
        gdf.loc[wrong_year, "fire_date"] = pd.NaT

    gdf["date_quality"] = "valid"
    gdf.loc[
        gdf["date_correction_reason"].eq("alternate_iso_date_format"),
        "date_quality",
    ] = "format_normalized"
    gdf.loc[
        gdf["date_correction_reason"].eq("manual_source_date_correction"),
        "date_quality",
    ] = "unverified"
    gdf.loc[
        gdf["date_correction_reason"].eq("unparseable_source_date"),
        "date_quality",
    ] = "unverified"
    gdf.loc[
        gdf["date_correction_reason"].eq("year_mismatch"),
        "date_quality",
    ] = "year_mismatch"

    gdf["temporal_reference_valid"] = gdf["date_quality"].isin(
        ["valid", "format_normalized"]
    )

    print("Date quality:")
    print(gdf["date_quality"].value_counts().to_string())

    print(f"Normalized or corrected dates: {int(gdf['date_corrected'].sum())}")

    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Missing or empty source geometries")

    gdf["geometry_original"] = gdf.geometry.to_wkb(hex=True)

    invalid = ~gdf.geometry.is_valid
    n_invalid = int(invalid.sum())

    if n_invalid:
        repaired = gdf.loc[invalid, "geometry"].make_valid()
        repaired = repaired.apply(polygon_only)
        gdf.loc[invalid, "geometry"] = repaired

    gdf["geometry_repaired"] = invalid.to_numpy()

    valid = gdf.geometry.is_valid.all()
    polygonal = gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    ).all()
    nonempty = (~gdf.geometry.is_empty).all()

    if not (valid and polygonal and nonempty):
        raise ValueError("Geometry validation failed after repair")

    gdf["year"] = year
    gdf["source"] = "CFVA_Sardegna"

    print(f"Unique source IDs: {source_ids.nunique()}")
    print(f"Repaired geometries: {n_invalid}")

    return gdf


def save_cfva(
    gdf: gpd.GeoDataFrame,
    year: int,
    directory: str | Path = "data/raw/cfva",
) -> Path:
    """Save validated annual GeoParquet."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"cfva_perimeters_{year}.parquet"
    temporary = output.with_suffix(".parquet.tmp")

    try:
        gdf.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    return output
