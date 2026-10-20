"""Tests for CFVA geometry handling and reference integrity."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon

from sardinia_fire.ingestion.cfva import YEAR_CONFIG, polygon_only


def test_all_years_configured():
    assert set(YEAR_CONFIG) == set(range(2019, 2026))


def test_polygon_only_preserves_polygon():
    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])
    result = polygon_only(polygon)

    assert result.equals(polygon)
    assert result.is_valid


def test_polygon_only_extracts_polygon_from_collection():
    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])
    line = LineString([(3, 3), (4, 4)])

    collection = GeometryCollection([polygon, line])
    result = polygon_only(collection)

    assert isinstance(result, MultiPolygon)
    assert result.is_valid
    assert abs(result.area - polygon.area) < 1e-10


def test_polygon_only_rejects_nonpolygon_components():
    line = LineString([(0, 0), (1, 1)])
    result = polygon_only(line)

    assert result.is_empty


def test_cfva_reference_integrity():
    path = "data/processed/cfva/cfva_benchmark_reference.parquet"
    if not Path(path).exists():
        pytest.skip("Local CFVA dataset not available")
    gdf = gpd.read_parquet(path)

    assert len(gdf) == 7779
    assert gdf["fire_id"].is_unique
    assert gdf.crs.to_epsg() == 7791
    assert gdf.geometry.is_valid.all()
    assert (~gdf.geometry.is_empty).all()
    assert int(gdf["temporal_reference_valid"].sum()) == 7776
    assert int(gdf["area_review_required"].sum()) == 4


def test_temporal_quality_flags():
    path = "data/processed/cfva/cfva_benchmark_reference.parquet"
    if not Path(path).exists():
        pytest.skip("Local CFVA dataset not available")
    gdf = gpd.read_parquet(path)

    invalid = gdf.loc[~gdf["temporal_reference_valid"]]

    assert len(invalid) == 3
    assert invalid["fire_id"].is_unique
    assert invalid["fire_date_raw"].notna().all()

    valid = gdf.loc[gdf["temporal_reference_valid"]]
    assert valid["fire_date"].notna().all()
    assert (
        valid["fire_date"].dt.year.astype(int)
        == valid["year"].astype(int)
    ).all()
