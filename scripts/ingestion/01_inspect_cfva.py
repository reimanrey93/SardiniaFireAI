
"""Inspect a sample of official CFVA wildfire perimeters."""

import geopandas as gpd
import requests

WFS_URL = "https://webgis.regione.sardegna.it/geoserver/ows"

LAYER = "dbu:areeincendiateperim2024"

params = {
    "service": "WFS",
    "version": "2.0.0",
    "request": "GetFeature",
    "typeNames": LAYER,
    "outputFormat": "application/json",
    "count": 10,
    "startIndex": 0,
}

print("Requesting CFVA data...")

response = requests.get(
    WFS_URL,
    params=params,
    timeout=120,
)
response.raise_for_status()

payload = response.json()

if payload.get("type") != "FeatureCollection":
    raise ValueError("Unexpected WFS response format")

features = payload.get("features", [])

print(f"Downloaded features: {len(features)}")
print(f"Matched features: {payload.get('numberMatched', 'unknown')}")

if not features:
    raise RuntimeError("No features returned by WFS")

gdf = gpd.GeoDataFrame.from_features(features)

if gdf.crs is None:
    crs = payload.get("crs")
    if crs:
        print(f"CRS metadata: {crs}")
    print("Warning: confirm GeoJSON coordinate reference system.")

print("\nColumns:")
print(gdf.columns.tolist())

print("\nFirst records:")
print(gdf.drop(columns="geometry").head().to_string(index=False))

print("\nGeometry types:")
print(gdf.geometry.geom_type.value_counts())

print("\nCRS:")
print(gdf.crs)

print("\nMissing values:")
print(gdf.isna().sum().to_string())
