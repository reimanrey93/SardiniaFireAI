
"""Discover official Sardinia wildfire layers via OGC WFS."""

from xml.etree import ElementTree as ET

import requests

WFS_ENDPOINTS = [
    "https://webgis.regione.sardegna.it/geoserver/ows",
    "https://webgis2.regione.sardegna.it/geoserver/dbu/ows",
]


def discover_cfva_layers() -> list[dict[str, str]]:
    layers = {}

    for endpoint in WFS_ENDPOINTS:
        params = {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetCapabilities",
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=60,
            )
            response.raise_for_status()

            root = ET.fromstring(response.content)

            for feature in root.iter():
                if not feature.tag.endswith("FeatureType"):
                    continue

                name = ""
                title = ""

                for child in feature:
                    if child.tag.endswith("Name"):
                        name = (child.text or "").strip()
                    elif child.tag.endswith("Title"):
                        title = (child.text or "").strip()

                search_text = f"{name} {title}".lower()

                if "incend" in search_text or "cfva" in search_text:
                    layers[(endpoint, name)] = {
                        "name": name,
                        "title": title,
                        "endpoint": endpoint,
                    }

            print(f"Connected: {endpoint}")

        except (requests.RequestException, ET.ParseError) as error:
            print(f"Could not query {endpoint}: {error}")

    return list(layers.values())


if __name__ == "__main__":
    results = discover_cfva_layers()

    print(f"\nFound {len(results)} candidate layers\n")

    for layer in results:
        print(f"Layer: {layer['name']}")
        print(f"Title: {layer['title']}")
        print(f"Endpoint: {layer['endpoint']}")
        print("-" * 60)
