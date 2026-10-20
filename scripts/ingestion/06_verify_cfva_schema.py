"""Inspect CFVA sorting, identifiers and date formats."""

from collections import Counter

import requests

URL = "https://webgis.regione.sardegna.it/geoserver/ows"

CONFIG = {
    2019: (["id_ince"], "dt_ince"),
    2020: (["id", "id_ince"], "data"),
    2021: (["id", "id_ince"], "data"),
    2022: (["objectid", "id_ince"], "dt_ince"),
    2023: (["objectid", "id_ince"], "data"),
    2024: (["id", "cod_incend"], "data"),
    2025: (["cod_incend", "id"], "data"),
}


def fetch(session, year, sort_field, start=0):
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": f"dbu:areeincendiateperim{year}",
        "outputFormat": "application/json",
        "srsName": "EPSG:7791",
        "count": 500,
        "startIndex": start,
        "sortBy": sort_field,
    }

    response = session.get(URL, params=params, timeout=120)

    if not response.ok:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    return payload if payload.get("type") == "FeatureCollection" else None


def main():
    with requests.Session() as session:
        for year, (sort_fields, date_field) in CONFIG.items():
            print(f"\n{'=' * 60}")
            print(f"CFVA {year}")
            print("=" * 60)

            for field in sort_fields:
                first = fetch(session, year, field)

                if first is None:
                    print(f"Sorting {field}: FAILED")
                    continue

                features = first.get("features", [])
                matched = first.get("numberMatched")

                print(f"Sorting {field}: OK")
                print(f"Matched: {matched}")

                values = [
                    feature.get("properties", {}).get(date_field)
                    for feature in features
                ]

                print(f"Date field: {date_field}")
                print(f"Date samples: {list(dict.fromkeys(map(str, values)))[:8]}")

                if len(features) == 500:
                    second = fetch(session, year, field, start=500)

                    if second is None:
                        print("Second page: FAILED")
                        continue

                    first_ids = [f.get("id") for f in features]
                    second_ids = [f.get("id") for f in second["features"]]

                    overlap = len(set(first_ids) & set(second_ids))
                    print(f"Repeated WFS feature IDs across pages: {overlap}")

                properties = [f.get("properties", {}) for f in features]
                identifiers = [p.get(field) for p in properties]
                counts = Counter(map(str, identifiers))
                duplicates = sum(count - 1 for count in counts.values() if count > 1)

                print(f"Duplicate sorting values in first page: {duplicates}")
                print("-" * 40)


if __name__ == "__main__":
    main()
