"""Diagnose annual CFVA WFS differences without modifying datasets."""

import pandas as pd
import requests

URL = "https://webgis.regione.sardegna.it/geoserver/ows"
YEARS = [2019, 2020, 2021, 2022, 2023, 2025]


def fetch_page(session, year, start=0, sort=True):
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": f"dbu:areeincendiateperim{year}",
        "outputFormat": "application/json",
        "srsName": "EPSG:7791",
        "count": 500,
        "startIndex": start,
    }

    if sort:
        params["sortBy"] = "id"

    response = session.get(URL, params=params, timeout=120)

    if not response.ok:
        print(f"HTTP {response.status_code}")
        print(response.text[:1200])
        return None

    try:
        return response.json()
    except ValueError:
        print("Unexpected response:")
        print(response.text[:1200])
        return None


def main():
    with requests.Session() as session:
        for year in YEARS:
            print(f"\n{'=' * 65}")
            print(f"CFVA {year}")
            print("=" * 65)

            payload = fetch_page(session, year, sort=True)

            if payload is None:
                print("\nRetrying without sortBy...")
                payload = fetch_page(session, year, sort=False)

            if payload is None:
                print("Both requests failed")
                continue

            features = payload.get("features", [])
            print(f"First page: {len(features)} records")
            print(f"numberMatched: {payload.get('numberMatched')}")
            print(f"numberReturned: {payload.get('numberReturned')}")

            if not features:
                continue

            rows = [f.get("properties", {}) for f in features]
            df = pd.DataFrame(rows)

            print("\nColumns:")
            print(df.columns.tolist())

            if "data" in df:
                dates = pd.to_datetime(
                    df["data"],
                    format="%d/%m/%Y",
                    errors="coerce",
                )
                invalid = df.loc[dates.isna(), "data"]

                print(f"\nInvalid dates in first page: {len(invalid)}")
                print(invalid.value_counts(dropna=False).head(10))

            if "id" in df:
                print(f"\nDuplicate IDs within first page: {df['id'].duplicated().sum()}")

            if "cod_incend" in df:
                print(
                    "Duplicate fire codes within first page:",
                    df["cod_incend"].duplicated().sum(),
                )

            print("\nChecking next page...")

            if len(features) == 500:
                next_page = fetch_page(
                    session,
                    year,
                    start=500,
                    sort=True,
                )

                if next_page is not None:
                    second = pd.DataFrame(
                        [f.get("properties", {}) for f in next_page["features"]]
                    )
                    combined = pd.concat([df, second], ignore_index=True)

                    for field in ("id", "cod_incend"):
                        if field in combined:
                            print(
                                f"Duplicate {field} across pages:",
                                combined[field].duplicated().sum(),
                            )
                            duplicates = combined.loc[
                                combined[field].duplicated(keep=False),
                                [c for c in ["id", "cod_incend"] if c in combined],
                            ]
                            print(duplicates.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
