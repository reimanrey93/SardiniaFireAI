
"""Download and inspect CFVA wildfire perimeters."""

from sardinia_fire.ingestion.cfva import download_cfva, save_cfva


def main():
    year = 2024

    print(f"Downloading CFVA perimeters for {year}...")

    gdf = download_cfva(year)
    output = save_cfva(gdf, year)

    print("\nDownload completed")
    print(f"Features: {len(gdf)}")
    print(f"Unique fire codes: {gdf['cod_incend'].nunique()}")
    print(f"CRS: {gdf.crs}")
    print(f"Date range: {gdf['fire_date'].min()} to {gdf['fire_date'].max()}")
    print(f"Output: {output}")

    print("\nGeometry types:")
    print(gdf.geometry.geom_type.value_counts())

    print("\nMost frequent fire codes:")
    print(gdf["cod_incend"].value_counts().head(10))


if __name__ == "__main__":
    main()
