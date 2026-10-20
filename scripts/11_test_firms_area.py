"""Test historical FIRMS acquisition over Sardinia."""

from pathlib import Path

from sardinia_fire.ingestion.firms import download_firms_area

SOURCE = "VIIRS_NOAA20_SP"
START_DATE = "2024-07-20"
DAYS = 5

OUTPUT = Path("data/raw/firms/viirs_noaa20_20240720_5d.csv")


def main():
    df = download_firms_area(
        source=SOURCE,
        date=START_DATE,
        days=DAYS,
    )

    print("\nFIRMS COLUMNS")
    print(df.columns.tolist())

    print("\nSAMPLE DETECTIONS")
    print(df.head(10).to_string(index=False))

    if df.empty:
        print("\nNo detections in the requested period.")
        return

    print("\nSATELLITES")
    print(df["satellite"].value_counts().to_string())

    print("\nACQUISITION DATES")
    print(df["acq_date"].value_counts().sort_index().to_string())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
