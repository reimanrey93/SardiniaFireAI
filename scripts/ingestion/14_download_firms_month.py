"""Download historical FIRMS observations for a requested month."""

import argparse

from sardinia_fire.ingestion.historical_download import ingest_firms_month


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    result = ingest_firms_month(
        year=args.year,
        month=args.month,
        overwrite=args.overwrite,
    )

    print("\nMONTHLY FIRMS SUMMARY")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
