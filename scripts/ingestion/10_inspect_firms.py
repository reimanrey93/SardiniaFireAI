"""Inspect NASA FIRMS product availability."""

from sardinia_fire.ingestion.firms import get_selected_availability


def main():
    availability = get_selected_availability()

    print("\nNASA FIRMS PRODUCT AVAILABILITY")
    print(availability.to_string(index=False))

    if availability.empty:
        raise RuntimeError("No selected FIRMS products were returned")


if __name__ == "__main__":
    main()
