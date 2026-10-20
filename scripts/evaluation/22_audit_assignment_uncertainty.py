"""Audit uncertainty in provisional FIRMS-CFVA assignments."""

from pathlib import Path

import pandas as pd

INPUT = Path("outputs/tables/firms_cfva_all_candidates_2024_07.csv")
OUTPUT = Path("outputs/tables/firms_cfva_assignment_uncertainty.csv")

DISTANCE_TOLERANCE_M = 1.0


def main():
    df = pd.read_csv(INPUT)

    required = {
        "detection_id", "fire_id",
        "distance_m", "delta_days", "candidate_rank",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    rows = []

    for detection_id, group in df.groupby("detection_id"):
        group = group.sort_values("candidate_rank")
        first = group.iloc[0]

        same_distance = group["distance_m"].sub(
            first["distance_m"]
        ).abs() <= DISTANCE_TOLERANCE_M

        same_date = (
            group["delta_days"].abs()
            == abs(first["delta_days"])
        )

        tied = group.loc[same_distance & same_date]

        rows.append({
            "detection_id": detection_id,
            "selected_fire_id": first["fire_id"],
            "candidate_count": group["fire_id"].nunique(),
            "distance_m": first["distance_m"],
            "abs_delta_days": abs(first["delta_days"]),
            "equivalent_candidates": tied["fire_id"].nunique(),
            "unresolved_tie": tied["fire_id"].nunique() > 1,
        })

    audit = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT, index=False)

    print("\nASSIGNMENT UNCERTAINTY AUDIT")
    print(f"Assigned detections: {len(audit)}")
    print(f"Single candidate: {(audit['candidate_count'] == 1).sum()}")
    print(f"Multiple candidates: {(audit['candidate_count'] > 1).sum()}")
    print(f"Unresolved near-ties: {audit['unresolved_tie'].sum()}")
    print(
        "Selected CFVA fires:",
        audit["selected_fire_id"].nunique(),
    )
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
