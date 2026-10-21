"""Controlled tabular ML comparison with a compact MLP."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(
    "data/processed/evaluation/ranking_dataset_2024_06_09.parquet"
)

OUTPUT = Path(
    "outputs/tables/ml_comparison_mlp_2024_06_09.csv"
)

FEATURES = [
    "detections",
    "frp_max_mw",
    "frp_p90_mw",
    "frp_group_peak_p90_mw",
]


def build_models():
    return {
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "HGB": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=7,
                min_samples_leaf=5,
                l2_regularization=10.0,
                learning_rate=0.05,
                random_state=42,
            ),
        ),
        "MLP": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="lbfgs",
                alpha=10.0,
                max_iter=600,
                random_state=42,
            ),
        ),
    }


def main():
    df = pd.read_parquet(DATA)

    df = df.loc[
        df["label_status"].eq("provisional_one_to_one")
        & df["reference_area_ha"].gt(0)
    ].copy()

    df = df.sort_values("cluster_id").reset_index(drop=True)

    assert len(df) == 56
    assert df["reference_fire_id"].is_unique

    x = df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    y_area = df["reference_area_ha"].to_numpy(dtype=float)
    y_log = np.log1p(y_area)

    groups = df["reference_fire_id"].astype(str)

    splits = list(
        GroupKFold(n_splits=5).split(x, y_log, groups)
    )

    records = []
    predictions = df[
        [
            "cluster_id",
            "reference_fire_id",
            "reference_area_ha",
            "quality_category",
        ]
    ].copy()

    for name, model in build_models().items():
        oof = np.full(len(df), np.nan)

        for train_idx, test_idx in splits:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", ConvergenceWarning
                )

                model.fit(
                    x.iloc[train_idx],
                    y_log[train_idx],
                )

            predicted_log = model.predict(
                x.iloc[test_idx]
            )

            # Bounds use training targets only.
            predicted_log = np.clip(
                predicted_log,
                0.0,
                float(y_log[train_idx].max()),
            )

            oof[test_idx] = np.expm1(predicted_log)

        assert np.isfinite(oof).all()

        predictions[name] = oof

        for subset_name, mask in {
            "All": np.ones(len(df), dtype=bool),
            "A+B": df["quality_category"].isin([
                "A_high_spatial_evidence",
                "B_partial",
            ]).to_numpy(),
        }.items():
            truth = y_area[mask]
            estimated = oof[mask]

            records.append({
                "subset": subset_name,
                "model": name,
                "n": len(truth),
                "mae_ha": mean_absolute_error(
                    truth, estimated
                ),
                "rmse_ha": np.sqrt(
                    mean_squared_error(truth, estimated)
                ),
                "spearman": spearmanr(
                    truth, estimated
                ).statistic,
            })

    results = pd.DataFrame(records)

    print("\nTABULAR ML COMPARISON")
    print(results.round(3).to_string(index=False))

    print("\nMLP AND RIDGE PREDICTION RANGES")
    for name in ("Ridge", "MLP"):
        print(
            name,
            "min:",
            round(predictions[name].min(), 2),
            "max:",
            round(predictions[name].max(), 2),
        )

    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)
    print("ML comparison completed.")


if __name__ == "__main__":
    main()
