"""Exploratory feature-family ablation for CFVA burned area."""

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
    "data/processed/evaluation/"
    "ranking_dataset_2024_06_09.parquet"
)

OUT = Path("outputs/tables")

NON_FRP = [
    "detections",
    "satellites",
    "products",
    "duration_h",
    "bbox_diagonal_km",
    "observation_groups",
    "active_days",
    "viirs_detections",
    "modis_detections",
    "detections_per_group_mean",
]

FRP = [
    "frp_max_mw",
    "frp_mean_mw",
    "frp_median_mw",
    "frp_p90_mw",
    "frp_group_peak_mean_mw",
    "frp_group_peak_p90_mw",
    "viirs_frp_median_mw",
    "viirs_frp_p90_mw",
    "modis_frp_median_mw",
    "modis_frp_p90_mw",
]

EXPERIMENTS = {
    "E1_non_frp": NON_FRP,
    "E2_frp_only": FRP,
    "E3_combined": NON_FRP + FRP,
}


def make_models():
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
                learning_rate=0.05,
                l2_regularization=10.0,
                random_state=42,
            ),
        ),
        "MLP": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(16, 8),
                solver="lbfgs",
                alpha=10.0,
                max_iter=600,
                random_state=42,
            ),
        ),
    }


def metrics(truth, estimate):
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    rho = (
        float(spearmanr(truth, estimate).statistic)
        if np.unique(estimate).size > 1
        else np.nan
    )

    return {
        "mae_ha": float(mean_absolute_error(truth, estimate)),
        "rmse_ha": float(
            np.sqrt(mean_squared_error(truth, estimate))
        ),
        "spearman": rho,
    }


def main():
    df = pd.read_parquet(DATA)

    df = df.loc[
        df["label_status"].eq("provisional_one_to_one")
        & df["reference_area_ha"].notna()
        & df["reference_area_ha"].gt(0)
    ].sort_values("cluster_id").reset_index(drop=True)

    assert len(df) == 56
    assert df["cluster_id"].is_unique
    assert df["reference_fire_id"].notna().all()
    assert df["reference_fire_id"].is_unique

    all_features = NON_FRP + FRP

    assert len(all_features) == len(set(all_features))

    missing = set(all_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing predictors: {sorted(missing)}")

    x_all = (
        df[all_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    if x_all.isna().all(axis=0).any():
        raise ValueError("Some features contain only missing values")

    y_area = df["reference_area_ha"].to_numpy(dtype=float)
    y_log = np.log1p(y_area)
    groups = df["reference_fire_id"].astype(str)

    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(x_all, y_log, groups))

    fold_ids = np.full(len(df), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        assert set(groups.iloc[train_idx]).isdisjoint(
            set(groups.iloc[test_idx])
        )
        fold_ids[test_idx] = fold

    assert (fold_ids > 0).all()

    predictions = df[
        [
            "cluster_id",
            "reference_fire_id",
            "reference_area_ha",
            "quality_category",
        ]
    ].copy()

    predictions["fold"] = fold_ids

    subsets = {
        "All": np.ones(len(df), dtype=bool),
        "A+B": df["quality_category"].isin(
            ["A_high_spatial_evidence", "B_partial"]
        ).to_numpy(),
    }

    records = []

    for experiment, features in EXPERIMENTS.items():
        x = x_all[features]

        print(
            f"\nRUNNING {experiment} "
            f"({len(features)} features)"
        )

        for model_name, model in make_models().items():
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

                # Training-derived cap, applied equally to all models.
                predicted_log = np.clip(
                    predicted_log,
                    0.0,
                    float(y_log[train_idx].max()),
                )

                oof[test_idx] = np.expm1(predicted_log)

            assert np.isfinite(oof).all()

            prediction_column = (
                f"{experiment}__{model_name}"
            )

            predictions[prediction_column] = oof

            for subset_name, mask in subsets.items():
                records.append({
                    "experiment": experiment,
                    "model": model_name,
                    "subset": subset_name,
                    "n": int(mask.sum()),
                    **metrics(y_area[mask], oof[mask]),
                })

    # Reference baselines; scores only, not predictions in hectares.
    count = x_all["detections"]
    frp_max = x_all["frp_max_mw"]

    baseline_scores = {
        "Detection_count": count,
        "FRP_max": frp_max,
        "Count_plus_FRP": (
            count.rank(pct=True) + frp_max.rank(pct=True)
        ) / 2,
    }

    for name, scores in baseline_scores.items():
        predictions[name] = scores.to_numpy(dtype=float)

        for subset_name, mask in subsets.items():
            records.append({
                "experiment": "Baseline",
                "model": name,
                "subset": subset_name,
                "n": int(mask.sum()),
                "mae_ha": np.nan,
                "rmse_ha": np.nan,
                "spearman": metrics(
                    y_area[mask],
                    scores.to_numpy(dtype=float)[mask],
                )["spearman"],
            })

    results = pd.DataFrame(records)

    print("\nFEATURE FAMILY ABLATION")
    print(
        results.loc[
            results["subset"] == "All"
        ].round(3).to_string(index=False)
    )

    print("\nHIGHER QUALITY REFERENCES (A+B)")
    print(
        results.loc[
            results["subset"] == "A+B"
        ].round(3).to_string(index=False)
    )

    print("\nTOP SUPERVISED RANKING RESULTS")
    supervised = results.loc[
        (results["subset"] == "All")
        & (results["experiment"] != "Baseline")
    ]

    print(
        supervised.sort_values(
            "spearman", ascending=False
        ).head(5).round(3).to_string(index=False)
    )

    metrics_path = OUT / "ml_feature_ablation_2024_06_09.csv"
    predictions_path = (
        OUT / "ml_feature_ablation_oof_2024_06_09.csv"
    )

    for path in (metrics_path, predictions_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    OUT.mkdir(parents=True, exist_ok=True)

    results.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nSaved:", metrics_path)
    print("Saved:", predictions_path)
    print("Feature-family ablation completed.")


if __name__ == "__main__":
    main()
