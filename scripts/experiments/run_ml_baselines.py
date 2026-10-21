"""First out-of-fold ML baselines for CFVA burned area."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "detections",
    "frp_max_mw",
    "frp_p90_mw",
    "frp_group_peak_p90_mw",
]

DATASET = Path(
    "data/processed/evaluation/"
    "ranking_dataset_2024_06_09.parquet"
)

OUTPUT = Path(
    "outputs/tables/ml_baselines_2024_06_09.csv"
)


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
                max_features=1.0,
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
    }


def evaluate(name, truth, prediction, subset):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    rho = (
        float(spearmanr(truth, prediction).statistic)
        if np.unique(prediction).size > 1
        else np.nan
    )

    return {
        "subset": subset,
        "model": name,
        "n": len(truth),
        "mae_ha": mean_absolute_error(truth, prediction),
        "rmse_ha": np.sqrt(
            mean_squared_error(truth, prediction)
        ),
        "spearman": rho,
    }


def main():
    df = pd.read_parquet(DATASET)

    required = set(FEATURES) | {
        "cluster_id",
        "reference_fire_id",
        "reference_area_ha",
        "label_status",
        "quality_category",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.loc[
        df["label_status"].eq("provisional_one_to_one")
        & df["reference_area_ha"].notna()
        & df["reference_area_ha"].gt(0)
    ].copy()

    df = df.sort_values("cluster_id").reset_index(drop=True)

    assert len(df) == 56
    assert df["cluster_id"].is_unique
    assert df["reference_fire_id"].notna().all()

    X = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    if X.isna().all(axis=0).any():
        raise ValueError("A predictor contains only missing values")

    y_area = df["reference_area_ha"].to_numpy(dtype=float)
    y_log = np.log1p(y_area)
    groups = df["reference_fire_id"].astype(str)

    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(X, y_log, groups))

    for train_idx, test_idx in splits:
        assert set(groups.iloc[train_idx]).isdisjoint(
            set(groups.iloc[test_idx])
        )

    predictions = df[
        [
            "cluster_id",
            "reference_fire_id",
            "reference_area_ha",
            "quality_category",
        ]
    ].copy()

    models = build_models()

    print("\nML DATASET")
    print("Events:", len(df))
    print("Unique CFVA fires:", groups.nunique())
    print("Predictors:", FEATURES)
    print("Folds:", len(splits))

    for name, model in models.items():
        oof_log = np.full(len(df), np.nan)

        for train_idx, test_idx in splits:
            model.fit(
                X.iloc[train_idx],
                y_log[train_idx],
            )

            oof_log[test_idx] = model.predict(
                X.iloc[test_idx]
            )

        assert np.isfinite(oof_log).all()

        predictions[name] = np.maximum(
            np.expm1(oof_log), 0.0
        )

    # Non-trained comparison scores.
    # These are ranking proxies, not area predictions.
    predictions["Detection count"] = X["detections"]
    predictions["FRP maximum"] = X["frp_max_mw"]
    predictions["Count + FRP"] = (
        X["detections"].rank(pct=True)
        + X["frp_max_mw"].rank(pct=True)
    ) / 2

    subsets = {
        "All": np.ones(len(df), dtype=bool),
        "A+B": df["quality_category"].isin([
            "A_high_spatial_evidence",
            "B_partial",
        ]).to_numpy(),
    }

    records = []

    for subset_name, mask in subsets.items():
        for name in models:
            records.append(
                evaluate(
                    name,
                    y_area[mask],
                    predictions.loc[mask, name],
                    subset_name,
                )
            )

        for name in (
            "Detection count",
            "FRP maximum",
            "Count + FRP",
        ):
            scores = predictions.loc[mask, name]

            records.append({
                "subset": subset_name,
                "model": name,
                "n": int(mask.sum()),
                "mae_ha": np.nan,
                "rmse_ha": np.nan,
                "spearman": scores.corr(
                    predictions.loc[
                        mask, "reference_area_ha"
                    ],
                    method="spearman",
                ),
            })

    results = pd.DataFrame(records)

    print("\nML BASELINE RESULTS")
    print(results.round(3).to_string(index=False))

    print("\nFOLD SIZES")
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)
    print("ML baseline experiment completed.")


if __name__ == "__main__":
    main()
