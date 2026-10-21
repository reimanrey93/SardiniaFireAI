"""Compare optimization methods for the same tabular PyTorch MLP."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from sardinia_fire.models.hybrid_fire import TabularFireMLP

DATA = Path(
    "data/processed/evaluation/ranking_dataset_2024_06_09.parquet"
)
OUT = Path(
    "outputs/tables/torch_optimizer_comparison_2024_06_09.csv"
)

NON_FRP = [
    "detections", "satellites", "products",
    "duration_h", "bbox_diagonal_km",
    "observation_groups", "active_days",
    "viirs_detections", "modis_detections",
    "detections_per_group_mean",
]

FRP = [
    "frp_max_mw", "frp_mean_mw",
    "frp_median_mw", "frp_p90_mw",
    "frp_group_peak_mean_mw", "frp_group_peak_p90_mw",
    "viirs_frp_median_mw", "viirs_frp_p90_mw",
    "modis_frp_median_mw", "modis_frp_p90_mw",
]

FEATURES = NON_FRP + FRP
SEED = 42


def fit_predict(
    x_train, y_train, x_test, optimizer_name, seed
):
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    model = TabularFireMLP(len(FEATURES)).double()

    xt = torch.as_tensor(x_train, dtype=torch.float64)
    yt = torch.as_tensor(y_train, dtype=torch.float64)
    xv = torch.as_tensor(x_test, dtype=torch.float64)

    if optimizer_name == "LBFGS":
        alpha = 10.0
        n = len(y_train)

        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=300,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()

            prediction = model(xt)

            mse = 0.5 * torch.mean(
                (prediction - yt).square()
            )

            penalty = sum(
                parameter.square().sum()
                for name, parameter in model.named_parameters()
                if name.endswith("weight")
            )

            loss = mse + alpha * penalty / (2.0 * n)
            loss.backward()
            return loss

        model.train()
        optimizer.step(closure)

    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.05,
        )

        model.train()
        for _ in range(120):
            optimizer.zero_grad()
            loss = torch.mean((model(xt) - yt).square())
            loss.backward()
            optimizer.step()

    else:
        raise ValueError(optimizer_name)

    model.eval()

    with torch.no_grad():
        log_pred = model(xv).numpy()

    log_pred = np.clip(
        log_pred,
        0.0,
        float(np.max(y_train)),
    )

    return np.expm1(log_pred)


def main():
    df = pd.read_parquet(DATA)

    df = df.loc[
        df["label_status"].eq("provisional_one_to_one")
        & df["reference_area_ha"].notna()
        & df["reference_area_ha"].gt(0)
    ].sort_values("cluster_id").reset_index(drop=True)

    assert len(df) == 56
    assert df["reference_fire_id"].is_unique

    x = df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    y = df["reference_area_ha"].to_numpy(float)
    y_log = np.log1p(y)

    splits = list(
        GroupKFold(n_splits=5).split(
            x,
            y_log,
            df["reference_fire_id"],
        )
    )

    results = df[
        [
            "cluster_id",
            "reference_fire_id",
            "reference_area_ha",
            "quality_category",
        ]
    ].copy()

    results["fold"] = -1
    results["AdamW"] = np.nan
    results["LBFGS"] = np.nan

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            imputer.fit_transform(x.iloc[train_idx])
        )
        x_test = scaler.transform(
            imputer.transform(x.iloc[test_idx])
        )

        results.loc[test_idx, "fold"] = fold

        for name in ("AdamW", "LBFGS"):
            prediction = fit_predict(
                x_train,
                y_log[train_idx],
                x_test,
                name,
                SEED + fold,
            )

            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Non-finite predictions: {name}, fold {fold}"
                )

            results.loc[test_idx, name] = prediction

        print(f"Fold {fold} completed")

    assert results[["AdamW", "LBFGS"]].notna().all().all()

    print("\nTORCH OPTIMIZER COMPARISON")

    for subset, mask in {
        "All": np.ones(len(df), dtype=bool),
        "A+B": df["quality_category"].isin(
            ["A_high_spatial_evidence", "B_partial"]
        ).to_numpy(),
    }.items():
        truth = y[mask]

        for name in ("AdamW", "LBFGS"):
            pred = results.loc[mask, name].to_numpy(float)

            rho = spearmanr(truth, pred).statistic
            mae = mean_absolute_error(truth, pred)
            rmse = np.sqrt(mean_squared_error(truth, pred))
            ndcg = ndcg_score(
                truth.reshape(1, -1),
                pred.reshape(1, -1),
                k=min(10, len(truth)),
            )

            print(
                f"{subset:4s} {name:5s} "
                f"Spearman={rho:.3f} "
                f"NDCG@10={ndcg:.3f} "
                f"MAE={mae:.2f} "
                f"RMSE={rmse:.2f}"
            )

    print("\nPER-FOLD SPEARMAN")

    for fold, part in results.groupby("fold"):
        print(
            f"Fold {fold}: "
            f"AdamW={spearmanr(part['reference_area_ha'], part['AdamW']).statistic:.3f} "
            f"LBFGS={spearmanr(part['reference_area_ha'], part['LBFGS']).statistic:.3f}"
        )

    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)

    print("\nSaved:", OUT)
    print("Optimizer comparison completed.")


if __name__ == "__main__":
    main()
