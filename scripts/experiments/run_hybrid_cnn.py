"""Controlled comparison of tabular MLP and hybrid FIRMS CNN."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, ndcg_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from sardinia_fire.models.hybrid_fire import (
    HybridFireCNN,
    TabularFireMLP,
)
from sardinia_fire.models.spatial_raster import rasterize_event

LABELS_PATH = Path(
    "data/processed/evaluation/ranking_dataset_2024_06_09.parquet"
)
DETECTIONS_PATH = Path(
    "data/processed/clustering/st_dbscan_2024_06_09.parquet"
)
OUTPUT = Path("outputs/tables/hybrid_cnn_oof_2024_06_09.csv")

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
EPOCHS = 120
SEED = 42


def fit_and_predict(
    model, tab_train, raster_train, y_train,
    tab_test, raster_test, hybrid,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.05,
    )
    loss_fn = nn.MSELoss()

    t_train = torch.as_tensor(tab_train, dtype=torch.float32)
    r_train = torch.as_tensor(raster_train, dtype=torch.float32)
    target = torch.as_tensor(y_train, dtype=torch.float32)
    t_test = torch.as_tensor(tab_test, dtype=torch.float32)
    r_test = torch.as_tensor(raster_test, dtype=torch.float32)

    model.train()

    for _ in range(EPOCHS):
        optimizer.zero_grad()

        prediction = (
            model(r_train, t_train)
            if hybrid
            else model(t_train)
        )

        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        output = (
            model(r_test, t_test)
            if hybrid
            else model(t_test)
        ).numpy()

    output = np.clip(output, 0.0, float(np.max(y_train)))
    return np.expm1(output)


def main():
    torch.set_num_threads(2)

    labels = pd.read_parquet(LABELS_PATH)
    labels = labels.loc[
        labels["label_status"].eq("provisional_one_to_one")
        & labels["reference_area_ha"].gt(0)
    ].sort_values("cluster_id").reset_index(drop=True)

    assert len(labels) == 56
    assert labels["reference_fire_id"].is_unique

    detections = gpd.read_parquet(DETECTIONS_PATH)
    detections = detections.to_crs("EPSG:32632")

    rasters = np.stack([
        rasterize_event(
            detections.loc[
                detections["cluster_id"] == cluster_id
            ],
            grid_size=32,
            window_m=8000.0,
        )
        for cluster_id in labels["cluster_id"]
    ])

    tabular = labels[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    y = labels["reference_area_ha"].to_numpy(float)
    y_log = np.log1p(y)
    groups = labels["reference_fire_id"].astype(str)

    splits = list(
        GroupKFold(n_splits=5).split(tabular, y_log, groups)
    )

    results = labels[
        [
            "cluster_id", "reference_fire_id",
            "reference_area_ha", "quality_category",
        ]
    ].copy()

    results["fold"] = -1

    for model_name in ("TabularMLP", "HybridCNN"):
        results[model_name] = np.nan

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        assert set(groups.iloc[train_idx]).isdisjoint(
            set(groups.iloc[test_idx])
        )

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        tab_train = scaler.fit_transform(
            imputer.fit_transform(tabular.iloc[train_idx])
        ).astype(np.float32)

        tab_test = scaler.transform(
            imputer.transform(tabular.iloc[test_idx])
        ).astype(np.float32)

        raster_train = np.log1p(rasters[train_idx])
        raster_test = np.log1p(rasters[test_idx])

        channel_scale = np.maximum(
            raster_train.max(axis=(0, 2, 3), keepdims=True),
            1e-6,
        )

        raster_train = raster_train / channel_scale
        raster_test = raster_test / channel_scale

        results.loc[test_idx, "fold"] = fold

        print(f"\nFold {fold}")

        for name, model_class, hybrid in [
            ("TabularMLP", TabularFireMLP, False),
            ("HybridCNN", HybridFireCNN, True),
        ]:
            torch.manual_seed(SEED + fold)

            model = model_class(len(FEATURES))

            predicted = fit_and_predict(
                model,
                tab_train,
                raster_train,
                y_log[train_idx],
                tab_test,
                raster_test,
                hybrid,
            )

            assert np.isfinite(predicted).all()

            results.loc[test_idx, name] = predicted

            print(
                f"  {name}: "
                f"{predicted.min():.2f}–"
                f"{predicted.max():.2f} ha"
            )

    assert results["fold"].ge(1).all()
    assert results[["TabularMLP", "HybridCNN"]].notna().all().all()

    baseline = (
        labels["detections"].rank(pct=True)
        + labels["frp_max_mw"].rank(pct=True)
    ) / 2

    results["Baseline"] = baseline.to_numpy()

    print("\nHYBRID MODEL COMPARISON")

    for subset, mask in {
        "All": np.ones(len(labels), dtype=bool),
        "A+B": labels["quality_category"].isin(
            ["A_high_spatial_evidence", "B_partial"]
        ).to_numpy(),
    }.items():
        truth = y[mask]

        for name in ("Baseline", "TabularMLP", "HybridCNN"):
            pred = results.loc[mask, name].to_numpy(float)

            rho = spearmanr(truth, pred).statistic
            ndcg = ndcg_score(
                truth.reshape(1, -1),
                pred.reshape(1, -1),
                k=min(10, len(truth)),
            )

            mae = (
                mean_absolute_error(truth, pred)
                if name != "Baseline"
                else np.nan
            )

            print(
                f"{subset:4s} {name:12s} "
                f"Spearman={rho:.3f} "
                f"NDCG@10={ndcg:.3f} "
                f"MAE={mae:.2f}"
            )

    if OUTPUT.exists():
        raise FileExistsError(f"Output exists: {OUTPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)
    print("Hybrid experiment completed.")


if __name__ == "__main__":
    main()
