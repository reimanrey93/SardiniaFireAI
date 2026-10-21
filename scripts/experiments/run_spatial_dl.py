"""Out-of-fold comparison of CNN and raster-based MLP."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
)
from sklearn.model_selection import GroupKFold
from torch import nn

from sardinia_fire.models.spatial_cnn import FireSpatialCNN
from sardinia_fire.models.spatial_raster import rasterize_event

DATA = Path(
    "data/processed/evaluation/ranking_dataset_2024_06_09.parquet"
)
DETECTIONS = Path(
    "data/processed/clustering/st_dbscan_2024_06_09.parquet"
)
OUT = Path("outputs/tables")
EPOCHS = 80
SEED = 42


class SpatialMLP(nn.Module):
    """Non-convolutional reference model for the same raster data."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_predict(model, x_train, y_train, x_test, seed):
    torch.manual_seed(seed)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.01,
    )
    loss_fn = nn.MSELoss()

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_test, dtype=torch.float32)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        prediction = model(xt)
        loss = loss_fn(prediction, yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_log = model(xv).cpu().numpy()

    # The limit uses training targets only.
    predicted_log = np.clip(
        predicted_log,
        0.0,
        float(np.max(y_train)),
    )

    return np.expm1(predicted_log)


def evaluate(y, prediction):
    return {
        "mae_ha": float(mean_absolute_error(y, prediction)),
        "rmse_ha": float(
            np.sqrt(mean_squared_error(y, prediction))
        ),
        "spearman": float(
            spearmanr(y, prediction).statistic
        ),
        "ndcg_at_10": float(
            ndcg_score(
                y.reshape(1, -1),
                prediction.reshape(1, -1),
                k=min(10, len(y)),
            )
        ),
    }


def main():
    torch.set_num_threads(2)

    labels = pd.read_parquet(DATA)
    labels = labels.loc[
        labels["label_status"].eq("provisional_one_to_one")
        & labels["reference_area_ha"].notna()
        & labels["reference_area_ha"].gt(0)
    ].sort_values("cluster_id").reset_index(drop=True)

    assert len(labels) == 56
    assert labels["reference_fire_id"].is_unique

    detections = gpd.read_parquet(DETECTIONS)
    detections = detections.to_crs("EPSG:32632")

    rasters = []
    for cluster_id in labels["cluster_id"]:
        group = detections.loc[
            detections["cluster_id"] == cluster_id
        ]
        image = rasterize_event(
            group,
            grid_size=32,
            window_m=8000.0,
        )
        rasters.append(image)

    x = np.stack(rasters).astype(np.float32)
    y = labels["reference_area_ha"].to_numpy(dtype=float)
    y_log = np.log1p(y)
    groups = labels["reference_fire_id"].astype(str)

    assert x.shape == (56, 3, 32, 32)
    assert np.isfinite(x).all()

    splits = list(
        GroupKFold(n_splits=5).split(x, y_log, groups)
    )

    predictions = labels[
        [
            "cluster_id",
            "reference_fire_id",
            "reference_area_ha",
            "quality_category",
        ]
    ].copy()

    predictions["fold"] = -1
    predictions["CNN"] = np.nan
    predictions["SpatialMLP"] = np.nan

    model_classes = {
        "CNN": FireSpatialCNN,
        "SpatialMLP": SpatialMLP,
    }

    print("\nSPATIAL DEEP LEARNING EXPERIMENT")
    print("Events:", len(labels))
    print("Raster shape:", x.shape)
    print("Epochs per fold:", EPOCHS)

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        assert set(groups.iloc[train_idx]).isdisjoint(
            set(groups.iloc[test_idx])
        )

        predictions.loc[test_idx, "fold"] = fold

        # Fixed monotonic transform, followed by train-only scaling.
        train_raw = np.log1p(x[train_idx])
        test_raw = np.log1p(x[test_idx])

        scale = np.maximum(
            train_raw.max(axis=(0, 2, 3), keepdims=True),
            1e-6,
        )

        x_train = train_raw / scale
        x_test = test_raw / scale

        print(
            f"\nFold {fold}: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

        for name, model_class in model_classes.items():
            seed = SEED + fold
            torch.manual_seed(seed)

            model = model_class()

            prediction = train_predict(
                model,
                x_train,
                y_log[train_idx],
                x_test,
                seed,
            )

            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Non-finite predictions: {name}, fold {fold}"
                )

            predictions.loc[test_idx, name] = prediction

            print(
                f"  {name}: "
                f"predicted range "
                f"{prediction.min():.2f}–"
                f"{prediction.max():.2f} ha"
            )

    assert predictions["fold"].ge(1).all()
    assert predictions[["CNN", "SpatialMLP"]].notna().all().all()

    baseline = (
        labels["detections"].rank(pct=True)
        + labels["frp_max_mw"].rank(pct=True)
    ) / 2

    predictions["Count_plus_FRP"] = baseline.to_numpy()

    rows = []
    subsets = {
        "All": np.ones(len(labels), dtype=bool),
        "A+B": labels["quality_category"].isin(
            ["A_high_spatial_evidence", "B_partial"]
        ).to_numpy(),
    }

    for subset_name, mask in subsets.items():
        truth = y[mask]

        for name in ("CNN", "SpatialMLP", "Count_plus_FRP"):
            estimated = predictions.loc[
                mask, name
            ].to_numpy(dtype=float)

            result = evaluate(truth, estimated)

            if name == "Count_plus_FRP":
                result["mae_ha"] = np.nan
                result["rmse_ha"] = np.nan

            rows.append({
                "subset": subset_name,
                "model": name,
                "n": int(mask.sum()),
                **result,
            })

    results = pd.DataFrame(rows)

    print("\nSPATIAL MODEL COMPARISON")
    print(results.round(3).to_string(index=False))

    metric_path = OUT / "spatial_dl_metrics_2024_06_09.csv"
    pred_path = OUT / "spatial_dl_oof_2024_06_09.csv"

    for path in (metric_path, pred_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(metric_path, index=False)
    predictions.to_csv(pred_path, index=False)

    print("\nSaved:", metric_path)
    print("Saved:", pred_path)
    print("Spatial deep learning experiment completed.")


if __name__ == "__main__":
    main()
