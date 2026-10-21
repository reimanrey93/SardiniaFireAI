"""Tabular and hybrid neural regressors for FIRMS events."""

import torch
from torch import nn


class TabularFireMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.tabular = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.head = nn.Linear(8, 1)

    def forward(self, tabular):
        return self.head(self.tabular(tabular)).squeeze(-1)


class HybridFireCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.spatial_projection = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.tabular = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, raster, tabular):
        spatial = self.spatial_projection(self.spatial(raster))
        features = self.tabular(tabular)
        return self.head(
            torch.cat([spatial, features], dim=1)
        ).squeeze(-1)
