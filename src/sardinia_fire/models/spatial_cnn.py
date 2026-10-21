"""Small convolutional network for FIRMS event rasters."""

from torch import nn


class FireSpatialCNN(nn.Module):
    """Predict log1p(burned area) from a three-channel event raster."""

    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected tensor [batch, channels, H, W]")

        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)
