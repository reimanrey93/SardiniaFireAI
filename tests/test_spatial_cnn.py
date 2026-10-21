import geopandas as gpd
import numpy as np
import pytest
import torch
from shapely.geometry import Point

from sardinia_fire.models.spatial_cnn import FireSpatialCNN
from sardinia_fire.models.spatial_raster import rasterize_event


def sample_event():
    return gpd.GeoDataFrame(
        {
            "source_product": [
                "VIIRS_SNPP_SP",
                "VIIRS_NOAA20_SP",
                "MODIS_SP",
            ],
            "frp_mw": [3.0, 7.0, 12.0],
        },
        geometry=[
            Point(500000, 4400000),
            Point(500250, 4400250),
            Point(500500, 4400500),
        ],
        crs="EPSG:32632",
    )


def test_raster_shape_and_detection_conservation():
    raster = rasterize_event(sample_event())

    assert raster.shape == (3, 32, 32)
    assert raster.dtype == np.float32
    assert raster[0].sum() == pytest.approx(3.0)


def test_raster_sensor_channels():
    raster = rasterize_event(sample_event())

    assert raster[1].sum() == pytest.approx(
        np.log1p(3.0) + np.log1p(7.0)
    )
    assert raster[2].sum() == pytest.approx(np.log1p(12.0))


def test_invalid_crs_rejected():
    event = sample_event().to_crs("EPSG:4326")

    with pytest.raises(ValueError):
        rasterize_event(event)


def test_unknown_product_rejected():
    event = sample_event()
    event.loc[0, "source_product"] = "UNKNOWN"

    with pytest.raises(ValueError):
        rasterize_event(event)


def test_cnn_forward_and_backward():
    torch.manual_seed(42)
    model = FireSpatialCNN()

    x = torch.randn(4, 3, 32, 32)
    predictions = model(x)

    assert predictions.shape == (4,)

    loss = predictions.square().mean()
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )


def test_cnn_rejects_invalid_dimensions():
    model = FireSpatialCNN()

    with pytest.raises(ValueError):
        model(torch.randn(3, 32, 32))
