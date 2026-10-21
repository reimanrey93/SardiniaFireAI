"""Convert a FIRMS event into a fixed-resolution spatial tensor."""

import numpy as np


def rasterize_event(
    event,
    grid_size=32,
    window_m=8000.0,
):
    """Return a [3, H, W] raster from a projected FIRMS event.

    Channels:
      0: detection counts
      1: sum of log1p(FRP) for VIIRS
      2: sum of log1p(FRP) for MODIS

    All coordinates must be in EPSG:32632.
    """
    if event.empty:
        raise ValueError("Empty event")

    if event.crs is None or event.crs.to_epsg() != 32632:
        raise ValueError("Event must use EPSG:32632")

    if grid_size < 4 or window_m <= 0:
        raise ValueError("Invalid raster parameters")

    coords_x = event.geometry.x.to_numpy(dtype=float)
    coords_y = event.geometry.y.to_numpy(dtype=float)

    center_x = float(np.median(coords_x))
    center_y = float(np.median(coords_y))

    left = center_x - window_m / 2
    bottom = center_y - window_m / 2
    cell_size = window_m / grid_size

    cols = np.floor((coords_x - left) / cell_size).astype(int)
    rows = np.floor((coords_y - bottom) / cell_size).astype(int)

    if (
        (cols < 0).any()
        or (cols >= grid_size).any()
        or (rows < 0).any()
        or (rows >= grid_size).any()
    ):
        raise ValueError("Event exceeds the spatial raster window")

    frp = event["frp_mw"].to_numpy(dtype=float)

    if not np.isfinite(frp).all() or (frp < 0).any():
        raise ValueError("Invalid FRP values")

    products = event["source_product"].astype(str)

    raster = np.zeros(
        (3, grid_size, grid_size),
        dtype=np.float32,
    )

    np.add.at(raster[0], (rows, cols), 1.0)

    viirs = products.str.startswith("VIIRS").to_numpy()
    modis = products.str.startswith("MODIS").to_numpy()

    if not (viirs | modis).all():
        raise ValueError("Unexpected FIRMS product")

    np.add.at(
        raster[1],
        (rows[viirs], cols[viirs]),
        np.log1p(frp[viirs]),
    )

    np.add.at(
        raster[2],
        (rows[modis], cols[modis]),
        np.log1p(frp[modis]),
    )

    return raster
