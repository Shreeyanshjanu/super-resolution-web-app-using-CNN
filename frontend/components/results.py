from __future__ import annotations

import numpy as np
import rasterio
from PIL import Image


def geotiff_to_rgb(
    geotiff_bytes: bytes,
    scale: float = 10000.0,
) -> Image.Image:
    """
    Convert a 4-band GeoTIFF into an RGB PIL image.

    Band convention:
        Band 1 = Red
        Band 2 = Green
        Band 3 = Blue
        Band 4 = NIR
    """

    if not geotiff_bytes:
        raise ValueError(
            "GeoTIFF data is empty."
        )

    with rasterio.MemoryFile(
        geotiff_bytes
    ) as memfile:

        with memfile.open() as src:

            if src.count < 3:
                raise ValueError(
                    "The GeoTIFF must contain at least "
                    "3 bands for RGB visualization."
                )

            red = src.read(1).astype(
                np.float32
            )

            green = src.read(2).astype(
                np.float32
            )

            blue = src.read(3).astype(
                np.float32
            )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    rgb /= scale

    # Robust contrast stretching.
    low = np.percentile(
        rgb,
        2,
    )

    high = np.percentile(
        rgb,
        98,
    )

    if high <= low:
        high = low + 1e-8

    rgb = (
        rgb - low
    ) / (
        high - low
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    rgb_uint8 = (
        rgb * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb_uint8,
        mode="RGB",
    )


def get_geotiff_metadata(
    geotiff_bytes: bytes,
) -> dict:
    """
    Extract metadata from a GeoTIFF.
    """

    if not geotiff_bytes:
        raise ValueError(
            "GeoTIFF data is empty."
        )

    with rasterio.MemoryFile(
        geotiff_bytes
    ) as memfile:

        with memfile.open() as src:

            return {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
                "crs": (
                    str(src.crs)
                    if src.crs
                    else None
                ),
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
            }


def get_rgb_array(
    geotiff_bytes: bytes,
    scale: float = 10000.0,
) -> np.ndarray:
    """
    Return RGB as H x W x 3 normalized array.
    """

    with rasterio.MemoryFile(
        geotiff_bytes
    ) as memfile:

        with memfile.open() as src:

            if src.count < 3:
                raise ValueError(
                    "The GeoTIFF must contain at least 3 bands."
                )

            red = src.read(1).astype(
                np.float32
            )

            green = src.read(2).astype(
                np.float32
            )

            blue = src.read(3).astype(
                np.float32
            )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    rgb /= scale

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


__all__ = [
    "geotiff_to_rgb",
    "get_geotiff_metadata",
    "get_rgb_array",
]