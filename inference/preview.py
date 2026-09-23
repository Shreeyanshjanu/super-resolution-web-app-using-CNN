"""Bounded display previews. Reprojection affects previews, never model inputs."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


def rgb_image(array, limits=None):
    array = np.ma.asarray(array, dtype=np.float32)
    valid = ~np.ma.getmaskarray(array).any(axis=0)
    values = array.filled(np.nan)
    valid &= np.isfinite(values).all(axis=0)
    if limits is None:
        samples = values[:, valid]
        if not samples.size:
            raise ValueError("No valid pixels available for a preview.")
        low, high = np.percentile(samples, (2, 98))
        limits = (float(low), float(max(high, low + 1e-6)))
    low, high = limits
    rgb = np.clip((np.nan_to_num(values) - low) / max(high - low, 1e-6), 0, 1)
    rgba = np.concatenate([(rgb.transpose(1, 2, 0) * 255).astype(np.uint8), (valid * 255).astype(np.uint8)[..., None]], axis=2)
    return Image.fromarray(rgba), limits


def create_preview(source: Path, destination: Path, indexes, max_size=768):
    with rasterio.open(source) as src:
        ratio = min(1, max_size / max(src.width, src.height))
        width, height = max(1, round(src.width * ratio)), max(1, round(src.height * ratio))
        data = src.read(indexes[:3], out_shape=(3, height, width), masked=True, resampling=Resampling.bilinear).astype(np.float32)
        bounds_wgs84 = None
        if src.crs:
            # Warp the small preview to Leaflet's Web Mercator grid.
            source_transform = src.transform * Affine.scale(src.width / width, src.height / height)
            transform, w, h = calculate_default_transform(src.crs, "EPSG:3857", width, height, *src.bounds)
            ratio = min(1, max_size / max(w, h))
            out_w, out_h = max(1, round(w * ratio)), max(1, round(h * ratio))
            transform = transform * Affine.scale(w / out_w, h / out_h)
            warped = np.full((3, out_h, out_w), np.nan, dtype=np.float32)
            reproject(data.filled(np.nan), warped, src_transform=source_transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:3857", src_nodata=np.nan,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
            data = np.ma.masked_invalid(warped)
            bounds = rasterio.transform.array_bounds(out_h, out_w, transform)
            bounds_wgs84 = list(transform_bounds("EPSG:3857", "EPSG:4326", *bounds))
        image, _ = rgb_image(data)
        image.save(destination, format="PNG")
        return {"preview_bounds_wgs84": bounds_wgs84}


def png_bytes(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
