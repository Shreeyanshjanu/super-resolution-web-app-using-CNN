"""Validate imagery and crop in its native pixel grid before model inference."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from affine import Affine
from pydantic import BaseModel, ConfigDict, model_validator
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

PATCH_SIZE = 128
OVERLAP = 8
FACTOR = 4
BANDS = ("B04", "B03", "B02", "B08")


class AOI(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["center", "pixel", "bbox"] = "center"
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_selection(self):
        pixels = (self.x, self.y, self.width, self.height)
        if self.mode == "pixel":
            if any(v is None for v in pixels) or self.bbox is not None:
                raise ValueError("Pixel selection needs x, y, width and height only.")
            if self.x < 0 or self.y < 0 or self.width < 1 or self.height < 1:
                raise ValueError("Pixel offsets must be nonnegative and dimensions positive.")
        elif self.mode == "bbox":
            if self.bbox is None or any(v is not None for v in pixels):
                raise ValueError("Map selection needs a WGS84 bbox only.")
            west, south, east, north = self.bbox
            if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
                raise ValueError("Select a nonempty rectangle within map bounds; antimeridian crossing is unsupported.")
        elif self.bbox is not None or any(v is not None for v in pixels):
            raise ValueError("Center selection takes no coordinates.")
        return self


def patch_count(width: int, height: int) -> int:
    """Match OpenSR 2.0's final edge window with size 128 and overlap 8."""
    def count(length):
        return 1 + math.ceil(max(0, length - PATCH_SIZE) / (PATCH_SIZE - OVERLAP))
    return count(width) * count(height)


def band_indexes(src, order: str = "auto") -> list[int]:
    descriptions = tuple((d or "").strip().upper() for d in src.descriptions)
    if order == "auto":
        if len(set(descriptions)) == 4 and set(descriptions) == set(BANDS):
            return [descriptions.index(band) + 1 for band in BANDS]
        raise ValueError(
            "This TIFF does not contain Sentinel-2 band descriptions (B02, B03, B04, B08). "
            "Please select RGBN or BGRN manually in the Input band order dropdown."
        )
    if order not in ("rgbn", "bgrn"):
        raise ValueError("Band order must be auto, rgbn or bgrn.")
    return [1, 2, 3, 4] if order == "rgbn" else [3, 2, 1, 4]


def inspect_raster(path: str | Path, order: str = "auto", value_scale: float = 10000) -> dict:
    if value_scale not in (1, 10000):
        raise ValueError("Reflectance scale must be 1 or 10000.")
    with rasterio.open(path) as src:
        if src.driver != "GTiff" or src.count != 4:
            raise ValueError("Upload a four-band RGB + NIR GeoTIFF.")
        indexes = band_indexes(src, order)
        if any(v != 1 for v in src.scales) or any(v != 0 for v in src.offsets):
            raise ValueError("Apply the raster's scale/offset metadata to reflectance before uploading.")
        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError("Rotated rasters must first be reprojected to a north-up 10 m grid.")
        geographic = src.crs is not None
        if geographic:
            if not src.crs.is_projected or src.crs.linear_units != "metre":
                raise ValueError("Reproject imagery to a projected CRS in metres at 10 m resolution.")
            if src.transform.a <= 0 or src.transform.e >= 0 or not np.allclose(src.res, (10, 10), atol=0.01, rtol=0):
                raise ValueError("Geographic SR requires a north-up 10 m Sentinel-2 grid.")
        warnings = []
        if not geographic:
            warnings.append("No CRS: only pixel crops are available. Ground resolution and geographic location are unknown; no coordinates will be invented.")
        sample = src.read(indexes, out_shape=(4, min(src.height, 256), min(src.width, 256)), masked=True)
        validate_values(sample, value_scale)
        return {
            "width": src.width, "height": src.height, "bands": src.count,
            "crs": str(src.crs) if geographic else None,
            "resolution_x": src.res[0], "resolution_y": src.res[1],
            "resolution_unit": "m" if geographic else "unknown",
            "bounds": dict(zip(("left", "bottom", "right", "top"), src.bounds)),
            "bounds_wgs84": list(transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)) if geographic else None,
            "band_indexes": indexes, "input_band_order": order,
            "output_band_order": list(BANDS), "value_scale": value_scale,
            "dtype": src.dtypes[0], "warnings": warnings,
            "full_image_patches": patch_count(src.width, src.height),
        }


def validate_values(data, scale):
    values = np.ma.asarray(data).compressed()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("The selected data contains no valid pixels or has unmasked NaN/infinite values.")
    if values.min() < 0 or values.max() > scale:
        raise ValueError(f"Expected surface reflectance in [0, {scale:g}]. Correct scaling, offsets and NoData before processing.")


def plan_crop(path: str | Path, selection: AOI, max_patches: int = 4) -> dict:
    with rasterio.open(path) as src:
        if selection.mode == "center":
            width, height = min(PATCH_SIZE, src.width), min(PATCH_SIZE, src.height)
            x, y = (src.width - width) // 2, (src.height - height) // 2
        elif selection.mode == "pixel":
            x, y, width, height = selection.x, selection.y, selection.width, selection.height
            if x + width > src.width or y + height > src.height:
                raise ValueError("Pixel selection extends outside the image.")
        else:
            if not src.crs:
                raise ValueError("Map AOIs require a CRS. Use a pixel crop for this raster.")
            native = transform_bounds("EPSG:4326", src.crs, *selection.bbox, densify_pts=21)
            if not all(math.isfinite(v) for v in native):
                raise ValueError("AOI cannot be transformed into this raster's CRS.")
            window = from_bounds(*native, transform=src.transform)
            x = max(0, math.floor(window.col_off + 1e-7))
            y = max(0, math.floor(window.row_off + 1e-7))
            right = min(src.width, math.ceil(window.col_off + window.width - 1e-7))
            bottom = min(src.height, math.ceil(window.row_off + window.height - 1e-7))
            width, height = right - x, bottom - y
            if width <= 0 or height <= 0:
                raise ValueError("The selected rectangle does not intersect the image.")
        count = patch_count(width, height)
        if count > max_patches:
            raise ValueError(f"AOI needs {count} patches; the limit is {max_patches}. Draw a smaller area (128×128 pixels needs one patch).")
        window = Window(x, y, width, height)
        bounds = rasterio.windows.bounds(window, src.transform)
        return {
            "window": {"x": x, "y": y, "width": width, "height": height},
            "patches": count, "full_image_patches": patch_count(src.width, src.height),
            "model_width": max(PATCH_SIZE, width), "model_height": max(PATCH_SIZE, height),
            "output_width": width * FACTOR, "output_height": height * FACTOR,
            "bounds": dict(zip(("left", "bottom", "right", "top"), bounds)),
            "bounds_wgs84": list(transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21)) if src.crs else None,
            "crs": str(src.crs) if src.crs else None,
            "input_resolution_m": 10 if src.crs else None,
            "output_resolution_m": 2.5 if src.crs else None,
            "padding_right": max(0, PATCH_SIZE - width), "padding_bottom": max(0, PATCH_SIZE - height),
        }


def write_crop(path: str | Path, directory: Path, plan: dict, indexes: list[int], value_scale: float) -> tuple[Path, Path]:
    """Save the exact AOI and a >=128x128 edge-padded model input. No resampling."""
    directory.mkdir(parents=True, exist_ok=True)
    crop_path, model_path = directory / "crop.tif", directory / "model_input.tif"
    w = plan["window"]
    window = Window(w["x"], w["y"], w["width"], w["height"])
    with rasterio.open(path) as src:
        data = src.read(indexes, window=window, masked=True)
        validate_values(data, value_scale)
        valid = ~np.ma.getmaskarray(data).any(axis=0)
        values = np.where(valid[None], data.filled(0), 0).astype(np.float32) * (10000.0 / value_scale)
        profile = dict(driver="GTiff", count=4, dtype="float32", crs=src.crs,
                       transform=src.window_transform(window), compress="deflate")
    padded = np.pad(values, ((0, 0), (0, plan["padding_bottom"]), (0, plan["padding_right"])), mode="edge")
    padded_valid = np.pad(valid, ((0, plan["padding_bottom"]), (0, plan["padding_right"])), mode="edge")
    for target, array, mask in ((crop_path, values, valid), (model_path, padded, padded_valid)):
        with rasterio.open(target, "w", width=array.shape[2], height=array.shape[1], **profile) as dst:
            dst.write(array)
            dst.write_mask(mask.astype(np.uint8) * 255)
            dst.descriptions = BANDS
            dst.update_tags(reflectance_scale="10000", band_order=",".join(BANDS))
    return crop_path, model_path


def finish_output(raw_output: Path, crop_path: Path, output_path: Path) -> Path:
    """Remove model padding and restore the exact AOI grid and NoData footprint."""
    with rasterio.open(crop_path) as crop, rasterio.open(raw_output) as sr:
        width, height = crop.width * FACTOR, crop.height * FACTOR
        if sr.count != 4 or sr.width < width or sr.height < height:
            raise RuntimeError("Model output has unexpected dimensions or bands.")
        output = sr.read(window=Window(0, 0, width, height))
        valid = np.repeat(np.repeat(crop.dataset_mask() > 0, FACTOR, axis=0), FACTOR, axis=1)
        output[:, ~valid] = 0
        profile = dict(driver="GTiff", width=width, height=height, count=4, dtype=sr.dtypes[0],
                       crs=crop.crs, transform=crop.transform * Affine.scale(1 / FACTOR), compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output)
        dst.write_mask(valid.astype(np.uint8) * 255)
        dst.descriptions = BANDS
        dst.update_tags(model="ESA LDSR-S2", reflectance_scale="10000", product="super-resolved estimate")
    return output_path
