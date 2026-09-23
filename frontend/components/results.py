from __future__ import annotations
from io import BytesIO
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window

from inference.preview import rgb_image, create_preview


def read_rgb(data: bytes):
    with rasterio.MemoryFile(data) as mem, mem.open() as src:
        ratio = min(1, 1024 / max(src.width, src.height))
        return src.read([1, 2, 3], out_shape=(3, max(1, round(src.height * ratio)), max(1, round(src.width * ratio))),
                        masked=True, resampling=Resampling.bilinear)


def geotiff_to_rgb(geotiff_bytes: bytes, scale=10000):
    return rgb_image(read_rgb(geotiff_bytes))[0]


def comparison_images(crop_bytes, sr_bytes):
    # A shared stretch avoids presenting different contrast as improved detail.
    original, limits = rgb_image(read_rgb(crop_bytes))
    result, _ = rgb_image(read_rgb(sr_bytes), limits)
    return original, result


def detail_images(crop_bytes, sr_bytes, x, y, width, height):
    """Show the same native SR pixels beside an explicitly bicubic LR baseline."""
    _, limits = rgb_image(read_rgb(crop_bytes))
    with rasterio.MemoryFile(crop_bytes) as mem, mem.open() as src:
        if x < 0 or y < 0 or width < 1 or height < 1 or x + width > src.width or y + height > src.height:
            raise ValueError("Detail window is outside the crop.")
        baseline = src.read([1, 2, 3], window=Window(x, y, width, height),
                            out_shape=(3, height * 4, width * 4), masked=True,
                            resampling=Resampling.cubic)
    with rasterio.MemoryFile(sr_bytes) as mem, mem.open() as src:
        detail = src.read([1, 2, 3], window=Window(x * 4, y * 4, width * 4, height * 4), masked=True)
    images = [rgb_image(data, limits)[0] for data in (baseline, detail)]
    # Integer magnification preserves the displayed pixels without smoothing.
    return tuple(image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST) for image in images)


def get_geotiff_metadata(geotiff_bytes: bytes):
    with rasterio.MemoryFile(geotiff_bytes) as mem, mem.open() as src:
        return dict(width=src.width, height=src.height, bands=src.count, crs=str(src.crs) if src.crs else None,
                    resolution_x=src.res[0], resolution_y=src.res[1], dtype=src.dtypes[0],
                    bounds=dict(zip(("left", "bottom", "right", "top"), src.bounds)))


def map_preview(geotiff_bytes):
    buffer = BytesIO()
    with rasterio.MemoryFile(geotiff_bytes) as mem:
        metadata = create_preview(mem.name, buffer, [1, 2, 3, 4])
    buffer.seek(0)
    return Image.open(buffer).copy(), metadata["preview_bounds_wgs84"]
