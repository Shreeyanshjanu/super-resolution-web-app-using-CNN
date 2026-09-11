from __future__ import annotations

from typing import Optional

import folium
import numpy as np
from pyproj import Transformer
from streamlit_folium import st_folium


def _transform_bounds_to_wgs84(
    bounds: dict,
    source_crs: str,
) -> tuple[float, float, float, float]:
    """
    Convert projected GeoTIFF bounds into WGS84.

    Returns:
        (south, west, north, east)
    """

    if not source_crs:
        raise ValueError(
            "GeoTIFF does not contain a CRS."
        )

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True,
    )

    left = bounds["left"]
    bottom = bounds["bottom"]
    right = bounds["right"]
    top = bounds["top"]

    corners = [
        transformer.transform(left, bottom),
        transformer.transform(left, top),
        transformer.transform(right, bottom),
        transformer.transform(right, top),
    ]

    longitudes = [
        point[0]
        for point in corners
    ]

    latitudes = [
        point[1]
        for point in corners
    ]

    west = min(longitudes)
    east = max(longitudes)

    south = min(latitudes)
    north = max(latitudes)

    return south, west, north, east


def create_map(
    metadata: dict,
    rgb_image,
    layer_name: str = "Super-Resolved Image",
) -> folium.Map:
    """
    Create an interactive Folium map with the SR image
    positioned using its actual GeoTIFF bounds.
    """

    crs = metadata.get("crs")

    if not crs:
        raise ValueError(
            "The GeoTIFF has no CRS."
        )

    bounds = metadata.get(
        "bounds"
    )

    if not bounds:
        raise ValueError(
            "The GeoTIFF has no bounds."
        )

    (
        south,
        west,
        north,
        east,
    ) = _transform_bounds_to_wgs84(
        bounds=bounds,
        source_crs=crs,
    )

    center_lat = (
        south + north
    ) / 2

    center_lon = (
        west + east
    ) / 2

    # Convert PIL image to RGB NumPy array.
    image_array = np.asarray(
        rgb_image.convert("RGB")
    )

    # --------------------------------------------------------
    # Create map
    # --------------------------------------------------------

    fmap = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=14,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    # --------------------------------------------------------
    # SR image overlay
    # --------------------------------------------------------

    folium.raster_layers.ImageOverlay(
        image=image_array,
        bounds=[
            [south, west],
            [north, east],
        ],
        opacity=0.85,
        interactive=True,
        cross_origin=False,
        zindex=2,
        name=layer_name,
    ).add_to(fmap)

    # --------------------------------------------------------
    # Footprint
    # --------------------------------------------------------

    footprint = [
        [south, west],
        [south, east],
        [north, east],
        [north, west],
    ]

    folium.Polygon(
        locations=footprint,
        weight=2,
        fill=False,
        popup=(
            f"<b>{layer_name}</b><br>"
            f"CRS: {crs}<br>"
            f"Resolution: "
            f"{metadata['resolution_x']:.2f} m"
        ),
        tooltip="Processed image footprint",
    ).add_to(fmap)

    # --------------------------------------------------------
    # Center marker
    # --------------------------------------------------------

    folium.Marker(
        location=[
            center_lat,
            center_lon,
        ],
        popup=(
            f"<b>SR Product</b><br>"
            f"Resolution: "
            f"{metadata['resolution_x']:.2f} m<br>"
            f"CRS: {crs}"
        ),
        tooltip="Super-resolution product",
    ).add_to(fmap)

    # --------------------------------------------------------
    # Layer control
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(fmap)

    # --------------------------------------------------------
    # Fit map to product
    # --------------------------------------------------------

    fmap.fit_bounds(
        [
            [south, west],
            [north, east],
        ]
    )

    return fmap


def show_map(
    metadata: dict,
    rgb_image,
    layer_name: str = "Super-Resolved Image",
    height: int = 600,
) -> dict:
    """
    Render the interactive map inside Streamlit.
    """

    fmap = create_map(
        metadata=metadata,
        rgb_image=rgb_image,
        layer_name=layer_name,
    )

    return st_folium(
        fmap,
        width=None,
        height=height,
        returned_objects=[
            "last_clicked",
            "bounds",
        ],
    )


__all__ = [
    "create_map",
    "show_map",
]