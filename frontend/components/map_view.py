from __future__ import annotations
from io import BytesIO
import folium
import numpy as np
from folium.plugins import Draw
from PIL import Image
from streamlit_folium import st_folium


def create_map(bounds, image, draw=False, selection_bounds=None):
    west, south, east, north = bounds
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], tiles="OpenStreetMap", control_scale=True)
    folium.raster_layers.ImageOverlay(np.asarray(image), bounds=[[south, west], [north, east]],
                                     name="Satellite image", opacity=1).add_to(fmap)
    folium.Rectangle([[south, west], [north, east]], color="#64748b", weight=1, fill=False,
                     tooltip="Image extent").add_to(fmap)
    if selection_bounds:
        w, s, e, n = selection_bounds
        folium.Rectangle([[s, w], [n, e]], color="#22c55e", fill=False, weight=2,
                         tooltip="Selected pixel extent").add_to(fmap)
    if draw:
        Draw(export=False, draw_options={"polyline": False, "polygon": False, "circle": False,
             "marker": False, "circlemarker": False, "rectangle": {"shapeOptions": {"color": "#22c55e"}}},
             edit_options={"edit": True, "remove": True}).add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])
    folium.LayerControl().add_to(fmap)
    return fmap


def drawing_bbox(drawings):
    if not drawings:
        return None
    geometry = drawings[-1].get("geometry", {})
    if geometry.get("type") != "Polygon":
        return None
    points = geometry["coordinates"][0]
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def select_aoi(scene, preview_bytes, draw=False, selection_bounds=None):
    image = Image.open(BytesIO(preview_bytes)).convert("RGBA")
    fmap = create_map(scene["preview_bounds_wgs84"], image, draw, selection_bounds)
    result = st_folium(fmap, key=f"scene-map-{scene['scene_id']}-{draw}", height=500,
                       use_container_width=True, returned_objects=["all_drawings"] if draw else [])
    return drawing_bbox(result.get("all_drawings")) if result else None


def show_result_map(bounds, image, job_id):
    st_folium(create_map(bounds, image), key=f"result-map-{job_id}", height=450,
              use_container_width=True, returned_objects=[])
