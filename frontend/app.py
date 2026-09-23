from __future__ import annotations
import hashlib
import time

import streamlit as st

from frontend.api_client import SRMApiClient
from frontend.components.comparison import before_after_slider
from frontend.components.map_view import select_aoi, show_result_map
from frontend.components.results import comparison_images, detail_images, get_geotiff_metadata, map_preview

st.set_page_config(page_title="Satellite SRM", page_icon="🛰️", layout="wide")
st.title("Satellite super-resolution")
st.write("Select a small area of Sentinel-2 imagery and generate a 4× super-resolved estimate with ESA LDSR-S2.")

api = SRMApiClient()
try:
    info = api.health()["model"]
except Exception as exc:
    st.error(f"The backend is unavailable or still loading the model. {exc}")
    st.code("python -m uvicorn backend.main:app --workers 1")
    st.stop()

with st.sidebar:
    st.header("Processing")
    st.write(f"**Model:** {info['model']}")
    st.write(f"**Device:** {info['device']}")
    quality_labels = {100: "Detailed — 100 steps", 50: "Balanced — 50 steps", 20: "Quick preview — 20 steps"}
    sampling_options = sorted(info.get("sampling_options", [20]), reverse=True)
    if "sampling_options" in info and not st.session_state.get("quality_settings_version"):
        st.session_state.sampling_steps = info["sampling_steps"]
        st.session_state.quality_settings_version = 1
    sampling_steps = st.selectbox("Reconstruction quality", sampling_options,
                                 format_func=lambda value: quality_labels[value], key="sampling_steps")
    if "sampling_options" not in info:
        st.warning("Restart the FastAPI server to enable the new quality settings.")
    st.write(f"**Maximum patches per AOI:** {info['max_patches']}")
    st.caption("100 steps uses five times as many denoising iterations as the old 20-step preview. Allow several minutes per patch; reconstruction quality varies with the scene.")
    with st.expander("Saved jobs"):
        if st.button("Refresh job history"):
            st.session_state.history = api.list_jobs()
        history = st.session_state.get("history", [])
        if history:
            selected = st.selectbox("Job", [job["job_id"] for job in history],
                                    format_func=lambda value: next(f"{j['filename']} · {j['status']} · {value[:8]}" for j in history if j["job_id"] == value))
            if st.button("Open job"):
                st.session_state.active_job = selected
                st.session_state.poll_started = time.monotonic()
                st.session_state.poll_enabled = True
                st.session_state.pop("products", None)
            if st.button("Delete finished job"):
                try:
                    api.delete_job(selected)
                    st.session_state.history = api.list_jobs()
                    if st.session_state.get("active_job") == selected:
                        st.session_state.pop("active_job", None)
                        st.session_state.pop("products", None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

st.subheader("1. Upload and inspect")
uploaded = st.file_uploader("Four-band Sentinel-2 L2A TIFF", type=["tif", "tiff"])
order_labels = {"auto": "Read band descriptions", "rgbn": "RGBN — B04, B03, B02, B08", "bgrn": "BGRN — B02, B03, B04, B08"}
order = st.selectbox("Input band order", list(order_labels), format_func=order_labels.get)
scale = st.selectbox("Reflectance storage", [10000, 1],
                     format_func=lambda x: "Reflectance × 10,000" if x == 10000 else "Reflectance in [0, 1]")
st.caption("For the unlabelled SEN2NAIP demo, choose RGBN. Labelled prepared files can use their band descriptions.")

fingerprint = None
if uploaded is not None:
    fingerprint = hashlib.sha256(uploaded.getbuffer()).hexdigest() + f":{order}:{scale}"
if st.session_state.get("upload_fingerprint") != fingerprint:
    # Contents and interpretation determine identity, even for identical filenames.
    for key in ("scene", "scene_preview", "active_job", "products", "poll_started"):
        st.session_state.pop(key, None)
    st.session_state.upload_fingerprint = fingerprint
    st.session_state.poll_enabled = False

if uploaded is not None and st.button("Inspect image", type="primary"):
    try:
        with st.spinner("Uploading and preparing a map preview…"):
            scene = api.upload_scene(uploaded.getvalue(), uploaded.name, order, scale)
            preview = api.preview(scene["scene_id"])
        st.session_state.scene = scene
        st.session_state.scene_preview = preview
    except Exception as exc:
        st.error(str(exc))

scene = st.session_state.get("scene")
if scene:
    for warning in scene["warnings"]:
        st.warning(warning)
    st.caption(f"{scene['width']} × {scene['height']} pixels · Full image: {scene['full_image_patches']} patches · Output bands: RGB + NIR")
    st.subheader("2. Select the area to process")
    modes = ["Center crop (up to 128×128)", "Pixel rectangle"]
    if scene["crs"]:
        modes.append("Draw on map")
    mode = st.radio("Area selection", modes, horizontal=True, key=f"mode-{scene['scene_id']}")
    aoi = {"mode": "center"}
    if mode == "Pixel rectangle":
        cols = st.columns(4)
        x = cols[0].number_input("Left column", min_value=0, max_value=scene["width"] - 1, value=0, step=1, key=f"x-{scene['scene_id']}")
        y = cols[1].number_input("Top row", min_value=0, max_value=scene["height"] - 1, value=0, step=1, key=f"y-{scene['scene_id']}")
        width = cols[2].number_input("Width (pixels)", min_value=1, max_value=scene["width"] - x, value=min(128, scene["width"] - x), step=1)
        height = cols[3].number_input("Height (pixels)", min_value=1, max_value=scene["height"] - y, value=min(128, scene["height"] - y), step=1)
        aoi = {"mode": "pixel", "x": x, "y": y, "width": width, "height": height}
    if scene["crs"]:
        default_plan = None
        if mode != "Draw on map":
            try:
                default_plan = api.plan(scene["scene_id"], aoi)
            except Exception:
                pass
        bbox = select_aoi(scene, st.session_state.scene_preview, draw=mode == "Draw on map",
                          selection_bounds=default_plan["bounds_wgs84"] if default_plan else None)
        if mode == "Draw on map":
            st.caption("Use the rectangle tool. The most recently drawn rectangle is selected; edit or delete it on the map. The crop snaps outward to source pixels.")
            aoi = {"mode": "bbox", "bbox": bbox} if bbox else None
    else:
        st.image(st.session_state.scene_preview, caption="Source preview — pixel coordinates only", width=650)

    plan = None
    if aoi:
        try:
            plan = api.plan(scene["scene_id"], aoi)
            a, b, c = st.columns(3)
            w = plan["window"]
            a.metric("Selected pixels", f"{w['width']} × {w['height']}")
            b.metric("Model patches", plan["patches"])
            c.metric("Output pixels", f"{plan['output_width']} × {plan['output_height']}")
            st.caption(f"Pixel window: column {w['x']}, row {w['y']}. {plan['full_image_patches']} full-image patches → {plan['patches']} AOI patches.")
            if plan["padding_right"] or plan["padding_bottom"]:
                st.caption("Small crops are padded to 128×128 for the model. Padding is removed from the final product.")
        except Exception as exc:
            st.error(str(exc))
    else:
        st.info("Draw a small rectangle to preview its patch count.")

    if st.button("Super-resolve selected area", disabled=plan is None or "sampling_options" not in info, type="primary"):
        try:
            job = api.submit_aoi(scene["scene_id"], aoi, sampling_steps=sampling_steps)
            st.session_state.active_job = job["job_id"]
            st.session_state.poll_started = time.monotonic()
            st.session_state.poll_enabled = True
            st.session_state.pop("products", None)
        except Exception as exc:
            st.error(str(exc))
    with st.expander("Source details"):
        st.json(scene)
        if st.button("Delete uploaded source"):
            try:
                api.delete_scene(scene["scene_id"])
                st.session_state.pop("scene", None)
                st.session_state.pop("scene_preview", None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


@st.fragment(run_every=2 if st.session_state.get("poll_enabled") else None)
def job_panel():
    job_id = st.session_state.get("active_job")
    if not job_id:
        return
    st.subheader("3. Result")
    try:
        job = api.get_job(job_id)
    except Exception as exc:
        st.warning(f"Unable to refresh job: {exc}. The server may still be processing it.")
        return
    st.caption(f"Job {job_id}")
    job_steps = job.get("sampling_steps", 20)
    st.caption(f"This result uses {job_steps} sampling steps.")
    if job_steps != sampling_steps:
        st.info("Changing quality does not update an existing result. Run the selected area again to generate a new product.")
    if job["status"] not in ("preparing", "queued", "processing") and st.session_state.get("poll_enabled"):
        st.session_state.poll_enabled = False
        st.rerun()
    if job["status"] in ("preparing", "queued", "processing"):
        started = st.session_state.setdefault("poll_started", time.monotonic())
        elapsed = time.monotonic() - started
        st.progress(job["progress"], text=job["message"])
        st.caption(f"Waiting {elapsed / 60:.1f} minutes · {job['patches_completed']} of {job['plan']['patches']} patches completed")
        if elapsed > 3600:
            st.warning("This job has taken over an hour. Its ID is saved; inspect backend logs or reopen it from job history.")
            if st.session_state.get("poll_enabled"):
                st.session_state.poll_enabled = False
                st.rerun()
        if not st.session_state.get("poll_enabled") and st.button("Resume status updates"):
            st.session_state.poll_started = time.monotonic()
            st.session_state.poll_enabled = True
            st.rerun()
        return
    if job["status"] == "failed":
        st.error(job.get("error") or "The job failed.")
        return
    st.success(f"Completed in {job.get('inference_seconds', 0):.1f} seconds of inference and output processing.")
    try:
        cached = st.session_state.get("products")
        if cached is None or cached["job_id"] != job_id:
            crop = api.download_result(job_id, "crop")
            result = api.download_result(job_id)
            before, after = comparison_images(crop, result)
            metadata = get_geotiff_metadata(result)
            mapped = map_preview(result) if metadata["crs"] else None
            cached = dict(job_id=job_id, crop=crop, result=result, before=before, after=after, metadata=metadata, mapped=mapped)
            st.session_state.products = cached
        before_after_slider(cached["before"], cached["after"], before_label="Selected original area", after_label="ESA LDSR-S2 · 4×")
        with st.expander("Inspect fine detail"):
            crop_width = cached["metadata"]["width"] // 4
            crop_height = cached["metadata"]["height"] // 4
            detail_width, detail_height = min(32, crop_width), min(32, crop_height)
            x_col, y_col = st.columns(2)
            x = x_col.number_input("Detail column", min_value=0, max_value=crop_width-detail_width,
                                   value=(crop_width-detail_width)//2, key=f"detail-x-{job_id}")
            y = y_col.number_input("Detail row", min_value=0, max_value=crop_height-detail_height,
                                   value=(crop_height-detail_height)//2, key=f"detail-y-{job_id}")
            baseline, detail = detail_images(cached["crop"], cached["result"], x, y, detail_width, detail_height)
            left_detail, right_detail = st.columns(2)
            left_detail.image(baseline, caption="Original · bicubic enlarged", width=baseline.width)
            right_detail.image(detail, caption=f"LDSR-S2 · {job_steps} steps", width=detail.width)
            st.caption("Matching area and contrast. Each SR pixel is displayed at 2× size without smoothing; no sharpening filter is applied.")
        left, right = st.columns(2)
        left.download_button("Download SR GeoTIFF", cached["result"], file_name=f"{job_id}_sr.tif", mime="image/tiff")
        right.download_button("Download original AOI", cached["crop"], file_name=f"{job_id}_crop.tif", mime="image/tiff")
        if cached["mapped"]:
            image, bounds = cached["mapped"]
            show_result_map(bounds, image, job_id)
            st.caption("Input: 10 m · Output grid: 2.5 m · Geographic extent preserved.")
        else:
            st.info("This TIFF has no CRS. The output has 4× more pixels; its ground resolution and location remain unknown.")
        st.caption("Super-resolution is a learned estimate, not a native 2.5 m observation. Added detail is not proof of increased accuracy.")
        with st.expander("Product and processing metadata"):
            st.json(cached["metadata"])
            st.json(job)
    except Exception as exc:
        st.error(f"Could not display the result: {exc}")


job_panel()
