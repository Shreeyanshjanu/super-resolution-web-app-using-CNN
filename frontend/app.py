from __future__ import annotations

import time

import streamlit as st

from frontend.api_client import SRMApiClient
from frontend.components.comparison import before_after_slider
from frontend.components.map_view import show_map
from frontend.components.results import (
    geotiff_to_rgb,
    get_geotiff_metadata,
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Satellite SRM",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API CLIENT
# ============================================================

@st.cache_resource
def get_api_client() -> SRMApiClient:
    return SRMApiClient(
        base_url=API_URL
    )


api = get_api_client()


# ============================================================
# SESSION STATE
# ============================================================

if "job_id" not in st.session_state:
    st.session_state.job_id = None

if "completed_job" not in st.session_state:
    st.session_state.completed_job = None

if "output_bytes" not in st.session_state:
    st.session_state.output_bytes = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ Satellite Super Resolution Mapping</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Deep Learning-Based Super Resolution for Sentinel-2 Imagery
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Transform 10 m Sentinel-2 RGB-NIR imagery into a "
    "2.5 m super-resolved estimate using the pretrained "
    "ESA LDSR-S2 model."
)


# ============================================================
# BACKEND CHECK
# ============================================================

try:

    health = api.health()

    if health.get("status") != "ok":

        st.error(
            "FastAPI is running, but the SRM service is not healthy."
        )

        st.stop()

    model_info = health["model"]

except Exception as exc:

    st.error(
        "❌ Could not connect to the FastAPI backend."
    )

    st.code(
        str(exc)
    )

    st.info(
        "Start FastAPI with:\n\n"
        "python -m uvicorn backend.main:app --reload"
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Model Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        model_info.get(
            "model",
            "ESA LDSR-S2",
        ),
    )

with col2:
    st.metric(
        "Input",
        f"{model_info.get('input_resolution_m', 10)} m",
    )

with col3:
    st.metric(
        "Output",
        f"{model_info.get('output_resolution_m', 2.5)} m",
    )

with col4:
    st.metric(
        "Scale",
        f"{model_info.get('scale_factor', 4)}×",
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write(
        f"**Backend:** `{API_URL}`"
    )

    st.write(
        f"**Device:** "
        f"`{model_info.get('device', 'unknown')}`"
    )

    st.write(
        "**Input:** Sentinel-2 L2A"
    )

    st.write(
        "**Bands:** B02, B03, B04, B08"
    )

    st.write(
        "**Target:** 2.5 m"
    )

    st.divider()

    st.info(
        "Current model: ESA LDSR-S2 pretrained model."
    )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Sentinel-2 GeoTIFF</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a 4-band Sentinel-2 L2A GeoTIFF",
    type=[
        "tif",
        "tiff",
    ],
    help=(
        "Expected bands: B02, B03, B04 and B08 "
        "at 10 m resolution."
    ),
)


# ============================================================
# FILE HANDLING
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()


    # --------------------------------------------------------
    # Detect new upload
    # --------------------------------------------------------

    if (
        st.session_state.uploaded_filename
        != uploaded_file.name
    ):

        st.session_state.job_id = None
        st.session_state.completed_job = None
        st.session_state.output_bytes = None
        st.session_state.uploaded_filename = (
            uploaded_file.name
        )

    st.session_state.uploaded_bytes = file_bytes


    st.success(
        f"✅ Loaded: {uploaded_file.name}"
    )


    file_size_mb = (
        len(file_bytes)
        / (1024 * 1024)
    )

    st.write(
        f"File size: **{file_size_mb:.2f} MB**"
    )


    # ========================================================
    # INPUT PREVIEW
    # ========================================================

    try:

        original_image = geotiff_to_rgb(
            file_bytes
        )

        st.subheader(
            "Input Preview"
        )

        st.image(
            original_image,
            caption="Sentinel-2 Input — 10 m",
            use_container_width=True,
        )

    except Exception as exc:

        original_image = None

        st.warning(
            "Could not generate an RGB preview."
        )

        st.code(
            str(exc)
        )


    st.divider()


    # ========================================================
    # RUN BUTTON
    # ========================================================

    if st.button(
        "🚀 Run Super Resolution",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.job_id = None
        st.session_state.completed_job = None
        st.session_state.output_bytes = None


        try:

            # ------------------------------------------------
            # Upload
            # ------------------------------------------------

            with st.spinner(
                "Uploading Sentinel-2 image..."
            ):

                response = api.submit_file(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                )


            job_id = response["job_id"]

            st.session_state.job_id = job_id


            # ------------------------------------------------
            # Processing
            # ------------------------------------------------

            st.subheader(
                "Processing"
            )

            progress_bar = st.progress(
                5
            )

            status_text = st.empty()


            while True:

                job = api.get_job(
                    job_id
                )

                status = job.get(
                    "status",
                    "unknown",
                )


                if status == "queued":

                    progress_bar.progress(
                        5
                    )

                    status_text.info(
                        "⏳ Job queued..."
                    )


                elif status == "processing":

                    progress_bar.progress(
                        10
                    )

                    status_text.info(
                        "🔄 Generating the 2.5 m "
                        "super-resolved product..."
                    )


                elif status == "completed":

                    progress_bar.progress(
                        100
                    )

                    status_text.success(
                        "✅ Super resolution completed."
                    )

                    st.session_state.completed_job = job

                    break


                elif status == "failed":

                    progress_bar.progress(
                        100
                    )

                    status_text.error(
                        "❌ Super-resolution failed."
                    )

                    st.error(
                        job.get(
                            "error",
                            "Unknown error.",
                        )
                    )

                    break


                else:

                    status_text.warning(
                        f"Unexpected status: {status}"
                    )


                time.sleep(2)


        except Exception as exc:

            st.error(
                "❌ Super-resolution request failed."
            )

            st.exception(
                exc
            )


# ============================================================
# RESULT
# ============================================================

job = st.session_state.get(
    "completed_job"
)


if job is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">2. Super-Resolution Result</div>',
        unsafe_allow_html=True,
    )


    try:

        # ====================================================
        # DOWNLOAD SR PRODUCT
        # ====================================================

        if (
            st.session_state.output_bytes
            is None
        ):

            with st.spinner(
                "Downloading SR GeoTIFF..."
            ):

                output_bytes = (
                    api.download_result(
                        job["job_id"]
                    )
                )

            st.session_state.output_bytes = (
                output_bytes
            )

        else:

            output_bytes = (
                st.session_state.output_bytes
            )


        # ====================================================
        # LOAD INPUT AND OUTPUT
        # ====================================================

        input_bytes = (
            st.session_state.uploaded_bytes
        )

        if input_bytes is None:

            raise RuntimeError(
                "Original uploaded image is no longer available."
            )


        original_image = geotiff_to_rgb(
            input_bytes
        )

        sr_image = geotiff_to_rgb(
            output_bytes
        )


        # ====================================================
        # BEFORE / AFTER
        # ====================================================

        st.subheader(
            "Before / After Comparison"
        )

        st.caption(
            "Drag the slider left and right to compare "
            "the original 10 m image with the "
            "2.5 m super-resolved estimate."
        )

        before_after_slider(
            before=original_image,
            after=sr_image,
            before_label="Original — 10 m",
            after_label="ESA LDSR-S2 — 2.5 m",
        )


        # ====================================================
        # PRODUCT INFORMATION
        # ====================================================

        st.subheader(
            "Product Information"
        )

        input_metadata = get_geotiff_metadata(
            input_bytes
        )

        output_metadata = get_geotiff_metadata(
            output_bytes
        )


        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Input Resolution",
                f"{input_metadata['resolution_x']:.1f} m",
            )

        with m2:

            st.metric(
                "Output Resolution",
                f"{output_metadata['resolution_x']:.1f} m",
            )

        with m3:

            st.metric(
                "Scale Factor",
                "4×",
            )

        with m4:

            st.metric(
                "Bands",
                str(output_metadata["bands"]),
            )


        # ====================================================
        # GEOGRAPHIC MAP
        # ====================================================

        st.subheader(
            "3. Geographic View"
        )

        st.caption(
            "The super-resolved product is displayed at its "
            "actual geographic location using the GeoTIFF "
            "coordinate reference system and bounds."
        )

        try:

            map_data = show_map(
                metadata=output_metadata,
                rgb_image=sr_image,
                layer_name="ESA LDSR-S2 — 2.5 m",
                height=600,
            )

            if map_data:

                clicked = map_data.get(
                    "last_clicked"
                )

                if clicked:

                    st.caption(
                        f"Clicked location: "
                        f"{clicked['lat']:.6f}, "
                        f"{clicked['lng']:.6f}"
                    )

        except Exception as map_exc:

            st.warning(
                "Could not display the geographic map."
            )

            st.code(
                str(map_exc)
            )


        # ====================================================
        # DETAILED METADATA
        # ====================================================

        with st.expander(
            "🔍 Detailed GeoTIFF Metadata"
        ):

            metadata_col1, metadata_col2 = (
                st.columns(2)
            )


            with metadata_col1:

                st.markdown(
                    "**Input Product**"
                )

                st.write(
                    f"Dimensions: "
                    f"{input_metadata['width']} × "
                    f"{input_metadata['height']}"
                )

                st.write(
                    f"Resolution: "
                    f"{input_metadata['resolution_x']:.2f} × "
                    f"{input_metadata['resolution_y']:.2f} m"
                )

                st.write(
                    f"Bands: "
                    f"{input_metadata['bands']}"
                )

                st.write(
                    f"CRS: "
                    f"{input_metadata['crs']}"
                )

                st.write(
                    f"Data type: "
                    f"{input_metadata['dtype']}"
                )


            with metadata_col2:

                st.markdown(
                    "**SR Product**"
                )

                st.write(
                    f"Dimensions: "
                    f"{output_metadata['width']} × "
                    f"{output_metadata['height']}"
                )

                st.write(
                    f"Resolution: "
                    f"{output_metadata['resolution_x']:.2f} × "
                    f"{output_metadata['resolution_y']:.2f} m"
                )

                st.write(
                    f"Bands: "
                    f"{output_metadata['bands']}"
                )

                st.write(
                    f"CRS: "
                    f"{output_metadata['crs']}"
                )

                st.write(
                    f"Data type: "
                    f"{output_metadata['dtype']}"
                )


        # ====================================================
        # SCIENTIFIC NOTE
        # ====================================================

        st.info(
            "The 2.5 m product is a learned super-resolution "
            "estimate generated from 10 m Sentinel-2 "
            "RGB-NIR imagery. It should not be interpreted "
            "as native 2.5 m satellite observation."
        )


        # ====================================================
        # PROCESSING DETAILS
        # ====================================================

        with st.expander(
            "⚙️ Processing Details"
        ):

            st.write(
                f"**Job ID:** "
                f"`{job.get('job_id', 'N/A')}`"
            )

            st.write(
                f"**Status:** "
                f"`{job.get('status', 'N/A')}`"
            )

            st.write(
                f"**Input file:** "
                f"`{job.get('filename', 'N/A')}`"
            )

            st.write(
                f"**Device:** "
                f"`{model_info.get('device', 'N/A')}`"
            )

            st.write(
                "**Model:** ESA LDSR-S2"
            )

            st.write(
                "**Bands:** B02, B03, B04, B08"
            )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.subheader(
            "Download"
        )

        output_filename = (
            f"{job['job_id']}_sr_2.5m.tif"
        )

        st.download_button(
            label="⬇️ Download 2.5 m GeoTIFF",
            data=output_bytes,
            file_name=output_filename,
            mime="image/tiff",
            type="primary",
            use_container_width=True,
        )


    except Exception as exc:

        st.error(
            "❌ Could not display the generated result."
        )

        st.exception(
            exc
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Satellite SRM • Sentinel-2 10 m → 2.5 m • "
    "ESA LDSR-S2 pretrained model"
)