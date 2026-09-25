# Evaluation dashboard: research-grade analysis tabs shown after SR job completion.
from __future__ import annotations

import io
import json

import numpy as np
import streamlit as st
from PIL import Image
from rasterio.enums import Resampling

import rasterio

BAND_NAMES = ["B04 — Red", "B03 — Green", "B02 — Blue", "B08 — NIR"]
BAND_COLORS = ["#d32f2f", "#388e3c", "#1976d2", "#7b1fa2"]


# Read a GeoTIFF from bytes and return masked float32 array [bands, H, W].
def _read_array(geotiff_bytes: bytes) -> np.ndarray:
    with rasterio.MemoryFile(geotiff_bytes) as mem, mem.open() as src:
        return src.read(masked=True).astype(np.float32)


# Render an array as an RGB PNG for display (percentile stretch, shared limits).
def _array_to_png(data: np.ndarray, limits=None) -> tuple[bytes, tuple]:
    values = np.ma.asarray(data[:3], dtype=np.float32)
    valid = ~np.ma.getmaskarray(values).any(axis=0)
    filled = values.filled(np.nan)
    valid &= np.isfinite(filled).all(axis=0)
    if limits is None:
        samples = filled[:, valid]
        if not samples.size:
            raise ValueError("No valid pixels.")
        low, high = np.percentile(samples, (2, 98))
        limits = (float(low), float(max(high, low + 1e-6)))
    low, high = limits
    rgb = np.clip((np.nan_to_num(filled) - low) / max(high - low, 1e-6), 0, 1)
    rgba = np.concatenate([
        (rgb.transpose(1, 2, 0) * 255).astype(np.uint8),
        (valid * 255).astype(np.uint8)[..., None],
    ], axis=2)
    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), limits


# Render the Overview tab with input/output/processing metrics.
def _render_overview(evaluation: dict):
    ov = evaluation["overview"]
    st.subheader("Overview")

    a, b, c = st.columns(3)
    with a:
        st.markdown("#### Input (10 m)")
        inp = ov["input"]
        st.metric("Dimensions", f"{inp['width']} × {inp['height']}")
        st.metric("Bands", inp["bands"])
        st.metric("Resolution", f"{inp['resolution_x']} m")
        st.caption(f"CRS: {inp['crs'] or 'None'}")
        st.caption(f"Type: {inp['dtype']}")
    with b:
        st.markdown("#### Output (2.5 m grid)")
        out = ov["output"]
        st.metric("Dimensions", f"{out['width']} × {out['height']}")
        st.metric("Bands", out["bands"])
        st.metric("Resolution", f"{out['resolution_x']} m")
        st.caption(f"CRS: {out['crs'] or 'None'}")
        st.caption(f"Type: {out['dtype']}")
    with c:
        st.markdown("#### Processing")
        sr = ov["super_resolution"]
        st.metric("Scale factor", f"{sr['scale_factor']}×")
        st.metric("Sampling steps", sr["sampling_steps"])
        st.metric("Patches", sr["patches"])
        st.metric("Inference time", f"{sr['inference_seconds']} s")
        st.caption(f"{sr['seconds_per_patch']} s/patch")

    st.info("Physical geographic coverage is preserved. The output has 4× more pixels on a 2.5 m grid.")


# Render the Image Comparison tab with Original / Bicubic / LDSR-S2 side by side.
def _render_comparison(crop_bytes: bytes, sr_bytes: bytes):
    st.subheader("Image Comparison")
    crop_data = _read_array(crop_bytes)
    sr_data = _read_array(sr_bytes)

    from backend.services.evaluation_service import bicubic_upscale, VALUE_SCALE
    bicubic = bicubic_upscale(crop_data / VALUE_SCALE, sr_data.shape[1], sr_data.shape[2])
    sr_norm = sr_data / VALUE_SCALE

    _, limits = _array_to_png(crop_data / VALUE_SCALE)
    orig_png, _ = _array_to_png(crop_data / VALUE_SCALE, limits)
    bicubic_png, _ = _array_to_png(bicubic, limits)
    sr_png, _ = _array_to_png(sr_norm, limits)

    view_mode = st.radio("Visualization", ["RGB composite", "NIR band"], horizontal=True, key="comp-view")

    if view_mode == "RGB composite":
        a, b, c = st.columns(3)
        a.image(orig_png, caption="Original · 10 m", use_container_width=True)
        b.image(bicubic_png, caption="Bicubic · 4× interpolation", use_container_width=True)
        c.image(sr_png, caption="LDSR-S2 · 4× super-resolved estimate", use_container_width=True)
        st.caption("All three images use the same contrast stretch (2nd–98th percentile from the original). "
                    "LDSR-S2 output is a learned estimate on a 2.5 m output grid, not native 2.5 m imagery.")
    else:
        nir_crop = crop_data[3:4] / VALUE_SCALE
        nir_sr = sr_data[3:4] / VALUE_SCALE
        from backend.services.evaluation_service import bicubic_upscale as _bu
        nir_bicubic = _bu(nir_crop, sr_data.shape[1], sr_data.shape[2])
        nir_limits = None
        nir_orig_png, nir_limits = _array_to_png(np.ma.concatenate([nir_crop, nir_crop, nir_crop], axis=0))
        nir_bic_png, _ = _array_to_png(np.ma.concatenate([nir_bicubic, nir_bicubic, nir_bicubic], axis=0), nir_limits)
        nir_sr_png, _ = _array_to_png(np.ma.concatenate([nir_sr, nir_sr, nir_sr], axis=0), nir_limits)
        a, b, c = st.columns(3)
        a.image(nir_orig_png, caption="Original NIR · 10 m", use_container_width=True)
        b.image(nir_bic_png, caption="Bicubic NIR · 4× interpolation", use_container_width=True)
        c.image(nir_sr_png, caption="LDSR-S2 NIR · 4× estimate", use_container_width=True)
        st.caption("NIR band (B08) shown as grayscale for all three. Same contrast limits applied.")


# Render the Quantitative Metrics tab.
def _render_metrics(evaluation: dict):
    st.subheader("Quantitative Metrics")
    if not evaluation["reference_available"]:
        st.info("Reference ground truth unavailable — quantitative reconstruction metrics are not computed for this image.")
        st.caption("PSNR, SSIM, RMSE and MAE require a valid high-resolution reference image (e.g. NAIP for SEN2NAIP). "
                    "When no reference is available, these metrics cannot be calculated.")
        return

    ba = evaluation["band_analysis"]
    rows = []
    for b in ba:
        rm = b["reference_metrics"]
        rows.append({
            "Band": b["name"],
            "PSNR (dB)": f"{rm['psnr']:.2f}" if rm["psnr"] is not None else "N/A",
            "SSIM": f"{rm['ssim']:.4f}" if rm["ssim"] is not None else "N/A",
            "RMSE": f"{rm['rmse']:.6f}" if rm["rmse"] is not None else "N/A",
            "MAE": f"{rm['mae']:.6f}" if rm["mae"] is not None else "N/A",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# Render the Band-wise Analysis tab.
def _render_band_analysis(evaluation: dict):
    st.subheader("Band-wise Analysis")
    ba = evaluation["band_analysis"]

    rows = []
    for b in ba:
        s = b["output_stats"]
        rows.append({
            "Band": b["name"],
            "Mean": f"{s['mean']:.4f}" if s["mean"] is not None else "N/A",
            "Std": f"{s['std']:.4f}" if s["std"] is not None else "N/A",
            "Min": f"{s['min']:.4f}" if s["min"] is not None else "N/A",
            "Max": f"{s['max']:.4f}" if s["max"] is not None else "N/A",
            "RMSE": "N/A — no reference",
            "MAE": "N/A — no reference",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("#### Per-band reflectance distributions")
    spectral = evaluation["spectral"]
    for b_idx in range(4):
        with st.expander(f"{BAND_NAMES[b_idx]}"):
            inp_h = spectral["input"]["per_band"][b_idx]
            out_h = spectral["output"]["per_band"][b_idx]
            if inp_h["counts"] and inp_h["bin_edges"]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input**")
                    st.bar_chart(
                        {"count": inp_h["counts"]},
                        x_label="Reflectance bin",
                        y_label="Pixel count",
                    )
                    st.caption(f"Mean: {inp_h['mean']:.4f} · Std: {inp_h['std']:.4f}")
                with col2:
                    st.markdown("**SR Output**")
                    st.bar_chart(
                        {"count": out_h["counts"]},
                        x_label="Reflectance bin",
                        y_label="Pixel count",
                    )
                    st.caption(f"Mean: {out_h['mean']:.4f} · Std: {out_h['std']:.4f}")


# Render the Error/Difference Maps tab.
def _render_error_maps(evaluation: dict):
    st.subheader("Error / Difference Maps")
    if not evaluation["reference_available"]:
        st.info("Reference-based residual analysis requires ground truth.")
        st.caption("Absolute error maps, squared error maps and error heatmaps compare the SR output against a "
                    "high-resolution reference. Without a valid reference, these maps cannot be generated.")
        return

    st.caption("Error maps would show per-pixel differences between the SR output and the HR reference.")


# Render the Spectral Analysis tab.
def _render_spectral(evaluation: dict):
    st.subheader("Spectral Analysis")
    sp = evaluation["spectral"]

    a, b = st.columns(2)
    with a:
        st.markdown("**Input statistics**")
        st.metric("Overall mean reflectance", f"{sp['input']['overall_mean']:.4f}")
        st.metric("Overall std", f"{sp['input']['overall_std']:.4f}")
    with b:
        st.markdown("**SR output statistics**")
        st.metric("Overall mean reflectance", f"{sp['output']['overall_mean']:.4f}")
        st.metric("Overall std", f"{sp['output']['overall_std']:.4f}")

    st.markdown("#### Per-band spectral error (mean reflectance shift)")
    se = sp["spectral_error"]
    for b_idx in range(4):
        err = se["per_band_mean"][b_idx]
        if err is not None:
            st.markdown(f"**{BAND_NAMES[b_idx]}:** {err:.6f}")

    st.caption(se["note"])

    st.markdown("#### Reflectance distribution comparison")
    for b_idx in range(4):
        with st.expander(f"{BAND_NAMES[b_idx]} histogram"):
            inp_h = sp["input"]["per_band"][b_idx]
            out_h = sp["output"]["per_band"][b_idx]
            if inp_h["counts"] and inp_h["bin_edges"]:
                st.bar_chart(
                    {"Input": inp_h["counts"], "SR Output": out_h["counts"]},
                    x_label="Reflectance bin",
                    y_label="Pixel count",
                )


# Render the Uncertainty/Confidence tab.
def _render_uncertainty(evaluation: dict):
    st.subheader("Uncertainty / Confidence")
    u = evaluation["uncertainty"]
    if not u["available"]:
        st.info(u["message"])
        with st.expander("Future extension"):
            st.caption("Per-pixel uncertainty estimation could be added via ensemble inference or "
                        "Monte Carlo dropout. This is not currently implemented in the ESA LDSR-S2 pipeline.")
    else:
        st.caption("Uncertainty map would be displayed here.")


# Render the Spatial Detail Analysis tab.
def _render_spatial(evaluation: dict):
    st.subheader("Spatial Detail Analysis")
    sd = evaluation["spatial_detail"]

    st.markdown("#### Comparison of spatial metrics")
    rows = []
    for label, key in [("Original (10 m)", "input"), ("Bicubic 4×", "bicubic"), ("LDSR-S2", "output")]:
        metrics = sd[key]
        avg_grad = np.mean([m["mean_gradient"] for m in metrics])
        avg_edge = np.mean([m["edge_density"] for m in metrics])
        avg_hfe = np.mean([m["high_frequency_energy"] for m in metrics])
        avg_contrast = np.mean([m["local_contrast"] for m in metrics])
        rows.append({
            "Method": label,
            "Avg gradient": f"{avg_grad:.6f}",
            "Edge density": f"{avg_edge:.6f}",
            "HF energy": f"{avg_hfe:.6f}",
            "Local contrast": f"{avg_contrast:.4f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.warning(sd["note"])

    with st.expander("Per-band spatial detail"):
        for b_idx in range(4):
            st.markdown(f"**{BAND_NAMES[b_idx]}**")
            cols = st.columns(3)
            for col, (label, key) in zip(cols, [("Original", "input"), ("Bicubic", "bicubic"), ("LDSR-S2", "output")]):
                m = sd[key][b_idx]
                with col:
                    st.caption(label)
                    st.metric("Gradient", f"{m['mean_gradient']:.6f}")
                    st.metric("HF energy", f"{m['high_frequency_energy']:.6f}")


# Render the Model Information tab.
def _render_model_info(evaluation: dict):
    st.subheader("Model Information")
    mi = evaluation["model_info"]

    a, b = st.columns(2)
    with a:
        st.markdown("| | |")
        st.markdown("|---|---|")
        st.markdown(f"| **Model** | {mi['model']} |")
        st.markdown(f"| **Architecture** | {mi['architecture']} |")
        st.markdown(f"| **Components** | {', '.join(mi['components'])} |")
        st.markdown(f"| **Parameters** | {mi['approximate_parameters']} |")
        st.markdown(f"| **Sampling method** | {mi['sampling_method']} |")
    with b:
        st.markdown("| | |")
        st.markdown("|---|---|")
        st.markdown(f"| **Input** | {mi['input']} |")
        st.markdown(f"| **Band order** | {mi['input_band_order']} |")
        st.markdown(f"| **Scale factor** | {mi['scale_factor']} |")
        st.markdown(f"| **Input resolution** | {mi['input_resolution']} |")
        st.markdown(f"| **Output grid** | {mi['output_grid']} |")
        st.markdown(f"| **Latent representation** | {mi['latent_representation']} |")
        st.markdown(f"| **Target output** | {mi['target_output']} |")

    st.caption(f"Sampling steps used for this job: **{mi['sampling_steps_used']}** "
                f"(options: {', '.join(str(s) for s in mi['sampling_steps_options'])})")
    st.caption("Sampling steps are denoising iterations, not neural-network layers. "
                "More steps increase computational cost but do not automatically improve accuracy.")


# Render the Dataset Information tab.
def _render_dataset_info(evaluation: dict):
    st.subheader("Dataset Information")
    di = evaluation["dataset_info"]
    a, b = st.columns(2)
    with a:
        st.markdown(f"| | |")
        st.markdown(f"|---|---|")
        st.markdown(f"| **Dataset** | {di['dataset']} |")
        st.markdown(f"| **Purpose** | {di['purpose']} |")
        st.markdown(f"| **Bands** | {di['bands']} |")
    with b:
        st.markdown(f"| | |")
        st.markdown(f"|---|---|")
        st.markdown(f"| **Input source** | {di['input_source']} |")
        st.markdown(f"| **HR reference** | {di['hr_reference']} |")
        st.markdown(f"| **Publication year** | {di['publication_year']} |")
    st.caption(di["note"])


# Render the Limitations card.
def _render_limitations(evaluation: dict):
    st.subheader("Limitations & Scientific Caveats")
    for i, lim in enumerate(evaluation["limitations"], 1):
        st.markdown(f"{i}. {lim}")


# Render the Patch Information section.
def _render_patch_info(evaluation: dict):
    st.subheader("Patch Information")
    pi = evaluation["patch_info"]
    a, b, c, d = st.columns(4)
    a.metric("Patch size", f"{pi['patch_size']} × {pi['patch_size']}")
    b.metric("Overlap", f"{pi['overlap']} px")
    c.metric("Patches used", pi["patches_used"])
    d.metric("Output", pi["output_dimensions"])

    st.markdown("```")
    st.markdown(f"128 × 128 LR patch  →  512 × 512 SR patch  ({pi['scale']}×)")
    st.markdown("```")


# Render the Georeferencing section.
def _render_georeferencing(evaluation: dict):
    st.subheader("Georeferencing")
    g = evaluation["georeferencing"]
    a, b = st.columns(2)
    with a:
        st.metric("CRS", g["crs"] or "None")
        st.metric("Resolution", f"{g['resolution_x']} × {g['resolution_y']} m")
        st.metric("Dimensions", f"{g['width']} × {g['height']}")
    with b:
        bounds = g["bounds"]
        st.metric("Left", f"{bounds['left']:.2f}")
        st.metric("Bottom", f"{bounds['bottom']:.2f}")
        st.metric("Right", f"{bounds['right']:.2f}")
        st.metric("Top", f"{bounds['top']:.2f}")
    if g["crs"]:
        st.caption("The SR GeoTIFF preserves CRS and transform metadata. Output resolution is 2.5 m on a 4× finer grid.")
    else:
        st.caption("No CRS: pixel-only output. Ground resolution and geographic location are unknown.")


# Render the Download Reports tab.
def _render_downloads(evaluation: dict, job: dict, api_client):
    st.subheader("Download Reports")
    st.caption("Download a text evaluation report containing all computed metrics and metadata.")

    if st.button("Download Evaluation Report", type="primary"):
        try:
            report_bytes = api_client.evaluation_report(job["job_id"])
            st.download_button(
                "Save report",
                report_bytes,
                file_name=f"{job['job_id']}_report.txt",
                mime="text/plain",
            )
        except Exception as exc:
            st.error(f"Could not generate report: {exc}")

    with st.expander("Evaluation data (JSON)"):
        st.json(evaluation)


# Main entry point: render the full evaluation dashboard.
def render_evaluation_dashboard(crop_bytes: bytes, sr_bytes: bytes, evaluation: dict, job: dict, api_client):
    st.markdown("---")
    st.header("Evaluation & Analysis")

    tabs = st.tabs([
        "Overview",
        "Image Comparison",
        "Quantitative Metrics",
        "Band-wise Analysis",
        "Error / Difference Maps",
        "Spectral Analysis",
        "Uncertainty / Confidence",
        "Spatial Detail",
        "Model & Dataset Info",
        "Downloads",
    ])

    with tabs[0]:
        _render_overview(evaluation)
        _render_patch_info(evaluation)
        _render_georeferencing(evaluation)

    with tabs[1]:
        _render_comparison(crop_bytes, sr_bytes)

    with tabs[2]:
        _render_metrics(evaluation)

    with tabs[3]:
        _render_band_analysis(evaluation)

    with tabs[4]:
        _render_error_maps(evaluation)

    with tabs[5]:
        _render_spectral(evaluation)

    with tabs[6]:
        _render_uncertainty(evaluation)

    with tabs[7]:
        _render_spatial(evaluation)

    with tabs[8]:
        _render_model_info(evaluation)
        _render_dataset_info(evaluation)
        _render_limitations(evaluation)

    with tabs[9]:
        _render_downloads(evaluation, job, api_client)
