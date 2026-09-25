# Evaluation service: computes metrics, statistics and analysis data for completed SR jobs.
from __future__ import annotations

import json
import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

LOG = logging.getLogger(__name__)

BAND_NAMES = ["B04 — Red", "B03 — Green", "B02 — Blue", "B08 — NIR"]
BAND_SHORT = ["B04", "B03", "B02", "B08"]
VALUE_SCALE = 10000


# Load a GeoTIFF from raw bytes and return (data, profile) where data is [bands, height, width].
def _read_geotiff(geotiff_bytes: bytes) -> tuple[np.ndarray, dict]:
    with rasterio.MemoryFile(geotiff_bytes) as mem, mem.open() as src:
        data = src.read(masked=True).astype(np.float32)
        profile = {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs) if src.crs else None,
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "dtype": src.dtypes[0],
            "bounds": dict(zip(("left", "bottom", "right", "top"), src.bounds)),
            "transform": list(src.transform)[:6],
        }
    return data, profile


# Compute bicubic 4× upscale of the LR crop for baseline comparison.
def bicubic_upscale(data: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    bands = data.shape[0]
    result = np.empty((bands, target_height, target_width), dtype=np.float32)
    for b in range(bands):
        band = data[b]
        mask = np.ma.getmaskarray(band) if isinstance(band, np.ma.MaskedArray) else None
        values = np.ma.filled(band, 0).astype(np.float32)
        img = Image.fromarray(values)
        resized = img.resize((target_width, target_height), Image.Resampling.BICUBIC)
        result[b] = np.array(resized, dtype=np.float32)
        if mask is not None:
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_resized = mask_img.resize((target_width, target_height), Image.Resampling.NEAREST)
            result[b] = np.ma.array(result[b], mask=np.array(mask_resized).astype(bool))
    return result


# Calculate per-band statistics (mean, std, min, max) for valid pixels.
def band_statistics(data: np.ndarray) -> list[dict]:
    stats = []
    for b in range(data.shape[0]):
        band = np.ma.asarray(data[b])
        valid = band.compressed()
        if valid.size == 0:
            stats.append({"mean": None, "std": None, "min": None, "max": None, "valid_pixels": 0})
        else:
            stats.append({
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "valid_pixels": int(valid.size),
            })
    return stats


# Compute MAE between prediction and target (per-band, for valid pixels).
def compute_mae(prediction: np.ndarray, target: np.ndarray) -> list[float | None]:
    result = []
    for b in range(prediction.shape[0]):
        p = np.ma.asarray(prediction[b]).compressed()
        t = np.ma.asarray(target[b]).compressed()
        n = min(p.size, t.size)
        if n == 0:
            result.append(None)
        else:
            result.append(float(np.mean(np.abs(p[:n] - t[:n]))))
    return result


# Compute RMSE between prediction and target (per-band, for valid pixels).
def compute_rmse(prediction: np.ndarray, target: np.ndarray) -> list[float | None]:
    result = []
    for b in range(prediction.shape[0]):
        p = np.ma.asarray(prediction[b]).compressed()
        t = np.ma.asarray(target[b]).compressed()
        n = min(p.size, t.size)
        if n == 0:
            result.append(None)
        else:
            result.append(float(np.sqrt(np.mean((p[:n] - t[:n]) ** 2))))
    return result


# Compute spatial detail metrics: gradient magnitude, edge density, high-frequency energy.
def spatial_detail_metrics(data: np.ndarray) -> list[dict]:
    metrics = []
    for b in range(data.shape[0]):
        band = np.ma.filled(data[b], 0).astype(np.float32)
        dx = np.abs(np.diff(band, axis=1))
        dy = np.abs(np.diff(band, axis=0))
        min_h = min(dx.shape[0], dy.shape[0])
        min_w = min(dx.shape[1], dy.shape[1])
        gradient_mag = np.sqrt(dx[:min_h, :min_w] ** 2 + dy[:min_h, :min_w] ** 2) if min_h > 0 and min_w > 0 else np.array([[0.0]])
        edge_threshold = np.mean(band) + 2 * np.std(band) if np.std(band) > 0 else 0
        edges = (np.abs(dx) > edge_threshold).sum() + (np.abs(dy) > edge_threshold).sum()
        total = dx.size + dy.size
        high_freq_energy = float(np.mean(gradient_mag ** 2))
        metrics.append({
            "mean_gradient": float(np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2,
            "edge_density": float(edges) / max(total, 1),
            "high_frequency_energy": high_freq_energy,
            "local_contrast": float(np.std(band)),
        })
    return metrics


# Compute histogram data for a single band (for spectral analysis plots).
def band_histogram(data: np.ndarray, n_bins: int = 50) -> dict:
    values = np.ma.asarray(data).compressed()
    if values.size == 0:
        return {"counts": [0] * n_bins, "bin_edges": [], "mean": None, "std": None}
    counts, edges = np.histogram(values, bins=n_bins)
    return {
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


# Build the overview section: input/output dimensions, resolution, processing info.
def build_overview(crop_profile: dict, sr_profile: dict, job: dict) -> dict:
    plan = job.get("plan", {})
    return {
        "input": {
            "width": crop_profile["width"],
            "height": crop_profile["height"],
            "bands": crop_profile["bands"],
            "resolution_x": crop_profile["resolution_x"],
            "resolution_y": crop_profile["resolution_y"],
            "crs": crop_profile["crs"],
            "dtype": crop_profile["dtype"],
        },
        "output": {
            "width": sr_profile["width"],
            "height": sr_profile["height"],
            "bands": sr_profile["bands"],
            "resolution_x": sr_profile["resolution_x"],
            "resolution_y": sr_profile["resolution_y"],
            "crs": sr_profile["crs"],
            "dtype": sr_profile["dtype"],
        },
        "super_resolution": {
            "scale_factor": 4,
            "input_resolution_m": 10,
            "output_resolution_m": 2.5,
            "sampling_steps": job.get("sampling_steps", 20),
            "patches": plan.get("patches", 0),
            "inference_seconds": round(job.get("inference_seconds", 0), 2),
            "seconds_per_patch": round(job.get("seconds_per_patch", 0), 2),
        },
    }


# Build the complete evaluation data for a completed job.
def compute_evaluation(crop_bytes: bytes, sr_bytes: bytes, job: dict) -> dict:
    crop_data, crop_profile = _read_geotiff(crop_bytes)
    sr_data, sr_profile = _read_geotiff(sr_bytes)

    overview = build_overview(crop_profile, sr_profile, job)

    crop_norm = crop_data / VALUE_SCALE
    sr_norm = sr_data / VALUE_SCALE

    bicubic = bicubic_upscale(crop_norm, sr_profile["height"], sr_profile["width"])

    sr_band_stats = band_statistics(sr_norm)
    crop_band_stats = band_statistics(crop_norm)

    band_analysis = []
    for b in range(4):
        band_info = {
            "name": BAND_NAMES[b],
            "short": BAND_SHORT[b],
            "input_stats": crop_band_stats[b],
            "output_stats": sr_band_stats[b],
            "reference_metrics": {
                "psnr": None,
                "ssim": None,
                "rmse": None,
                "mae": None,
                "unavailable_reason": "Reference ground truth unavailable — quantitative reconstruction metrics are not computed for this image.",
            },
        }
        band_analysis.append(band_info)

    spectral = {
        "input": {
            "per_band": [band_histogram(crop_norm[b]) for b in range(4)],
            "overall_mean": float(np.mean(crop_norm)) if crop_norm.size else None,
            "overall_std": float(np.std(crop_norm)) if crop_norm.size else None,
        },
        "output": {
            "per_band": [band_histogram(sr_norm[b]) for b in range(4)],
            "overall_mean": float(np.mean(sr_norm)) if sr_norm.size else None,
            "overall_std": float(np.std(sr_norm)) if sr_norm.size else None,
        },
        "spectral_error": {
            "per_band_mean": [
                float(abs(np.mean(sr_norm[b]) - np.mean(crop_norm[b])))
                if sr_norm[b].size and crop_norm[b].size else None
                for b in range(4)
            ],
            "note": "Spectral error measures the shift in mean reflectance between input and SR output. It does not indicate accuracy without a ground-truth reference.",
        },
    }

    spatial = {
        "input": spatial_detail_metrics(crop_norm),
        "output": spatial_detail_metrics(sr_norm),
        "bicubic": spatial_detail_metrics(bicubic),
        "note": "Increased high-frequency content does not automatically indicate increased truth. Visual sharpness alone does not prove reconstruction accuracy.",
    }

    uncertainty = {
        "available": False,
        "message": "Uncertainty visualization is not currently exposed by this inference pipeline. The ESA LDSR-S2 model does not produce per-pixel confidence estimates.",
    }

    confusion_matrix = {
        "applicable": False,
        "message": "Confusion matrix is not applicable to the current super-resolution reconstruction task. It can be added when a downstream classification task is evaluated.",
    }

    model_info = {
        "model": "ESA LDSR-S2",
        "architecture": "Latent Diffusion",
        "components": ["Autoencoder", "Denoising U-Net"],
        "input": "4-band Sentinel-2 RGB-NIR",
        "input_band_order": "B04, B03, B02, B08",
        "scale_factor": "4×",
        "input_resolution": "10 m",
        "output_grid": "2.5 m",
        "latent_representation": "4 × 128 × 128",
        "target_output": "512 × 512",
        "approximate_parameters": "~170 million",
        "sampling_method": "DDIM",
        "sampling_steps_options": [20, 50, 100],
        "sampling_steps_used": job.get("sampling_steps", 20),
    }

    dataset_info = {
        "dataset": "SEN2NAIP",
        "purpose": "Cross-sensor satellite super-resolution evaluation/training data",
        "bands": "B02, B03, B04, B08",
        "input_source": "Sentinel-2",
        "hr_reference": "NAIP where available",
        "publication_year": 2024,
        "note": "Dataset publication year (2024) and image acquisition years are different concepts.",
    }

    limitations = [
        "LDSR-S2 produces a learned estimate, not native 2.5 m satellite observations.",
        "The 2.5 m output is a finer output grid, not a native sensor resolution.",
        "Generated fine details may be hallucinated by the model.",
        "Cross-sensor evaluation involves differences between Sentinel-2 and NAIP characteristics.",
        "Quantitative metrics (PSNR, SSIM, RMSE) require valid high-resolution ground truth.",
        "Visual sharpness alone does not prove reconstruction accuracy.",
        "Spectral consistency is important for multispectral remote sensing applications.",
        "Results may vary depending on scene characteristics and land cover type.",
    ]

    patch_info = {
        "patch_size": 128,
        "overlap": 8,
        "scale": 4,
        "patches_used": job.get("plan", {}).get("patches", 0),
        "output_dimensions": f"{sr_profile['width']} × {sr_profile['height']}",
    }

    georeferencing = {
        "crs": sr_profile["crs"],
        "transform": sr_profile["transform"],
        "resolution_x": sr_profile["resolution_x"],
        "resolution_y": sr_profile["resolution_y"],
        "bounds": sr_profile["bounds"],
        "width": sr_profile["width"],
        "height": sr_profile["height"],
    }

    return {
        "overview": overview,
        "band_analysis": band_analysis,
        "spectral": spectral,
        "spatial_detail": spatial,
        "uncertainty": uncertainty,
        "confusion_matrix": confusion_matrix,
        "model_info": model_info,
        "dataset_info": dataset_info,
        "limitations": limitations,
        "patch_info": patch_info,
        "georeferencing": georeferencing,
        "reference_available": False,
    }


# Generate a text-based evaluation report for download.
def generate_report(evaluation: dict, job: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("SATELLITE SUPER-RESOLUTION EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    ov = evaluation["overview"]
    lines.append("INPUT")
    lines.append(f"  Dimensions: {ov['input']['width']} × {ov['input']['height']}")
    lines.append(f"  Bands: {ov['input']['bands']}")
    lines.append(f"  Resolution: {ov['input']['resolution_x']} m")
    lines.append(f"  CRS: {ov['input']['crs'] or 'None'}")
    lines.append(f"  Data type: {ov['input']['dtype']}")
    lines.append("")

    lines.append("OUTPUT")
    lines.append(f"  Dimensions: {ov['output']['width']} × {ov['output']['height']}")
    lines.append(f"  Bands: {ov['output']['bands']}")
    lines.append(f"  Resolution: {ov['output']['resolution_x']} m")
    lines.append(f"  CRS: {ov['output']['crs'] or 'None'}")
    lines.append(f"  Data type: {ov['output']['dtype']}")
    lines.append("")

    sr = ov["super_resolution"]
    lines.append("SUPER-RESOLUTION")
    lines.append(f"  Scale factor: {sr['scale_factor']}×")
    lines.append(f"  Input resolution: {sr['input_resolution_m']} m")
    lines.append(f"  Output grid: {sr['output_resolution_m']} m")
    lines.append(f"  Sampling steps: {sr['sampling_steps']}")
    lines.append(f"  Patches: {sr['patches']}")
    lines.append(f"  Inference time: {sr['inference_seconds']} s")
    lines.append("")

    lines.append("MODEL")
    mi = evaluation["model_info"]
    lines.append(f"  Model: {mi['model']}")
    lines.append(f"  Architecture: {mi['architecture']}")
    lines.append(f"  Parameters: {mi['approximate_parameters']}")
    lines.append(f"  Sampling: {mi['sampling_method']}")
    lines.append("")

    lines.append("DATASET")
    di = evaluation["dataset_info"]
    lines.append(f"  Dataset: {di['dataset']}")
    lines.append(f"  Purpose: {di['purpose']}")
    lines.append(f"  Publication year: {di['publication_year']}")
    lines.append("")

    lines.append("QUANTITATIVE METRICS")
    if evaluation["reference_available"]:
        for ba in evaluation["band_analysis"]:
            rm = ba["reference_metrics"]
            lines.append(f"  {ba['name']}:")
            lines.append(f"    PSNR: {rm['psnr']:.2f} dB" if rm['psnr'] is not None else "    PSNR: N/A")
            lines.append(f"    SSIM: {rm['ssim']:.4f}" if rm['ssim'] is not None else "    SSIM: N/A")
            lines.append(f"    RMSE: {rm['rmse']:.6f}" if rm['rmse'] is not None else "    RMSE: N/A")
            lines.append(f"    MAE: {rm['mae']:.6f}" if rm['mae'] is not None else "    MAE: N/A")
    else:
        lines.append("  N/A — no valid reference available")
    lines.append("")

    lines.append("BAND-WISE STATISTICS (SR OUTPUT)")
    for ba in evaluation["band_analysis"]:
        s = ba["output_stats"]
        if s["mean"] is not None:
            lines.append(f"  {ba['name']}: mean={s['mean']:.4f}, std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}")
        else:
            lines.append(f"  {ba['name']}: no valid pixels")
    lines.append("")

    lines.append("SPATIAL DETAIL ANALYSIS")
    sd = evaluation["spatial_detail"]
    for label, key in [("Input", "input"), ("Bicubic", "bicubic"), ("LDSR-S2", "output")]:
        avg_grad = np.mean([m["mean_gradient"] for m in sd[key]])
        avg_hfe = np.mean([m["high_frequency_energy"] for m in sd[key]])
        lines.append(f"  {label}: avg_gradient={avg_grad:.6f}, avg_hf_energy={avg_hfe:.6f}")
    lines.append(f"  Note: {sd['note']}")
    lines.append("")

    lines.append("SPECTRAL ANALYSIS")
    sp = evaluation["spectral"]
    lines.append(f"  Input overall: mean={sp['input']['overall_mean']:.4f}, std={sp['input']['overall_std']:.4f}")
    lines.append(f"  Output overall: mean={sp['output']['overall_mean']:.4f}, std={sp['output']['overall_std']:.4f}")
    for b in range(4):
        err = evaluation["spectral"]["spectral_error"]["per_band_mean"][b]
        if err is not None:
            lines.append(f"  {BAND_NAMES[b]} spectral error (mean shift): {err:.6f}")
    lines.append("")

    lines.append("LIMITATIONS")
    for i, lim in enumerate(evaluation["limitations"], 1):
        lines.append(f"  {i}. {lim}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)
