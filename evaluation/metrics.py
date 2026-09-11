from __future__ import annotations

import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


def normalize_lr(lr: np.ndarray) -> np.ndarray:
    """
    SEN2NAIP LR convention:
    integer Sentinel-2 reflectance values / 10000.
    """
    return lr.astype(np.float32) / 10000.0


def normalize_hr(hr: np.ndarray) -> np.ndarray:
    """
    SEN2NAIP HR convention:
    uint8 values / 255.
    """
    return hr.astype(np.float32) / 255.0


def normalize_sr(sr: np.ndarray) -> np.ndarray:
    """
    ESA LDSR-S2 output from our GeoTIFF is stored on the
    Sentinel-2-style reflectance*10000 scale.
    """
    return sr.astype(np.float32) / 10000.0


def histogram_match_hr_to_lr(
    hr: np.ndarray,
    lr: np.ndarray,
) -> np.ndarray:
    """
    Match each HR band to the corresponding LR band.

    This adjusts radiometric distribution while preserving
    the spatial structure of the HR reference.
    """

    if hr.shape[0] != lr.shape[0]:
        raise ValueError(
            "HR and LR must contain the same number of bands."
        )

    matched = np.empty_like(hr, dtype=np.float32)

    for band in range(hr.shape[0]):
        matched[band] = match_histograms(
            hr[band],
            lr[band],
        ).astype(np.float32)

    return matched


def calculate_band_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """
    Calculate PSNR, SSIM and RMSE for one band.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )

    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)

    mse = np.mean(
        (prediction - target) ** 2
    )

    rmse = float(np.sqrt(mse))

    data_range = float(
        target.max() - target.min()
    )

    if data_range == 0:
        data_range = 1.0

    psnr = float(
        peak_signal_noise_ratio(
            target,
            prediction,
            data_range=data_range,
        )
    )

    ssim = float(
        structural_similarity(
            target,
            prediction,
            data_range=data_range,
        )
    )

    return {
        "psnr": psnr,
        "ssim": ssim,
        "rmse": rmse,
    }