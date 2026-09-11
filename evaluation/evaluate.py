from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from evaluation.metrics import (
    calculate_band_metrics,
    histogram_match_hr_to_lr,
    normalize_hr,
    normalize_lr,
    normalize_sr,
)


BASE_DIR = Path(
    "data/raw/demo/demo/cross-sensor/ROI_0000"
)

LR_PATH = BASE_DIR / "lr.tif"
HR_PATH = BASE_DIR / "hr.tif"
SR_PATH = BASE_DIR / "sr.tif"


BAND_NAMES = [
    "B04 - Red",
    "B03 - Green",
    "B02 - Blue",
    "B08 - NIR",
]


def load_image(path: Path) -> np.ndarray:
    """
    Load a GeoTIFF as [bands, height, width].
    """
    with rasterio.open(path) as src:
        return src.read()


def bicubic_upscale(
    lr: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Bicubic baseline from LR to HR spatial dimensions.

    LR is expected to already be in reflectance [0,1].
    """

    output = np.empty(
        (
            lr.shape[0],
            target_height,
            target_width,
        ),
        dtype=np.float32,
    )

    for band in range(lr.shape[0]):

        image = Image.fromarray(
            lr[band].astype(np.float32),
            mode="F",
        )

        image = image.resize(
            (target_width, target_height),
            Image.Resampling.BICUBIC,
        )

        output[band] = np.asarray(
            image,
            dtype=np.float32,
        )

    return output


def print_metrics(
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    results = {}

    for index, band_name in enumerate(BAND_NAMES):

        metrics = calculate_band_metrics(
            prediction[index],
            target[index],
        )

        results[band_name] = metrics

        print(
            f"{band_name:<14}"
            f"PSNR: {metrics['psnr']:>8.3f} dB   "
            f"SSIM: {metrics['ssim']:.5f}   "
            f"RMSE: {metrics['rmse']:.6f}"
        )

    avg_psnr = np.mean(
        [v["psnr"] for v in results.values()]
    )

    avg_ssim = np.mean(
        [v["ssim"] for v in results.values()]
    )

    avg_rmse = np.mean(
        [v["rmse"] for v in results.values()]
    )

    print("-" * 70)
    print(
        f"{'AVERAGE':<14}"
        f"PSNR: {avg_psnr:>8.3f} dB   "
        f"SSIM: {avg_ssim:.5f}   "
        f"RMSE: {avg_rmse:.6f}"
    )

    return results


def main() -> None:

    for path in (LR_PATH, HR_PATH, SR_PATH):

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    print("=" * 70)
    print("SEN2NAIP CROSS-SENSOR SR EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load files
    # ---------------------------------------------------------

    lr_raw = load_image(LR_PATH)
    hr_raw = load_image(HR_PATH)
    sr_raw = load_image(SR_PATH)

    print(f"LR shape: {lr_raw.shape}")
    print(f"HR shape: {hr_raw.shape}")
    print(f"SR shape: {sr_raw.shape}")

    # ---------------------------------------------------------
    # Convert to comparable floating-point ranges
    # ---------------------------------------------------------

    lr = normalize_lr(lr_raw)
    hr = normalize_hr(hr_raw)
    sr = normalize_sr(sr_raw)

    # ---------------------------------------------------------
    # Radiometric harmonization
    #
    # HR is NAIP, while SR follows the Sentinel-2 reference.
    # Match HR histograms to the LR Sentinel-2 distribution.
    # ---------------------------------------------------------

    hr_matched = histogram_match_hr_to_lr(
        hr,
        lr,
    )

    # ---------------------------------------------------------
    # Bicubic baseline
    # ---------------------------------------------------------

    bicubic = bicubic_upscale(
        lr,
        target_height=hr.shape[1],
        target_width=hr.shape[2],
    )

    # ---------------------------------------------------------
    # Check value ranges
    # ---------------------------------------------------------

    print()
    print("Value ranges after normalization:")
    print(
        f"LR      : "
        f"{lr.min():.5f} → {lr.max():.5f}"
    )
    print(
        f"HR      : "
        f"{hr.min():.5f} → {hr.max():.5f}"
    )
    print(
        f"HR match: "
        f"{hr_matched.min():.5f} → {hr_matched.max():.5f}"
    )
    print(
        f"SR      : "
        f"{sr.min():.5f} → {sr.max():.5f}"
    )

    # ---------------------------------------------------------
    # Evaluate bicubic
    # ---------------------------------------------------------

    bicubic_results = print_metrics(
        "BICUBIC BASELINE vs HARMONIZED HR",
        bicubic,
        hr_matched,
    )

    # ---------------------------------------------------------
    # Evaluate ESA SR
    # ---------------------------------------------------------

    sr_results = print_metrics(
        "ESA LDSR-S2 vs HARMONIZED HR",
        sr,
        hr_matched,
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    bicubic_psnr = np.mean(
        [x["psnr"] for x in bicubic_results.values()]
    )

    sr_psnr = np.mean(
        [x["psnr"] for x in sr_results.values()]
    )

    print(
        f"Bicubic average PSNR : {bicubic_psnr:.3f} dB"
    )

    print(
        f"ESA SR average PSNR  : {sr_psnr:.3f} dB"
    )

    print(
        f"PSNR improvement     : "
        f"{sr_psnr - bicubic_psnr:+.3f} dB"
    )


if __name__ == "__main__":
    main()