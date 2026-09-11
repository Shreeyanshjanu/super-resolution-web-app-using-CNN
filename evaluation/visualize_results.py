from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


BASE_DIR = Path(
    "data/raw/demo/demo/cross-sensor/ROI_0000"
)


def load_rgb(path: Path, scale: float) -> np.ndarray:
    """
    Load B04, B03, B02 and return RGB in HxWx3 format.
    """

    with rasterio.open(path) as src:
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)

    rgb = np.stack(
        [red, green, blue],
        axis=-1,
    )

    rgb /= scale

    # Robust display stretching.
    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    rgb = (rgb - low) / (high - low + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    return rgb


def main() -> None:

    lr_path = BASE_DIR / "lr.tif"
    sr_path = BASE_DIR / "sr.tif"
    hr_path = BASE_DIR / "hr.tif"

    # LR and SR use Sentinel-2-like 10000 scaling.
    lr = load_rgb(lr_path, 10000.0)
    sr = load_rgb(sr_path, 10000.0)

    # HR uses 8-bit NAIP values.
    hr = load_rgb(hr_path, 255.0)

    # Resize LR only for visualization.
    from PIL import Image

    lr_image = Image.fromarray(
        (lr * 255).astype(np.uint8)
    )

    lr_image = lr_image.resize(
        (484, 484),
        Image.Resampling.BICUBIC,
    )

    lr = np.asarray(lr_image).astype(np.float32) / 255.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(lr)
    axes[0].set_title("LR → Bicubic")
    axes[0].axis("off")

    axes[1].imshow(sr)
    axes[1].set_title("ESA LDSR-S2")
    axes[1].axis("off")

    axes[2].imshow(hr)
    axes[2].set_title("HR Reference")
    axes[2].axis("off")

    plt.tight_layout()

    output = BASE_DIR / "comparison_rgb.png"

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved comparison to: {output}")


if __name__ == "__main__":
    main()