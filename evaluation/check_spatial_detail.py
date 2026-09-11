from pathlib import Path

import numpy as np
import rasterio


BASE_DIR = Path(
    "data/raw/demo/demo/cross-sensor/ROI_0000"
)


def load_band(path: Path, band: int) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band).astype(np.float32)


def report(name: str, image: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * 50)
    print(f"Shape : {image.shape}")
    print(f"Min   : {image.min():.4f}")
    print(f"Max   : {image.max():.4f}")
    print(f"Mean  : {image.mean():.4f}")
    print(f"Std   : {image.std():.4f}")


def main() -> None:
    lr = load_band(BASE_DIR / "lr.tif", 1)
    sr = load_band(BASE_DIR / "sr.tif", 1)
    hr = load_band(BASE_DIR / "hr.tif", 1)

    # Down/up relationship:
    # LR is 121x121, SR/HR are 484x484.
    # Compare standard deviation first.
    report("LR Band 1", lr)
    report("SR Band 1", sr)
    report("HR Band 1", hr)

    # Difference between neighboring pixels gives a simple
    # measure of local spatial variation.
    lr_dx = np.abs(np.diff(lr, axis=1)).mean()
    sr_dx = np.abs(np.diff(sr, axis=1)).mean()
    hr_dx = np.abs(np.diff(hr, axis=1)).mean()

    lr_dy = np.abs(np.diff(lr, axis=0)).mean()
    sr_dy = np.abs(np.diff(sr, axis=0)).mean()
    hr_dy = np.abs(np.diff(hr, axis=0)).mean()

    print("\nAverage horizontal change")
    print(f"LR: {lr_dx:.4f}")
    print(f"SR: {sr_dx:.4f}")
    print(f"HR: {hr_dx:.4f}")

    print("\nAverage vertical change")
    print(f"LR: {lr_dy:.4f}")
    print(f"SR: {sr_dy:.4f}")
    print(f"HR: {hr_dy:.4f}")


if __name__ == "__main__":
    main()