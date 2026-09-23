from pathlib import Path

import rasterio

from inference.geospatial import super_resolve
from inference.aoi import AOI, inspect_raster, plan_crop, write_crop, finish_output
from inference.model import load_model


INPUT_PATH = Path(
    "data/raw/demo/demo/cross-sensor/ROI_0000/lr.tif"
)


def inspect_output(path: Path) -> None:
    print()
    print("=" * 60)
    print("OUTPUT INSPECTION")
    print("=" * 60)

    with rasterio.open(path) as src:
        print(f"File       : {path}")
        print(f"Width      : {src.width}")
        print(f"Height     : {src.height}")
        print(f"Bands      : {src.count}")
        print(f"Resolution : {src.res}")
        print(f"CRS        : {src.crs}")
        print(f"Dtypes     : {src.dtypes}")
        print(f"Transform  : {src.transform}")


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{INPUT_PATH}"
        )

    print("=" * 60)
    print("REAL SENTINEL-2 SUPER-RESOLUTION TEST")
    print("=" * 60)

    print(f"Input: {INPUT_PATH}")

    directory = Path("outputs/real-image-test")
    metadata = inspect_raster(INPUT_PATH, order="rgbn")
    plan = plan_crop(INPUT_PATH, AOI(), max_patches=1)
    crop, model_input = write_crop(INPUT_PATH, directory, plan, metadata["band_indexes"], 10000)
    model, device = load_model(device="cpu", sampling_steps=20)
    raw = super_resolve(model_input, model=model, device=device, expected_patches=1)
    output_path = finish_output(raw, crop, directory / "result.tif")

    print()
    print(f"SR output: {output_path}")

    inspect_output(output_path)


if __name__ == "__main__":
    main()
