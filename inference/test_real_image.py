from pathlib import Path

import rasterio

from inference.geospatial import super_resolve


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

    output_path = super_resolve(
        input_path=INPUT_PATH,
        device="cpu",
        debug=False,
    )

    output_path = Path(output_path)

    print()
    print(f"SR output: {output_path}")

    inspect_output(output_path)


if __name__ == "__main__":
    main()