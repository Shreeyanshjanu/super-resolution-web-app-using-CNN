from pathlib import Path

import rasterio


BASE_DIR = Path(
    "data/raw/demo/demo/cross-sensor/ROI_0000"
)


FILES = {
    "LR": BASE_DIR / "lr.tif",
    "HR": BASE_DIR / "hr.tif",
    "SR": BASE_DIR / "sr.tif",
}


def inspect_file(name: str, path: Path) -> None:
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Path: {path}")

    with rasterio.open(path) as src:

        print(f"Width      : {src.width}")
        print(f"Height     : {src.height}")
        print(f"Bands      : {src.count}")
        print(f"Resolution : {src.res}")
        print(f"CRS        : {src.crs}")
        print(f"Dtypes     : {src.dtypes}")
        print(f"NoData     : {src.nodata}")

        for band_number in range(1, src.count + 1):
            data = src.read(band_number)

            print(
                f"Band {band_number}: "
                f"min={data.min()} "
                f"max={data.max()} "
                f"mean={data.mean():.4f}"
            )


def main() -> None:

    for name, path in FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

        inspect_file(name, path)


if __name__ == "__main__":
    main()