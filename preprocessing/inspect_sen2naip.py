from pathlib import Path

import rasterio


DATA_DIR = Path("data/raw/demo/demo/cross-sensor/ROI_0000")


def inspect_raster(path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"FILE: {path.name}")
    print("=" * 60)

    with rasterio.open(path) as src:
        print(f"Driver       : {src.driver}")
        print(f"Width        : {src.width}")
        print(f"Height       : {src.height}")
        print(f"Bands        : {src.count}")
        print(f"CRS          : {src.crs}")
        print(f"Transform    : {src.transform}")
        print(f"Resolution   : {src.res}")
        print(f"Dtypes       : {src.dtypes}")
        print(f"NoData       : {src.nodata}")

        for band_index in range(1, src.count + 1):
            band = src.read(band_index, masked=True)

            print(f"\nBand {band_index}")
            print(f"  Min        : {band.min()}")
            print(f"  Max        : {band.max()}")
            print(f"  Mean       : {band.mean()}")
            print(f"  Masked     : {band.mask.any()}")


def main() -> None:
    lr_path = DATA_DIR / "lr.tif"
    hr_path = DATA_DIR / "hr.tif"

    if not lr_path.exists():
        raise FileNotFoundError(
            f"LR file not found:\n{lr_path}"
        )

    if not hr_path.exists():
        raise FileNotFoundError(
            f"HR file not found:\n{hr_path}"
        )

    inspect_raster(lr_path)
    inspect_raster(hr_path)


if __name__ == "__main__":
    main()