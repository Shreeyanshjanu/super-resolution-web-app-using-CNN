from pathlib import Path

import rasterio


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "test_images" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_images" / "prepared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def combine_bands(b2b3b4_path: Path, b8_path: Path, output_path: Path):
    print(f"\nProcessing: {b2b3b4_path.name}")

    with rasterio.open(b2b3b4_path) as rgb_src:
        with rasterio.open(b8_path) as nir_src:

            # Do not combine bands from different geographic pixel grids.
            if rgb_src.crs != nir_src.crs or not rgb_src.transform.almost_equals(nir_src.transform):
                raise ValueError("RGB and NIR must have the same CRS and pixel transform.")
            if rgb_src.count != 3 or nir_src.count != 1:
                raise ValueError("Expected a 3-band B02/B03/B04 file and a 1-band B08 file.")
            if rgb_src.crs is None:
                print("WARNING: source has no CRS; this output supports pixel crops only.")
            # Check dimensions
            if rgb_src.width != nir_src.width or rgb_src.height != nir_src.height:
                raise ValueError(
                    f"Size mismatch:\n"
                    f"{rgb_src.name}: {rgb_src.width}x{rgb_src.height}\n"
                    f"{nir_src.name}: {nir_src.width}x{nir_src.height}"
                )

            # Read B2, B3, B4
            b2 = rgb_src.read(1)
            b3 = rgb_src.read(2)
            b4 = rgb_src.read(3)

            # Read B8
            b8 = nir_src.read(1)

            # Copy metadata from B2/B3/B4 file
            profile = rgb_src.profile.copy()

            # We are creating a 4-band GeoTIFF
            profile.update(
                count=4,
                dtype=b2.dtype,
                compress="deflate"
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(b4, 1)
                dst.write(b3, 2)
                dst.write(b2, 3)
                dst.write(b8, 4)
                dst.write_mask(rgb_src.dataset_mask() & nir_src.dataset_mask())

                # Band descriptions
                dst.set_band_description(1, "B04")
                dst.set_band_description(2, "B03")
                dst.set_band_description(3, "B02")
                dst.set_band_description(4, "B08")

    print(f"Created: {output_path.name}")


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: Folder does not exist:")
        print(RAW_DIR)
        return

    b2b3b4_files = sorted(RAW_DIR.glob("*_10m_B2B3B4.tif"))

    if not b2b3b4_files:
        print("No B2B3B4 TIFF files found.")
        print(f"Expected files inside: {RAW_DIR}")
        return

    print(f"Found {len(b2b3b4_files)} B2B3B4 files.")

    successful = 0

    for b2b3b4_path in b2b3b4_files:

        # Example:
        # 001_10m_B2B3B4.tif
        # becomes:
        # 001_10m_B8.tif

        prefix = b2b3b4_path.name.replace("_10m_B2B3B4.tif", "")
        b8_path = RAW_DIR / f"{prefix}_10m_B8.tif"

        if not b8_path.exists():
            print(f"\nSkipping {prefix}: B8 file not found.")
            continue

        output_path = OUTPUT_DIR / f"{prefix}_RGBN.tif"

        try:
            combine_bands(
                b2b3b4_path,
                b8_path,
                output_path
            )

            successful += 1

        except Exception as e:
            print(f"\nERROR processing {prefix}: {e}")

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Successfully created: {successful} files")
    print(f"Output folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
