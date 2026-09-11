from pathlib import Path

from opensr_utils.pipeline import large_file_processing


INPUT_PATH = Path(
    "data/raw/demo/demo/cross-sensor/ROI_0000/lr.tif"
)


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found:\n{INPUT_PATH}"
        )

    print("=" * 60)
    print("ESA PIPELINE INPUT TEST")
    print("=" * 60)

    job = large_file_processing(
        root=str(INPUT_PATH),
        model=None,
        window_size=(128, 128),
        factor=4,
        overlap=8,
        eliminate_border_px=0,
        device="cpu",
        gpus=None,
        debug=True,
        auto_run=False,
        batch_size=1,
        num_workers=0,
    )

    print()
    print("Pipeline accepted the image.")
    print(f"Input type : {job.input_type}")
    print(f"Width      : {job.image_meta['width']}")
    print(f"Height     : {job.image_meta['height']}")
    print(f"Bands      : {job.image_meta['bands']}")
    print(f"CRS        : {job.image_meta['crs']}")
    print(f"Windows    : {len(job.image_meta['image_windows'])}")
    print(f"Output dir : {job.output_dir}")


if __name__ == "__main__":
    main()