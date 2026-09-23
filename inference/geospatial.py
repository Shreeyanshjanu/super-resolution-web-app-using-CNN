from __future__ import annotations

from pathlib import Path

from opensr_utils.pipeline import large_file_processing


def super_resolve(
    input_path: str | Path,
    model,
    device: str,
    debug: bool = False,
    max_patches: int = 4,
    expected_patches: int | None = None,
    progress_callback=None,
) -> Path:
    """
    Run the already-loaded ESA LDSR-S2 model on a
    Sentinel-2 GeoTIFF.

    The model is supplied by the caller so that the
    pretrained checkpoint is loaded only once.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input does not exist: {input_path}"
        )

    if device not in ("cpu", "cuda"):
        raise ValueError(
            "device must be 'cpu' or 'cuda'"
        )

    if model is None:
        raise ValueError("A loaded ESA LDSR-S2 model is required.")
    import rasterio
    from inference.aoi import patch_count
    with rasterio.open(input_path) as src:
        patches = patch_count(src.width, src.height)
        if min(src.width, src.height) < 128:
            raise ValueError("Pad the selected crop to at least 128×128 before inference.")
        if patches > max_patches:
            raise ValueError(f"This crop needs {patches} patches; limit: {max_patches}.")

    print("=" * 60)
    print("SENTINEL-2 SUPER RESOLUTION")
    print("=" * 60)

    print(f"Input : {input_path}")
    print(f"Device: {device}")
    print(f"Debug : {debug}")

    job = large_file_processing(
        root=str(input_path),
        model=model,

        # 10 m -> 2.5 m
        window_size=(128, 128),
        factor=4,

        # Reduce tile boundary artifacts.
        overlap=8,
        eliminate_border_px=0,

        device=device,
        gpus=None,

        save_preview=False,

        debug=debug,
        auto_run=False,

        cleanup=True,
        overwrite=True,

        # Conservative CPU settings.
        batch_size=1,
        num_workers=0,
    )

    print(f"Input type: {job.input_type}")
    print(
        f"Number of patches: "
        f"{len(job.image_meta['image_windows'])}"
    )

    print("Starting SR inference...")

    actual = len(job.image_meta["image_windows"])
    if actual > max_patches or (expected_patches is not None and actual != expected_patches):
        raise RuntimeError(f"Unexpected patch count: {actual}. Refusing unplanned inference.")
    if progress_callback:
        original_step = job.model.predict_step
        def predict_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            predict_step.completed += 1
            progress_callback(predict_step.completed, actual)
            return result
        predict_step.completed = 0
        job.model.predict_step = predict_step

    output_path = job.run()

    print(f"SR output: {output_path}")

    return Path(output_path)
