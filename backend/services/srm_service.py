from __future__ import annotations
import threading
from pathlib import Path
from time import perf_counter

from backend.settings import MAX_PATCHES, SAMPLING_STEPS, SAMPLING_OPTIONS
from inference.aoi import finish_output
from inference.model import load_model


class SRMService:
    """One unchanged ESA LDSR-S2 instance; only prepared AOI crops reach it."""

    def __init__(self):
        self.model, self.device = load_model(sampling_steps=SAMPLING_STEPS)
        self._lock = threading.Lock()

    def process(self, input_path: Path, job_directory: Path, plan: dict, progress_callback=None,
                sampling_steps: int = SAMPLING_STEPS) -> dict:
        from inference.geospatial import super_resolve
        if sampling_steps not in SAMPLING_OPTIONS:
            raise ValueError("Choose 20, 50 or 100 sampling steps.")
        with self._lock:
            start = perf_counter()
            previous_steps = self.model.config.denoiser_settings.sampling_steps
            self.model.config.denoiser_settings.sampling_steps = sampling_steps
            try:
                raw = super_resolve(
                    input_path=input_path, model=self.model, device=self.device,
                    max_patches=MAX_PATCHES, expected_patches=plan["patches"],
                    progress_callback=progress_callback,
                )
            finally:
                self.model.config.denoiser_settings.sampling_steps = previous_steps
            output = finish_output(raw, job_directory / "crop.tif", job_directory / "result.tif")
            import rasterio
            with rasterio.open(output, "r+") as product:
                product.update_tags(sampling_steps=str(sampling_steps))
            elapsed = perf_counter() - start
            return {"output_path": str(output), "inference_seconds": elapsed,
                    "seconds_per_patch": elapsed / plan["patches"], "sampling_steps": sampling_steps}

    def info(self):
        return {"device": self.device, "model": "ESA LDSR-S2", "scale_factor": 4,
                "input_resolution_m": 10, "output_resolution_m": 2.5, "channels": 4,
                "sampling_steps": SAMPLING_STEPS, "patch_size": 128, "overlap": 8,
                "sampling_options": list(SAMPLING_OPTIONS),
                "max_patches": MAX_PATCHES}
