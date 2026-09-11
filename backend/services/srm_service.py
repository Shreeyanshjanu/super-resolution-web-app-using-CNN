from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from inference.model import load_model
from inference.geospatial import super_resolve


class SRMService:
    """
    Owns the pretrained SR model and processes SR jobs.

    The model is loaded once when this service is created.
    """

    def __init__(self) -> None:
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print()
        print("=" * 60)
        print("INITIALIZING SRM SERVICE")
        print("=" * 60)

        self.model, self.device = load_model(
            device=self.device,
            sampling_steps=20,
        )

        # The ESA pipeline modifies temporary attributes on the
        # model while processing. For now, serialize jobs.
        self._lock = threading.Lock()

        print("SRM service ready.")

    def process(
        self,
        input_path: Path,
        job_directory: Path,
    ) -> Path:

        output_path: Path | None = None

        with self._lock:
            job_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = super_resolve(
                input_path=input_path,
                model=self.model,
                device=self.device,
                debug=False,
            )

        return output_path

    def info(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "model": "ESA LDSR-S2",
            "scale_factor": 4,
            "input_resolution_m": 10,
            "output_resolution_m": 2.5,
            "channels": 4,
        }