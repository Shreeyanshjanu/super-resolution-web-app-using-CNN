from __future__ import annotations

import torch

from opensr_utils.model_utils.get_models import get_ldsrs2


def get_device() -> str:
    """Return CUDA when available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    device: str | None = None,
    sampling_steps: int | None = None,
):
    """
    Load the pretrained ESA LDSR-S2 model.

    sampling_steps can be reduced for CPU development/testing.
    """

    if device is None:
        device = get_device()

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available."
        )

    print(f"Loading ESA LDSR-S2 on {device}...")

    model = get_ldsrs2(device=device)

    # Reduce diffusion steps for CPU testing.
    if sampling_steps is not None:
        model.config.denoiser_settings.sampling_steps = sampling_steps

    model.eval()

    print("ESA LDSR-S2 loaded successfully.")

    if sampling_steps is not None:
        print(f"Sampling steps: {sampling_steps}")

    return model, device