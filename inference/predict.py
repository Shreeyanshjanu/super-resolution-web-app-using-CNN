from __future__ import annotations

import torch

from inference.model import load_model


def predict_tensor(
    model,
    image: torch.Tensor,
    sampling_steps: int = 100,
) -> torch.Tensor:
    """
    Run super-resolution on a Sentinel-2 tensor.

    Expected input:
        [C, H, W]
        or
        [B, C, H, W]

    Values must be Sentinel-2 L2A reflectance in [0, 1].

    Output spatial resolution is 4x larger.
    """

    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")

    if image.ndim not in (3, 4):
        raise ValueError(
            "Expected tensor shape [C,H,W] or [B,C,H,W]. "
            f"Received: {tuple(image.shape)}"
        )

    if image.shape[-3] != 4:
        raise ValueError(
            "ESAOpenSR LDSR-S2 expects 4 channels "
            "(RGB + NIR). "
            f"Received: {image.shape[-3]} channels."
        )

    image = image.float()

    min_value = image.min().item()
    max_value = image.max().item()

    if min_value < 0 or max_value > 1:
        raise ValueError(
            "Input reflectance must be in [0,1]. "
            f"Received range [{min_value:.4f}, {max_value:.4f}]"
        )

    with torch.inference_mode():
        sr = model.forward(
            image,
            sampling_steps=sampling_steps,
        )

    return sr


def predict(
    image: torch.Tensor,
    sampling_steps: int = 100,
) -> torch.Tensor:
    """
    Convenience function that loads the pretrained model
    and performs inference.
    """
    model, _ = load_model()

    return predict_tensor(
        model=model,
        image=image,
        sampling_steps=sampling_steps,
    )