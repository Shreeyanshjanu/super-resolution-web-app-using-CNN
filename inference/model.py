from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "ldsrs2.yaml"
CHECKPOINT_NAME = "opensr-ldsrs2_v1_0_0.ckpt"


def get_device() -> str:
    """Return CUDA when available, otherwise CPU."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    device: str | None = None,
    sampling_steps: int = 100,
):
    """
    Load the pretrained ESA LDSR-S2 model.

    The default is the upstream 100-step reconstruction setting.
    """

    import torch
    from omegaconf import OmegaConf
    from opensr_model import SRLatentDiffusion

    if device is None:
        device = get_device()

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available."
        )

    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda.")
    if device == "cpu":
        threads = int(os.environ.get("SRM_CPU_THREADS", str(min(4, os.cpu_count() or 1))))
        if threads < 1:
            raise ValueError("SRM_CPU_THREADS must be positive.")
        torch.set_num_threads(threads)
    if not 1 <= sampling_steps <= 1000:
        raise ValueError("sampling_steps must be in [1, 1000].")
    checkpoint = Path(os.environ.get("SRM_CHECKPOINT", str(PROJECT_ROOT / CHECKPOINT_NAME))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run python -m scripts.download_model first.")
    config = OmegaConf.load(CONFIG_PATH)
    config.denoiser_settings.sampling_steps = sampling_steps
    print(f"Loading ESA LDSR-S2 on {device}...", flush=True)
    model = SRLatentDiffusion(config, device=device)
    model.load_pretrained(str(checkpoint))

    model.eval()
    print(f"CPU threads: {torch.get_num_threads()}", flush=True)

    print("ESA LDSR-S2 loaded successfully.")

    if sampling_steps is not None:
        print(f"Sampling steps: {sampling_steps}")

    return model, device
