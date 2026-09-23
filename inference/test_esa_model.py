from pathlib import Path

import torch
import opensr_model
from inference.model import load_model
from omegaconf import OmegaConf


def get_config_path() -> Path:
    """
    Locate the config bundled with the installed opensr-model package.
    """
    package_dir = Path(opensr_model.__file__).resolve().parent
    config_path = package_dir / "configs" / "config_10m.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find ESAOpenSR 10m config at:\n{config_path}"
        )

    return config_path


def main() -> None:
    print("=" * 60)
    print("ESAOpenSR MODEL TEST")
    print("=" * 60)

    # Automatically use GPU when one is available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # Locate the config from the installed Python package
    # ---------------------------------------------------------
    config_path = get_config_path()

    print(f"Config: {config_path}")

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------
    config = OmegaConf.load(config_path)

    print(f"Checkpoint version: {config.ckpt_version}")

    # ---------------------------------------------------------
    # Create the pretrained SR model
    # ---------------------------------------------------------
    print("Creating ESAOpenSR model...")

    model, device = load_model(device=device, sampling_steps=20)

    # ---------------------------------------------------------
    # Download/load pretrained checkpoint
    # ---------------------------------------------------------
    print("Loading pretrained weights...")


    model.eval()

    print("Pretrained model loaded successfully.")

    # ---------------------------------------------------------
    # Create a dummy Sentinel-2 RGB-NIR input
    #
    # Shape:
    #   batch     = 1
    #   channels  = 4
    #   height    = 128
    #   width     = 128
    #
    # 4 channels = RGB + NIR
    # ---------------------------------------------------------
    lr = torch.rand(
        1,
        4,
        128,
        128,
        device=device,
    )

    print(f"Input shape:  {tuple(lr.shape)}")

    # ---------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------
    print("Running inference...")

    with torch.inference_mode():
        sr = model.forward(
            lr,
            sampling_steps=20,
        )

    print(f"Output shape: {tuple(sr.shape)}")

    # ---------------------------------------------------------
    # Verify expected 4x spatial scaling
    # ---------------------------------------------------------
    expected_height = 128 * 4
    expected_width = 128 * 4

    if sr.shape[-2] != expected_height or sr.shape[-1] != expected_width:
        raise RuntimeError(
            "Unexpected output resolution.\n"
            f"Expected: {expected_height}x{expected_width}\n"
            f"Received: {sr.shape[-2]}x{sr.shape[-1]}"
        )

    print("4x super-resolution verified.")
    print("Model inference test completed successfully.")


if __name__ == "__main__":
    main()
