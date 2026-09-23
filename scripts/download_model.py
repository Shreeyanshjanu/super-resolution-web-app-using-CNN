"""Download and verify the exact pretrained checkpoint used by this application."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import requests

from inference.model import PROJECT_ROOT, CHECKPOINT_NAME

SHA256 = "e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a"
URL = f"https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/{CHECKPOINT_NAME}"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    path = Path(os.environ.get("SRM_CHECKPOINT", str(PROJECT_ROOT / CHECKPOINT_NAME))).resolve()
    if path.exists():
        if checksum(path) != SHA256:
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}. The existing file was not modified.")
        print(f"Checkpoint verified: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with requests.get(URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if checksum(temporary) != SHA256:
            raise RuntimeError("Downloaded checkpoint checksum mismatch.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Checkpoint downloaded and verified: {path}")


if __name__ == "__main__":
    main()
