from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SRM_DATA_ROOT", str(PROJECT_ROOT / "outputs" / "app"))).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("SRM_MAX_UPLOAD_MB", "256")) * 1024 * 1024
MAX_PATCHES = int(os.environ.get("SRM_MAX_PATCHES", "4"))
MAX_PENDING_JOBS = int(os.environ.get("SRM_MAX_PENDING_JOBS", "8"))
SAMPLING_STEPS = 100
SAMPLING_OPTIONS = (20, 50, 100)
if not 1 <= MAX_PATCHES <= 64 or not 1 <= MAX_PENDING_JOBS <= 100 or MAX_UPLOAD_BYTES <= 0:
    raise ValueError("Invalid SRM limits.")
