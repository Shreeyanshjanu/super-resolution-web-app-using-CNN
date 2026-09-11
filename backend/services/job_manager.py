from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from backend.services.srm_service import SRMService


class JobManager:

    def __init__(
        self,
        srm_service: SRMService,
        jobs_directory: str = "outputs/jobs",
    ) -> None:

        self.srm_service = srm_service
        self.jobs_directory = Path(jobs_directory)

        self.jobs_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.jobs: dict[str, dict] = {}

        self._jobs_lock = Lock()

    def create_job(
        self,
        filename: str,
    ) -> str:

        job_id = uuid4().hex

        job_dir = (
            self.jobs_directory / job_id
        )

        job_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        job = {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "message": "Job queued.",
            "created_at": self._now(),
            "completed_at": None,
            "input_path": None,
            "output_path": None,
            "error": None,
        }

        with self._jobs_lock:
            self.jobs[job_id] = job

        return job_id

    def get_job(
        self,
        job_id: str,
    ) -> dict | None:

        with self._jobs_lock:
            return self.jobs.get(job_id)

    def run_job(
        self,
        job_id: str,
        input_path: Path,
    ) -> None:

        self.update_job(
            job_id,
            status="processing",
            progress=10,
            message="Starting super-resolution.",
            input_path=str(input_path),
        )

        try:

            job = self.get_job(job_id)

            if job is None:
                raise RuntimeError(
                    f"Unknown job: {job_id}"
                )

            job_dir = (
                self.jobs_directory / job_id
            )

            output_path = (
                self.srm_service.process(
                    input_path=input_path,
                    job_directory=job_dir,
                )
            )

            self.update_job(
                job_id,
                status="completed",
                progress=100,
                message="Super-resolution completed.",
                output_path=str(output_path),
                completed_at=self._now(),
            )

        except Exception as exc:

            self.update_job(
                job_id,
                status="failed",
                progress=100,
                message="Super-resolution failed.",
                error=str(exc),
                completed_at=self._now(),
            )

    def update_job(
        self,
        job_id: str,
        **changes,
    ) -> None:

        with self._jobs_lock:

            if job_id not in self.jobs:
                return

            self.jobs[job_id].update(changes)

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()