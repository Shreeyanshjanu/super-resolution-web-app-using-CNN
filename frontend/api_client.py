from __future__ import annotations

import time
from typing import Any

import requests


class SRMApiClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
    ) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/health",
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    def submit_file(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> dict[str, Any]:

        files = {
            "file": (
                filename,
                file_bytes,
                "image/tiff",
            )
        }

        response = requests.post(
            f"{self.base_url}/api/super-resolve",
            files=files,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    def get_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:

        response = requests.get(
            f"{self.base_url}/api/jobs/{job_id}",
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 3600,
    ) -> dict[str, Any]:

        start = time.monotonic()

        while True:

            job = self.get_job(job_id)

            status = job.get("status")

            if status in {
                "completed",
                "failed",
            }:
                return job

            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    "Timed out waiting for SR job."
                )

            time.sleep(poll_interval)

    def download_result(
        self,
        job_id: str,
    ) -> bytes:

        response = requests.get(
            f"{self.base_url}/api/jobs/{job_id}/download",
            timeout=300,
        )

        response.raise_for_status()

        return response.content