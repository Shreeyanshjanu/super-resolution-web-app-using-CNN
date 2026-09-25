from __future__ import annotations
import os
import requests


class SRMApiClient:
    def __init__(self, base_url=None):
        self.base_url = (base_url or os.environ.get("SRM_API_URL", "http://127.0.0.1:8000")).rstrip("/")

    def _request(self, method, path, **kwargs):
        response = requests.request(method, self.base_url + path, timeout=kwargs.pop("timeout", 30), **kwargs)
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"API {response.status_code}: {detail}")
        return response

    def health(self):
        return self._request("GET", "/api/health", timeout=5).json()

    def upload_scene(self, file_bytes, filename, band_order="auto", value_scale=10000):
        return self._request("POST", "/api/scenes", files={"file": (filename, file_bytes, "image/tiff")},
                             data={"band_order": band_order, "value_scale": value_scale}, timeout=300).json()

    def preview(self, scene_id):
        return self._request("GET", f"/api/scenes/{scene_id}/preview").content

    def plan(self, scene_id, aoi):
        return self._request("POST", f"/api/scenes/{scene_id}/plan", json=aoi).json()

    def submit_aoi(self, scene_id, aoi, sampling_steps=100):
        return self._request("POST", "/api/super-resolve",
                             json={"scene_id": scene_id, "aoi": aoi, "sampling_steps": sampling_steps}).json()

    def get_job(self, job_id):
        return self._request("GET", f"/api/jobs/{job_id}").json()

    def list_jobs(self):
        return self._request("GET", "/api/jobs").json()

    def download_result(self, job_id, product="sr"):
        return self._request("GET", f"/api/jobs/{job_id}/download", params={"product": product}, timeout=300).content

    def delete_job(self, job_id):
        return self._request("DELETE", f"/api/jobs/{job_id}").json()

    def delete_scene(self, scene_id):
        return self._request("DELETE", f"/api/scenes/{scene_id}").json()

    def evaluation(self, job_id):
        return self._request("GET", f"/api/jobs/{job_id}/evaluation", timeout=60).json()

    def evaluation_report(self, job_id):
        return self._request("GET", f"/api/jobs/{job_id}/evaluation/report", timeout=60).content
