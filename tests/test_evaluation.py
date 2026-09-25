# Tests for the evaluation service and API endpoint.
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import UploadLimit
from backend.routes.evaluation import router as evaluation_router
from backend.routes.super_resolution import router as sr_router
from backend.services.evaluation_service import (
    band_histogram,
    band_statistics,
    bicubic_upscale,
    compute_evaluation,
    compute_mae,
    compute_rmse,
    generate_report,
    spatial_detail_metrics,
)
from backend.services.job_manager import JobManager
from backend.services.storage import SceneStore
from inference.aoi import finish_output
from tests.helpers import RasterTestCase


class StubService:
    """Deterministic test double for evaluation tests."""

    def process(self, input_path, job_directory, plan, progress_callback=None, sampling_steps=100):
        with rasterio.open(input_path) as src:
            data = np.repeat(np.repeat(src.read(), 4, axis=1), 4, axis=2)
            profile = src.profile.copy()
            profile.update(width=src.width * 4, height=src.height * 4,
                           transform=src.transform * Affine.scale(0.25))
        raw = job_directory / "raw.tif"
        with rasterio.open(raw, "w", **profile) as dst:
            dst.write(data)
        if progress_callback:
            progress_callback(plan["patches"], plan["patches"])
        output = finish_output(raw, job_directory / "crop.tif", job_directory / "result.tif")
        return {"output_path": str(output), "inference_seconds": 0.01,
                "seconds_per_patch": 0.01, "sampling_steps": sampling_steps}


class EvaluationServiceTests(RasterTestCase):
    # Band statistics should return mean/std/min/max for valid data.
    def test_band_statistics_computes_correctly(self):
        data = np.ma.array([[[1.0, 2.0], [3.0, 4.0]]])
        stats = band_statistics(data)
        self.assertEqual(len(stats), 1)
        self.assertAlmostEqual(stats[0]["mean"], 2.5)
        self.assertAlmostEqual(stats[0]["std"], np.std([1, 2, 3, 4]))
        self.assertAlmostEqual(stats[0]["min"], 1.0)
        self.assertAlmostEqual(stats[0]["max"], 4.0)

    # Band statistics should handle empty/masked data.
    def test_band_statistics_handles_empty_data(self):
        data = np.ma.array([[[0.0, 0.0]]], mask=[[[True, True]]])
        stats = band_statistics(data)
        self.assertEqual(stats[0]["valid_pixels"], 0)
        self.assertIsNone(stats[0]["mean"])

    # Bicubic upscale should produce correct output shape.
    def test_bicubic_upscale_shape(self):
        data = np.ma.array(np.ones((4, 32, 32), dtype=np.float32))
        result = bicubic_upscale(data, 128, 128)
        self.assertEqual(result.shape, (4, 128, 128))

    # MAE should compute correctly for matching arrays.
    def test_compute_mae_identical_arrays(self):
        data = np.ma.array([[[1.0, 2.0], [3.0, 4.0]]])
        mae = compute_mae(data, data)
        self.assertAlmostEqual(mae[0], 0.0)

    # MAE should compute correctly for different arrays.
    def test_compute_mae_different_arrays(self):
        pred = np.ma.array([[[1.0, 2.0]]])
        target = np.ma.array([[[2.0, 4.0]]])
        mae = compute_mae(pred, target)
        self.assertAlmostEqual(mae[0], 1.5)

    # RMSE should compute correctly.
    def test_compute_rmse(self):
        pred = np.ma.array([[[1.0, 2.0]]])
        target = np.ma.array([[[3.0, 4.0]]])
        rmse = compute_rmse(pred, target)
        self.assertAlmostEqual(rmse[0], 2.0)

    # Spatial detail metrics should return all required fields.
    def test_spatial_detail_metrics_fields(self):
        data = np.ma.array(np.random.rand(4, 32, 32).astype(np.float32))
        metrics = spatial_detail_metrics(data)
        self.assertEqual(len(metrics), 4)
        for m in metrics:
            self.assertIn("mean_gradient", m)
            self.assertIn("edge_density", m)
            self.assertIn("high_frequency_energy", m)
            self.assertIn("local_contrast", m)

    # Band histogram should return counts and edges.
    def test_band_histogram(self):
        data = np.ma.array(np.random.rand(32, 32).astype(np.float32))
        hist = band_histogram(data, n_bins=20)
        self.assertEqual(len(hist["counts"]), 20)
        self.assertEqual(len(hist["bin_edges"]), 21)
        self.assertIsNotNone(hist["mean"])
        self.assertIsNotNone(hist["std"])


class EvaluationAPITests(RasterTestCase):
    def setUp(self):
        super().setUp()
        self.service = StubService()
        self.manager = JobManager(self.service, self.root / "jobs", max_pending=2)
        self.app = FastAPI()
        self.app.state.job_manager = self.manager
        self.app.state.scenes = SceneStore(self.root / "scenes")
        self.app.include_router(sr_router)
        self.app.include_router(evaluation_router)
        self.app.add_middleware(UploadLimit)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.manager.shutdown()
        self.client.close()
        super().tearDown()

    def upload(self):
        response = self.client.post("/api/scenes",
                                    files={"file": ("source.tif", self.source.read_bytes(), "image/tiff")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def wait_job(self, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = self.client.get(f"/api/jobs/{job_id}").json()
            if result["status"] in ("completed", "failed"):
                return result
            time.sleep(0.02)
        self.fail("Test job timed out")

    def run_job(self):
        scene = self.upload()
        response = self.client.post("/api/super-resolve",
                                    json={"scene_id": scene["scene_id"], "aoi": {"mode": "center"}})
        self.assertEqual(response.status_code, 202, response.text)
        job = self.wait_job(response.json()["job_id"])
        self.assertEqual(job["status"], "completed", job)
        return job

    # Evaluation endpoint should return data for a completed job.
    def test_evaluation_endpoint_returns_data(self):
        job = self.run_job()
        response = self.client.get(f"/api/jobs/{job['job_id']}/evaluation")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("overview", data)
        self.assertIn("band_analysis", data)
        self.assertIn("spectral", data)
        self.assertIn("spatial_detail", data)
        self.assertIn("model_info", data)
        self.assertIn("limitations", data)

    # Evaluation should indicate no reference is available.
    def test_evaluation_no_reference_available(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        self.assertFalse(data["reference_available"])
        for band in data["band_analysis"]:
            self.assertIsNone(band["reference_metrics"]["psnr"])
            self.assertIsNone(band["reference_metrics"]["ssim"])
            self.assertIn("unavailable_reason", band["reference_metrics"])

    # Evaluation should not fabricate metrics.
    def test_evaluation_does_not_fabricate_metrics(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        for band in data["band_analysis"]:
            rm = band["reference_metrics"]
            self.assertIsNone(rm["psnr"])
            self.assertIsNone(rm["ssim"])
            self.assertIsNone(rm["rmse"])
            self.assertIsNone(rm["mae"])

    # Uncertainty should be marked as unavailable.
    def test_evaluation_uncertainty_unavailable(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        self.assertFalse(data["uncertainty"]["available"])
        self.assertIn("message", data["uncertainty"])

    # Confusion matrix should be marked as not applicable.
    def test_evaluation_confusion_matrix_not_applicable(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        self.assertFalse(data["confusion_matrix"]["applicable"])
        self.assertIn("message", data["confusion_matrix"])

    # Evaluation report endpoint should return text.
    def test_evaluation_report_endpoint(self):
        job = self.run_job()
        response = self.client.get(f"/api/jobs/{job['job_id']}/evaluation/report")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/plain", response.headers["content-type"])
        text = response.text
        self.assertIn("EVALUATION REPORT", text)
        self.assertIn("INPUT", text)
        self.assertIn("OUTPUT", text)
        self.assertIn("MODEL", text)

    # Evaluation should fail for non-completed jobs.
    def test_evaluation_fails_for_active_job(self):
        scene = self.upload()
        with patch.object(self.manager, "enqueue"):
            response = self.client.post("/api/super-resolve",
                                        json={"scene_id": scene["scene_id"], "aoi": {"mode": "center"}})
        job_id = response.json()["job_id"]
        response = self.client.get(f"/api/jobs/{job_id}/evaluation")
        self.assertEqual(response.status_code, 409)

    # Evaluation overview should contain correct dimensions.
    def test_evaluation_overview_dimensions(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        ov = data["overview"]
        self.assertEqual(ov["input"]["width"], 128)
        self.assertEqual(ov["input"]["height"], 128)
        self.assertEqual(ov["output"]["width"], 512)
        self.assertEqual(ov["output"]["height"], 512)
        self.assertEqual(ov["super_resolution"]["scale_factor"], 4)

    # Band analysis should have 4 bands with correct names.
    def test_evaluation_band_analysis_structure(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        ba = data["band_analysis"]
        self.assertEqual(len(ba), 4)
        names = [b["name"] for b in ba]
        self.assertIn("B04", names[0])
        self.assertIn("B08", names[3])

    # Generate report should produce valid text.
    def test_generate_report(self):
        job = self.run_job()
        raw = self.manager.get_job(job["job_id"])
        crop_bytes = Path(raw["crop_path"]).read_bytes()
        sr_bytes = Path(raw["output_path"]).read_bytes()
        evaluation = compute_evaluation(crop_bytes, sr_bytes, raw)
        report = generate_report(evaluation, raw)
        self.assertIn("SATELLITE SUPER-RESOLUTION EVALUATION REPORT", report)
        self.assertIn("N/A — no valid reference available", report)
        self.assertIn("LIMITATIONS", report)

    # Model info should contain expected fields.
    def test_evaluation_model_info(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        mi = data["model_info"]
        self.assertEqual(mi["model"], "ESA LDSR-S2")
        self.assertEqual(mi["architecture"], "Latent Diffusion")
        self.assertEqual(mi["scale_factor"], "4×")
        self.assertIn(20, mi["sampling_steps_options"])
        self.assertIn(100, mi["sampling_steps_options"])

    # Dataset info should be present.
    def test_evaluation_dataset_info(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        di = data["dataset_info"]
        self.assertEqual(di["dataset"], "SEN2NAIP")
        self.assertEqual(di["publication_year"], 2024)

    # Limitations should be a non-empty list.
    def test_evaluation_limitations(self):
        job = self.run_job()
        data = self.client.get(f"/api/jobs/{job['job_id']}/evaluation").json()
        self.assertIsInstance(data["limitations"], list)
        self.assertGreater(len(data["limitations"]), 0)
