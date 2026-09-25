# Evaluation API endpoint: returns computed metrics and analysis for completed SR jobs.
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from backend.services.evaluation_service import compute_evaluation, generate_report

router = APIRouter(prefix="/api", tags=["evaluation"])


# Return the job or raise 404.
def job_or_404(request: Request, job_id: str) -> dict:
    job = request.app.state.job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


# GET /api/jobs/{job_id}/evaluation — return full evaluation data for a completed job.
@router.get("/jobs/{job_id}/evaluation")
def get_evaluation(request: Request, job_id: str):
    job = job_or_404(request, job_id)
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']}. Evaluation requires a completed job.")
    output_path = job.get("output_path")
    crop_path = job.get("crop_path")
    if not output_path or not crop_path or not Path(output_path).is_file() or not Path(crop_path).is_file():
        raise HTTPException(404, "Job products are unavailable.")
    crop_bytes = Path(crop_path).read_bytes()
    sr_bytes = Path(output_path).read_bytes()
    return compute_evaluation(crop_bytes, sr_bytes, job)


# GET /api/jobs/{job_id}/evaluation/report — return a text report for download.
@router.get("/jobs/{job_id}/evaluation/report")
def get_evaluation_report(request: Request, job_id: str):
    job = job_or_404(request, job_id)
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']}. Evaluation requires a completed job.")
    output_path = job.get("output_path")
    crop_path = job.get("crop_path")
    if not output_path or not crop_path or not Path(output_path).is_file() or not Path(crop_path).is_file():
        raise HTTPException(404, "Job products are unavailable.")
    crop_bytes = Path(crop_path).read_bytes()
    sr_bytes = Path(output_path).read_bytes()
    evaluation = compute_evaluation(crop_bytes, sr_bytes, job)
    report = generate_report(evaluation, job)
    return PlainTextResponse(report, media_type="text/plain",
                             headers={"Content-Disposition": f"attachment; filename={job_id}_report.txt"})
