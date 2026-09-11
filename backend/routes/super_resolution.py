from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/api",
    tags=["super-resolution"],
)


ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
}


@router.post("/super-resolve")
async def create_super_resolution_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .tif and .tiff files "
                "are currently supported."
            ),
        )

    job_manager = (
        request.app.state.job_manager
    )

    job_id = job_manager.create_job(
        file.filename
    )

    job_dir = (
        job_manager.jobs_directory / job_id
    )

    input_path = (
        job_dir / f"input{extension}"
    )

    try:

        with input_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as exc:

        job_manager.update_job(
            job_id,
            status="failed",
            message="Could not save uploaded file.",
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded file.",
        ) from exc

    background_tasks.add_task(
        job_manager.run_job,
        job_id,
        input_path,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": (
            "File uploaded. "
            "Super-resolution job started."
        ),
    }


@router.get("/jobs/{job_id}")
def get_job(
    request: Request,
    job_id: str,
):

    job_manager = (
        request.app.state.job_manager
    )

    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return job


@router.get("/jobs/{job_id}/download")
def download_result(
    request: Request,
    job_id: str,
):

    job_manager = (
        request.app.state.job_manager
    )

    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job is not completed. "
                f"Current status: {job['status']}"
            ),
        )

    output_path = Path(
        job["output_path"]
    )

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Result file no longer exists.",
        )

    return FileResponse(
        path=output_path,
        media_type="image/tiff",
        filename=f"{job_id}_sr.tif",
    )