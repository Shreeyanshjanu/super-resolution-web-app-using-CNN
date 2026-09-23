from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal
from uuid import uuid4

import rasterio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.settings import MAX_PATCHES, MAX_UPLOAD_BYTES, SAMPLING_STEPS
from backend.services.job_manager import QueueFullError
from backend.services.storage import remove_object
from inference.aoi import AOI, inspect_raster, plan_crop, write_crop
from inference.preview import create_preview

router = APIRouter(prefix="/api", tags=["super-resolution"])
LOG = logging.getLogger(__name__)


class SubmitJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    aoi: AOI  # Explicit selection is mandatory; there is no full-image fallback.
    sampling_steps: Literal[20, 50, 100] = SAMPLING_STEPS


def scene_or_404(request, scene_id):
    scene = request.app.state.scenes.get(scene_id)
    if scene is None:
        raise HTTPException(404, "Scene not found. Upload the image again.")
    return scene


def job_or_404(request, job_id):
    job = request.app.state.job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


def public_job(job):
    return {k: v for k, v in job.items() if not k.endswith("_path")}


@router.post("/scenes", status_code=201)
def upload_scene(request: Request, file: UploadFile = File(...),
                 band_order: Literal["auto", "rgbn", "bgrn"] = Form("auto"),
                 value_scale: int = Form(10000)):
    if not file.filename or Path(file.filename).suffix.lower() not in (".tif", ".tiff"):
        raise HTTPException(400, "Upload a .tif or .tiff file.")
    store = request.app.state.scenes
    scene_id = uuid4().hex
    directory = store.path(scene_id)
    directory.mkdir()
    path = directory / "source.tif"
    digest, total = hashlib.sha256(), 0
    try:
        with path.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                digest.update(chunk)
                output.write(chunk)
        if total == 0:
            raise ValueError("Uploaded file is empty.")
        metadata = inspect_raster(path, band_order, value_scale)
        preview = create_preview(path, directory / "preview.png", metadata["band_indexes"])
        scene = dict(scene_id=scene_id, filename=Path(file.filename).name,
                     sha256=digest.hexdigest(), size_bytes=total, **metadata, **preview)
        store.save(scene)
        return scene
    except (ValueError, rasterio.errors.RasterioError) as exc:
        remove_object(store.directory, scene_id)
        raise HTTPException(422, str(exc)) from exc
    except Exception:
        remove_object(store.directory, scene_id)
        raise
    finally:
        file.file.close()


@router.get("/scenes/{scene_id}")
def get_scene(request: Request, scene_id: str):
    return scene_or_404(request, scene_id)


@router.get("/scenes/{scene_id}/preview")
def scene_preview(request: Request, scene_id: str):
    scene_or_404(request, scene_id)
    return FileResponse(request.app.state.scenes.path(scene_id) / "preview.png", media_type="image/png")


@router.post("/scenes/{scene_id}/plan")
def preview_plan(request: Request, scene_id: str, aoi: AOI):
    scene_or_404(request, scene_id)
    try:
        return plan_crop(request.app.state.scenes.path(scene_id) / "source.tif", aoi, MAX_PATCHES)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/super-resolve", status_code=202)
def create_super_resolution_job(request: Request, submission: SubmitJob):
    scene = scene_or_404(request, submission.scene_id)
    source = request.app.state.scenes.path(submission.scene_id) / "source.tif"
    manager = request.app.state.job_manager
    job_id = None
    try:
        plan = plan_crop(source, submission.aoi, MAX_PATCHES)
        job_id = manager.create_job(scene["filename"], scene_id=scene["scene_id"],
                                    source_sha256=scene["sha256"], plan=plan,
                                    selection=submission.aoi.model_dump(), sampling_steps=submission.sampling_steps)
        directory = manager.jobs_directory / job_id
        crop, model_input = write_crop(source, directory, plan, scene["band_indexes"], scene["value_scale"])
        manager.update_job(job_id, status="queued", input_path=str(model_input), crop_path=str(crop),
                           message=f"Queued: {plan['patches']} patches.")
        manager.enqueue(job_id)
        return public_job(manager.get_job(job_id))
    except QueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    except (ValueError, rasterio.errors.RasterioError) as exc:
        if job_id:
            manager.update_job(job_id, status="failed", error=str(exc), completed_at=manager._now())
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        if job_id:
            manager.update_job(job_id, status="failed", error=str(exc), completed_at=manager._now())
        LOG.exception("Could not prepare AOI")
        raise HTTPException(500, "Could not prepare the selected area.") from exc


@router.get("/jobs")
def list_jobs(request: Request):
    return [public_job(job) for job in request.app.state.job_manager.list_jobs()]


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    return public_job(job_or_404(request, job_id))


@router.get("/jobs/{job_id}/download")
def download_result(request: Request, job_id: str, product: Literal["sr", "crop"] = "sr"):
    job = job_or_404(request, job_id)
    if product == "sr" and job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']}.")
    value = job.get("output_path" if product == "sr" else "crop_path")
    if not value or not Path(value).is_file():
        raise HTTPException(404, "Product is unavailable.")
    return FileResponse(value, media_type="image/tiff", filename=f"{job_id}_{product}.tif")


@router.delete("/jobs/{job_id}")
def delete_job(request: Request, job_id: str):
    try:
        if not request.app.state.job_manager.delete_job(job_id):
            raise HTTPException(404, "Job not found.")
        return {"deleted": job_id}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/scenes/{scene_id}")
def delete_scene(request: Request, scene_id: str):
    scene_or_404(request, scene_id)
    if any(job.get("scene_id") == scene_id for job in request.app.state.job_manager.list_jobs()):
        raise HTTPException(409, "Delete this scene's jobs first.")
    remove_object(request.app.state.scenes.directory, scene_id)
    return {"deleted": scene_id}
