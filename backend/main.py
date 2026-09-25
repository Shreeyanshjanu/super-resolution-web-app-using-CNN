from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from filelock import FileLock, Timeout

from backend.routes.evaluation import router as evaluation_router
from backend.routes.health import router as health_router
from backend.routes.super_resolution import router as sr_router
from backend.services.job_manager import JobManager
from backend.services.srm_service import SRMService
from backend.services.storage import SceneStore
from backend.settings import DATA_ROOT, MAX_PENDING_JOBS, MAX_UPLOAD_BYTES


class UploadLimit:
    """Limit the request body before multipart parsing and disk spooling."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD_BYTES + 1024 * 1024
        headers = dict(scope.get("headers", []))
        try:
            size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            size = -1
        if size < 0 or size > limit:
            return await JSONResponse({"detail": "Upload exceeds the configured request size limit."}, status_code=413)(scope, receive, send)
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > limit:
                from starlette.exceptions import HTTPException
                raise HTTPException(413, "Upload exceeds the configured request size limit.")
            return message
        await self.app(scope, limited_receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(DATA_ROOT / "worker.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise RuntimeError("Use one API worker per storage directory; another SRM process is running.") from exc
    manager = None
    try:
        service = SRMService()
        manager = JobManager(service, DATA_ROOT / "jobs", MAX_PENDING_JOBS)
        app.state.srm_service = service
        app.state.job_manager = manager
        app.state.scenes = SceneStore(DATA_ROOT / "scenes")
        manager.recover()
        yield
    finally:
        if manager:
            manager.shutdown()
        lock.release()


app = FastAPI(title="Satellite SRM API", version="0.2.0",
              description="Selected-area Sentinel-2 super-resolution with ESA LDSR-S2.", lifespan=lifespan)
app.add_middleware(UploadLimit)
app.include_router(health_router)
app.include_router(sr_router)
app.include_router(evaluation_router)


@app.get("/")
def root():
    return {"name": "Satellite SRM API", "status": "running", "docs": "/docs"}
