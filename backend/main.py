from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routes.health import router as health_router
from backend.routes.super_resolution import (
    router as sr_router,
)
from backend.services.job_manager import JobManager
from backend.services.srm_service import SRMService


@asynccontextmanager
async def lifespan(app: FastAPI):

    print()
    print("=" * 70)
    print("STARTING SATELLITE SRM API")
    print("=" * 70)

    # Load ESA model exactly once.
    srm_service = SRMService()

    job_manager = JobManager(
        srm_service=srm_service,
    )

    app.state.srm_service = srm_service
    app.state.job_manager = job_manager

    print("API initialization complete.")

    yield

    print("Shutting down SRM API.")


app = FastAPI(
    title="Satellite SRM API",
    description=(
        "Deep Learning-Based Super Resolution "
        "Mapping for Sentinel-2 imagery."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(
    health_router
)

app.include_router(
    sr_router
)


@app.get("/")
def root():

    return {
        "name": "Satellite SRM API",
        "status": "running",
        "docs": "/docs",
    }
    
