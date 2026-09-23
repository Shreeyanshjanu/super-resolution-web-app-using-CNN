"""A fast browser-test API. Never use this interpolation double for real SR."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI

from backend.routes.health import router as health
from backend.routes.super_resolution import router
from backend.services.job_manager import JobManager
from backend.services.storage import SceneStore
from tests.test_api import StubService


class BrowserService(StubService):
    def info(self):
        return {"model":"ESA LDSR-S2 (UI test double)","device":"cpu","sampling_steps":100,"sampling_options":[20,50,100],"max_patches":4}


@asynccontextmanager
async def lifespan(app):
    root=Path("outputs/browser-test")
    service=BrowserService()
    manager=JobManager(service,root/"jobs")
    app.state.srm_service=service
    app.state.job_manager=manager
    app.state.scenes=SceneStore(root/"scenes")
    yield
    manager.shutdown()


app=FastAPI(lifespan=lifespan)
app.include_router(health)
app.include_router(router)
