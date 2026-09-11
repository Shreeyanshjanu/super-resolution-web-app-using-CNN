from fastapi import APIRouter, Request


router = APIRouter(
    prefix="/api",
    tags=["system"],
)


@router.get("/health")
def health(request: Request):

    service = request.app.state.srm_service

    return {
        "status": "ok",
        "model": service.info(),
    }