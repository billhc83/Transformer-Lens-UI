import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/models", tags=["models"])


class LoadModelRequest(BaseModel):
    model_name: str


@router.get("/available")
def get_available_models():
    return {"models": model_manager.list_available_models()}


@router.post("/load")
async def load_model(req: LoadModelRequest):
    try:
        loop = asyncio.get_event_loop()
        config = await loop.run_in_executor(None, model_manager.load_model, req.model_name)
        return {"status": "loaded", "config": config}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/loaded")
def get_loaded_model():
    info = model_manager.get_loaded_model_info()
    if info is None:
        return {"loaded": False}
    return {"loaded": True, "config": info}
