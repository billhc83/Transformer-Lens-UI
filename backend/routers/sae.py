from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.cache_store import cache_store
from backend.services.model_manager import model_manager
from backend.services.sae_manager import sae_manager

router = APIRouter(prefix="/api/sae", tags=["sae"])


class LoadRequest(BaseModel):
    layer: int  # 0-11 = hook_resid_pre, 12 = final hook_resid_post


class DecomposeRequest(BaseModel):
    layer: int
    top_k: int = 15


@router.get("/status")
async def sae_status():
    model_info = model_manager.get_loaded_model_info()
    model_name = model_info["name"] if model_info else None
    release = sae_manager.get_release(model_name) if model_name else None
    return {
        "model": model_name,
        "supported": release is not None,
        "release": release,
        "loaded_layer_keys": sae_manager.loaded_layers,
    }


@router.post("/load")
async def load_sae(request: LoadRequest):
    model_info = model_manager.get_loaded_model_info()
    if not model_info:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    if request.layer < 0 or request.layer > 12:
        raise HTTPException(status_code=400, detail="layer must be 0-12 (12 = final resid_post).")

    try:
        info = sae_manager.load_layer(model_info["name"], request.layer)
        return info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAE download failed: {e}")


@router.post("/decompose")
async def decompose_sae(request: DecomposeRequest):
    model_info = model_manager.get_loaded_model_info()
    if not model_info:
        raise HTTPException(status_code=400, detail="No model loaded.")

    if not cache_store.cache:
        raise HTTPException(
            status_code=400,
            detail="No activation cache. Run /api/inference/run_with_cache first.",
        )

    if request.layer < 0 or request.layer > 12:
        raise HTTPException(status_code=400, detail="layer must be 0-12.")

    cache_key = (
        "blocks.11.hook_resid_post"
        if request.layer == 12
        else f"blocks.{request.layer}.hook_resid_pre"
    )

    if cache_key not in cache_store.cache:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Key '{cache_key}' not in cache. "
                "Ensure run_with_cache ran in 'all' mode and this layer exists."
            ),
        )

    activation = cache_store.cache[cache_key]  # [1, pos, d_model]

    try:
        result = sae_manager.decompose(
            activation,
            model_info["name"],
            request.layer,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decomposition failed: {e}")

    return {
        **result,
        "str_tokens": cache_store.str_tokens,
    }
