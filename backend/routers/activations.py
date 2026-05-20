from fastapi import APIRouter, HTTPException, Query
import numpy as np

from backend.services.cache_store import cache_store

router = APIRouter(prefix="/api/activations", tags=["activations"])


@router.get("/keys")
async def get_keys():
    if not cache_store.cache:
        raise HTTPException(status_code=400, detail="No cache available. Run /api/inference/run_with_cache first.")
    return {"keys": cache_store.keys_with_shapes(), "str_tokens": cache_store.str_tokens}


@router.get("/attention/{layer}")
async def get_attention(layer: int):
    if not cache_store.cache:
        raise HTTPException(status_code=400, detail="No cache available. Run /api/inference/run_with_cache first.")

    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in cache_store.cache:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found in cache. Run run_with_cache first.")

    tensor = cache_store.cache[key]
    arr = tensor.detach().cpu().float().numpy()[0]  # [n_heads, query_pos, key_pos]

    n_heads, n_positions, _ = arr.shape
    return {
        "layer": layer,
        "n_heads": n_heads,
        "n_positions": n_positions,
        "str_tokens": cache_store.str_tokens,
        "patterns": arr.tolist(),
    }


@router.get("/{key:path}")
async def get_activation(key: str, max_elements: int = Query(4096)):
    if not cache_store.cache:
        raise HTTPException(status_code=400, detail="No cache available. Run /api/inference/run_with_cache first.")
    if key not in cache_store.cache:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found in cache.")

    tensor = cache_store.cache[key]
    arr = tensor.detach().cpu().float().numpy()
    shape = list(arr.shape)

    stats = {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }

    total = int(np.prod(shape))
    if total <= max_elements:
        data = arr.tolist()
    else:
        # Truncate last axis so total fits within max_elements
        leading = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        last_dim = max(1, max_elements // leading)
        arr_trunc = arr[..., :last_dim]
        data = arr_trunc.tolist()
        shape = list(arr_trunc.shape)

    return {
        "key": key,
        "shape": shape,
        "original_shape": list(tensor.shape),
        "dtype": "float32",
        "stats": stats,
        "data": data,
        "str_tokens": cache_store.str_tokens,
    }
