from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from scipy.special import softmax

from backend.services.model_manager import model_manager
from backend.services.cache_store import cache_store

router = APIRouter(prefix="/api/inference", tags=["inference"])


class TokenizeRequest(BaseModel):
    text: str


class ForwardRequest(BaseModel):
    text: str
    top_k: int = 10


class RunWithCacheRequest(BaseModel):
    text: str


class LogitLensRequest(BaseModel):
    top_k: int = 5


@router.post("/tokenize")
async def tokenize_endpoint(request: TokenizeRequest):
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    tokens = model.to_tokens(request.text, prepend_bos=True)
    token_ids = tokens[0].tolist()
    str_tokens = model.to_str_tokens(request.text, prepend_bos=True)

    return {
        "token_ids": token_ids,
        "str_tokens": list(str_tokens),
        "n_tokens": len(token_ids),
    }


@router.post("/forward")
async def forward_endpoint(request: ForwardRequest):
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    tokens = model.to_tokens(request.text, prepend_bos=True)
    logits = model(tokens, return_type="logits")
    logits_np = logits[0].detach().cpu().float().numpy()  # [seq_len, d_vocab]

    str_tokens = list(model.to_str_tokens(request.text, prepend_bos=True))
    top_k = min(request.top_k, logits_np.shape[-1])

    predictions = []
    for i, row in enumerate(logits_np):
        probs = softmax(row).astype(float)
        top_k_indices = np.argsort(-probs)[:top_k]
        top_k_entries = []
        for tk_id in top_k_indices:
            tk_id_int = int(tk_id)
            tk_str = model.tokenizer.convert_ids_to_tokens(tk_id_int) or str(tk_id_int)
            top_k_entries.append({
                "token_id": tk_id_int,
                "token_str": tk_str,
                "probability": float(probs[tk_id]),
                "logit": float(row[tk_id]),
            })
        predictions.append({
            "position": i,
            "token": str_tokens[i] if i < len(str_tokens) else f"[{i}]",
            "top_k": top_k_entries,
        })

    return {
        "logits_shape": list(logits_np.shape),
        "str_tokens": str_tokens,
        "predictions": predictions,
    }


@router.post("/run_with_cache")
async def run_with_cache_endpoint(request: RunWithCacheRequest):
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    tokens = model.to_tokens(request.text, prepend_bos=True)
    str_tokens = list(model.to_str_tokens(request.text, prepend_bos=True))

    _, cache = model.run_with_cache(tokens)
    cache_dict = dict(cache.cache_dict)
    cache_store.set(cache_dict, str_tokens)

    keys = [{"key": k, "shape": list(v.shape)} for k, v in sorted(cache_dict.items())]
    return {
        "keys": keys,
        "n_keys": len(keys),
        "str_tokens": str_tokens,
    }


@router.post("/logit_lens")
async def logit_lens_endpoint(request: LogitLensRequest):
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if not cache_store.cache:
        raise HTTPException(status_code=400, detail="No cache. Run /api/inference/run_with_cache first.")

    n_layers = model.cfg.n_layers
    str_tokens = cache_store.str_tokens
    results = []

    for layer_idx in range(n_layers + 1):
        if layer_idx == 0:
            embed = cache_store.cache.get("hook_embed")
            pos_embed = cache_store.cache.get("hook_pos_embed")
            if embed is None:
                continue
            resid = embed + pos_embed if pos_embed is not None else embed
            label = "embed"
        else:
            key = f"blocks.{layer_idx - 1}.hook_resid_post"
            if key not in cache_store.cache:
                continue
            resid = cache_store.cache[key]
            label = f"L{layer_idx - 1}"

        with torch.no_grad():
            ln_out = model.ln_final(resid)
            logits = model.unembed(ln_out)

        logits_np = logits[0].detach().cpu().float().numpy()  # [pos, d_vocab]
        probs = softmax(logits_np, axis=-1)
        top_k = min(request.top_k, probs.shape[-1])

        preds_per_pos = []
        for pos_i, prob_row in enumerate(probs):
            top_indices = np.argsort(-prob_row)[:top_k]
            preds_per_pos.append({
                "position": pos_i,
                "top_k": [
                    {
                        "token_id": int(idx),
                        "token_str": model.tokenizer.decode([int(idx)]),
                        "probability": float(prob_row[idx]),
                    }
                    for idx in top_indices
                ],
            })

        results.append({"layer": layer_idx, "label": label, "predictions": preds_per_pos})

    return {"results": results, "str_tokens": str_tokens, "n_layers": n_layers}
