from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import torch
from typing import List
from transformer_lens.patching import get_act_patch_resid_pre, get_act_patch_attn_out, get_act_patch_mlp_out
from transformer_lens import ActivationCache

from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/patching", tags=["patching"])


class PatchingRequest(BaseModel):
    clean_prompt: str
    corrupted_prompt: str
    answer_token: str
    baseline_token: str
    activation_types: List[str] = Field(default=["residual", "attn_out", "mlp_out"])


@router.post("/run")
async def run_patching(request: PatchingRequest):
    try:
        model = model_manager.model
        if model is None:
            raise HTTPException(status_code=400, detail="No model loaded")

        clean_tokens = model.to_tokens(request.clean_prompt, prepend_bos=True)
        corrupted_tokens = model.to_tokens(request.corrupted_prompt, prepend_bos=True)

        if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt token lengths differ: clean={clean_tokens.shape[1]}, corrupted={corrupted_tokens.shape[1]}. Pad prompts to equal length.",
            )

        logits_clean, clean_cache = model.run_with_cache(clean_tokens)
        logits_corrupted = model(corrupted_tokens)

        answer_id = int(model.to_single_token(request.answer_token))
        baseline_id = int(model.to_single_token(request.baseline_token))
        seq_len = clean_tokens.shape[1]
        last = seq_len - 1

        clean_diff = (logits_clean[0, last, answer_id] - logits_clean[0, last, baseline_id]).item()
        corr_diff = (logits_corrupted[0, last, answer_id] - logits_corrupted[0, last, baseline_id]).item()
        denom = (clean_diff - corr_diff) if abs(clean_diff - corr_diff) > 1e-6 else 1.0

        def metric(logits):
            return (logits[0, last, answer_id] - logits[0, last, baseline_id] - corr_diff) / denom

        clean_cache_obj = ActivationCache(dict(clean_cache.cache_dict), model)
        results = {}

        for act_type in request.activation_types:
            if act_type == "residual":
                result = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache_obj, metric)
            elif act_type == "attn_out":
                result = get_act_patch_attn_out(model, corrupted_tokens, clean_cache_obj, metric)
            elif act_type == "mlp_out":
                result = get_act_patch_mlp_out(model, corrupted_tokens, clean_cache_obj, metric)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown activation type: {act_type}")

            results[act_type] = result.detach().cpu().float().tolist()

        return {
            "results": results,
            "str_tokens": list(model.to_str_tokens(request.clean_prompt, prepend_bos=True)),
            "n_layers": model.cfg.n_layers,
            "clean_diff": clean_diff,
            "corrupted_diff": corr_diff,
            "answer_token": request.answer_token,
            "baseline_token": request.baseline_token,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
