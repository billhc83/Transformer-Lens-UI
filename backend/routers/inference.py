from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from scipy.special import softmax
import einops
from transformer_lens import ActivationCache

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


class AttributionRequest(BaseModel):
    answer_token: str
    mode: str = "full"  # "full" | "by_layer" | "by_head"


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
    # Drop the BOS token from display — it's needed for inference but shouldn't
    # appear as a chip. Logits at position i predict the token at position i+1,
    # so after slicing, logits_np[i] still predicts what follows str_tokens[i].
    str_tokens = str_tokens[1:]
    logits_np = logits_np[1:]

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


@router.post("/attribution")
async def attribution_endpoint(request: AttributionRequest):
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if not cache_store.cache:
        raise HTTPException(status_code=400, detail="No cache. Run /api/inference/run_with_cache first.")

    try:
        answer_token_id = int(model.to_single_token(request.answer_token))
    except Exception:
        toks = model.to_tokens(request.answer_token, prepend_bos=False)
        answer_token_id = int(toks[0, 0].item())

    cache = ActivationCache(cache_store.cache, model)
    str_tokens = cache_store.str_tokens

    with torch.no_grad():
        if request.mode == "by_head":
            head_stack, head_labels = cache.stack_head_results(return_labels=True)
            mlp_parts = []
            mlp_labels = []
            for l in range(model.cfg.n_layers):
                key = f"blocks.{l}.hook_mlp_out"
                if key in cache_store.cache:
                    mlp_parts.append(cache_store.cache[key])
                    mlp_labels.append(f"L{l}_mlp")
            if mlp_parts:
                mlp_tensor = torch.stack(mlp_parts, dim=0)
                combined = torch.cat([head_stack, mlp_tensor], dim=0)
                labels = list(head_labels) + mlp_labels
            else:
                combined = head_stack
                labels = list(head_labels)
            combined_ln = cache.apply_ln_to_stack(combined, layer=-1, has_batch_dim=True)

        else:
            stack, raw_labels = cache.decompose_resid(mode="all", return_labels=True, apply_ln=False)
            combined_ln = cache.apply_ln_to_stack(stack, layer=-1, has_batch_dim=True)
            labels = list(raw_labels)

            if request.mode == "by_layer":
                new_parts = []
                new_labels = []
                embed_i = labels.index("embed") if "embed" in labels else 0
                pos_i = labels.index("pos_embed") if "pos_embed" in labels else -1
                embed_contrib = combined_ln[embed_i] + (combined_ln[pos_i] if pos_i >= 0 else 0)
                new_parts.append(embed_contrib)
                new_labels.append("embed")
                for l in range(model.cfg.n_layers):
                    attn_lbl = f"{l}_attn_out"
                    mlp_lbl = f"{l}_mlp_out"
                    ai = labels.index(attn_lbl) if attn_lbl in labels else -1
                    mi = labels.index(mlp_lbl) if mlp_lbl in labels else -1
                    if ai >= 0 and mi >= 0:
                        new_parts.append(combined_ln[ai] + combined_ln[mi])
                    elif ai >= 0:
                        new_parts.append(combined_ln[ai])
                    elif mi >= 0:
                        new_parts.append(combined_ln[mi])
                    else:
                        continue
                    new_labels.append(f"L{l}")
                combined_ln = torch.stack(new_parts, dim=0)
                labels = new_labels

        # Project only onto the answer token direction to avoid building the
        # full (comp, batch, pos, d_vocab) tensor, which is ~280 MB for gpt2 by_head.
        w_answer = model.W_U[:, answer_token_id]  # [d_model]
        scores = einops.einsum(
            combined_ln[:, 0], w_answer,
            "comp pos d_model, d_model -> comp pos",
        ).detach().cpu().float().numpy()

    return {
        "scores": scores.tolist(),
        "labels": labels,
        "str_tokens": str_tokens,
        "answer_token_id": answer_token_id,
        "answer_token_str": request.answer_token,
        "mode": request.mode,
    }
