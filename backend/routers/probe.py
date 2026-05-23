import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.special import softmax as scipy_softmax

from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/probe", tags=["probe"])

SESSIONS_DIR = Path("backend/data/probe_sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class ProbeSessionRequest(BaseModel):
    concept: str
    normalization_cues: list[str]
    control_cues: list[str]
    probe_prompts: list[str]
    top_k: int = 20
    notes: str = ""


def _decode_token(model, token_id: int) -> str:
    try:
        return model.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except Exception:
        return f"[{token_id}]"


def _get_last_pos_probs(model, text: str) -> np.ndarray:
    tokens = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    last_logits = logits[0, -1, :].detach().cpu().float().numpy()
    return scipy_softmax(last_logits).astype(np.float64)


def _run_session(req: ProbeSessionRequest):
    model = model_manager.model
    if model is None:
        yield f"data: {json.dumps({'type': 'error', 'detail': 'No model loaded'})}\n\n"
        return

    n = len(req.probe_prompts)
    probe_results = []
    kl_divergences = []
    # full delta arrays for sway aggregation: list of np.ndarray [d_vocab]
    all_deltas: list[np.ndarray] = []

    for i, probe_prompt in enumerate(req.probe_prompts):
        treatment_ctx = " ".join(req.normalization_cues) + " " + probe_prompt
        control_ctx = " ".join(req.control_cues) + " " + probe_prompt

        t_probs = _get_last_pos_probs(model, treatment_ctx)
        c_probs = _get_last_pos_probs(model, control_ctx)

        delta = t_probs - c_probs
        all_deltas.append(delta)

        # KL(treatment || control)
        kl = float(np.sum(t_probs * np.log((t_probs + 1e-10) / (c_probs + 1e-10))))
        kl_divergences.append(kl)

        # top_k by treatment prob
        top_ids = np.argsort(-t_probs)[: req.top_k]
        treatment_top_k = [
            {
                "token": _decode_token(model, int(tid)),
                "token_id": int(tid),
                "prob": float(t_probs[tid]),
            }
            for tid in top_ids
        ]
        control_top_k = [
            {
                "token": _decode_token(model, int(tid)),
                "token_id": int(tid),
                "prob": float(c_probs[tid]),
            }
            for tid in top_ids
        ]

        # delta_top: top 20 by |delta|
        delta_top_ids = np.argsort(-np.abs(delta))[:20]
        delta_top = [
            {
                "token": _decode_token(model, int(tid)),
                "token_id": int(tid),
                "treatment_prob": float(t_probs[tid]),
                "control_prob": float(c_probs[tid]),
                "delta": float(delta[tid]),
            }
            for tid in delta_top_ids
        ]

        probe_results.append(
            {
                "probe_index": i,
                "probe_prompt": probe_prompt,
                "kl_divergence": kl,
                "treatment_top_k": treatment_top_k,
                "control_top_k": control_top_k,
                "delta_top": delta_top,
            }
        )

        yield f"data: {json.dumps({'type': 'progress', 'done': i + 1, 'total': n, 'probe_index': i, 'kl': kl})}\n\n"

    # --- Aggregate sway metrics ---
    delta_matrix = np.stack(all_deltas, axis=0)  # [N, d_vocab]
    mean_kl = float(np.mean(kl_divergences))
    n_above = sum(1 for kl in kl_divergences if kl > 0.001)
    sway_score = float(mean_kl * (n_above / n)) if n > 0 else 0.0

    # consistent tokens: sign consistent across >= 70% of probes
    signs = np.sign(delta_matrix)  # [N, d_vocab], values -1/0/1
    pos_frac = (signs > 0).mean(axis=0)  # fraction of probes where delta > 0
    neg_frac = (signs < 0).mean(axis=0)
    consistency = np.maximum(pos_frac, neg_frac)  # per-token consistency fraction
    mean_delta = delta_matrix.mean(axis=0)
    mean_abs_delta = np.abs(mean_delta)

    consistent_mask = consistency >= 0.70
    consistent_ids = np.where(consistent_mask)[0]
    # sort by mean |delta| descending, take top 50
    sorted_consistent = sorted(
        consistent_ids, key=lambda tid: -mean_abs_delta[tid]
    )[:50]

    consistent_tokens = [
        {
            "token": _decode_token(model, int(tid)),
            "token_id": int(tid),
            "mean_delta": float(mean_delta[tid]),
            "consistency_pct": float(consistency[tid] * 100),
        }
        for tid in sorted_consistent
    ]

    direction_map = {
        entry["token"]: {
            "mean_delta": entry["mean_delta"],
            "consistency_pct": entry["consistency_pct"],
            "count": n,
        }
        for entry in consistent_tokens
    }

    sway = {
        "mean_kl": mean_kl,
        "sway_score": sway_score,
        "consistent_tokens": consistent_tokens,
        "direction_map": direction_map,
    }

    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "concept": req.concept,
        "model_name": getattr(model_manager, "loaded_model_name", "unknown"),
        "notes": req.notes,
        "normalization_cues": req.normalization_cues,
        "control_cues": req.control_cues,
        "probe_prompts": req.probe_prompts,
        "sway": sway,
        "probes": probe_results,
    }

    session_file = SESSIONS_DIR / f"{session_id}.json"
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

    yield f"data: {json.dumps({'type': 'complete', 'session_id': session_id, 'sway': sway})}\n\n"


@router.post("/session/run")
async def run_probe_session(req: ProbeSessionRequest):
    return StreamingResponse(_run_session(req), media_type="text/event-stream")


@router.get("/sessions")
async def list_sessions():
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            sessions.append(
                {
                    "session_id": data.get("session_id"),
                    "timestamp": data.get("timestamp"),
                    "concept": data.get("concept"),
                    "model_name": data.get("model_name"),
                    "probe_count": len(data.get("probes", [])),
                    "sway_score": data.get("sway", {}).get("sway_score", 0.0),
                }
            )
        except Exception:
            continue
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    f = SESSIONS_DIR / f"{session_id}.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    with open(f) as fh:
        return json.load(fh)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    f = SESSIONS_DIR / f"{session_id}.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    f.unlink()
    return {"deleted": True}


class ProbeResponseRequest(BaseModel):
    normalization_cues: list[str]
    control_cues: list[str]
    probe_prompt: str
    max_new_tokens: int = 100


def _generate_text(model, context: str, max_new_tokens: int) -> str:
    tokens = model.to_tokens(context, prepend_bos=True)
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # decode only the newly generated tokens
    new_tokens = output[0, tokens.shape[1]:]
    return model.tokenizer.decode(new_tokens.tolist(), clean_up_tokenization_spaces=False)


@router.post("/response")
async def generate_probe_response(req: ProbeResponseRequest):
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    treatment_ctx = " ".join(req.normalization_cues) + " " + req.probe_prompt
    control_ctx = " ".join(req.control_cues) + " " + req.probe_prompt

    treatment_response = _generate_text(model, treatment_ctx, req.max_new_tokens)
    control_response = _generate_text(model, control_ctx, req.max_new_tokens)

    return {
        "treatment_context": treatment_ctx,
        "control_context": control_ctx,
        "treatment_response": treatment_response,
        "control_response": control_response,
    }
