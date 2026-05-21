import asyncio

from fastapi import APIRouter, HTTPException
import torch

from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/circuits", tags=["circuits"])


def _check_model():
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    return model


@router.get("/head_labels")
async def get_head_labels():
    try:
        model = _check_model()
        return {
            "labels": model.all_head_labels(),
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qk/{layer}/{head}")
async def get_qk(layer: int, head: int):
    try:
        model = _check_model()
        if layer >= model.cfg.n_layers or head >= model.cfg.n_heads:
            raise HTTPException(status_code=400, detail="Layer/head out of range")

        W_Q = model.blocks[layer].attn.W_Q[head].detach().cpu().float()  # [d_model, d_head]
        W_K = model.blocks[layer].attn.W_K[head].detach().cpu().float()  # [d_model, d_head]

        U_q, S_q, Vh_q = torch.linalg.svd(W_Q, full_matrices=False)
        U_k, S_k, Vh_k = torch.linalg.svd(W_K, full_matrices=False)
        top = min(10, S_q.shape[0])

        return {
            "layer": layer,
            "head": head,
            "d_model": W_Q.shape[0],
            "d_head": W_Q.shape[1],
            "S_Q": S_q[:top].tolist(),
            "U_Q": U_q[:, :top].tolist(),
            "Vh_Q": Vh_q[:top, :].tolist(),
            "S_K": S_k[:top].tolist(),
            "U_K": U_k[:, :top].tolist(),
            "Vh_K": Vh_k[:top, :].tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ov/{layer}/{head}")
async def get_ov(layer: int, head: int):
    try:
        model = _check_model()
        if layer >= model.cfg.n_layers or head >= model.cfg.n_heads:
            raise HTTPException(status_code=400, detail="Layer/head out of range")

        W_V = model.blocks[layer].attn.W_V[head].detach().cpu().float()  # [d_model, d_head]
        W_O = model.blocks[layer].attn.W_O[head].detach().cpu().float()  # [d_head, d_model]

        U_v, S_v, Vh_v = torch.linalg.svd(W_V, full_matrices=False)
        U_o, S_o, Vh_o = torch.linalg.svd(W_O, full_matrices=False)
        top = min(10, S_v.shape[0])

        return {
            "layer": layer,
            "head": head,
            "d_model": W_V.shape[0],
            "d_head": W_V.shape[1],
            "S_V": S_v[:top].tolist(),
            "U_V": U_v[:, :top].tolist(),
            "Vh_V": Vh_v[:top, :].tolist(),
            "S_O": S_o[:top].tolist(),
            "U_O": U_o[:, :top].tolist(),
            "Vh_O": Vh_o[:top, :].tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _compute_composition_scores(model):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_total = n_layers * n_heads

    W_Vs = torch.stack([
        model.blocks[l].attn.W_V[h].detach().cpu().float()
        for l in range(n_layers) for h in range(n_heads)
    ])  # [N, d_model, d_head]
    W_Os = torch.stack([
        model.blocks[l].attn.W_O[h].detach().cpu().float()
        for l in range(n_layers) for h in range(n_heads)
    ])  # [N, d_head, d_model]
    W_Qs = torch.stack([
        model.blocks[l].attn.W_Q[h].detach().cpu().float()
        for l in range(n_layers) for h in range(n_heads)
    ])  # [N, d_model, d_head]
    W_Ks = torch.stack([
        model.blocks[l].attn.W_K[h].detach().cpu().float()
        for l in range(n_layers) for h in range(n_heads)
    ])  # [N, d_model, d_head]

    ov_norms = (
        W_Vs.reshape(n_total, -1).norm(dim=1) *
        W_Os.reshape(n_total, -1).norm(dim=1)
    )  # [N]
    W_Q_norms = W_Qs.reshape(n_total, -1).norm(dim=1)
    W_K_norms = W_Ks.reshape(n_total, -1).norm(dim=1)
    W_V_norms = W_Vs.reshape(n_total, -1).norm(dim=1)

    # P[i] = W_V[i]^T @ W_V[i]: [N, d_head, d_head]
    P_all = torch.bmm(W_Vs.permute(0, 2, 1), W_Vs)

    d_head = W_Vs.shape[2]
    # Chunk outer loop so each batch stays ~400 MB: 6 tensors of [C, N, d_head, d_head]
    bytes_per_row = n_total * d_head * d_head * 4 * 6
    chunk_size = max(1, min(n_total, int(400 * 1024 ** 2 // bytes_per_row)))

    q_scores = torch.zeros(n_total, n_total)
    k_scores = torch.zeros(n_total, n_total)
    v_scores = torch.zeros(n_total, n_total)

    for i0 in range(0, n_total, chunk_size):
        i1 = min(i0 + chunk_size, n_total)
        Wo = W_Os[i0:i1]              # [C, d_head, d_model]
        P  = P_all[i0:i1]             # [C, d_head, d_head]
        denom = ov_norms[i0:i1, None] # [C, 1]

        # M[i,j] = W_O[i] @ W_X[j]: [C, N, d_head, d_head]
        Mq = torch.einsum('iab,jbc->ijac', Wo, W_Qs)
        Mk = torch.einsum('iab,jbc->ijac', Wo, W_Ks)
        Mv = torch.einsum('iab,jbc->ijac', Wo, W_Vs)

        # score[i,j] = sqrt(Tr(M^T P M)) / denom
        PMq = torch.einsum('iab,ijbc->ijac', P, Mq)
        PMk = torch.einsum('iab,ijbc->ijac', P, Mk)
        PMv = torch.einsum('iab,ijbc->ijac', P, Mv)

        q_scores[i0:i1] = (Mq * PMq).sum((-2, -1)).clamp(min=0).sqrt() / (denom * W_Q_norms + 1e-8)
        k_scores[i0:i1] = (Mk * PMk).sum((-2, -1)).clamp(min=0).sqrt() / (denom * W_K_norms + 1e-8)
        v_scores[i0:i1] = (Mv * PMv).sum((-2, -1)).clamp(min=0).sqrt() / (denom * W_V_norms + 1e-8)

    return {
        "q_scores": q_scores.tolist(),
        "k_scores": k_scores.tolist(),
        "v_scores": v_scores.tolist(),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "labels": model.all_head_labels(),
    }


@router.get("/composition_scores")
async def get_composition_scores():
    try:
        model = _check_model()
        return await asyncio.to_thread(_compute_composition_scores, model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
