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


@router.get("/composition_scores")
async def get_composition_scores():
    try:
        model = _check_model()
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        n_total = n_layers * n_heads

        # Collect weight matrices for all heads
        W_Vs = torch.stack([
            model.blocks[l].attn.W_V[h].detach().cpu().float()
            for l in range(n_layers) for h in range(n_heads)
        ])  # [n_total, d_model, d_head]
        W_Os = torch.stack([
            model.blocks[l].attn.W_O[h].detach().cpu().float()
            for l in range(n_layers) for h in range(n_heads)
        ])  # [n_total, d_head, d_model]
        W_Qs = torch.stack([
            model.blocks[l].attn.W_Q[h].detach().cpu().float()
            for l in range(n_layers) for h in range(n_heads)
        ])  # [n_total, d_model, d_head]
        W_Ks = torch.stack([
            model.blocks[l].attn.W_K[h].detach().cpu().float()
            for l in range(n_layers) for h in range(n_heads)
        ])  # [n_total, d_model, d_head]

        # Precompute norms
        # Approximate ||OV||_F ≈ ||W_V||_F * ||W_O||_F (submultiplicativity bound)
        ov_norms = (
            W_Vs.reshape(n_total, -1).norm(dim=1) *
            W_Os.reshape(n_total, -1).norm(dim=1)
        )  # [n_total]
        W_Q_norms = W_Qs.reshape(n_total, -1).norm(dim=1)  # [n_total]
        W_K_norms = W_Ks.reshape(n_total, -1).norm(dim=1)  # [n_total]
        W_V_norms = W_Vs.reshape(n_total, -1).norm(dim=1)  # [n_total]

        q_scores = torch.zeros(n_total, n_total)
        k_scores = torch.zeros(n_total, n_total)
        v_scores = torch.zeros(n_total, n_total)

        # For each source head i, compute composition score with all target heads j
        # Q-comp: ||W_V_i @ W_O_i @ W_Q_j||_F / (||OV_i||_F * ||W_Q_j||_F)
        # Efficient: W_O_i @ W_Q_j = [d_head, d_head], then W_V_i @ result = [d_model, d_head]
        for i in range(n_total):
            # Q-composition
            M_q = torch.einsum('ab,jbc->jac', W_Os[i], W_Qs)   # [n_total, d_head, d_head]
            cq = torch.einsum('ab,jbc->jac', W_Vs[i], M_q)      # [n_total, d_model, d_head]
            q_scores[i] = cq.reshape(n_total, -1).norm(dim=1) / (ov_norms[i] * W_Q_norms + 1e-8)

            # K-composition
            M_k = torch.einsum('ab,jbc->jac', W_Os[i], W_Ks)   # [n_total, d_head, d_head]
            ck = torch.einsum('ab,jbc->jac', W_Vs[i], M_k)      # [n_total, d_model, d_head]
            k_scores[i] = ck.reshape(n_total, -1).norm(dim=1) / (ov_norms[i] * W_K_norms + 1e-8)

            # V-composition: W_O_i [d_head,d_model] @ W_V_j [d_model,d_head] → [d_head,d_head]
            M_v = torch.einsum('ab,jbc->jac', W_Os[i], W_Vs)   # [n_total, d_head, d_head]
            cv = torch.einsum('ab,jbc->jac', W_Vs[i], M_v)      # [n_total, d_model, d_head]
            v_scores[i] = cv.reshape(n_total, -1).norm(dim=1) / (ov_norms[i] * W_V_norms + 1e-8)

        return {
            "q_scores": q_scores.tolist(),
            "k_scores": k_scores.tolist(),
            "v_scores": v_scores.tolist(),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "labels": model.all_head_labels(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
