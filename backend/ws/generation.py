import json

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.model_manager import model_manager

router = APIRouter()


@router.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()

    model = model_manager.model
    if model is None:
        await websocket.send_json({"type": "error", "message": "No model loaded"})
        await websocket.close()
        return

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)

        prompt: str = payload.get("prompt", "")
        max_tokens: int = int(payload.get("max_tokens", 50))
        temperature: float = float(payload.get("temperature", 1.0))
        top_k: int = int(payload.get("top_k", 40))
        monitors: list[str] = payload.get("monitors", [])

        # tokens shape: [1, seq_len]
        tokens = model.to_tokens(prompt)

        names_filter = monitors if monitors else None

        for step in range(max_tokens):
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

            # last-position logits: [d_vocab]
            last_logits = logits[0, -1] / max(temperature, 1e-6)

            # top-k masking
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(last_logits, min(top_k, last_logits.shape[-1]))
                mask = torch.full_like(last_logits, float("-inf"))
                mask[topk_idx] = topk_vals
                last_logits = mask

            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            logprob = float(torch.log(probs[next_id] + 1e-10).item())

            # top-5 by prob for bar chart
            top5_probs, top5_ids = torch.topk(probs, min(5, probs.shape[-1]))
            top_tokens = [
                {"token": model.tokenizer.decode([tid.item()]), "prob": float(p.item())}
                for tid, p in zip(top5_ids, top5_probs)
            ]

            # activation summaries for monitored hooks
            activations: dict = {}
            for name in monitors:
                if name in cache:
                    t = cache[name]
                    activations[name] = {
                        "shape": list(t.shape),
                        "mean": float(t.float().mean().item()),
                        "max": float(t.float().max().item()),
                        "min": float(t.float().min().item()),
                    }

            token_str = model.tokenizer.decode([next_id])

            await websocket.send_json({
                "type": "token",
                "token": token_str,
                "token_id": next_id,
                "logprob": logprob,
                "step": step,
                "top_tokens": top_tokens,
                "activations": activations,
            })

            # append next token [1, 1] → [1, seq+1]
            next_tensor = torch.tensor([[next_id]], dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, next_tensor], dim=1)

            # stop at EOS
            if hasattr(model.cfg, "eos_token") and model.cfg.eos_token is not None:
                if next_id == model.cfg.eos_token:
                    break

        await websocket.send_json({"type": "done", "total_tokens": step + 1})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
