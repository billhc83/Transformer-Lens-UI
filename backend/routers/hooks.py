import ast
from typing import List, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.hook_registry import hook_registry
from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/hooks", tags=["hooks"])

FORBIDDEN_CALLS = {
    "eval", "exec", "open", "compile", "__import__",
    "getattr", "setattr", "delattr", "vars", "dir",
    "input", "breakpoint",
}

_SAFE_BUILTINS = {
    "print": print,
    "range": range,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "float": float,
    "int": int,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "None": None,
    "True": True,
    "False": False,
}


def _validate_and_compile(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed in hook code")
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name and name in FORBIDDEN_CALLS:
                raise ValueError(f"Call to '{name}' is not allowed in hook code")

    namespace: dict = {"torch": torch, "__builtins__": _SAFE_BUILTINS}
    try:
        exec(code, namespace)
    except Exception as e:
        raise ValueError(f"Hook code raised an error: {e}")

    if "hook_fn" not in namespace or not callable(namespace["hook_fn"]):
        raise ValueError("Hook code must define a callable named 'hook_fn(value, hook)'")

    return namespace["hook_fn"]


def _top_k(model, logits: torch.Tensor, k: int) -> List[dict]:
    last = logits[0, -1, :]
    probs = torch.softmax(last, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    return [
        {
            "token": model.to_str_tokens(torch.tensor([int(tid)]))[0],
            "token_id": int(tid),
            "prob": float(p),
        }
        for tid, p in zip(top_ids, top_probs)
    ]


# ---------- request models ----------

class AddHookRequest(BaseModel):
    hook_name_filter: str
    hook_code: str


class PreviewRequest(BaseModel):
    prompt: str
    top_k: int = 5


# ---------- endpoints ----------

@router.post("/add")
async def add_hook(request: AddHookRequest):
    try:
        _validate_and_compile(request.hook_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    hook_id = hook_registry.add(request.hook_name_filter, request.hook_code)
    return {"id": hook_id}


@router.delete("/{hook_id}")
async def remove_hook(hook_id: str):
    deleted = hook_registry.remove(hook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Hook '{hook_id}' not found")
    return {"deleted": True}


@router.get("")
async def list_hooks():
    return {"hooks": hook_registry.list_hooks()}


@router.get("/hook-points")
async def list_hook_points():
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    return {"hook_points": list(model.hook_dict.keys())}


@router.post("/preview")
async def preview_hooks(request: PreviewRequest):
    model = model_manager.model
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    try:
        tokens = model.to_tokens(request.prompt, prepend_bos=True)
        str_tokens = list(model.to_str_tokens(request.prompt, prepend_bos=True))

        with torch.no_grad():
            baseline_logits = model(tokens)
        baseline_top = _top_k(model, baseline_logits, request.top_k)

        active = hook_registry.list_hooks()
        if not active:
            return {
                "baseline": baseline_top,
                "modified": baseline_top,
                "str_tokens": str_tokens,
                "hook_count": 0,
            }

        fwd_hooks = []
        for h in active:
            try:
                fn = _validate_and_compile(h["code"])
                fwd_hooks.append((h["hook_name"], fn))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Hook {h['id']}: {e}")

        with torch.no_grad():
            modified_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        modified_top = _top_k(model, modified_logits, request.top_k)

        return {
            "baseline": baseline_top,
            "modified": modified_top,
            "str_tokens": str_tokens,
            "hook_count": len(fwd_hooks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
