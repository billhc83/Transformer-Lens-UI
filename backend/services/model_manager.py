import threading
from pathlib import Path
from collections import OrderedDict

import torch
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# ProcessWeights unconditionally upcasts bfloat16 → float32 even when no weight
# processing is requested, requiring ~3× the model's weight RAM at peak and causing
# swap thrashing / indefinite hangs on systems where RAM is insufficient.
# When all processing flags are False the upcast is a no-op, so we skip it.
try:
    from transformer_lens.weight_processing import ProcessWeights as _PW
    _orig_process_weights = _PW.process_weights

    def _memory_safe_process_weights(
        state_dict, cfg,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
        fold_value_biases=False, refactor_factored_attn_matrices=False, **kw
    ):
        if not any([fold_ln, center_writing_weights, center_unembed,
                    fold_value_biases, refactor_factored_attn_matrices]):
            return state_dict
        return _orig_process_weights(
            state_dict, cfg,
            fold_ln=fold_ln, center_writing_weights=center_writing_weights,
            center_unembed=center_unembed, fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices, **kw,
        )

    _PW.process_weights = staticmethod(_memory_safe_process_weights)
except Exception:
    pass  # older TL without weight_processing — no patch needed


def _model_slug(name: str) -> str:
    return "models--" + name.replace("/", "--")


def _is_local(name: str) -> bool:
    return (HF_CACHE / _model_slug(name)).exists()


def _config_dict(model: HookedTransformer, model_name: str) -> dict:
    cfg = model.cfg
    return {
        "name": model_name,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_mlp": cfg.d_mlp,
        "n_ctx": cfg.n_ctx,
        "d_vocab": cfg.d_vocab,
        "act_fn": cfg.act_fn,
        "normalization_type": cfg.normalization_type,
        "device": str(next(model.parameters()).device),
    }


class ModelManager:
    def __init__(self):
        self._cache: OrderedDict[str, HookedTransformer] = OrderedDict()
        self._max_size = 2
        self._active: str | None = None
        self._lock = threading.Lock()

    def list_available_models(self) -> list[dict]:
        return [{"name": n, "is_local": _is_local(n)} for n in OFFICIAL_MODEL_NAMES]

    def load_model(self, model_name: str) -> dict:
        with self._lock:
            if model_name in self._cache:
                self._cache.move_to_end(model_name)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                # dtype=bfloat16 keeps the TL skeleton at ~half the float32 size,
                # preventing peak RAM from exceeding available memory on large models.
                # All processing flags off: fold_ln etc. require the float32 upcast
                # (avoided by _memory_safe_process_weights above); RMSNorm scales are
                # preserved explicitly in the loaded state dict instead.
                model = HookedTransformer.from_pretrained(
                    model_name,
                    device="cpu",
                    dtype=torch.bfloat16,
                    fold_ln=False,
                    center_unembed=False,
                    center_writing_weights=False,
                    fold_value_biases=False,
                    refactor_factored_attn_matrices=False,
                )
                self._cache[model_name] = model
            self._active = model_name
        return _config_dict(self._cache[model_name], model_name)

    def get_loaded_model_info(self) -> dict | None:
        with self._lock:
            if self._active and self._active in self._cache:
                return _config_dict(self._cache[self._active], self._active)
        return None

    @property
    def model(self) -> HookedTransformer | None:
        with self._lock:
            if self._active:
                return self._cache.get(self._active)
        return None


model_manager = ModelManager()
