import threading
from pathlib import Path
from collections import OrderedDict

from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


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
                model = HookedTransformer.from_pretrained(model_name)
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
