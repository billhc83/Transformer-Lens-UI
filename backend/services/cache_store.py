from typing import Dict, List
import torch


class CacheStore:
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}
        self.str_tokens: List[str] = []

    def set(self, cache_dict: Dict, str_tokens: List[str]):
        self.cache = dict(cache_dict)
        self.str_tokens = list(str_tokens)

    def clear(self):
        self.cache.clear()
        self.str_tokens.clear()

    def keys_with_shapes(self) -> List[Dict]:
        return [{"key": k, "shape": list(v.shape)} for k, v in sorted(self.cache.items())]


cache_store = CacheStore()
