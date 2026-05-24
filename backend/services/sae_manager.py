import threading
from typing import Dict, List, Optional

import numpy as np
import torch

from sae_lens import SAE

SUPPORTED_RELEASES: dict[str, str] = {
    "gpt2": "gpt2-small-res-jb",
}

# gpt2-small-res-jb has resid_pre for layers 0-11 and resid_post for layer 11.
# We expose layer indices 0-11 as resid_pre and index 12 as the final resid_post.
_RESID_PRE = "blocks.{layer}.hook_resid_pre"
_RESID_POST_FINAL = "blocks.11.hook_resid_post"


def _hook_for_layer(layer: int) -> str:
    if layer == 12:
        return _RESID_POST_FINAL
    return _RESID_PRE.format(layer=layer)


def _cache_key_for_layer(layer: int) -> str:
    """Key used by TransformerLens activation cache for the given layer index."""
    if layer == 12:
        return "blocks.11.hook_resid_post"
    return f"blocks.{layer}.hook_resid_pre"


class SAEManager:
    def __init__(self):
        self._saes: Dict[str, SAE] = {}
        self._lock = threading.Lock()

    def get_release(self, model_name: str) -> Optional[str]:
        # Match on the base model name (e.g. "gpt2" from "gpt2-medium")
        for key, release in SUPPORTED_RELEASES.items():
            if model_name == key or model_name.startswith(key + "-"):
                return release
        return None

    def load_layer(self, model_name: str, layer: int) -> dict:
        """Download + cache the SAE for a specific residual-stream layer (0-12)."""
        release = self.get_release(model_name)
        if not release:
            raise ValueError(
                f"No pretrained SAE available for '{model_name}'. "
                f"Supported base models: {list(SUPPORTED_RELEASES)}"
            )

        hook = _hook_for_layer(layer)
        cache_key = f"{release}::{hook}"

        with self._lock:
            if cache_key not in self._saes:
                sae, _, _ = SAE.from_pretrained(release=release, sae_id=hook, device="cpu")
                self._saes[cache_key] = sae
            info = self._saes[cache_key]

        n_features = int(info.W_enc.shape[1])
        return {
            "release": release,
            "hook": hook,
            "layer": layer,
            "n_features": n_features,
            "neuronpedia_id": f"gpt2-small/{layer}-res-jb",
        }

    def decompose(
        self,
        activation: torch.Tensor,
        model_name: str,
        layer: int,
        top_k: int = 15,
    ) -> dict:
        """
        Decompose a [1, pos, d_model] residual-stream tensor into sparse SAE features.

        Returns a dict with per-token and global feature data ready to serialise.
        """
        release = self.get_release(model_name)
        if not release:
            raise ValueError(f"No SAE release for model '{model_name}'.")

        hook = _hook_for_layer(layer)
        cache_key = f"{release}::{hook}"

        with self._lock:
            sae = self._saes.get(cache_key)

        if sae is None:
            raise ValueError(
                f"SAE for layer {layer} not loaded. Call POST /api/sae/load first."
            )

        act = activation.float().cpu()  # ensure CPU float32
        with torch.no_grad():
            features = sae.encode(act)  # [1, pos, n_features]

        feats_np: np.ndarray = features[0].numpy()  # [pos, n_features]
        n_pos, n_features = feats_np.shape
        top_k = min(top_k, n_features)

        # Per-token top-k
        per_token = []
        for pos_i in range(n_pos):
            row = feats_np[pos_i]
            positive_mask = row > 0
            n_active = int(positive_mask.sum())
            top_indices = np.argsort(-row)[:top_k]
            top_features = [
                {"feature_id": int(idx), "activation": float(row[idx])}
                for idx in top_indices
                if row[idx] > 0
            ]
            per_token.append({
                "position": pos_i,
                "top_features": top_features,
                "total_activation": float(row[positive_mask].sum()),
                "n_active": n_active,
            })

        # Global: features with the highest max activation across all positions
        global_max = feats_np.max(axis=0)  # [n_features]
        top_global_idx = np.argsort(-global_max)[:top_k * 3]
        global_top = [
            {"feature_id": int(idx), "max_activation": float(global_max[idx])}
            for idx in top_global_idx
            if global_max[idx] > 0
        ]

        # Heatmap: collect the union of features appearing in any token's top-k
        active_fids = sorted({
            f["feature_id"]
            for tok in per_token
            for f in tok["top_features"]
        })
        # Sort by total activation across positions, descending
        active_fids.sort(key=lambda fid: -float(feats_np[:, fid].sum()))

        heatmap = [
            {
                "feature_id": fid,
                "activations": feats_np[:, fid].tolist(),
                "max_activation": float(feats_np[:, fid].max()),
            }
            for fid in active_fids
        ]

        return {
            "layer": layer,
            "hook": hook,
            "per_token": per_token,
            "global_top_features": global_top,
            "heatmap": heatmap,
            "n_active_features": len(active_fids),
            "neuronpedia_base_url": f"https://www.neuronpedia.org/gpt2-small/{layer}-res-jb",
        }

    @property
    def loaded_layers(self) -> List[str]:
        return list(self._saes.keys())


sae_manager = SAEManager()
