import { create } from 'zustand'

export interface ModelConfig {
  name: string
  d_model: number
  n_layers: number
  n_heads: number
  d_mlp: number
  n_ctx: number
  d_vocab: number
  act_fn: string
  normalization_type: string
  device: string
}

interface ModelStore {
  loadedConfig: ModelConfig | null
  setLoadedConfig: (cfg: ModelConfig | null) => void
}

export const useModelStore = create<ModelStore>((set) => ({
  loadedConfig: null,
  setLoadedConfig: (cfg) => set({ loadedConfig: cfg }),
}))
