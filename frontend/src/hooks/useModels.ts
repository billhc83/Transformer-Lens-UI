import { useState, useEffect } from 'react'

export interface ModelInfo {
  name: string
  is_local: boolean
}

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

export default function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadingModel, setLoadingModel] = useState(false)

  useEffect(() => {
    fetch('/api/models/available')
      .then((r) => r.json())
      .then((d) => setModels([...d.models].sort((a, b) => Number(b.is_local) - Number(a.is_local))))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  async function loadModel(name: string): Promise<ModelConfig> {
    setLoadingModel(true)
    try {
      const r = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: name }),
      })
      if (!r.ok) throw new Error(`Load failed: ${r.status}`)
      const d = await r.json()
      return d.config as ModelConfig
    } finally {
      setLoadingModel(false)
    }
  }

  return { models, loading, error, loadModel, loadingModel }
}
