import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { useSessionStore } from '../../store/sessionStore'

const API = ''

// ── types ────────────────────────────────────────────────────────────────────

interface TopFeature { feature_id: number; activation: number }
interface PerToken {
  position: number
  top_features: TopFeature[]
  total_activation: number
  n_active: number
}
interface HeatmapRow {
  feature_id: number
  activations: number[]
  max_activation: number
}
interface GlobalTopFeature { feature_id: number; max_activation: number }
interface DecomposeResponse {
  layer: number
  hook: string
  str_tokens: string[]
  per_token: PerToken[]
  heatmap: HeatmapRow[]
  global_top_features: GlobalTopFeature[]
  n_active_features: number
  neuronpedia_base_url: string
}
interface StatusResponse {
  model: string | null
  supported: boolean
  release: string | null
  loaded_layer_keys: string[]
}

// ── palette helpers ──────────────────────────────────────────────────────────

function heatColor(value: number, max: number): string {
  if (max === 0) return 'rgba(255,255,255,0.03)'
  const t = Math.min(1, value / max)
  // 0 = dark navy, 1 = bright amber
  const r = Math.round(t * 245)
  const g = Math.round(t * 158)
  const b = Math.round((1 - t) * 80)
  return `rgba(${r},${g},${b},${Math.max(0.05, t * 0.9)})`
}

function tokenIntensityColor(total: number, maxTotal: number): string {
  if (maxTotal === 0) return 'rgba(0,212,255,0.05)'
  const t = Math.min(1, total / maxTotal)
  return `rgba(0,212,255,${0.06 + t * 0.35})`
}

// ── sub-components ───────────────────────────────────────────────────────────

function TokenStrip({
  strTokens,
  perToken,
  selectedPos,
  onSelect,
}: {
  strTokens: string[]
  perToken: PerToken[]
  selectedPos: number | null
  onSelect: (pos: number) => void
}) {
  const maxTotal = Math.max(...perToken.map(t => t.total_activation), 0.0001)
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, padding: '8px 20px' }}>
      {strTokens.map((tok, i) => {
        const pt = perToken[i]
        const total = pt?.total_activation ?? 0
        const nActive = pt?.n_active ?? 0
        const isSelected = selectedPos === i
        return (
          <button
            key={i}
            onClick={() => onSelect(i)}
            title={`${nActive} active features  ·  total activation ${total.toFixed(2)}`}
            style={{
              padding: '4px 8px',
              borderRadius: 5,
              border: isSelected
                ? '1px solid #f59e0b'
                : '1px solid rgba(255,255,255,0.08)',
              background: isSelected
                ? 'rgba(245,158,11,0.15)'
                : tokenIntensityColor(total, maxTotal),
              color: isSelected ? '#f59e0b' : 'rgba(255,255,255,0.8)',
              fontSize: 11,
              fontFamily: 'JetBrains Mono, monospace',
              cursor: 'pointer',
              position: 'relative',
            }}
          >
            {tok}
            {nActive > 0 && (
              <span style={{
                position: 'absolute',
                top: -5,
                right: -5,
                fontSize: 7,
                background: 'rgba(245,158,11,0.8)',
                color: '#000',
                borderRadius: 4,
                padding: '0 3px',
                fontWeight: 700,
              }}>
                {nActive}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

function FeatureHeatmap({
  heatmap,
  strTokens,
  selectedFeatureId,
  highlightPos,
  onSelectFeature,
  neuronpediaBase,
}: {
  heatmap: HeatmapRow[]
  strTokens: string[]
  selectedFeatureId: number | null
  highlightPos: number | null
  onSelectFeature: (fid: number) => void
  neuronpediaBase: string
}) {
  const globalMax = Math.max(...heatmap.map(r => r.max_activation), 0.0001)

  return (
    <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 340, flexShrink: 0 }}>
      <div style={{ minWidth: 'max-content', padding: '0 20px 8px' }}>
        {/* Column headers */}
        <div style={{ display: 'flex', gap: 2, marginBottom: 4, paddingLeft: 140 }}>
          {strTokens.map((tok, i) => (
            <div
              key={i}
              style={{
                width: 44,
                flexShrink: 0,
                fontSize: 8,
                fontFamily: 'JetBrains Mono, monospace',
                color: highlightPos === i ? '#f59e0b' : 'rgba(255,255,255,0.25)',
                textAlign: 'center',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {tok.slice(0, 5)}
            </div>
          ))}
        </div>

        {/* Feature rows */}
        {heatmap.map(row => {
          const isSelected = selectedFeatureId === row.feature_id
          return (
            <div
              key={row.feature_id}
              onClick={() => onSelectFeature(row.feature_id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                marginBottom: 2,
                cursor: 'pointer',
                borderRadius: 4,
                background: isSelected ? 'rgba(168,85,247,0.08)' : 'transparent',
                border: isSelected ? '1px solid rgba(168,85,247,0.3)' : '1px solid transparent',
              }}
            >
              {/* Feature ID label */}
              <div style={{
                width: 130,
                flexShrink: 0,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                paddingLeft: 4,
              }}>
                <span style={{
                  fontSize: 9,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: isSelected ? '#a855f7' : 'rgba(255,255,255,0.35)',
                  whiteSpace: 'nowrap',
                }}>
                  F{row.feature_id}
                </span>
                <a
                  href={`${neuronpediaBase}/${row.feature_id}`}
                  target="_blank"
                  rel="noreferrer"
                  onClick={e => e.stopPropagation()}
                  style={{
                    fontSize: 7,
                    color: 'rgba(0,212,255,0.4)',
                    textDecoration: 'none',
                  }}
                  title="Open on Neuronpedia"
                >
                  ↗
                </a>
                <span style={{
                  fontSize: 7,
                  color: 'rgba(255,255,255,0.2)',
                  fontFamily: 'JetBrains Mono, monospace',
                }}>
                  {row.max_activation.toFixed(1)}
                </span>
              </div>

              {/* Activation cells */}
              {row.activations.map((val, colIdx) => (
                <div
                  key={colIdx}
                  title={`F${row.feature_id} @ pos${colIdx} (${strTokens[colIdx]}): ${val.toFixed(3)}`}
                  style={{
                    width: 44,
                    height: 18,
                    flexShrink: 0,
                    borderRadius: 2,
                    background: heatColor(val, globalMax),
                    border: highlightPos === colIdx && val > 0
                      ? '1px solid rgba(245,158,11,0.5)'
                      : '1px solid transparent',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {val > 0.1 && (
                    <span style={{
                      fontSize: 7,
                      color: 'rgba(255,255,255,0.6)',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}>
                      {val.toFixed(1)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function FeatureDetail({
  featureId,
  heatmapRow,
  strTokens,
  layer,
  neuronpediaBase,
  onInterpret,
  interpretation,
  interpreting,
}: {
  featureId: number
  heatmapRow: HeatmapRow | undefined
  strTokens: string[]
  layer: number
  neuronpediaBase: string
  onInterpret: () => void
  interpretation: string
  interpreting: boolean
}) {
  if (!heatmapRow) return null

  const pairs = strTokens.map((tok, i) => ({
    tok,
    val: heatmapRow.activations[i] ?? 0,
  }))
  const maxVal = Math.max(...pairs.map(p => p.val), 0.0001)
  const barWidth = 140

  return (
    <div style={{
      padding: '12px 20px',
      borderTop: '1px solid rgba(168,85,247,0.15)',
      background: 'rgba(168,85,247,0.04)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: '#a855f7', fontFamily: 'JetBrains Mono, monospace' }}>
          Feature {featureId}
        </span>
        <a
          href={`${neuronpediaBase}/${featureId}`}
          target="_blank"
          rel="noreferrer"
          style={{
            fontSize: 10,
            color: '#00d4ff',
            textDecoration: 'none',
            border: '1px solid rgba(0,212,255,0.3)',
            padding: '2px 8px',
            borderRadius: 4,
          }}
        >
          View on Neuronpedia ↗
        </a>
        <button
          onClick={onInterpret}
          disabled={interpreting}
          style={{
            fontSize: 10,
            color: '#f59e0b',
            background: 'rgba(245,158,11,0.1)',
            border: '1px solid rgba(245,158,11,0.35)',
            borderRadius: 4,
            padding: '2px 10px',
            cursor: interpreting ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit',
          }}
        >
          {interpreting ? 'Interpreting…' : '✦ Interpret with AI'}
        </button>
        <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.2)', fontFamily: 'JetBrains Mono, monospace' }}>
          max={heatmapRow.max_activation.toFixed(3)} · layer {layer}
        </span>
      </div>

      {/* Per-token activation bars */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
        {pairs.map(({ tok, val }, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <div style={{
              width: 32,
              height: Math.max(2, Math.round((val / maxVal) * 36)),
              background: val > 0 ? `rgba(168,85,247,${0.2 + (val / maxVal) * 0.7})` : 'rgba(255,255,255,0.04)',
              borderRadius: 2,
              alignSelf: 'flex-end',
            }} />
            <span style={{
              fontSize: 8,
              fontFamily: 'JetBrains Mono, monospace',
              color: val > 0 ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.15)',
              maxWidth: 32,
              textAlign: 'center',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {tok.slice(0, 4)}
            </span>
            {val > 0 && (
              <span style={{
                fontSize: 7,
                color: 'rgba(168,85,247,0.7)',
                fontFamily: 'JetBrains Mono, monospace',
              }}>
                {val.toFixed(1)}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* AI Interpretation output */}
      {interpretation && (
        <div style={{
          background: 'rgba(0,0,0,0.3)',
          border: '1px solid rgba(245,158,11,0.2)',
          borderRadius: 6,
          padding: '8px 12px',
          fontSize: 11,
          color: 'rgba(255,255,255,0.75)',
          lineHeight: 1.55,
          fontFamily: 'inherit',
          whiteSpace: 'pre-wrap',
          maxHeight: 140,
          overflowY: 'auto',
        }}>
          <span style={{ color: '#f59e0b', fontSize: 9, display: 'block', marginBottom: 4 }}>
            AI INTERPRETATION
          </span>
          {interpretation}
        </div>
      )}
    </div>
  )
}

// ── main page ────────────────────────────────────────────────────────────────

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

export default function SAEStudio() {
  const [layer, setLayer] = useState(8)
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingMsg, setLoadingMsg] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<DecomposeResponse | null>(null)
  const [selectedFeatureId, setSelectedFeatureId] = useState<number | null>(null)
  const [selectedPos, setSelectedPos] = useState<number | null>(null)
  const [interpretation, setInterpretation] = useState('')
  const [interpreting, setInterpreting] = useState(false)
  const addFinding = useSessionStore(s => s.addFinding)
  const abortRef = useRef<AbortController | null>(null)

  // Fetch status on mount
  useEffect(() => {
    axios.get<StatusResponse>(`${API}/api/sae/status`).then(r => setStatus(r.data)).catch(() => {})
  }, [])

  async function loadAndDecompose() {
    setLoading(true)
    setError(null)
    setData(null)
    setSelectedFeatureId(null)
    setSelectedPos(null)
    setInterpretation('')

    try {
      setLoadingMsg('Loading SAE weights (first run downloads ~50 MB)…')
      await axios.post(`${API}/api/sae/load`, { layer })

      setLoadingMsg('Decomposing activations…')
      const { data: res } = await axios.post<DecomposeResponse>(`${API}/api/sae/decompose`, {
        layer,
        top_k: 15,
      })
      setData(res)

      // Auto-select the globally dominant feature
      if (res.global_top_features.length > 0) {
        setSelectedFeatureId(res.global_top_features[0].feature_id)
      }

      addFinding({
        page: 'sae-studio',
        headline: `L${layer}: ${res.n_active_features} active SAE features`,
        data: {
          layer,
          n_active_features: res.n_active_features,
          top_feature: res.global_top_features[0]?.feature_id ?? null,
          str_tokens: res.str_tokens,
        },
      })
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Request failed. Run run_with_cache first.')
    } finally {
      setLoading(false)
      setLoadingMsg('')
    }
  }

  async function interpret() {
    if (selectedFeatureId === null || !data) return
    const row = data.heatmap.find(r => r.feature_id === selectedFeatureId)
    if (!row) return

    setInterpreting(true)
    setInterpretation('')

    if (abortRef.current) abortRef.current.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      const resp = await fetch(`${API}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          page_type: 'sae-studio',
          data: {
            layer: data.layer,
            feature_id: selectedFeatureId,
            str_tokens: data.str_tokens,
            activations: row.activations,
            n_active_features: data.n_active_features,
            neuronpedia_url: `${data.neuronpedia_base_url}/${selectedFeatureId}`,
          },
        }),
        signal: ctrl.signal,
      })

      const reader = resp.body!.getReader()
      const dec = new TextDecoder()
      let text = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        text += dec.decode(value, { stream: true })
        setInterpretation(text)
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') setInterpretation('AI interpretation unavailable.')
    } finally {
      setInterpreting(false)
    }
  }

  const selectedRow = data?.heatmap.find(r => r.feature_id === selectedFeatureId)

  // When a token is clicked, auto-select the top feature for that position
  function handleTokenSelect(pos: number) {
    setSelectedPos(pos)
    const pt = data?.per_token[pos]
    if (pt && pt.top_features.length > 0) {
      setSelectedFeatureId(pt.top_features[0].feature_id)
      setInterpretation('')
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#0a0a0f', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        padding: '12px 20px',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#a855f7', fontFamily: 'JetBrains Mono, monospace' }}>
          SAE Feature Studio
        </span>
        <span style={{
          fontSize: 9,
          padding: '2px 7px',
          borderRadius: 4,
          background: 'rgba(168,85,247,0.12)',
          border: '1px solid rgba(168,85,247,0.3)',
          color: '#a855f7',
        }}>Phase 12</span>
        {status && (
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.25)', fontFamily: 'JetBrains Mono, monospace' }}>
            {status.supported
              ? `release: ${status.release}`
              : `no SAE for model "${status.model ?? 'none'}"`}
          </span>
        )}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>

        {/* Controls */}
        <div style={{
          padding: '10px 20px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          flexShrink: 0,
        }}>
          <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', fontFamily: 'JetBrains Mono, monospace' }}>
            Residual stream layer:
          </label>
          <input
            type="number"
            value={layer}
            min={0}
            max={12}
            onChange={e => setLayer(Number(e.target.value))}
            style={{
              width: 52,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 6,
              color: '#fff',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
              padding: '4px 8px',
            }}
          />
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.2)', fontFamily: 'JetBrains Mono, monospace' }}>
            {layer <= 3 ? 'early · syntax' : layer <= 7 ? 'mid · semantics' : 'late · facts'}
            {layer === 12 ? ' (final resid_post)' : ''}
          </span>

          <button
            onClick={loadAndDecompose}
            disabled={loading}
            style={{
              padding: '6px 18px',
              background: loading ? 'rgba(168,85,247,0.08)' : 'rgba(168,85,247,0.15)',
              border: '1px solid rgba(168,85,247,0.4)',
              borderRadius: 6,
              color: '#a855f7',
              fontSize: 11,
              cursor: loading ? 'not-allowed' : 'pointer',
              fontFamily: 'inherit',
            }}
          >
            {loading ? loadingMsg || 'Working…' : '⬡ Decompose Layer'}
          </button>

          {data && (
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.25)', fontFamily: 'JetBrains Mono, monospace' }}>
              {data.n_active_features} active features · {data.str_tokens.length} tokens
            </span>
          )}
        </div>

        {error && (
          <div style={{ padding: '8px 20px', color: '#ff6b6b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}>
            {error}
          </div>
        )}

        {/* Empty state */}
        {!data && !loading && (
          <div style={{ padding: 24, display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
            <div style={{ ...panel, textAlign: 'center', maxWidth: 400 }}>
              <div style={{ fontSize: 28, marginBottom: 12, opacity: 0.15, color: '#a855f7' }}>⬡</div>
              <div style={{ fontSize: 12, color: '#a855f7', fontFamily: 'JetBrains Mono, monospace', marginBottom: 8 }}>
                SAE Feature Studio
              </div>
              <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)', lineHeight: 1.6 }}>
                Decomposes the residual stream into sparse, human-interpretable features
                using a pretrained Sparse Autoencoder (gpt2-small-res-jb).<br /><br />
                Run <span style={{ color: '#00d4ff' }}>run_with_cache</span> on a prompt,
                then choose a layer and click <span style={{ color: '#a855f7' }}>Decompose Layer</span>.
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {data && (
          <>
            {/* Section label */}
            <div style={{
              padding: '6px 20px 2px',
              fontSize: 9,
              color: 'rgba(168,85,247,0.5)',
              fontFamily: 'JetBrains Mono, monospace',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              flexShrink: 0,
            }}>
              Token Activation Map · click a token to see its top feature
            </div>

            <TokenStrip
              strTokens={data.str_tokens}
              perToken={data.per_token}
              selectedPos={selectedPos}
              onSelect={handleTokenSelect}
            />

            <div style={{
              padding: '6px 20px 2px',
              fontSize: 9,
              color: 'rgba(168,85,247,0.5)',
              fontFamily: 'JetBrains Mono, monospace',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              flexShrink: 0,
            }}>
              Feature Heatmap · click a row to inspect · ↗ opens Neuronpedia
            </div>

            <FeatureHeatmap
              heatmap={data.heatmap}
              strTokens={data.str_tokens}
              selectedFeatureId={selectedFeatureId}
              highlightPos={selectedPos}
              onSelectFeature={fid => { setSelectedFeatureId(fid); setInterpretation('') }}
              neuronpediaBase={data.neuronpedia_base_url}
            />

            {/* Feature detail panel */}
            {selectedFeatureId !== null && (
              <FeatureDetail
                featureId={selectedFeatureId}
                heatmapRow={selectedRow}
                strTokens={data.str_tokens}
                layer={data.layer}
                neuronpediaBase={data.neuronpedia_base_url}
                onInterpret={interpret}
                interpretation={interpretation}
                interpreting={interpreting}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}
