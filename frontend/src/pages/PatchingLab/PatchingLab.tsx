import { useState } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

const API = 'http://localhost:8000'

interface PatchingResponse {
  results: Record<string, number[][]>
  str_tokens: string[]
  n_layers: number
  clean_diff: number
  corrupted_diff: number
  answer_token: string
  baseline_token: string
}

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

const inputStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.05)',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 6,
  color: '#fff',
  fontFamily: 'JetBrains Mono, monospace',
  fontSize: 11,
  padding: '4px 8px',
}

const ACTIVATION_TYPES = [
  { id: 'residual', label: 'Residual Stream' },
  { id: 'attn_out', label: 'Attn Output' },
  { id: 'mlp_out', label: 'MLP Output' },
]

export default function PatchingLab() {
  const [cleanPrompt, setCleanPrompt] = useState(
    'When Mary and John went to the store, John gave a drink to'
  )
  const [corruptedPrompt, setCorruptedPrompt] = useState(
    'When Mary and John went to the store, Mary gave a drink to'
  )
  const [answerToken, setAnswerToken] = useState(' Mary')
  const [baselineToken, setBaselineToken] = useState(' John')
  const [selectedTypes, setSelectedTypes] = useState<string[]>(['residual', 'attn_out', 'mlp_out'])
  const [data, setData] = useState<PatchingResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const toggleType = (id: string) => {
    setSelectedTypes(prev =>
      prev.includes(id) ? prev.filter(t => t !== id) : [...prev, id]
    )
  }

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const { data: res } = await axios.post<PatchingResponse>(
        `${API}/api/patching/run`,
        {
          clean_prompt: cleanPrompt,
          corrupted_prompt: corruptedPrompt,
          answer_token: answerToken,
          baseline_token: baselineToken,
          activation_types: selectedTypes,
        }
      )
      setData(res)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Request failed')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  const heatmapLayout = (title: string) => ({
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(255,255,255,0.02)',
    margin: { t: 10, b: 50, l: 50, r: 60 },
    height: 220,
    autosize: true,
    xaxis: {
      tickfont: { color: 'rgba(255,255,255,0.6)', size: 9, family: 'JetBrains Mono' },
      gridcolor: 'rgba(255,255,255,0.05)',
      color: 'rgba(255,255,255,0.3)',
      title: { text: title, font: { color: 'rgba(255,255,255,0.3)', size: 9 } },
    },
    yaxis: {
      tickfont: { color: 'rgba(255,255,255,0.6)', size: 9, family: 'JetBrains Mono' },
      gridcolor: 'rgba(255,255,255,0.05)',
      color: 'rgba(255,255,255,0.3)',
      autorange: 'reversed' as const,
    },
    font: { family: 'JetBrains Mono', color: 'rgba(255,255,255,0.5)' },
  })

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
        <span style={{ fontSize: 14, fontWeight: 600, color: '#00d4ff', fontFamily: 'JetBrains Mono, monospace' }}>
          Patching Lab
        </span>
        <span style={{
          fontSize: 9,
          padding: '2px 7px',
          borderRadius: 4,
          background: 'rgba(168,85,247,0.12)',
          border: '1px solid rgba(168,85,247,0.3)',
          color: '#a855f7',
        }}>Phase 7</span>
      </div>

      {/* Controls */}
      <div style={{
        padding: '10px 20px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}>
        {/* Prompts row */}
        <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', fontFamily: 'JetBrains Mono, monospace' }}>
              Clean prompt
            </label>
            <textarea
              value={cleanPrompt}
              onChange={e => setCleanPrompt(e.target.value)}
              rows={2}
              style={{ ...inputStyle, resize: 'vertical', width: '100%' }}
            />
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', fontFamily: 'JetBrains Mono, monospace' }}>
              Corrupted prompt
            </label>
            <textarea
              value={corruptedPrompt}
              onChange={e => setCorruptedPrompt(e.target.value)}
              rows={2}
              style={{ ...inputStyle, resize: 'vertical', width: '100%' }}
            />
          </div>
        </div>

        {/* Token + type controls row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', fontFamily: 'JetBrains Mono, monospace' }}>
            Answer:
          </label>
          <input
            value={answerToken}
            onChange={e => setAnswerToken(e.target.value)}
            style={{ ...inputStyle, width: 90 }}
          />
          <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', fontFamily: 'JetBrains Mono, monospace' }}>
            Baseline:
          </label>
          <input
            value={baselineToken}
            onChange={e => setBaselineToken(e.target.value)}
            style={{ ...inputStyle, width: 90 }}
          />

          {/* Activation type toggles */}
          <div style={{ display: 'flex', gap: 4 }}>
            {ACTIVATION_TYPES.map(t => {
              const active = selectedTypes.includes(t.id)
              return (
                <button
                  key={t.id}
                  onClick={() => toggleType(t.id)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: 5,
                    border: `1px solid ${active ? 'rgba(0,212,255,0.5)' : 'rgba(255,255,255,0.1)'}`,
                    background: active ? 'rgba(0,212,255,0.12)' : 'transparent',
                    color: active ? '#00d4ff' : 'rgba(255,255,255,0.35)',
                    fontSize: 10,
                    cursor: 'pointer',
                    fontFamily: 'JetBrains Mono, monospace',
                  }}
                >
                  {t.label}
                </button>
              )
            })}
          </div>

          <button
            onClick={run}
            disabled={loading || selectedTypes.length === 0}
            style={{
              padding: '6px 18px',
              background: loading ? 'rgba(0,212,255,0.08)' : 'rgba(0,212,255,0.15)',
              border: '1px solid rgba(0,212,255,0.4)',
              borderRadius: 6,
              color: '#00d4ff',
              fontSize: 11,
              cursor: loading || selectedTypes.length === 0 ? 'not-allowed' : 'pointer',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            {loading ? 'Running…' : 'Run Patching'}
          </button>

          {data && (
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', fontFamily: 'JetBrains Mono, monospace' }}>
              {data.n_layers} layers · {data.str_tokens.length} tokens
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: '8px 20px', color: '#ff6b6b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', flexShrink: 0 }}>
          {error}
        </div>
      )}

      {/* Body */}
      {loading && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ fontSize: 12, color: '#00d4ff', fontFamily: 'JetBrains Mono, monospace', opacity: 0.7 }}>
            Running patching experiments…
          </span>
        </div>
      )}

      {!data && !loading && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ ...panel, textAlign: 'center', maxWidth: 440 }}>
            <div style={{ fontSize: 28, marginBottom: 12, opacity: 0.2 }}>⊗</div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.7 }}>
              Enter a <span style={{ color: '#00d4ff' }}>clean</span> and <span style={{ color: '#ff6b6b' }}>corrupted</span> prompt,<br />
              set the <span style={{ color: '#a855f7' }}>answer</span> and <span style={{ color: '#a855f7' }}>baseline</span> tokens,<br />
              then click <span style={{ color: '#00d4ff' }}>Run Patching</span>.<br />
              <span style={{ opacity: 0.5, fontSize: 10 }}>Default: IOI task — Mary vs John</span>
            </div>
          </div>
        </div>
      )}

      {data && !loading && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Summary */}
          <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', fontFamily: 'JetBrains Mono, monospace' }}>
            clean logit diff: <span style={{ color: '#4ade80' }}>{data.clean_diff > 0 ? '+' : ''}{data.clean_diff.toFixed(3)}</span>
            &nbsp;·&nbsp;
            corrupted: <span style={{ color: '#ff6b6b' }}>{data.corrupted_diff > 0 ? '+' : ''}{data.corrupted_diff.toFixed(3)}</span>
            &nbsp;·&nbsp;normalized: 0 = fully corrupted, 1 = fully clean
          </div>

          {/* Heatmap per activation type */}
          {ACTIVATION_TYPES.filter(t => data.results[t.id]).map(t => (
            <div key={t.id} style={panel}>
              <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 8 }}>
                {t.label.toUpperCase()} — LAYER × POSITION
                <span style={{ marginLeft: 10, color: '#00d4ff' }}>■ 1.0 = clean</span>
                <span style={{ marginLeft: 6, color: '#1a1a2e', textShadow: '0 0 0 rgba(168,85,247,0.8)' }}>■ 0.0 = corrupted</span>
              </div>
              <Plot
                data={[{
                  z: data.results[t.id],
                  x: data.str_tokens,
                  y: Array.from({ length: data.n_layers }, (_, i) => `L${i}`),
                  type: 'heatmap' as const,
                  colorscale: [
                    [0,   '#0a0a1f'],
                    [0.5, '#a855f7'],
                    [1,   '#00d4ff'],
                  ],
                  zmin: 0,
                  zmax: 1,
                  showscale: true,
                  colorbar: {
                    thickness: 10,
                    len: 0.8,
                    tickfont: { color: 'rgba(255,255,255,0.5)', size: 9, family: 'JetBrains Mono' },
                    bgcolor: 'transparent',
                    bordercolor: 'transparent',
                  },
                  hovertemplate: `${t.label} · Layer %{y} · Token %{x}<br>norm diff: %{z:.3f}<extra></extra>`,
                } as any]}
                layout={heatmapLayout(t.label) as any}
                style={{ width: '100%' }}
                useResizeHandler
                config={{ displayModeBar: false, responsive: true }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
