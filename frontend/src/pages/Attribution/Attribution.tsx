import { useState, useMemo } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

const API = ''

interface AttributionResponse {
  scores: number[][]       // [n_components, n_pos]
  labels: string[]
  str_tokens: string[]
  answer_token_id: number
  answer_token_str: string
  mode: string
}

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

type Mode = 'full' | 'by_layer' | 'by_head'

const MODES: { id: Mode; label: string }[] = [
  { id: 'full', label: 'Full' },
  { id: 'by_layer', label: 'By Layer' },
  { id: 'by_head', label: 'By Head' },
]

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v))
}

export default function Attribution() {
  const [answerToken, setAnswerToken] = useState(' Paris')
  const [mode, setMode] = useState<Mode>('full')
  const [focusPos, setFocusPos] = useState<number | null>(null)
  const [data, setData] = useState<AttributionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const { data: res } = await axios.post<AttributionResponse>(
        `${API}/api/inference/attribution`,
        { answer_token: answerToken, mode },
      )
      setData(res)
      setFocusPos(res.str_tokens.length - 1)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Request failed. Run run_with_cache first.')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  // Sorted top contributors at the focused position
  const topContribs = useMemo(() => {
    if (!data || focusPos === null) return []
    const pos = clamp(focusPos, 0, data.str_tokens.length - 1)
    return data.labels
      .map((lbl, i) => ({ label: lbl, score: data.scores[i]?.[pos] ?? 0 }))
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
      .slice(0, 20)
  }, [data, focusPos])

  // Plotly heatmap data
  const heatmapTrace = useMemo(() => {
    if (!data) return null
    const z = data.scores  // [n_comp, n_pos]
    const maxAbs = Math.max(...z.flat().map(Math.abs), 0.001)
    return {
      z,
      x: data.str_tokens,
      y: data.labels,
      type: 'heatmap' as const,
      colorscale: [
        [0,    '#ff6b6b'],
        [0.5,  '#1a1a2e'],
        [1,    '#4ade80'],
      ],
      zmin: -maxAbs,
      zmax: maxAbs,
      showscale: true,
      colorbar: {
        thickness: 10,
        len: 0.8,
        tickfont: { color: 'rgba(255,255,255,0.5)', size: 9, family: 'JetBrains Mono' },
        bgcolor: 'transparent',
        bordercolor: 'transparent',
      },
      hoverongaps: false,
      hovertemplate: '%{y} @ %{x}<br>attribution: %{z:.4f}<extra></extra>',
    }
  }, [data])

  const heatmapLayout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(255,255,255,0.02)',
    margin: { t: 10, b: 60, l: 100, r: 60 },
    xaxis: {
      tickfont: { color: 'rgba(255,255,255,0.6)', size: 9, family: 'JetBrains Mono' },
      gridcolor: 'rgba(255,255,255,0.05)',
      color: 'rgba(255,255,255,0.3)',
    },
    yaxis: {
      tickfont: { color: 'rgba(255,255,255,0.5)', size: 8, family: 'JetBrains Mono' },
      gridcolor: 'rgba(255,255,255,0.05)',
      color: 'rgba(255,255,255,0.3)',
      autorange: 'reversed' as const,
    },
    font: { family: 'JetBrains Mono', color: 'rgba(255,255,255,0.5)' },
  }), [])

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
          Attribution Analyzer
        </span>
        <span style={{
          fontSize: 9,
          padding: '2px 7px',
          borderRadius: 4,
          background: 'rgba(168,85,247,0.12)',
          border: '1px solid rgba(168,85,247,0.3)',
          color: '#a855f7',
        }}>Phase 6</span>
      </div>

      {/* Controls */}
      <div style={{
        padding: '10px 20px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        flexWrap: 'wrap',
      }}>
        <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', fontFamily: 'JetBrains Mono, monospace' }}>
          Answer token:
        </label>
        <input
          value={answerToken}
          onChange={e => setAnswerToken(e.target.value)}
          placeholder=" Paris"
          style={{
            width: 120,
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 6,
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 12,
            padding: '4px 8px',
          }}
        />

        {/* Mode selector */}
        <div style={{ display: 'flex', gap: 4 }}>
          {MODES.map(m => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              style={{
                padding: '4px 12px',
                borderRadius: 5,
                border: `1px solid ${mode === m.id ? 'rgba(0,212,255,0.5)' : 'rgba(255,255,255,0.1)'}`,
                background: mode === m.id ? 'rgba(0,212,255,0.12)' : 'transparent',
                color: mode === m.id ? '#00d4ff' : 'rgba(255,255,255,0.4)',
                fontSize: 10,
                cursor: 'pointer',
                fontFamily: 'JetBrains Mono, monospace',
              }}
            >
              {m.label}
            </button>
          ))}
        </div>

        <button
          onClick={run}
          disabled={loading}
          style={{
            padding: '6px 18px',
            background: loading ? 'rgba(0,212,255,0.08)' : 'rgba(0,212,255,0.15)',
            border: '1px solid rgba(0,212,255,0.4)',
            borderRadius: 6,
            color: '#00d4ff',
            fontSize: 11,
            cursor: loading ? 'not-allowed' : 'pointer',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        >
          {loading ? 'Running…' : 'Run Attribution'}
        </button>

        {data && (
          <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', fontFamily: 'JetBrains Mono, monospace' }}>
            {data.labels.length} components · {data.str_tokens.length} tokens
          </span>
        )}
      </div>

      {error && (
        <div style={{ padding: '8px 20px', color: '#ff6b6b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', flexShrink: 0 }}>
          {error}
        </div>
      )}

      {loading && !data && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)', fontFamily: 'JetBrains Mono, monospace' }}>
            Computing attribution scores…
          </div>
        </div>
      )}

      {data ? (
        <div style={{ flex: 1, display: 'flex', gap: 12, padding: '12px 20px', overflow: 'hidden', minHeight: 0 }}>
          {/* Heatmap panel */}
          <div style={{ ...panel, flex: '1 1 60%', display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 8 }}>
              ATTRIBUTION HEATMAP — components × positions
              <span style={{ marginLeft: 12, color: 'rgba(74,222,128,0.7)' }}>■ positive</span>
              <span style={{ marginLeft: 8, color: 'rgba(255,107,107,0.7)' }}>■ negative</span>
            </div>
            <div style={{ flex: 1, minHeight: 0 }}>
              {heatmapTrace && (
                <Plot
                  data={[heatmapTrace as any]}
                  layout={{
                    ...heatmapLayout,
                    height: undefined,
                    autosize: true,
                  }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler
                  config={{ displayModeBar: false, responsive: true }}
                  onClick={(event: any) => {
                    const pt = event.points?.[0]
                    if (pt != null) setFocusPos(pt.pointIndex?.[1] ?? pt.x as number)
                  }}
                />
              )}
            </div>
          </div>

          {/* Top contributors panel */}
          <div style={{ ...panel, width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', fontFamily: 'JetBrains Mono, monospace', marginBottom: 8 }}>
              TOP CONTRIBUTORS
            </div>

            {/* Position selector */}
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 10 }}>
              {data.str_tokens.map((tok, i) => (
                <button
                  key={i}
                  onClick={() => setFocusPos(i)}
                  style={{
                    padding: '2px 6px',
                    borderRadius: 4,
                    border: `1px solid ${focusPos === i ? 'rgba(168,85,247,0.6)' : 'rgba(255,255,255,0.08)'}`,
                    background: focusPos === i ? 'rgba(168,85,247,0.15)' : 'transparent',
                    color: focusPos === i ? '#a855f7' : 'rgba(255,255,255,0.4)',
                    fontSize: 9,
                    cursor: 'pointer',
                    fontFamily: 'JetBrains Mono, monospace',
                    whiteSpace: 'nowrap',
                    maxWidth: 64,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                  title={tok}
                >
                  {tok.slice(0, 7)}
                </button>
              ))}
            </div>

            {/* Bar chart */}
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {topContribs.map(({ label, score }) => {
                const maxScore = Math.max(...topContribs.map(c => Math.abs(c.score)), 0.001)
                const pct = Math.abs(score) / maxScore * 100
                const isPos = score >= 0
                return (
                  <div key={label} style={{ marginBottom: 5 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <span style={{
                        fontSize: 9,
                        fontFamily: 'JetBrains Mono, monospace',
                        color: isPos ? 'rgba(74,222,128,0.8)' : 'rgba(255,107,107,0.8)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        maxWidth: 140,
                      }}>
                        {label}
                      </span>
                      <span style={{
                        fontSize: 9,
                        fontFamily: 'JetBrains Mono, monospace',
                        color: isPos ? 'rgba(74,222,128,0.6)' : 'rgba(255,107,107,0.6)',
                        flexShrink: 0,
                        marginLeft: 4,
                      }}>
                        {score > 0 ? '+' : ''}{score.toFixed(3)}
                      </span>
                    </div>
                    <div style={{
                      height: 4,
                      borderRadius: 2,
                      background: 'rgba(255,255,255,0.05)',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${pct}%`,
                        borderRadius: 2,
                        background: isPos
                          ? 'linear-gradient(90deg, rgba(74,222,128,0.6), rgba(74,222,128,0.9))'
                          : 'linear-gradient(90deg, rgba(255,107,107,0.6), rgba(255,107,107,0.9))',
                        transition: 'width 0.3s ease',
                      }} />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      ) : (
        !loading && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ ...panel, textAlign: 'center', maxWidth: 400 }}>
              <div style={{ fontSize: 28, marginBottom: 12, opacity: 0.2 }}>⊛</div>
              <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.7 }}>
                Run <span style={{ color: '#00d4ff' }}>run_with_cache</span> on a prompt first,<br />
                then enter an answer token (e.g. <span style={{ color: '#a855f7' }}>" Paris"</span>)<br />
                and click <span style={{ color: '#00d4ff' }}>Run Attribution</span>.
              </div>
            </div>
          </div>
        )
      )}
    </div>
  )
}
