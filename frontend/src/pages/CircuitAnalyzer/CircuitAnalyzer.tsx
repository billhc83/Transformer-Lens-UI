import { useState, useCallback } from 'react'
import axios from 'axios'
import _PlotLib from 'react-plotly.js'
const Plot = ((_PlotLib as any).default ?? _PlotLib) as any
import CircuitGraph from '../../components/viz/CircuitGraph'
import InterpretationModal, { type InterpretationGuide } from '../../components/shared/InterpretationModal'

const API = ''

const GUIDE: InterpretationGuide = {
  overview:
    'Circuit Analyzer computes composition scores between every pair of attention heads — how much head B uses the output of head A as its input. ' +
    'Three composition types: Q-composition (head B\'s queries use A\'s output), K-composition (B\'s keys), V-composition (B\'s values). ' +
    'High score = strong compositional relationship = a circuit edge. ' +
    'The node graph visualises the top edges above a tunable threshold. ' +
    'Click a node to see the QK and OV matrix decompositions for that head (singular value spectra).',
  example: {
    prompt: 'Load head info + compute composition scores (default threshold 0.05)',
    output:
      'Top K-composition scores:\n' +
      '  L1H8 → L9H6:  0.341  (strong)\n' +
      '  L0H3 → L5H5:  0.189\n' +
      'Top Q-composition:\n' +
      '  L2H2 → L10H7: 0.127\n' +
      'OV spectrum for L9H6: singular values [0.42, 0.38, 0.21, ...]',
    interpretation:
      'L1H8 → L9H6 via K-composition: L9H6\'s keys are shaped by L1H8\'s output.\n' +
      'This is a hallmark of the induction circuit: L1H8 is an induction head that\n' +
      'K-composes with later name-mover heads.\n' +
      'A flat OV singular value spectrum means the head copies uniformly across many directions.\n' +
      'A peaked spectrum (one dominant singular value) means the head is a specialised mover\n' +
      'for a narrow set of features.',
  },
}

const GUIDE_BTN: React.CSSProperties = {
  fontSize: 11,
  padding: '3px 10px',
  borderRadius: 6,
  border: '1px solid rgba(0,212,255,0.4)',
  background: 'transparent',
  color: '#00d4ff',
  cursor: 'pointer',
  fontFamily: 'inherit',
  letterSpacing: '0.04em',
}

interface HeadLabelsResponse {
  labels: string[]
  n_layers: number
  n_heads: number
}

interface CompositionResponse {
  q_scores: number[][]
  k_scores: number[][]
  v_scores: number[][]
  n_layers: number
  n_heads: number
  labels: string[]
}

interface QKResponse {
  layer: number
  head: number
  d_model: number
  d_head: number
  S_Q: number[]
  S_K: number[]
}

interface OVResponse {
  layer: number
  head: number
  d_model: number
  d_head: number
  S_V: number[]
  S_O: number[]
}

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

const btn: React.CSSProperties = {
  background: 'rgba(0,212,255,0.1)',
  border: '1px solid rgba(0,212,255,0.3)',
  borderRadius: 6,
  color: '#00d4ff',
  fontSize: 11,
  padding: '5px 14px',
  cursor: 'pointer',
  fontFamily: 'inherit',
}

const label11: React.CSSProperties = {
  fontSize: 11,
  color: 'rgba(255,255,255,0.5)',
  fontFamily: 'JetBrains Mono, monospace',
}

function SvdBars({
  title, s1, name1, color1, s2, name2, color2,
}: {
  title: string
  s1: number[]; name1: string; color1: string
  s2: number[]; name2: string; color2: string
}) {
  const xs = Array.from({ length: Math.max(s1.length, s2.length) }, (_, i) => i + 1)
  return (
    <Plot
      data={[
        { x: xs, y: s1, type: 'bar', name: name1, marker: { color: color1 } },
        { x: xs, y: s2, type: 'bar', name: name2, marker: { color: color2 } },
      ]}
      layout={{
        title: { text: title, font: { size: 11, color: 'rgba(255,255,255,0.8)' } },
        height: 170,
        margin: { l: 32, r: 8, t: 28, b: 24 },
        barmode: 'group',
        showlegend: true,
        legend: { font: { size: 9, color: 'rgba(255,255,255,0.6)' }, bgcolor: 'rgba(0,0,0,0)' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(255,255,255,0.02)',
        xaxis: {
          title: { text: 'vector', font: { size: 9 } },
          tickfont: { size: 8, color: 'rgba(255,255,255,0.4)' },
          gridcolor: 'rgba(255,255,255,0.05)',
        },
        yaxis: {
          title: { text: 'σ', font: { size: 9 } },
          tickfont: { size: 8, color: 'rgba(255,255,255,0.4)' },
          gridcolor: 'rgba(255,255,255,0.05)',
        },
        font: { color: 'rgba(255,255,255,0.6)' },
      } as object}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

function getTopEdges(matrix: number[][], labels: string[], n: number) {
  const edges: { src: string; dst: string; score: number }[] = []
  for (let i = 0; i < matrix.length; i++)
    for (let j = 0; j < (matrix[i]?.length ?? 0); j++)
      edges.push({ src: labels[i] ?? `${i}`, dst: labels[j] ?? `${j}`, score: matrix[i][j] })
  return edges.sort((a, b) => b.score - a.score).slice(0, n)
}

export default function CircuitAnalyzer() {
  const [info, setInfo] = useState<HeadLabelsResponse | null>(null)
  const [scores, setScores] = useState<CompositionResponse | null>(null)
  const [threshold, setThreshold] = useState(0.05)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [qkData, setQkData] = useState<QKResponse | null>(null)
  const [ovData, setOvData] = useState<OVResponse | null>(null)
  const [loadingInfo, setLoadingInfo] = useState(false)
  const [loadingScores, setLoadingScores] = useState(false)
  const [loadingHead, setLoadingHead] = useState(false)
  const [guideOpen, setGuideOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchInfo = async () => {
    setLoadingInfo(true)
    setError(null)
    try {
      const res = await axios.get<HeadLabelsResponse>(`${API}/api/circuits/head_labels`)
      setInfo(res.data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to fetch head labels')
    } finally {
      setLoadingInfo(false)
    }
  }

  const fetchScores = async () => {
    setLoadingScores(true)
    setError(null)
    try {
      const res = await axios.get<CompositionResponse>(`${API}/api/circuits/composition_scores`, {
        timeout: 120_000,
      })
      setScores(res.data)
      setInfo(i => i ?? { labels: res.data.labels, n_layers: res.data.n_layers, n_heads: res.data.n_heads })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to compute composition scores')
    } finally {
      setLoadingScores(false)
    }
  }

  const handleNodeClick = useCallback(async (layer: number, head: number) => {
    const label = `L${layer}H${head}`
    setSelectedNode(label)
    setQkData(null)
    setOvData(null)
    setLoadingHead(true)
    try {
      const [qkRes, ovRes] = await Promise.all([
        axios.get<QKResponse>(`${API}/api/circuits/qk/${layer}/${head}`),
        axios.get<OVResponse>(`${API}/api/circuits/ov/${layer}/${head}`),
      ])
      setQkData(qkRes.data)
      setOvData(ovRes.data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to fetch head data')
    } finally {
      setLoadingHead(false)
    }
  }, [])

  const hasGraph = scores !== null && info !== null
  const edgeCount = scores
    ? (() => {
        let c = 0
        const n = scores.labels.length
        for (let i = 0; i < n; i++)
          for (let j = i + 1; j < n; j++)
            if (Math.max(scores.q_scores[i][j], scores.k_scores[i][j], scores.v_scores[i][j]) >= threshold) c++
        return c
      })()
    : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: 20, gap: 16, overflow: 'hidden' }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#00d4ff', letterSpacing: '0.05em' }}>
              Circuit Analyzer
            </div>
            <button style={GUIDE_BTN} onClick={() => setGuideOpen(true)}>? How to read this</button>
          </div>
          <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', marginTop: 2 }}>
            QK / OV composition scores · head node graph
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, marginLeft: 'auto', alignItems: 'center' }}>
          {!info && (
            <button style={btn} onClick={fetchInfo} disabled={loadingInfo}>
              {loadingInfo ? 'Loading…' : 'Load Model Info'}
            </button>
          )}
          {info && !scores && (
            <button style={btn} onClick={fetchScores} disabled={loadingScores}>
              {loadingScores ? 'Computing…' : 'Compute Composition Scores'}
            </button>
          )}
          {info && (
            <span style={{ ...label11, marginLeft: 4 }}>
              {info.n_layers}L × {info.n_heads}H = {info.n_layers * info.n_heads} heads
            </span>
          )}
        </div>
      </div>

      {error && (
        <div style={{ padding: '8px 12px', background: 'rgba(255,107,107,0.1)', border: '1px solid rgba(255,107,107,0.3)', borderRadius: 8, fontSize: 11, color: '#ff6b6b' }}>
          {error}
        </div>
      )}

      {/* Main area */}
      <div style={{ flex: 1, display: 'flex', gap: 16, overflow: 'hidden', minHeight: 0 }}>
        {/* Left: graph */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12, minWidth: 0 }}>
          {/* Controls */}
          {hasGraph && (
            <div style={{ ...panel, display: 'flex', alignItems: 'center', gap: 16, padding: '10px 16px', flexShrink: 0 }}>
              {/* Legend */}
              <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                {[['Q-comp', '#00d4ff'], ['K-comp', '#a855f7'], ['V-comp', '#4ade80']].map(([lbl, col]) => (
                  <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <div style={{ width: 20, height: 2, background: col }} />
                    <span style={{ ...label11, color: 'rgba(255,255,255,0.6)' }}>{lbl}</span>
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>
                <span style={label11}>threshold</span>
                <input
                  type="range" min={0.01} max={0.5} step={0.01}
                  value={threshold}
                  onChange={e => setThreshold(parseFloat(e.target.value))}
                  style={{ width: 120, accentColor: '#00d4ff' }}
                />
                <span style={{ ...label11, color: '#00d4ff', minWidth: 34 }}>{threshold.toFixed(2)}</span>
                <span style={{ ...label11, color: 'rgba(255,255,255,0.3)' }}>
                  {edgeCount} edges
                </span>
              </div>
            </div>
          )}

          {/* Graph canvas */}
          <div style={{ flex: 1, ...panel, padding: 0, overflow: 'hidden', position: 'relative', minHeight: 0 }}>
            {!info && !loadingInfo && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 12 }}>
                <div style={{ fontSize: 32, opacity: 0.1 }}>⬡</div>
                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.25)' }}>Load a model, then click "Load Model Info"</div>
              </div>
            )}
            {loadingInfo && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }}>Loading head labels…</span>
              </div>
            )}
            {info && !scores && !loadingScores && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 12 }}>
                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)' }}>
                  {info.n_layers * info.n_heads} heads ready · click "Compute Composition Scores"
                </div>
                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.18)' }}>
                  This may take 10–30 seconds for large models
                </div>
              </div>
            )}
            {loadingScores && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 8 }}>
                <div style={{ fontSize: 11, color: '#00d4ff' }}>Computing composition scores…</div>
                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)' }}>
                  Iterating over {info ? info.n_layers * info.n_heads : '?'} heads
                </div>
              </div>
            )}
            {hasGraph && (
              <CircuitGraph
                nLayers={scores!.n_layers}
                nHeads={scores!.n_heads}
                labels={scores!.labels}
                qScores={scores!.q_scores}
                kScores={scores!.k_scores}
                vScores={scores!.v_scores}
                threshold={threshold}
                onNodeClick={handleNodeClick}
                selectedNode={selectedNode}
              />
            )}
          </div>
        </div>

        {/* Right: side panel */}
        <div style={{ width: 280, display: 'flex', flexDirection: 'column', gap: 12, flexShrink: 0, overflowY: 'auto' }}>
          {/* Head info */}
          <div style={panel}>
            {!selectedNode && (
              <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.25)', textAlign: 'center', padding: '20px 0' }}>
                Click a head node to inspect
              </div>
            )}
            {selectedNode && (
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#00d4ff', fontFamily: 'JetBrains Mono, monospace' }}>
                  {selectedNode}
                </div>
                {qkData && (
                  <div style={{ marginTop: 6, display: 'flex', gap: 12 }}>
                    <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }}>
                      d_model={qkData.d_model}
                    </div>
                    <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }}>
                      d_head={qkData.d_head}
                    </div>
                  </div>
                )}
                {loadingHead && (
                  <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', marginTop: 8 }}>
                    Fetching circuit data…
                  </div>
                )}
              </div>
            )}
          </div>

          {/* QK singular values */}
          {qkData && (
            <div style={panel}>
              <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)', marginBottom: 4 }}>QK CIRCUIT</div>
              <SvdBars
                title="Query · Key singular values"
                s1={qkData.S_Q}
                name1="W_Q"
                color1="#00d4ff"
                s2={qkData.S_K}
                name2="W_K"
                color2="#a855f7"
              />
              <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.25)', marginTop: 4 }}>
                σ₁(W_Q)={qkData.S_Q[0]?.toFixed(2)} · σ₁(W_K)={qkData.S_K[0]?.toFixed(2)}
              </div>
            </div>
          )}

          {/* OV singular values */}
          {ovData && (
            <div style={panel}>
              <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)', marginBottom: 4 }}>OV CIRCUIT</div>
              <SvdBars
                title="Value · Output singular values"
                s1={ovData.S_V}
                name1="W_V"
                color1="#4ade80"
                s2={ovData.S_O}
                name2="W_O"
                color2="#ff6b6b"
              />
              <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.25)', marginTop: 4 }}>
                σ₁(W_V)={ovData.S_V[0]?.toFixed(2)} · σ₁(W_O)={ovData.S_O[0]?.toFixed(2)}
              </div>
            </div>
          )}

          {/* Composition scores for selected node */}
          {selectedNode && scores && (() => {
            const idx = scores.labels.indexOf(selectedNode)
            if (idx === -1) return null
            const outgoing = scores.labels.map((lbl, j) => ({
              lbl,
              q: scores.q_scores[idx][j] ?? 0,
              k: scores.k_scores[idx][j] ?? 0,
              v: scores.v_scores[idx][j] ?? 0,
              max: Math.max(scores.q_scores[idx][j] ?? 0, scores.k_scores[idx][j] ?? 0, scores.v_scores[idx][j] ?? 0),
            })).filter((x, j) => j !== idx && x.max > 0.01)
              .sort((a, b) => b.max - a.max)
              .slice(0, 8)
            if (outgoing.length === 0) return null
            return (
              <div style={panel}>
                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)', marginBottom: 8 }}>
                  TOP COMPOSITIONS FROM {selectedNode}
                </div>
                {outgoing.map(({ lbl, q, k, v, max }) => {
                  const type = max === q ? 'Q' : max === k ? 'K' : 'V'
                  const color = type === 'Q' ? '#00d4ff' : type === 'K' ? '#a855f7' : '#4ade80'
                  return (
                    <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#fff', width: 36 }}>{lbl}</span>
                      <span style={{ fontSize: 8, color, width: 12, textAlign: 'center' }}>{type}</span>
                      <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${Math.min(max * 200, 100)}%`, background: color, borderRadius: 2 }} />
                      </div>
                      <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: 'rgba(255,255,255,0.45)', width: 32, textAlign: 'right' }}>
                        {max.toFixed(3)}
                      </span>
                    </div>
                  )
                })}
              </div>
            )
          })()}
        </div>
      </div>
      <InterpretationModal
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
        pageTitle="Circuit Analyzer"
        pageType="circuit-analyzer"
        guide={GUIDE}
        liveData={scores && info ? {
          labels: info.labels,
          n_layers: info.n_layers,
          n_heads: info.n_heads,
          top_k_edges: getTopEdges(scores.k_scores, info.labels, 10),
          top_q_edges: getTopEdges(scores.q_scores, info.labels, 10),
          top_v_edges: getTopEdges(scores.v_scores, info.labels, 10),
          selected_node: selectedNode,
          qk_data: qkData ? { S_Q: qkData.S_Q.slice(0, 10), S_K: qkData.S_K.slice(0, 10) } : null,
          ov_data: ovData ? { S_V: ovData.S_V.slice(0, 10), S_O: ovData.S_O.slice(0, 10) } : null,
        } : null}
      />
    </div>
  )
}
