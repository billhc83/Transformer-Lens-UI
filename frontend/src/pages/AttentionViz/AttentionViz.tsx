import { useState, useCallback } from 'react'
import axios from 'axios'
import AttentionHeatmap from '../../components/viz/AttentionHeatmap'
import InterpretationModal, { type InterpretationGuide } from '../../components/shared/InterpretationModal'

const API = ''

const GUIDE: InterpretationGuide = {
  overview:
    'Attention Viz shows the attention weight matrix for every head in a transformer layer. ' +
    'Each cell [row i, col j] is the attention weight head h assigns to token j when processing token i. ' +
    'Rows sum to 1.0 (softmax). ' +
    'Bright = high attention, dark = near-zero. ' +
    'Different heads specialise: some attend to previous tokens (previous-token heads), some copy subject tokens (duplicate-token heads), ' +
    'some attend to punctuation or the [BOS] token. ' +
    'Use the layer slider to step through layers 0–11 and the head grid to zoom into a single head.',
  example: {
    prompt: 'Layer 5, IOI prompt "When Mary and John went to the store, John gave a book to"',
    output:
      '12 heads, 8×8 attention matrix per head\n' +
      'Head 5,1: strong diagonal (each token attends to itself)\n' +
      'Head 5,5: " John" (pos 2) attends heavily to " Mary" (pos 1)\n' +
      'Head 5,9: most tokens attend to pos 0 (BOS token)',
    interpretation:
      'Head 5,5 exhibiting cross-name attention is consistent with a "subject-mover" head.\n' +
      'BOS-attending heads (like 5,9) act as a null/no-op signal — the model routes attention\n' +
      'there when it doesn\'t need information from the context.\n' +
      'Diagonal heads are induction-adjacent: they copy the current token\'s own representation\n' +
      'forward, providing a residual shortcut through the layer.',
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

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

function statChip(label: string, value: number) {
  return (
    <span
      key={label}
      style={{
        background: 'rgba(0,212,255,0.08)',
        border: '1px solid rgba(0,212,255,0.2)',
        borderRadius: 6,
        padding: '2px 10px',
        fontSize: 11,
        fontFamily: 'JetBrains Mono, monospace',
        color: '#00d4ff',
        marginRight: 6,
      }}
    >
      {label}: {value.toFixed(4)}
    </span>
  )
}

function headStats(pattern: number[][]) {
  const flat = pattern.flat()
  const min = Math.min(...flat)
  const max = Math.max(...flat)
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length
  return { min, max, mean }
}

export default function AttentionViz() {
  const [layer, setLayer] = useState(0)
  const [maxLayer, setMaxLayer] = useState(11)
  const [patterns, setPatterns] = useState<number[][][] | null>(null)
  const [strTokens, setStrTokens] = useState<string[]>([])
  const [selectedHead, setSelectedHead] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [guideOpen, setGuideOpen] = useState(false)

  const fetchLayer = useCallback(async (l: number) => {
    setLoading(true)
    setError(null)
    setSelectedHead(null)
    try {
      const { data } = await axios.get(`${API}/api/activations/attention/${l}`)
      setPatterns(data.patterns)
      setStrTokens(data.str_tokens)
      // infer max layer from n_heads (gpt2 has 12 layers)
      if (data.n_heads) setMaxLayer(11)
    } catch (e: any) {
      const msg = e?.response?.data?.detail ?? 'Fetch failed. Run /api/inference/run_with_cache first.'
      setError(msg)
      setPatterns(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleLayerInput = (val: number) => {
    const l = Math.max(0, Math.min(maxLayer, val))
    setLayer(l)
    setPatterns(null)
    setError(null)
    setSelectedHead(null)
  }

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        padding: 20,
        overflowY: 'auto',
        fontFamily: 'JetBrains Mono, monospace',
        color: '#fff',
        minHeight: 0,
      }}
    >
      {/* Top bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 18, fontWeight: 600, letterSpacing: 1 }}>Attention Patterns</span>
        <button style={GUIDE_BTN} onClick={() => setGuideOpen(true)}>? How to read this</button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>
          <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>Layer</span>
          <input
            type="number"
            value={layer}
            min={0}
            max={maxLayer}
            onChange={(e) => handleLayerInput(Number(e.target.value))}
            style={{
              width: 50,
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 6,
              color: '#fff',
              padding: '3px 6px',
              fontSize: 13,
              fontFamily: 'inherit',
              textAlign: 'center',
            }}
          />
          <input
            type="range"
            min={0}
            max={maxLayer}
            value={layer}
            onChange={(e) => handleLayerInput(Number(e.target.value))}
            style={{ accentColor: '#00d4ff', width: 140 }}
          />
          <button
            onClick={() => fetchLayer(layer)}
            disabled={loading}
            style={{
              background: loading ? 'rgba(0,212,255,0.1)' : 'rgba(0,212,255,0.15)',
              border: '1px solid rgba(0,212,255,0.4)',
              borderRadius: 8,
              color: '#00d4ff',
              fontSize: 12,
              padding: '5px 16px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontFamily: 'inherit',
            }}
          >
            {loading ? 'Loading…' : 'Load Layer'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            ...panel,
            color: '#ff6b6b',
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}

      {/* Hint when no data */}
      {!patterns && !error && !loading && (
        <div style={{ ...panel, color: 'rgba(255,255,255,0.3)', fontSize: 13, textAlign: 'center', padding: 32 }}>
          Select a layer and click <span style={{ color: '#00d4ff' }}>Load Layer</span>.
          Make sure you ran the cache first (Activations page).
        </div>
      )}

      {/* Head grid */}
      {patterns && (
        <div style={panel}>
          <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)', marginBottom: 12 }}>
            Layer {layer} — {patterns.length} heads
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            {patterns.map((pat, h) => (
              <div
                key={h}
                style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}
              >
                <AttentionHeatmap
                  compact
                  pattern={pat}
                  strTokens={strTokens}
                  headIndex={h}
                  layerIndex={layer}
                  onClick={() => setSelectedHead(h === selectedHead ? null : h)}
                  selected={selectedHead === h}
                />
                <span style={{ fontSize: 10, color: '#00d4ff' }}>H{h}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expanded view */}
      {patterns && selectedHead !== null && (
        <div style={panel}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>
              Layer {layer} · Head {selectedHead}
            </span>
            <div>
              {(() => {
                const s = headStats(patterns[selectedHead])
                return (
                  <>
                    {statChip('min', s.min)}
                    {statChip('max', s.max)}
                    {statChip('mean', s.mean)}
                  </>
                )
              })()}
            </div>
          </div>
          <AttentionHeatmap
            compact={false}
            pattern={patterns[selectedHead]}
            strTokens={strTokens}
            headIndex={selectedHead}
            layerIndex={layer}
          />
        </div>
      )}
      <InterpretationModal
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
        pageTitle="Attention Viz"
        pageType="attention-viz"
        guide={GUIDE}
        liveData={patterns ? { layer, str_tokens: strTokens, patterns } : null}
      />
    </div>
  )
}
