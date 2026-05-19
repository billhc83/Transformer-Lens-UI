import { useState } from 'react'

const CHIP_COLORS = ['#00d4ff', '#a855f7', '#14b8a6', '#ec4899', '#f97316']

interface TopKEntry {
  token_id: number
  token_str: string
  probability: number
  logit: number
}

interface Prediction {
  position: number
  token: string
  top_k: TopKEntry[]
}

interface ForwardResponse {
  logits_shape: number[]
  str_tokens: string[]
  predictions: Prediction[]
}

export default function ForwardPass() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<ForwardResponse | null>(null)
  const [selectedPos, setSelectedPos] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function runForward() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/inference/forward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, top_k: 10 }),
      })
      if (res.status === 400) {
        const data = await res.json()
        setError(data.detail ?? 'No model loaded')
        setResult(null)
        return
      }
      const data: ForwardResponse = await res.json()
      setResult(data)
      setSelectedPos(data.predictions.length - 1)
    } catch (e) {
      setError('Request failed — is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const focused = result && selectedPos !== null ? result.predictions[selectedPos] : null
  const maxProb = focused ? Math.max(...focused.top_k.map(t => t.probability)) : 1

  return (
    <div style={{ flex: 1, overflow: 'auto', padding: '28px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header */}
      <div>
        <div style={{ fontSize: '16px', fontWeight: 600, color: '#00d4ff', marginBottom: '4px' }}>Forward Pass</div>
        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.35)' }}>Run a forward pass and inspect top predictions per position</div>
      </div>

      {/* Input */}
      <div style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '12px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) runForward() }}
          placeholder="Enter text for forward pass..."
          rows={3}
          style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '8px',
            padding: '10px 12px',
            color: 'rgba(255,255,255,0.85)',
            fontFamily: 'inherit',
            fontSize: '13px',
            resize: 'vertical',
            outline: 'none',
            width: '100%',
            boxSizing: 'border-box',
          }}
        />
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={runForward}
            disabled={loading || !text.trim()}
            style={{
              background: loading ? 'rgba(0,212,255,0.3)' : '#00d4ff',
              border: 'none',
              borderRadius: '8px',
              padding: '8px 20px',
              color: '#0a0a0f',
              fontFamily: 'inherit',
              fontSize: '12px',
              fontWeight: 700,
              cursor: loading || !text.trim() ? 'not-allowed' : 'pointer',
              letterSpacing: '0.05em',
            }}
          >
            {loading ? 'Running...' : 'Run Forward Pass'}
          </button>
          <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.25)' }}>⌘↵ to run</span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: 'rgba(255,107,107,0.08)',
          border: '1px solid rgba(255,107,107,0.3)',
          borderRadius: '8px',
          padding: '10px 14px',
          fontSize: '12px',
          color: '#ff6b6b',
        }}>
          {error}
        </div>
      )}

      {/* Token chips */}
      {result && (
        <div style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
        }}>
          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.35)' }}>
            Click a token to see its prediction distribution
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {result.predictions.map((pred) => {
              const isSelected = selectedPos === pred.position
              const color = CHIP_COLORS[pred.position % CHIP_COLORS.length]
              return (
                <div
                  key={pred.position}
                  onClick={() => setSelectedPos(pred.position)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '3px',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{
                    background: isSelected ? `${color}33` : `${color}15`,
                    border: `1px solid ${isSelected ? color : color + '44'}`,
                    borderRadius: '6px',
                    padding: '4px 8px',
                    fontSize: '12px',
                    color: isSelected ? color : `${color}bb`,
                    fontFamily: 'inherit',
                    whiteSpace: 'pre',
                    transition: 'all 0.12s',
                    boxShadow: isSelected ? `0 0 8px ${color}33` : 'none',
                  }}>
                    {pred.token}
                  </div>
                  <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.3)' }}>{pred.position}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Predictions bar chart */}
      {focused && (
        <div style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '14px',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.4)' }}>Position {focused.position}</span>
            <span style={{
              background: 'rgba(0,212,255,0.12)',
              border: '1px solid rgba(0,212,255,0.3)',
              borderRadius: '4px',
              padding: '1px 7px',
              fontSize: '12px',
              color: '#00d4ff',
              fontFamily: 'inherit',
              whiteSpace: 'pre',
            }}>
              {focused.token}
            </span>
            <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)' }}>— top {focused.top_k.length} predictions</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
            {focused.top_k.map((entry, i) => {
              const barWidth = maxProb > 0 ? (entry.probability / maxProb) * 100 : 0
              const opacity = 0.35 + 0.65 * (entry.probability / maxProb)
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  {/* Token label */}
                  <div style={{
                    width: '120px',
                    flexShrink: 0,
                    fontSize: '11px',
                    color: 'rgba(255,255,255,0.7)',
                    fontFamily: 'inherit',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'pre',
                    textAlign: 'right',
                  }}>
                    {entry.token_str}
                  </div>
                  {/* Bar */}
                  <div style={{ flex: 1, height: '18px', background: 'rgba(255,255,255,0.04)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${barWidth}%`,
                      background: `rgba(0,212,255,${opacity})`,
                      borderRadius: '3px',
                      transition: 'width 0.2s ease',
                    }} />
                  </div>
                  {/* Probability */}
                  <div style={{
                    width: '52px',
                    flexShrink: 0,
                    fontSize: '10px',
                    color: 'rgba(255,255,255,0.45)',
                    textAlign: 'right',
                    fontFamily: 'inherit',
                  }}>
                    {(entry.probability * 100).toFixed(2)}%
                  </div>
                </div>
              )
            })}
          </div>

          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)' }}>
            logits shape: [{result!.logits_shape.join(' × ')}] · d_vocab = {result!.logits_shape[1]}
          </div>
        </div>
      )}
    </div>
  )
}
