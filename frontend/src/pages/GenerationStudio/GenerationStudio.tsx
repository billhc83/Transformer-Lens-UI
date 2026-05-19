import { useState, useRef, useCallback } from 'react'

const WS_URL = `ws://${window.location.host}/ws/generate`

interface GeneratedToken {
  token: string
  token_id: number
  logprob: number
  step: number
  top_tokens: { token: string; prob: number }[]
  activations: Record<string, { shape: number[]; mean: number; max: number; min: number }>
}

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

const lbl: React.CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.12em',
  color: 'rgba(255,255,255,0.4)',
  marginBottom: 6,
  display: 'block',
  textTransform: 'uppercase',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 6,
  color: '#e0e0e0',
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: 12,
  padding: '6px 10px',
  outline: 'none',
  boxSizing: 'border-box',
}

const btn = (color = '#00d4ff'): React.CSSProperties => ({
  background: 'transparent',
  border: `1px solid ${color}`,
  borderRadius: 6,
  color,
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: 11,
  padding: '6px 14px',
  cursor: 'pointer',
  transition: 'background 0.15s',
})

function logprobToOpacity(logprob: number): number {
  // logprob in range roughly -10 to 0; map to 0.35–1.0
  const clamped = Math.max(-8, Math.min(0, logprob))
  return 0.35 + ((clamped + 8) / 8) * 0.65
}

export default function GenerationStudio() {
  const [prompt, setPrompt] = useState('The Eiffel Tower is in')
  const [maxTokens, setMaxTokens] = useState(50)
  const [temperature, setTemperature] = useState(1.0)
  const [topK, setTopK] = useState(40)
  const [monitors, setMonitors] = useState<string[]>([])
  const [monitorInput, setMonitorInput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [tokens, setTokens] = useState<GeneratedToken[]>([])
  const [scrubStep, setScrubStep] = useState<number | null>(null)
  const [error, setError] = useState('')

  const wsRef = useRef<WebSocket | null>(null)

  const addMonitor = () => {
    const name = monitorInput.trim()
    if (name && !monitors.includes(name)) {
      setMonitors(prev => [...prev, name])
    }
    setMonitorInput('')
  }

  const removeMonitor = (name: string) => setMonitors(prev => prev.filter(m => m !== name))

  const startGeneration = useCallback(() => {
    wsRef.current?.close()
    setTokens([])
    setScrubStep(null)
    setError('')
    setIsGenerating(true)

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({ prompt, max_tokens: maxTokens, temperature, top_k: topK, monitors }))
    }

    ws.onmessage = (e: MessageEvent) => {
      const msg = JSON.parse(e.data as string)
      if (msg.type === 'token') {
        setTokens(prev => [...prev, msg as GeneratedToken])
      } else if (msg.type === 'done' || msg.type === 'error') {
        if (msg.type === 'error') setError(msg.message as string)
        setIsGenerating(false)
      }
    }

    ws.onclose = () => setIsGenerating(false)
    ws.onerror = () => { setError('WebSocket connection error'); setIsGenerating(false) }
  }, [prompt, maxTokens, temperature, topK, monitors])

  const stopGeneration = () => {
    wsRef.current?.close()
    setIsGenerating(false)
  }

  const activeIdx = scrubStep ?? tokens.length - 1
  const activeToken = tokens[activeIdx] ?? null

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden', fontFamily: '"JetBrains Mono", monospace', color: '#e0e0e0' }}>

      {/* ── LEFT: controls ──────────────────────────────────────────── */}
      <div style={{ width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column', padding: 18, gap: 12, overflowY: 'auto', borderRight: '1px solid rgba(255,255,255,0.07)' }}>

        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ fontSize: 17, fontWeight: 700, color: '#00d4ff', letterSpacing: '0.05em' }}>
              Generation Studio
            </div>
            {tokens.length > 0 && (
              <span style={{ fontSize: 9, padding: '2px 6px', borderRadius: 4, background: 'rgba(0,212,255,0.12)', color: '#00d4ff' }}>
                {tokens.length} tok
              </span>
            )}
          </div>
          <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', marginTop: 4 }}>
            Autoregressive streaming with live activations.
          </div>
        </div>

        {/* Prompt */}
        <div style={panel}>
          <span style={lbl}>Prompt</span>
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            rows={4}
            spellCheck={false}
            style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.6 }}
          />
        </div>

        {/* Max tokens */}
        <div style={panel}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={lbl}>Max Tokens</span>
            <span style={{ fontSize: 12, color: '#00d4ff' }}>{maxTokens}</span>
          </div>
          <input type="range" min={10} max={200} value={maxTokens}
            onChange={e => setMaxTokens(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#00d4ff' }}
          />
        </div>

        {/* Temperature */}
        <div style={panel}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={lbl}>Temperature</span>
            <span style={{ fontSize: 12, color: '#00d4ff' }}>{temperature.toFixed(1)}</span>
          </div>
          <input type="range" min={0.1} max={2.0} step={0.1} value={temperature}
            onChange={e => setTemperature(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#00d4ff' }}
          />
        </div>

        {/* Top-K */}
        <div style={panel}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={lbl}>Top-K</span>
            <span style={{ fontSize: 12, color: '#00d4ff' }}>{topK}</span>
          </div>
          <input type="range" min={1} max={100} value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#00d4ff' }}
          />
        </div>

        {/* Activation monitors */}
        <div style={panel}>
          <span style={lbl}>Activation Monitors</span>
          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            <input
              type="text"
              value={monitorInput}
              onChange={e => setMonitorInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') addMonitor() }}
              placeholder="blocks.5.attn.hook_z"
              style={{ ...inputStyle, flex: 1 }}
            />
            <button onClick={addMonitor} style={{ ...btn('#a855f7'), padding: '6px 10px', flexShrink: 0 }}>+</button>
          </div>
          {monitors.length === 0 && (
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.2)' }}>No monitors. Generation still works.</div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {monitors.map(m => (
              <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, padding: '2px 8px', background: 'rgba(168,85,247,0.1)', border: '1px solid rgba(168,85,247,0.25)', borderRadius: 20, color: '#a855f7' }}>
                <span style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m}</span>
                <span style={{ cursor: 'pointer', opacity: 0.7 }} onClick={() => removeMonitor(m)}>×</span>
              </div>
            ))}
          </div>
        </div>

        {/* Generate / Stop */}
        {!isGenerating ? (
          <button onClick={startGeneration} style={{ ...btn('#4ade80'), padding: '9px 0', fontSize: 12, fontWeight: 600 }}>
            ▶ Generate
          </button>
        ) : (
          <button onClick={stopGeneration} style={{ ...btn('#ff6b6b'), padding: '9px 0', fontSize: 12, fontWeight: 600 }}>
            ■ Stop
          </button>
        )}

        {error && (
          <div style={{ fontSize: 10, color: '#ff6b6b', background: 'rgba(255,107,107,0.07)', border: '1px solid rgba(255,107,107,0.2)', borderRadius: 6, padding: '8px 10px' }}>
            {error}
          </div>
        )}
      </div>

      {/* ── CENTER: token stream + scrubber ─────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 18, gap: 14, overflow: 'hidden' }}>

        <div style={{ fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.08em' }}>
          Generated Output
        </div>

        {/* Token stream display */}
        <div style={{
          flex: 1,
          ...panel,
          overflowY: 'auto',
          lineHeight: 1.8,
          fontSize: 14,
          wordBreak: 'break-word',
          position: 'relative',
        }}>
          {tokens.length === 0 && !isGenerating && (
            <div style={{ color: 'rgba(255,255,255,0.2)', fontSize: 12 }}>
              Enter a prompt and click Generate…
            </div>
          )}

          {/* Prompt prefix */}
          {tokens.length > 0 && (
            <span style={{ color: 'rgba(255,255,255,0.3)' }}>{prompt}</span>
          )}

          {tokens.map((t, i) => {
            const opacity = logprobToOpacity(t.logprob)
            const isActive = i === activeIdx && scrubStep !== null
            return (
              <span
                key={i}
                onClick={() => setScrubStep(i)}
                style={{
                  color: `rgba(0,212,255,${opacity})`,
                  background: isActive ? 'rgba(0,212,255,0.12)' : 'transparent',
                  borderRadius: 3,
                  cursor: 'pointer',
                  transition: 'background 0.1s',
                  padding: '1px 2px',
                }}
                title={`logprob: ${t.logprob.toFixed(3)}`}
              >
                {t.token}
              </span>
            )
          })}

          {isGenerating && (
            <span style={{ display: 'inline-block', width: 8, height: 14, background: '#00d4ff', marginLeft: 2, animation: 'blink 1s step-end infinite', verticalAlign: 'text-bottom', opacity: 0.8 }} />
          )}
        </div>

        {/* Timeline scrubber */}
        {tokens.length > 1 && (
          <div style={{ ...panel, padding: '12px 16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={lbl}>Timeline Scrubber</span>
              <span style={{ fontSize: 10, color: '#a855f7' }}>
                {scrubStep !== null ? `Step ${scrubStep}` : 'Live'}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={tokens.length - 1}
              value={scrubStep ?? tokens.length - 1}
              onChange={e => setScrubStep(Number(e.target.value))}
              style={{ width: '100%', accentColor: '#a855f7' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 9, color: 'rgba(255,255,255,0.3)' }}>
              <span>t=0</span>
              <button
                onClick={() => setScrubStep(null)}
                style={{ ...btn('#a855f7'), padding: '1px 8px', fontSize: 9 }}
              >
                → Live
              </button>
              <span>t={tokens.length - 1}</span>
            </div>
          </div>
        )}
      </div>

      {/* ── RIGHT: per-token analysis ────────────────────────────────── */}
      <div style={{ width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column', padding: 18, gap: 14, overflowY: 'auto', borderLeft: '1px solid rgba(255,255,255,0.07)' }}>

        <div style={{ fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.08em' }}>
          {activeToken ? `Step ${activeToken.step}` : 'Token Analysis'}
        </div>

        {!activeToken && (
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.2)' }}>
            {tokens.length === 0 ? 'Start generating to see per-token stats.' : 'Click a token to inspect it.'}
          </div>
        )}

        {activeToken && (
          <>
            {/* Token info */}
            <div style={panel}>
              <span style={lbl}>Token</span>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#00d4ff', marginBottom: 6 }}>
                {JSON.stringify(activeToken.token)}
              </div>
              <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }}>
                id={activeToken.token_id} &nbsp;|&nbsp; logp={activeToken.logprob.toFixed(3)} &nbsp;|&nbsp; p={Math.exp(activeToken.logprob).toFixed(4)}
              </div>
            </div>

            {/* Top-5 probability distribution */}
            <div style={panel}>
              <span style={lbl}>Probability Distribution</span>
              {activeToken.top_tokens.map((t, i) => (
                <div key={i} style={{ marginBottom: 7 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                    <span style={{ fontSize: 11, color: i === 0 ? '#00d4ff' : '#e0e0e0' }}>
                      {JSON.stringify(t.token)}
                    </span>
                    <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)' }}>
                      {(t.prob * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2 }}>
                    <div style={{
                      width: `${t.prob * 100}%`,
                      height: '100%',
                      background: i === 0 ? '#00d4ff' : 'rgba(0,212,255,0.4)',
                      borderRadius: 2,
                      transition: 'width 0.3s',
                    }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Activation monitor stats */}
            {Object.keys(activeToken.activations).length > 0 && (
              <div style={panel}>
                <span style={lbl}>Activation Monitors</span>
                {Object.entries(activeToken.activations).map(([name, stats]) => (
                  <div key={name} style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 10, color: '#a855f7', marginBottom: 4, wordBreak: 'break-all' }}>{name}</div>
                    <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.5)', marginBottom: 3 }}>
                      shape: [{stats.shape.join(', ')}]
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3px 8px', fontSize: 10 }}>
                      {[
                        { k: 'mean', v: stats.mean },
                        { k: 'max',  v: stats.max },
                        { k: 'min',  v: stats.min },
                      ].map(({ k, v }) => (
                        <div key={k} style={{ color: 'rgba(255,255,255,0.4)' }}>
                          {k}: <span style={{ color: '#e0e0e0' }}>{v.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      <style>{`
        @keyframes blink { 0%,100%{opacity:0.8} 50%{opacity:0} }
      `}</style>
    </div>
  )
}
