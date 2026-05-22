import { useEffect, useRef, useState } from 'react'

interface Props {
  pageType: string
  liveData: Record<string, unknown> | null
}

export default function InlineInsight({ pageType, liveData }: Props) {
  const [insight, setInsight] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const firedForRef = useRef<string | null>(null)

  useEffect(() => {
    if (!liveData) return
    const key = JSON.stringify(liveData)
    if (firedForRef.current === key) return
    firedForRef.current = key

    const timer = setTimeout(() => fire(liveData), 400)
    return () => clearTimeout(timer)
  }, [liveData]) // eslint-disable-line react-hooks/exhaustive-deps

  async function fire(data: Record<string, unknown>) {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setInsight('')
    setError(null)
    setLoading(true)
    setCollapsed(false)
    try {
      const res = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page_type: pageType, data }),
        signal: ctrl.signal,
      })
      if (!res.ok) {
        setError(`Ollama error ${res.status}`)
        return
      }
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        setInsight((prev) => prev + decoder.decode(value, { stream: true }))
      }
    } catch (e: unknown) {
      if ((e as Error).name !== 'AbortError') setError('Ollama unreachable')
    } finally {
      setLoading(false)
    }
  }

  if (!liveData) return null

  return (
    <div style={{
      borderTop: '1px solid rgba(168,85,247,0.12)',
      background: 'rgba(168,85,247,0.02)',
      flexShrink: 0,
    }}>
      {/* Header bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '7px 20px',
          cursor: 'pointer',
          userSelect: 'none',
        }}
        onClick={() => setCollapsed((c) => !c)}
      >
        <span style={{
          fontSize: 9,
          color: '#a855f7',
          fontFamily: 'JetBrains Mono, monospace',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
        }}>
          ✦ AI Insight
        </span>
        <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.2)', fontFamily: 'JetBrains Mono, monospace' }}>
          qwen3:14b
        </span>
        {loading && (
          <span style={{ fontSize: 9, color: '#a855f7', fontFamily: 'JetBrains Mono, monospace', opacity: 0.7 }}>
            ⟳ generating…
          </span>
        )}
        {!loading && insight && !collapsed && (
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.2)', fontFamily: 'JetBrains Mono, monospace', marginLeft: 'auto' }}>
            ▴ collapse
          </span>
        )}
        {!loading && insight && collapsed && (
          <span
            style={{
              fontSize: 10,
              color: 'rgba(255,255,255,0.4)',
              fontFamily: 'JetBrains Mono, monospace',
              flex: 1,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {insight.slice(0, 80)}…  ▾
          </span>
        )}
        {!loading && !insight && !error && (
          <button
            onClick={(e) => { e.stopPropagation(); if (liveData) fire(liveData) }}
            style={{
              fontSize: 9,
              padding: '2px 8px',
              borderRadius: 4,
              border: '1px solid rgba(168,85,247,0.35)',
              background: 'transparent',
              color: '#a855f7',
              cursor: 'pointer',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            Generate
          </button>
        )}
        {error && (
          <span style={{ fontSize: 9, color: '#ff6b6b', fontFamily: 'JetBrains Mono, monospace' }}>
            {error}
          </span>
        )}
        {!loading && insight && (
          <button
            onClick={(e) => { e.stopPropagation(); if (liveData) fire(liveData) }}
            style={{
              fontSize: 9,
              padding: '2px 8px',
              borderRadius: 4,
              border: '1px solid rgba(168,85,247,0.25)',
              background: 'transparent',
              color: 'rgba(168,85,247,0.6)',
              cursor: 'pointer',
              fontFamily: 'JetBrains Mono, monospace',
              marginLeft: 'auto',
              flexShrink: 0,
            }}
          >
            ↺
          </button>
        )}
      </div>

      {/* Body */}
      {!collapsed && (insight || loading) && (
        <div style={{
          padding: '0 20px 12px',
          fontSize: 11,
          fontFamily: 'JetBrains Mono, monospace',
          color: 'rgba(255,255,255,0.75)',
          lineHeight: 1.75,
          whiteSpace: 'pre-wrap',
        }}>
          {insight}
          {loading && (
            <span style={{
              display: 'inline-block',
              width: 7,
              height: 11,
              background: '#a855f7',
              marginLeft: 2,
              animation: 'blink 1s step-end infinite',
              verticalAlign: 'text-bottom',
            }} />
          )}
          <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}`}</style>
        </div>
      )}
    </div>
  )
}
