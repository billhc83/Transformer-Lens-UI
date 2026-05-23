import { useRef, useState } from 'react'
import { useSessionStore } from '../../store/sessionStore'
import { useModelStore } from '../../store/modelStore'

const PAGE_LABELS: Record<string, string> = {
  'token-inspector': 'Token Inspector',
  'forward-pass': 'Forward Pass',
  'activation-browser': 'Activation Browser',
  'attention-viz': 'Attention Viz',
  'logit-lens': 'Logit Lens',
  'attribution': 'Attribution',
  'patching-lab': 'Patching Lab',
  'circuit-analyzer': 'Circuit Analyzer',
  'hook-lab': 'Hook Lab',
  'generation-studio': 'Generation Studio',
}

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

function timeAgo(ts: number) {
  const d = Math.floor((Date.now() - ts) / 1000)
  if (d < 60) return `${d}s ago`
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  return `${Math.floor(d / 3600)}h ago`
}

export default function ReportStudio() {
  const findings = useSessionStore((s) => s.findings)
  const clearFindings = useSessionStore((s) => s.clearFindings)
  const modelConfig = useModelStore((s) => s.loadedConfig)

  const [report, setReport] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  async function generate() {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setReport('')
    setError(null)
    setLoading(true)
    try {
      const res = await fetch('/api/report/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelConfig?.name ?? 'unknown',
          findings,
        }),
        signal: ctrl.signal,
      })
      if (!res.ok) {
        setError(`Server error ${res.status}`)
        return
      }
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        setReport((prev) => prev + decoder.decode(value, { stream: true }))
      }
    } catch (e: unknown) {
      if ((e as Error).name !== 'AbortError') {
        setError('Request failed — is Ollama running?')
      }
    } finally {
      setLoading(false)
    }
  }

  function download() {
    const blob = new Blob([report], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `transformerlens-report-${Date.now()}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#0a0a0f',
      overflow: 'hidden',
      fontFamily: 'JetBrains Mono, monospace',
      color: '#fff',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 20px',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#00d4ff' }}>
          Report Studio
        </span>
        <span style={{
          fontSize: 9,
          padding: '2px 7px',
          borderRadius: 4,
          background: 'rgba(168,85,247,0.12)',
          border: '1px solid rgba(168,85,247,0.3)',
          color: '#a855f7',
        }}>
          {findings.length} findings
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {report && (
            <button
              onClick={download}
              style={{
                fontSize: 11,
                padding: '4px 12px',
                borderRadius: 6,
                border: '1px solid rgba(0,212,255,0.4)',
                background: 'transparent',
                color: '#00d4ff',
                cursor: 'pointer',
              }}
            >
              ↓ Download .md
            </button>
          )}
          {report && (
            <button
              onClick={() => navigator.clipboard.writeText(report)}
              style={{
                fontSize: 11,
                padding: '4px 12px',
                borderRadius: 6,
                border: '1px solid rgba(255,255,255,0.15)',
                background: 'transparent',
                color: 'rgba(255,255,255,0.5)',
                cursor: 'pointer',
              }}
            >
              Copy
            </button>
          )}
          <button
            onClick={generate}
            disabled={loading || findings.length === 0}
            style={{
              fontSize: 11,
              padding: '4px 16px',
              borderRadius: 6,
              border: `1px solid ${findings.length === 0 ? 'rgba(255,255,255,0.1)' : 'rgba(168,85,247,0.5)'}`,
              background: loading ? 'rgba(168,85,247,0.1)' : findings.length === 0 ? 'transparent' : 'rgba(168,85,247,0.12)',
              color: findings.length === 0 ? 'rgba(255,255,255,0.2)' : '#a855f7',
              cursor: findings.length === 0 || loading ? 'not-allowed' : 'pointer',
            }}
          >
            {loading ? '⟳ Generating…' : report ? '↺ Regenerate' : '✦ Generate Report'}
          </button>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', gap: 0, overflow: 'hidden', minHeight: 0 }}>
        {/* Findings sidebar */}
        <div style={{
          width: 260,
          flexShrink: 0,
          borderRight: '1px solid rgba(255,255,255,0.07)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '10px 14px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid rgba(255,255,255,0.05)',
            flexShrink: 0,
          }}>
            <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.35)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              Session Findings
            </span>
            {findings.length > 0 && (
              <button
                onClick={clearFindings}
                style={{
                  fontSize: 9,
                  padding: '1px 6px',
                  borderRadius: 3,
                  border: '1px solid rgba(255,107,107,0.25)',
                  background: 'transparent',
                  color: 'rgba(255,107,107,0.5)',
                  cursor: 'pointer',
                }}
              >
                Clear
              </button>
            )}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
            {findings.length === 0 ? (
              <div style={{
                padding: '24px 14px',
                textAlign: 'center',
                fontSize: 10,
                color: 'rgba(255,255,255,0.2)',
                lineHeight: 1.7,
              }}>
                No findings yet.<br />
                Run analyses on other pages to collect findings here.
              </div>
            ) : (
              findings.map((f, i) => (
                <div key={i} style={{
                  padding: '8px 14px',
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 3,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{
                      fontSize: 8,
                      color: '#a855f7',
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                    }}>
                      {PAGE_LABELS[f.page] ?? f.page}
                    </span>
                    <span style={{ fontSize: 8, color: 'rgba(255,255,255,0.2)' }}>
                      {timeAgo(f.timestamp)}
                    </span>
                  </div>
                  <div style={{
                    fontSize: 10,
                    color: 'rgba(255,255,255,0.65)',
                    lineHeight: 1.4,
                  }}>
                    {f.headline}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Report panel */}
        <div style={{ flex: 1, overflowY: 'auto', padding: 20, minWidth: 0 }}>
          {error && (
            <div style={{ color: '#ff6b6b', fontSize: 11, marginBottom: 12 }}>{error}</div>
          )}

          {!report && !loading && (
            <div style={{ ...panel, textAlign: 'center', maxWidth: 420, margin: '40px auto' }}>
              <div style={{ fontSize: 32, marginBottom: 16, opacity: 0.15 }}>◈</div>
              <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', lineHeight: 1.8 }}>
                {findings.length === 0
                  ? 'Collect findings by running analyses on the other pages.\nEach page automatically records a finding when data loads.'
                  : `${findings.length} finding${findings.length > 1 ? 's' : ''} ready.\nClick Generate Report to synthesize them.`}
              </div>
            </div>
          )}

          {(report || loading) && (
            <div style={{
              ...panel,
              fontSize: 12,
              color: 'rgba(255,255,255,0.8)',
              lineHeight: 1.85,
              whiteSpace: 'pre-wrap',
              maxWidth: 760,
            }}>
              {report}
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
      </div>
    </div>
  )
}
