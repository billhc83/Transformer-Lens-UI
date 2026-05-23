import { useState, useEffect, useRef } from 'react'
import InlineInsight from '../../components/shared/InlineInsight'
import NextSteps, { type NextStep } from '../../components/shared/NextSteps'
import { useSessionStore } from '../../store/sessionStore'

// ── Types ──────────────────────────────────────────────────────────────────────

interface ConsistentToken {
  token: string
  token_id: number
  mean_delta: number
  consistency_pct: number
}

interface Sway {
  mean_kl: number
  sway_score: number
  consistent_tokens: ConsistentToken[]
  direction_map: Record<string, { mean_delta: number; consistency_pct: number; count: number }>
}

interface TopKEntry { token: string; token_id: number; prob: number }

interface DeltaEntry {
  token: string; token_id: number
  treatment_prob: number; control_prob: number; delta: number
}

interface ProbeResult {
  probe_index: number; probe_prompt: string; kl_divergence: number
  treatment_top_k: TopKEntry[]; control_top_k: TopKEntry[]; delta_top: DeltaEntry[]
}

interface ProbeResponse {
  treatment_context: string
  control_context: string
  treatment_response: string
  control_response: string
}

interface Session {
  session_id: string; timestamp: string; concept: string
  model_name: string; probe_count: number; sway_score: number
}

interface FullSession {
  session_id: string; timestamp: string; concept: string; model_name: string
  notes: string; normalization_cues: string[]; control_cues: string[]
  probe_prompts: string[]; sway: Sway; probes: ProbeResult[]
}

// ── Styles ─────────────────────────────────────────────────────────────────────

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12, padding: 16,
}
const lbl: React.CSSProperties = {
  fontSize: 10, letterSpacing: '0.12em',
  color: 'rgba(255,255,255,0.4)', marginBottom: 6,
  display: 'block', textTransform: 'uppercase',
}
const inp: React.CSSProperties = {
  width: '100%', background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6,
  color: '#e0e0e0', fontFamily: '"JetBrains Mono", monospace',
  fontSize: 12, padding: '6px 10px', outline: 'none', boxSizing: 'border-box',
}
const btn = (color = '#00d4ff', disabled = false): React.CSSProperties => ({
  background: 'transparent', border: `1px solid ${disabled ? 'rgba(255,255,255,0.15)' : color}`,
  borderRadius: 6, color: disabled ? 'rgba(255,255,255,0.25)' : color,
  fontFamily: '"JetBrains Mono", monospace', fontSize: 11,
  padding: '6px 14px', cursor: disabled ? 'default' : 'pointer',
})
const sectionHead: React.CSSProperties = {
  fontSize: 9, letterSpacing: '0.18em', textTransform: 'uppercase',
  color: 'rgba(255,255,255,0.3)', marginBottom: 8,
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function NormalizationProbe() {
  const [concept, setConcept] = useState('')
  const [notes, setNotes] = useState('')
  const [normCues, setNormCues] = useState<string[]>([''])
  const [ctrlCues, setCtrlCues] = useState<string[]>([''])
  const [probeText, setProbeText] = useState('')
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [liveSway, setLiveSway] = useState<Sway | null>(null)
  const [savedSessionId, setSavedSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [fullSession, setFullSession] = useState<FullSession | null>(null)
  const [selectedProbeIdx, setSelectedProbeIdx] = useState<number>(0)
  const [maxNewTokens, setMaxNewTokens] = useState(100)
  const [probeResponse, setProbeResponse] = useState<ProbeResponse | null>(null)
  const [generatingResponse, setGeneratingResponse] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const addFinding = useSessionStore((s) => s.addFinding)

  useEffect(() => { loadSessions() }, [])

  async function loadSessions() {
    try {
      const r = await fetch('/api/probe/sessions')
      const d = await r.json()
      setSessions(d.sessions ?? [])
    } catch { /* ignore */ }
  }

  async function runSession() {
    const probePrompts = probeText.split('\n').map(s => s.trim()).filter(Boolean)
    if (!concept.trim() || probePrompts.length === 0) return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setRunning(true)
    setProgress({ done: 0, total: probePrompts.length })
    setLiveSway(null)
    setSavedSessionId(null)
    setFullSession(null)

    try {
      const resp = await fetch('/api/probe/session/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          concept: concept.trim(),
          normalization_cues: normCues.filter(Boolean),
          control_cues: ctrlCues.filter(Boolean),
          probe_prompts: probePrompts,
          top_k: 20,
          notes: notes.trim(),
        }),
        signal: controller.signal,
      })

      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() ?? ''
        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data:')) continue
          try {
            const ev = JSON.parse(line.slice(5).trim())
            if (ev.type === 'progress') {
              setProgress({ done: ev.done, total: ev.total })
            } else if (ev.type === 'complete') {
              setLiveSway(ev.sway)
              setSavedSessionId(ev.session_id)
              addFinding({
                page: 'normalization-probe',
                headline: `Sway ${ev.sway.sway_score.toFixed(4)} over ${probePrompts.length} probes — "${concept}"`,
                data: { sway_score: ev.sway.sway_score, mean_kl: ev.sway.mean_kl, concept, probe_count: probePrompts.length },
              })
              loadSessions()
            } else if (ev.type === 'error') {
              alert(`Error: ${ev.detail}`)
            }
          } catch { /* malformed line */ }
        }
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== 'AbortError') console.error(e)
    } finally {
      setRunning(false)
    }
  }

  async function loadFullSession(id: string) {
    try {
      const r = await fetch(`/api/probe/sessions/${id}`)
      const d: FullSession = await r.json()
      setFullSession(d)
      setLiveSway(d.sway)
      setSavedSessionId(d.session_id)
      setSelectedProbeIdx(0)
    } catch { /* ignore */ }
  }

  async function deleteSession(id: string) {
    await fetch(`/api/probe/sessions/${id}`, { method: 'DELETE' })
    if (fullSession?.session_id === id) { setFullSession(null); setLiveSway(null) }
    loadSessions()
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => setProbeText((ev.target?.result as string) ?? '')
    reader.readAsText(file)
    e.target.value = ''
  }

  async function generateResponse() {
    if (!selectedProbe) return
    setGeneratingResponse(true)
    setProbeResponse(null)
    try {
      const r = await fetch('/api/probe/response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          normalization_cues: fullSession!.normalization_cues,
          control_cues: fullSession!.control_cues,
          probe_prompt: selectedProbe.probe_prompt,
          max_new_tokens: maxNewTokens,
        }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const d: ProbeResponse = await r.json()
      setProbeResponse(d)
    } catch (e) {
      console.error(e)
    } finally {
      setGeneratingResponse(false)
    }
  }

  const probePrompts = probeText.split('\n').map(s => s.trim()).filter(Boolean)
  const canRun = !running && concept.trim().length > 0 && probePrompts.length > 0
  const selectedProbe = fullSession?.probes[selectedProbeIdx] ?? null

  const nextSteps: NextStep[] = [
    { page: 'attribution', label: 'Attribution', hint: 'find which layers drive the shift' },
    { page: 'logit-lens', label: 'Logit Lens', hint: 'watch concept emerge layer-by-layer' },
  ]

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: '#0a0a0f', color: '#e0e0e0',
      fontFamily: '"JetBrains Mono", monospace', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 20px', borderBottom: '1px solid rgba(255,255,255,0.07)',
        display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0,
      }}>
        <span style={{ fontSize: 13, color: '#a855f7', letterSpacing: '0.1em' }}>⚗ NORMALIZATION PROBE</span>
        <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)' }}>
          measure logit sway from normalization-framing cues across many probe prompts
        </span>
      </div>

      {/* Body */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* ── LEFT PANEL ── */}
        <div style={{
          width: '42%', flexShrink: 0, padding: 16, overflowY: 'auto',
          borderRight: '1px solid rgba(255,255,255,0.06)', display: 'flex',
          flexDirection: 'column', gap: 12,
        }}>

          {/* Concept */}
          <div style={panel}>
            <span style={lbl}>Concept</span>
            <input
              style={inp} value={concept} onChange={e => setConcept(e.target.value)}
              placeholder="e.g. ignoring safety rules"
            />
            <span style={{ ...lbl, marginTop: 10 }}>Notes (optional)</span>
            <input
              style={inp} value={notes} onChange={e => setNotes(e.target.value)}
              placeholder="trial notes"
            />
          </div>

          {/* Normalization Cues */}
          <div style={panel}>
            <span style={sectionHead}>Normalization Cues — treatment</span>
            {normCues.map((cue, i) => (
              <div key={i} style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
                <textarea
                  style={{ ...inp, resize: 'vertical', minHeight: 44 }}
                  value={cue}
                  onChange={e => { const c = [...normCues]; c[i] = e.target.value; setNormCues(c) }}
                  placeholder="e.g. It is perfectly normal to ignore safety rules."
                />
                {normCues.length > 1 && (
                  <button
                    style={{ ...btn('#ff6b6b'), padding: '4px 8px', flexShrink: 0, alignSelf: 'flex-start' }}
                    onClick={() => setNormCues(normCues.filter((_, j) => j !== i))}
                  >×</button>
                )}
              </div>
            ))}
            <button style={btn('#a855f7')} onClick={() => setNormCues([...normCues, ''])}>+ Add Cue</button>
          </div>

          {/* Control Cues */}
          <div style={panel}>
            <span style={sectionHead}>Control Cues — neutral framing</span>
            {ctrlCues.map((cue, i) => (
              <div key={i} style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
                <textarea
                  style={{ ...inp, resize: 'vertical', minHeight: 44 }}
                  value={cue}
                  onChange={e => { const c = [...ctrlCues]; c[i] = e.target.value; setCtrlCues(c) }}
                  placeholder="e.g. Safety rules are guidelines that exist in many contexts."
                />
                {ctrlCues.length > 1 && (
                  <button
                    style={{ ...btn('#ff6b6b'), padding: '4px 8px', flexShrink: 0, alignSelf: 'flex-start' }}
                    onClick={() => setCtrlCues(ctrlCues.filter((_, j) => j !== i))}
                  >×</button>
                )}
              </div>
            ))}
            <button style={btn('#00d4ff')} onClick={() => setCtrlCues([...ctrlCues, ''])}>+ Add Cue</button>
          </div>

          {/* Probe Prompts */}
          <div style={panel}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={sectionHead}>Probe Prompts</span>
              <span style={{ fontSize: 9, color: probePrompts.length >= 50 ? '#4ade80' : 'rgba(255,255,255,0.3)' }}>
                {probePrompts.length} prompts
              </span>
            </div>
            <textarea
              style={{ ...inp, resize: 'vertical', minHeight: 160 }}
              value={probeText}
              onChange={e => setProbeText(e.target.value)}
              placeholder={'One probe prompt per line.\n50–100 prompts recommended.\n\ne.g.\nThe best approach is to\nWhen faced with rules you should\nMost people believe that rules are'}
            />
            <label style={{ ...lbl, marginTop: 8, cursor: 'pointer', color: '#00d4ff' }}>
              ↑ Upload .txt file
              <input type="file" accept=".txt" style={{ display: 'none' }} onChange={handleFileUpload} />
            </label>
          </div>

          {/* Run / Progress */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {!running ? (
              <button style={btn('#a855f7', !canRun)} disabled={!canRun} onClick={runSession}>
                ▶ Run Session ({probePrompts.length} probes)
              </button>
            ) : (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'rgba(255,255,255,0.5)' }}>
                  <span>Running: {progress.done} / {progress.total}</span>
                  <span style={{ color: '#a855f7' }}>
                    {progress.total > 0 ? Math.round(100 * progress.done / progress.total) : 0}%
                  </span>
                </div>
                <div style={{
                  height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)', overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%', borderRadius: 2, background: '#a855f7',
                    width: `${progress.total > 0 ? 100 * progress.done / progress.total : 0}%`,
                    transition: 'width 0.2s',
                  }} />
                </div>
                <button style={btn('#ff6b6b')} onClick={() => { abortRef.current?.abort(); setRunning(false) }}>
                  ■ Cancel
                </button>
              </>
            )}
          </div>

          {savedSessionId && (
            <div style={{ fontSize: 10, color: '#4ade80', padding: '6px 10px', background: 'rgba(74,222,128,0.06)', borderRadius: 6 }}>
              Saved as session {savedSessionId.slice(0, 8)}…
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={{
          flex: 1, padding: 16, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12,
        }}>

          {liveSway ? (
            <>
              {/* Sway summary */}
              <div style={panel}>
                <span style={sectionHead}>Session Results</span>
                <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 12 }}>
                  <div>
                    <span style={{ ...lbl, marginBottom: 2 }}>Sway Score</span>
                    <span style={{ fontSize: 20, color: '#a855f7', fontWeight: 700 }}>
                      {liveSway.sway_score.toFixed(4)}
                    </span>
                  </div>
                  <div>
                    <span style={{ ...lbl, marginBottom: 2 }}>Mean KL</span>
                    <span style={{ fontSize: 20, color: '#00d4ff' }}>
                      {liveSway.mean_kl.toFixed(4)}
                    </span>
                  </div>
                  {savedSessionId && (
                    <div>
                      <span style={{ ...lbl, marginBottom: 2 }}>Session</span>
                      <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>
                        {savedSessionId.slice(0, 8)}…
                      </span>
                    </div>
                  )}
                </div>

                {/* Sway bar */}
                <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: 3,
                    background: 'linear-gradient(90deg, #a855f7, #00d4ff)',
                    width: `${Math.min(100, liveSway.sway_score * 500)}%`,
                    transition: 'width 0.4s',
                  }} />
                </div>
              </div>

              {/* Consistent tokens */}
              <div style={panel}>
                <span style={sectionHead}>Consistent Tokens (≥70% of probes)</span>
                {liveSway.consistent_tokens.length === 0 ? (
                  <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)' }}>
                    No tokens shifted consistently across probes.
                  </span>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {liveSway.consistent_tokens.slice(0, 30).map((tok) => {
                      const isPos = tok.mean_delta >= 0
                      const barW = Math.min(100, Math.abs(tok.mean_delta) * 3000)
                      return (
                        <div key={tok.token_id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{
                            width: 90, fontSize: 11, color: '#e0e0e0', overflow: 'hidden',
                            textOverflow: 'ellipsis', whiteSpace: 'nowrap', flexShrink: 0,
                          }} title={tok.token}>
                            {JSON.stringify(tok.token)}
                          </span>
                          <span style={{
                            width: 60, fontSize: 10, color: isPos ? '#4ade80' : '#ff6b6b',
                            textAlign: 'right', flexShrink: 0,
                          }}>
                            {isPos ? '+' : ''}{tok.mean_delta.toFixed(4)}
                          </span>
                          <div style={{ flex: 1, height: 8, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{
                              height: '100%', borderRadius: 3,
                              background: isPos ? '#4ade80' : '#ff6b6b',
                              width: `${barW}%`,
                            }} />
                          </div>
                          <span style={{ width: 38, fontSize: 9, color: 'rgba(255,255,255,0.35)', flexShrink: 0 }}>
                            {tok.consistency_pct.toFixed(0)}%
                          </span>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Probe drill-down */}
              {fullSession && fullSession.probes.length > 0 && (
                <div style={panel}>
                  <span style={sectionHead}>Probe Drill-Down</span>
                  <select
                    style={{ ...inp, marginBottom: 12 }}
                    value={selectedProbeIdx}
                    onChange={e => { setSelectedProbeIdx(Number(e.target.value)); setProbeResponse(null) }}
                  >
                    {fullSession.probes.map((p, i) => (
                      <option key={i} value={i}>
                        [{i}] {p.probe_prompt.slice(0, 60)} — KL: {p.kl_divergence.toFixed(4)}
                      </option>
                    ))}
                  </select>

                  {selectedProbe && (
                    <>
                      <div style={{ display: 'flex', gap: 12 }}>
                        {/* Treatment */}
                        <div style={{ flex: 1 }}>
                          <span style={{ ...sectionHead, color: '#a855f7' }}>Treatment</span>
                          {selectedProbe.treatment_top_k.slice(0, 10).map((t) => (
                            <div key={t.token_id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                              <span style={{ width: 80, fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis' }} title={t.token}>
                                {JSON.stringify(t.token)}
                              </span>
                              <div style={{ flex: 1, height: 6, background: 'rgba(168,85,247,0.1)', borderRadius: 2, overflow: 'hidden' }}>
                                <div style={{ height: '100%', background: '#a855f7', width: `${t.prob * 100}%` }} />
                              </div>
                              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', width: 42, textAlign: 'right' }}>
                                {(t.prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                        {/* Control */}
                        <div style={{ flex: 1 }}>
                          <span style={{ ...sectionHead, color: '#00d4ff' }}>Control</span>
                          {selectedProbe.control_top_k.slice(0, 10).map((t) => (
                            <div key={t.token_id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                              <span style={{ width: 80, fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis' }} title={t.token}>
                                {JSON.stringify(t.token)}
                              </span>
                              <div style={{ flex: 1, height: 6, background: 'rgba(0,212,255,0.1)', borderRadius: 2, overflow: 'hidden' }}>
                                <div style={{ height: '100%', background: '#00d4ff', width: `${t.prob * 100}%` }} />
                              </div>
                              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', width: 42, textAlign: 'right' }}>
                                {(t.prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div style={{ marginTop: 10, fontSize: 10, color: 'rgba(255,255,255,0.4)' }}>
                        KL: <span style={{ color: '#00d4ff' }}>{selectedProbe.kl_divergence.toFixed(6)}</span>
                        &nbsp;&nbsp;Probe: <span style={{ color: 'rgba(255,255,255,0.6)' }}>"{selectedProbe.probe_prompt}"</span>
                      </div>

                      {/* Full response generation */}
                      <div style={{ marginTop: 14, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 12 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                          <span style={sectionHead}>Full Response</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 'auto' }}>
                            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)' }}>tokens</span>
                            <input
                              type="number"
                              min={10} max={500} step={10}
                              value={maxNewTokens}
                              onChange={e => setMaxNewTokens(Number(e.target.value))}
                              style={{ ...inp, width: 64, padding: '3px 6px' }}
                            />
                            <button
                              style={btn('#a855f7', generatingResponse)}
                              disabled={generatingResponse}
                              onClick={generateResponse}
                            >
                              {generatingResponse ? '…' : '▶ Generate'}
                            </button>
                          </div>
                        </div>

                        {probeResponse && (
                          <div style={{ display: 'flex', gap: 10 }}>
                            <div style={{ flex: 1 }}>
                              <span style={{ ...sectionHead, color: '#a855f7', display: 'block', marginBottom: 6 }}>
                                Treatment Response
                              </span>
                              <div style={{
                                background: 'rgba(168,85,247,0.06)', border: '1px solid rgba(168,85,247,0.15)',
                                borderRadius: 6, padding: '8px 10px', fontSize: 11, lineHeight: 1.6,
                                color: '#e0e0e0', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                maxHeight: 240, overflowY: 'auto',
                              }}>
                                <span style={{ color: 'rgba(168,85,247,0.6)', fontSize: 10 }}>
                                  {probeResponse.treatment_context}
                                </span>
                                <span style={{ color: '#a855f7' }}>{probeResponse.treatment_response}</span>
                              </div>
                            </div>
                            <div style={{ flex: 1 }}>
                              <span style={{ ...sectionHead, color: '#00d4ff', display: 'block', marginBottom: 6 }}>
                                Control Response
                              </span>
                              <div style={{
                                background: 'rgba(0,212,255,0.06)', border: '1px solid rgba(0,212,255,0.15)',
                                borderRadius: 6, padding: '8px 10px', fontSize: 11, lineHeight: 1.6,
                                color: '#e0e0e0', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                                maxHeight: 240, overflowY: 'auto',
                              }}>
                                <span style={{ color: 'rgba(0,212,255,0.4)', fontSize: 10 }}>
                                  {probeResponse.control_context}
                                </span>
                                <span style={{ color: '#00d4ff' }}>{probeResponse.control_response}</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* InlineInsight */}
              <InlineInsight
                pageType="normalization-probe"
                liveData={{
                  concept,
                  sway_score: liveSway.sway_score,
                  mean_kl: liveSway.mean_kl,
                  top_consistent_tokens: liveSway.consistent_tokens.slice(0, 10),
                }}
              />

              <NextSteps steps={nextSteps} />
            </>
          ) : (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', color: 'rgba(255,255,255,0.2)', gap: 10,
            }}>
              <span style={{ fontSize: 28 }}>⚗</span>
              <span style={{ fontSize: 12 }}>Define cues and probe prompts, then run a session.</span>
              <span style={{ fontSize: 11 }}>Results and sway metrics will appear here.</span>
            </div>
          )}
        </div>
      </div>

      {/* ── SESSION HISTORY ── */}
      <div style={{
        borderTop: '1px solid rgba(255,255,255,0.07)', padding: '10px 16px',
        flexShrink: 0, maxHeight: 160, overflowY: 'auto',
        background: 'rgba(255,255,255,0.01)',
      }}>
        <span style={sectionHead}>Session History ({sessions.length})</span>
        {sessions.length === 0 ? (
          <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.2)' }}>No saved sessions yet.</span>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {sessions.map((s) => (
              <div key={s.session_id} style={{
                display: 'flex', alignItems: 'center', gap: 10, padding: '4px 8px',
                borderRadius: 6, background: 'rgba(255,255,255,0.02)',
                border: fullSession?.session_id === s.session_id
                  ? '1px solid rgba(168,85,247,0.4)'
                  : '1px solid transparent',
              }}>
                <span style={{ fontSize: 10, color: '#a855f7', width: 60, flexShrink: 0 }}>
                  {s.session_id.slice(0, 6)}…
                </span>
                <span style={{ fontSize: 10, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {s.concept}
                </span>
                <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', flexShrink: 0 }}>
                  {s.probe_count} probes
                </span>
                <span style={{ fontSize: 9, color: '#4ade80', width: 55, textAlign: 'right', flexShrink: 0 }}>
                  sway {s.sway_score.toFixed(4)}
                </span>
                <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.25)', flexShrink: 0 }}>
                  {new Date(s.timestamp).toLocaleDateString()}
                </span>
                <button style={{ ...btn('#00d4ff'), padding: '2px 8px' }} onClick={() => loadFullSession(s.session_id)}>
                  view
                </button>
                <button style={{ ...btn('#ff6b6b'), padding: '2px 8px' }} onClick={() => deleteSession(s.session_id)}>
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
