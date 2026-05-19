import { useState, useEffect } from 'react'
import axios from 'axios'

const API = ''

const TEMPLATES = [
  {
    label: 'Zero Ablation',
    code: `def hook_fn(value, hook):\n    return torch.zeros_like(value)`,
  },
  {
    label: 'Mean Ablation',
    code: `def hook_fn(value, hook):\n    mean_value = value.mean(dim=-1, keepdim=True)\n    return mean_value.expand_as(value)`,
  },
  {
    label: 'Scale × 0.5',
    code: `def hook_fn(value, hook):\n    return value * 0.5`,
  },
  {
    label: 'Log (inspect)',
    code: `def hook_fn(value, hook):\n    print(f"Shape: {value.shape}, Mean: {value.mean()}")\n    return value`,
  },
]

interface ActiveHook {
  id: string
  hook_name: string
  code: string
}

interface PredEntry {
  token: string
  token_id: number
  prob: number
}

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

const label: React.CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.12em',
  color: 'rgba(255,255,255,0.4)',
  marginBottom: 6,
  display: 'block',
  textTransform: 'uppercase',
}

const input: React.CSSProperties = {
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

export default function HookLab() {
  const [hookCode, setHookCode] = useState(TEMPLATES[0].code)
  const [hookName, setHookName] = useState('blocks.5.attn.hook_z')
  const [activeHooks, setActiveHooks] = useState<ActiveHook[]>([])
  const [prompt, setPrompt] = useState('The Eiffel Tower is in')
  const [baseline, setBaseline] = useState<PredEntry[]>([])
  const [modified, setModified] = useState<PredEntry[]>([])
  const [strTokens, setStrTokens] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    axios.get(`${API}/api/hooks`).then(r => setActiveHooks(r.data.hooks)).catch(() => {})
  }, [])

  const handleTemplateChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const t = TEMPLATES.find(t => t.label === e.target.value)
    if (t) setHookCode(t.code)
  }

  const addHook = async () => {
    setError('')
    try {
      const r = await axios.post(`${API}/api/hooks/add`, {
        hook_name_filter: hookName,
        hook_code: hookCode,
      })
      const newHook: ActiveHook = { id: r.data.id, hook_name: hookName, code: hookCode }
      setActiveHooks(prev => [...prev, newHook])
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      setError(msg)
    }
  }

  const removeHook = async (id: string) => {
    try {
      await axios.delete(`${API}/api/hooks/${id}`)
      setActiveHooks(prev => prev.filter(h => h.id !== id))
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      setError(msg)
    }
  }

  const preview = async () => {
    setError('')
    setLoading(true)
    try {
      const r = await axios.post(`${API}/api/hooks/preview`, { prompt, top_k: 5 })
      setBaseline(r.data.baseline)
      setModified(r.data.modified)
      setStrTokens(r.data.str_tokens)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden', fontFamily: '"JetBrains Mono", monospace', color: '#e0e0e0' }}>

      {/* ── LEFT: editor + hook controls ─────────────────────────────── */}
      <div style={{ width: '42%', display: 'flex', flexDirection: 'column', padding: 20, gap: 14, overflowY: 'auto', borderRight: '1px solid rgba(255,255,255,0.07)' }}>

        {/* header */}
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#00d4ff', letterSpacing: '0.05em' }}>Hook Lab</div>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', marginTop: 4 }}>
            Write Python hook functions and apply them to any hook point.
          </div>
        </div>

        {/* template picker */}
        <div style={panel}>
          <span style={label}>Template</span>
          <select
            onChange={handleTemplateChange}
            style={{ ...input, cursor: 'pointer' }}
          >
            {TEMPLATES.map(t => <option key={t.label} value={t.label}>{t.label}</option>)}
          </select>
        </div>

        {/* code editor */}
        <div style={panel}>
          <span style={label}>Hook Function (Python)</span>
          <textarea
            value={hookCode}
            onChange={e => setHookCode(e.target.value)}
            rows={10}
            spellCheck={false}
            style={{
              ...input,
              color: '#00d4ff',
              resize: 'vertical',
              lineHeight: 1.6,
              fontSize: 12,
            }}
          />
          <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.25)', marginTop: 6 }}>
            Must define <span style={{ color: '#a855f7' }}>hook_fn(value, hook)</span>. Import-free. Only torch available.
          </div>
        </div>

        {/* hook point filter */}
        <div style={panel}>
          <span style={label}>Hook Point</span>
          <input
            type="text"
            value={hookName}
            onChange={e => setHookName(e.target.value)}
            placeholder="blocks.5.attn.hook_z"
            style={input}
          />
          <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.25)', marginTop: 6 }}>
            Exact hook name (e.g. blocks.5.attn.hook_z, hook_embed, blocks.0.mlp.hook_post)
          </div>
        </div>

        {/* add button */}
        <button onClick={addHook} style={{ ...btn(), padding: '8px 0', fontSize: 12 }}>
          ⊕ Add Hook
        </button>

        {/* active hooks */}
        <div style={panel}>
          <span style={label}>Active Hooks ({activeHooks.length})</span>
          {activeHooks.length === 0 && (
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.2)' }}>No hooks registered.</div>
          )}
          {activeHooks.map(h => (
            <div key={h.id} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              padding: '8px 10px',
              marginBottom: 6,
              background: 'rgba(168,85,247,0.06)',
              border: '1px solid rgba(168,85,247,0.15)',
              borderRadius: 6,
            }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 11, color: '#a855f7', marginBottom: 2 }}>{h.hook_name}</div>
                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', whiteSpace: 'pre', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {h.code.split('\n')[0]}
                </div>
              </div>
              <button
                onClick={() => removeHook(h.id)}
                style={{ ...btn('#ff6b6b'), padding: '2px 8px', marginLeft: 8, fontSize: 11, flexShrink: 0 }}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        {error && (
          <div style={{ fontSize: 11, color: '#ff6b6b', background: 'rgba(255,107,107,0.08)', border: '1px solid rgba(255,107,107,0.2)', borderRadius: 6, padding: '8px 12px' }}>
            {error}
          </div>
        )}
      </div>

      {/* ── RIGHT: preview ───────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 20, gap: 16, overflowY: 'auto' }}>

        <div style={{ fontSize: 13, fontWeight: 600, color: 'rgba(255,255,255,0.7)', letterSpacing: '0.05em' }}>
          Before / After Preview
        </div>

        {/* prompt row */}
        <div style={{ display: 'flex', gap: 10 }}>
          <input
            type="text"
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Enter prompt…"
            style={{ ...input, flex: 1 }}
          />
          <button onClick={preview} disabled={loading} style={{ ...btn(), flexShrink: 0, opacity: loading ? 0.5 : 1 }}>
            {loading ? '…' : '▶ Preview'}
          </button>
        </div>

        {/* tokens strip */}
        {strTokens.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {strTokens.map((t, i) => (
              <span key={i} style={{
                fontSize: 11,
                padding: '2px 6px',
                background: 'rgba(0,212,255,0.08)',
                border: '1px solid rgba(0,212,255,0.2)',
                borderRadius: 4,
                color: '#00d4ff',
              }}>
                {t}
              </span>
            ))}
          </div>
        )}

        {/* comparison columns */}
        <div style={{ display: 'flex', gap: 16, flex: 1 }}>
          <PredColumn title="Baseline" entries={baseline} color="#00d4ff" />
          <PredColumn title={`Modified (${activeHooks.length} hook${activeHooks.length !== 1 ? 's' : ''})`} entries={modified} color="#a855f7" />
        </div>

        {/* diff hint */}
        {baseline.length > 0 && modified.length > 0 && (
          <DiffSummary baseline={baseline} modified={modified} />
        )}
      </div>
    </div>
  )
}

function PredColumn({ title, entries, color }: { title: string; entries: PredEntry[]; color: string }) {
  return (
    <div style={{ flex: 1, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: 16 }}>
      <div style={{ fontSize: 11, color, marginBottom: 12, fontWeight: 600, letterSpacing: '0.08em' }}>{title}</div>
      {entries.length === 0 ? (
        <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.2)' }}>Run preview to see predictions.</div>
      ) : entries.map((e, i) => (
        <div key={i} style={{ marginBottom: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
            <span style={{ fontSize: 12, color: '#e0e0e0' }}>{JSON.stringify(e.token)}</span>
            <span style={{ fontSize: 11, color }}>{(e.prob * 100).toFixed(2)}%</span>
          </div>
          <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ width: `${e.prob * 100}%`, height: '100%', background: color, borderRadius: 2, transition: 'width 0.4s' }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function DiffSummary({ baseline, modified }: { baseline: PredEntry[]; modified: PredEntry[] }) {
  const baseTop = baseline[0]
  const modTop = modified[0]
  const changed = baseTop?.token !== modTop?.token
  return (
    <div style={{
      background: changed ? 'rgba(255,107,107,0.07)' : 'rgba(74,222,128,0.07)',
      border: `1px solid ${changed ? 'rgba(255,107,107,0.2)' : 'rgba(74,222,128,0.2)'}`,
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 11,
      color: changed ? '#ff6b6b' : '#4ade80',
    }}>
      {changed
        ? `Top prediction changed: ${JSON.stringify(baseTop?.token)} → ${JSON.stringify(modTop?.token)}`
        : `Top prediction unchanged: ${JSON.stringify(baseTop?.token)} (hook had no effect on this output)`}
    </div>
  )
}
