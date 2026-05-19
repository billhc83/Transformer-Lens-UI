import { useState, useMemo } from 'react'
import useModels, { type ModelConfig } from '../../hooks/useModels'
import ArchitectureGraph from '../../components/viz/ArchitectureGraph'
import { useModelStore } from '../../store/modelStore'

const PANEL: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '12px',
  boxShadow: '0 4px 24px rgba(0,212,255,0.05)',
}

const LABEL_ROW: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '8px 12px',
  borderBottom: '1px solid rgba(255,255,255,0.05)',
  fontSize: '10px',
  color: 'rgba(255,255,255,0.4)',
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
}

function StatChip({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '2px',
      padding: '10px 14px',
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: '8px',
      minWidth: '80px',
    }}>
      <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.4)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ fontSize: '14px', color: '#00d4ff', fontWeight: 600 }}>{value}</span>
    </div>
  )
}

function ConfigCard({ config }: { config: ModelConfig }) {
  return (
    <div style={{ ...PANEL, padding: '0' }}>
      <div style={LABEL_ROW}>
        <span>architecture</span>
        <span style={{ color: '#4ade80' }}>{config.device}</span>
      </div>
      <div style={{ padding: '14px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        <StatChip label="d_model"   value={config.d_model} />
        <StatChip label="layers"    value={config.n_layers} />
        <StatChip label="heads"     value={config.n_heads} />
        <StatChip label="d_mlp"     value={config.d_mlp} />
        <StatChip label="n_ctx"     value={config.n_ctx} />
        <StatChip label="d_vocab"   value={config.d_vocab.toLocaleString()} />
        <StatChip label="act_fn"    value={config.act_fn} />
        <StatChip label="norm"      value={config.normalization_type} />
      </div>
    </div>
  )
}

export default function ModelHub() {
  const { models, loading, error, loadModel, loadingModel } = useModels()
  const { loadedConfig, setLoadedConfig } = useModelStore()

  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<'all' | 'local' | 'remote'>('all')
  const [selected, setSelected] = useState<string>('')
  const [loadError, setLoadError] = useState<string | null>(null)

  const filtered = useMemo(() => {
    return models.filter((m) => {
      const matchSearch = m.name.toLowerCase().includes(search.toLowerCase())
      const matchFilter =
        filter === 'all' ||
        (filter === 'local' && m.is_local) ||
        (filter === 'remote' && !m.is_local)
      return matchSearch && matchFilter
    })
  }, [models, search, filter])

  async function handleLoad() {
    if (!selected) return
    setLoadError(null)
    try {
      const cfg = await loadModel(selected)
      setLoadedConfig(cfg)
    } catch (e: unknown) {
      setLoadError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '24px', gap: '20px', overflow: 'auto' }}>
      {/* Header */}
      <div>
        <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 600, color: '#e2e8f0', letterSpacing: '-0.02em' }}>
          Model Hub
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>
          Load a TransformerLens model to begin analysis
        </p>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        {/* Search */}
        <input
          type="text"
          placeholder="Search models…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            flex: '1',
            minWidth: '200px',
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            padding: '8px 12px',
            color: '#e2e8f0',
            fontSize: '12px',
            fontFamily: 'inherit',
            outline: 'none',
          }}
        />

        {/* Filter tabs */}
        {(['all', 'local', 'remote'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              padding: '7px 14px',
              borderRadius: '8px',
              fontSize: '11px',
              fontFamily: 'inherit',
              cursor: 'pointer',
              border: '1px solid',
              transition: 'all 0.15s',
              background: filter === f ? 'rgba(0,212,255,0.1)' : 'transparent',
              borderColor: filter === f ? '#00d4ff' : 'rgba(255,255,255,0.12)',
              color: filter === f ? '#00d4ff' : 'rgba(255,255,255,0.5)',
            }}
          >
            {f === 'all' ? `All (${models.length})` : f === 'local' ? `Local (${models.filter(m => m.is_local).length})` : `Remote (${models.filter(m => !m.is_local).length})`}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '20px', flex: 1, minHeight: 0 }}>
        {/* Model list */}
        <div style={{ ...PANEL, width: '340px', flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={LABEL_ROW}>
            <span>models ({filtered.length})</span>
            {loading && <span style={{ color: '#00d4ff' }}>loading…</span>}
            {error && <span style={{ color: '#ff6b6b' }}>error</span>}
          </div>
          <div style={{ overflowY: 'auto', flex: 1 }}>
            {filtered.map((m) => {
              const active = selected === m.name
              return (
                <div
                  key={m.name}
                  onClick={() => setSelected(m.name)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '9px 12px',
                    cursor: 'pointer',
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    background: active ? 'rgba(0,212,255,0.06)' : 'transparent',
                    borderLeft: `2px solid ${active ? '#00d4ff' : 'transparent'}`,
                    transition: 'background 0.1s',
                  }}
                >
                  {/* LOCAL / DOWNLOAD badge */}
                  <span style={{
                    fontSize: '8px',
                    padding: '2px 5px',
                    borderRadius: '3px',
                    fontWeight: 600,
                    letterSpacing: '0.06em',
                    flexShrink: 0,
                    background: m.is_local ? 'rgba(74,222,128,0.12)' : 'rgba(168,85,247,0.12)',
                    color: m.is_local ? '#4ade80' : '#a855f7',
                    border: `1px solid ${m.is_local ? 'rgba(74,222,128,0.25)' : 'rgba(168,85,247,0.25)'}`,
                  }}>
                    {m.is_local ? 'LOCAL' : 'DL'}
                  </span>
                  <span style={{
                    fontSize: '11px',
                    color: active ? '#00d4ff' : '#e2e8f0',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {m.name}
                  </span>
                </div>
              )
            })}
            {!loading && filtered.length === 0 && (
              <div style={{ padding: '24px', textAlign: 'center', color: 'rgba(255,255,255,0.3)', fontSize: '11px' }}>
                No models match
              </div>
            )}
          </div>

          {/* Load button */}
          <div style={{ padding: '12px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            {loadError && (
              <div style={{ fontSize: '10px', color: '#ff6b6b', marginBottom: '8px', wordBreak: 'break-all' }}>
                {loadError}
              </div>
            )}
            <button
              onClick={handleLoad}
              disabled={!selected || loadingModel}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '8px',
                fontSize: '12px',
                fontFamily: 'inherit',
                cursor: selected && !loadingModel ? 'pointer' : 'default',
                border: '1px solid',
                transition: 'all 0.2s',
                background: selected && !loadingModel ? 'rgba(0,212,255,0.1)' : 'rgba(255,255,255,0.03)',
                borderColor: selected && !loadingModel ? '#00d4ff' : 'rgba(255,255,255,0.08)',
                color: selected && !loadingModel ? '#00d4ff' : 'rgba(255,255,255,0.25)',
                fontWeight: 500,
              }}
            >
              {loadingModel ? '⟳ Loading…' : selected ? `Load ${selected}` : 'Select a model'}
            </button>
          </div>
        </div>

        {/* Right panel */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', minWidth: 0 }}>
          {loadedConfig ? (
            <>
              {/* Config card */}
              <ConfigCard config={loadedConfig} />

              {/* Architecture graph */}
              <div style={{ ...PANEL, flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <div style={LABEL_ROW}>
                  <span>architecture graph — {loadedConfig.name}</span>
                  <span style={{ color: '#a855f7', fontSize: '9px' }}>embed → blocks → unembed</span>
                </div>
                <div style={{ flex: 1, position: 'relative' }}>
                  <ArchitectureGraph config={loadedConfig} />
                </div>
              </div>
            </>
          ) : (
            <div style={{
              ...PANEL,
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              gap: '12px',
            }}>
              <div style={{ fontSize: '40px', opacity: 0.12 }}>⬡</div>
              <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.3)' }}>
                Select and load a model to see its architecture
              </div>
              <div style={{ fontSize: '10px', color: 'rgba(0,212,255,0.4)' }}>
                Try <strong style={{ color: '#00d4ff' }}>gpt2</strong> — it's already in your local cache
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
