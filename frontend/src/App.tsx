import ModelHub from './pages/ModelHub/ModelHub'
import TokenInspector from './pages/TokenInspector/TokenInspector'
import ForwardPass from './pages/ForwardPass/ForwardPass'
import ActivationBrowser from './pages/ActivationBrowser/ActivationBrowser'
import AttentionViz from './pages/AttentionViz/AttentionViz'
import LogitLens from './pages/LogitLens/LogitLens'
import Attribution from './pages/Attribution/Attribution'
import PatchingLab from './pages/PatchingLab/PatchingLab'
import CircuitAnalyzer from './pages/CircuitAnalyzer/CircuitAnalyzer'
import HookLab from './pages/HookLab/HookLab'
import GenerationStudio from './pages/GenerationStudio/GenerationStudio'
import ReportStudio from './pages/ReportStudio/ReportStudio'
import SAEStudio from './pages/SAEStudio/SAEStudio'
import NormalizationProbe from './pages/NormalizationProbe/NormalizationProbe'
import { useSessionStore, type PageId } from './store/sessionStore'

interface NavItem {
  id: PageId
  label: string
  phase: number
  icon: string
  available: boolean
  layerPage?: boolean
}

const NAV: NavItem[] = [
  { id: 'model-hub',          label: 'Model Hub',        phase: 1,  icon: '⬡', available: true  },
  { id: 'token-inspector',    label: 'Token Inspector',  phase: 2,  icon: '◈', available: true  },
  { id: 'forward-pass',       label: 'Forward Pass',     phase: 2,  icon: '▶', available: true  },
  { id: 'activation-browser', label: 'Activations',      phase: 3,  icon: '⊞', available: true, layerPage: true },
  { id: 'attention-viz',      label: 'Attention',        phase: 4,  icon: '◎', available: true, layerPage: true },
  { id: 'logit-lens',         label: 'Logit Lens',       phase: 5,  icon: '◉', available: true  },
  { id: 'attribution',        label: 'Attribution',      phase: 6,  icon: '⊛', available: true  },
  { id: 'patching-lab',       label: 'Patching Lab',     phase: 7,  icon: '⊗', available: true  },
  { id: 'circuit-analyzer',   label: 'Circuit Analyzer', phase: 8,  icon: '⬡', available: true  },
  { id: 'hook-lab',           label: 'Hook Lab',         phase: 9,  icon: '⊕', available: true  },
  { id: 'generation-studio',  label: 'Generation',       phase: 10, icon: '▷', available: true  },
  { id: 'report-studio',      label: 'Report Studio',    phase: 11, icon: '◈', available: true  },
  { id: 'sae-studio',          label: 'SAE Features',     phase: 12, icon: '⬡', available: true  },
  { id: 'normalization-probe', label: 'Norm Probe',       phase: 13, icon: '⚗', available: true  },
]

type ComponentMap = Partial<Record<PageId, React.ComponentType>>

const PAGE_COMPONENTS: ComponentMap = {
  'model-hub': ModelHub,
  'token-inspector': TokenInspector,
  'forward-pass': ForwardPass,
  'activation-browser': ActivationBrowser,
  'attention-viz': AttentionViz,
  'logit-lens': LogitLens,
  'attribution': Attribution,
  'patching-lab': PatchingLab,
  'circuit-analyzer': CircuitAnalyzer,
  'hook-lab': HookLab,
  'generation-studio': GenerationStudio,
  'report-studio': ReportStudio,
  'sae-studio': SAEStudio,
  'normalization-probe': NormalizationProbe,
}

export default function App() {
  const activePage = useSessionStore((s) => s.activePage)
  const navigateTo = useSessionStore((s) => s.navigateTo)
  const mountedPages = useSessionStore((s) => s.mountedPages)
  const visitedPages = useSessionStore((s) => s.visitedPages)
  const exploredLayers = useSessionStore((s) => s.exploredLayers)
  const findings = useSessionStore((s) => s.findings)

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: '#0a0a0f' }}>
      {/* Sidebar */}
      <nav style={{
        display: 'flex',
        flexDirection: 'column',
        width: '200px',
        flexShrink: 0,
        borderRight: '1px solid rgba(255,255,255,0.07)',
        background: 'rgba(255,255,255,0.015)',
        padding: '16px 0',
        gap: '2px',
        overflowY: 'auto',
      }}>
        {/* Logo */}
        <div style={{
          padding: '0 16px 16px',
          marginBottom: '8px',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
        }}>
          <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.15em', color: '#00d4ff' }}>TRANSFORMER</div>
          <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.15em', color: '#a855f7' }}>LENS  UI</div>
        </div>

        {NAV.map((item) => {
          const active = activePage === item.id
          const visited = visitedPages.includes(item.id)
          const layerCount = item.layerPage ? (exploredLayers[item.id] ?? []).length : 0
          const findingCount = findings.filter((f) => f.page === item.id).length
          const isReport = item.id === 'report-studio'

          return (
            <button
              key={item.id}
              onClick={() => item.available && navigateTo(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '8px 16px',
                background: active ? 'rgba(0,212,255,0.06)' : 'transparent',
                borderLeft: active
                  ? '2px solid #00d4ff'
                  : isReport && findings.length > 0
                    ? '2px solid rgba(168,85,247,0.5)'
                    : '2px solid transparent',
                color: !item.available ? 'rgba(255,255,255,0.2)' : active ? '#00d4ff' : 'rgba(255,255,255,0.65)',
                fontSize: '11px',
                cursor: item.available ? 'pointer' : 'default',
                border: 'none',
                textAlign: 'left',
                transition: 'all 0.15s',
                fontFamily: 'inherit',
              }}
            >
              <span style={{ width: '14px', textAlign: 'center', fontSize: '13px' }}>{item.icon}</span>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {item.label}
              </span>

              {/* Layer coverage badge */}
              {item.layerPage && layerCount > 0 && (
                <span style={{
                  fontSize: '8px',
                  padding: '1px 4px',
                  borderRadius: '3px',
                  background: 'rgba(0,212,255,0.1)',
                  color: 'rgba(0,212,255,0.6)',
                  flexShrink: 0,
                }}>
                  L×{layerCount}
                </span>
              )}

              {/* Finding count badge */}
              {findingCount > 0 && !isReport && (
                <span style={{
                  fontSize: '8px',
                  padding: '1px 4px',
                  borderRadius: '3px',
                  background: 'rgba(168,85,247,0.12)',
                  color: '#a855f7',
                  flexShrink: 0,
                }}>
                  {findingCount}
                </span>
              )}

              {/* Report Studio total findings pill */}
              {isReport && findings.length > 0 && (
                <span style={{
                  fontSize: '8px',
                  padding: '1px 5px',
                  borderRadius: '3px',
                  background: 'rgba(168,85,247,0.2)',
                  color: '#a855f7',
                  flexShrink: 0,
                }}>
                  {findings.length}
                </span>
              )}

              {/* Visited dot */}
              {visited && !active && !findingCount && !layerCount && !isReport && (
                <span style={{
                  width: 5,
                  height: 5,
                  borderRadius: '50%',
                  background: 'rgba(0,212,255,0.35)',
                  flexShrink: 0,
                  display: 'inline-block',
                }} />
              )}

              {!item.available && (
                <span style={{
                  fontSize: '8px',
                  padding: '1px 4px',
                  borderRadius: '3px',
                  background: 'rgba(168,85,247,0.12)',
                  color: '#a855f7',
                  flexShrink: 0,
                }}>
                  P{item.phase}
                </span>
              )}
            </button>
          )
        })}
      </nav>

      {/* Main content — keep all mounted pages in DOM to preserve state */}
      <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {NAV.map((item) => {
          if (!mountedPages.includes(item.id)) return null
          const Page = PAGE_COMPONENTS[item.id]
          if (!Page) return null
          return (
            <div
              key={item.id}
              style={{
                flex: 1,
                overflow: 'hidden',
                display: activePage === item.id ? 'flex' : 'none',
                flexDirection: 'column',
              }}
            >
              <Page />
            </div>
          )
        })}
      </main>
    </div>
  )
}
