import React, { useState } from 'react'
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

type PageId =
  | 'model-hub'
  | 'token-inspector'
  | 'forward-pass'
  | 'activation-browser'
  | 'attention-viz'
  | 'logit-lens'
  | 'attribution'
  | 'patching-lab'
  | 'circuit-analyzer'
  | 'hook-lab'
  | 'generation-studio'

interface NavItem {
  id: PageId
  label: string
  phase: number
  icon: string
  available: boolean
}

const NAV: NavItem[] = [
  { id: 'model-hub',          label: 'Model Hub',        phase: 1,  icon: '⬡', available: true  },
  { id: 'token-inspector',    label: 'Token Inspector',  phase: 2,  icon: '◈', available: true  },
  { id: 'forward-pass',       label: 'Forward Pass',     phase: 2,  icon: '▶', available: true  },
  { id: 'activation-browser', label: 'Activations',      phase: 3,  icon: '⊞', available: true  },
  { id: 'attention-viz',      label: 'Attention',        phase: 4,  icon: '◎', available: true  },
  { id: 'logit-lens',         label: 'Logit Lens',       phase: 5,  icon: '◉', available: true  },
  { id: 'attribution',        label: 'Attribution',      phase: 6,  icon: '⊛', available: true  },
  { id: 'patching-lab',       label: 'Patching Lab',     phase: 7,  icon: '⊗', available: true  },
  { id: 'circuit-analyzer',   label: 'Circuit Analyzer', phase: 8,  icon: '⬡', available: true  },
  { id: 'hook-lab',           label: 'Hook Lab',         phase: 9,  icon: '⊕', available: true  },
  { id: 'generation-studio',  label: 'Generation',       phase: 10, icon: '▷', available: true  },
]

const PAGE_COMPONENTS: Record<PageId, React.ComponentType> = {
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
}

export default function App() {
  const [page, setPage] = useState<PageId>('model-hub')
  const [mounted, setMounted] = useState<Set<PageId>>(() => new Set(['model-hub']))

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
          const active = page === item.id
          return (
            <button
              key={item.id}
              onClick={() => {
                if (!item.available) return
                setMounted(prev => new Set([...prev, item.id]))
                setPage(item.id)
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '8px 16px',
                background: active ? 'rgba(0,212,255,0.06)' : 'transparent',
                borderLeft: active ? '2px solid #00d4ff' : '2px solid transparent',
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

      {/* Main content */}
      <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {NAV.map(item => {
          if (!mounted.has(item.id)) return null
          const Page = PAGE_COMPONENTS[item.id]
          return (
            <div
              key={item.id}
              style={{
                flex: 1,
                overflow: 'hidden',
                display: page === item.id ? 'flex' : 'none',
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
