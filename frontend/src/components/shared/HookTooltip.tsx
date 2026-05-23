import { useState } from 'react'
import { resolveHookDef } from '../../data/hookDefs'

interface Props {
  hookKey: string
  children: React.ReactNode
}

export default function HookTooltip({ hookKey, children }: Props) {
  const [visible, setVisible] = useState(false)
  const def = resolveHookDef(hookKey)
  if (!def) return <>{children}</>

  return (
    <span
      style={{ position: 'relative', display: 'inline-block', cursor: 'help' }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <span style={{
          position: 'absolute',
          bottom: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginBottom: 6,
          background: '#1a1a2e',
          border: '1px solid rgba(0,212,255,0.25)',
          borderRadius: 6,
          padding: '6px 10px',
          fontSize: 10,
          fontFamily: 'JetBrains Mono, monospace',
          color: 'rgba(255,255,255,0.8)',
          whiteSpace: 'nowrap',
          maxWidth: 280,
          whiteSpaceCollapse: 'preserve',
          zIndex: 999,
          pointerEvents: 'none',
          boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
        }}>
          <span style={{ color: '#00d4ff', display: 'block', marginBottom: 3, fontSize: 9, letterSpacing: '0.05em' }}>
            HOOK
          </span>
          {def}
        </span>
      )}
    </span>
  )
}
