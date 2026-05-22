import { useSessionStore, type PageId } from '../../store/sessionStore'

export interface NextStep {
  page: PageId
  label: string
  hint: string
  badge?: string
}

interface Props {
  steps: NextStep[]
}

export default function NextSteps({ steps }: Props) {
  const setActivePage = useSessionStore((s) => s.setActivePage)
  if (steps.length === 0) return null

  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: 8,
      padding: '10px 20px',
      borderTop: '1px solid rgba(255,255,255,0.05)',
      background: 'rgba(168,85,247,0.03)',
      flexShrink: 0,
      flexWrap: 'wrap',
    }}>
      <span style={{
        fontSize: 9,
        color: '#a855f7',
        fontFamily: 'JetBrains Mono, monospace',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        alignSelf: 'center',
        flexShrink: 0,
        marginRight: 4,
      }}>
        Next →
      </span>
      {steps.map((step) => (
        <button
          key={step.page + step.label}
          onClick={() => setActivePage(step.page)}
          title={step.hint}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            padding: '4px 10px',
            background: 'rgba(168,85,247,0.08)',
            border: '1px solid rgba(168,85,247,0.3)',
            borderRadius: 6,
            color: '#c084fc',
            fontSize: 10,
            fontFamily: 'JetBrains Mono, monospace',
            cursor: 'pointer',
            transition: 'all 0.15s',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(168,85,247,0.18)'
            ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(168,85,247,0.6)'
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(168,85,247,0.08)'
            ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(168,85,247,0.3)'
          }}
        >
          {step.badge && (
            <span style={{
              fontSize: 8,
              padding: '1px 4px',
              background: 'rgba(168,85,247,0.2)',
              borderRadius: 3,
              color: '#a855f7',
            }}>
              {step.badge}
            </span>
          )}
          {step.label}
        </button>
      ))}
    </div>
  )
}
