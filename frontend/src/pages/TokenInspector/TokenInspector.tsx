import { useState } from 'react'
import InterpretationModal, { type InterpretationGuide } from '../../components/shared/InterpretationModal'

const CHIP_COLORS = ['#00d4ff', '#a855f7', '#14b8a6', '#ec4899', '#f97316']

const GUIDE: InterpretationGuide = {
  overview:
    'Token Inspector converts raw text into the discrete tokens a transformer actually processes. ' +
    'Each coloured chip represents one token — the atomic unit the model reads. ' +
    'Hover a chip to see its numeric token ID and sequence position. ' +
    'Token boundaries are model-specific (GPT-2 uses BPE): common words are single tokens, rare words split into sub-words. ' +
    'A leading space character (· ) is part of the token — " Paris" (ID 6342) and "Paris" (ID 40313) are different tokens with different model representations.',
  example: {
    prompt: '"The Eiffel Tower is in"',
    output:
      '6 tokens\n' +
      'Pos 0 → "The"     (ID 464)\n' +
      'Pos 1 → " Eiffel" (ID 15389)\n' +
      'Pos 2 → " Tower"  (ID 10580)\n' +
      'Pos 3 → " is"     (ID 318)\n' +
      'Pos 4 → " in"     (ID 287)',
    interpretation:
      '"Eiffel" and "Tower" each get their own token — common enough in GPT-2\'s vocabulary.\n' +
      'Every content word has a leading space baked in (e.g. " Eiffel").\n' +
      'The model will predict the token at position 5, which should be " Paris" (ID 6342).\n' +
      'If a rare proper noun split into sub-words (e.g. "Ei", "ff", "el"), that would\n' +
      'hint at lower model familiarity and typically weaker next-token confidence.',
  },
}

const GUIDE_BTN: React.CSSProperties = {
  fontSize: 11,
  padding: '3px 10px',
  borderRadius: 6,
  border: '1px solid rgba(0,212,255,0.4)',
  background: 'transparent',
  color: '#00d4ff',
  cursor: 'pointer',
  fontFamily: 'inherit',
  letterSpacing: '0.04em',
}

interface TokenizeResponse {
  token_ids: number[]
  str_tokens: string[]
  n_tokens: number
}

export default function TokenInspector() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<TokenizeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)
  const [guideOpen, setGuideOpen] = useState(false)

  async function tokenize() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/inference/tokenize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (res.status === 400) {
        const data = await res.json()
        setError(data.detail ?? 'No model loaded')
        setResult(null)
        return
      }
      const data: TokenizeResponse = await res.json()
      setResult(data)
    } catch (e) {
      setError('Request failed — is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ flex: 1, overflow: 'auto', padding: '28px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '4px' }}>
          <div style={{ fontSize: '16px', fontWeight: 600, color: '#00d4ff' }}>Token Inspector</div>
          <button style={GUIDE_BTN} onClick={() => setGuideOpen(true)}>? How to read this</button>
        </div>
        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.35)' }}>Tokenize text and inspect token IDs</div>
      </div>

      {/* Input panel */}
      <div style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '12px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) tokenize() }}
          placeholder="Enter text to tokenize..."
          rows={3}
          style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '8px',
            padding: '10px 12px',
            color: 'rgba(255,255,255,0.85)',
            fontFamily: 'inherit',
            fontSize: '13px',
            resize: 'vertical',
            outline: 'none',
            width: '100%',
            boxSizing: 'border-box',
          }}
        />
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={tokenize}
            disabled={loading || !text.trim()}
            style={{
              background: loading ? 'rgba(0,212,255,0.3)' : '#00d4ff',
              border: 'none',
              borderRadius: '8px',
              padding: '8px 20px',
              color: '#0a0a0f',
              fontFamily: 'inherit',
              fontSize: '12px',
              fontWeight: 700,
              cursor: loading || !text.trim() ? 'not-allowed' : 'pointer',
              letterSpacing: '0.05em',
            }}
          >
            {loading ? 'Tokenizing...' : 'Tokenize'}
          </button>
          <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.25)' }}>⌘↵ to run</span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: 'rgba(255,107,107,0.08)',
          border: '1px solid rgba(255,107,107,0.3)',
          borderRadius: '8px',
          padding: '10px 14px',
          fontSize: '12px',
          color: '#ff6b6b',
        }}>
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
        }}>
          <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>
            {result.n_tokens} tokens
          </div>

          {/* Token chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'flex-start' }}>
            {result.token_ids.map((id, idx) => {
              const color = CHIP_COLORS[idx % CHIP_COLORS.length]
              const isHovered = hoveredIdx === idx
              return (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '3px',
                    position: 'relative',
                  }}
                >
                  <div style={{
                    background: `${color}22`,
                    border: `1px solid ${color}55`,
                    borderRadius: '6px',
                    padding: '4px 8px',
                    fontSize: '12px',
                    color: color,
                    fontFamily: 'inherit',
                    whiteSpace: 'pre',
                    cursor: 'default',
                    transition: 'all 0.1s',
                    ...(isHovered ? { background: `${color}33`, border: `1px solid ${color}88` } : {}),
                  }}>
                    {result.str_tokens[idx]}
                  </div>
                  <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.3)' }}>{idx}</div>

                  {/* Tooltip */}
                  {isHovered && (
                    <div style={{
                      position: 'absolute',
                      bottom: 'calc(100% + 6px)',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      background: 'rgba(10,10,15,0.95)',
                      border: '1px solid rgba(255,255,255,0.12)',
                      borderRadius: '6px',
                      padding: '5px 8px',
                      fontSize: '10px',
                      color: 'rgba(255,255,255,0.7)',
                      whiteSpace: 'nowrap',
                      zIndex: 10,
                      pointerEvents: 'none',
                    }}>
                      ID: {id} · pos: {idx}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* IDs row */}
          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.25)', fontFamily: 'inherit' }}>
            IDs: [{result.token_ids.join(', ')}]
          </div>
        </div>
      )}
      <InterpretationModal
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
        pageTitle="Token Inspector"
        pageType="token-inspector"
        guide={GUIDE}
        liveData={result as Record<string, unknown> | null}
      />
    </div>
  )
}
