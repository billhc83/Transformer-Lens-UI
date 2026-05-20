import { useEffect, useState, useRef } from 'react';

export interface InterpretationGuide {
  overview: string;
  example: {
    prompt: string;
    output: string;
    interpretation: string;
  };
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  pageTitle: string;
  pageType: string;
  guide: InterpretationGuide;
  liveData?: Record<string, unknown> | null;
}

const BACKDROP: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(0,0,0,0.72)',
  zIndex: 1000,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 24,
};

const CARD: React.CSSProperties = {
  background: '#0d0d14',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 14,
  width: '100%',
  maxWidth: 680,
  maxHeight: '85vh',
  overflow: 'auto',
  padding: '28px 32px',
  position: 'relative',
  boxShadow: '0 8px 48px rgba(0,212,255,0.08)',
};

const CLOSE_BTN: React.CSSProperties = {
  position: 'absolute',
  top: 16,
  right: 16,
  background: 'transparent',
  border: '1px solid rgba(255,255,255,0.12)',
  color: 'rgba(255,255,255,0.5)',
  borderRadius: 6,
  width: 28,
  height: 28,
  cursor: 'pointer',
  fontSize: 14,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  lineHeight: 1,
};

const SECTION_LABEL: React.CSSProperties = {
  fontSize: 10,
  fontFamily: 'var(--font-mono, monospace)',
  letterSpacing: '0.12em',
  textTransform: 'uppercase',
  color: '#00d4ff',
  marginBottom: 8,
  marginTop: 24,
};

const BODY_TEXT: React.CSSProperties = {
  fontSize: 13,
  lineHeight: 1.7,
  color: 'rgba(255,255,255,0.75)',
  fontFamily: 'var(--font-mono, monospace)',
  margin: 0,
};

const CODE_BLOCK: React.CSSProperties = {
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 8,
  padding: '10px 14px',
  fontSize: 12,
  fontFamily: 'var(--font-mono, monospace)',
  color: 'rgba(255,255,255,0.65)',
  whiteSpace: 'pre-wrap',
  lineHeight: 1.6,
  margin: 0,
};

export default function InterpretationModal({
  isOpen, onClose, pageTitle, pageType, guide, liveData,
}: Props) {
  const [insight, setInsight] = useState('');
  const [insightLoading, setInsightLoading] = useState(false);
  const [insightError, setInsightError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const prevDataStr = useRef<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  // Reset insight when modal closes or when liveData content changes (user regenerated data)
  useEffect(() => {
    if (!isOpen) {
      abortRef.current?.abort();
      setInsight('');
      setInsightError(null);
      setInsightLoading(false);
      prevDataStr.current = null;
      return;
    }
    const str = liveData ? JSON.stringify(liveData) : null;
    if (prevDataStr.current !== null && prevDataStr.current !== str) {
      abortRef.current?.abort();
      setInsight('');
      setInsightError(null);
      setInsightLoading(false);
    }
    prevDataStr.current = str;
  }, [isOpen, liveData]);

  async function getInsight() {
    if (!liveData || insightLoading) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setInsight('');
    setInsightError(null);
    setInsightLoading(true);
    try {
      const res = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page_type: pageType, data: liveData }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        setInsightError(`Server error ${res.status} — is Ollama running?`);
        return;
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        setInsight(prev => prev + decoder.decode(value, { stream: true }));
      }
    } catch (e: unknown) {
      if ((e as Error).name !== 'AbortError') {
        setInsightError('Request failed — is Ollama running with qwen3:14b loaded?');
      }
    } finally {
      setInsightLoading(false);
    }
  }

  if (!isOpen) return null;

  const hasData = !!liveData;

  return (
    <div style={BACKDROP} onClick={onClose}>
      <div style={CARD} onClick={e => e.stopPropagation()}>
        <button style={CLOSE_BTN} onClick={onClose}>×</button>

        <div style={{ fontSize: 11, color: '#a855f7', fontFamily: 'var(--font-mono, monospace)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>
          How to read this
        </div>
        <h2 style={{ margin: 0, fontSize: 18, color: '#ffffff', fontFamily: 'var(--font-mono, monospace)', fontWeight: 600 }}>
          {pageTitle}
        </h2>

        <div style={{ ...SECTION_LABEL, marginTop: 20 }}>Overview</div>
        <p style={BODY_TEXT}>{guide.overview}</p>

        <div style={SECTION_LABEL}>Example Interpretation</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', fontFamily: 'var(--font-mono, monospace)', marginBottom: 4, letterSpacing: '0.08em' }}>PROMPT</div>
            <pre style={CODE_BLOCK}>{guide.example.prompt}</pre>
          </div>
          <div>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', fontFamily: 'var(--font-mono, monospace)', marginBottom: 4, letterSpacing: '0.08em' }}>OUTPUT</div>
            <pre style={CODE_BLOCK}>{guide.example.output}</pre>
          </div>
          <div>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', fontFamily: 'var(--font-mono, monospace)', marginBottom: 4, letterSpacing: '0.08em' }}>INTERPRETATION</div>
            <pre style={{ ...CODE_BLOCK, color: '#00d4ff', borderColor: 'rgba(0,212,255,0.15)', background: 'rgba(0,212,255,0.04)' }}>{guide.example.interpretation}</pre>
          </div>
        </div>

        {/* AI Insight divider */}
        <div style={{ margin: '28px 0 0', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono, monospace)', letterSpacing: '0.12em', textTransform: 'uppercase', color: '#a855f7' }}>
              AI Insight
            </span>
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.25)', fontFamily: 'var(--font-mono, monospace)' }}>
              qwen3:14b · current data
            </span>
            {!hasData && (
              <span style={{ fontSize: 10, color: 'rgba(255,107,107,0.7)', fontFamily: 'var(--font-mono, monospace)' }}>
                — run the page first
              </span>
            )}
          </div>

          <button
            onClick={getInsight}
            disabled={!hasData || insightLoading}
            style={{
              fontSize: 11,
              padding: '6px 16px',
              borderRadius: 6,
              border: `1px solid ${hasData ? 'rgba(168,85,247,0.5)' : 'rgba(255,255,255,0.1)'}`,
              background: insightLoading ? 'rgba(168,85,247,0.15)' : 'transparent',
              color: hasData ? '#a855f7' : 'rgba(255,255,255,0.2)',
              cursor: hasData && !insightLoading ? 'pointer' : 'not-allowed',
              fontFamily: 'var(--font-mono, monospace)',
              letterSpacing: '0.04em',
              transition: 'all 0.15s',
            }}
          >
            {insightLoading ? '⟳ Generating...' : insight ? '↺ Regenerate insight' : '✦ Get AI insight'}
          </button>

          {insightError && (
            <div style={{ marginTop: 10, fontSize: 11, color: '#ff6b6b', fontFamily: 'var(--font-mono, monospace)' }}>
              {insightError}
            </div>
          )}

          {(insight || insightLoading) && (
            <div style={{
              marginTop: 12,
              background: 'rgba(168,85,247,0.04)',
              border: '1px solid rgba(168,85,247,0.15)',
              borderRadius: 8,
              padding: '12px 16px',
              fontSize: 12,
              fontFamily: 'var(--font-mono, monospace)',
              color: 'rgba(255,255,255,0.8)',
              lineHeight: 1.7,
              whiteSpace: 'pre-wrap',
              minHeight: 48,
            }}>
              {insight}
              {insightLoading && (
                <span style={{ display: 'inline-block', width: 8, height: 12, background: '#a855f7', marginLeft: 2, animation: 'blink 1s step-end infinite', verticalAlign: 'text-bottom' }} />
              )}
            </div>
          )}
        </div>

        <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
      </div>
    </div>
  );
}
