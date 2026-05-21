import React, { useState, useCallback, useRef } from 'react';
import HookTree from './HookTree';
import TensorViewer from './TensorViewer';
import type { ActivationData } from './TensorViewer';
import InterpretationModal, { type InterpretationGuide } from '../../components/shared/InterpretationModal';

const GUIDE: InterpretationGuide = {
  overview:
    'Activation Browser exposes every internal tensor captured during a forward pass via TransformerLens\'s hook system. ' +
    'After clicking "Run Cache", a tree of 200+ hook points appears on the left. ' +
    'Hook names follow the pattern blocks.{layer}.{module}.{hook_name} — for example, ' +
    'blocks.9.hook_resid_post is the residual stream after all of layer 9\'s computations. ' +
    'Selecting a hook fetches its raw tensor and renders it as a colour heatmap (rows = sequence positions, columns = features). ' +
    'Bright cyan = strong positive activation; bright red = strong negative activation; near-black = zero.',
  example: {
    prompt: 'Run cache on "The Eiffel Tower is in", then select blocks.9.hook_resid_post',
    output:
      'Shape: [1, 5, 768]  (batch=1, seq=5, d_model=768)\n' +
      'Heatmap: 5 rows × 768 columns\n' +
      'Row 1 (" Eiffel") shows a distinctive pattern of bright columns around dim 200–250',
    interpretation:
      'blocks.9.hook_resid_post captures the residual stream just before the model unembeds to logits.\n' +
      'Bright columns shared across multiple token positions encode general syntactic features.\n' +
      'Columns that light up uniquely for " Eiffel" are features the model uses to encode\n' +
      '"this is a famous landmark" — the same features that steer the final prediction to " Paris".\n' +
      'Comparing this hook to blocks.0.hook_resid_post shows how representations evolve.',
  },
};

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
};

const REMOVE_BTN: React.CSSProperties = {
  fontSize: 14,
  lineHeight: 1,
  padding: '0 6px',
  borderRadius: 4,
  border: '1px solid rgba(255,255,255,0.15)',
  background: 'transparent',
  color: '#888',
  cursor: 'pointer',
  fontFamily: 'inherit',
};

const ActivationBrowser: React.FC = () => {
  const [text, setText] = useState('The Eiffel Tower is in');
  const [keys, setKeys] = useState<{ key: string; shape: number[] }[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [activationData, setActivationData] = useState<ActivationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [tensorLoading, setTensorLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [guideOpen, setGuideOpen] = useState(false);
  const [cacheMode, setCacheMode] = useState<'all' | 'essential'>('all');

  // Compare mode state
  const [compareMode, setCompareMode] = useState(false);
  const [pinnedKeys, setPinnedKeys] = useState<string[]>([]);
  const [pinnedData, setPinnedData] = useState<Record<string, ActivationData>>({});
  const [loadingKeys, setLoadingKeys] = useState<Set<string>>(new Set());
  // Ref to avoid stale closure in togglePinned
  const pinnedDataRef = useRef<Record<string, ActivationData>>({});

  const runWithCache = useCallback(async () => {
    setLoading(true);
    setError(null);
    setActivationData(null);
    setSelectedKey(null);
    try {
      const res = await fetch('/api/inference/run_with_cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, cache_mode: cacheMode }),
      });
      if (!res.ok) {
        let msg = res.statusText;
        try { const d = await res.json(); msg = d.detail ?? msg; } catch { msg = await res.text().catch(() => msg); }
        throw new Error(msg);
      }
      const data = await res.json();
      setKeys(data.keys);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [text, cacheMode]);

  const fetchActivation = useCallback(async (key: string): Promise<ActivationData | null> => {
    try {
      const res = await fetch(`/api/activations/${encodeURIComponent(key)}`);
      if (!res.ok) {
        let msg = res.statusText;
        try { const d = await res.json(); msg = d.detail ?? msg; } catch { msg = await res.text().catch(() => msg); }
        throw new Error(msg);
      }
      return await res.json();
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  }, []);

  const selectKey = useCallback(async (key: string) => {
    if (compareMode) {
      // Toggle pinned
      setPinnedKeys(prev => {
        if (prev.includes(key)) {
          // Remove
          setPinnedData(d => { const next = { ...d }; delete next[key]; pinnedDataRef.current = next; return next; });
          setLoadingKeys(s => { const n = new Set(s); n.delete(key); return n; });
          return prev.filter(k => k !== key);
        } else {
          // Add and fetch if not cached
          if (!pinnedDataRef.current[key]) {
            setLoadingKeys(s => new Set(s).add(key));
            fetchActivation(key).then(data => {
              if (data) {
                pinnedDataRef.current = { ...pinnedDataRef.current, [key]: data };
                setPinnedData(d => ({ ...d, [key]: data }));
              }
              setLoadingKeys(s => { const n = new Set(s); n.delete(key); return n; });
            });
          }
          return [...prev, key];
        }
      });
    } else {
      setSelectedKey(key);
      setTensorLoading(true);
      setError(null);
      const data = await fetchActivation(key);
      setActivationData(data);
      setTensorLoading(false);
    }
  }, [compareMode, fetchActivation]);

  const removePinned = useCallback((key: string) => {
    setPinnedKeys(prev => prev.filter(k => k !== key));
    setPinnedData(d => { const next = { ...d }; delete next[key]; pinnedDataRef.current = next; return next; });
  }, []);

  const toggleCompareMode = useCallback(() => {
    setCompareMode(prev => {
      if (prev) {
        // Turning off — clear pinned state
        setPinnedKeys([]);
        setPinnedData({});
        pinnedDataRef.current = {};
        setLoadingKeys(new Set());
      }
      return !prev;
    });
  }, []);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#0a0a0f', color: 'white' }}>
      {/* Top bar */}
      <div style={{ padding: 16, display: 'flex', gap: 12, alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.08)', flexShrink: 0 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#00d4ff', whiteSpace: 'nowrap' }}>Activation Browser</span>
        <button style={GUIDE_BTN} onClick={() => setGuideOpen(true)}>? How to read this</button>
        <button
          onClick={toggleCompareMode}
          style={{
            ...GUIDE_BTN,
            background: compareMode ? 'rgba(0,212,255,0.15)' : 'transparent',
            border: compareMode ? '1px solid #00d4ff' : '1px solid rgba(0,212,255,0.4)',
            fontWeight: compareMode ? 700 : 400,
            whiteSpace: 'nowrap',
          }}
        >
          {compareMode ? `⊞ Comparing (${pinnedKeys.length})` : '⊞ Compare'}
        </button>
        <input
          type="text"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && runWithCache()}
          placeholder="Enter prompt…"
          style={{
            flex: 1,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 8,
            color: 'white',
            padding: '8px 12px',
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 13,
          }}
        />
        <button
          onClick={runWithCache}
          disabled={loading}
          style={{
            background: 'linear-gradient(135deg, #00d4ff, #a855f7)',
            color: '#000',
            fontWeight: 700,
            padding: '8px 20px',
            borderRadius: 8,
            cursor: loading ? 'wait' : 'pointer',
            border: 'none',
            fontFamily: '"JetBrains Mono", monospace',
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? 'Running…' : 'Run Cache'}
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#888', fontFamily: '"JetBrains Mono", monospace', cursor: 'pointer', userSelect: 'none' }}>
          <input
            type="checkbox"
            checked={cacheMode === 'essential'}
            onChange={e => setCacheMode(e.target.checked ? 'essential' : 'all')}
            style={{ accentColor: '#a855f7' }}
          />
          Low VRAM mode
        </label>
        {keys.length > 0 && (
          <span style={{ color: '#666', fontSize: 12, fontFamily: '"JetBrains Mono", monospace' }}>
            {keys.length} activations
          </span>
        )}
      </div>

      {error && (
        <div style={{ color: '#ff6b6b', padding: '8px 16px', fontSize: 12, fontFamily: '"JetBrains Mono", monospace', borderBottom: '1px solid rgba(255,107,107,0.2)', flexShrink: 0 }}>
          {error}
        </div>
      )}

      {/* Main area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left: tree */}
        <div style={{ width: 300, borderRight: '1px solid rgba(255,255,255,0.08)', overflowY: 'auto', flexShrink: 0 }}>
          {keys.length === 0 ? (
            <div style={{ color: '#444', padding: 24, fontSize: 12, fontFamily: '"JetBrains Mono", monospace' }}>
              Run cache to see hook points.
            </div>
          ) : (
            <HookTree keys={keys} selectedKey={selectedKey} onSelect={selectKey} compareMode={compareMode} pinnedKeys={pinnedKeys} />
          )}
        </div>

        {/* Right: tensor viewer or compare panel */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {compareMode ? (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              {pinnedKeys.length === 0 ? (
                <div style={{ color: '#555', padding: 24, fontFamily: '"JetBrains Mono", monospace', fontSize: 13 }}>
                  Select hook points in the tree to compare.
                </div>
              ) : (
                pinnedKeys.map(key => (
                  <div key={key} style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                    <div style={{ padding: '6px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(168,85,247,0.08)', borderBottom: '1px solid rgba(168,85,247,0.15)', flexShrink: 0 }}>
                      <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 12, color: '#a855f7' }}>{key}</span>
                      <button onClick={() => removePinned(key)} style={REMOVE_BTN}>×</button>
                    </div>
                    <TensorViewer activationData={pinnedData[key] ?? null} loading={loadingKeys.has(key)} />
                  </div>
                ))
              )}
            </div>
          ) : (
            <TensorViewer activationData={activationData} loading={tensorLoading} />
          )}
        </div>
      </div>
      <InterpretationModal
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
        pageTitle="Activation Browser"
        pageType="activation-browser"
        guide={GUIDE}
        liveData={activationData ? {
          key: selectedKey,
          shape: activationData.shape,
          original_shape: activationData.original_shape,
          stats: activationData.stats,
          str_tokens: activationData.str_tokens,
        } : null}
      />
    </div>
  );
};

export default ActivationBrowser;
