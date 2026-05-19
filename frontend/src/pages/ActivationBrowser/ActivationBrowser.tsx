import React, { useState, useCallback } from 'react';
import HookTree from './HookTree';
import TensorViewer from './TensorViewer';
import type { ActivationData } from './TensorViewer';

const ActivationBrowser: React.FC = () => {
  const [text, setText] = useState('The Eiffel Tower is in');
  const [keys, setKeys] = useState<{ key: string; shape: number[] }[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [activationData, setActivationData] = useState<ActivationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [tensorLoading, setTensorLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runWithCache = useCallback(async () => {
    setLoading(true);
    setError(null);
    setActivationData(null);
    setSelectedKey(null);
    try {
      const res = await fetch('/api/inference/run_with_cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      const data = await res.json();
      setKeys(data.keys);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [text]);

  const selectKey = useCallback(async (key: string) => {
    setSelectedKey(key);
    setTensorLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/activations/${encodeURIComponent(key)}`);
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      const data = await res.json();
      setActivationData(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setTensorLoading(false);
    }
  }, []);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#0a0a0f', color: 'white' }}>
      {/* Top bar */}
      <div style={{ padding: 16, display: 'flex', gap: 12, alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.08)', flexShrink: 0 }}>
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
            <HookTree keys={keys} selectedKey={selectedKey} onSelect={selectKey} />
          )}
        </div>

        {/* Right: tensor viewer */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <TensorViewer activationData={activationData} loading={tensorLoading} />
        </div>
      </div>
    </div>
  );
};

export default ActivationBrowser;
