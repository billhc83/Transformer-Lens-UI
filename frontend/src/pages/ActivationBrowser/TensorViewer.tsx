import React, { useState } from 'react';

export interface ActivationData {
  key: string;
  shape: number[];
  original_shape: number[];
  dtype: string;
  stats: { min: number; max: number; mean: number; std: number };
  data: any;
  str_tokens: string[];
}

interface Props {
  activationData: ActivationData | null;
  loading: boolean;
}

function valueToColor(val: number, min: number, max: number): string {
  const t = max === min ? 0.5 : Math.max(0, Math.min(1, (val - min) / (max - min)));
  // blue (#0000ff) → cyan (#00d4ff) → white (#ffffff)
  if (t < 0.5) {
    const s = t * 2;
    const r = Math.round(0 + s * 0);
    const g = Math.round(0 + s * 212);
    const b = 255;
    return `rgb(${r},${g},${b})`;
  } else {
    const s = (t - 0.5) * 2;
    const r = Math.round(0 + s * 255);
    const g = Math.round(212 + s * (255 - 212));
    const b = Math.round(255 + s * (255 - 255));
    return `rgb(${r},${g},${b})`;
  }
}

const STAT_CHIP: React.CSSProperties = {
  display: 'inline-block', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 6, padding: '2px 10px', fontSize: 12, fontFamily: '"JetBrains Mono", monospace',
  color: '#c0c0c0', marginRight: 8,
};

const TensorViewer: React.FC<Props> = ({ activationData, loading }) => {
  const [sliceIdx, setSliceIdx] = useState(0);

  if (loading) return <div style={{ color: '#00d4ff', padding: 24 }}>Loading...</div>;
  if (!activationData) return (
    <div style={{ color: '#555', padding: 24, fontFamily: '"JetBrains Mono", monospace', fontSize: 13 }}>
      Select a hook point to inspect its tensor.
    </div>
  );

  const { shape, original_shape, stats, data, str_tokens } = activationData;
  const { min, max, mean, std } = stats;

  const DIM_NAMES = ['batch', 'pos', 'd_model', 'd_head', 'n_heads', 'd_mlp', 'd_vocab'];

  const shapeLabel = original_shape.map((d, i) => {
    const label = i === 1 && str_tokens.length ? `pos=${d}` : `${DIM_NAMES[i] ?? `d${i}`}=${d}`;
    return label;
  }).join(', ');

  // Resolve to 2D slice for heatmap
  const resolveSlice = (): number[][] | number[] | null => {
    if (!data) return null;
    if (shape.length === 1) return data as number[];
    if (shape.length === 2) return data as number[][];
    if (shape.length === 3) return data[Math.min(sliceIdx, shape[0] - 1)] as number[][];
    if (shape.length >= 4) {
      const s0 = Math.min(sliceIdx, shape[0] - 1);
      return data[s0][0] as number[][];
    }
    return null;
  };

  const sliceData = resolveSlice();

  return (
    <div style={{ padding: 16, height: '100%', overflowY: 'auto', boxSizing: 'border-box' }}>
      {/* Shape label */}
      <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 12, color: '#a855f7', marginBottom: 12 }}>
        [{shapeLabel}]
        <span style={{ color: '#555', marginLeft: 8 }}>({activationData.dtype})</span>
      </div>

      {/* Stats chips */}
      <div style={{ marginBottom: 16 }}>
        <span style={STAT_CHIP}>min {min.toFixed(3)}</span>
        <span style={STAT_CHIP}>max {max.toFixed(3)}</span>
        <span style={STAT_CHIP}>mean {mean.toFixed(3)}</span>
        <span style={STAT_CHIP}>std {std.toFixed(3)}</span>
      </div>

      {/* Slice selector for 3D+ */}
      {shape.length >= 3 && (
        <div style={{ marginBottom: 12, fontFamily: '"JetBrains Mono", monospace', fontSize: 12, color: '#c0c0c0' }}>
          Slice axis 0:{' '}
          <input
            type="number" min={0} max={shape[0] - 1} value={sliceIdx}
            onChange={e => setSliceIdx(Math.max(0, Math.min(shape[0] - 1, Number(e.target.value))))}
            style={{ width: 60, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 4, color: '#00d4ff', padding: '2px 6px', fontFamily: 'inherit' }}
          />
          <span style={{ color: '#555', marginLeft: 8 }}>of {shape[0]}</span>
        </div>
      )}

      {/* Visualization */}
      {shape.length === 1 && Array.isArray(sliceData) && (
        <div style={{ overflowY: 'auto', maxHeight: 400 }}>
          {(sliceData as number[]).map((val, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 2, fontFamily: '"JetBrains Mono", monospace', fontSize: 11 }}>
              <span style={{ color: '#666', width: 40, textAlign: 'right', marginRight: 8 }}>{i}</span>
              <div style={{ flex: 1, height: 12, background: 'rgba(255,255,255,0.04)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${Math.max(0, Math.min(1, (val - min) / (max - min || 1))) * 100}%`, height: '100%', background: '#00d4ff', borderRadius: 2 }} />
              </div>
              <span style={{ color: '#c0c0c0', width: 70, textAlign: 'right', marginLeft: 8 }}>{val.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}

      {shape.length >= 2 && Array.isArray(sliceData) && Array.isArray((sliceData as number[][])[0]) && (
        <div style={{ overflowX: 'auto' }}>
          <div style={{ display: 'flex', marginBottom: 2 }}>
            <div style={{ width: 60 }} />
            {/* No column labels — too many */}
          </div>
          {(sliceData as number[][]).map((row, ri) => (
            <div key={ri} style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 10, color: '#666', width: 60, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {str_tokens[ri] ?? ri}
              </span>
              {row.map((val, ci) => (
                <div
                  key={ci}
                  title={`[${ri},${ci}] = ${val.toFixed(4)}`}
                  style={{ width: 8, height: 8, flexShrink: 0, backgroundColor: valueToColor(val, min, max) }}
                />
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TensorViewer;
