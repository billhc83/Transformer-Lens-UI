import { useState, useCallback } from 'react'

interface AttentionHeatmapProps {
  pattern: number[][]
  strTokens: string[]
  headIndex?: number
  layerIndex?: number
  compact?: boolean
  onClick?: () => void
  selected?: boolean
  headLabel?: string
  headLabelColor?: string
}

// Plasma colormap: dark blue → purple → magenta → orange → yellow
const PLASMA = [
  [13, 8, 135],
  [126, 3, 168],
  [204, 71, 120],
  [248, 149, 64],
  [240, 249, 33],
] as const

function cellColor(v: number): string {
  const t = Math.max(0, Math.min(1, v))
  const scaled = t * (PLASMA.length - 1)
  const i = Math.min(Math.floor(scaled), PLASMA.length - 2)
  const frac = scaled - i
  const [r0, g0, b0] = PLASMA[i]
  const [r1, g1, b1] = PLASMA[i + 1]
  return `rgb(${Math.round(r0 + frac * (r1 - r0))},${Math.round(g0 + frac * (g1 - g0))},${Math.round(b0 + frac * (b1 - b0))})`
}

interface Tooltip { q: string; k: string; v: number; x: number; y: number }

export default function AttentionHeatmap({
  pattern,
  strTokens,
  headIndex,
  compact = false,
  onClick,
  selected = false,
  headLabel: _headLabel,
  headLabelColor = '#00d4ff',
}: AttentionHeatmapProps) {
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)

  const cellSize = compact ? 14 : 28
  const nQ = pattern.length
  const nK = pattern[0]?.length ?? 0

  // Axis label space for full mode
  const labelW = compact ? 0 : 64
  const labelH = compact ? 0 : 56
  const svgW = labelW + nK * cellSize
  const svgH = nQ * cellSize + labelH

  const handleMove = useCallback(
    (qi: number, ki: number, e: React.MouseEvent) => {
      setTooltip({
        q: strTokens[qi] ?? String(qi),
        k: strTokens[ki] ?? String(ki),
        v: pattern[qi][ki],
        x: e.clientX + 12,
        y: e.clientY + 12,
      })
    },
    [strTokens, pattern],
  )

  return (
    <div
      style={{ position: 'relative', display: 'inline-block', cursor: onClick ? 'pointer' : 'default' }}
      onClick={onClick}
    >
      <svg
        width={svgW}
        height={svgH}
        style={{
          display: 'block',
          border: selected ? '2px solid #00d4ff' : '2px solid transparent',
          borderRadius: 4,
        }}
      >
        {/* Grid cells */}
        {pattern.map((row, qi) =>
          row.map((val, ki) => (
            <rect
              key={`${qi}-${ki}`}
              x={labelW + ki * cellSize}
              y={qi * cellSize}
              width={cellSize}
              height={cellSize}
              fill={cellColor(val)}
              onMouseMove={(e) => handleMove(qi, ki, e)}
              onMouseLeave={() => setTooltip(null)}
            />
          )),
        )}

        {/* Y-axis token labels (query, left side) */}
        {!compact &&
          strTokens.slice(0, nQ).map((t, qi) => (
            <text
              key={`yl-${qi}`}
              x={labelW - 6}
              y={qi * cellSize + cellSize / 2 + 4}
              textAnchor="end"
              fontSize={9}
              fill="rgba(255,255,255,0.55)"
              fontFamily="JetBrains Mono, monospace"
            >
              {t.length > 7 ? t.slice(0, 7) : t}
            </text>
          ))}

        {/* X-axis token labels (key, bottom) */}
        {!compact &&
          strTokens.slice(0, nK).map((t, ki) => (
            <text
              key={`xl-${ki}`}
              x={labelW + ki * cellSize + cellSize / 2}
              y={nQ * cellSize + 10}
              textAnchor="start"
              fontSize={9}
              fill="rgba(255,255,255,0.55)"
              fontFamily="JetBrains Mono, monospace"
              transform={`rotate(45,${labelW + ki * cellSize + cellSize / 2},${nQ * cellSize + 10})`}
            >
              {t.length > 7 ? t.slice(0, 7) : t}
            </text>
          ))}

        {/* Compact: head number badge */}
        {compact && headIndex !== undefined && (
          <text
            x={2}
            y={11}
            fontSize={8}
            fill={headLabelColor}
            fontFamily="JetBrains Mono, monospace"
          >
            H{headIndex}
          </text>
        )}
      </svg>

      {/* Color legend (expanded mode only) */}
      {!compact && (
        <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', fontFamily: 'JetBrains Mono, monospace' }}>0</span>
          <div
            style={{
              flex: 1,
              height: 8,
              borderRadius: 4,
              background: `linear-gradient(to right, ${PLASMA.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${(i / (PLASMA.length - 1)) * 100}%`).join(', ')})`,
            }}
          />
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', fontFamily: 'JetBrains Mono, monospace' }}>1</span>
        </div>
      )}

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x,
            top: tooltip.y,
            background: 'rgba(10,10,15,0.95)',
            border: '1px solid rgba(0,212,255,0.3)',
            borderRadius: 4,
            padding: '4px 8px',
            fontSize: 11,
            fontFamily: 'JetBrains Mono, monospace',
            color: 'rgba(255,255,255,0.85)',
            pointerEvents: 'none',
            zIndex: 9999,
            whiteSpace: 'nowrap',
          }}
        >
          Q: "{tooltip.q}" → K: "{tooltip.k}" — {tooltip.v.toFixed(3)}
        </div>
      )}
    </div>
  )
}
