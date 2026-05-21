import { useState, useMemo } from 'react'
import axios from 'axios'
import InterpretationModal, { type InterpretationGuide } from '../../components/shared/InterpretationModal'

const API = ''

const GUIDE: InterpretationGuide = {
  overview:
    'Logit Lens unembeds the residual stream at the output of each layer — giving you a "prediction" at every depth of the model, ' +
    'not just the final layer. ' +
    'Each row is a layer (0 = earliest, final = output). Each column is a sequence position. ' +
    'The chip in each cell is the top-1 predicted token at that layer, with its probability. ' +
    'A green ✦ EMERGES badge marks the first layer where the correct answer breaks through above a 5% threshold. ' +
    'Early layers predict grammatical fillers ("the", "a"); the correct semantic answer crystallises in mid-to-late layers.\n\n' +
    'The Generated Response section runs the same analysis on tokens the model auto-regressively produces. ' +
    'Each column is a generated token; the predictions shown are what each layer "voted for" at the moment that token was produced.',
  example: {
    prompt: 'Run run_with_cache on "The Eiffel Tower is in", then click Run Logit Lens. Select position 4 (" in").',
    output:
      'Layer 0:  "the"     0.8%\n' +
      'Layer 3:  "the"     1.2%\n' +
      'Layer 6:  "France"  3.1%\n' +
      'Layer 9:  "Paris"   7.4%  ✦ EMERGES\n' +
      'Layer 11: "Paris"  18.4%',
    interpretation:
      'Layers 0–8 predict function words — the model hasn\'t yet located the fact.\n' +
      'At layer 9, " Paris" crosses 5% — this is where geographic knowledge enters the residual stream.\n' +
      'The green ✦ badge highlights this emergence point.\n' +
      'Layers 9–11 refine confidence from 7% to 18%.\n' +
      'This tells you the factual recall happens in the L9 MLP block, not in attention.',
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

interface TokenPred { token_id: number; token_str: string; probability: number }
interface PosPreds { position: number; top_k: TokenPred[] }
interface LayerResult { layer: number; label: string; predictions: PosPreds[] }
interface LensResponse { results: LayerResult[]; str_tokens: string[]; n_layers: number }
interface GenResponse { results: LayerResult[]; str_tokens: string[]; prompt_len: number; n_layers: number }

const panel: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 12,
  padding: 16,
}

function TokenChip({
  token, isTarget, isFirst, accent,
}: { token: TokenPred; isTarget: boolean; isFirst: boolean; accent: string }) {
  const opacity = Math.min(1, token.probability * 0.85 + 0.15)
  return (
    <span style={{
      display: 'inline-block',
      background: `rgba(0,212,255,${opacity * 0.25})`,
      border: `1px solid ${isFirst && isTarget ? '#4ade80' : isTarget ? accent : 'rgba(0,212,255,0.2)'}`,
      borderRadius: 4,
      padding: '2px 6px',
      fontSize: 10,
      fontFamily: 'JetBrains Mono, monospace',
      color: isTarget ? '#fff' : 'rgba(255,255,255,0.5)',
      marginRight: 2,
      whiteSpace: 'nowrap',
      maxWidth: 90,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
    }}>
      {token.token_str.slice(0, 8)}{' '}
      <span style={{ color: `rgba(0,212,255,${opacity})`, fontSize: 9 }}>
        {(token.probability * 100).toFixed(1)}%
      </span>
    </span>
  )
}

function LayerRow({
  result, targetPos, firstHitLayer, strTokens, cellWidth, accent, positionOffset,
}: {
  result: LayerResult
  targetPos: number
  firstHitLayer: number | null
  strTokens: string[]
  cellWidth: number
  accent: string
  positionOffset: number   // absolute position in full sequence for strTokens[0]
}) {
  const isFirst = firstHitLayer === result.layer
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 4,
      padding: '3px 0',
      borderLeft: isFirst ? '3px solid #4ade80' : '3px solid transparent',
      paddingLeft: 8,
    }}>
      <div style={{
        width: 120, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 4,
        position: 'sticky', left: 0, background: '#0a0a0f', zIndex: 1,
      }}>
        <span style={{
          width: 52,
          flexShrink: 0,
          fontSize: 10,
          fontFamily: 'JetBrains Mono, monospace',
          color: isFirst ? '#4ade80' : 'rgba(255,255,255,0.35)',
          textAlign: 'right',
          paddingRight: 8,
        }}>
          {result.label}
        </span>
        <span style={{
          fontSize: 8,
          padding: '1px 5px',
          borderRadius: 3,
          background: 'rgba(74,222,128,0.15)',
          border: '1px solid rgba(74,222,128,0.4)',
          color: '#4ade80',
          flexShrink: 0,
          visibility: isFirst ? 'visible' : 'hidden',
        }}>✦ EMERGES</span>
      </div>

      <div style={{ display: 'flex', gap: 3, flexWrap: 'nowrap' }}>
        {strTokens.map((tok, colIdx) => {
          const absPos = positionOffset + colIdx
          const posPred = result.predictions.find(p => p.position === absPos)
          if (!posPred || posPred.top_k.length === 0) return null
          if (colIdx === targetPos) {
            return (
              <div key={colIdx} style={{ width: cellWidth + 20, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
                {posPred.top_k.slice(0, 3).map((t, i) => (
                  <TokenChip key={t.token_id} token={t} isTarget={true} isFirst={isFirst && i === 0} accent={accent} />
                ))}
              </div>
            )
          }
          const top1 = posPred.top_k[0]
          return (
            <span key={colIdx} style={{
              display: 'inline-block',
              width: cellWidth,
              flexShrink: 0,
              fontSize: 9,
              fontFamily: 'JetBrains Mono, monospace',
              color: 'rgba(255,255,255,0.2)',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 3,
              padding: '1px 4px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              boxSizing: 'border-box',
            }}>
              {top1.token_str.slice(0, 6)}
            </span>
          )
        })}
      </div>
    </div>
  )
}

function SectionLabel({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <div style={{
      padding: '6px 20px 2px',
      fontSize: 9,
      fontFamily: 'JetBrains Mono, monospace',
      color,
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
      flexShrink: 0,
    }}>
      {children}
    </div>
  )
}

export default function LogitLens() {
  const [targetPos, setTargetPos] = useState(7)
  const [data, setData] = useState<LensResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [guideOpen, setGuideOpen] = useState(false)

  const [genData, setGenData] = useState<GenResponse | null>(null)
  const [genLoading, setGenLoading] = useState(false)
  const [genError, setGenError] = useState<string | null>(null)
  const [maxNewTokens, setMaxNewTokens] = useState(8)
  const [genTargetPos, setGenTargetPos] = useState(0)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const { data: res } = await axios.post<LensResponse>(`${API}/api/inference/logit_lens`, { top_k: 3 })
      setData(res)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Request failed. Run /api/inference/run_with_cache first.')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  const runGeneration = async () => {
    setGenLoading(true)
    setGenError(null)
    try {
      const { data: res } = await axios.post<GenResponse>(
        `${API}/api/inference/logit_lens_generation`,
        { top_k: 3, max_new_tokens: maxNewTokens },
      )
      setGenData(res)
      setGenTargetPos(0)
    } catch (e: any) {
      setGenError(e?.response?.data?.detail ?? 'Generation failed. Run /api/inference/run_with_cache first.')
      setGenData(null)
    } finally {
      setGenLoading(false)
    }
  }

  const cellWidth = useMemo(() => {
    if (!data) return 48
    const maxLen = Math.max(...data.str_tokens.map(t => Math.min(t.length, 8)), 3)
    return maxLen * 6 + 10
  }, [data])

  const genCellWidth = useMemo(() => {
    if (!genData) return 48
    const genTokens = genData.str_tokens.slice(genData.prompt_len)
    const maxLen = Math.max(...genTokens.map(t => Math.min(t.length, 8)), 3)
    return maxLen * 6 + 10
  }, [genData])

  const firstHitLayer: number | null = (() => {
    if (!data) return null
    for (const lr of data.results) {
      const p = lr.predictions.find(p => p.position === targetPos)
      if (p && p.top_k[0]?.probability > 0.05) return lr.layer
    }
    return null
  })()

  // For generation: find first layer where top-1 at the "producing" position exceeds 5%
  // Generated token I is produced by predictions at position (prompt_len - 1 + I)
  const genFirstHitLayer: number | null = (() => {
    if (!genData) return null
    const absPos = genData.prompt_len - 1 + genTargetPos
    for (const lr of genData.results) {
      const p = lr.predictions.find(p => p.position === absPos)
      if (p && p.top_k[0]?.probability > 0.05) return lr.layer
    }
    return null
  })()

  // Remap generation results so LayerRow can find predictions by colIdx
  // Generated token I (0-indexed col) corresponds to absPos = prompt_len - 1 + I
  // We inject a synthetic positionOffset so LayerRow's absPos = positionOffset + colIdx lines up
  const genPositionOffset = genData ? genData.prompt_len - 1 : 0
  const genTokens = genData ? genData.str_tokens.slice(genData.prompt_len) : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#0a0a0f', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        padding: '12px 20px',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#00d4ff', fontFamily: 'JetBrains Mono, monospace' }}>
          Logit Lens
        </span>
        <button style={GUIDE_BTN} onClick={() => setGuideOpen(true)}>? How to read this</button>
        <span style={{
          fontSize: 9,
          padding: '2px 7px',
          borderRadius: 4,
          background: 'rgba(168,85,247,0.12)',
          border: '1px solid rgba(168,85,247,0.3)',
          color: '#a855f7',
        }}>Phase 5</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>

        {/* ── INPUT ANALYSIS ─────────────────────────────────────── */}
        <SectionLabel color="rgba(0,212,255,0.5)">Input Analysis</SectionLabel>

        {/* Input controls */}
        <div style={{ padding: '8px 20px', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 12 }}>
          <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', fontFamily: 'JetBrains Mono, monospace' }}>
            Target position:
          </label>
          <input
            type="number"
            value={targetPos}
            min={0}
            max={30}
            onChange={e => setTargetPos(Number(e.target.value))}
            style={{
              width: 60,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 6,
              color: '#fff',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
              padding: '4px 8px',
            }}
          />
          <button
            onClick={run}
            disabled={loading}
            style={{
              padding: '6px 18px',
              background: loading ? 'rgba(0,212,255,0.1)' : 'rgba(0,212,255,0.15)',
              border: '1px solid rgba(0,212,255,0.4)',
              borderRadius: 6,
              color: '#00d4ff',
              fontSize: 11,
              cursor: loading ? 'not-allowed' : 'pointer',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            {loading ? 'Running…' : 'Run Logit Lens'}
          </button>

          {data && (
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', fontFamily: 'JetBrains Mono, monospace' }}>
              {data.n_layers} layers · {data.str_tokens.length} tokens
              {firstHitLayer !== null && (
                <span style={{ color: '#4ade80', marginLeft: 10 }}>
                  ✦ answer emerges at {data.results[firstHitLayer]?.label}
                </span>
              )}
            </span>
          )}
        </div>

        {error && (
          <div style={{ padding: '8px 20px', color: '#ff6b6b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', flexShrink: 0 }}>
            {error}
          </div>
        )}

        {/* Input column headers + layer rows — shared horizontal scroll */}
        {data && (
          <div style={{ overflowX: 'auto', flexShrink: 0 }}>
            <div style={{ padding: '4px 20px', display: 'flex', gap: 4, alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.04)', minWidth: 'max-content' }}>
              <span style={{ width: 128, flexShrink: 0 }} />
              {data.str_tokens.map((t, i) => (
                <span key={i} style={{
                  fontSize: 9,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: i === targetPos ? '#a855f7' : 'rgba(255,255,255,0.2)',
                  background: i === targetPos ? 'rgba(168,85,247,0.1)' : 'transparent',
                  border: i === targetPos ? '1px solid rgba(168,85,247,0.3)' : '1px solid transparent',
                  borderRadius: 3,
                  padding: '1px 4px',
                  width: i === targetPos ? cellWidth + 20 : cellWidth,
                  flexShrink: 0,
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  boxSizing: 'border-box',
                }}>
                  {t.slice(0, 8)}
                </span>
              ))}
            </div>
            <div style={{ padding: '4px 20px 12px', minWidth: 'max-content' }}>
              {data.results.map(lr => (
                <LayerRow
                  key={lr.layer}
                  result={lr}
                  targetPos={targetPos}
                  firstHitLayer={firstHitLayer}
                  strTokens={data.str_tokens}
                  cellWidth={cellWidth}
                  accent="#a855f7"
                  positionOffset={0}
                />
              ))}
            </div>
          </div>
        )}
        {!data && (
          !loading && (
            <div style={{ padding: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ ...panel, textAlign: 'center', maxWidth: 360 }}>
                <div style={{ fontSize: 28, marginBottom: 12, opacity: 0.2 }}>◉</div>
                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', fontFamily: 'JetBrains Mono, monospace' }}>
                  Run <span style={{ color: '#00d4ff' }}>/api/inference/run_with_cache</span> first,<br />
                  then click Run Logit Lens.
                </div>
              </div>
            </div>
          )
        )}

        {/* ── GENERATED RESPONSE ANALYSIS ───────────────────────── */}
        <div style={{ borderTop: '1px solid rgba(245,158,11,0.15)', marginTop: 4 }} />
        <SectionLabel color="rgba(245,158,11,0.6)">Generated Response Analysis</SectionLabel>

        {/* Generation controls */}
        <div style={{ padding: '8px 20px', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 12 }}>
          <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', fontFamily: 'JetBrains Mono, monospace' }}>
            Generate tokens:
          </label>
          <input
            type="number"
            value={maxNewTokens}
            min={1}
            max={20}
            onChange={e => setMaxNewTokens(Number(e.target.value))}
            style={{
              width: 60,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 6,
              color: '#fff',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
              padding: '4px 8px',
            }}
          />
          <button
            onClick={runGeneration}
            disabled={genLoading}
            style={{
              padding: '6px 18px',
              background: genLoading ? 'rgba(245,158,11,0.1)' : 'rgba(245,158,11,0.15)',
              border: '1px solid rgba(245,158,11,0.4)',
              borderRadius: 6,
              color: '#f59e0b',
              fontSize: 11,
              cursor: genLoading ? 'not-allowed' : 'pointer',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            {genLoading ? 'Generating…' : 'Generate & Analyze'}
          </button>

          {genData && (
            <>
              <label style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', fontFamily: 'JetBrains Mono, monospace', marginLeft: 8 }}>
                Target token:
              </label>
              <input
                type="number"
                value={genTargetPos}
                min={0}
                max={Math.max(0, genTokens.length - 1)}
                onChange={e => setGenTargetPos(Number(e.target.value))}
                style={{
                  width: 60,
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 6,
                  color: '#fff',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: 12,
                  padding: '4px 8px',
                }}
              />
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', fontFamily: 'JetBrains Mono, monospace' }}>
                {genTokens.length} tokens generated
                {genFirstHitLayer !== null && (
                  <span style={{ color: '#4ade80', marginLeft: 10 }}>
                    ✦ decision solidifies at {genData.results[genFirstHitLayer]?.label}
                  </span>
                )}
              </span>
            </>
          )}
        </div>

        {genError && (
          <div style={{ padding: '8px 20px', color: '#ff6b6b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', flexShrink: 0 }}>
            {genError}
          </div>
        )}

        {/* Generated token column headers + layer rows — shared horizontal scroll */}
        {genData && genTokens.length > 0 && (
          <div style={{ overflowX: 'auto', flexShrink: 0 }}>
            <div style={{ padding: '4px 20px', display: 'flex', gap: 4, alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.04)', minWidth: 'max-content' }}>
              <span style={{ width: 128, flexShrink: 0 }} />
              {genTokens.map((t, i) => (
                <span key={i} style={{
                  fontSize: 9,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: i === genTargetPos ? '#f59e0b' : 'rgba(255,255,255,0.2)',
                  background: i === genTargetPos ? 'rgba(245,158,11,0.1)' : 'transparent',
                  border: i === genTargetPos ? '1px solid rgba(245,158,11,0.3)' : '1px solid transparent',
                  borderRadius: 3,
                  padding: '1px 4px',
                  width: i === genTargetPos ? genCellWidth + 20 : genCellWidth,
                  flexShrink: 0,
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  boxSizing: 'border-box',
                }}>
                  {t.slice(0, 8)}
                </span>
              ))}
            </div>
            <div style={{ padding: '4px 20px 20px', minWidth: 'max-content' }}>
              {genData.results.map(lr => (
                <LayerRow
                  key={lr.layer}
                  result={lr}
                  targetPos={genTargetPos}
                  firstHitLayer={genFirstHitLayer}
                  strTokens={genTokens}
                  cellWidth={genCellWidth}
                  accent="#f59e0b"
                  positionOffset={genPositionOffset}
                />
              ))}
            </div>
          </div>
        )}
        {(!genData || genTokens.length === 0) && (
          !genLoading && (
            <div style={{ padding: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ ...panel, textAlign: 'center', maxWidth: 400, borderColor: 'rgba(245,158,11,0.12)' }}>
                <div style={{ fontSize: 22, marginBottom: 10, opacity: 0.2 }}>▶</div>
                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', fontFamily: 'JetBrains Mono, monospace' }}>
                  After running Input Analysis, click{' '}
                  <span style={{ color: '#f59e0b' }}>Generate & Analyze</span>
                  {' '}to see layer-by-layer predictions<br />
                  for each generated token.
                </div>
              </div>
            </div>
          )
        )}

      </div>

      <InterpretationModal
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
        pageTitle="Logit Lens"
        pageType="logit-lens"
        guide={GUIDE}
        liveData={data ? {
          str_tokens: data.str_tokens,
          n_layers: data.n_layers,
          target_pos: targetPos,
          emergence_layer: firstHitLayer,
          results: data.results.map(r => ({
            layer: r.layer,
            label: r.label,
            predictions: r.predictions.filter(p => p.position === targetPos).map(p => ({ position: p.position, top_k: p.top_k.slice(0, 3) })),
          })),
        } : null}
      />
    </div>
  )
}
