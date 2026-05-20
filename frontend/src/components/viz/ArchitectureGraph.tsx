import { useMemo } from 'react'
import {
  ReactFlow,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { ModelConfig } from '../../hooks/useModels'

interface Props {
  config: ModelConfig
}

const NODE_W = 160
const NODE_H = 44
const GAP = 28

function makeNode(id: string, label: string, sub: string, y: number, accent: string): Node {
  return {
    id,
    position: { x: 0, y },
    data: { label: `${label}|${sub}` },
    type: 'default',
    style: {
      width: NODE_W,
      height: NODE_H,
      background: 'rgba(255,255,255,0.03)',
      border: `1px solid ${accent}44`,
      borderRadius: '8px',
      color: accent,
      fontSize: '10px',
      fontFamily: 'JetBrains Mono, monospace',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '4px',
      boxShadow: `0 0 12px ${accent}22`,
      cursor: 'default',
    },
  }
}

function makeEdge(src: string, tgt: string): Edge {
  return {
    id: `${src}-${tgt}`,
    source: src,
    target: tgt,
    type: 'smoothstep',
    style: { stroke: 'url(#edge-grad)', strokeWidth: 1.5, opacity: 0.6 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#a855f7', width: 8, height: 8 },
  }
}

export default function ArchitectureGraph({ config }: Props) {
  const { n_layers, d_model, n_heads, d_mlp, act_fn } = config

  const { nodes: initNodes, edges: initEdges } = useMemo(() => {
    const nodes: Node[] = []
    const edges: Edge[] = []
    let y = 0

    nodes.push(makeNode('embed', 'Embedding', `d_vocab → d_model=${d_model}`, y, '#00d4ff'))
    y += NODE_H + GAP

    nodes.push(makeNode('pos', 'Pos Embed', `n_ctx=${config.n_ctx}`, y, '#00d4ff'))
    edges.push(makeEdge('embed', 'pos'))
    y += NODE_H + GAP

    let prev = 'pos'
    for (let i = 0; i < n_layers; i++) {
      const id = `block${i}`
      const label = `Block ${i}`
      const sub = `${n_heads}h · d_mlp=${d_mlp} · ${act_fn}`
      nodes.push(makeNode(id, label, sub, y, '#a855f7'))
      edges.push(makeEdge(prev, id))
      prev = id
      y += NODE_H + GAP
    }

    nodes.push(makeNode('ln_final', 'LayerNorm', config.normalization_type, y, '#00d4ff'))
    edges.push(makeEdge(prev, 'ln_final'))
    y += NODE_H + GAP

    nodes.push(makeNode('unembed', 'Unembed', `d_model → d_vocab=${config.d_vocab}`, y, '#4ade80'))
    edges.push(makeEdge('ln_final', 'unembed'))

    return { nodes, edges }
  }, [config])

  const [nodes, , onNodesChange] = useNodesState(initNodes)
  const [edges, , onEdgesChange] = useEdgesState(initEdges)

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <svg width="0" height="0">
        <defs>
          <linearGradient id="edge-grad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#00d4ff" />
            <stop offset="100%" stopColor="#a855f7" />
          </linearGradient>
        </defs>
      </svg>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        style={{ background: 'transparent' }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="rgba(255,255,255,0.03)" gap={20} size={1} />
      </ReactFlow>
    </div>
  )
}
