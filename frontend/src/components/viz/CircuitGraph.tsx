import { useMemo, useEffect, useCallback } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

interface CircuitGraphProps {
  nLayers: number
  nHeads: number
  labels: string[]
  qScores: number[][]
  kScores: number[][]
  vScores: number[][]
  threshold: number
  onNodeClick: (layer: number, head: number) => void
  selectedNode: string | null
}

const NODE_W = 52
const NODE_H = 52
const GAP_X = 110
const GAP_Y = 72

export default function CircuitGraph({
  nHeads,
  labels,
  qScores,
  kScores,
  vScores,
  threshold,
  onNodeClick,
  selectedNode,
}: CircuitGraphProps) {
  const computedNodes = useMemo<Node[]>(() => {
    return labels.map((label, i) => {
      const layer = Math.floor(i / nHeads)
      const head = i % nHeads
      const isSelected = selectedNode === label
      return {
        id: label,
        position: { x: layer * GAP_X, y: head * GAP_Y },
        data: { label },
        type: 'default',
        style: {
          width: NODE_W,
          height: NODE_H,
          background: isSelected ? 'rgba(0,212,255,0.12)' : 'rgba(255,255,255,0.04)',
          border: `1px solid ${isSelected ? '#00d4ff' : 'rgba(255,255,255,0.12)'}`,
          borderRadius: '8px',
          color: isSelected ? '#00d4ff' : 'rgba(255,255,255,0.65)',
          fontSize: '9px',
          fontFamily: 'JetBrains Mono, monospace',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: isSelected ? '0 0 14px rgba(0,212,255,0.25)' : 'none',
          padding: 0,
          cursor: 'pointer',
        },
      }
    })
  }, [labels, nHeads, selectedNode])

  const computedEdges = useMemo<Edge[]>(() => {
    const edges: Edge[] = []
    const n = labels.length
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const q = qScores[i]?.[j] ?? 0
        const k = kScores[i]?.[j] ?? 0
        const v = vScores[i]?.[j] ?? 0
        const maxScore = Math.max(q, k, v)
        if (maxScore < threshold) continue

        let color: string
        if (maxScore === q) color = '#00d4ff'
        else if (maxScore === k) color = '#a855f7'
        else color = '#4ade80'

        const width = Math.min(Math.max(maxScore * 8, 0.5), 5)
        edges.push({
          id: `${labels[i]}-${labels[j]}`,
          source: labels[i],
          target: labels[j],
          animated: maxScore > 0.5,
          style: { stroke: color, strokeWidth: width, opacity: 0.75 },
        })
      }
    }
    return edges
  }, [labels, qScores, kScores, vScores, threshold])

  const [nodes, setNodes, onNodesChange] = useNodesState(computedNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(computedEdges)

  useEffect(() => { setNodes(computedNodes) }, [computedNodes, setNodes])
  useEffect(() => { setEdges(computedEdges) }, [computedEdges, setEdges])

  const handleNodeClick: NodeMouseHandler = useCallback((_evt, node) => {
    const idx = labels.indexOf(node.id)
    if (idx === -1) return
    onNodeClick(Math.floor(idx / nHeads), idx % nHeads)
  }, [labels, nHeads, onNodeClick])

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
        fitView
        fitViewOptions={{ padding: 0.1 }}
        style={{ background: 'transparent' }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="rgba(255,255,255,0.025)" gap={20} size={1} />
        <Controls
          showInteractive={false}
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
          }}
        />
      </ReactFlow>
    </div>
  )
}
