import { create } from 'zustand'

export type PageId =
  | 'model-hub'
  | 'token-inspector'
  | 'forward-pass'
  | 'activation-browser'
  | 'attention-viz'
  | 'logit-lens'
  | 'attribution'
  | 'patching-lab'
  | 'circuit-analyzer'
  | 'hook-lab'
  | 'generation-studio'
  | 'report-studio'

export interface Finding {
  page: PageId
  timestamp: number
  headline: string
  data: Record<string, unknown>
}

interface SessionState {
  activePage: PageId
  setActivePage: (page: PageId) => void

  visitedPages: PageId[]
  markVisited: (page: PageId) => void

  exploredLayers: Record<string, number[]>
  markLayerExplored: (page: string, layer: number) => void

  findings: Finding[]
  addFinding: (f: Omit<Finding, 'timestamp'>) => void
  clearFindings: () => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  activePage: 'model-hub',
  setActivePage: (page) => {
    set({ activePage: page })
    get().markVisited(page)
  },

  visitedPages: [],
  markVisited: (page) =>
    set((s) => ({
      visitedPages: s.visitedPages.includes(page)
        ? s.visitedPages
        : [...s.visitedPages, page],
    })),

  exploredLayers: {},
  markLayerExplored: (page, layer) =>
    set((s) => {
      const current = s.exploredLayers[page] ?? []
      if (current.includes(layer)) return s
      return { exploredLayers: { ...s.exploredLayers, [page]: [...current, layer] } }
    }),

  findings: [],
  addFinding: (f) =>
    set((s) => ({
      findings: [
        ...s.findings.filter(
          (x) => !(x.page === f.page && x.headline === f.headline)
        ),
        { ...f, timestamp: Date.now() },
      ],
    })),
  clearFindings: () => set({ findings: [] }),
}))
