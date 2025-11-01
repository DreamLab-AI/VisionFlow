import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createLogger, createErrorMetadata } from '../../../utils/loggerConfig'
import { debugState } from '../../../utils/clientDebugState'
import { produce } from 'immer'
import type { GraphNode, GraphEdge } from '../../graph/types/graphTypes'
import { unifiedApiClient } from '../../../services/api'

const logger = createLogger('AnalyticsStore')

// SSSP-specific types
export interface SSSPResult {
  sourceNodeId: string
  distances: Record<string, number>
  predecessors: Record<string, string | null>
  unreachableCount: number
  computationTime: number
  timestamp: number
  algorithm: 'dijkstra' | 'bellman-ford' | 'floyd-warshall'
}

export interface SSSPCache {
  [sourceNodeId: string]: {
    result: SSSPResult
    lastAccessed: number
    graphHash: string 
  }
}

export interface AnalyticsMetrics {
  totalComputations: number
  cacheHits: number
  cacheMisses: number
  averageComputationTime: number
  lastComputationTime: number
}

interface AnalyticsState {
  
  currentResult: SSSPResult | null
  cache: SSSPCache
  loading: boolean
  error: string | null
  
  
  metrics: AnalyticsMetrics
  
  
  lastGraphHash: string | null
  
  
  computeSSSP: (
    nodes: GraphNode[], 
    edges: GraphEdge[], 
    sourceNodeId: string,
    algorithm?: 'dijkstra' | 'bellman-ford' | 'floyd-warshall'
  ) => Promise<SSSPResult>
  
  clearResults: () => void
  clearCache: () => void
  getCachedResult: (sourceNodeId: string, graphHash: string) => SSSPResult | null
  normalizeDistances: (result: SSSPResult) => Record<string, number>
  getUnreachableNodes: (result: SSSPResult) => string[]
  
  
  invalidateCache: () => void
  cleanExpiredCache: (maxAge?: number) => void
  
  
  setError: (error: string | null) => void
  
  
  updateMetrics: (computationTime: number, fromCache: boolean) => void
  resetMetrics: () => void
}

// Hash function for graph structure
function hashGraph(nodes: GraphNode[], edges: GraphEdge[]): string {
  const nodeIds = nodes.map(n => n.id).sort().join(',')
  const edgeIds = edges.map(e => `${e.source}-${e.target}-${e.weight || 1}`).sort().join(',')
  return btoa(`${nodeIds}|${edgeIds}`)
}

// Dijkstra's algorithm implementation
function dijkstra(nodes: GraphNode[], edges: GraphEdge[], sourceNodeId: string): Omit<SSSPResult, 'timestamp' | 'computationTime' | 'algorithm'> {
  const distances: Record<string, number> = {}
  const predecessors: Record<string, string | null> = {}
  const visited = new Set<string>()
  const nodeIds = new Set(nodes.map(n => n.id))
  
  
  for (const node of nodes) {
    distances[node.id] = node.id === sourceNodeId ? 0 : Infinity
    predecessors[node.id] = null
  }
  
  
  const adjacencyList: Record<string, Array<{ nodeId: string; weight: number }>> = {}
  for (const node of nodes) {
    adjacencyList[node.id] = []
  }
  
  for (const edge of edges) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      const weight = edge.weight || 1
      adjacencyList[edge.source].push({ nodeId: edge.target, weight })
      
      adjacencyList[edge.target].push({ nodeId: edge.source, weight })
    }
  }
  
  
  while (visited.size < nodes.length) {
    
    let currentNode: string | null = null
    let minDistance = Infinity
    
    for (const nodeId of Object.keys(distances)) {
      if (!visited.has(nodeId) && distances[nodeId] < minDistance) {
        minDistance = distances[nodeId]
        currentNode = nodeId
      }
    }
    
    if (currentNode === null || minDistance === Infinity) break
    
    visited.add(currentNode)
    
    
    for (const neighbor of adjacencyList[currentNode] || []) {
      if (!visited.has(neighbor.nodeId)) {
        const newDistance = distances[currentNode] + neighbor.weight
        if (newDistance < distances[neighbor.nodeId]) {
          distances[neighbor.nodeId] = newDistance
          predecessors[neighbor.nodeId] = currentNode
        }
      }
    }
  }
  
  
  const unreachableCount = Object.values(distances).filter(d => d === Infinity).length
  
  return {
    sourceNodeId,
    distances,
    predecessors,
    unreachableCount
  }
}

// Bellman-Ford algorithm (handles negative weights)
function bellmanFord(nodes: GraphNode[], edges: GraphEdge[], sourceNodeId: string): Omit<SSSPResult, 'timestamp' | 'computationTime' | 'algorithm'> {
  const distances: Record<string, number> = {}
  const predecessors: Record<string, string | null> = {}
  
  
  for (const node of nodes) {
    distances[node.id] = node.id === sourceNodeId ? 0 : Infinity
    predecessors[node.id] = null
  }
  
  
  for (let i = 0; i < nodes.length - 1; i++) {
    for (const edge of edges) {
      const weight = edge.weight || 1
      if (distances[edge.source] !== Infinity) {
        const newDistance = distances[edge.source] + weight
        if (newDistance < distances[edge.target]) {
          distances[edge.target] = newDistance
          predecessors[edge.target] = edge.source
        }
      }
      
      if (distances[edge.target] !== Infinity) {
        const newDistance = distances[edge.target] + weight
        if (newDistance < distances[edge.source]) {
          distances[edge.source] = newDistance
          predecessors[edge.source] = edge.target
        }
      }
    }
  }
  
  
  for (const edge of edges) {
    const weight = edge.weight || 1
    if (distances[edge.source] !== Infinity && 
        distances[edge.source] + weight < distances[edge.target]) {
      logger.warn('Negative cycle detected in graph')
    }
  }
  
  const unreachableCount = Object.values(distances).filter(d => d === Infinity).length
  
  return {
    sourceNodeId,
    distances,
    predecessors,
    unreachableCount
  }
}

export const useAnalyticsStore = create<AnalyticsState>()(
  persist(
    (set, get) => ({
      
      currentResult: null,
      cache: {},
      loading: false,
      error: null,
      lastGraphHash: null,
      metrics: {
        totalComputations: 0,
        cacheHits: 0,
        cacheMisses: 0,
        averageComputationTime: 0,
        lastComputationTime: 0
      },

      computeSSSP: async (nodes, edges, sourceNodeId, algorithm = 'dijkstra') => {
        const startTime = performance.now()

        set({ loading: true, error: null })

        try {
          
          if (!nodes.length || !sourceNodeId) {
            throw new Error('Invalid input: nodes array is empty or sourceNodeId is missing')
          }

          const sourceNode = nodes.find(n => n.id === sourceNodeId)
          if (!sourceNode) {
            throw new Error(`Source node with id ${sourceNodeId} not found`)
          }

          
          const graphHash = hashGraph(nodes, edges)

          
          const cachedResult = get().getCachedResult(sourceNodeId, graphHash)
          if (cachedResult) {
            const computationTime = performance.now() - startTime
            get().updateMetrics(computationTime, true)

            set({
              currentResult: cachedResult,
              loading: false,
              lastGraphHash: graphHash
            })

            if (debugState.isEnabled()) {
              logger.info('SSSP result retrieved from cache', { sourceNodeId, algorithm })
            }

            return cachedResult
          }

          
          let result: SSSPResult

          
          try {
            const response = await unifiedApiClient.post('/api/analytics/shortest-path', {
              source_node_id: parseInt(sourceNodeId), 
            })

            const data = response.data

            if (!data.success) {
              throw new Error(data.error || 'SSSP computation failed on server')
            }

            
            const distances: Record<string, number> = {}
            const predecessors: Record<string, string | null> = {}

            for (const [nodeId, distance] of Object.entries(data.distances || {})) {
              distances[nodeId] = distance === null ? Infinity : distance as number
              
              predecessors[nodeId] = null
            }

            const computationTime = performance.now() - startTime

            result = {
              sourceNodeId,
              distances,
              predecessors,
              unreachableCount: data.unreachable_count || 0,
              algorithm,
              computationTime,
              timestamp: Date.now()
            }

            if (debugState.isEnabled()) {
              logger.info('SSSP computed on server', {
                sourceNodeId,
                algorithm,
                unreachableCount: result.unreachableCount,
                computationTime
              })
            }

          } catch (apiError) {
            
            logger.warn('Server SSSP failed, falling back to local computation', apiError)

            
            let baseResult: Omit<SSSPResult, 'timestamp' | 'computationTime' | 'algorithm'>

            switch (algorithm) {
              case 'bellman-ford':
                baseResult = bellmanFord(nodes, edges, sourceNodeId)
                break
              case 'dijkstra':
              default:
                baseResult = dijkstra(nodes, edges, sourceNodeId)
                break
            }

            const computationTime = performance.now() - startTime

            result = {
              ...baseResult,
              algorithm,
              computationTime,
              timestamp: Date.now()
            }
          }

          
          set(state => produce(state, draft => {
            draft.currentResult = result
            draft.loading = false
            draft.lastGraphHash = graphHash
            
            
            draft.cache[sourceNodeId] = {
              result,
              lastAccessed: Date.now(),
              graphHash
            }
            
            
            const cacheEntries = Object.entries(draft.cache)
            if (cacheEntries.length > 50) {
              
              const sortedEntries = cacheEntries.sort(([,a], [,b]) => b.lastAccessed - a.lastAccessed)
              draft.cache = Object.fromEntries(sortedEntries.slice(0, 50))
            }
          }))
          
          
          get().updateMetrics(computationTime, false)
          
          if (debugState.isEnabled()) {
            logger.info('SSSP computation completed', {
              sourceNodeId,
              algorithm,
              computationTime: `${computationTime.toFixed(2)}ms`,
              unreachableCount: result.unreachableCount,
              totalNodes: nodes.length
            })
          }
          
          return result
          
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error during SSSP computation'
          
          logger.error('SSSP computation failed:', createErrorMetadata(error))
          
          set({ 
            loading: false, 
            error: errorMessage 
          })
          
          throw error
        }
      },

      clearResults: () => {
        set({ 
          currentResult: null, 
          error: null 
        })
        
        if (debugState.isEnabled()) {
          logger.info('SSSP results cleared')
        }
      },

      clearCache: () => {
        set({ cache: {} })
        
        if (debugState.isEnabled()) {
          logger.info('SSSP cache cleared')
        }
      },

      getCachedResult: (sourceNodeId, graphHash) => {
        const state = get()
        const cached = state.cache[sourceNodeId]
        
        if (cached && cached.graphHash === graphHash) {
          
          set(state => produce(state, draft => {
            if (draft.cache[sourceNodeId]) {
              draft.cache[sourceNodeId].lastAccessed = Date.now()
            }
          }))
          
          return cached.result
        }
        
        return null
      },

      normalizeDistances: (result) => {
        if (!result) return {}
        
        const distances = { ...result.distances }
        const finiteDistances = Object.values(distances).filter(d => isFinite(d))
        
        if (finiteDistances.length === 0) return distances
        
        const maxDistance = Math.max(...finiteDistances)
        const minDistance = Math.min(...finiteDistances)
        const range = maxDistance - minDistance
        
        if (range === 0) {
          
          Object.keys(distances).forEach(nodeId => {
            if (isFinite(distances[nodeId])) {
              distances[nodeId] = 1
            }
          })
        } else {
          
          Object.keys(distances).forEach(nodeId => {
            if (isFinite(distances[nodeId])) {
              distances[nodeId] = (distances[nodeId] - minDistance) / range
            }
          })
        }
        
        return distances
      },

      getUnreachableNodes: (result) => {
        if (!result) return []
        
        return Object.entries(result.distances)
          .filter(([, distance]) => !isFinite(distance))
          .map(([nodeId]) => nodeId)
      },

      invalidateCache: () => {
        set({ 
          cache: {},
          lastGraphHash: null 
        })
        
        if (debugState.isEnabled()) {
          logger.info('SSSP cache invalidated')
        }
      },

      cleanExpiredCache: (maxAge = 24 * 60 * 60 * 1000) => { 
        const now = Date.now()
        
        set(state => produce(state, draft => {
          Object.entries(draft.cache).forEach(([sourceNodeId, cached]) => {
            if (now - cached.lastAccessed > maxAge) {
              delete draft.cache[sourceNodeId]
            }
          })
        }))
        
        if (debugState.isEnabled()) {
          logger.info('Expired SSSP cache entries cleaned', { maxAge })
        }
      },

      setError: (error) => {
        set({ error })
      },

      updateMetrics: (computationTime, fromCache) => {
        set(state => produce(state, draft => {
          draft.metrics.totalComputations += 1
          draft.metrics.lastComputationTime = computationTime
          
          if (fromCache) {
            draft.metrics.cacheHits += 1
          } else {
            draft.metrics.cacheMisses += 1
            
            
            const totalNonCacheComputations = draft.metrics.cacheMisses
            const currentAverage = draft.metrics.averageComputationTime
            draft.metrics.averageComputationTime = 
              (currentAverage * (totalNonCacheComputations - 1) + computationTime) / totalNonCacheComputations
          }
        }))
      },

      resetMetrics: () => {
        set(state => produce(state, draft => {
          draft.metrics = {
            totalComputations: 0,
            cacheHits: 0,
            cacheMisses: 0,
            averageComputationTime: 0,
            lastComputationTime: 0
          }
        }))
        
        if (debugState.isEnabled()) {
          logger.info('SSSP metrics reset')
        }
      }
    }),
    {
      name: 'analytics-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        cache: state.cache,
        metrics: state.metrics
      }),
      onRehydrateStorage: () => (state) => {
        if (state && debugState.isEnabled()) {
          logger.info('Analytics store rehydrated', {
            cacheEntries: Object.keys(state.cache || {}).length,
            metrics: state.metrics
          })
        }
      }
    }
  )
)

// Utility hooks for common operations
export const useCurrentSSSPResult = () => useAnalyticsStore(state => state.currentResult)
export const useSSSPLoading = () => useAnalyticsStore(state => state.loading)
export const useSSSPError = () => useAnalyticsStore(state => state.error)
export const useSSSPMetrics = () => useAnalyticsStore(state => state.metrics)

