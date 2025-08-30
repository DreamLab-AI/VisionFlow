import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { useAnalyticsStore } from './analyticsStore'
import type { GraphNode, GraphEdge } from '../../graph/types/graphTypes'

// Mock the logger and debug utilities
vi.mock('../../../utils/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn()
  }),
  createErrorMetadata: vi.fn()
}))

vi.mock('../../../utils/clientDebugState', () => ({
  debugState: {
    isEnabled: () => false
  }
}))

// Mock global objects for Node.js testing environment
global.window = global.window || {}
global.localStorage = {
  getItem: vi.fn(() => null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
}

describe('AnalyticsStore', () => {
  const sampleNodes: GraphNode[] = [
    { id: 'A', label: 'Node A', position: { x: 0, y: 0, z: 0 } },
    { id: 'B', label: 'Node B', position: { x: 1, y: 0, z: 0 } },
    { id: 'C', label: 'Node C', position: { x: 2, y: 0, z: 0 } },
    { id: 'D', label: 'Node D', position: { x: 0, y: 1, z: 0 } }
  ]

  const sampleEdges: GraphEdge[] = [
    { id: 'e1', source: 'A', target: 'B', weight: 1 },
    { id: 'e2', source: 'B', target: 'C', weight: 2 },
    { id: 'e3', source: 'A', target: 'D', weight: 3 }
  ]

  beforeEach(() => {
    // Reset the store to initial state
    useAnalyticsStore.setState({
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
      }
    })
  })

  afterEach(() => {
    // Clean up after each test
    vi.clearAllMocks()
  })

  describe('SSSP Computation', () => {
    it('should compute shortest paths correctly using Dijkstra', async () => {
      const result = await useAnalyticsStore.getState().computeSSSP(
        sampleNodes, 
        sampleEdges, 
        'A'
      )

      expect(result).toBeDefined()
      expect(result.sourceNodeId).toBe('A')
      expect(result.algorithm).toBe('dijkstra')
      expect(result.distances['A']).toBe(0)
      expect(result.distances['B']).toBe(1)
      expect(result.distances['C']).toBe(3) // A->B->C = 1+2 = 3
      expect(result.distances['D']).toBe(3) // A->D = 3
      expect(result.unreachableCount).toBe(0)
      expect(result.computationTime).toBeGreaterThan(0)
    })

    it('should compute shortest paths correctly using Bellman-Ford', async () => {
      const result = await useAnalyticsStore.getState().computeSSSP(
        sampleNodes, 
        sampleEdges, 
        'A', 
        'bellman-ford'
      )

      expect(result.algorithm).toBe('bellman-ford')
      expect(result.distances['A']).toBe(0)
      expect(result.distances['B']).toBe(1)
      expect(result.distances['C']).toBe(3)
      expect(result.distances['D']).toBe(3)
    })

    it('should handle unreachable nodes', async () => {
      const isolatedNodes: GraphNode[] = [
        ...sampleNodes,
        { id: 'E', label: 'Isolated', position: { x: 5, y: 5, z: 0 } }
      ]

      const result = await useAnalyticsStore.getState().computeSSSP(
        isolatedNodes, 
        sampleEdges, 
        'A'
      )

      expect(result.unreachableCount).toBe(1)
      expect(result.distances['E']).toBe(Infinity)
    })

    it('should handle invalid input gracefully', async () => {
      await expect(
        useAnalyticsStore.getState().computeSSSP([], [], 'A')
      ).rejects.toThrow('Invalid input')

      await expect(
        useAnalyticsStore.getState().computeSSSP(sampleNodes, sampleEdges, 'INVALID')
      ).rejects.toThrow('Source node with id INVALID not found')
    })

    it('should update loading state during computation', async () => {
      const store = useAnalyticsStore.getState()
      expect(store.loading).toBe(false)

      // Start computation and check loading state immediately
      const computationPromise = store.computeSSSP(sampleNodes, sampleEdges, 'A')
      
      // Wait for the promise to resolve
      await computationPromise
      
      // Should not be loading after completion
      expect(useAnalyticsStore.getState().loading).toBe(false)
    })
  })

  describe('Caching', () => {
    it('should cache computation results', async () => {
      const store = useAnalyticsStore.getState()
      
      // First computation
      const result1 = await store.computeSSSP(sampleNodes, sampleEdges, 'A')
      const metrics1 = useAnalyticsStore.getState().metrics
      expect(metrics1.cacheMisses).toBe(1)
      expect(metrics1.cacheHits).toBe(0)

      // Second computation with same parameters should use cache
      const result2 = await store.computeSSSP(sampleNodes, sampleEdges, 'A')
      const metrics2 = useAnalyticsStore.getState().metrics
      expect(metrics2.cacheHits).toBe(1)
      expect(result1.timestamp).toBe(result2.timestamp) // Same cached result
    })

    it('should invalidate cache when graph changes', async () => {
      const store = useAnalyticsStore.getState()
      
      // First computation
      await store.computeSSSP(sampleNodes, sampleEdges, 'A')
      
      // Change graph structure
      const newEdges = [...sampleEdges, { id: 'e4', source: 'C', target: 'D', weight: 1 }]
      
      // Should not use cache due to graph change
      await store.computeSSSP(sampleNodes, newEdges, 'A')
      const metrics = useAnalyticsStore.getState().metrics
      expect(metrics.cacheMisses).toBe(2)
    })

    it('should clean expired cache entries', async () => {
      const store = useAnalyticsStore.getState()
      
      // First, add a cache entry by computing
      await store.computeSSSP(sampleNodes, sampleEdges, 'A')
      
      // Manually set an old timestamp by accessing the store directly
      const currentState = useAnalyticsStore.getState()
      useAnalyticsStore.setState({
        ...currentState,
        cache: {
          ...currentState.cache,
          'A': {
            ...currentState.cache['A'],
            lastAccessed: Date.now() - 25 * 60 * 60 * 1000 // 25 hours ago
          }
        }
      })

      expect(Object.keys(useAnalyticsStore.getState().cache)).toContain('A')
      
      store.cleanExpiredCache(24 * 60 * 60 * 1000) // 24 hours
      
      expect(Object.keys(useAnalyticsStore.getState().cache)).not.toContain('A')
    })
  })

  describe('Distance Normalization', () => {
    it('should normalize distances to 0-1 range', async () => {
      const result = await useAnalyticsStore.getState().computeSSSP(
        sampleNodes, 
        sampleEdges, 
        'A'
      )
      
      const normalized = useAnalyticsStore.getState().normalizeDistances(result)
      
      expect(normalized['A']).toBe(0) // Source node
      expect(normalized['B']).toBeGreaterThanOrEqual(0)
      expect(normalized['B']).toBeLessThanOrEqual(1)
      expect(normalized['C']).toBeGreaterThanOrEqual(0)
      expect(normalized['C']).toBeLessThanOrEqual(1)
      expect(normalized['D']).toBeGreaterThanOrEqual(0)
      expect(normalized['D']).toBeLessThanOrEqual(1)
    })

    it('should handle case where all reachable nodes have same distance', () => {
      const result = {
        sourceNodeId: 'A',
        distances: { A: 0, B: 5, C: 5, D: 5 },
        predecessors: { A: null, B: 'A', C: 'A', D: 'A' },
        unreachableCount: 0,
        computationTime: 1,
        timestamp: Date.now(),
        algorithm: 'dijkstra' as const
      }
      
      const normalized = useAnalyticsStore.getState().normalizeDistances(result)
      
      expect(normalized['A']).toBe(0)
      expect(normalized['B']).toBe(1)
      expect(normalized['C']).toBe(1)
      expect(normalized['D']).toBe(1)
    })
  })

  describe('Utility Functions', () => {
    it('should identify unreachable nodes', async () => {
      const isolatedNodes: GraphNode[] = [
        ...sampleNodes,
        { id: 'E', label: 'Isolated', position: { x: 5, y: 5, z: 0 } }
      ]

      const result = await useAnalyticsStore.getState().computeSSSP(
        isolatedNodes, 
        sampleEdges, 
        'A'
      )
      
      const unreachable = useAnalyticsStore.getState().getUnreachableNodes(result)
      expect(unreachable).toEqual(['E'])
    })

    it('should clear results', () => {
      const store = useAnalyticsStore.getState()
      
      // Set some state
      store.setError('Test error')
      
      store.clearResults()
      
      expect(store.currentResult).toBeNull()
      expect(store.error).toBeNull()
    })

    it('should clear cache', async () => {
      const store = useAnalyticsStore.getState()
      
      // Add cache entry by computing
      await store.computeSSSP(sampleNodes, sampleEdges, 'A')
      expect(Object.keys(useAnalyticsStore.getState().cache)).toHaveLength(1)
      
      store.clearCache()
      
      expect(Object.keys(useAnalyticsStore.getState().cache)).toHaveLength(0)
    })
  })

  describe('Metrics', () => {
    it('should track computation metrics', async () => {
      const store = useAnalyticsStore.getState()
      
      await store.computeSSSP(sampleNodes, sampleEdges, 'A')
      
      const metrics = useAnalyticsStore.getState().metrics
      expect(metrics.totalComputations).toBe(1)
      expect(metrics.cacheMisses).toBe(1)
      expect(metrics.cacheHits).toBe(0)
      expect(metrics.averageComputationTime).toBeGreaterThan(0)
      expect(metrics.lastComputationTime).toBeGreaterThan(0)
    })

    it('should reset metrics', () => {
      const store = useAnalyticsStore.getState()
      
      // Set some metrics
      store.updateMetrics(100, false)
      
      store.resetMetrics()
      
      expect(store.metrics.totalComputations).toBe(0)
      expect(store.metrics.cacheMisses).toBe(0)
      expect(store.metrics.cacheHits).toBe(0)
      expect(store.metrics.averageComputationTime).toBe(0)
      expect(store.metrics.lastComputationTime).toBe(0)
    })
  })

  describe('Store State Management', () => {
    it('should maintain current result in store', async () => {
      const result = await useAnalyticsStore.getState().computeSSSP(
        sampleNodes, 
        sampleEdges, 
        'A'
      )
      
      expect(useAnalyticsStore.getState().currentResult).toEqual(result)
    })

    it('should handle errors properly', async () => {
      await expect(
        useAnalyticsStore.getState().computeSSSP([], [], 'invalid')
      ).rejects.toThrow()
      
      const state = useAnalyticsStore.getState()
      expect(state.loading).toBe(false)
      expect(state.error).toBeTruthy()
    })
  })
})