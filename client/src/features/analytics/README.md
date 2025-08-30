# Analytics Feature

The Analytics feature provides graph analysis capabilities, with a focus on Single Source Shortest Path (SSSP) algorithms and related computations.

## Features

### SSSP Analysis
- **Dijkstra's Algorithm**: For graphs with non-negative edge weights
- **Bellman-Ford Algorithm**: For graphs with potentially negative edge weights
- **Caching**: Intelligent caching of computation results with automatic invalidation
- **Performance Metrics**: Detailed tracking of computation times and cache efficiency
- **Distance Normalization**: Option to normalize distances to 0-1 range for visualization

## Store Architecture

### `analyticsStore.ts`

The main Zustand store that manages SSSP state and computations:

```typescript
interface AnalyticsState {
  // SSSP State
  currentResult: SSSPResult | null
  cache: SSSPCache
  loading: boolean
  error: string | null
  
  // Analytics metrics
  metrics: AnalyticsMetrics
  
  // Actions
  computeSSSP: (nodes, edges, sourceNodeId, algorithm?) => Promise<SSSPResult>
  clearResults: () => void
  clearCache: () => void
  normalizeDistances: (result) => Record<string, number>
  // ... more methods
}
```

### Key Types

```typescript
interface SSSPResult {
  sourceNodeId: string
  distances: Record<string, number>      // Node ID -> distance
  predecessors: Record<string, string | null>  // Node ID -> predecessor
  unreachableCount: number
  computationTime: number
  timestamp: number
  algorithm: 'dijkstra' | 'bellman-ford'
}

interface SSSPCache {
  [sourceNodeId: string]: {
    result: SSSPResult
    lastAccessed: number
    graphHash: string  // Detects graph structure changes
  }
}
```

## Components

### `SSSPAnalysisPanel`

A comprehensive React component for SSSP analysis:

```typescript
interface SSSPAnalysisPanelProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  className?: string
}
```

**Features:**
- Source node selection dropdown
- Algorithm selection (Dijkstra/Bellman-Ford)
- Real-time computation with loading states
- Results table with distances and paths
- Normalized distance view option
- Performance metrics display
- Error handling

## Usage Examples

### Basic SSSP Computation

```typescript
import { useAnalyticsStore } from '../features/analytics'

const MyComponent = () => {
  const { computeSSSP, currentResult, loading, error } = useAnalyticsStore()
  
  const handleAnalyze = async () => {
    try {
      const result = await computeSSSP(nodes, edges, 'sourceNodeId', 'dijkstra')
      console.log('Shortest distances:', result.distances)
    } catch (err) {
      console.error('Analysis failed:', err)
    }
  }
  
  return (
    <div>
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? 'Computing...' : 'Analyze Graph'}
      </button>
      {currentResult && (
        <div>
          <p>Unreachable nodes: {currentResult.unreachableCount}</p>
          <p>Computation time: {currentResult.computationTime}ms</p>
        </div>
      )}
    </div>
  )
}
```

### Using the Analysis Panel

```typescript
import { SSSPAnalysisPanel } from '../features/analytics'

const GraphAnalysis = ({ graphData }) => {
  return (
    <div className="analysis-container">
      <SSSPAnalysisPanel 
        nodes={graphData.nodes}
        edges={graphData.edges}
        className="my-custom-styles"
      />
    </div>
  )
}
```

### Hook-based Access

```typescript
import { 
  useCurrentSSSPResult, 
  useSSSPLoading, 
  useSSSPError,
  useSSSPMetrics 
} from '../features/analytics'

const ResultsDisplay = () => {
  const result = useCurrentSSSPResult()
  const loading = useSSSPLoading()
  const error = useSSSPError()
  const metrics = useSSSPMetrics()
  
  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage error={error} />
  if (!result) return <EmptyState />
  
  return (
    <div>
      <h3>Analysis Results</h3>
      <p>Cache hit rate: {(metrics.cacheHits / metrics.totalComputations * 100).toFixed(1)}%</p>
      {/* Display results... */}
    </div>
  )
}
```

## Algorithms

### Dijkstra's Algorithm
- **Time Complexity**: O((V + E) log V) with binary heap
- **Space Complexity**: O(V)
- **Requirements**: Non-negative edge weights
- **Use Case**: Most common shortest path problems

### Bellman-Ford Algorithm
- **Time Complexity**: O(V × E)
- **Space Complexity**: O(V)
- **Requirements**: Can handle negative edge weights
- **Features**: Detects negative cycles
- **Use Case**: Graphs with negative weights

## Caching Strategy

The store implements an intelligent caching system:

1. **Graph Hashing**: Creates a hash from node IDs and edge structure to detect changes
2. **LRU Eviction**: Keeps the 50 most recently accessed results
3. **Automatic Invalidation**: Clears cache when graph structure changes
4. **Expiration**: Configurable expiration time (default: 24 hours)

### Cache Methods

```typescript
// Manual cache management
clearCache()                          // Clear all cached results
cleanExpiredCache(maxAge?: number)    // Remove expired entries
invalidateCache()                     // Force cache invalidation

// Automatic cache usage
getCachedResult(sourceNodeId, graphHash)  // Internal cache lookup
```

## Performance Monitoring

The store tracks detailed performance metrics:

```typescript
interface AnalyticsMetrics {
  totalComputations: number        // All computations (cached + fresh)
  cacheHits: number               // Successful cache retrievals
  cacheMisses: number             // Computations that required calculation
  averageComputationTime: number  // Average time for non-cached computations
  lastComputationTime: number     // Most recent computation time
}
```

## Testing

Comprehensive test suite includes:

- Algorithm correctness (both Dijkstra and Bellman-Ford)
- Caching behavior and invalidation
- Error handling for invalid inputs
- Distance normalization
- Metrics tracking
- Store state management

Run tests:
```bash
npm test -- src/features/analytics/store/analyticsStore.test.ts
```

## Integration Points

### With Graph Store
```typescript
// Example integration with graph data
const graphData = useGraphStore(state => state.currentGraph)
const result = await computeSSSP(
  graphData.nodes, 
  graphData.edges, 
  selectedNodeId
)
```

### With Visualization
```typescript
// Use normalized distances for visual styling
const normalizedDistances = normalizeDistances(result)
nodes.forEach(node => {
  const distance = normalizedDistances[node.id]
  node.visualStyle = {
    color: distanceToColor(distance),
    size: distanceToSize(distance)
  }
})
```

## Future Enhancements

- **All-Pairs Shortest Path**: Floyd-Warshall algorithm
- **Path Reconstruction**: Detailed path sequences between nodes
- **Graph Centrality**: Betweenness, closeness, eigenvector centrality
- **Community Detection**: Louvain, label propagation algorithms
- **Performance Optimization**: Web Workers for large graphs
- **Visualization Integration**: Direct rendering of shortest paths

## API Reference

### Store Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `computeSSSP` | `nodes, edges, sourceNodeId, algorithm?` | `Promise<SSSPResult>` | Computes shortest paths |
| `clearResults` | - | `void` | Clears current results |
| `clearCache` | - | `void` | Clears all cached results |
| `normalizeDistances` | `result` | `Record<string, number>` | Normalizes distances 0-1 |
| `getUnreachableNodes` | `result` | `string[]` | Gets unreachable node IDs |
| `updateMetrics` | `computationTime, fromCache` | `void` | Updates performance metrics |
| `resetMetrics` | - | `void` | Resets all metrics |

### Utility Hooks

- `useCurrentSSSPResult()` - Current SSSP result
- `useSSSPLoading()` - Loading state
- `useSSSPError()` - Error state  
- `useSSSPMetrics()` - Performance metrics