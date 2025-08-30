import React, { useState, useCallback, useEffect } from 'react'
import { Card } from '../../design-system/components/Card'
import { Button } from '../../design-system/components/Button'
import { Select } from '../../design-system/components/Select'
import { LoadingSpinner } from '../../design-system/components/LoadingSpinner'
import { Badge } from '../../design-system/components/Badge'
import { Separator } from '../../design-system/components/Separator'
import { 
  useAnalyticsStore, 
  useCurrentSSSPResult, 
  useSSSPLoading, 
  useSSSPError,
  useSSSPMetrics,
  type SSSPResult 
} from '../store/analyticsStore'
import type { GraphNode, GraphEdge } from '../../graph/types/graphTypes'

interface SSSPAnalysisPanelProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  className?: string
}

export const SSSPAnalysisPanel: React.FC<SSSPAnalysisPanelProps> = ({
  nodes,
  edges,
  className = ''
}) => {
  const [selectedSourceNode, setSelectedSourceNode] = useState<string>('')
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<'dijkstra' | 'bellman-ford'>('dijkstra')
  const [showNormalized, setShowNormalized] = useState(false)

  const computeSSSP = useAnalyticsStore(state => state.computeSSSP)
  const normalizeDistances = useAnalyticsStore(state => state.normalizeDistances)
  const getUnreachableNodes = useAnalyticsStore(state => state.getUnreachableNodes)
  const clearResults = useAnalyticsStore(state => state.clearResults)
  
  const currentResult = useCurrentSSSPResult()
  const loading = useSSSPLoading()
  const error = useSSSPError()
  const metrics = useSSSPMetrics()

  // Set default source node when nodes change
  useEffect(() => {
    if (nodes.length > 0 && !selectedSourceNode) {
      setSelectedSourceNode(nodes[0].id)
    }
  }, [nodes, selectedSourceNode])

  const handleComputeSSSP = useCallback(async () => {
    if (!selectedSourceNode || nodes.length === 0) return

    try {
      await computeSSSP(nodes, edges, selectedSourceNode, selectedAlgorithm)
    } catch (err) {
      console.error('Failed to compute SSSP:', err)
    }
  }, [computeSSSP, nodes, edges, selectedSourceNode, selectedAlgorithm])

  const handleClearResults = useCallback(() => {
    clearResults()
    setShowNormalized(false)
  }, [clearResults])

  const formatDistance = (distance: number): string => {
    if (!isFinite(distance)) return '∞'
    return distance.toFixed(2)
  }

  const getDistancesToDisplay = (): Record<string, number> => {
    if (!currentResult) return {}
    return showNormalized ? normalizeDistances(currentResult) : currentResult.distances
  }

  const unreachableNodes = currentResult ? getUnreachableNodes(currentResult) : []

  return (
    <Card className={`p-6 space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Single Source Shortest Path Analysis</h3>
        <div className="flex items-center gap-2">
          <Badge variant={loading ? 'secondary' : currentResult ? 'success' : 'outline'}>
            {loading ? 'Computing...' : currentResult ? 'Complete' : 'Ready'}
          </Badge>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Source Node</label>
          <Select
            value={selectedSourceNode}
            onValueChange={setSelectedSourceNode}
            disabled={loading || nodes.length === 0}
          >
            {nodes.map(node => (
              <option key={node.id} value={node.id}>
                {node.label || node.id}
              </option>
            ))}
          </Select>
        </div>

        <div>
          <label className="text-sm font-medium mb-2 block">Algorithm</label>
          <Select
            value={selectedAlgorithm}
            onValueChange={(value) => setSelectedAlgorithm(value as 'dijkstra' | 'bellman-ford')}
            disabled={loading}
          >
            <option value="dijkstra">Dijkstra</option>
            <option value="bellman-ford">Bellman-Ford</option>
          </Select>
        </div>

        <div className="flex items-end">
          <Button
            onClick={handleComputeSSSP}
            disabled={loading || !selectedSourceNode || nodes.length === 0}
            className="w-full"
          >
            {loading ? <LoadingSpinner size="sm" className="mr-2" /> : null}
            Compute Paths
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 text-red-700">
          <h4 className="font-medium">Error</h4>
          <p className="text-sm mt-1">{error}</p>
        </div>
      )}

      {/* Results */}
      {currentResult && (
        <div className="space-y-4">
          <Separator />
          
          {/* Result Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-blue-50 p-3 rounded-md">
              <div className="font-medium text-blue-700">Source Node</div>
              <div className="text-lg font-bold text-blue-900">
                {nodes.find(n => n.id === currentResult.sourceNodeId)?.label || currentResult.sourceNodeId}
              </div>
            </div>
            
            <div className="bg-green-50 p-3 rounded-md">
              <div className="font-medium text-green-700">Reachable Nodes</div>
              <div className="text-lg font-bold text-green-900">
                {Object.keys(currentResult.distances).length - unreachableNodes.length}
              </div>
            </div>
            
            <div className="bg-orange-50 p-3 rounded-md">
              <div className="font-medium text-orange-700">Unreachable</div>
              <div className="text-lg font-bold text-orange-900">{unreachableNodes.length}</div>
            </div>
            
            <div className="bg-purple-50 p-3 rounded-md">
              <div className="font-medium text-purple-700">Computation Time</div>
              <div className="text-lg font-bold text-purple-900">
                {currentResult.computationTime.toFixed(2)}ms
              </div>
            </div>
          </div>

          {/* Distance Controls */}
          <div className="flex items-center justify-between">
            <h4 className="font-medium">Shortest Distances</h4>
            <div className="flex items-center gap-4">
              <label className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={showNormalized}
                  onChange={(e) => setShowNormalized(e.target.checked)}
                  className="mr-2"
                />
                Show Normalized (0-1)
              </label>
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearResults}
              >
                Clear Results
              </Button>
            </div>
          </div>

          {/* Distance Table */}
          <div className="max-h-64 overflow-y-auto border rounded-md">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="text-left p-3 border-b">Node</th>
                  <th className="text-right p-3 border-b">Distance</th>
                  <th className="text-left p-3 border-b">Via</th>
                  <th className="text-center p-3 border-b">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(getDistancesToDisplay()).map(([nodeId, distance]) => {
                  const node = nodes.find(n => n.id === nodeId)
                  const predecessor = currentResult.predecessors[nodeId]
                  const predecessorNode = predecessor ? nodes.find(n => n.id === predecessor) : null
                  const isUnreachable = !isFinite(distance)
                  const isSource = nodeId === currentResult.sourceNodeId
                  
                  return (
                    <tr key={nodeId} className={`border-b hover:bg-gray-50 ${isSource ? 'bg-blue-50' : ''}`}>
                      <td className="p-3 font-medium">
                        {node?.label || nodeId}
                        {isSource && <Badge variant="outline" className="ml-2 text-xs">Source</Badge>}
                      </td>
                      <td className="p-3 text-right font-mono">
                        {formatDistance(distance)}
                      </td>
                      <td className="p-3 text-gray-600">
                        {predecessor ? (predecessorNode?.label || predecessor) : isSource ? '-' : 'N/A'}
                      </td>
                      <td className="p-3 text-center">
                        <Badge 
                          variant={isUnreachable ? 'destructive' : isSource ? 'success' : 'secondary'}
                          className="text-xs"
                        >
                          {isUnreachable ? 'Unreachable' : isSource ? 'Source' : 'Reachable'}
                        </Badge>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Performance Metrics */}
          {metrics.totalComputations > 0 && (
            <>
              <Separator />
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs text-gray-600">
                <div>
                  <div className="font-medium">Total Computations</div>
                  <div className="text-lg font-bold text-gray-900">{metrics.totalComputations}</div>
                </div>
                <div>
                  <div className="font-medium">Cache Hits</div>
                  <div className="text-lg font-bold text-green-900">{metrics.cacheHits}</div>
                </div>
                <div>
                  <div className="font-medium">Cache Misses</div>
                  <div className="text-lg font-bold text-orange-900">{metrics.cacheMisses}</div>
                </div>
                <div>
                  <div className="font-medium">Avg Computation</div>
                  <div className="text-lg font-bold text-blue-900">
                    {metrics.averageComputationTime.toFixed(2)}ms
                  </div>
                </div>
                <div>
                  <div className="font-medium">Cache Hit Rate</div>
                  <div className="text-lg font-bold text-purple-900">
                    {metrics.totalComputations > 0 
                      ? ((metrics.cacheHits / metrics.totalComputations) * 100).toFixed(1)
                      : 0
                    }%
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </Card>
  )
}

export default SSSPAnalysisPanel