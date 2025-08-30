import React, { useState } from 'react'
import { useAnalyticsStore, useCurrentSSSPResult, useSSSPLoading } from '../store/analyticsStore'
import type { GraphNode, GraphEdge } from '../../graph/types/graphTypes'

// Example usage of the Analytics Store
export const BasicUsageExample: React.FC = () => {
  // Sample graph data
  const [nodes] = useState<GraphNode[]>([
    { id: 'A', label: 'Node A', position: { x: 0, y: 0, z: 0 } },
    { id: 'B', label: 'Node B', position: { x: 1, y: 0, z: 0 } },
    { id: 'C', label: 'Node C', position: { x: 2, y: 0, z: 0 } },
    { id: 'D', label: 'Node D', position: { x: 0, y: 1, z: 0 } }
  ])

  const [edges] = useState<GraphEdge[]>([
    { id: 'e1', source: 'A', target: 'B', weight: 1 },
    { id: 'e2', source: 'B', target: 'C', weight: 2 },
    { id: 'e3', source: 'A', target: 'D', weight: 3 }
  ])

  // Use the store actions and selectors
  const computeSSSP = useAnalyticsStore(state => state.computeSSSP)
  const clearResults = useAnalyticsStore(state => state.clearResults)
  const normalizeDistances = useAnalyticsStore(state => state.normalizeDistances)
  
  // Use the convenience hooks
  const result = useCurrentSSSPResult()
  const loading = useSSSPLoading()

  const handleAnalyze = async () => {
    try {
      await computeSSSP(nodes, edges, 'A', 'dijkstra')
    } catch (error) {
      console.error('Analysis failed:', error)
    }
  }

  const handleClear = () => {
    clearResults()
  }

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">Analytics Store Example</h2>
      
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">Graph Structure</h3>
        <div className="bg-gray-50 p-4 rounded">
          <p><strong>Nodes:</strong> {nodes.map(n => n.label).join(', ')}</p>
          <p><strong>Edges:</strong></p>
          <ul className="ml-4">
            {edges.map(e => (
              <li key={e.id}>
                {nodes.find(n => n.id === e.source)?.label} → {nodes.find(n => n.id === e.target)?.label} (weight: {e.weight})
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="mb-6">
        <button 
          onClick={handleAnalyze} 
          disabled={loading}
          className="bg-blue-500 text-white px-4 py-2 rounded mr-2 disabled:opacity-50"
        >
          {loading ? 'Computing...' : 'Compute Shortest Paths from A'}
        </button>
        
        <button 
          onClick={handleClear}
          className="bg-gray-500 text-white px-4 py-2 rounded"
        >
          Clear Results
        </button>
      </div>

      {result && (
        <div className="bg-white border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Results</h3>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <h4 className="font-medium">Shortest Distances</h4>
              <ul className="text-sm">
                {Object.entries(result.distances).map(([nodeId, distance]) => (
                  <li key={nodeId} className="flex justify-between">
                    <span>To {nodes.find(n => n.id === nodeId)?.label}:</span>
                    <span className="font-mono">
                      {isFinite(distance) ? distance : '∞'}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium">Normalized Distances (0-1)</h4>
              <ul className="text-sm">
                {Object.entries(normalizeDistances(result)).map(([nodeId, distance]) => (
                  <li key={nodeId} className="flex justify-between">
                    <span>To {nodes.find(n => n.id === nodeId)?.label}:</span>
                    <span className="font-mono">
                      {isFinite(distance) ? distance.toFixed(3) : '∞'}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-sm text-gray-600">
            <div>
              <strong>Algorithm:</strong> {result.algorithm}
            </div>
            <div>
              <strong>Unreachable:</strong> {result.unreachableCount} nodes
            </div>
            <div>
              <strong>Time:</strong> {result.computationTime.toFixed(2)}ms
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default BasicUsageExample