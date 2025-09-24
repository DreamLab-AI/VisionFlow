import React, { useState, useEffect } from 'react'
import VircadiaScene from '../components/VircadiaScene'
import { useSettingsStore } from '../store/settingsStore'
import { createLogger } from '../utils/loggerConfig'

const logger = createLogger('VircadiaXRExample')

// Sample graph data for demonstration
const sampleGraphData = {
  nodes: [
    { id: 'node1', label: 'Central Hub', x: 0, y: 0, z: 0, color: '#FF6B6B', size: 0.1 },
    { id: 'node2', label: 'Data Node A', x: 1, y: 0.5, z: 0, color: '#4ECDC4', size: 0.08 },
    { id: 'node3', label: 'Data Node B', x: -1, y: 0.5, z: 0, color: '#45B7D1', size: 0.08 },
    { id: 'node4', label: 'Process X', x: 0, y: 1, z: 1, color: '#96CEB4', size: 0.07 },
    { id: 'node5', label: 'Process Y', x: 0, y: 1, z: -1, color: '#FECA57', size: 0.07 },
    { id: 'node6', label: 'Output 1', x: 1.5, y: -0.5, z: 0.5, color: '#FF8B94', size: 0.06 },
    { id: 'node7', label: 'Output 2', x: -1.5, y: -0.5, z: 0.5, color: '#A8E6CF', size: 0.06 },
    { id: 'node8', label: 'Storage', x: 0, y: -1, z: 0, color: '#FFD93D', size: 0.09 }
  ],
  edges: [
    { id: 'edge1', source: 'node1', target: 'node2', color: '#888888' },
    { id: 'edge2', source: 'node1', target: 'node3', color: '#888888' },
    { id: 'edge3', source: 'node1', target: 'node4', color: '#888888' },
    { id: 'edge4', source: 'node1', target: 'node5', color: '#888888' },
    { id: 'edge5', source: 'node2', target: 'node6', color: '#4ECDC4' },
    { id: 'edge6', source: 'node3', target: 'node7', color: '#45B7D1' },
    { id: 'edge7', source: 'node4', target: 'node8', color: '#96CEB4' },
    { id: 'edge8', source: 'node5', target: 'node8', color: '#FECA57' },
    { id: 'edge9', source: 'node6', target: 'node8', color: '#FF8B94' },
    { id: 'edge10', source: 'node7', target: 'node8', color: '#A8E6CF' }
  ]
}

export const VircadiaXRExample: React.FC = () => {
  const settings = useSettingsStore(state => state.settings)
  const updateSettings = useSettingsStore(state => state.updateSettings)
  
  const [sceneReady, setSceneReady] = useState(false)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [showPhysics, setShowPhysics] = useState(true)
  
  // Enable Vircadia XR mode in settings
  useEffect(() => {
    updateSettings({
      xr: {
        ...settings?.xr,
        enabled: true,
        mode: 'vircadia',
        enableMultiUser: true,
        enableSpatialAudio: true,
        graphScale: 0.8
      }
    })
    
    return () => {
      // Reset to default XR mode on unmount
      updateSettings({
        xr: {
          ...settings?.xr,
          mode: 'threejs'
        }
      })
    }
  }, [])
  
  const handleSceneReady = (scene: any) => {
    setSceneReady(true)
    logger.info('Vircadia example scene ready')
    
    // Setup click handler for nodes
    scene.onPointerObservable.add((pointerInfo: any) => {
      if (pointerInfo.type === 'BABYLON.PointerEventTypes.POINTERPICK') {
        if (pointerInfo.pickInfo.hit) {
          const mesh = pointerInfo.pickInfo.pickedMesh
          if (mesh && mesh.name.startsWith('node_')) {
            const nodeId = mesh.metadata?.id
            setSelectedNode(nodeId)
            logger.info('Node selected in example:', nodeId)
          }
        }
      }
    })
  }
  
  return (
    <div className="vircadia-xr-example">
      <h2>Vircadia XR Example</h2>
      <p>This example demonstrates the Vircadia-based XR implementation with Babylon.js</p>
      
      <div className="example-controls">
        <button 
          onClick={() => setShowPhysics(!showPhysics)}
          className="control-button"
        >
          {showPhysics ? 'Stop' : 'Start'} Physics
        </button>
        
        <button 
          onClick={() => setSelectedNode(null)}
          className="control-button"
          disabled={!selectedNode}
        >
          Clear Selection
        </button>
        
        <div className="status-info">
          <span>Scene Ready: {sceneReady ? '✅' : '⏳'}</span>
          <span>Selected Node: {selectedNode || 'None'}</span>
        </div>
      </div>
      
      <div className="scene-container">
        <VircadiaScene
          graphData={sampleGraphData}
          onReady={handleSceneReady}
          className="example-scene"
        />
      </div>
      
      {selectedNode && (
        <div className="node-info">
          <h3>Selected Node Info</h3>
          <pre>{JSON.stringify(
            sampleGraphData.nodes.find(n => n.id === selectedNode),
            null,
            2
          )}</pre>
        </div>
      )}
      
      <div className="features-list">
        <h3>Vircadia XR Features:</h3>
        <ul>
          <li>✅ Babylon.js-based rendering</li>
          <li>✅ WebXR support for Quest 3</li>
          <li>✅ Force-directed graph physics</li>
          <li>✅ Multi-user session support</li>
          <li>✅ Spatial audio integration</li>
          <li>✅ Hand tracking support</li>
          <li>✅ AR passthrough mode</li>
          <li>✅ Performance optimizations</li>
        </ul>
      </div>
      
      <style jsx>{`
        .vircadia-xr-example {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        h2 {
          color: #4ECDC4;
          margin-bottom: 10px;
        }
        
        .example-controls {
          display: flex;
          gap: 10px;
          align-items: center;
          margin: 20px 0;
          padding: 15px;
          background: #f5f5f5;
          border-radius: 8px;
        }
        
        .control-button {
          padding: 8px 16px;
          background: #4ECDC4;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.3s ease;
        }
        
        .control-button:hover:not(:disabled) {
          background: #3eb8ae;
          transform: translateY(-1px);
        }
        
        .control-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .status-info {
          margin-left: auto;
          display: flex;
          gap: 20px;
          font-size: 14px;
        }
        
        .scene-container {
          width: 100%;
          height: 600px;
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          overflow: hidden;
          margin: 20px 0;
        }
        
        .example-scene {
          width: 100%;
          height: 100%;
        }
        
        .node-info {
          background: #f9f9f9;
          padding: 15px;
          border-radius: 8px;
          margin: 20px 0;
        }
        
        .node-info h3 {
          margin-top: 0;
          color: #333;
        }
        
        .node-info pre {
          background: white;
          padding: 10px;
          border-radius: 4px;
          overflow-x: auto;
        }
        
        .features-list {
          background: #f0f8f7;
          padding: 20px;
          border-radius: 8px;
          margin-top: 30px;
        }
        
        .features-list h3 {
          color: #2a7a75;
          margin-top: 0;
        }
        
        .features-list ul {
          list-style: none;
          padding: 0;
          margin: 10px 0 0 0;
        }
        
        .features-list li {
          padding: 5px 0;
          color: #555;
        }
      `}</style>
    </div>
  )
}

export default VircadiaXRExample