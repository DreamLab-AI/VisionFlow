import React, { useEffect, useRef, useState } from 'react'
import { useSettingsStore } from '../../../store/settingsStore'
import { useMultiUserStore } from '../../../store/multiUserStore'
import VircadiaScene from '../../../components/VircadiaScene'
import { createLogger } from '../../../utils/loggerConfig'

const logger = createLogger('VircadiaXRIntegration')

interface VircadiaXRIntegrationProps {
  graphData?: {
    nodes: Array<{
      id: string
      label?: string
      x: number
      y: number
      z: number
      color?: string
      size?: number
      metadata?: any
    }>
    edges: Array<{
      id: string
      source: string
      target: string
      weight?: number
      color?: string
    }>
  }
  onNodeSelect?: (nodeId: string) => void
  onNodeHover?: (nodeId: string | null) => void
}

export const VircadiaXRIntegration: React.FC<VircadiaXRIntegrationProps> = ({
  graphData,
  onNodeSelect,
  onNodeHover
}) => {
  const settings = useSettingsStore(state => state.settings)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const sceneRef = useRef<any>(null)
  
  // Transform graph data from the existing format
  const transformedGraphData = useRef<any>(null)
  
  useEffect(() => {
    if (!graphData) return
    
    // Transform nodes and edges to Vircadia format
    transformedGraphData.current = {
      nodes: graphData.nodes.map(node => ({
        ...node,
        // Ensure 3D positioning
        x: node.x || 0,
        y: node.y || 0,
        z: node.z || 0,
        color: node.color || settings?.graph?.nodeColor || '#4ECDC4',
        size: node.size || settings?.graph?.nodeSize || 0.05
      })),
      edges: graphData.edges.map(edge => ({
        ...edge,
        color: edge.color || settings?.graph?.edgeColor || '#666666'
      }))
    }
    
    logger.info('Transformed graph data for Vircadia', {
      nodeCount: transformedGraphData.current.nodes.length,
      edgeCount: transformedGraphData.current.edges.length
    })
  }, [graphData, settings])
  
  const handleSceneReady = (scene: any) => {
    sceneRef.current = scene
    setIsReady(true)
    
    // Setup node interaction handlers
    scene.onPointerObservable.add((pointerInfo: any) => {
      switch (pointerInfo.type) {
        case 'BABYLON.PointerEventTypes.POINTERPICK':
          if (pointerInfo.pickInfo.hit) {
            const mesh = pointerInfo.pickInfo.pickedMesh
            if (mesh && mesh.name.startsWith('node_')) {
              const nodeId = mesh.metadata?.id
              if (nodeId && onNodeSelect) {
                onNodeSelect(nodeId)
                logger.info('Node selected:', nodeId)
              }
            }
          }
          break
          
        case 'BABYLON.PointerEventTypes.POINTERMOVE':
          if (pointerInfo.pickInfo.hit) {
            const mesh = pointerInfo.pickInfo.pickedMesh
            if (mesh && mesh.name.startsWith('node_')) {
              const nodeId = mesh.metadata?.id
              if (nodeId && onNodeHover) {
                onNodeHover(nodeId)
              }
            }
          } else if (onNodeHover) {
            onNodeHover(null)
          }
          break
      }
    })
    
    logger.info('Vircadia scene ready and interaction handlers set up')
  }
  
  // Handle XR settings changes
  useEffect(() => {
    if (!settings?.xr || !sceneRef.current) return
    
    // Apply XR-specific settings
    if (settings.xr.graphScale) {
      // Update graph scale in scene
      logger.info('Updating graph scale:', settings.xr.graphScale)
    }
    
    if (settings.xr.enableOptimizedNodes) {
      // Enable/disable optimized rendering
      logger.info('Optimized nodes:', settings.xr.enableOptimizedNodes)
    }
  }, [settings?.xr])
  
  return (
    <div className="vircadia-xr-integration">
      <VircadiaScene
        graphData={transformedGraphData.current}
        onReady={handleSceneReady}
        className="vircadia-xr-canvas"
      />
      
      {/* Debug overlay */}
      {settings?.debug?.enabled && isReady && (
        <div className="vircadia-debug-overlay">
          <h4>Vircadia XR Debug</h4>
          <p>Scene Ready: {isReady ? 'Yes' : 'No'}</p>
          <p>Nodes: {transformedGraphData.current?.nodes?.length || 0}</p>
          <p>Edges: {transformedGraphData.current?.edges?.length || 0}</p>
          <p>Multi-User: {settings?.xr?.enableMultiUser ? 'Enabled' : 'Disabled'}</p>
          <p>Spatial Audio: {settings?.xr?.enableSpatialAudio ? 'Enabled' : 'Disabled'}</p>
        </div>
      )}
      
      {error && (
        <div className="vircadia-error">
          <p>Error: {error}</p>
        </div>
      )}
      
      <style jsx>{`
        .vircadia-xr-integration {
          position: relative;
          width: 100%;
          height: 100%;
        }
        
        .vircadia-xr-canvas {
          width: 100%;
          height: 100%;
        }
        
        .vircadia-debug-overlay {
          position: absolute;
          top: 10px;
          left: 10px;
          background: rgba(0, 0, 0, 0.8);
          color: white;
          padding: 10px;
          border-radius: 5px;
          font-size: 12px;
          z-index: 100;
        }
        
        .vircadia-debug-overlay h4 {
          margin: 0 0 10px 0;
          font-size: 14px;
          color: #4ECDC4;
        }
        
        .vircadia-debug-overlay p {
          margin: 2px 0;
        }
        
        .vircadia-error {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: rgba(255, 0, 0, 0.1);
          border: 2px solid #ff0000;
          color: #ff0000;
          padding: 20px;
          border-radius: 5px;
        }
      `}</style>
    </div>
  )
}

export default VircadiaXRIntegration