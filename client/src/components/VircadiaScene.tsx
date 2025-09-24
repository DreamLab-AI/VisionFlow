import React, { useEffect, useRef, useState } from 'react'
import * as BABYLON from '@babylonjs/core'
import '@babylonjs/loaders'
import { useSettingsStore } from '../store/settingsStore'
import { useMultiUserStore } from '../store/multiUserStore'
import { createLogger } from '../utils/loggerConfig'
import { useVircadiaXR } from '../hooks/useVircadiaXR'
import { VircadiaService } from '../services/vircadia/VircadiaService'

const logger = createLogger('VircadiaScene')

interface VircadiaSceneProps {
  graphData?: any
  className?: string
  onReady?: (scene: BABYLON.Scene) => void
}

export const VircadiaScene: React.FC<VircadiaSceneProps> = ({ 
  graphData, 
  className,
  onReady 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<BABYLON.Engine | null>(null)
  const sceneRef = useRef<BABYLON.Scene | null>(null)
  const vircadiaServiceRef = useRef<VircadiaService | null>(null)
  
  const settings = useSettingsStore(state => state.settings)
  const multiUserState = useMultiUserStore(state => state)
  const { setupXR, enterXR, exitXR, isXRSupported, isInXR } = useVircadiaXR()
  
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Initialize Babylon.js scene
  useEffect(() => {
    if (!canvasRef.current) return
    
    const initializeScene = async () => {
      try {
        setIsLoading(true)
        setError(null)
        
        // Create engine
        const engine = new BABYLON.Engine(canvasRef.current, true, {
          preserveDrawingBuffer: true,
          stencil: true,
          powerPreference: 'high-performance'
        })
        engineRef.current = engine
        
        // Create scene
        const scene = new BABYLON.Scene(engine)
        scene.clearColor = new BABYLON.Color4(0, 0, 0, 0) // Transparent for AR
        sceneRef.current = scene
        
        // Create Vircadia service
        const vircadiaService = new VircadiaService(scene, engine)
        vircadiaServiceRef.current = vircadiaService
        
        // Initialize basic scene setup
        await vircadiaService.initialize({
          enableMultiUser: settings?.xr?.enableMultiUser !== false,
          enableSpatialAudio: settings?.xr?.enableSpatialAudio !== false,
          graphScale: settings?.xr?.graphScale || 0.8
        })
        
        // Setup camera
        const camera = new BABYLON.UniversalCamera(
          'camera', 
          new BABYLON.Vector3(0, 1.6, -3), 
          scene
        )
        camera.setTarget(BABYLON.Vector3.Zero())
        camera.attachControl(canvasRef.current, true)
        
        // Setup lighting for AR/VR
        const ambientLight = new BABYLON.HemisphericLight(
          'ambient',
          new BABYLON.Vector3(0, 1, 0),
          scene
        )
        ambientLight.intensity = 0.7
        
        const directionalLight = new BABYLON.DirectionalLight(
          'directional',
          new BABYLON.Vector3(-1, -2, -1),
          scene
        )
        directionalLight.position = new BABYLON.Vector3(3, 10, 3)
        directionalLight.intensity = 1.2
        
        // Shadow generator
        const shadowGenerator = new BABYLON.ShadowGenerator(2048, directionalLight)
        shadowGenerator.useBlurExponentialShadowMap = true
        shadowGenerator.blurKernel = 32
        
        // AR shadow plane
        const shadowPlane = BABYLON.MeshBuilder.CreateGround(
          'shadowPlane',
          { width: 10, height: 10 },
          scene
        )
        shadowPlane.receiveShadows = true
        const shadowMaterial = new BABYLON.StandardMaterial('shadowMat', scene)
        shadowMaterial.alpha = 0.3
        shadowMaterial.backFaceCulling = false
        shadowPlane.material = shadowMaterial
        
        // Setup XR if supported
        if (await isXRSupported()) {
          await setupXR(scene, engine)
        }
        
        // Load graph data if provided
        if (graphData) {
          await vircadiaService.loadGraphData(graphData)
        }
        
        // Connect multi-user if enabled
        if (settings?.xr?.enableMultiUser) {
          await vircadiaService.connectMultiUser(multiUserState)
        }
        
        // Start render loop
        engine.runRenderLoop(() => {
          scene.render()
        })
        
        // Handle resize
        const handleResize = () => {
          engine.resize()
        }
        window.addEventListener('resize', handleResize)
        
        // Notify ready
        if (onReady) {
          onReady(scene)
        }
        
        setIsLoading(false)
        logger.info('Vircadia scene initialized successfully')
        
        // Cleanup
        return () => {
          window.removeEventListener('resize', handleResize)
          vircadiaService.dispose()
          scene.dispose()
          engine.dispose()
        }
        
      } catch (err) {
        logger.error('Failed to initialize Vircadia scene:', err)
        setError(err instanceof Error ? err.message : 'Failed to initialize scene')
        setIsLoading(false)
      }
    }
    
    initializeScene()
  }, [graphData, settings, multiUserState, setupXR, isXRSupported, onReady])
  
  // Handle XR mode toggle
  const handleXRToggle = async () => {
    if (!sceneRef.current || !engineRef.current) return
    
    try {
      if (isInXR) {
        await exitXR()
      } else {
        await enterXR(sceneRef.current, engineRef.current)
      }
    } catch (err) {
      logger.error('Failed to toggle XR mode:', err)
      setError('Failed to enter/exit XR mode')
    }
  }
  
  // Update graph data
  useEffect(() => {
    if (!vircadiaServiceRef.current || !graphData) return
    
    vircadiaServiceRef.current.updateGraphData(graphData)
      .catch(err => {
        logger.error('Failed to update graph data:', err)
      })
  }, [graphData])
  
  // Update multi-user state
  useEffect(() => {
    if (!vircadiaServiceRef.current || !settings?.xr?.enableMultiUser) return
    
    vircadiaServiceRef.current.updateMultiUserState(multiUserState)
  }, [multiUserState, settings?.xr?.enableMultiUser])
  
  // Store scene to memory for coordination
  useEffect(() => {
    if (!sceneRef.current || !vircadiaServiceRef.current) return
    
    const storeSceneInfo = async () => {
      try {
        await fetch('npx claude-flow@alpha hooks post-edit --file "VircadiaScene.tsx" --memory-key "hive/implementation/vircadia-scene"', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scene: 'initialized',
            hasXR: await isXRSupported(),
            multiUserEnabled: settings?.xr?.enableMultiUser !== false
          })
        })
      } catch (err) {
        logger.warn('Failed to store scene info to memory:', err)
      }
    }
    
    storeSceneInfo()
  }, [isXRSupported, settings?.xr?.enableMultiUser])
  
  return (
    <div className={`vircadia-scene-container ${className || ''}`}>
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner">Loading Vircadia Scene...</div>
        </div>
      )}
      
      {error && (
        <div className="error-overlay">
          <div className="error-message">Error: {error}</div>
        </div>
      )}
      
      <canvas 
        ref={canvasRef}
        className="vircadia-canvas"
        style={{ 
          width: '100%', 
          height: '100%',
          display: 'block',
          touchAction: 'none'
        }}
      />
      
      {/* XR Controls */}
      {!isLoading && !error && (
        <div className="xr-controls">
          <button 
            onClick={handleXRToggle}
            className="xr-toggle-button"
            disabled={!isXRSupported}
          >
            {isInXR ? 'Exit XR' : 'Enter XR'}
          </button>
        </div>
      )}
      
      <style jsx>{`
        .vircadia-scene-container {
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
        }
        
        .loading-overlay,
        .error-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(0, 0, 0, 0.8);
          color: white;
          z-index: 10;
        }
        
        .loading-spinner {
          font-size: 1.2rem;
          animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        
        .error-message {
          color: #ff4444;
          font-size: 1.1rem;
        }
        
        .xr-controls {
          position: absolute;
          bottom: 20px;
          right: 20px;
          z-index: 5;
        }
        
        .xr-toggle-button {
          background: #4ECDC4;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 6px;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
        }
        
        .xr-toggle-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(78, 205, 196, 0.4);
        }
        
        .xr-toggle-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .vircadia-canvas {
          touch-action: none;
          -webkit-touch-callout: none;
          -webkit-user-select: none;
          user-select: none;
        }
      `}</style>
    </div>
  )
}

export default VircadiaScene