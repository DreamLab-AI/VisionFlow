import * as BABYLON from '@babylonjs/core'
import { createLogger } from '../../utils/logger'
import { MultiUserManager } from './MultiUserManager'
import { SpatialAudioManager } from './SpatialAudioManager'
import { GraphVisualizationManager } from './GraphVisualizationManager'

const logger = createLogger('VircadiaService')

interface VircadiaConfig {
  enableMultiUser?: boolean
  enableSpatialAudio?: boolean
  graphScale?: number
}

export class VircadiaService {
  private scene: BABYLON.Scene
  private engine: BABYLON.Engine
  private multiUserManager?: MultiUserManager
  private spatialAudioManager?: SpatialAudioManager
  private graphVisualizationManager: GraphVisualizationManager
  private isInitialized = false
  
  constructor(scene: BABYLON.Scene, engine: BABYLON.Engine) {
    this.scene = scene
    this.engine = engine
    this.graphVisualizationManager = new GraphVisualizationManager(scene)
  }
  
  async initialize(config: VircadiaConfig = {}): Promise<void> {
    if (this.isInitialized) {
      logger.warn('VircadiaService already initialized')
      return
    }
    
    try {
      logger.info('Initializing VircadiaService with config:', config)
      
      // Initialize graph visualization
      await this.graphVisualizationManager.initialize({
        scale: config.graphScale || 0.8
      })
      
      // Initialize multi-user support if enabled
      if (config.enableMultiUser) {
        this.multiUserManager = new MultiUserManager(this.scene)
        await this.multiUserManager.initialize()
      }
      
      // Initialize spatial audio if enabled
      if (config.enableSpatialAudio) {
        this.spatialAudioManager = new SpatialAudioManager(this.scene)
        await this.spatialAudioManager.initialize()
      }
      
      // Setup scene optimizations
      this.setupSceneOptimizations()
      
      this.isInitialized = true
      logger.info('VircadiaService initialized successfully')
      
    } catch (err) {
      logger.error('Failed to initialize VircadiaService:', err)
      throw err
    }
  }
  
  private setupSceneOptimizations(): void {
    // Enable scene optimizations for Quest 3
    this.scene.autoClear = false
    this.scene.autoClearDepthAndStencil = false
    
    // Freeze materials that won't change
    this.scene.onReadyObservable.addOnce(() => {
      this.scene.freezeMaterials()
    })
    
    // Enable frustum culling
    this.scene.setRenderingAutoClearDepthStencil(0, true, true, true)
    
    // Optimize for mobile/XR
    const options = BABYLON.SceneOptimizerOptions.ModerateDegradationAllowed()
    const optimizer = new BABYLON.SceneOptimizer(this.scene, options)
    optimizer.start()
    
    logger.info('Scene optimizations applied')
  }
  
  async loadGraphData(graphData: any): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('VircadiaService not initialized')
    }
    
    try {
      logger.info('Loading graph data into Vircadia scene')
      await this.graphVisualizationManager.loadGraphData(graphData)
    } catch (err) {
      logger.error('Failed to load graph data:', err)
      throw err
    }
  }
  
  async updateGraphData(graphData: any): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('VircadiaService not initialized')
    }
    
    try {
      await this.graphVisualizationManager.updateGraphData(graphData)
    } catch (err) {
      logger.error('Failed to update graph data:', err)
      throw err
    }
  }
  
  async connectMultiUser(multiUserState: any): Promise<void> {
    if (!this.multiUserManager) {
      logger.warn('Multi-user not enabled')
      return
    }
    
    try {
      await this.multiUserManager.connect(multiUserState)
    } catch (err) {
      logger.error('Failed to connect multi-user:', err)
      throw err
    }
  }
  
  updateMultiUserState(multiUserState: any): void {
    if (!this.multiUserManager) return
    
    this.multiUserManager.updateState(multiUserState)
  }
  
  enableSpatialAudio(audioContext: AudioContext): void {
    if (!this.spatialAudioManager) {
      logger.warn('Spatial audio not enabled')
      return
    }
    
    this.spatialAudioManager.setAudioContext(audioContext)
  }
  
  // Force-directed graph physics
  startGraphPhysics(): void {
    this.graphVisualizationManager.startPhysicsSimulation()
  }
  
  stopGraphPhysics(): void {
    this.graphVisualizationManager.stopPhysicsSimulation()
  }
  
  // Node selection
  selectNode(nodeId: string): void {
    this.graphVisualizationManager.selectNode(nodeId)
  }
  
  deselectAllNodes(): void {
    this.graphVisualizationManager.deselectAllNodes()
  }
  
  // Camera utilities
  focusOnNode(nodeId: string): void {
    this.graphVisualizationManager.focusOnNode(nodeId)
  }
  
  resetCameraView(): void {
    const camera = this.scene.activeCamera
    if (camera && camera instanceof BABYLON.UniversalCamera) {
      camera.position = new BABYLON.Vector3(0, 1.6, -3)
      camera.setTarget(BABYLON.Vector3.Zero())
    }
  }
  
  dispose(): void {
    logger.info('Disposing VircadiaService')
    
    if (this.multiUserManager) {
      this.multiUserManager.dispose()
    }
    
    if (this.spatialAudioManager) {
      this.spatialAudioManager.dispose()
    }
    
    this.graphVisualizationManager.dispose()
    
    this.isInitialized = false
  }
  
  // Debug utilities
  getDebugInfo(): any {
    return {
      isInitialized: this.isInitialized,
      sceneStats: {
        totalMeshes: this.scene.meshes.length,
        activeMeshes: this.scene.getActiveMeshes().length,
        totalVertices: this.scene.getTotalVertices(),
        fps: this.engine.getFps()
      },
      graphStats: this.graphVisualizationManager.getStats(),
      multiUserConnected: this.multiUserManager?.isConnected() || false,
      spatialAudioEnabled: this.spatialAudioManager?.isEnabled() || false
    }
  }
}