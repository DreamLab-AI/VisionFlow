import * as BABYLON from '@babylonjs/core'
import { createLogger } from '../../utils/loggerConfig'

const logger = createLogger('SpatialAudioManager')

interface AudioSource {
  id: string
  node: AudioNode
  panner: PannerNode
  position: BABYLON.Vector3
  gain: GainNode
}

export class SpatialAudioManager {
  private scene: BABYLON.Scene
  private audioContext?: AudioContext
  private audioSources: Map<string, AudioSource> = new Map()
  private listenerPosition: BABYLON.Vector3 = new BABYLON.Vector3(0, 1.6, 0)
  private enabled = false
  
  constructor(scene: BABYLON.Scene) {
    this.scene = scene
  }
  
  async initialize(): Promise<void> {
    logger.info('Initializing SpatialAudioManager')
    
    // Create or get audio context
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      
      // Resume audio context if suspended
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume()
      }
    }
    
    // Setup listener orientation updates
    this.setupListenerTracking()
    
    this.enabled = true
    logger.info('SpatialAudioManager initialized')
  }
  
  setAudioContext(audioContext: AudioContext): void {
    this.audioContext = audioContext
    this.setupListenerTracking()
  }
  
  private setupListenerTracking(): void {
    if (!this.audioContext) return
    
    // Track camera position/rotation for audio listener
    this.scene.onBeforeRenderObservable.add(() => {
      if (!this.enabled || !this.audioContext) return
      
      const camera = this.scene.activeCamera
      if (!camera) return
      
      // Update listener position
      const listener = this.audioContext.listener
      
      // Check if we're in XR mode
      const xr = this.scene.xr
      let position = camera.position
      let forward = camera.getForwardRay().direction
      let up = camera.upVector
      
      if (xr && xr.baseExperience && xr.baseExperience.state === BABYLON.WebXRState.IN_XR) {
        const xrCamera = xr.baseExperience.camera
        if (xrCamera) {
          position = xrCamera.position
          forward = xrCamera.getForwardRay().direction
          up = xrCamera.upVector
        }
      }
      
      // Update AudioListener position
      if (listener.positionX) {
        // Modern API
        listener.positionX.setValueAtTime(position.x, this.audioContext.currentTime)
        listener.positionY.setValueAtTime(position.y, this.audioContext.currentTime)
        listener.positionZ.setValueAtTime(position.z, this.audioContext.currentTime)
        
        listener.forwardX.setValueAtTime(-forward.x, this.audioContext.currentTime)
        listener.forwardY.setValueAtTime(-forward.y, this.audioContext.currentTime)
        listener.forwardZ.setValueAtTime(-forward.z, this.audioContext.currentTime)
        
        listener.upX.setValueAtTime(up.x, this.audioContext.currentTime)
        listener.upY.setValueAtTime(up.y, this.audioContext.currentTime)
        listener.upZ.setValueAtTime(up.z, this.audioContext.currentTime)
      } else {
        // Legacy API
        listener.setPosition(position.x, position.y, position.z)
        listener.setOrientation(-forward.x, -forward.y, -forward.z, up.x, up.y, up.z)
      }
      
      this.listenerPosition = position.clone()
    })
  }
  
  async createAudioSource(
    id: string, 
    audioBuffer: AudioBuffer | MediaStream,
    position: BABYLON.Vector3
  ): Promise<AudioSource> {
    if (!this.audioContext) {
      throw new Error('Audio context not initialized')
    }
    
    // Create source node
    let sourceNode: AudioNode
    
    if (audioBuffer instanceof AudioBuffer) {
      const bufferSource = this.audioContext.createBufferSource()
      bufferSource.buffer = audioBuffer
      bufferSource.loop = true
      sourceNode = bufferSource
    } else {
      // MediaStream for voice chat
      sourceNode = this.audioContext.createMediaStreamSource(audioBuffer)
    }
    
    // Create panner node for 3D positioning
    const panner = this.audioContext.createPanner()
    panner.panningModel = 'HRTF'
    panner.distanceModel = 'inverse'
    panner.refDistance = 1
    panner.maxDistance = 100
    panner.rolloffFactor = 1
    panner.coneInnerAngle = 360
    panner.coneOuterAngle = 360
    panner.coneOuterGain = 0
    
    // Set initial position
    if (panner.positionX) {
      panner.positionX.setValueAtTime(position.x, this.audioContext.currentTime)
      panner.positionY.setValueAtTime(position.y, this.audioContext.currentTime)
      panner.positionZ.setValueAtTime(position.z, this.audioContext.currentTime)
    } else {
      panner.setPosition(position.x, position.y, position.z)
    }
    
    // Create gain node for volume control
    const gain = this.audioContext.createGain()
    gain.gain.setValueAtTime(1, this.audioContext.currentTime)
    
    // Connect nodes
    sourceNode.connect(panner)
    panner.connect(gain)
    gain.connect(this.audioContext.destination)
    
    // Create audio source object
    const audioSource: AudioSource = {
      id,
      node: sourceNode,
      panner,
      position: position.clone(),
      gain
    }
    
    // Store and start if buffer source
    this.audioSources.set(id, audioSource)
    
    if (sourceNode instanceof AudioBufferSourceNode) {
      sourceNode.start()
    }
    
    logger.info(`Created audio source: ${id}`)
    return audioSource
  }
  
  updateAudioSourcePosition(id: string, position: BABYLON.Vector3): void {
    const audioSource = this.audioSources.get(id)
    if (!audioSource || !this.audioContext) return
    
    audioSource.position = position.clone()
    
    if (audioSource.panner.positionX) {
      audioSource.panner.positionX.setValueAtTime(position.x, this.audioContext.currentTime)
      audioSource.panner.positionY.setValueAtTime(position.y, this.audioContext.currentTime)
      audioSource.panner.positionZ.setValueAtTime(position.z, this.audioContext.currentTime)
    } else {
      audioSource.panner.setPosition(position.x, position.y, position.z)
    }
  }
  
  setAudioSourceVolume(id: string, volume: number): void {
    const audioSource = this.audioSources.get(id)
    if (!audioSource || !this.audioContext) return
    
    audioSource.gain.gain.setValueAtTime(
      Math.max(0, Math.min(1, volume)), 
      this.audioContext.currentTime
    )
  }
  
  removeAudioSource(id: string): void {
    const audioSource = this.audioSources.get(id)
    if (!audioSource) return
    
    // Stop and disconnect
    if (audioSource.node instanceof AudioBufferSourceNode) {
      audioSource.node.stop()
    }
    
    audioSource.node.disconnect()
    audioSource.panner.disconnect()
    audioSource.gain.disconnect()
    
    this.audioSources.delete(id)
    logger.info(`Removed audio source: ${id}`)
  }
  
  // Utility to load audio files
  async loadAudioFile(url: string): Promise<AudioBuffer> {
    if (!this.audioContext) {
      throw new Error('Audio context not initialized')
    }
    
    try {
      const response = await fetch(url)
      const arrayBuffer = await response.arrayBuffer()
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer)
      return audioBuffer
    } catch (err) {
      logger.error('Failed to load audio file:', err)
      throw err
    }
  }
  
  // Create ambient sound
  async createAmbientSound(id: string, url: string, volume: number = 0.5): Promise<void> {
    try {
      const audioBuffer = await this.loadAudioFile(url)
      const source = await this.createAudioSource(
        id, 
        audioBuffer, 
        this.listenerPosition
      )
      
      // Ambient sounds follow the listener
      this.scene.onBeforeRenderObservable.add(() => {
        this.updateAudioSourcePosition(id, this.listenerPosition)
      })
      
      this.setAudioSourceVolume(id, volume)
      
    } catch (err) {
      logger.error('Failed to create ambient sound:', err)
      throw err
    }
  }
  
  // Create positional sound attached to a mesh
  async createPositionalSound(
    id: string, 
    url: string, 
    mesh: BABYLON.Mesh,
    volume: number = 1
  ): Promise<void> {
    try {
      const audioBuffer = await this.loadAudioFile(url)
      const source = await this.createAudioSource(
        id, 
        audioBuffer, 
        mesh.position
      )
      
      // Update position with mesh
      this.scene.onBeforeRenderObservable.add(() => {
        this.updateAudioSourcePosition(id, mesh.position)
      })
      
      this.setAudioSourceVolume(id, volume)
      
    } catch (err) {
      logger.error('Failed to create positional sound:', err)
      throw err
    }
  }
  
  // Voice chat support
  async createVoiceChannel(
    userId: string, 
    mediaStream: MediaStream,
    position: BABYLON.Vector3
  ): Promise<void> {
    try {
      await this.createAudioSource(`voice_${userId}`, mediaStream, position)
      logger.info(`Created voice channel for user: ${userId}`)
    } catch (err) {
      logger.error('Failed to create voice channel:', err)
      throw err
    }
  }
  
  updateVoicePosition(userId: string, position: BABYLON.Vector3): void {
    this.updateAudioSourcePosition(`voice_${userId}`, position)
  }
  
  removeVoiceChannel(userId: string): void {
    this.removeAudioSource(`voice_${userId}`)
  }
  
  isEnabled(): boolean {
    return this.enabled
  }
  
  mute(): void {
    if (!this.audioContext) return
    this.audioContext.suspend()
  }
  
  unmute(): void {
    if (!this.audioContext) return
    this.audioContext.resume()
  }
  
  dispose(): void {
    logger.info('Disposing SpatialAudioManager')
    
    // Remove all audio sources
    this.audioSources.forEach((_, id) => this.removeAudioSource(id))
    
    // Suspend audio context
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.suspend()
    }
    
    this.audioSources.clear()
    this.enabled = false
  }
}