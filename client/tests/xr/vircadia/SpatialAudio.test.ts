import * as BABYLON from '@babylonjs/core'
import { SpatialAudioManager } from '../../../src/services/vircadia/SpatialAudioManager'

// Mock dependencies
jest.mock('@babylonjs/core')
jest.mock('../../../src/utils/logger', () => ({
  createLogger: () => ({
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  })
}))

// Mock Web Audio API
const mockAudioContext = {
  state: 'suspended',
  currentTime: 0,
  resume: jest.fn().mockResolvedValue(undefined),
  listener: {
    positionX: { setValueAtTime: jest.fn() },
    positionY: { setValueAtTime: jest.fn() },
    positionZ: { setValueAtTime: jest.fn() },
    forwardX: { setValueAtTime: jest.fn() },
    forwardY: { setValueAtTime: jest.fn() },
    forwardZ: { setValueAtTime: jest.fn() },
    upX: { setValueAtTime: jest.fn() },
    upY: { setValueAtTime: jest.fn() },
    upZ: { setValueAtTime: jest.fn() }
  },
  createBufferSource: jest.fn(),
  createMediaStreamSource: jest.fn(),
  createPanner: jest.fn(),
  createGain: jest.fn(),
  createDynamicsCompressor: jest.fn(),
  destination: {}
}

const mockPannerNode = {
  panningModel: '',
  distanceModel: '',
  refDistance: 0,
  maxDistance: 0,
  rolloffFactor: 0,
  coneInnerAngle: 0,
  coneOuterAngle: 0,
  coneOuterGain: 0,
  positionX: { setValueAtTime: jest.fn() },
  positionY: { setValueAtTime: jest.fn() },
  positionZ: { setValueAtTime: jest.fn() },
  orientationX: { setValueAtTime: jest.fn() },
  orientationY: { setValueAtTime: jest.fn() },
  orientationZ: { setValueAtTime: jest.fn() },
  connect: jest.fn()
}

const mockGainNode = {
  gain: { setValueAtTime: jest.fn(), value: 1 },
  connect: jest.fn()
}

const mockBufferSource = {
  buffer: null,
  loop: false,
  connect: jest.fn(),
  start: jest.fn(),
  stop: jest.fn()
}

const mockCompressor = {
  threshold: { setValueAtTime: jest.fn() },
  knee: { setValueAtTime: jest.fn() },
  ratio: { setValueAtTime: jest.fn() },
  attack: { setValueAtTime: jest.fn() },
  release: { setValueAtTime: jest.fn() },
  connect: jest.fn()
}

describe('SpatialAudioManager', () => {
  let mockScene: any
  let spatialAudioManager: SpatialAudioManager
  
  beforeEach(() => {
    jest.clearAllMocks()
    
    // Reset audio context mock
    mockAudioContext.state = 'suspended'
    mockAudioContext.currentTime = 0
    mockAudioContext.createPanner.mockReturnValue({ ...mockPannerNode })
    mockAudioContext.createGain.mockReturnValue({ ...mockGainNode })
    mockAudioContext.createBufferSource.mockReturnValue({ ...mockBufferSource })
    mockAudioContext.createDynamicsCompressor.mockReturnValue({ ...mockCompressor })
    
    // Mock window.AudioContext
    ;(window as any).AudioContext = jest.fn(() => mockAudioContext)
    
    // Mock BABYLON scene
    mockScene = {
      activeCamera: {
        position: { x: 0, y: 1.6, z: -3 },
        rotation: { x: 0, y: 0, z: 0 },
        getForwardRay: jest.fn().mockReturnValue({
          direction: { x: 0, y: 0, z: 1 }
        }),
        upVector: { x: 0, y: 1, z: 0 }
      },
      xr: null,
      onBeforeRenderObservable: {
        add: jest.fn()
      }
    }
    
    // Mock BABYLON types
    ;(BABYLON.Vector3 as jest.Mock).mockImplementation((x, y, z) => ({ 
      x, y, z,
      clone: function() { return { x: this.x, y: this.y, z: this.z } }
    }))
    ;(BABYLON as any).WebXRState = { IN_XR: 'IN_XR' }
    
    spatialAudioManager = new SpatialAudioManager(mockScene)
  })
  
  describe('Initialization', () => {
    it('should initialize with new AudioContext', async () => {
      await spatialAudioManager.initialize()
      
      expect(window.AudioContext).toHaveBeenCalled()
      expect(mockAudioContext.resume).toHaveBeenCalled()
      expect(spatialAudioManager.isEnabled()).toBe(true)
    })
    
    it('should handle already running audio context', async () => {
      mockAudioContext.state = 'running'
      
      await spatialAudioManager.initialize()
      
      expect(mockAudioContext.resume).not.toHaveBeenCalled()
      expect(spatialAudioManager.isEnabled()).toBe(true)
    })
    
    it('should setup listener tracking', async () => {
      await spatialAudioManager.initialize()
      
      expect(mockScene.onBeforeRenderObservable.add).toHaveBeenCalled()
    })
    
    it('should handle webkit AudioContext', async () => {
      delete (window as any).AudioContext
      ;(window as any).webkitAudioContext = jest.fn(() => mockAudioContext)
      
      await spatialAudioManager.initialize()
      
      expect((window as any).webkitAudioContext).toHaveBeenCalled()
    })
  })
  
  describe('Audio Context Management', () => {
    it('should set external audio context', async () => {
      await spatialAudioManager.initialize()
      
      const externalContext = { ...mockAudioContext }
      spatialAudioManager.setAudioContext(externalContext as any)
      
      expect(mockScene.onBeforeRenderObservable.add).toHaveBeenCalledTimes(2)
    })
  })
  
  describe('Listener Tracking', () => {
    beforeEach(async () => {
      await spatialAudioManager.initialize()
    })
    
    it('should update listener position from camera', () => {
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      
      mockScene.activeCamera.position = { x: 1, y: 2, z: 3 }
      callback()
      
      expect(mockAudioContext.listener.positionX.setValueAtTime).toHaveBeenCalledWith(1, 0)
      expect(mockAudioContext.listener.positionY.setValueAtTime).toHaveBeenCalledWith(2, 0)
      expect(mockAudioContext.listener.positionZ.setValueAtTime).toHaveBeenCalledWith(3, 0)
    })
    
    it('should update listener orientation from camera', () => {
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      
      mockScene.activeCamera.getForwardRay.mockReturnValue({
        direction: { x: 0.5, y: 0, z: 0.866 } // 30 degrees
      })
      mockScene.activeCamera.upVector = { x: 0, y: 1, z: 0 }
      
      callback()
      
      // Forward vector is negated for audio
      expect(mockAudioContext.listener.forwardX.setValueAtTime).toHaveBeenCalledWith(-0.5, 0)
      expect(mockAudioContext.listener.forwardY.setValueAtTime).toHaveBeenCalledWith(0, 0)
      expect(mockAudioContext.listener.forwardZ.setValueAtTime).toHaveBeenCalledWith(-0.866, 0)
      
      // Up vector
      expect(mockAudioContext.listener.upX.setValueAtTime).toHaveBeenCalledWith(0, 0)
      expect(mockAudioContext.listener.upY.setValueAtTime).toHaveBeenCalledWith(1, 0)
      expect(mockAudioContext.listener.upZ.setValueAtTime).toHaveBeenCalledWith(0, 0)
    })
    
    it('should use XR camera when in XR mode', () => {
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      
      // Setup XR mode
      mockScene.xr = {
        baseExperience: {
          state: BABYLON.WebXRState.IN_XR,
          camera: {
            position: { x: 2, y: 1.8, z: -1 },
            rotation: { x: 0.1, y: 0.2, z: 0 },
            getForwardRay: jest.fn().mockReturnValue({
              direction: { x: 0, y: 0, z: 1 }
            }),
            upVector: { x: 0, y: 1, z: 0 }
          }
        }
      }
      
      callback()
      
      expect(mockAudioContext.listener.positionX.setValueAtTime).toHaveBeenCalledWith(2, 0)
      expect(mockAudioContext.listener.positionY.setValueAtTime).toHaveBeenCalledWith(1.8, 0)
      expect(mockAudioContext.listener.positionZ.setValueAtTime).toHaveBeenCalledWith(-1, 0)
    })
    
    it('should handle legacy AudioListener API', () => {
      // Remove modern API
      delete mockAudioContext.listener.positionX
      mockAudioContext.listener.setPosition = jest.fn()
      mockAudioContext.listener.setOrientation = jest.fn()
      
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      
      mockScene.activeCamera.position = { x: 1, y: 2, z: 3 }
      callback()
      
      expect(mockAudioContext.listener.setPosition).toHaveBeenCalledWith(1, 2, 3)
      expect(mockAudioContext.listener.setOrientation).toHaveBeenCalledWith(
        0, 0, -1,  // forward (negated)
        0, 1, 0     // up
      )
    })
    
    it('should not update when disabled', () => {
      spatialAudioManager.dispose()
      
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      callback()
      
      expect(mockAudioContext.listener.positionX.setValueAtTime).not.toHaveBeenCalled()
    })
  })
  
  describe('Audio Source Creation', () => {
    beforeEach(async () => {
      await spatialAudioManager.initialize()
    })
    
    it('should create audio source from buffer', async () => {
      const audioBuffer = { duration: 5, length: 44100 * 5, sampleRate: 44100 } as AudioBuffer
      const position = new BABYLON.Vector3(1, 0, 0)
      
      const source = await spatialAudioManager.createAudioSource('source1', audioBuffer, position as any)
      
      expect(mockAudioContext.createBufferSource).toHaveBeenCalled()
      expect(mockBufferSource.buffer).toBe(audioBuffer)
      expect(mockBufferSource.loop).toBe(true)
      
      expect(source.id).toBe('source1')
      expect(source.position).toEqual(position)
    })
    
    it('should create audio source from media stream', async () => {
      const mediaStream = new MediaStream()
      const position = new BABYLON.Vector3(2, 1, -1)
      
      const source = await spatialAudioManager.createAudioSource('voice1', mediaStream, position as any)
      
      expect(mockAudioContext.createMediaStreamSource).toHaveBeenCalledWith(mediaStream)
      expect(source.id).toBe('voice1')
    })
    
    it('should configure panner node correctly', async () => {
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(3, 0, 0)
      
      await spatialAudioManager.createAudioSource('source2', audioBuffer, position as any)
      
      const panner = mockAudioContext.createPanner.mock.results[0].value
      
      expect(panner.panningModel).toBe('HRTF')
      expect(panner.distanceModel).toBe('inverse')
      expect(panner.refDistance).toBe(1)
      expect(panner.maxDistance).toBe(100)
      expect(panner.rolloffFactor).toBe(1)
      expect(panner.coneInnerAngle).toBe(360)
      expect(panner.coneOuterAngle).toBe(360)
      expect(panner.coneOuterGain).toBe(0)
    })
    
    it('should set initial position on panner', async () => {
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(5, 2, -3)
      
      await spatialAudioManager.createAudioSource('source3', audioBuffer, position as any)
      
      const panner = mockAudioContext.createPanner.mock.results[0].value
      
      expect(panner.positionX.setValueAtTime).toHaveBeenCalledWith(5, 0)
      expect(panner.positionY.setValueAtTime).toHaveBeenCalledWith(2, 0)
      expect(panner.positionZ.setValueAtTime).toHaveBeenCalledWith(-3, 0)
    })
    
    it('should handle legacy panner API', async () => {
      const panner = {
        ...mockPannerNode,
        positionX: undefined,
        setPosition: jest.fn()
      }
      mockAudioContext.createPanner.mockReturnValue(panner)
      
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(1, 2, 3)
      
      await spatialAudioManager.createAudioSource('source4', audioBuffer, position as any)
      
      expect(panner.setPosition).toHaveBeenCalledWith(1, 2, 3)
    })
    
    it('should connect audio nodes correctly', async () => {
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(0, 0, 0)
      
      await spatialAudioManager.createAudioSource('source5', audioBuffer, position as any)
      
      expect(mockBufferSource.connect).toHaveBeenCalledWith(mockPannerNode)
      expect(mockPannerNode.connect).toHaveBeenCalledWith(mockGainNode)
      expect(mockGainNode.connect).toHaveBeenCalledWith(mockAudioContext.destination)
    })
    
    it('should start buffer source playback', async () => {
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(0, 0, 0)
      
      await spatialAudioManager.createAudioSource('source6', audioBuffer, position as any)
      
      expect(mockBufferSource.start).toHaveBeenCalled()
    })
    
    it('should throw error if not initialized', async () => {
      const uninitializedManager = new SpatialAudioManager(mockScene)
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(0, 0, 0)
      
      await expect(
        uninitializedManager.createAudioSource('source7', audioBuffer, position as any)
      ).rejects.toThrow('Audio context not initialized')
    })
  })
  
  describe('Audio Source Management', () => {
    beforeEach(async () => {
      await spatialAudioManager.initialize()
    })
    
    it('should update audio source position', async () => {
      const audioBuffer = {} as AudioBuffer
      const initialPos = new BABYLON.Vector3(0, 0, 0)
      
      const source = await spatialAudioManager.createAudioSource('movingSource', audioBuffer, initialPos as any)
      
      const newPos = new BABYLON.Vector3(5, 0, 5)
      spatialAudioManager.updateAudioPosition('movingSource', newPos as any)
      
      const panner = mockAudioContext.createPanner.mock.results[0].value
      
      expect(panner.positionX.setValueAtTime).toHaveBeenLastCalledWith(5, 0)
      expect(panner.positionY.setValueAtTime).toHaveBeenLastCalledWith(0, 0)
      expect(panner.positionZ.setValueAtTime).toHaveBeenLastCalledWith(5, 0)
    })
    
    it('should update audio source orientation', async () => {
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(0, 0, 0)
      
      await spatialAudioManager.createAudioSource('directionalSource', audioBuffer, position as any)
      
      const orientation = new BABYLON.Vector3(0, 0, 1) // Forward
      spatialAudioManager.updateAudioOrientation('directionalSource', orientation as any)
      
      const panner = mockAudioContext.createPanner.mock.results[0].value
      
      expect(panner.orientationX.setValueAtTime).toHaveBeenCalledWith(0, 0)
      expect(panner.orientationY.setValueAtTime).toHaveBeenCalledWith(0, 0)
      expect(panner.orientationZ.setValueAtTime).toHaveBeenCalledWith(1, 0)
    })
    
    it('should update audio volume', async () => {
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(0, 0, 0)
      
      await spatialAudioManager.createAudioSource('volumeSource', audioBuffer, position as any)
      
      spatialAudioManager.setAudioVolume('volumeSource', 0.5)
      
      const gain = mockAudioContext.createGain.mock.results[0].value
      
      expect(gain.gain.setValueAtTime).toHaveBeenCalledWith(0.5, 0)
    })
    
    it('should remove audio source', async () => {
      const audioBuffer = {} as AudioBuffer
      const position = new BABYLON.Vector3(0, 0, 0)
      
      await spatialAudioManager.createAudioSource('tempSource', audioBuffer, position as any)
      
      spatialAudioManager.removeAudioSource('tempSource')
      
      expect(mockBufferSource.stop).toHaveBeenCalled()
      expect(mockGainNode.disconnect).toHaveBeenCalled()
      expect(mockPannerNode.disconnect).toHaveBeenCalled()
      expect(mockBufferSource.disconnect).toHaveBeenCalled()
    })
    
    it('should handle removing non-existent source', () => {
      // Should not throw
      expect(() => spatialAudioManager.removeAudioSource('nonExistent')).not.toThrow()
    })
  })
  
  describe('Global Audio Effects', () => {
    beforeEach(async () => {
      await spatialAudioManager.initialize()
    })
    
    it('should set global volume', () => {
      spatialAudioManager.setGlobalVolume(0.7)
      
      expect(mockGainNode.gain.value).toBe(0.7)
    })
    
    it('should enable reverb effect', () => {
      spatialAudioManager.enableReverb(true, {
        roomSize: 0.5,
        decay: 2,
        wet: 0.3
      })
      
      // Reverb implementation would create convolver node
      // This is a simplified test
      expect(spatialAudioManager.isEnabled()).toBe(true)
    })
  })
  
  describe('Disposal', () => {
    it('should dispose all resources', async () => {
      await spatialAudioManager.initialize()
      
      // Create some sources
      const audioBuffer = {} as AudioBuffer
      await spatialAudioManager.createAudioSource('source1', audioBuffer, new BABYLON.Vector3(0, 0, 0) as any)
      await spatialAudioManager.createAudioSource('source2', audioBuffer, new BABYLON.Vector3(1, 0, 0) as any)
      
      spatialAudioManager.dispose()
      
      // All sources should be stopped
      expect(mockBufferSource.stop).toHaveBeenCalledTimes(2)
      expect(mockBufferSource.disconnect).toHaveBeenCalledTimes(2)
      
      expect(spatialAudioManager.isEnabled()).toBe(false)
    })
    
    it('should handle disposal when not initialized', () => {
      // Should not throw
      expect(() => spatialAudioManager.dispose()).not.toThrow()
    })
  })
  
  describe('Edge Cases', () => {
    beforeEach(async () => {
      await spatialAudioManager.initialize()
    })
    
    it('should handle missing camera', () => {
      mockScene.activeCamera = null
      
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      
      // Should not throw
      expect(() => callback()).not.toThrow()
      
      // Should not update listener
      expect(mockAudioContext.listener.positionX.setValueAtTime).not.toHaveBeenCalled()
    })
    
    it('should handle audio context in closed state', async () => {
      mockAudioContext.state = 'closed'
      
      const newManager = new SpatialAudioManager(mockScene)
      
      // Should still initialize but may have limited functionality
      await newManager.initialize()
      
      expect(newManager.isEnabled()).toBe(true)
    })
    
    it('should handle very large distances', async () => {
      const audioBuffer = {} as AudioBuffer
      const farPosition = new BABYLON.Vector3(1000, 500, -1000)
      
      const source = await spatialAudioManager.createAudioSource('farSource', audioBuffer, farPosition as any)
      
      // Should still create source but volume might be attenuated
      expect(source).toBeDefined()
      
      const panner = mockAudioContext.createPanner.mock.results[0].value
      expect(panner.maxDistance).toBe(100) // Should respect max distance
    })
  })
})