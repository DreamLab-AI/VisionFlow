import React from 'react'
import { render, waitFor, fireEvent, screen } from '@testing-library/react'
import '@testing-library/jest-dom'
import * as BABYLON from '@babylonjs/core'
import { VircadiaScene } from '../../../src/components/VircadiaScene'
import { useSettingsStore } from '../../../src/store/settingsStore'
import { useMultiUserStore } from '../../../src/store/multiUserStore'
import { useVircadiaXR } from '../../../src/hooks/useVircadiaXR'
import { VircadiaService } from '../../../src/services/vircadia/VircadiaService'

// Mock dependencies
jest.mock('@babylonjs/core')
jest.mock('../../../src/store/settingsStore')
jest.mock('../../../src/store/multiUserStore')
jest.mock('../../../src/hooks/useVircadiaXR')
jest.mock('../../../src/services/vircadia/VircadiaService')
jest.mock('../../../src/utils/logger', () => ({
  createLogger: () => ({
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  })
}))

describe('VircadiaScene', () => {
  let mockEngine: any
  let mockScene: any
  let mockVircadiaService: any
  let mockCanvas: HTMLCanvasElement
  
  const mockXRHook = {
    setupXR: jest.fn().mockResolvedValue(undefined),
    enterXR: jest.fn().mockResolvedValue(undefined),
    exitXR: jest.fn().mockResolvedValue(undefined),
    isXRSupported: jest.fn().mockResolvedValue(true),
    isInXR: false
  }
  
  const mockSettings = {
    xr: {
      enableMultiUser: true,
      enableSpatialAudio: true,
      graphScale: 0.8
    }
  }
  
  const mockMultiUserState = {
    users: {},
    localUserId: 'test-user',
    sendUpdate: jest.fn()
  }
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks()
    
    // Setup canvas mock
    mockCanvas = document.createElement('canvas')
    mockCanvas.getBoundingClientRect = jest.fn().mockReturnValue({
      width: 800,
      height: 600,
      top: 0,
      left: 0,
      right: 800,
      bottom: 600
    })
    
    // Setup BABYLON mocks
    mockScene = {
      clearColor: null,
      render: jest.fn(),
      dispose: jest.fn(),
      effectLayers: [],
      xr: null,
      onBeforeRenderObservable: { add: jest.fn() }
    }
    
    mockEngine = {
      runRenderLoop: jest.fn(),
      resize: jest.fn(),
      dispose: jest.fn(),
      getFps: jest.fn().mockReturnValue(60)
    }
    
    ;(BABYLON.Engine as jest.Mock).mockImplementation(() => mockEngine)
    ;(BABYLON.Scene as jest.Mock).mockImplementation(() => mockScene)
    ;(BABYLON.Color4 as jest.Mock).mockImplementation((r, g, b, a) => ({ r, g, b, a }))
    ;(BABYLON.Vector3 as any).Zero = jest.fn().mockReturnValue({ x: 0, y: 0, z: 0 })
    ;(BABYLON.Vector3 as jest.Mock).mockImplementation((x, y, z) => ({ x, y, z }))
    
    // Mock camera
    const mockCamera = {
      setTarget: jest.fn(),
      attachControl: jest.fn()
    }
    ;(BABYLON.UniversalCamera as jest.Mock).mockImplementation(() => mockCamera)
    
    // Mock lights
    ;(BABYLON.HemisphericLight as jest.Mock).mockImplementation(() => ({ intensity: 0 }))
    ;(BABYLON.DirectionalLight as jest.Mock).mockImplementation(() => ({ 
      intensity: 0,
      position: { x: 0, y: 0, z: 0 }
    }))
    
    // Mock shadow generator
    ;(BABYLON.ShadowGenerator as jest.Mock).mockImplementation(() => ({
      useBlurExponentialShadowMap: false,
      blurKernel: 0
    }))
    
    // Mock mesh builders
    ;(BABYLON.MeshBuilder as any) = {
      CreateGround: jest.fn().mockReturnValue({
        receiveShadows: false,
        material: null
      })
    }
    
    ;(BABYLON.StandardMaterial as jest.Mock).mockImplementation(() => ({
      alpha: 1,
      backFaceCulling: true
    }))
    
    // Mock VircadiaService
    mockVircadiaService = {
      initialize: jest.fn().mockResolvedValue(undefined),
      loadGraphData: jest.fn().mockResolvedValue(undefined),
      updateGraphData: jest.fn().mockResolvedValue(undefined),
      connectMultiUser: jest.fn().mockResolvedValue(undefined),
      updateMultiUserState: jest.fn(),
      dispose: jest.fn()
    }
    ;(VircadiaService as jest.Mock).mockImplementation(() => mockVircadiaService)
    
    // Mock store hooks
    ;(useSettingsStore as jest.Mock).mockReturnValue(mockSettings)
    ;(useMultiUserStore as jest.Mock).mockReturnValue(mockMultiUserState)
    ;(useVircadiaXR as jest.Mock).mockReturnValue(mockXRHook)
    
    // Mock fetch for memory storage
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({})
    })
  })
  
  afterEach(() => {
    jest.restoreAllMocks()
  })
  
  describe('Initialization', () => {
    it('should render without crashing', () => {
      const { container } = render(<VircadiaScene />)
      expect(container.querySelector('.vircadia-scene-container')).toBeInTheDocument()
    })
    
    it('should display loading spinner initially', () => {
      const { container } = render(<VircadiaScene />)
      expect(container.querySelector('.loading-spinner')).toBeInTheDocument()
    })
    
    it('should initialize Babylon.js engine and scene', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(BABYLON.Engine).toHaveBeenCalledWith(
          expect.any(HTMLCanvasElement),
          true,
          expect.objectContaining({
            preserveDrawingBuffer: true,
            stencil: true,
            powerPreference: 'high-performance'
          })
        )
        expect(BABYLON.Scene).toHaveBeenCalledWith(mockEngine)
      })
    })
    
    it('should initialize VircadiaService with correct config', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(VircadiaService).toHaveBeenCalledWith(mockScene, mockEngine)
        expect(mockVircadiaService.initialize).toHaveBeenCalledWith({
          enableMultiUser: true,
          enableSpatialAudio: true,
          graphScale: 0.8
        })
      })
    })
    
    it('should setup XR when supported', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockXRHook.isXRSupported).toHaveBeenCalled()
        expect(mockXRHook.setupXR).toHaveBeenCalledWith(mockScene, mockEngine)
      })
    })
    
    it('should handle XR not supported', async () => {
      mockXRHook.isXRSupported.mockResolvedValue(false)
      
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockXRHook.setupXR).not.toHaveBeenCalled()
      })
    })
    
    it('should call onReady callback when scene is ready', async () => {
      const onReady = jest.fn()
      
      render(<VircadiaScene onReady={onReady} />)
      
      await waitFor(() => {
        expect(onReady).toHaveBeenCalledWith(mockScene)
      })
    })
    
    it('should handle initialization errors', async () => {
      const error = new Error('Init failed')
      mockVircadiaService.initialize.mockRejectedValue(error)
      
      const { container } = render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(container.querySelector('.error-message')).toHaveTextContent('Init failed')
      })
    })
  })
  
  describe('Graph Data Loading', () => {
    const mockGraphData = {
      nodes: [
        { id: 'node1', x: 0, y: 0, z: 0 },
        { id: 'node2', x: 1, y: 0, z: 0 }
      ],
      edges: [
        { id: 'edge1', source: 'node1', target: 'node2' }
      ]
    }
    
    it('should load graph data on mount if provided', async () => {
      render(<VircadiaScene graphData={mockGraphData} />)
      
      await waitFor(() => {
        expect(mockVircadiaService.loadGraphData).toHaveBeenCalledWith(mockGraphData)
      })
    })
    
    it('should update graph data when prop changes', async () => {
      const { rerender } = render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockVircadiaService.loadGraphData).not.toHaveBeenCalled()
      })
      
      const newGraphData = { ...mockGraphData, nodes: [...mockGraphData.nodes] }
      rerender(<VircadiaScene graphData={newGraphData} />)
      
      await waitFor(() => {
        expect(mockVircadiaService.updateGraphData).toHaveBeenCalledWith(newGraphData)
      })
    })
    
    it('should handle graph data loading errors', async () => {
      mockVircadiaService.loadGraphData.mockRejectedValue(new Error('Load failed'))
      
      render(<VircadiaScene graphData={mockGraphData} />)
      
      await waitFor(() => {
        expect(mockVircadiaService.loadGraphData).toHaveBeenCalled()
        // Error should be logged but not crash the component
      })
    })
  })
  
  describe('Multi-User Integration', () => {
    it('should connect multi-user when enabled', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockVircadiaService.connectMultiUser).toHaveBeenCalledWith(mockMultiUserState)
      })
    })
    
    it('should not connect multi-user when disabled', async () => {
      ;(useSettingsStore as jest.Mock).mockReturnValue({
        settings: { xr: { enableMultiUser: false } }
      })
      
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockVircadiaService.initialize).toHaveBeenCalled()
      })
      
      expect(mockVircadiaService.connectMultiUser).not.toHaveBeenCalled()
    })
    
    it('should update multi-user state when store changes', async () => {
      const { rerender } = render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockVircadiaService.initialize).toHaveBeenCalled()
      })
      
      const newMultiUserState = {
        ...mockMultiUserState,
        users: { 'user2': { id: 'user2', position: [0, 0, 0] } }
      }
      ;(useMultiUserStore as jest.Mock).mockReturnValue(newMultiUserState)
      
      rerender(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockVircadiaService.updateMultiUserState).toHaveBeenCalledWith(newMultiUserState)
      })
    })
  })
  
  describe('XR Mode Controls', () => {
    it('should show XR toggle button when XR is supported', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(screen.getByText('Enter XR')).toBeInTheDocument()
      })
    })
    
    it('should disable XR button when not supported', async () => {
      mockXRHook.isXRSupported.mockResolvedValue(false)
      
      render(<VircadiaScene />)
      
      await waitFor(() => {
        const button = screen.getByText('Enter XR')
        expect(button).toBeDisabled()
      })
    })
    
    it('should enter XR mode on button click', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(screen.getByText('Enter XR')).toBeInTheDocument()
      })
      
      fireEvent.click(screen.getByText('Enter XR'))
      
      await waitFor(() => {
        expect(mockXRHook.enterXR).toHaveBeenCalledWith(mockScene, mockEngine)
      })
    })
    
    it('should exit XR mode when in XR', async () => {
      ;(useVircadiaXR as jest.Mock).mockReturnValue({
        ...mockXRHook,
        isInXR: true
      })
      
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(screen.getByText('Exit XR')).toBeInTheDocument()
      })
      
      fireEvent.click(screen.getByText('Exit XR'))
      
      expect(mockXRHook.exitXR).toHaveBeenCalled()
    })
    
    it('should handle XR toggle errors gracefully', async () => {
      mockXRHook.enterXR.mockRejectedValue(new Error('XR failed'))
      
      const { container } = render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(screen.getByText('Enter XR')).toBeInTheDocument()
      })
      
      fireEvent.click(screen.getByText('Enter XR'))
      
      await waitFor(() => {
        expect(container.querySelector('.error-message')).toHaveTextContent('Failed to enter/exit XR mode')
      })
    })
  })
  
  describe('Scene Lifecycle', () => {
    it('should start render loop after initialization', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockEngine.runRenderLoop).toHaveBeenCalledWith(expect.any(Function))
      })
      
      // Test render loop callback
      const renderLoopCallback = mockEngine.runRenderLoop.mock.calls[0][0]
      renderLoopCallback()
      expect(mockScene.render).toHaveBeenCalled()
    })
    
    it('should handle window resize', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockEngine.runRenderLoop).toHaveBeenCalled()
      })
      
      // Trigger resize event
      fireEvent.resize(window)
      
      expect(mockEngine.resize).toHaveBeenCalled()
    })
    
    it('should cleanup on unmount', async () => {
      const { unmount } = render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(mockVircadiaService.initialize).toHaveBeenCalled()
      })
      
      unmount()
      
      expect(mockVircadiaService.dispose).toHaveBeenCalled()
      expect(mockScene.dispose).toHaveBeenCalled()
      expect(mockEngine.dispose).toHaveBeenCalled()
    })
  })
  
  describe('Custom Styling', () => {
    it('should apply custom className', () => {
      const { container } = render(<VircadiaScene className="custom-scene" />)
      
      expect(container.querySelector('.vircadia-scene-container')).toHaveClass('custom-scene')
    })
    
    it('should apply correct canvas styles', () => {
      const { container } = render(<VircadiaScene />)
      
      const canvas = container.querySelector('canvas')
      expect(canvas).toHaveStyle({
        width: '100%',
        height: '100%',
        display: 'block',
        touchAction: 'none'
      })
    })
  })
  
  describe('Memory Coordination', () => {
    it('should store scene info to memory after initialization', async () => {
      render(<VircadiaScene />)
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('hooks post-edit'),
          expect.objectContaining({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: expect.stringContaining('"scene":"initialized"')
          })
        )
      })
    })
  })
})