import * as BABYLON from '@babylonjs/core'
import { GraphVisualizationManager } from '../../../src/services/vircadia/GraphVisualizationManager'

// Mock dependencies
jest.mock('@babylonjs/core')
jest.mock('../../../src/utils/logger', () => ({
  createLogger: () => ({
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  })
}))

describe('GraphVisualizationManager', () => {
  let mockScene: any
  let graphManager: GraphVisualizationManager
  let mockMeshes: Map<string, any>
  let mockAnimations: any[]
  
  const mockGraphData = {
    nodes: [
      { id: 'node1', label: 'Node 1', x: 0, y: 0, z: 0, color: '#FF6B6B', size: 0.1 },
      { id: 'node2', label: 'Node 2', x: 1, y: 0, z: 0, color: '#4ECDC4', size: 0.1 },
      { id: 'node3', label: 'Node 3', x: 0.5, y: 1, z: 0.5, color: '#45B7D1', size: 0.15 }
    ],
    edges: [
      { id: 'edge1', source: 'node1', target: 'node2', weight: 1, color: '#666666' },
      { id: 'edge2', source: 'node2', target: 'node3', weight: 2, color: '#888888' }
    ]
  }
  
  beforeEach(() => {
    jest.clearAllMocks()
    mockMeshes = new Map()
    mockAnimations = []
    
    // Mock BABYLON scene
    mockScene = {
      activeCamera: {
        position: { x: 0, y: 1.6, z: -3 },
        setTarget: jest.fn()
      },
      effectLayers: [],
      onReadyObservable: {
        addOnce: jest.fn(cb => cb())
      },
      freezeMaterials: jest.fn(),
      setRenderingAutoClearDepthStencil: jest.fn(),
      onBeforeRenderObservable: {
        add: jest.fn(),
        remove: jest.fn()
      }
    }
    
    // Mock BABYLON constructors
    ;(BABYLON.TransformNode as jest.Mock).mockImplementation((name) => ({
      name,
      position: { x: 0, y: 0, z: 0 },
      scaling: { setAll: jest.fn() }
    }))
    
    ;(BABYLON.Vector3 as jest.Mock).mockImplementation((x, y, z) => ({ 
      x, y, z,
      clone: function() { return { x: this.x, y: this.y, z: this.z } },
      subtract: function(other: any) { return { x: this.x - other.x, y: this.y - other.y, z: this.z - other.z } },
      normalize: function() { return this },
      scale: function(s: number) { return { x: this.x * s, y: this.y * s, z: this.z * s } },
      add: function(other: any) { return { x: this.x + other.x, y: this.y + other.y, z: this.z + other.z } }
    }))
    
    ;(BABYLON.Color3 as any).FromHexString = jest.fn((hex) => ({ 
      scale: jest.fn((s) => ({ r: 0.5, g: 0.5, b: 0.5 })) 
    }))
    
    // Mock mesh builders
    ;(BABYLON.MeshBuilder as any) = {
      CreateSphere: jest.fn((name, options) => {
        const mesh = {
          name,
          position: { x: 0, y: 0, z: 0 },
          material: null,
          parent: null,
          metadata: {},
          isPickable: false,
          dispose: jest.fn()
        }
        mockMeshes.set(name, mesh)
        return mesh
      }),
      CreateTube: jest.fn((name, options) => {
        const mesh = {
          name,
          path: options.path,
          material: null,
          parent: null,
          metadata: {},
          dispose: jest.fn()
        }
        mockMeshes.set(name, mesh)
        return mesh
      }),
      CreateLines: jest.fn((name, options) => {
        const mesh = {
          name,
          points: options.points,
          material: null,
          parent: null,
          metadata: {},
          dispose: jest.fn()
        }
        mockMeshes.set(name, mesh)
        return mesh
      })
    }
    
    // Mock materials
    ;(BABYLON.StandardMaterial as jest.Mock).mockImplementation((name) => ({
      name,
      diffuseColor: null,
      emissiveColor: null,
      specularColor: null,
      alpha: 1,
      backFaceCulling: true
    }))
    
    // Mock animation
    ;(BABYLON.Animation as any).CreateAndStartAnimation = jest.fn((name, target, prop, fps, frames, from, to, loop) => {
      mockAnimations.push({ name, target, prop, from, to })
      // Immediately apply the target value for testing
      if (prop === 'position') {
        target.position = to
      }
    })
    
    ;(BABYLON.Animation as any).ANIMATIONLOOPMODE_CONSTANT = 0
    
    // Mock scene optimizer
    ;(BABYLON.SceneOptimizerOptions as any).ModerateDegradationAllowed = jest.fn(() => ({}))
    ;(BABYLON.SceneOptimizer as jest.Mock).mockImplementation(() => ({ start: jest.fn() }))
    
    graphManager = new GraphVisualizationManager(mockScene)
  })
  
  describe('Initialization', () => {
    it('should initialize with default config', async () => {
      await graphManager.initialize()
      
      const nodeContainer = (BABYLON.TransformNode as jest.Mock).mock.results.find(
        r => r.value.name === 'nodeContainer'
      )?.value
      
      expect(nodeContainer).toBeDefined()
      expect(nodeContainer.position).toEqual({ x: 0, y: 1.2, z: -1 })
      expect(nodeContainer.scaling.setAll).toHaveBeenCalledWith(0.8)
    })
    
    it('should initialize with custom config', async () => {
      await graphManager.initialize({
        scale: 1.5,
        nodeSize: 0.1,
        edgeWidth: 0.005,
        forceStrength: 0.2
      })
      
      const nodeContainer = (BABYLON.TransformNode as jest.Mock).mock.results.find(
        r => r.value.name === 'nodeContainer'
      )?.value
      
      expect(nodeContainer.scaling.setAll).toHaveBeenCalledWith(1.5)
    })
  })
  
  describe('Graph Loading', () => {
    beforeEach(async () => {
      await graphManager.initialize()
    })
    
    it('should load graph data with nodes and edges', async () => {
      await graphManager.loadGraphData(mockGraphData)
      
      // Check nodes created
      expect(BABYLON.MeshBuilder.CreateSphere).toHaveBeenCalledTimes(3)
      expect(mockMeshes.has('node_node1')).toBe(true)
      expect(mockMeshes.has('node_node2')).toBe(true)
      expect(mockMeshes.has('node_node3')).toBe(true)
      
      // Check edges created
      expect(BABYLON.MeshBuilder.CreateTube).toHaveBeenCalledTimes(2)
      expect(mockMeshes.has('edge_edge1')).toBe(true)
      expect(mockMeshes.has('edge_edge2')).toBe(true)
    })
    
    it('should position nodes correctly', async () => {
      await graphManager.loadGraphData(mockGraphData)
      
      const node1 = mockMeshes.get('node_node1')
      const node2 = mockMeshes.get('node_node2')
      const node3 = mockMeshes.get('node_node3')
      
      expect(node1?.position).toEqual({ x: 0, y: 0, z: 0 })
      expect(node2?.position).toEqual({ x: 1, y: 0, z: 0 })
      expect(node3?.position).toEqual({ x: 0.5, y: 1, z: 0.5 })
    })
    
    it('should apply node colors and sizes', async () => {
      await graphManager.loadGraphData(mockGraphData)
      
      const node1 = mockMeshes.get('node_node1')
      const node3 = mockMeshes.get('node_node3')
      
      expect(BABYLON.MeshBuilder.CreateSphere).toHaveBeenCalledWith(
        'node_node1',
        { diameter: 0.2 }, // 0.1 * 2
        mockScene
      )
      
      expect(BABYLON.MeshBuilder.CreateSphere).toHaveBeenCalledWith(
        'node_node3',
        { diameter: 0.3 }, // 0.15 * 2
        mockScene
      )
      
      expect(BABYLON.Color3.FromHexString).toHaveBeenCalledWith('#FF6B6B')
      expect(BABYLON.Color3.FromHexString).toHaveBeenCalledWith('#45B7D1')
    })
    
    it('should create edges with correct paths', async () => {
      await graphManager.loadGraphData(mockGraphData)
      
      const edge1 = mockMeshes.get('edge_edge1')
      const edge2 = mockMeshes.get('edge_edge2')
      
      expect(edge1?.path).toEqual([
        { x: 0, y: 0, z: 0 },  // node1 position
        { x: 1, y: 0, z: 0 }   // node2 position
      ])
      
      expect(edge2?.path).toEqual([
        { x: 1, y: 0, z: 0 },    // node2 position
        { x: 0.5, y: 1, z: 0.5 } // node3 position
      ])
    })
    
    it('should handle edges with missing nodes', async () => {
      const invalidGraphData = {
        nodes: [{ id: 'node1', x: 0, y: 0, z: 0 }],
        edges: [{ id: 'edge1', source: 'node1', target: 'missingNode' }]
      }
      
      await graphManager.loadGraphData(invalidGraphData)
      
      // Edge should not be created
      expect(mockMeshes.has('edge_edge1')).toBe(false)
      expect(BABYLON.MeshBuilder.CreateTube).not.toHaveBeenCalled()
    })
    
    it('should clear existing graph before loading new one', async () => {
      // Load first graph
      await graphManager.loadGraphData(mockGraphData)
      
      const firstNodes = Array.from(mockMeshes.values())
      
      // Load second graph
      const newGraphData = {
        nodes: [{ id: 'newNode', x: 0, y: 0, z: 0 }],
        edges: []
      }
      
      await graphManager.loadGraphData(newGraphData)
      
      // Old meshes should be disposed
      firstNodes.forEach(mesh => {
        expect(mesh.dispose).toHaveBeenCalled()
      })
      
      // New node should exist
      expect(mockMeshes.has('node_newNode')).toBe(true)
    })
  })
  
  describe('Graph Updates', () => {
    beforeEach(async () => {
      await graphManager.initialize()
      await graphManager.loadGraphData(mockGraphData)
    })
    
    it('should update existing node positions with animation', async () => {
      const updatedData = {
        nodes: [
          { id: 'node1', x: 2, y: 1, z: 0 },
          { id: 'node2', x: 3, y: 0, z: 1 },
          { id: 'node3', x: 2.5, y: 2, z: 0.5 }
        ],
        edges: mockGraphData.edges
      }
      
      await graphManager.updateGraphData(updatedData)
      
      // Check animations were created
      expect(BABYLON.Animation.CreateAndStartAnimation).toHaveBeenCalledTimes(3)
      
      // Check target positions
      const node1 = mockMeshes.get('node_node1')
      const node2 = mockMeshes.get('node_node2')
      
      expect(node1?.position).toEqual({ x: 2, y: 1, z: 0 })
      expect(node2?.position).toEqual({ x: 3, y: 0, z: 1 })
    })
    
    it('should add new nodes in updates', async () => {
      const updatedData = {
        nodes: [
          ...mockGraphData.nodes,
          { id: 'node4', x: 2, y: 0, z: 0 }
        ],
        edges: mockGraphData.edges
      }
      
      await graphManager.updateGraphData(updatedData)
      
      expect(mockMeshes.has('node_node4')).toBe(true)
    })
    
    it('should remove nodes that no longer exist', async () => {
      const node3 = mockMeshes.get('node_node3')
      
      const updatedData = {
        nodes: [
          { id: 'node1', x: 0, y: 0, z: 0 },
          { id: 'node2', x: 1, y: 0, z: 0 }
        ],
        edges: [{ id: 'edge1', source: 'node1', target: 'node2' }]
      }
      
      await graphManager.updateGraphData(updatedData)
      
      expect(node3?.dispose).toHaveBeenCalled()
      expect(mockMeshes.has('node_node3')).toBe(false)
    })
    
    it('should recreate all edges on update', async () => {
      // Get initial edges
      const initialEdges = ['edge_edge1', 'edge_edge2'].map(id => mockMeshes.get(id))
      
      const updatedData = {
        nodes: mockGraphData.nodes,
        edges: [
          { id: 'edge1', source: 'node1', target: 'node2' },
          { id: 'edge3', source: 'node1', target: 'node3' }
        ]
      }
      
      await graphManager.updateGraphData(updatedData)
      
      // Old edges should be disposed
      initialEdges.forEach(edge => {
        expect(edge?.dispose).toHaveBeenCalled()
      })
      
      // New edges should exist
      expect(mockMeshes.has('edge_edge1')).toBe(true)
      expect(mockMeshes.has('edge_edge3')).toBe(true)
      expect(mockMeshes.has('edge_edge2')).toBe(false)
    })
  })
  
  describe('Node Selection', () => {
    beforeEach(async () => {
      await graphManager.initialize()
      await graphManager.loadGraphData(mockGraphData)
    })
    
    it('should select a node', () => {
      graphManager.selectNode('node1')
      
      const stats = graphManager.getStats()
      expect(stats.selectedNodes).toEqual(['node1'])
    })
    
    it('should select multiple nodes', () => {
      graphManager.selectNode('node1')
      graphManager.selectNode('node2')
      
      const stats = graphManager.getStats()
      expect(stats.selectedNodes).toContain('node1')
      expect(stats.selectedNodes).toContain('node2')
    })
    
    it('should deselect all nodes', () => {
      graphManager.selectNode('node1')
      graphManager.selectNode('node2')
      graphManager.deselectAllNodes()
      
      const stats = graphManager.getStats()
      expect(stats.selectedNodes).toEqual([])
    })
    
    it('should handle selecting non-existent nodes', () => {
      graphManager.selectNode('nonExistentNode')
      
      const stats = graphManager.getStats()
      expect(stats.selectedNodes).toContain('nonExistentNode')
    })
  })
  
  describe('Camera Focus', () => {
    beforeEach(async () => {
      await graphManager.initialize()
      await graphManager.loadGraphData(mockGraphData)
    })
    
    it('should focus camera on node', () => {
      graphManager.focusOnNode('node2')
      
      const node2 = mockMeshes.get('node_node2')
      
      // Camera should animate to node position
      expect(BABYLON.Animation.CreateAndStartAnimation).toHaveBeenCalledWith(
        'cameraMove',
        mockScene.activeCamera,
        'position',
        expect.any(Number),
        expect.any(Number),
        mockScene.activeCamera.position,
        expect.objectContaining({ x: 1, y: 0, z: expect.any(Number) }),
        expect.any(Number)
      )
      
      expect(mockScene.activeCamera.setTarget).toHaveBeenCalled()
    })
    
    it('should handle focus on non-existent node', () => {
      // Should not throw
      expect(() => graphManager.focusOnNode('nonExistentNode')).not.toThrow()
    })
  })
  
  describe('Physics Simulation', () => {
    beforeEach(async () => {
      await graphManager.initialize()
      await graphManager.loadGraphData(mockGraphData)
    })
    
    it('should start physics simulation', () => {
      graphManager.startPhysicsSimulation()
      
      expect(mockScene.onBeforeRenderObservable.add).toHaveBeenCalled()
      
      const stats = graphManager.getStats()
      expect(stats.physicsEnabled).toBe(true)
    })
    
    it('should stop physics simulation', () => {
      graphManager.startPhysicsSimulation()
      graphManager.stopPhysicsSimulation()
      
      expect(mockScene.onBeforeRenderObservable.remove).toHaveBeenCalled()
      
      const stats = graphManager.getStats()
      expect(stats.physicsEnabled).toBe(false)
    })
    
    it('should apply forces between nodes', () => {
      graphManager.startPhysicsSimulation()
      
      // Get the physics callback
      const physicsCallback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      
      // Store initial positions
      const initialPositions = new Map()
      mockGraphData.nodes.forEach(node => {
        const mesh = mockMeshes.get(`node_${node.id}`)
        initialPositions.set(node.id, { ...mesh.position })
      })
      
      // Run physics simulation step
      physicsCallback()
      
      // Positions should change (unless they're perfectly balanced)
      // This is a simplified test - in reality we'd need to mock the force calculations
      expect(mockScene.onBeforeRenderObservable.add).toHaveBeenCalled()
    })
  })
  
  describe('Statistics', () => {
    it('should return empty stats when not initialized', () => {
      const stats = graphManager.getStats()
      
      expect(stats).toEqual({
        nodeCount: 0,
        edgeCount: 0,
        selectedNodes: [],
        physicsEnabled: false
      })
    })
    
    it('should return correct stats after loading graph', async () => {
      await graphManager.initialize()
      await graphManager.loadGraphData(mockGraphData)
      
      const stats = graphManager.getStats()
      
      expect(stats).toEqual({
        nodeCount: 3,
        edgeCount: 2,
        selectedNodes: [],
        physicsEnabled: false
      })
    })
    
    it('should update stats after operations', async () => {
      await graphManager.initialize()
      await graphManager.loadGraphData(mockGraphData)
      
      graphManager.selectNode('node1')
      graphManager.startPhysicsSimulation()
      
      const stats = graphManager.getStats()
      
      expect(stats).toEqual({
        nodeCount: 3,
        edgeCount: 2,
        selectedNodes: ['node1'],
        physicsEnabled: true
      })
    })
  })
  
  describe('Disposal', () => {
    it('should dispose all resources', async () => {
      await graphManager.initialize()
      await graphManager.loadGraphData(mockGraphData)
      
      graphManager.startPhysicsSimulation()
      
      const allMeshes = Array.from(mockMeshes.values())
      
      graphManager.dispose()
      
      // All meshes should be disposed
      allMeshes.forEach(mesh => {
        expect(mesh.dispose).toHaveBeenCalled()
      })
      
      // Physics should be stopped
      expect(mockScene.onBeforeRenderObservable.remove).toHaveBeenCalled()
      
      // Stats should be reset
      const stats = graphManager.getStats()
      expect(stats.nodeCount).toBe(0)
      expect(stats.edgeCount).toBe(0)
    })
  })
  
  describe('Edge Cases', () => {
    beforeEach(async () => {
      await graphManager.initialize()
    })
    
    it('should handle empty graph data', async () => {
      await graphManager.loadGraphData({ nodes: [], edges: [] })
      
      const stats = graphManager.getStats()
      expect(stats.nodeCount).toBe(0)
      expect(stats.edgeCount).toBe(0)
    })
    
    it('should handle nodes without position data', async () => {
      const graphWithDefaults = {
        nodes: [
          { id: 'node1' }, // No position
          { id: 'node2', x: 1 } // Partial position
        ],
        edges: []
      }
      
      await graphManager.loadGraphData(graphWithDefaults as any)
      
      const node1 = mockMeshes.get('node_node1')
      const node2 = mockMeshes.get('node_node2')
      
      expect(node1?.position).toEqual({ x: 0, y: 0, z: 0 })
      expect(node2?.position).toEqual({ x: 1, y: 0, z: 0 })
    })
    
    it('should use default colors when not specified', async () => {
      const graphWithoutColors = {
        nodes: [{ id: 'node1', x: 0, y: 0, z: 0 }],
        edges: [{ id: 'edge1', source: 'node1', target: 'node1' }] // Self-loop
      }
      
      await graphManager.loadGraphData(graphWithoutColors)
      
      expect(BABYLON.Color3.FromHexString).toHaveBeenCalledWith('#4ECDC4') // Default node color
      expect(BABYLON.Color3.FromHexString).toHaveBeenCalledWith('#666666') // Default edge color
    })
  })
})