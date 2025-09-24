import * as BABYLON from '@babylonjs/core'
import { createLogger } from '../../utils/loggerConfig'

const logger = createLogger('GraphVisualizationManager')

interface Node {
  id: string
  label?: string
  x: number
  y: number
  z: number
  color?: string
  size?: number
  metadata?: any
}

interface Edge {
  id: string
  source: string
  target: string
  weight?: number
  color?: string
}

interface GraphData {
  nodes: Node[]
  edges: Edge[]
}

interface GraphConfig {
  scale?: number
  nodeSize?: number
  edgeWidth?: number
  forceStrength?: number
}

export class GraphVisualizationManager {
  private scene: BABYLON.Scene
  private config: Required<GraphConfig>
  private nodeContainer: BABYLON.TransformNode
  private edgeContainer: BABYLON.TransformNode
  private nodeMeshes: Map<string, BABYLON.Mesh> = new Map()
  private edgeMeshes: Map<string, BABYLON.Mesh> = new Map()
  private selectedNodes: Set<string> = new Set()
  private physicsEnabled = false
  private physicsObserver: BABYLON.Nullable<BABYLON.Observer<BABYLON.Scene>> = null
  
  constructor(scene: BABYLON.Scene) {
    this.scene = scene
    this.config = {
      scale: 0.8,
      nodeSize: 0.05,
      edgeWidth: 0.002,
      forceStrength: 0.1
    }
    
    // Create containers
    this.nodeContainer = new BABYLON.TransformNode('nodeContainer', scene)
    this.edgeContainer = new BABYLON.TransformNode('edgeContainer', scene)
  }
  
  async initialize(config: GraphConfig = {}): Promise<void> {
    this.config = { ...this.config, ...config }
    
    // Position containers
    this.nodeContainer.position = new BABYLON.Vector3(0, 1.2, -1)
    this.nodeContainer.scaling.setAll(this.config.scale)
    
    this.edgeContainer.position = new BABYLON.Vector3(0, 1.2, -1)
    this.edgeContainer.scaling.setAll(this.config.scale)
    
    logger.info('GraphVisualizationManager initialized')
  }
  
  async loadGraphData(graphData: GraphData): Promise<void> {
    logger.info(`Loading graph with ${graphData.nodes.length} nodes and ${graphData.edges.length} edges`)
    
    // Clear existing graph
    this.clearGraph()
    
    // Create nodes
    for (const nodeData of graphData.nodes) {
      await this.createNode(nodeData)
    }
    
    // Create edges
    for (const edgeData of graphData.edges) {
      await this.createEdge(edgeData)
    }
    
    // Apply initial layout
    this.applyForceDirectedLayout()
    
    logger.info('Graph data loaded successfully')
  }
  
  async updateGraphData(graphData: GraphData): Promise<void> {
    // Implement incremental updates
    const currentNodeIds = new Set(this.nodeMeshes.keys())
    const newNodeIds = new Set(graphData.nodes.map(n => n.id))
    
    // Remove nodes that no longer exist
    for (const nodeId of currentNodeIds) {
      if (!newNodeIds.has(nodeId)) {
        this.removeNode(nodeId)
      }
    }
    
    // Add or update nodes
    for (const nodeData of graphData.nodes) {
      if (this.nodeMeshes.has(nodeData.id)) {
        this.updateNode(nodeData)
      } else {
        await this.createNode(nodeData)
      }
    }
    
    // Update edges
    this.clearEdges()
    for (const edgeData of graphData.edges) {
      await this.createEdge(edgeData)
    }
  }
  
  private async createNode(nodeData: Node): Promise<void> {
    // Create sphere mesh for node
    const nodeMesh = BABYLON.MeshBuilder.CreateSphere(
      `node_${nodeData.id}`,
      { diameter: nodeData.size || this.config.nodeSize * 2 },
      this.scene
    )
    
    // Position
    nodeMesh.position = new BABYLON.Vector3(
      nodeData.x || 0,
      nodeData.y || 0,
      nodeData.z || 0
    )
    
    // Material
    const material = new BABYLON.StandardMaterial(`nodeMat_${nodeData.id}`, this.scene)
    const color = nodeData.color || '#4ECDC4'
    material.diffuseColor = BABYLON.Color3.FromHexString(color)
    material.emissiveColor = BABYLON.Color3.FromHexString(color).scale(0.3)
    material.specularColor = new BABYLON.Color3(0.2, 0.2, 0.2)
    nodeMesh.material = material
    
    // Parent to container
    nodeMesh.parent = this.nodeContainer
    
    // Store metadata
    nodeMesh.metadata = { ...nodeData }
    
    // Add to map
    this.nodeMeshes.set(nodeData.id, nodeMesh)
    
    // Make it interactive
    nodeMesh.isPickable = true
    
    // Add glow effect
    if (this.scene.effectLayers) {
      const glowLayer = this.scene.effectLayers.find(
        layer => layer.name === 'glowLayer'
      ) as BABYLON.GlowLayer
      if (glowLayer) {
        glowLayer.addIncludedOnlyMesh(nodeMesh)
      }
    }
  }
  
  private updateNode(nodeData: Node): void {
    const nodeMesh = this.nodeMeshes.get(nodeData.id)
    if (!nodeMesh) return
    
    // Update position with animation
    BABYLON.Animation.CreateAndStartAnimation(
      'nodeMove',
      nodeMesh,
      'position',
      30,
      15,
      nodeMesh.position,
      new BABYLON.Vector3(nodeData.x, nodeData.y, nodeData.z),
      BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT
    )
    
    // Update color if changed
    if (nodeData.color && nodeMesh.material instanceof BABYLON.StandardMaterial) {
      nodeMesh.material.diffuseColor = BABYLON.Color3.FromHexString(nodeData.color)
      nodeMesh.material.emissiveColor = BABYLON.Color3.FromHexString(nodeData.color).scale(0.3)
    }
    
    // Update metadata
    nodeMesh.metadata = { ...nodeMesh.metadata, ...nodeData }
  }
  
  private removeNode(nodeId: string): void {
    const nodeMesh = this.nodeMeshes.get(nodeId)
    if (!nodeMesh) return
    
    nodeMesh.dispose()
    this.nodeMeshes.delete(nodeId)
    this.selectedNodes.delete(nodeId)
  }
  
  private async createEdge(edgeData: Edge): Promise<void> {
    const sourceNode = this.nodeMeshes.get(edgeData.source)
    const targetNode = this.nodeMeshes.get(edgeData.target)
    
    if (!sourceNode || !targetNode) {
      logger.warn(`Cannot create edge ${edgeData.id}: nodes not found`)
      return
    }
    
    // Create tube mesh for edge
    const path = [sourceNode.position, targetNode.position]
    const edgeMesh = BABYLON.MeshBuilder.CreateTube(
      `edge_${edgeData.id}`,
      {
        path: path,
        radius: this.config.edgeWidth,
        tessellation: 16,
        updatable: true
      },
      this.scene
    )
    
    // Material
    const material = new BABYLON.StandardMaterial(`edgeMat_${edgeData.id}`, this.scene)
    const color = edgeData.color || '#666666'
    material.diffuseColor = BABYLON.Color3.FromHexString(color)
    material.alpha = 0.7
    material.backFaceCulling = false
    edgeMesh.material = material
    
    // Parent to container
    edgeMesh.parent = this.edgeContainer
    
    // Store metadata
    edgeMesh.metadata = { ...edgeData, sourceNode, targetNode }
    
    // Add to map
    this.edgeMeshes.set(edgeData.id, edgeMesh)
  }
  
  private clearEdges(): void {
    this.edgeMeshes.forEach(mesh => mesh.dispose())
    this.edgeMeshes.clear()
  }
  
  private clearGraph(): void {
    this.nodeMeshes.forEach(mesh => mesh.dispose())
    this.nodeMeshes.clear()
    this.clearEdges()
    this.selectedNodes.clear()
  }
  
  // Force-directed layout
  private applyForceDirectedLayout(): void {
    const nodes = Array.from(this.nodeMeshes.values())
    const iterations = 50
    
    for (let i = 0; i < iterations; i++) {
      // Repulsive forces between all nodes
      for (let j = 0; j < nodes.length; j++) {
        for (let k = j + 1; k < nodes.length; k++) {
          const node1 = nodes[j]
          const node2 = nodes[k]
          
          const delta = node1.position.subtract(node2.position)
          const distance = Math.max(delta.length(), 0.1)
          const force = this.config.forceStrength / (distance * distance)
          
          const displacement = delta.normalize().scale(force)
          node1.position.addInPlace(displacement)
          node2.position.subtractInPlace(displacement)
        }
      }
      
      // Attractive forces along edges
      this.edgeMeshes.forEach(edge => {
        const { sourceNode, targetNode } = edge.metadata
        if (!sourceNode || !targetNode) return
        
        const delta = targetNode.position.subtract(sourceNode.position)
        const distance = delta.length()
        const force = distance * this.config.forceStrength * 0.1
        
        const displacement = delta.normalize().scale(force)
        sourceNode.position.addInPlace(displacement)
        targetNode.position.subtractInPlace(displacement)
      })
      
      // Center the graph
      const center = new BABYLON.Vector3(0, 0, 0)
      nodes.forEach(node => {
        const toCenter = center.subtract(node.position).scale(0.01)
        node.position.addInPlace(toCenter)
      })
    }
    
    // Update edge positions
    this.updateEdgePositions()
  }
  
  private updateEdgePositions(): void {
    this.edgeMeshes.forEach(edge => {
      const { sourceNode, targetNode } = edge.metadata
      if (!sourceNode || !targetNode) return
      
      const path = [sourceNode.position, targetNode.position]
      const tubeData = BABYLON.MeshBuilder.CreateTube(
        'temp',
        { path: path, radius: this.config.edgeWidth },
        this.scene
      )
      
      edge.geometry = tubeData.geometry
      tubeData.dispose()
    })
  }
  
  startPhysicsSimulation(): void {
    if (this.physicsEnabled) return
    
    this.physicsEnabled = true
    this.physicsObserver = this.scene.onBeforeRenderObservable.add(() => {
      if (this.physicsEnabled) {
        this.applyForceDirectedLayout()
      }
    })
    
    logger.info('Physics simulation started')
  }
  
  stopPhysicsSimulation(): void {
    this.physicsEnabled = false
    if (this.physicsObserver) {
      this.scene.onBeforeRenderObservable.remove(this.physicsObserver)
      this.physicsObserver = null
    }
    
    logger.info('Physics simulation stopped')
  }
  
  selectNode(nodeId: string): void {
    const nodeMesh = this.nodeMeshes.get(nodeId)
    if (!nodeMesh) return
    
    this.selectedNodes.add(nodeId)
    
    // Highlight selected node
    if (nodeMesh.material instanceof BABYLON.StandardMaterial) {
      nodeMesh.material.emissiveColor = BABYLON.Color3.FromHexString('#FFD700')
    }
    
    // Scale up
    BABYLON.Animation.CreateAndStartAnimation(
      'selectNode',
      nodeMesh,
      'scaling',
      30,
      10,
      nodeMesh.scaling,
      new BABYLON.Vector3(1.5, 1.5, 1.5),
      BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT
    )
  }
  
  deselectAllNodes(): void {
    this.selectedNodes.forEach(nodeId => {
      const nodeMesh = this.nodeMeshes.get(nodeId)
      if (!nodeMesh) return
      
      // Reset highlight
      if (nodeMesh.material instanceof BABYLON.StandardMaterial) {
        const originalColor = nodeMesh.metadata.color || '#4ECDC4'
        nodeMesh.material.emissiveColor = BABYLON.Color3.FromHexString(originalColor).scale(0.3)
      }
      
      // Reset scale
      BABYLON.Animation.CreateAndStartAnimation(
        'deselectNode',
        nodeMesh,
        'scaling',
        30,
        10,
        nodeMesh.scaling,
        new BABYLON.Vector3(1, 1, 1),
        BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT
      )
    })
    
    this.selectedNodes.clear()
  }
  
  focusOnNode(nodeId: string): void {
    const nodeMesh = this.nodeMeshes.get(nodeId)
    if (!nodeMesh) return
    
    const camera = this.scene.activeCamera
    if (!camera) return
    
    // Animate camera to focus on node
    const targetPosition = nodeMesh.position.add(new BABYLON.Vector3(0, 0.5, 2))
    
    BABYLON.Animation.CreateAndStartAnimation(
      'cameraMove',
      camera,
      'position',
      30,
      30,
      camera.position,
      targetPosition,
      BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT
    )
    
    if (camera instanceof BABYLON.UniversalCamera) {
      camera.setTarget(nodeMesh.position)
    }
  }
  
  getStats(): any {
    return {
      nodeCount: this.nodeMeshes.size,
      edgeCount: this.edgeMeshes.size,
      selectedCount: this.selectedNodes.size,
      physicsEnabled: this.physicsEnabled
    }
  }
  
  dispose(): void {
    this.stopPhysicsSimulation()
    this.clearGraph()
    this.nodeContainer.dispose()
    this.edgeContainer.dispose()
    logger.info('GraphVisualizationManager disposed')
  }
}