import {
  Scene,
  InstancedMesh,
  Mesh,
  MeshBuilder,
  StandardMaterial,
  Color3,
  Vector3
} from '@babylonjs/core';
import { AdvancedDynamicTexture, TextBlock } from '@babylonjs/gui';

/**
 * Handles rendering of the knowledge graph in 3D space
 * Manages nodes, edges, and labels with high performance instancing
 */
export class GraphRenderer {
  private scene: Scene;
  private nodeMasterMesh: Mesh | null = null;
  private nodeInstances: InstancedMesh[] = [];
  private nodeInstanceMap = new Map<string, InstancedMesh>(); // Track nodes by ID for incremental updates
  private edgeLineSystem: Mesh | null = null;
  private labelTexture: AdvancedDynamicTexture | null = null;
  private labelBlocks: Map<string, TextBlock> = new Map();

  constructor(scene: Scene) {
    this.scene = scene;
    this.initialize();
    this.subscribeToData();
  }

  private initialize(): void {
    // Create node mesh template
    this.nodeMasterMesh = MeshBuilder.CreateSphere('nodeMasterMesh', { diameter: 0.1 }, this.scene);
    const nodeMaterial = new StandardMaterial('nodeMaterial', this.scene);
    nodeMaterial.diffuseColor = Color3.Blue();
    this.nodeMasterMesh.material = nodeMaterial;
    this.nodeMasterMesh.isVisible = false; // Hide master mesh

    // Initialize GUI for labels
    this.labelTexture = AdvancedDynamicTexture.CreateFullscreenUI('labelUI');
  }

  private subscribeToData(): void {
    // Real data subscriptions will be managed by the useImmersiveData hook
    // This component will receive data through the updateNodes/updateEdges/updateLabels methods
    console.log('GraphRenderer: Ready to receive data from useImmersiveData hook');
  }

  public updateNodes(nodes: any[], positions?: Float32Array): void {
    if (!this.nodeMasterMesh) return;

    // If we have positions but no nodes, create minimal nodes from positions
    let nodeList = nodes;
    if ((!nodes || nodes.length === 0) && positions && positions.length > 0) {
      const nodeCount = Math.floor(positions.length / 3);
      nodeList = Array.from({ length: nodeCount }, (_, i) => ({
        id: String(i),
        label: `Node ${i}`,
        type: 'default'
      }));
    }

    // DEDUPLICATION: Remove duplicate nodes by ID (HIGH PRIORITY FIX)
    const uniqueNodes = new Map<string, any>();
    let duplicateCount = 0;
    nodeList.forEach(node => {
      if (!uniqueNodes.has(node.id)) {
        uniqueNodes.set(node.id, node);
      } else {
        duplicateCount++;
        console.warn(`GraphRenderer: Duplicate node ID detected: ${node.id}`);
      }
    });

    const deduplicatedNodes = Array.from(uniqueNodes.values());

    console.log(
      `GraphRenderer: Updating nodes - Input: ${nodeList.length}, Unique: ${deduplicatedNodes.length}` +
      (duplicateCount > 0 ? `, Duplicates removed: ${duplicateCount}` : '')
    );

    // INCREMENTAL UPDATES: Track which nodes to keep, update, or create
    const newNodeIds = new Set(deduplicatedNodes.map(n => n.id));

    // Remove nodes that no longer exist
    const nodesToRemove: string[] = [];
    this.nodeInstanceMap.forEach((instance, nodeId) => {
      if (!newNodeIds.has(nodeId)) {
        instance.dispose();
        nodesToRemove.push(nodeId);
        // Remove from instances array
        const idx = this.nodeInstances.indexOf(instance);
        if (idx !== -1) {
          this.nodeInstances.splice(idx, 1);
        }
      }
    });

    nodesToRemove.forEach(nodeId => this.nodeInstanceMap.delete(nodeId));

    if (nodesToRemove.length > 0) {
      console.log(`GraphRenderer: Removed ${nodesToRemove.length} obsolete nodes`);
    }

    // Update existing nodes or create new ones
    let createdCount = 0;
    let updatedCount = 0;

    deduplicatedNodes.forEach((node, i) => {
      let instance = this.nodeInstanceMap.get(node.id);

      if (!instance) {
        // Create new instance
        instance = this.nodeMasterMesh!.createInstance(`node_${node.id}`);
        this.nodeInstanceMap.set(node.id, instance);
        this.nodeInstances.push(instance);
        createdCount++;
      } else {
        updatedCount++;
      }

      // Set position from physics simulation or node data
      let x = node.position?.x || node.x || 0;
      let y = node.position?.y || node.y || 0;
      let z = node.position?.z || node.z || 0;

      if (positions && i * 3 + 2 < positions.length) {
        x = positions[i * 3];
        y = positions[i * 3 + 1];
        z = positions[i * 3 + 2];
      }

      // Update position (for both new and existing nodes)
      instance.position.x = x;
      instance.position.y = y;
      instance.position.z = z;

      // Update metadata
      instance.metadata = { nodeId: node.id, nodeData: node };
    });

    if (createdCount > 0 || updatedCount > 0) {
      console.log(`GraphRenderer: Created ${createdCount} new nodes, updated ${updatedCount} existing nodes`);
    }
  }

  public updateEdges(edges: any[], nodePositions?: Float32Array): void {
    console.log('GraphRenderer: Updating edges', edges.length);

    if (!edges.length) return;

    // Dispose old line system
    if (this.edgeLineSystem) {
      this.edgeLineSystem.dispose();
    }

    // Create lines for edges
    const lines: Vector3[][] = [];

    for (const edge of edges) {
      const sourceNode = this.getNodePosition(edge.source, nodePositions);
      const targetNode = this.getNodePosition(edge.target, nodePositions);

      if (sourceNode && targetNode) {
        lines.push([sourceNode, targetNode]);
      }
    }

    if (lines.length > 0) {
      this.edgeLineSystem = MeshBuilder.CreateLineSystem('edges', { lines }, this.scene);
      // Create edge material with emissive properties for XR visibility
      const edgeMaterial = new StandardMaterial('edgeMaterial', this.scene);
      edgeMaterial.diffuseColor = new Color3(0.7, 0.7, 0.8);
      edgeMaterial.emissiveColor = new Color3(0.3, 0.3, 0.4); // Glow for visibility
      edgeMaterial.specularColor = new Color3(0.5, 0.5, 0.6);
      this.edgeLineSystem.material = edgeMaterial;
    }
  }

  public updateLabels(nodes: any[]): void {
    if (!this.labelTexture) return;

    console.log('GraphRenderer: Updating labels', nodes.length);

    // Clear existing labels
    this.labelBlocks.forEach(block => block.dispose());
    this.labelBlocks.clear();

    // Create new labels
    for (const node of nodes) {
      if (node.label) {
        const textBlock = new TextBlock(node.id + '_label', node.label);
        textBlock.color = '#FFFFFF'; // Bright white for XR visibility
        textBlock.fontSize = 18; // Larger for better readability in XR
        textBlock.outlineWidth = 2;
        textBlock.outlineColor = '#000000'; // Black outline for contrast
        // TODO: Position label relative to node
        this.labelTexture.addControl(textBlock);
        this.labelBlocks.set(node.id, textBlock);
      }
    }
  }

  private getNodeColor(node: any): Color3 {
    // Default color logic based on node properties
    if (node.type === 'agent') return Color3.Blue();
    if (node.type === 'document') return Color3.Green();
    if (node.type === 'entity') return Color3.Red();
    return Color3.White();
  }

  private getNodePosition(nodeId: string | number, positions?: Float32Array): Vector3 | null {
    // Convert nodeId to number if it's a string number
    const nodeIndex = typeof nodeId === 'string' ? parseInt(nodeId, 10) : nodeId;

    // Check if we have positions array and the index is valid
    if (positions && !isNaN(nodeIndex)) {
      const idx = nodeIndex * 3;
      if (idx + 2 < positions.length) {
        return new Vector3(
          positions[idx],
          positions[idx + 1],
          positions[idx + 2]
        );
      }
    }

    // Improved error handling: Log warning for debugging
    console.warn(
      `GraphRenderer: Node position not found for ID "${nodeId}" ` +
      `(index: ${nodeIndex}, positions array length: ${positions?.length || 0}). ` +
      `Using fallback random position.`
    );

    // Fallback to a random position if not found
    return new Vector3(
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10
    );
  }

  public dispose(): void {
    // Dispose all node instances
    this.nodeInstances.forEach(instance => instance.dispose());
    this.nodeInstances = [];
    this.nodeInstanceMap.clear();

    if (this.nodeMasterMesh) {
      this.nodeMasterMesh.dispose();
    }
    if (this.edgeLineSystem) {
      this.edgeLineSystem.dispose();
    }
    if (this.labelTexture) {
      this.labelTexture.dispose();
    }
    this.labelBlocks.clear();
  }
}