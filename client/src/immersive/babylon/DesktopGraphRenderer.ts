import {
  Scene,
  InstancedMesh,
  Mesh,
  MeshBuilder,
  StandardMaterial,
  Color3,
  Vector3,
  LineSystem,
  ArcRotateCamera,
  PickingInfo,
  PointerEventTypes,
  PointerInfo
} from '@babylonjs/core';
import { AdvancedDynamicTexture, TextBlock } from '@babylonjs/gui';

/**
 * Desktop version of Graph Renderer with orbit camera and mouse interaction
 * Provides traditional mouse + keyboard navigation for non-VR users
 */
export class DesktopGraphRenderer {
  private scene: Scene;
  private camera: ArcRotateCamera;
  private nodeMasterMesh: Mesh | null = null;
  private nodeInstances: InstancedMesh[] = [];
  private edgeLineSystem: LineSystem | null = null;
  private labelTexture: AdvancedDynamicTexture | null = null;
  private labelBlocks: Map<string, TextBlock> = new Map();
  private selectedNode: InstancedMesh | null = null;
  private onNodeSelectCallback: ((nodeId: string) => void) | null = null;

  constructor(scene: Scene, canvas: HTMLCanvasElement) {
    this.scene = scene;
    this.initializeCamera(canvas);
    this.initializeRenderer();
    this.initializeInteractions(canvas);
  }

  /**
   * Initialize orbit camera for desktop navigation
   */
  private initializeCamera(canvas: HTMLCanvasElement): void {
    // Create orbit camera centered at origin
    this.camera = new ArcRotateCamera(
      'desktopCamera',
      Math.PI / 2,  // Alpha (horizontal rotation)
      Math.PI / 3,  // Beta (vertical rotation)
      15,           // Radius (distance from target)
      new Vector3(0, 0, 0), // Target position
      this.scene
    );

    // Attach camera controls to canvas
    this.camera.attachControl(canvas, true);

    // Configure camera behavior
    this.camera.wheelPrecision = 50;           // Mouse wheel zoom sensitivity
    this.camera.lowerRadiusLimit = 2;          // Minimum zoom distance
    this.camera.upperRadiusLimit = 100;        // Maximum zoom distance
    this.camera.lowerBetaLimit = 0.1;          // Prevent camera from going below ground
    this.camera.upperBetaLimit = Math.PI - 0.1; // Prevent camera from going above scene
    this.camera.panningSensibility = 50;       // Middle mouse pan sensitivity
    this.camera.pinchPrecision = 50;           // Touch pinch zoom sensitivity

    // Enable panning with middle mouse or Ctrl+left mouse
    this.camera.panningAxis = new Vector3(1, 1, 0); // Pan on X and Y, not Z

    console.log('DesktopGraphRenderer: Camera initialized with orbit controls');
  }

  /**
   * Initialize node/edge rendering
   */
  private initializeRenderer(): void {
    // Create node mesh template
    this.nodeMasterMesh = MeshBuilder.CreateSphere('nodeMasterMesh', { diameter: 0.2 }, this.scene);
    const nodeMaterial = new StandardMaterial('nodeMaterial', this.scene);
    nodeMaterial.diffuseColor = Color3.Blue();
    nodeMaterial.specularColor = Color3.White();
    nodeMaterial.emissiveColor = new Color3(0.1, 0.1, 0.2); // Slight glow
    this.nodeMasterMesh.material = nodeMaterial;
    this.nodeMasterMesh.isVisible = false; // Hide master mesh

    // Initialize GUI for labels
    this.labelTexture = AdvancedDynamicTexture.CreateFullscreenUI('labelUI');

    console.log('DesktopGraphRenderer: Renderer initialized');
  }

  /**
   * Setup mouse interactions for node selection
   */
  private initializeInteractions(canvas: HTMLCanvasElement): void {
    // Pointer observable for click events
    this.scene.onPointerObservable.add((pointerInfo: PointerInfo) => {
      if (pointerInfo.type === PointerEventTypes.POINTERDOWN) {
        this.handlePointerDown(pointerInfo);
      }
    });

    // Optional: Hover effect
    this.scene.onPointerObservable.add((pointerInfo: PointerInfo) => {
      if (pointerInfo.type === PointerEventTypes.POINTERMOVE) {
        this.handlePointerMove(pointerInfo);
      }
    });

    console.log('DesktopGraphRenderer: Mouse interactions initialized');
  }

  /**
   * Handle pointer down (click) events
   */
  private handlePointerDown(pointerInfo: PointerInfo): void {
    const pickResult = this.scene.pick(
      this.scene.pointerX,
      this.scene.pointerY,
      (mesh) => mesh instanceof InstancedMesh && mesh.name.startsWith('node_')
    );

    if (pickResult && pickResult.hit && pickResult.pickedMesh) {
      const node = pickResult.pickedMesh as InstancedMesh;
      this.selectNode(node);
    } else {
      // Clicked on empty space - deselect
      this.deselectNode();
    }
  }

  /**
   * Handle pointer move (hover) events
   */
  private handlePointerMove(pointerInfo: PointerInfo): void {
    const pickResult = this.scene.pick(
      this.scene.pointerX,
      this.scene.pointerY,
      (mesh) => mesh instanceof InstancedMesh && mesh.name.startsWith('node_')
    );

    if (pickResult && pickResult.hit && pickResult.pickedMesh) {
      // Change cursor to pointer when hovering over nodes
      this.scene.getEngine().getRenderingCanvas()!.style.cursor = 'pointer';
    } else {
      // Reset cursor
      this.scene.getEngine().getRenderingCanvas()!.style.cursor = 'default';
    }
  }

  /**
   * Select a node and highlight it
   */
  private selectNode(node: InstancedMesh): void {
    // Deselect previous node
    if (this.selectedNode && this.selectedNode !== node) {
      this.resetNodeAppearance(this.selectedNode);
    }

    // Select new node
    this.selectedNode = node;
    this.highlightNode(node);

    // Fire callback
    const nodeId = node.metadata?.nodeId;
    if (nodeId && this.onNodeSelectCallback) {
      this.onNodeSelectCallback(nodeId);
    }

    console.log('DesktopGraphRenderer: Selected node', nodeId);
  }

  /**
   * Deselect currently selected node
   */
  private deselectNode(): void {
    if (this.selectedNode) {
      this.resetNodeAppearance(this.selectedNode);
      this.selectedNode = null;
    }
  }

  /**
   * Highlight a node visually
   */
  private highlightNode(node: InstancedMesh): void {
    // Increase size and change color
    node.scaling = new Vector3(1.5, 1.5, 1.5);

    // Create highlight material
    const highlightMaterial = new StandardMaterial('highlightMaterial', this.scene);
    highlightMaterial.diffuseColor = Color3.Yellow();
    highlightMaterial.emissiveColor = Color3.Yellow().scale(0.5);
    highlightMaterial.specularColor = Color3.White();

    // Note: InstancedMesh inherits material from source mesh, but we can modify scaling/visibility
    // For full material control, we'd need to convert to a regular mesh or use thin instances
  }

  /**
   * Reset node to default appearance
   */
  private resetNodeAppearance(node: InstancedMesh): void {
    node.scaling = new Vector3(1, 1, 1);
  }

  /**
   * Set callback for node selection events
   */
  public onNodeSelect(callback: (nodeId: string) => void): void {
    this.onNodeSelectCallback = callback;
  }

  /**
   * Update nodes in the scene
   */
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

    console.log('DesktopGraphRenderer: Updating', nodeList.length, 'nodes');

    // Clear existing instances
    this.nodeInstances.forEach(instance => instance.dispose());
    this.nodeInstances = [];
    this.selectedNode = null;

    // Create new instances for each node
    for (let i = 0; i < nodeList.length; i++) {
      const node = nodeList[i];

      // Create instance
      const instance = this.nodeMasterMesh.createInstance(`node_${node.id}`);

      // Set position from physics simulation or node data
      let x = node.position?.x || node.x || 0;
      let y = node.position?.y || node.y || 0;
      let z = node.position?.z || node.z || 0;

      if (positions && i * 3 + 2 < positions.length) {
        x = positions[i * 3];
        y = positions[i * 3 + 1];
        z = positions[i * 3 + 2];
      }

      // Set position
      instance.position.set(x, y, z);

      // Store instance metadata for interaction
      instance.metadata = { nodeId: node.id, nodeData: node };

      // Add to instances array
      this.nodeInstances.push(instance);
    }
  }

  /**
   * Update edges in the scene
   */
  public updateEdges(edges: any[], nodePositions?: Float32Array): void {
    console.log('DesktopGraphRenderer: Updating', edges.length, 'edges');

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

      // Create edge material
      const edgeMaterial = new StandardMaterial('edgeMaterial', this.scene);
      edgeMaterial.diffuseColor = new Color3(0.6, 0.6, 0.7);
      edgeMaterial.emissiveColor = new Color3(0.2, 0.2, 0.3);
      edgeMaterial.specularColor = new Color3(0.4, 0.4, 0.5);
      edgeMaterial.alpha = 0.8; // Slightly transparent
      this.edgeLineSystem.material = edgeMaterial;
    }
  }

  /**
   * Update labels in the scene
   */
  public updateLabels(nodes: any[]): void {
    if (!this.labelTexture) return;

    console.log('DesktopGraphRenderer: Updating', nodes.length, 'labels');

    // Clear existing labels
    this.labelBlocks.forEach(block => block.dispose());
    this.labelBlocks.clear();

    // Create new labels
    for (const node of nodes) {
      if (node.label) {
        const textBlock = new TextBlock(node.id + '_label', node.label);
        textBlock.color = '#FFFFFF';
        textBlock.fontSize = 14;
        textBlock.outlineWidth = 1;
        textBlock.outlineColor = '#000000';

        // Position label above node (will need to update in render loop for 3D positioning)
        // For now, just add to texture
        this.labelTexture.addControl(textBlock);
        this.labelBlocks.set(node.id, textBlock);
      }
    }
  }

  /**
   * Get node position by ID
   */
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

    // Fallback to searching instances
    const instanceName = `node_${nodeId}`;
    const instance = this.nodeInstances.find(inst => inst.name === instanceName);
    if (instance) {
      return instance.position.clone();
    }

    // Final fallback to random position
    return new Vector3(
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10
    );
  }

  /**
   * Focus camera on a specific node
   */
  public focusOnNode(nodeId: string, animated: boolean = true): void {
    const instanceName = `node_${nodeId}`;
    const instance = this.nodeInstances.find(inst => inst.name === instanceName);

    if (instance) {
      if (animated) {
        // Animate camera to node
        this.camera.setTarget(instance.position);
      } else {
        // Instantly move camera
        this.camera.target = instance.position.clone();
      }

      console.log('DesktopGraphRenderer: Focused camera on node', nodeId);
    }
  }

  /**
   * Reset camera to default position
   */
  public resetCamera(): void {
    this.camera.setTarget(new Vector3(0, 0, 0));
    this.camera.alpha = Math.PI / 2;
    this.camera.beta = Math.PI / 3;
    this.camera.radius = 15;

    console.log('DesktopGraphRenderer: Camera reset to default');
  }

  /**
   * Get current camera instance
   */
  public getCamera(): ArcRotateCamera {
    return this.camera;
  }

  /**
   * Dispose of all resources
   */
  public dispose(): void {
    // Dispose all node instances
    this.nodeInstances.forEach(instance => instance.dispose());
    this.nodeInstances = [];

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

    console.log('DesktopGraphRenderer: Disposed all resources');
  }
}
