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

  
  private initializeCamera(canvas: HTMLCanvasElement): void {
    
    this.camera = new ArcRotateCamera(
      'desktopCamera',
      Math.PI / 2,  
      Math.PI / 3,  
      15,           
      new Vector3(0, 0, 0), 
      this.scene
    );

    
    this.camera.attachControl(canvas, true);

    
    this.camera.wheelPrecision = 50;           
    this.camera.lowerRadiusLimit = 2;          
    this.camera.upperRadiusLimit = 100;        
    this.camera.lowerBetaLimit = 0.1;          
    this.camera.upperBetaLimit = Math.PI - 0.1; 
    this.camera.panningSensibility = 50;       
    this.camera.pinchPrecision = 50;           

    
    this.camera.panningAxis = new Vector3(1, 1, 0); 

    console.log('DesktopGraphRenderer: Camera initialized with orbit controls');
  }

  
  private initializeRenderer(): void {
    
    this.nodeMasterMesh = MeshBuilder.CreateSphere('nodeMasterMesh', { diameter: 0.2 }, this.scene);
    const nodeMaterial = new StandardMaterial('nodeMaterial', this.scene);
    nodeMaterial.diffuseColor = Color3.Blue();
    nodeMaterial.specularColor = Color3.White();
    nodeMaterial.emissiveColor = new Color3(0.1, 0.1, 0.2); 
    this.nodeMasterMesh.material = nodeMaterial;
    this.nodeMasterMesh.isVisible = false; 

    
    this.labelTexture = AdvancedDynamicTexture.CreateFullscreenUI('labelUI');

    console.log('DesktopGraphRenderer: Renderer initialized');
  }

  
  private initializeInteractions(canvas: HTMLCanvasElement): void {
    
    this.scene.onPointerObservable.add((pointerInfo: PointerInfo) => {
      if (pointerInfo.type === PointerEventTypes.POINTERDOWN) {
        this.handlePointerDown(pointerInfo);
      }
    });

    
    this.scene.onPointerObservable.add((pointerInfo: PointerInfo) => {
      if (pointerInfo.type === PointerEventTypes.POINTERMOVE) {
        this.handlePointerMove(pointerInfo);
      }
    });

    console.log('DesktopGraphRenderer: Mouse interactions initialized');
  }

  
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
      
      this.deselectNode();
    }
  }

  
  private handlePointerMove(pointerInfo: PointerInfo): void {
    const pickResult = this.scene.pick(
      this.scene.pointerX,
      this.scene.pointerY,
      (mesh) => mesh instanceof InstancedMesh && mesh.name.startsWith('node_')
    );

    if (pickResult && pickResult.hit && pickResult.pickedMesh) {
      
      this.scene.getEngine().getRenderingCanvas()!.style.cursor = 'pointer';
    } else {
      
      this.scene.getEngine().getRenderingCanvas()!.style.cursor = 'default';
    }
  }

  
  private selectNode(node: InstancedMesh): void {
    
    if (this.selectedNode && this.selectedNode !== node) {
      this.resetNodeAppearance(this.selectedNode);
    }

    
    this.selectedNode = node;
    this.highlightNode(node);

    
    const nodeId = node.metadata?.nodeId;
    if (nodeId && this.onNodeSelectCallback) {
      this.onNodeSelectCallback(nodeId);
    }

    console.log('DesktopGraphRenderer: Selected node', nodeId);
  }

  
  private deselectNode(): void {
    if (this.selectedNode) {
      this.resetNodeAppearance(this.selectedNode);
      this.selectedNode = null;
    }
  }

  
  private highlightNode(node: InstancedMesh): void {
    
    node.scaling = new Vector3(1.5, 1.5, 1.5);

    
    const highlightMaterial = new StandardMaterial('highlightMaterial', this.scene);
    highlightMaterial.diffuseColor = Color3.Yellow();
    highlightMaterial.emissiveColor = Color3.Yellow().scale(0.5);
    highlightMaterial.specularColor = Color3.White();

    
    
  }

  
  private resetNodeAppearance(node: InstancedMesh): void {
    node.scaling = new Vector3(1, 1, 1);
  }

  
  public onNodeSelect(callback: (nodeId: string) => void): void {
    this.onNodeSelectCallback = callback;
  }

  
  public updateNodes(nodes: any[], positions?: Float32Array): void {
    if (!this.nodeMasterMesh) return;

    
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

    
    this.nodeInstances.forEach(instance => instance.dispose());
    this.nodeInstances = [];
    this.selectedNode = null;

    
    for (let i = 0; i < nodeList.length; i++) {
      const node = nodeList[i];

      
      const instance = this.nodeMasterMesh.createInstance(`node_${node.id}`);

      
      let x = node.position?.x || node.x || 0;
      let y = node.position?.y || node.y || 0;
      let z = node.position?.z || node.z || 0;

      if (positions && i * 3 + 2 < positions.length) {
        x = positions[i * 3];
        y = positions[i * 3 + 1];
        z = positions[i * 3 + 2];
      }

      
      instance.position.set(x, y, z);

      
      instance.metadata = { nodeId: node.id, nodeData: node };

      
      this.nodeInstances.push(instance);
    }
  }

  
  public updateEdges(edges: any[], nodePositions?: Float32Array): void {
    console.log('DesktopGraphRenderer: Updating', edges.length, 'edges');

    if (!edges.length) return;

    
    if (this.edgeLineSystem) {
      this.edgeLineSystem.dispose();
    }

    
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

      
      const edgeMaterial = new StandardMaterial('edgeMaterial', this.scene);
      edgeMaterial.diffuseColor = new Color3(0.6, 0.6, 0.7);
      edgeMaterial.emissiveColor = new Color3(0.2, 0.2, 0.3);
      edgeMaterial.specularColor = new Color3(0.4, 0.4, 0.5);
      edgeMaterial.alpha = 0.8; 
      this.edgeLineSystem.material = edgeMaterial;
    }
  }

  
  public updateLabels(nodes: any[]): void {
    if (!this.labelTexture) return;

    console.log('DesktopGraphRenderer: Updating', nodes.length, 'labels');

    
    this.labelBlocks.forEach(block => block.dispose());
    this.labelBlocks.clear();

    
    for (const node of nodes) {
      if (node.label) {
        const textBlock = new TextBlock(node.id + '_label', node.label);
        textBlock.color = '#FFFFFF';
        textBlock.fontSize = 14;
        textBlock.outlineWidth = 1;
        textBlock.outlineColor = '#000000';

        
        
        this.labelTexture.addControl(textBlock);
        this.labelBlocks.set(node.id, textBlock);
      }
    }
  }

  
  private getNodePosition(nodeId: string | number, positions?: Float32Array): Vector3 | null {
    
    const nodeIndex = typeof nodeId === 'string' ? parseInt(nodeId, 10) : nodeId;

    
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

    
    const instanceName = `node_${nodeId}`;
    const instance = this.nodeInstances.find(inst => inst.name === instanceName);
    if (instance) {
      return instance.position.clone();
    }

    
    return new Vector3(
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10
    );
  }

  
  public focusOnNode(nodeId: string, animated: boolean = true): void {
    const instanceName = `node_${nodeId}`;
    const instance = this.nodeInstances.find(inst => inst.name === instanceName);

    if (instance) {
      if (animated) {
        
        this.camera.setTarget(instance.position);
      } else {
        
        this.camera.target = instance.position.clone();
      }

      console.log('DesktopGraphRenderer: Focused camera on node', nodeId);
    }
  }

  
  public resetCamera(): void {
    this.camera.setTarget(new Vector3(0, 0, 0));
    this.camera.alpha = Math.PI / 2;
    this.camera.beta = Math.PI / 3;
    this.camera.radius = 15;

    console.log('DesktopGraphRenderer: Camera reset to default');
  }

  
  public getCamera(): ArcRotateCamera {
    return this.camera;
  }

  
  public dispose(): void {
    
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
