import * as BABYLON from '@babylonjs/core';
import { Color3, Color4, DirectionalLight } from '@babylonjs/core';
import { GraphRenderer } from './GraphRenderer';
import { XRManager } from './XRManager';
import { XRUI } from './XRUI';


export class BabylonScene {
  private engine: BABYLON.Engine;
  private scene: BABYLON.Scene;
  private camera: BABYLON.UniversalCamera;
  private graphRenderer: GraphRenderer;
  public xrManager: XRManager; 
  private xrUI: XRUI;

  constructor(canvas: HTMLCanvasElement) {
    
    this.engine = new BABYLON.Engine(canvas, true, {
      preserveDrawingBuffer: true,
      stencil: true,
      antialias: true
    });

    
    this.scene = new BABYLON.Scene(this.engine);
    
    this.scene.clearColor = new BABYLON.Color4(0, 0, 0, 0);

    
    this.camera = new BABYLON.UniversalCamera('camera', new BABYLON.Vector3(0, 1.6, -5), this.scene);
    this.camera.setTarget(BABYLON.Vector3.Zero());
    this.camera.attachControl(canvas, true);

    
    const hemisphericLight = new BABYLON.HemisphericLight('hemisphericLight', new BABYLON.Vector3(0, 1, 0), this.scene);
    hemisphericLight.intensity = 1.2;
    hemisphericLight.groundColor = new BABYLON.Color3(0.2, 0.2, 0.3);

    
    const directionalLight = new BABYLON.DirectionalLight('directionalLight', new BABYLON.Vector3(-1, -2, -1), this.scene);
    directionalLight.position = new BABYLON.Vector3(3, 9, 3);
    directionalLight.intensity = 0.8;

    
    this.scene.ambientColor = new BABYLON.Color3(0.3, 0.3, 0.4);

    
    this.graphRenderer = new GraphRenderer(this.scene);
    this.xrManager = new XRManager(this.scene, this.camera);
    this.xrUI = new XRUI(this.scene);

    
    window.addEventListener('resize', () => {
      this.engine.resize();
    });
  }

  
  public updateGraph(graphData: any, nodePositions?: Float32Array): void {
    if (graphData) {
      this.graphRenderer.updateNodes(graphData.nodes || [], nodePositions);
      this.graphRenderer.updateEdges(graphData.edges || [], nodePositions);
      this.graphRenderer.updateLabels(graphData.nodes || []);
    }
  }

  
  public setBotsData(data: any): void {
    
    if (data) {
      
      if (data.graphData) {
        this.updateGraph(data.graphData, data.nodePositions);
      } else {
        
        const graphData = {
          nodes: data.nodes || [],
          edges: data.edges || []
        };
        this.updateGraph(graphData, data.nodePositions);
      }
    }
  }

  
  public setSettings(settings: any): void {
    
    console.log('Settings updated:', settings);
  }

  
  public getScene(): BABYLON.Scene {
    return this.scene;
  }

  
  public getEngine(): BABYLON.Engine {
    return this.engine;
  }

  
  public run(): void {
    this.engine.runRenderLoop(() => {
      this.scene.render();
    });
  }

  
  public dispose(): void {
    this.graphRenderer.dispose();
    this.xrManager.dispose();
    this.xrUI.dispose();
    this.scene.dispose();
    this.engine.dispose();
  }
}