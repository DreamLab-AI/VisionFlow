import * as BABYLON from '@babylonjs/core';
import { GraphRenderer } from './GraphRenderer';
import { XRManager } from './XRManager';
import { XRUI } from './XRUI';

/**
 * Core Babylon.js scene management class
 * Handles engine initialization, scene setup, and component coordination
 */
export class BabylonScene {
  private engine: BABYLON.Engine;
  private scene: BABYLON.Scene;
  private camera: BABYLON.UniversalCamera;
  private graphRenderer: GraphRenderer;
  public xrManager: XRManager; // Make public for access from ImmersiveApp
  private xrUI: XRUI;

  constructor(canvas: HTMLCanvasElement) {
    // Initialize Babylon.js engine
    this.engine = new BABYLON.Engine(canvas, true, {
      preserveDrawingBuffer: true,
      stencil: true,
      antialias: true
    });

    // Create the 3D scene
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color3(0.1, 0.1, 0.1);

    // Setup camera
    this.camera = new BABYLON.UniversalCamera('camera', new BABYLON.Vector3(0, 1.6, -5), this.scene);
    this.camera.setTarget(BABYLON.Vector3.Zero());
    this.camera.attachControl(canvas, true);

    // Add lighting
    const light = new BABYLON.HemisphericLight('light', new BABYLON.Vector3(0, 1, 0), this.scene);
    light.intensity = 0.7;

    // Initialize components
    this.graphRenderer = new GraphRenderer(this.scene);
    this.xrManager = new XRManager(this.scene, this.camera);
    this.xrUI = new XRUI(this.scene);

    // Handle window resize
    window.addEventListener('resize', () => {
      this.engine.resize();
    });
  }

  /**
   * Update graph visualization with new data
   */
  public updateGraph(graphData: any, nodePositions?: Float32Array): void {
    if (graphData) {
      this.graphRenderer.updateNodes(graphData.nodes || [], nodePositions);
      this.graphRenderer.updateEdges(graphData.edges || [], nodePositions);
      this.graphRenderer.updateLabels(graphData.nodes || []);
    }
  }

  /**
   * Set bots data (alias for compatibility)
   */
  public setBotsData(data: any): void {
    // Handle the data format from ImmersiveApp
    if (data) {
      // If we have graphData directly, use it
      if (data.graphData) {
        this.updateGraph(data.graphData, data.nodePositions);
      } else {
        // Otherwise try to construct it
        const graphData = {
          nodes: data.nodes || [],
          edges: data.edges || []
        };
        this.updateGraph(graphData, data.nodePositions);
      }
    }
  }

  /**
   * Set settings (for compatibility)
   */
  public setSettings(settings: any): void {
    // Store settings for future use
    console.log('Settings updated:', settings);
  }

  /**
   * Get the scene instance for external access
   */
  public getScene(): BABYLON.Scene {
    return this.scene;
  }

  /**
   * Get the engine instance for external access
   */
  public getEngine(): BABYLON.Engine {
    return this.engine;
  }

  /**
   * Start the render loop
   */
  public run(): void {
    this.engine.runRenderLoop(() => {
      this.scene.render();
    });
  }

  /**
   * Dispose of all resources
   */
  public dispose(): void {
    this.graphRenderer.dispose();
    this.xrManager.dispose();
    this.xrUI.dispose();
    this.scene.dispose();
    this.engine.dispose();
  }
}