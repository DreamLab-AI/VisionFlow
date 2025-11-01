import {
  Scene,
  Mesh,
  MeshBuilder,
  StandardMaterial,
  Vector3,
  Color3
} from '@babylonjs/core';
import * as BABYLON from '@babylonjs/core';
import {
  AdvancedDynamicTexture,
  Control,
  StackPanel,
  Button,
  Slider,
  Checkbox,
  TextBlock
} from '@babylonjs/gui';

// Import settings store types for proper integration
interface Settings {
  graph?: {
    nodeSize?: number;
    showLabels?: boolean;
    edgeOpacity?: number;
  };
  visualization?: {
    showBots?: boolean;
    showEdges?: boolean;
  };
  performance?: {
    maxNodes?: number;
    enablePhysics?: boolean;
  };
}


export class XRUI {
  private scene: Scene;
  private uiPlane: Mesh | null = null;
  private uiTexture: AdvancedDynamicTexture | null = null;
  private mainPanel: StackPanel | null = null;
  private settings: Settings = {};
  private onSettingsChange?: (path: string, value: any) => void;

  constructor(scene: Scene) {
    this.scene = scene;
    this.initialize();
  }

  
  public setSettingsChangeCallback(callback: (path: string, value: any) => void): void {
    this.onSettingsChange = callback;
  }

  private initialize(): void {
    
    this.uiPlane = MeshBuilder.CreatePlane('uiPlane', { width: 2, height: 1.5 }, this.scene);
    this.uiPlane.position = new Vector3(2, 1.6, 0); 

    
    const uiMaterial = new StandardMaterial('uiMaterial', this.scene);
    uiMaterial.emissiveColor = new BABYLON.Color3(0.1, 0.1, 0.15); 
    uiMaterial.diffuseColor = new BABYLON.Color3(0.15, 0.15, 0.2);
    uiMaterial.specularColor = new BABYLON.Color3(0, 0, 0);
    this.uiPlane.material = uiMaterial;

    
    this.uiTexture = AdvancedDynamicTexture.CreateForMesh(this.uiPlane);
    this.uiTexture.background = "rgba(30, 30, 40, 0.9)"; 

    
    this.createControlPanel();
  }

  private createControlPanel(): void {
    if (!this.uiTexture) return;

    
    this.mainPanel = new StackPanel('mainPanel');
    this.mainPanel.widthInPixels = 400;
    this.mainPanel.heightInPixels = 600;
    this.mainPanel.background = 'rgba(0, 0, 0, 0.8)';
    this.mainPanel.cornerRadius = 10;
    this.uiTexture.addControl(this.mainPanel);

    
    const title = new TextBlock('title', 'VisionFlow Settings');
    title.heightInPixels = 50;
    title.color = 'white';
    title.fontSize = 24;
    this.mainPanel.addControl(title);

    
    
    
    

    this.createGraphControls();
    this.createRenderingControls();
    this.createInteractionControls();
  }

  private createGraphControls(): void {
    if (!this.mainPanel) return;

    
    const graphTitle = new TextBlock('graphTitle', 'Graph Visualization');
    graphTitle.heightInPixels = 30;
    graphTitle.color = 'cyan';
    graphTitle.fontSize = 18;
    this.mainPanel.addControl(graphTitle);

    
    const nodeSizeSlider = new Slider('nodeSize');
    nodeSizeSlider.heightInPixels = 30;
    nodeSizeSlider.minimum = 0.1;
    nodeSizeSlider.maximum = 2.0;
    nodeSizeSlider.value = this.settings.graph?.nodeSize || 1.0;
    nodeSizeSlider.color = 'white';
    nodeSizeSlider.background = 'gray';
    this.mainPanel.addControl(nodeSizeSlider);

    
    nodeSizeSlider.onValueChangedObservable.add((value) => {
      if (this.onSettingsChange) {
        this.onSettingsChange('graph.nodeSize', value);
      }
    });

    
    const edgeOpacitySlider = new Slider('edgeOpacity');
    edgeOpacitySlider.heightInPixels = 30;
    edgeOpacitySlider.minimum = 0.0;
    edgeOpacitySlider.maximum = 1.0;
    edgeOpacitySlider.value = this.settings.graph?.edgeOpacity || 0.7;
    edgeOpacitySlider.color = 'white';
    edgeOpacitySlider.background = 'gray';
    this.mainPanel.addControl(edgeOpacitySlider);

    edgeOpacitySlider.onValueChangedObservable.add((value) => {
      if (this.onSettingsChange) {
        this.onSettingsChange('graph.edgeOpacity', value);
      }
    });
  }

  private createRenderingControls(): void {
    if (!this.mainPanel) return;

    
    const renderTitle = new TextBlock('renderTitle', 'Rendering Options');
    renderTitle.heightInPixels = 30;
    renderTitle.color = 'cyan';
    renderTitle.fontSize = 18;
    this.mainPanel.addControl(renderTitle);

    
    const showLabelsCheck = new Checkbox('showLabels');
    showLabelsCheck.heightInPixels = 30;
    showLabelsCheck.color = 'white';
    showLabelsCheck.background = 'gray';
    showLabelsCheck.isChecked = this.settings.graph?.showLabels !== false;
    this.mainPanel.addControl(showLabelsCheck);

    
    showLabelsCheck.onIsCheckedChangedObservable.add((value) => {
      if (this.onSettingsChange) {
        this.onSettingsChange('graph.showLabels', value);
      }
    });

    
    const showBotsCheck = new Checkbox('showBots');
    showBotsCheck.heightInPixels = 30;
    showBotsCheck.color = 'white';
    showBotsCheck.background = 'gray';
    showBotsCheck.isChecked = this.settings.visualization?.showBots !== false;
    this.mainPanel.addControl(showBotsCheck);

    showBotsCheck.onIsCheckedChangedObservable.add((value) => {
      if (this.onSettingsChange) {
        this.onSettingsChange('visualization.showBots', value);
      }
    });
  }

  private createInteractionControls(): void {
    if (!this.mainPanel) return;

    
    const interactionTitle = new TextBlock('interactionTitle', 'Interaction');
    interactionTitle.heightInPixels = 30;
    interactionTitle.color = 'cyan';
    interactionTitle.fontSize = 18;
    this.mainPanel.addControl(interactionTitle);

    
    const resetButton = Button.CreateSimpleButton('resetCamera', 'Reset View');
    resetButton.heightInPixels = 40;
    resetButton.color = 'white';
    resetButton.background = 'blue';
    this.mainPanel.addControl(resetButton);

    resetButton.onPointerClickObservable.add(() => {
      
      console.log('Reset camera view');
    });
  }

  
  public updateFromSettings(settings: Settings): void {
    this.settings = settings;
    console.log('XRUI: Updating from settings', settings);

    
    this.syncUIWithSettings();
  }

  private syncUIWithSettings(): void {
    if (!this.uiTexture) return;

    
    const nodeSizeSlider = this.uiTexture.getControlByName('nodeSize') as Slider;
    if (nodeSizeSlider && this.settings.graph?.nodeSize !== undefined) {
      nodeSizeSlider.value = this.settings.graph.nodeSize;
    }

    
    const edgeOpacitySlider = this.uiTexture.getControlByName('edgeOpacity') as Slider;
    if (edgeOpacitySlider && this.settings.graph?.edgeOpacity !== undefined) {
      edgeOpacitySlider.value = this.settings.graph.edgeOpacity;
    }

    
    const showLabelsCheck = this.uiTexture.getControlByName('showLabels') as Checkbox;
    if (showLabelsCheck && this.settings.graph?.showLabels !== undefined) {
      showLabelsCheck.isChecked = this.settings.graph.showLabels;
    }

    
    const showBotsCheck = this.uiTexture.getControlByName('showBots') as Checkbox;
    if (showBotsCheck && this.settings.visualization?.showBots !== undefined) {
      showBotsCheck.isChecked = this.settings.visualization.showBots;
    }
  }

  
  public setVisible(visible: boolean): void {
    if (this.uiPlane) {
      this.uiPlane.setEnabled(visible);
    }
  }

  
  public showNodeDetails(nodeId: string): void {
    
    console.log('Showing details for node:', nodeId);
  }

  
  public showWelcomeMessage(): void {
    
    console.log('Welcome to VisionFlow immersive mode!');
  }

  
  public onExitRequest(callback: () => void): void {
    
    const exitButton = this.uiTexture?.getControlByName('resetCamera') as Button;
    if (exitButton) {
      exitButton.onPointerClickObservable.add(callback);
    }
  }

  public dispose(): void {
    if (this.uiTexture) {
      this.uiTexture.dispose();
    }
    if (this.uiPlane) {
      this.uiPlane.dispose();
    }
  }
}