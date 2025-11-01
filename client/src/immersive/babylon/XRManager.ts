import {
  Scene,
  UniversalCamera,
  WebXRExperienceHelper,
  WebXRHandTracking,
  WebXRInputSource,
  WebXRState,
  Vector3,
  Ray
} from '@babylonjs/core';
import { createRemoteLogger } from '../../services/remoteLogger';

const logger = createRemoteLogger('XRManager');


export class XRManager {
  private scene: Scene;
  private camera: UniversalCamera;
  private xrHelper: WebXRExperienceHelper | null = null;
  private handTracking: WebXRHandTracking | null = null;

  
  private pinnedNode: string | null = null;
  private activeInputSource: WebXRInputSource | null = null;

  
  private uiPanel: any | null = null;

  constructor(scene: Scene, camera: UniversalCamera) {
    this.scene = scene;
    this.camera = camera;
    this.initializeXR();
  }

  private async initializeXR(): Promise<void> {
    
    logger.info('====== XRManager: WebXR Initialization Starting ======');
    logger.info('User Agent: ' + navigator.userAgent);
    logger.info('Protocol: ' + window.location.protocol);
    logger.info('Hostname: ' + window.location.hostname);

    console.log('====== XRManager: WebXR Initialization Starting ======');
    console.log('User Agent:', navigator.userAgent);
    console.log('Protocol:', window.location.protocol);
    console.log('Hostname:', window.location.hostname);

    
    if ('xr' in navigator) {
      logger.info('✅ WebXR API is available');
      console.log('✅ WebXR API is available');
      try {
        const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
        const arSupported = await navigator.xr.isSessionSupported('immersive-ar');
        logger.info('VR Support: ' + (vrSupported ? '✅ YES' : '❌ NO'));
        logger.info('AR Support: ' + (arSupported ? '✅ YES' : '❌ NO'));
        console.log('VR Support:', vrSupported ? '✅ YES' : '❌ NO');
        console.log('AR Support:', arSupported ? '✅ YES' : '❌ NO');
      } catch (checkError) {
        logger.error('Error checking XR support', checkError);
        console.error('Error checking XR support:', checkError);
      }
    } else {
      logger.error('❌ WebXR API is NOT available in this browser');
      console.error('❌ WebXR API is NOT available in this browser');
    }

    try {
      console.log('Creating Babylon XR experience with immersive-ar mode...');

      
      
      const xrOptions = {
        floorMeshes: [], 
        uiOptions: {
          sessionMode: 'immersive-ar', 
          referenceSpaceType: 'local-floor',
          
          customButtons: undefined 
        },
        optionalFeatures: true
      };

      console.log('XR Options:', JSON.stringify(xrOptions, null, 2));
      this.xrHelper = await this.scene.createDefaultXRExperienceAsync(xrOptions);

      logger.info('✅ XR Helper created successfully');
      logger.info('XR Helper baseExperience exists: ' + !!this.xrHelper.baseExperience);
      logger.info('XR UI exists: ' + !!this.xrHelper.enterExitUI);

      console.log('✅ XR Helper created successfully');
      console.log('XR Helper baseExperience exists:', !!this.xrHelper.baseExperience);
      console.log('XR UI exists:', !!this.xrHelper.enterExitUI);

      
      if (this.xrHelper.enterExitUI) {
        logger.info('XR Button Details:');
        console.log('XR Button Details:');
        const button = (this.xrHelper.enterExitUI as any).renderTarget?.firstChild;
        if (button) {
          const buttonInfo = {
            exists: !!button,
            position: button.style?.position,
            display: button.style?.display,
            zIndex: button.style?.zIndex,
            visibility: button.style?.visibility
          };
          logger.info('Button element info', buttonInfo);
          console.log('- Button element exists:', !!button);
          console.log('- Button position:', button.style?.position);
          console.log('- Button display:', button.style?.display);
          console.log('- Button z-index:', button.style?.zIndex);
          console.log('- Button visibility:', button.style?.visibility);
        } else {
          logger.warn('⚠️ XR button element not found in DOM');
          console.warn('⚠️ XR button element not found in DOM');
        }
      }

      
      if (this.xrHelper.baseExperience) {
        console.log('Configuring XR features...');

        
        this.xrHelper.baseExperience.onStateChangedObservable.add((state) => {
          console.log('🔄 XR State Changed:', WebXRState[state]);
          if (state === WebXRState.IN_XR) {
            console.log('✅ Successfully entered XR mode!');
          } else if (state === WebXRState.NOT_IN_XR) {
            console.log('⬅️ Exited XR mode');
          }
        });
        
        const handTrackingFeature = this.xrHelper.baseExperience.featuresManager.enableFeature(
          WebXRHandTracking.Name,
          'latest'
        ) as WebXRHandTracking;

        if (handTrackingFeature) {
          this.handTracking = handTrackingFeature;
          this.setupHandInteractions();
        }

        
        this.setupControllerInteractions();
      }

      console.log('====== XRManager: WebXR Initialization Complete ======');
      console.log('Summary:');
      console.log('- XR Helper:', this.xrHelper ? '✅' : '❌');
      console.log('- Base Experience:', this.xrHelper?.baseExperience ? '✅' : '❌');
      console.log('- Enter/Exit UI:', this.xrHelper?.enterExitUI ? '✅' : '❌');
      console.log('- Hand Tracking:', this.handTracking ? '✅' : '❌');
    } catch (error) {
      console.error('❌ XRManager: Failed to initialize WebXR');
      console.error('Error details:', error);

      
      if (error instanceof Error) {
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);

        if (error.message.includes('HTTPS')) {
          console.warn('🔒 HTTPS Required: WebXR requires a secure context (HTTPS)');
          console.warn('For Quest 3 testing, you need to:');
          console.warn('1. Use HTTPS (even with self-signed certificate)');
          console.warn('2. Or use localhost with port forwarding via adb');
        } else if (error.message.includes('session')) {
          console.warn('🎮 Session Error: Browser may not support the requested XR session mode');
        } else if (error.message.includes('permission')) {
          console.warn('🔐 Permission Error: User may have denied XR permissions');
        }
      }

      console.log('====== XRManager: Initialization Failed ======');
    }
  }

  private setupHandInteractions(): void {
    if (!this.handTracking) return;

    
    this.handTracking.onHandAddedObservable.add((hand) => {
      console.log('Hand detected:', hand.handedness);

      
      const indexTip = hand.getJoint('index-finger-tip');
      if (indexTip) {
        
        this.scene.onBeforeRenderObservable.add(() => {
          this.handleFingerTipInteraction(indexTip, hand.handedness);
        });
      }
    });

    console.log('XRManager: Hand interactions configured');
  }

  private setupControllerInteractions(): void {
    if (!this.xrHelper?.baseExperience) return;

    
    this.xrHelper.baseExperience.onXRInputSourceObservable.add((inputSource) => {
      console.log('Controller connected:', inputSource.uniqueId);

      
      inputSource.onComponentChangedObservable.add((component) => {
        if (component.id === 'xr-standard-trigger') {
          this.handleTriggerAction(component, inputSource);
        }
        if (component.id === 'xr-standard-squeeze') {
          this.handleSqueezeAction(component, inputSource);
        }
      });

      
      inputSource.onPointerRayChangedObservable.add((ray) => {
        this.handleControllerRay(ray, inputSource);
      });
    });

    console.log('XRManager: Controller interactions configured');
  }

  private handleFingerTipInteraction(indexTip: any, handedness: string): void {
    
    const tipPosition = indexTip.position;
    const tipForward = indexTip.forward || Vector3.Forward();

    const ray = new Ray(tipPosition, tipForward);
    this.performRaySelection(ray, 'hand_' + handedness);
  }

  private handleTriggerAction(component: any, inputSource: any): void {
    if (component.pressed) {
      
      this.startNodeInteraction(inputSource);
    } else {
      
      this.endNodeInteraction(inputSource);
    }
  }

  private handleSqueezeAction(component: any, inputSource: any): void {
    if (component.pressed) {
      
      this.toggleUIPanel();
    }
  }

  private handleControllerRay(ray: Ray, inputSource: any): void {
    this.performRaySelection(ray, inputSource.uniqueId);
  }

  private performRaySelection(ray: Ray, sourceId: string): void {
    
    const hit = this.scene.pickWithRay(ray);

    if (hit?.pickedMesh) {
      
      const nodeId = this.getNodeIdFromMesh(hit.pickedMesh);
      if (nodeId) {
        this.selectNode(nodeId, sourceId);
      }
    }
  }

  private getNodeIdFromMesh(mesh: any): string | null {
    
    return mesh.metadata?.nodeId || mesh.name?.replace('node_', '') || null;
  }

  private selectNode(nodeId: string, sourceId: string): void {
    console.log(`Node ${nodeId} selected by ${sourceId}`);

    
    this.scene.onNodeSelectedObservable?.notifyObservers({
      nodeId,
      sourceId,
      timestamp: Date.now()
    });
  }

  private startNodeInteraction(inputSource: any): void {
    
    console.log('Starting node interaction with', inputSource.uniqueId);

    
    
    if (this.scene.onNodeSelectedObservable) {
      const selectedNodeEvent = this.scene.onNodeSelectedObservable as any;
      if (selectedNodeEvent._observers && selectedNodeEvent._observers.length > 0) {
        const lastEvent = selectedNodeEvent._lastNotified;
        if (lastEvent && lastEvent.nodeId) {
          this.pinnedNode = lastEvent.nodeId;
          console.log('XRManager: Pinned node for dragging:', this.pinnedNode);

          
          if (this.scene.onPhysicsPinObservable) {
            (this.scene.onPhysicsPinObservable as any).notifyObservers({
              nodeId: this.pinnedNode,
              pinned: true,
              timestamp: Date.now()
            });
          }
        }
      }
    }

    
    this.activeInputSource = inputSource;

    
    const updateObserver = this.scene.onBeforeRenderObservable.add(() => {
      if (this.activeInputSource && this.pinnedNode) {
        
        const grip = this.activeInputSource.grip || this.activeInputSource.pointer;
        if (grip) {
          const position = grip.position;

          
          if (this.scene.onNodePositionUpdateObservable) {
            (this.scene.onNodePositionUpdateObservable as any).notifyObservers({
              nodeId: this.pinnedNode,
              position: { x: position.x, y: position.y, z: position.z },
              source: 'xr-input',
              timestamp: Date.now()
            });
          }
        }
      }
    });

    
    (this.scene as any)._xrDragUpdateObserver = updateObserver;
  }

  private endNodeInteraction(inputSource: any): void {
    
    console.log('Ending node interaction with', inputSource.uniqueId);

    
    if (this.pinnedNode) {
      console.log('XRManager: Unpinning node:', this.pinnedNode);

      
      if (this.scene.onPhysicsPinObservable) {
        (this.scene.onPhysicsPinObservable as any).notifyObservers({
          nodeId: this.pinnedNode,
          pinned: false,
          timestamp: Date.now()
        });
      }

      this.pinnedNode = null;
    }

    
    this.activeInputSource = null;

    
    if ((this.scene as any)._xrDragUpdateObserver) {
      this.scene.onBeforeRenderObservable.remove((this.scene as any)._xrDragUpdateObserver);
      (this.scene as any)._xrDragUpdateObserver = null;
    }
  }

  private toggleUIPanel(): void {
    
    console.log('XRManager: Toggling UI panel');

    
    if (this.uiPanel) {
      
      if (typeof this.uiPanel.toggle === 'function') {
        this.uiPanel.toggle();
        console.log('XRManager: UI panel toggled via direct method');
      } else if (typeof this.uiPanel.setVisibility === 'function') {
        
        const currentVisibility = this.uiPanel.isVisible || false;
        this.uiPanel.setVisibility(!currentVisibility);
        console.log('XRManager: UI panel toggled via setVisibility:', !currentVisibility);
      } else {
        console.warn('XRManager: UI panel does not have toggle or setVisibility method');
      }
    } else {
      console.warn('XRManager: UI panel reference not set. Use setUIPanel() to connect XRUI instance.');
    }

    
    if (this.scene.onUIToggleObservable) {
      (this.scene.onUIToggleObservable as any).notifyObservers({
        source: 'xr-manager',
        timestamp: Date.now()
      });
    }
  }

  
  public setUIPanel(uiPanel: any): void {
    this.uiPanel = uiPanel;
    console.log('XRManager: UI panel reference set');
  }

  
  public getXRHelper(): WebXRExperienceHelper | null {
    return this.xrHelper;
  }

  
  public isInXR(): boolean {
    return this.xrHelper?.baseExperience?.state === WebXRState.IN_XR;
  }

  public async exitXR(): Promise<void> {
    if (this.xrHelper?.baseExperience) {
      await this.xrHelper.baseExperience.exitXRAsync();
    }
  }

  public dispose(): void {
    if (this.xrHelper) {
      this.xrHelper.dispose();
    }
  }
}