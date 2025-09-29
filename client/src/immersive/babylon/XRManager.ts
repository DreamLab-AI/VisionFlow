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

/**
 * Manages WebXR session and interactions for Quest 3 AR
 * Handles immersive AR mode, hand tracking, and controller input
 */
export class XRManager {
  private scene: Scene;
  private camera: UniversalCamera;
  private xrHelper: WebXRExperienceHelper | null = null;
  private handTracking: WebXRHandTracking | null = null;

  constructor(scene: Scene, camera: UniversalCamera) {
    this.scene = scene;
    this.camera = camera;
    this.initializeXR();
  }

  private async initializeXR(): Promise<void> {
    // First, check WebXR capabilities
    logger.info('====== XRManager: WebXR Initialization Starting ======');
    logger.info('User Agent: ' + navigator.userAgent);
    logger.info('Protocol: ' + window.location.protocol);
    logger.info('Hostname: ' + window.location.hostname);

    console.log('====== XRManager: WebXR Initialization Starting ======');
    console.log('User Agent:', navigator.userAgent);
    console.log('Protocol:', window.location.protocol);
    console.log('Hostname:', window.location.hostname);

    // Check for WebXR support
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

      // Create default XR experience for Quest 3 AR
      // The default UI button will be created automatically by Babylon
      const xrOptions = {
        floorMeshes: [], // No floor mesh for AR passthrough
        uiOptions: {
          sessionMode: 'immersive-ar', // Request AR mode for passthrough
          referenceSpaceType: 'local-floor',
          // Let Babylon handle the button creation - this is what Quest browser expects
          customButtons: undefined // Use default button
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

      // Log button details
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

      // Configure for immersive AR mode
      if (this.xrHelper.baseExperience) {
        console.log('Configuring XR features...');

        // Listen for XR state changes
        this.xrHelper.baseExperience.onStateChangedObservable.add((state) => {
          console.log('🔄 XR State Changed:', WebXRState[state]);
          if (state === WebXRState.IN_XR) {
            console.log('✅ Successfully entered XR mode!');
          } else if (state === WebXRState.NOT_IN_XR) {
            console.log('⬅️ Exited XR mode');
          }
        });
        // Enable hand tracking
        const handTrackingFeature = this.xrHelper.baseExperience.featuresManager.enableFeature(
          WebXRHandTracking.Name,
          'latest'
        ) as WebXRHandTracking;

        if (handTrackingFeature) {
          this.handTracking = handTrackingFeature;
          this.setupHandInteractions();
        }

        // Setup controller interactions
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

      // Log specific error information
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

    // Track hand joint data for interaction
    this.handTracking.onHandAddedObservable.add((hand) => {
      console.log('Hand detected:', hand.handedness);

      // Get index finger tip joint for pointing/selection
      const indexTip = hand.getJoint('index-finger-tip');
      if (indexTip) {
        // Create a continuous observer for finger tip position
        this.scene.onBeforeRenderObservable.add(() => {
          this.handleFingerTipInteraction(indexTip, hand.handedness);
        });
      }
    });

    console.log('XRManager: Hand interactions configured');
  }

  private setupControllerInteractions(): void {
    if (!this.xrHelper?.baseExperience) return;

    // Listen for controller input
    this.xrHelper.baseExperience.onXRInputSourceObservable.add((inputSource) => {
      console.log('Controller connected:', inputSource.uniqueId);

      // Setup trigger events for selection
      inputSource.onComponentChangedObservable.add((component) => {
        if (component.id === 'xr-standard-trigger') {
          this.handleTriggerAction(component, inputSource);
        }
        if (component.id === 'xr-standard-squeeze') {
          this.handleSqueezeAction(component, inputSource);
        }
      });

      // Setup pointer ray for selection
      inputSource.onPointerRayChangedObservable.add((ray) => {
        this.handleControllerRay(ray, inputSource);
      });
    });

    console.log('XRManager: Controller interactions configured');
  }

  private handleFingerTipInteraction(indexTip: any, handedness: string): void {
    // Create ray from finger tip for node selection
    const tipPosition = indexTip.position;
    const tipForward = indexTip.forward || Vector3.Forward();

    const ray = new Ray(tipPosition, tipForward);
    this.performRaySelection(ray, 'hand_' + handedness);
  }

  private handleTriggerAction(component: any, inputSource: any): void {
    if (component.pressed) {
      // Trigger pressed - start selection/dragging
      this.startNodeInteraction(inputSource);
    } else {
      // Trigger released - end selection/dragging
      this.endNodeInteraction(inputSource);
    }
  }

  private handleSqueezeAction(component: any, inputSource: any): void {
    if (component.pressed) {
      // Squeeze for additional actions like menu toggle
      this.toggleUIPanel();
    }
  }

  private handleControllerRay(ray: Ray, inputSource: any): void {
    this.performRaySelection(ray, inputSource.uniqueId);
  }

  private performRaySelection(ray: Ray, sourceId: string): void {
    // Perform ray casting to detect node intersections
    const hit = this.scene.pickWithRay(ray);

    if (hit?.pickedMesh) {
      // Check if the picked mesh is a graph node
      const nodeId = this.getNodeIdFromMesh(hit.pickedMesh);
      if (nodeId) {
        this.selectNode(nodeId, sourceId);
      }
    }
  }

  private getNodeIdFromMesh(mesh: any): string | null {
    // Extract node ID from mesh metadata or name
    return mesh.metadata?.nodeId || mesh.name?.replace('node_', '') || null;
  }

  private selectNode(nodeId: string, sourceId: string): void {
    console.log(`Node ${nodeId} selected by ${sourceId}`);

    // Dispatch selection event
    this.scene.onNodeSelectedObservable?.notifyObservers({
      nodeId,
      sourceId,
      timestamp: Date.now()
    });
  }

  private startNodeInteraction(inputSource: any): void {
    // Start dragging the currently selected node
    console.log('Starting node interaction with', inputSource.uniqueId);

    // TODO: Pin node in physics simulation
    // TODO: Track input source for continuous position updates
  }

  private endNodeInteraction(inputSource: any): void {
    // End dragging interaction
    console.log('Ending node interaction with', inputSource.uniqueId);

    // TODO: Unpin node in physics simulation
  }

  private toggleUIPanel(): void {
    // Toggle the 3D UI panel visibility
    console.log('Toggling UI panel');

    // TODO: Communicate with XRUI component
  }

  /**
   * Gets the XR helper for external access
   * The built-in UI button will handle entering XR automatically
   */
  public getXRHelper(): WebXRExperienceHelper | null {
    return this.xrHelper;
  }

  /**
   * Check if currently in XR session
   */
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