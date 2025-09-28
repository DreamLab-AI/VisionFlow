import {
  Scene,
  UniversalCamera,
  WebXRExperienceHelper,
  WebXRHandTracking,
  WebXRInputSource,
  Vector3,
  Ray
} from '@babylonjs/core';

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
    try {
      // Attempt to create XR experience regardless of protocol
      // Quest 3 browser may allow WebXR in development scenarios
      console.log('XRManager: Attempting to initialize WebXR...');

      // Create default XR experience for Quest 3 AR
      this.xrHelper = await this.scene.createDefaultXRExperienceAsync({
        floorMeshes: [], // No floor mesh for AR passthrough
        optionalFeatures: true,
        // Disable the default UI that shows HTTPS warning
        uiOptions: {
          sessionMode: 'immersive-ar',
          referenceSpaceType: 'local-floor',
          ignoreHTTPSWarning: true // Try to ignore HTTPS warning
        }
      } as any);

      // Configure for immersive AR mode
      if (this.xrHelper.baseExperience) {
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

      console.log('XRManager: WebXR initialized successfully');
    } catch (error) {
      // This is expected in non-HTTPS environments
      if (error instanceof Error && error.message.includes('HTTPS')) {
        console.info('XRManager: Running in non-HTTPS mode. WebXR features disabled for development.');
      } else {
        console.warn('XRManager: WebXR not available or failed to initialize', error);
      }
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

  public async enterXR(): Promise<void> {
    if (!this.xrHelper?.baseExperience) {
      console.warn('XRManager: XR not initialized, attempting to create session directly');

      // Try to create a session directly if the helper isn't available
      if ('xr' in navigator && navigator.xr) {
        try {
          // Check what's supported
          const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
          const arSupported = await navigator.xr.isSessionSupported('immersive-ar');

          console.log('XRManager: VR supported:', vrSupported, 'AR supported:', arSupported);

          // Try AR first if supported, then fall back to VR
          const sessionMode = arSupported ? 'immersive-ar' : vrSupported ? 'immersive-vr' : null;

          if (sessionMode) {
            const session = await navigator.xr.requestSession(sessionMode, {
              requiredFeatures: ['local-floor'],
              optionalFeatures: ['hand-tracking', 'bounded-floor', 'unbounded']
            });

            console.log('XRManager: Session started in', sessionMode, 'mode');

            // TODO: Set up the session with Babylon.js
            // This would need proper integration with the Babylon.js scene
          } else {
            console.error('XRManager: No immersive modes supported');
          }
        } catch (error) {
          console.error('XRManager: Failed to start XR session:', error);
        }
      }
      return;
    }

    try {
      // First try immersive-ar for Quest 3 passthrough
      console.log('XRManager: Attempting to enter immersive-ar mode...');
      await this.xrHelper.baseExperience.enterXRAsync('immersive-ar', 'local-floor', {
        optionalFeatures: ['hand-tracking', 'mesh-detection', 'plane-detection']
      });
    } catch (arError) {
      console.warn('XRManager: immersive-ar failed, trying immersive-vr:', arError);

      try {
        // Fall back to immersive-vr if AR is not available
        await this.xrHelper.baseExperience.enterXRAsync('immersive-vr', 'local-floor', {
          optionalFeatures: ['hand-tracking', 'bounded-floor', 'unbounded']
        });
        console.log('XRManager: Successfully entered immersive-vr mode');
      } catch (vrError) {
        console.error('XRManager: Failed to enter any immersive mode:', vrError);
        throw vrError;
      }
    }
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