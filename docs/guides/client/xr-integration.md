---
title: XR/AR Integration Guide
description: VisionFlow supports **Quest 3 VR/AR** through Babylon. js and WebXR.
category: guide
tags:
  - architecture
  - design
  - api
  - api
  - http
related-docs:
  - guides/client/three-js-rendering.md
  - guides/client/state-management.md
  - QUICK_NAVIGATION.md
  - README.md
  - concepts/architecture/core/client.md
updated-date: 2025-12-18
difficulty-level: beginner
---

# XR/AR Integration Guide

**Target Audience**: Frontend developers implementing VR/AR features
**Prerequisites**: WebXR basics, Babylon.js or Three.js experience
**Last Updated**: 2025-12-02

---

## Overview

VisionFlow supports **Quest 3 VR/AR** through Babylon.js and WebXR. This guide explains device detection, immersive scene setup, controller handling, and known issues with XR support.

**⚠️ Status**: XR support is **experimental**. Quest 3 detection is fragile, and performance varies significantly across devices.

---

## Table of Contents

1. [Device Detection](#device-detection)
2. [Babylon.js XR Scene](#babylonjs-xr-scene)
3. [Controller Input Handling](#controller-input-handling)
4. [Performance Optimizations](#performance-optimizations)
5. [Known Issues & Workarounds](#known-issues--workarounds)
6. [Browser Compatibility](#browser-compatibility)

---

## Device Detection

### Quest 3 User-Agent Sniffing

**File**: `client/src/app/App.tsx`

```typescript
const shouldUseImmersiveClient = (): boolean => {
  const userAgent = navigator.userAgent;

  // Quest 3 detection (fragile!)
  const isQuest3Browser = userAgent.includes('Quest 3') ||
                          userAgent.includes('Quest3') ||
                          userAgent.includes('OculusBrowser') ||
                          (userAgent.includes('VR') && userAgent.includes('Quest'));

  // Manual override via URL parameter
  const forceQuest3 = window.location.search.includes('force=quest3') ||
                      window.location.search.includes('directar=true') ||
                      window.location.search.includes('immersive=true');

  return (isQuest3Browser || forceQuest3) && initialized;
};

// Render decision
return shouldUseImmersiveClient() ? (
  <ImmersiveApp />  // Babylon.js XR scene
) : (
  <MainLayout />    // Three.js desktop scene
);
```

### User-Agent Strings by Quest Version

| Device | Firmware | User-Agent String |
|--------|----------|-------------------|
| **Quest 3** | v55-v57 | `Quest 3` or `Quest3` |
| **Quest 3** | v58+ | `OculusBrowser` + `VR` |
| **Quest 2** | Latest | `OculusBrowser` (no `Quest 3`) |
| **Quest Pro** | Latest | `OculusBrowser` + `Quest Pro` |

**⚠️ Problem**: User-agent detection is unreliable. Quest 2 devices with `OculusBrowser` trigger false positives.

### Improved Detection with WebXR API

```typescript
// Proposed improvement (not yet implemented)
const shouldUseImmersiveClient = async (): Promise<boolean> => {
  // Check WebXR support
  if (!navigator.xr) {
    return false;
  }

  // Check for immersive-vr session support
  try {
    const supported = await navigator.xr.isSessionSupported('immersive-vr');
    if (!supported) {
      return false;
    }

    // Verify XR features (hand tracking, AR mode)
    const session = await navigator.xr.requestSession('immersive-vr', {
      requiredFeatures: ['local-floor', 'hand-tracking']
    });
    await session.end();

    return true;
  } catch (error) {
    console.warn('WebXR not supported', error);
    return false;
  }
};
```

**Benefits**:
- No user-agent parsing
- Future-proof (works with Quest 4, Pico, etc.)
- Checks actual XR capabilities

**Trade-off**: Requires user gesture to create XR session (can't run at startup).

### Manual Override

Users can force immersive mode via URL parameter:
```
https://visionflow.app/?force=quest3
https://visionflow.app/?immersive=true
https://visionflow.app/?directar=true
```

---

## Babylon.js XR Scene

### Why Babylon.js Instead of Three.js?

| Feature | Babylon.js | Three.js + WebXR |
|---------|------------|------------------|
| **WebXR Support** | Native, built-in | Requires custom setup |
| **Controller Handling** | Automatic mapping | Manual implementation |
| **Hand Tracking** | Built-in | Requires polyfill |
| **Foveated Rendering** | Automatic | Manual setup |
| **Performance** | Optimized for VR | General-purpose |

**Decision**: Babylon.js chosen for **better XR support out-of-the-box**.

**⚠️ Trade-off**: Maintaining two rendering engines (Three.js + Babylon.js) increases bundle size by 1.2MB.

### Immersive App Component

**File**: `client/src/immersive/components/ImmersiveApp.tsx` (212 lines)

```typescript
import React, { useEffect, useRef, useState } from 'react';
import { BabylonScene } from '../babylon/BabylonScene';
import { useImmersiveData } from '../hooks/useImmersiveData';
import { createLogger, createRemoteLogger } from '../../utils/loggerConfig';

const logger = createLogger('ImmersiveApp');
const remoteLog = createRemoteLogger('ImmersiveApp'); // Logs to remote server

export const ImmersiveApp: React.FC<ImmersiveAppProps> = ({ onExit, initialData }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [babylonScene, setBabylonScene] = useState<BabylonScene | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch graph data via WebSocket
  const {
    graphData,
    nodePositions,
    isLoading,
    error: dataError,
    updateNodePosition,
    selectNode,
    selectedNode
  } = useImmersiveData(initialData);

  // Initialize Babylon.js scene
  useEffect(() => {
    const initializeImmersiveEnvironment = () => {
      try {
        if (!canvasRef.current) {
          throw new Error('Canvas reference not available');
        }

        logger.info('Initializing immersive environment...');
        remoteLog.info('Initializing on ' + navigator.userAgent);

        // Detect Quest device
        const isQuest = /OculusBrowser|Quest/i.test(navigator.userAgent);
        if (isQuest) {
          remoteLog.info('🎮 Quest device detected!');
        }

        // Create Babylon.js scene
        const scene = new BabylonScene(canvasRef.current);
        setBabylonScene(scene);

        // Start render loop
        scene.run();

        setIsInitialized(true);
        logger.info('Immersive environment initialized successfully');
        remoteLog.info('✅ Immersive environment ready');

      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        logger.error('Failed to initialize:', errorMessage);
        remoteLog.error('Initialization failed', err);
        setError(errorMessage);
      }
    };

    initializeImmersiveEnvironment();

    // Cleanup on unmount
    return () => {
      logger.info('Cleaning up immersive environment...');
      babylonScene?.dispose();
      setBabylonScene(null);
      setIsInitialized(false);
    };
  }, []);

  // Update graph data when received
  useEffect(() => {
    if (babylonScene && graphData && !isLoading) {
      babylonScene.setBotsData({
        graphData: graphData,
        nodePositions: nodePositions,
        nodes: graphData.nodes || [],
        edges: graphData.edges || []
      });
    }
  }, [babylonScene, graphData, nodePositions, isLoading]);

  // Error state
  if (error) {
    return (
      <div className="immersive-error">
        <h2>XR Initialization Failed</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
        <button onClick={onExit}>Exit to Desktop Mode</button>
      </div>
    );
  }

  // Loading state
  if (!isInitialized) {
    return (
      <div className="immersive-loading">
        <p>Initializing VR environment...</p>
      </div>
    );
  }

  return <canvas ref={canvasRef} className="immersive-canvas" />;
};
```

### Babylon.js Scene Setup

**File**: `client/src/immersive/babylon/BabylonScene.ts`

```typescript
import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders';

export class BabylonScene {
  private engine: BABYLON.Engine;
  private scene: BABYLON.Scene;
  private camera: BABYLON.FreeCamera | BABYLON.WebXRCamera;
  private xrHelper: BABYLON.WebXRDefaultExperience | null = null;

  constructor(canvas: HTMLCanvasElement) {
    // Create engine
    this.engine = new BABYLON.Engine(canvas, true, {
      stencil: true,
      antialias: true,
      powerPreference: 'high-performance'
    });

    // Create scene
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color4(0, 0, 0, 1);

    // Create camera
    this.camera = new BABYLON.FreeCamera(
      'camera',
      new BABYLON.Vector3(0, 1.6, -5), // Eye level
      this.scene
    );
    this.camera.attachControl(canvas, true);

    // Add lighting
    const hemisphericLight = new BABYLON.HemisphericLight(
      'light',
      new BABYLON.Vector3(0, 1, 0),
      this.scene
    );
    hemisphericLight.intensity = 0.7;

    // Enable XR
    this.initializeXR();
  }

  private async initializeXR(): Promise<void> {
    try {
      // Create WebXR experience
      this.xrHelper = await this.scene.createDefaultXRExperienceAsync({
        uiOptions: {
          sessionMode: 'immersive-vr',
          referenceSpaceType: 'local-floor'
        },
        optionalFeatures: true // Enable hand tracking, hit-test, etc.
      });

      console.log('WebXR initialized:', this.xrHelper);

      // Setup controller input
      this.setupControllers();

      // Setup hand tracking (if available)
      this.setupHandTracking();

    } catch (error) {
      console.error('WebXR initialization failed:', error);
      // Fallback to non-XR mode
    }
  }

  private setupControllers(): void {
    if (!this.xrHelper?.input) return;

    this.xrHelper.input.onControllerAddedObservable.add((controller) => {
      controller.onMotionControllerInitObservable.add((motionController) => {
        console.log('Controller connected:', motionController.handedness);

        // Map controller buttons
        const triggerComponent = motionController.getComponent('xr-standard-trigger');
        const squeezeComponent = motionController.getComponent('xr-standard-squeeze');
        const thumbstickComponent = motionController.getComponent('xr-standard-thumbstick');

        if (triggerComponent) {
          triggerComponent.onButtonStateChangedObservable.add((component) => {
            if (component.pressed) {
              this.handleTriggerPress(controller);
            }
          });
        }

        if (squeezeComponent) {
          squeezeComponent.onButtonStateChangedObservable.add((component) => {
            if (component.pressed) {
              this.handleSqueezePress(controller);
            }
          });
        }

        if (thumbstickComponent) {
          thumbstickComponent.onAxisValueChangedObservable.add((axes) => {
            this.handleThumbstickMove(controller, axes.x, axes.y);
          });
        }
      });
    });
  }

  private setupHandTracking(): void {
    if (!this.xrHelper?.input) return;

    const handTracking = this.xrHelper.featuresManager.enableFeature(
      BABYLON.WebXRFeatureName.HAND_TRACKING,
      'latest',
      { xrInput: this.xrHelper.input }
    );

    if (handTracking) {
      console.log('Hand tracking enabled');
      // Implement hand gesture recognition
    }
  }

  // Set graph data from WebSocket
  public setBotsData(data: {
    graphData: any;
    nodePositions: Float32Array;
    nodes: any[];
    edges: any[];
  }): void {
    // Clear existing meshes
    this.scene.meshes.forEach(mesh => {
      if (mesh.name.startsWith('node_')) {
        mesh.dispose();
      }
    });

    // Create instanced meshes for nodes
    const sphereGeometry = BABYLON.MeshBuilder.CreateSphere('nodeSphere', {
      diameter: 1,
      segments: 16
    }, this.scene);

    data.nodes.forEach((node, i) => {
      const i3 = i * 3;
      const instance = sphereGeometry.createInstance(`node_${node.id}`);
      instance.position = new BABYLON.Vector3(
        data.nodePositions[i3],
        data.nodePositions[i3 + 1],
        data.nodePositions[i3 + 2]
      );

      // Assign color
      const material = new BABYLON.StandardMaterial(`mat_${node.id}`, this.scene);
      material.diffuseColor = BABYLON.Color3.FromHexString(node.color || '#00ffff');
      material.emissiveColor = BABYLON.Color3.FromHexString(node.color || '#00ffff');
      instance.material = material;
    });

    sphereGeometry.setEnabled(false); // Hide template
  }

  // Render loop
  public run(): void {
    this.engine.runRenderLoop(() => {
      this.scene.render();
    });

    // Handle resize
    window.addEventListener('resize', () => {
      this.engine.resize();
    });
  }

  // Cleanup
  public dispose(): void {
    this.scene.dispose();
    this.engine.dispose();
  }
}
```

---

## Controller Input Handling

### Quest Touch Controllers

```typescript
// Trigger button (index finger)
triggerComponent.onButtonStateChangedObservable.add((component) => {
  if (component.pressed) {
    // Raycast from controller
    const ray = controller.getWorldPointerRayToRef(new BABYLON.Ray());
    const hit = this.scene.pickWithRay(ray);

    if (hit.pickedMesh) {
      console.log('Selected node:', hit.pickedMesh.name);
      this.selectNode(hit.pickedMesh.name);
    }
  }
});

// Squeeze button (grip)
squeezeComponent.onButtonStateChangedObservable.add((component) => {
  if (component.pressed) {
    // Grab and move nodes
    this.startNodeDrag(controller);
  } else {
    this.endNodeDrag(controller);
  }
});

// Thumbstick (locomotion)
thumbstickComponent.onAxisValueChangedObservable.add((axes) => {
  const speed = 0.1;
  const forward = controller.pointer.forward.scale(axes.y * speed);
  const right = controller.pointer.right.scale(axes.x * speed);

  this.camera.position.addInPlace(forward);
  this.camera.position.addInPlace(right);
});
```

### Hand Tracking

```typescript
private setupHandTracking(): void {
  const handTracking = this.xrHelper.featuresManager.enableFeature(
    BABYLON.WebXRFeatureName.HAND_TRACKING,
    'latest',
    { xrInput: this.xrHelper.input }
  );

  if (handTracking) {
    handTracking.onHandAddedObservable.add((hand) => {
      console.log('Hand detected:', hand.xrController.inputSource.handedness);

      // Pinch gesture detection
      const indexTip = hand.getJointMesh(BABYLON.WebXRHandJoint.INDEX_FINGER_TIP);
      const thumbTip = hand.getJointMesh(BABYLON.WebXRHandJoint.THUMB_TIP);

      this.scene.onBeforeRenderObservable.add(() => {
        const distance = BABYLON.Vector3.Distance(
          indexTip.position,
          thumbTip.position
        );

        if (distance < 0.02) {
          // Pinch detected
          this.handlePinchGesture(hand);
        }
      });
    });
  }
}
```

---

## Performance Optimizations

### 1. Foveated Rendering

```typescript
// Enable foveated rendering (Quest 3 only)
if (this.xrHelper?.baseExperience) {
  const foveation = this.xrHelper.featuresManager.enableFeature(
    BABYLON.WebXRFeatureName.FOVEATED_RENDERING,
    'latest'
  );

  if (foveation) {
    foveation.foveationLevel = 0.5; // 0.0 = off, 1.0 = maximum
    console.log('Foveated rendering enabled');
  }
}
```

**Impact**: 30% GPU performance gain on Quest 3.

### 2. Level of Detail (LOD)

```typescript
// Create LOD meshes
const sphereHigh = BABYLON.MeshBuilder.CreateSphere('sphere_high', {
  diameter: 1,
  segments: 32 // High detail
}, this.scene);

const sphereMed = BABYLON.MeshBuilder.CreateSphere('sphere_med', {
  diameter: 1,
  segments: 16 // Medium detail
}, this.scene);

const sphereLow = BABYLON.MeshBuilder.CreateSphere('sphere_low', {
  diameter: 1,
  segments: 8 // Low detail
}, this.scene);

// Add LOD levels
sphereHigh.addLODLevel(10, sphereMed);   // Switch at 10 units
sphereHigh.addLODLevel(50, sphereLow);   // Switch at 50 units
sphereHigh.addLODLevel(200, null);       // Hide beyond 200 units
```

**Impact**: Maintains 72 FPS (Quest 3 native refresh rate) with 5,000+ nodes.

### 3. Instanced Meshes

```typescript
// Use instances instead of clones
const template = BABYLON.MeshBuilder.CreateSphere('template', {
  diameter: 1,
  segments: 16
}, this.scene);

nodes.forEach(node => {
  const instance = template.createInstance(`node_${node.id}`);
  instance.position = new BABYLON.Vector3(node.x, node.y, node.z);
});

template.setEnabled(false); // Hide template
```

**Impact**: 80% memory reduction vs individual meshes.

### 4. Reduce Draw Calls

```typescript
// Merge static geometry
const merged = BABYLON.Mesh.MergeMeshes(
  staticMeshes,
  true,   // Dispose source meshes
  true,   // Allow instancing
  undefined,
  false,  // Multimat
  true    // Preserve UVs
);
```

**Impact**: Reduces draw calls from 500 → 20.

---

## Known Issues & Workarounds

### ⚠️ Issue 1: Quest 3 Detection Fragility

**Problem**: User-agent string varies by firmware version.

**Current Detection**:
```typescript
const isQuest3 = userAgent.includes('Quest 3') ||
                 userAgent.includes('OculusBrowser');
```

**False Positives**: Quest 2 devices with `OculusBrowser` trigger XR mode.

**Workaround**: URL parameter `?force=quest3` for manual override.

**Proposed Fix**: Use `navigator.xr.isSessionSupported('immersive-vr')`.

### ⚠️ Issue 2: Babylon.js Bundle Size

**Problem**: Babylon.js adds 1.2MB (gzipped) to bundle.

**Current Size**:
- Three.js: 850KB
- Babylon.js: 1.2MB
- **Total**: 2.05MB for 3D engines

**Impact**: Slower initial page load.

**Workaround**: Code-splitting (load Babylon.js only when entering XR mode).

**Proposed Fix**:
```typescript
// Lazy load Babylon.js
const loadBabylonScene = async () => {
  const { BabylonScene } = await import('./babylon/BabylonScene');
  return BabylonScene;
};
```

### ⚠️ Issue 3: XR Session Requires User Gesture

**Problem**: `navigator.xr.requestSession()` requires user interaction (button click).

**Impact**: Cannot auto-detect XR at startup.

**Workaround**: Show "Enter VR" button on Quest devices.

```typescript
const EnterVRButton: React.FC = () => {
  const handleEnterVR = async () => {
    const supported = await navigator.xr.isSessionSupported('immersive-vr');
    if (supported) {
      window.location.href = '/?force=quest3';
    }
  };

  return <button onClick={handleEnterVR}>Enter VR Mode</button>;
};
```

### ⚠️ Issue 4: WebXR Not Available in Desktop Chrome

**Problem**: Desktop Chrome doesn't support `navigator.xr`.

**Impact**: Cannot test XR mode without Quest device.

**Workaround**: Use WebXR Emulator extension:
- [Chrome Extension](https://chrome.google.com/webstore/detail/webxr-emulator/mjddjgeghkdijejnciaefnkjmkafnnje)
- Emulates Quest controllers, hand tracking, and spatial tracking

### ⚠️ Issue 5: Graph Data Not Updating in XR

**Problem**: `babylonScene.setBotsData()` doesn't update existing meshes.

**Cause**: Meshes not disposed before creating new ones.

**Fix**:
```typescript
public setBotsData(data: any): void {
  // Dispose existing node meshes
  this.scene.meshes
    .filter(mesh => mesh.name.startsWith('node_'))
    .forEach(mesh => mesh.dispose());

  // Create new meshes
  data.nodes.forEach(node => {
    // ... create mesh
  });
}
```

---

## Browser Compatibility

| Feature | Quest 3 Browser | Desktop Chrome | Desktop Safari | Firefox |
|---------|----------------|----------------|----------------|---------|
| **WebXR API** | ✅ Full | ⚠️ Emulator only | ❌ Not supported | ⚠️ Experimental |
| **Hand Tracking** | ✅ Native | ❌ No | ❌ No | ❌ No |
| **Controller Input** | ✅ Touch Controllers | ⚠️ Emulated | ❌ No | ❌ No |
| **Foveated Rendering** | ✅ Automatic | ❌ No | ❌ No | ❌ No |
| **72 FPS Target** | ✅ Yes | N/A | N/A | N/A |

**Recommendation**: XR features should be considered Quest-exclusive. Desktop testing requires emulator.

---

## Testing XR Mode

### 1. Quest 3 Device Testing

**Steps**:
1. Connect Quest 3 to Wi-Fi
2. Open Quest Browser
3. Navigate to `https://visionflow.app`
4. App automatically enters XR mode (if detection works)
5. If not, add `?force=quest3` to URL

### 2. Desktop Emulator Testing

**Setup**:
1. Install WebXR Emulator extension
2. Open DevTools → WebXR tab
3. Select "Meta Quest 3" device
4. Add `?force=quest3` to URL
5. Click "Enter VR" button

### 3. Remote Logging

**Issue**: Quest browser has no DevTools.

**Solution**: Remote logging to server.

```typescript
import { createRemoteLogger } from '../../utils/loggerConfig';

const remoteLog = createRemoteLogger('ImmersiveApp');

remoteLog.info('XR session started');
remoteLog.error('XR initialization failed', error);
```

**View Logs**: Server endpoint `/api/logs?source=ImmersiveApp`

---

---

---

## Related Documentation

- [Client State Management with Zustand](state-management.md)
- [Three.js Rendering Pipeline](three-js-rendering.md)
- [Adding Features](../developer/04-adding-features.md)
- [Testing Guide](../../archive/docs/guides/developer/05-testing-guide.md)
- [Working with Agents](../../archive/docs/guides/user/working-with-agents.md)

## Future Improvements

### 1. Unified Rendering Engine

**Current**: Three.js (desktop) + Babylon.js (XR)

**Proposed**: Three.js with WebXR polyfill for both

**Benefits**:
- -1.2MB bundle size
- Single scene graph implementation
- Easier maintenance

**Implementation**:
```typescript
import { VRButton } from 'three/examples/jsm/webxr/VRButton';

// Three.js XR setup
renderer.xr.enabled = true;
document.body.appendChild(VRButton.createButton(renderer));

// Render loop
renderer.setAnimationLoop(() => {
  renderer.render(scene, camera);
});
```

### 2. Passthrough AR Mode

**Current**: Fully immersive VR only.

**Proposed**: Mixed reality with passthrough.

```typescript
const session = await navigator.xr.requestSession('immersive-ar', {
  requiredFeatures: ['local-floor'],
  optionalFeatures: ['hand-tracking', 'hit-test']
});
```

**Use Case**: Overlay graph visualization on physical environment.

### 3. Multi-User Collaboration

**Current**: Single-user XR experience.

**Proposed**: Share XR session with other users.

**Implementation**:
- WebRTC for avatar synchronization
- Shared cursor/pointer for collaboration
- Voice chat integration

---
