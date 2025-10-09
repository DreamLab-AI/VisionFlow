Okay, this is an exciting and advanced feature! Adding head-tracked parallax viewing will create a much more immersive "window into another world" effect. Here's a comprehensive breakdown of how we can implement this, touching all the necessary parts of your application.

### Summary of Changes

1.  **Dependency**: Add `@mediapipe/tasks-vision` for face and eye tracking.
2.  **New Hook (`useHeadTracking.ts`)**: A dedicated hook to manage webcam access, initialize MediaPipe's Face Landmarker, and process the video stream to calculate head position.
3.  **New Component (`HeadTrackedParallaxController.tsx`)**: A React Three Fiber component that uses the head tracking data to modify the scene camera, creating the parallax effect.
4.  **Settings Integration**: Add new settings to enable/disable head tracking and adjust its sensitivity, making it a configurable feature.
5.  **UI Integration**: Add a toggle button in the `IntegratedControlPanel` (specifically in the `GraphInteractionTab`) to control the feature.
6.  **Canvas Integration**: Add the new `HeadTrackedParallaxController` to your main `GraphCanvas` to apply the effect.

Here are the detailed file modifications and new files required:

### 1. Update `package.json`

First, let's add the MediaPipe dependency.

```json:client/package.json
{
  "name": "visionflow-client",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "npm run types:generate && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives",
    "test": "echo 'Testing disabled due to supply chain attack - see SECURITY_ALERT.md'",
    "test:ui": "echo 'Testing disabled due to supply chain attack - see SECURITY_ALERT.md'",
    "test:coverage": "echo 'Testing disabled due to supply chain attack - see SECURITY_ALERT.md'",
    "types:generate": "cd .. && cargo run --bin generate_types",
    "types:watch": "cd .. && cargo watch -x 'run --bin generate_types'",
    "types:clean": "rm -rf src/types/generated",
    "prebuild": "npm run types:generate",
    "preinstall": "node scripts/block-test-packages.js",
    "security:check": "node scripts/block-test-packages.js && npm audit"
  },
  "dependencies": {
    "@babylonjs/core": "8.28.0",
    "@babylonjs/gui": "8.29.0",
    "@babylonjs/loaders": "8.28.0",
    "@babylonjs/materials": "8.28.0",
    "@getalby/sdk": "^4.1.1",
    "@mediapipe/tasks-vision": "^0.10.14",
    "@radix-ui/react-collapsible": "^1.1.4",
    "@radix-ui/react-dialog": "^1.1.7",
    "@radix-ui/react-dropdown-menu": "^2.1.7",
    "@radix-ui/react-label": "^2.1.3",
    "@radix-ui/react-radio-group": "^1.1.3",
    "@radix-ui/react-select": "^2.2.4",
    "@radix-ui/react-slider": "^1.2.4",
    "@radix-ui/react-slot": "^1.2.0",
    "@radix-ui/react-switch": "^1.1.4",
    "@radix-ui/react-toast": "^1.2.7",
    "@radix-ui/react-tooltip": "^1.2.0",
    "@radix-ui/themes": "^3.2.1",
    "@react-three/drei": "^9.80.0",
    "@react-three/fiber": "^8.15.0",
    "@react-three/postprocessing": "^2.15.0",
    "@types/lodash": "4.17.20",
    "@types/react-window": "^1.8.8",
    "@types/three": "^0.175.0",
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "comlink": "^4.4.1",
    "framer-motion": "^12.6.5",
    "hls.js": "^1.6.2",
    "immer": "^10.1.1",
    "lodash": "4.17.21",
    "lucide-react": "^0.487.0",
    "nostr-tools": "^2.12.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-markdown": "^10.1.0",
    "react-resizable-panels": "^3.0.3",
    "react-rnd": "^10.5.2",
    "react-syntax-highlighter": "^15.6.6",
    "react-window": "^1.8.10",
    "remark-gfm": "^4.0.1",
    "tailwind-merge": "^3.2.0",
    "three": "^0.175.0",
    "uuid": "^11.1.0"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4.1.7",
    "@types/node": "^22.14.1",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@types/uuid": "^10.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "autoprefixer": "^10.4.21",
    "jsdom": "^25.0.1",
    "postcss": "^8.5.3",
    "tailwindcss": "^4.1.3",
    "typescript": "^5.8.3",
    "vite": "^6.2.6",
    "wscat": "^6.1.0"
  },
  "overrides": {
    "ansi-regex": "6.1.0",
    "ansi-styles": "6.2.1",
    "color-name": "2.0.0",
    "color-convert": "2.0.1",
    "supports-color": "9.4.0",
    "strip-ansi": "7.1.0",
    "string-width": "7.2.0",
    "wrap-ansi": "9.0.0",
    "esbuild": "0.25.9",
    "prismjs": "1.30.0"
  }
}
```

### 2. Add MediaPipe Model Files

You'll need to download the Face Landmarker model and place it in your `public` directory.

1.  Create a new directory: `client/public/models/`.
2.  Download the model file from: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
3.  Place the downloaded `face_landmarker.task` file inside `client/public/models/`.

### 3. Create the Head Tracking Hook

This new hook will encapsulate all the logic for accessing the webcam and running MediaPipe's face landmark detection.

```typescript:client/src/hooks/useHeadTracking.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as THREE from 'three';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useHeadTracking');

let faceLandmarker: FaceLandmarker | undefined;
let lastVideoTime = -1;

const smoothingFactor = 0.1; // Apply smoothing for less jittery movement

export function useHeadTracking() {
  const [isEnabled, setIsEnabled] = useState(false);
  const [isTracking, setIsTracking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [headPosition, setHeadPosition] = useState<THREE.Vector2 | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const smoothedPosition = useRef(new THREE.Vector2(0, 0));

  const initialize = useCallback(async () => {
    if (faceLandmarker) return;
    try {
      logger.info('Initializing MediaPipe Face Landmarker...');
      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: `/models/face_landmarker.task`,
          delegate: 'GPU',
        },
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
        runningMode: 'VIDEO',
        numFaces: 1,
      });
      logger.info('Face Landmarker initialized.');
    } catch (e: any) {
      logger.error('Failed to initialize Face Landmarker', e);
      setError('Failed to load head tracking model. Please check your network connection.');
    }
  }, []);

  const predictWebcam = useCallback(() => {
    if (!videoRef.current || !faceLandmarker || !videoRef.current.srcObject) {
        if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
        return;
    }

    const video = videoRef.current;
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const results = faceLandmarker.detectForVideo(video, Date.now());

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        // Use nose tip (landmark 1) as a stable point for head position
        const noseTip = results.faceLandmarks[0][1];
        if (noseTip) {
          // Normalize position to [-1, 1] range
          // MediaPipe gives normalized coordinates [0, 1]
          // We map x:[0,1] -> [-1,1] and y:[0,1] -> [1,-1] (inverted y)
          const newPos = new THREE.Vector2(
            (noseTip.x - 0.5) * 2,
            -(noseTip.y - 0.5) * 2
          );

          // Apply smoothing (Lerp)
          smoothedPosition.current.lerp(newPos, smoothingFactor);
          setHeadPosition(smoothedPosition.current.clone());
        }
      } else {
        setHeadPosition(null);
      }
    }

    animationFrameId.current = requestAnimationFrame(predictWebcam);
  }, []);

  const start = useCallback(async () => {
    if (isTracking) return;
    setError(null);

    await initialize();
    if (!faceLandmarker) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      if (!videoRef.current) {
        const video = document.createElement('video');
        video.style.display = 'none';
        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;
        document.body.appendChild(video);
        videoRef.current = video;
      }

      videoRef.current.srcObject = stream;
      videoRef.current.addEventListener('loadeddata', () => {
        videoRef.current?.play();
        setIsTracking(true);
        predictWebcam();
      });
    } catch (err: any) {
      logger.error('Failed to get webcam access', err);
      setError('Webcam access denied. Please allow camera permissions to use head tracking.');
      setIsEnabled(false);
    }
  }, [isTracking, initialize, predictWebcam]);

  const stop = useCallback(() => {
    if (!isTracking && !videoRef.current?.srcObject) return;

    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsTracking(false);
    setHeadPosition(null);
    smoothedPosition.current.set(0, 0);
  }, [isTracking]);

  useEffect(() => {
    if (isEnabled) {
      start();
    } else {
      stop();
    }
  }, [isEnabled, start, stop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
      if (videoRef.current) {
        document.body.removeChild(videoRef.current);
        videoRef.current = null;
      }
      // Do not close faceLandmarker, it can be reused across the app lifecycle.
    };
  }, [stop]);

  return { isEnabled, setIsEnabled, isTracking, headPosition, error };
}
```

### 4. Update Settings Configuration

Let's define the new settings for head tracking.

```typescript:client/src/features/settings/config/settings.ts
// ... (keep existing content)

// Add these new interfaces
export interface HeadTrackedParallaxSettings {
  enabled: boolean;
  sensitivity: number;
  cameraMode: 'offset' | 'asymmetricFrustum';
}

export interface InteractionSettings {
  headTrackedParallax: HeadTrackedParallaxSettings;
}

// Modify VisualisationSettings
export interface VisualisationSettings {
  // ... (existing properties)
  interaction?: InteractionSettings;
}

// ... (rest of the file)
```

Now, define the UI for these settings.

```typescript:client/src/features/settings/config/settingsUIDefinition.ts
// ... (keep existing content)

// Add this new subsection to the `visualization` category's `subsections` object
// in the `settingsUIDefinition` constant.

// ... inside visualization.subsections
    interaction: {
      label: 'Advanced Interaction',
      settings: {
        headTrackedParallaxEnabled: {
          label: 'Head-Tracked Parallax',
          type: 'toggle',
          path: 'visualisation.interaction.headTrackedParallax.enabled',
          description: 'Enable parallax viewing effect using your webcam for head tracking.'
        },
        headTrackedParallaxSensitivity: {
          label: 'Parallax Sensitivity',
          type: 'slider',
          min: 0.1,
          max: 2.0,
          step: 0.1,
          path: 'visualisation.interaction.headTrackedParallax.sensitivity',
          description: 'Controls the intensity of the head-tracked parallax effect.'
        },
        headTrackedParallaxMode: {
          label: 'Parallax Camera Mode',
          type: 'select',
          options: [
            { value: 'asymmetricFrustum', label: 'Asymmetric Frustum (Recommended)' },
            { value: 'offset', label: 'Camera Offset (Simpler)' }
          ],
          path: 'visualisation.interaction.headTrackedParallax.cameraMode',
          description: 'Method used to create the parallax effect. Asymmetric Frustum is more realistic.'
        }
      }
    },
// ... (rest of the subsections)
```

### 5. Add the UI Toggle to the Control Panel

We'll add a new card to the `GraphInteractionTab` for controlling this feature.

```tsx:client/src/features/visualisation/components/tabs/GraphInteractionTab.tsx
/**
 * Graph Interaction Tab Component
 * Advanced interaction modes and controls with UK English localisation
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Clock,
  Users,
  Glasses,
  Play,
  Pause,
  RotateCcw,
  FastForward,
  Rewind,
  MapPin,
  Radio,
  Gamepad2,
  AlertCircle,
  Sparkles,
  Navigation,
  Eye // Import the Eye icon
} from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Badge } from '@/features/design-system/components/Badge';
import { Slider } from '@/features/design-system/components/Slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Progress } from '@/features/design-system/components/Progress';
import { toast } from '@/features/design-system/components/Toast';
import { interactionApi, type GraphProcessingProgress, type GraphProcessingResult } from '@/services/interactionApi';
import { webSocketService } from '@/services/WebSocketService';
import { useSettingsStore } from '@/store/settingsStore'; // Import settings store

interface GraphInteractionTabProps {
  graphId?: string;
  onFeatureUpdate?: (feature: string, data: any) => void;
}

interface ProcessingState {
  taskId: string | null;
  isProcessing: boolean;
  progress: number;
  stage: string;
  currentOperation: string;
  estimatedTimeRemaining?: number;
  metrics?: {
    stepsProcessed: number;
    totalSteps: number;
    currentStep: string;
    operationsCompleted: number;
  };
  error?: string;
}

export const GraphInteractionTab: React.FC<GraphInteractionTabProps> = ({
  graphId = 'default',
  onFeatureUpdate
}) => {
  // ... (existing state and hooks)
  const { settings, updateSettings } = useSettingsStore();
  const headTrackingEnabled = settings?.visualisation?.interaction?.headTrackedParallax?.enabled ?? false;


  // ... (existing handlers)

  const handleHeadTrackingToggle = useCallback((enabled: boolean) => {
    updateSettings(draft => {
      if (!draft.visualisation) draft.visualisation = {};
      if (!draft.visualisation.interaction) {
        draft.visualisation.interaction = { headTrackedParallax: { enabled: false, sensitivity: 1.0, cameraMode: 'asymmetricFrustum' } };
      }
      if (!draft.visualisation.interaction.headTrackedParallax) {
        draft.visualisation.interaction.headTrackedParallax = { enabled: false, sensitivity: 1.0, cameraMode: 'asymmetricFrustum' };
      }
      draft.visualisation.interaction.headTrackedParallax.enabled = enabled;
    });
    onFeatureUpdate?.('headTracking', { enabled });
    toast({
      title: `Head Tracking ${enabled ? 'Enabled' : 'Disabled'}`,
      description: enabled ? 'Webcam will be used to create a parallax effect.' : 'Head tracking has been turned off.'
    });
  }, [updateSettings, onFeatureUpdate]);


  return (
    <div className="space-y-4">
      {/* Head-Tracked Parallax */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Head-Tracked Parallax
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="head-tracking-toggle">Enable Head Tracking</Label>
            <Switch
              id="head-tracking-toggle"
              checked={headTrackingEnabled}
              onCheckedChange={handleHeadTrackingToggle}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Uses your webcam to create a 3D parallax effect based on your head position.
          </p>
        </CardContent>
      </Card>

      {/* Time Travel Mode */}
      <Card>
        {/* ... existing Time Travel content ... */}
      </Card>

      {/* Collaboration */}
      <Card>
        {/* ... existing Collaboration content ... */}
      </Card>

      {/* VR/AR Modes */}
      <Card>
        {/* ... existing VR/AR content ... */}
      </Card>

      {/* Exploration Tools */}
      <Card>
        {/* ... existing Exploration content ... */}
      </Card>
    </div>
  );
};

export default GraphInteractionTab;
```

### 6. Create the Camera Controller Component

This component will live inside the R3F Canvas and apply the parallax effect every frame.

```typescript:client/src/features/visualisation/components/HeadTrackedParallaxController.tsx
import React, { useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSettingsStore } from '@/store/settingsStore';
import { useHeadTracking } from '@/hooks/useHeadTracking';
import { toast } from '@/features/design-system/components/Toast';

export function HeadTrackedParallaxController() {
  const { camera, size } = useThree();
  const { isEnabled, setIsEnabled, isTracking, headPosition, error } = useHeadTracking();

  const trackingEnabled = useSettingsStore(state => state.settings?.visualisation?.interaction?.headTrackedParallax?.enabled);
  const sensitivity = useSettingsStore(state => state.settings?.visualisation?.interaction?.headTrackedParallax?.sensitivity ?? 1.0);
  const cameraMode = useSettingsStore(state => state.settings?.visualisation?.interaction?.headTrackedParallax?.cameraMode ?? 'asymmetricFrustum');

  useEffect(() => {
    setIsEnabled(!!trackingEnabled);
  }, [trackingEnabled, setIsEnabled]);

  useEffect(() => {
    if (error) {
      toast({
        title: 'Head Tracking Error',
        description: error,
        variant: 'destructive',
      });
    }
  }, [error]);

  useFrame(() => {
    if (isTracking && headPosition && camera instanceof THREE.PerspectiveCamera) {
      if (cameraMode === 'asymmetricFrustum') {
        const virtualScreenScale = 1.0 + sensitivity * 0.5;
        const fullWidth = size.width * virtualScreenScale;
        const fullHeight = size.height * virtualScreenScale;

        const x_offset = -headPosition.x * (fullWidth - size.width) / 2;
        const y_offset = headPosition.y * (fullHeight - size.height) / 2;

        camera.setViewOffset(
          fullWidth,
          fullHeight,
          x_offset,
          y_offset,
          size.width,
          size.height
        );
        camera.updateProjectionMatrix();
      } else { // 'offset' mode
        // This is a simpler parallax that might fight with OrbitControls
        // It's kept as an alternative.
        const offsetX = headPosition.x * sensitivity * -0.5;
        const offsetY = headPosition.y * sensitivity * 0.5;

        const offsetVector = new THREE.Vector3(offsetX, offsetY, 0);
        // This is a simplified approach and might not be perfect.
        // It's applied as a small "nudge" to the camera's matrix.
        const nudgeMatrix = new THREE.Matrix4().makeTranslation(offsetVector.x, offsetVector.y, 0);
        camera.projectionMatrix.multiply(nudgeMatrix);
      }
    } else {
      if (camera.view) {
        camera.clearViewOffset();
        camera.updateProjectionMatrix();
      }
    }
  });

  return null;
}
```

### 7. Integrate into the Main Canvas

Finally, add the new controller component to your main canvas component.

```tsx:client/src/features/graph/components/GraphCanvas.tsx
import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';

// GraphManager for rendering the actual graph
import GraphManager from './GraphManager';
// Post-processing effects - using modern R3F selective bloom
import { SelectiveBloom } from '../../../rendering/SelectiveBloom';
// Bots visualization for agent graph
import { BotsVisualization } from '../../bots/components';
import { AgentPollingStatus } from '../../bots/components/AgentPollingStatus';
// SpacePilot Integration - using simpler version that works with useFrame
import SpacePilotSimpleIntegration from '../../visualisation/components/SpacePilotSimpleIntegration';
// Head Tracking for Parallax
import { HeadTrackedParallaxController } from '../../visualisation/components/HeadTrackedParallaxController';
// Hologram environment removed
// XR Support - causes graph to disappear
// import XRController from '../../xr/components/XRController';
// import XRVisualisationConnector from '../../xr/components/XRVisualisationConnector';

// Store and utils
import { useSettingsStore } from '../../../store/settingsStore';
import { graphDataManager, type GraphData } from '../managers/graphDataManager';
import { createLogger } from '../../../utils/loggerConfig';
import { HologramContent } from '../../visualisation/components/HolographicDataSphere';

const logger = createLogger('GraphCanvas');

// Main GraphCanvas component
const GraphCanvas: React.FC = () => {

    const containerRef = useRef<HTMLDivElement>(null);
    const orbitControlsRef = useRef<any>(null);
    const { settings } = useSettingsStore();
    const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false;
    const xrEnabled = settings?.xr?.enabled !== false;
    const enableBloom = settings?.visualisation?.bloom?.enabled ?? false;
    const enableGlow = settings?.visualisation?.glow?.enabled ?? false;
    const useMultiLayerBloom = enableBloom || enableGlow; // Use multi-layer if either is enabled
    const enableHologram = settings?.visualisation?.graphs?.logseq?.nodes?.enableHologram ?? false;

    // Graph data state
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
    const [canvasReady, setCanvasReady] = useState(false);

    // Subscribe to graph data updates
    useEffect(() => {
        let mounted = true;


        const handleGraphData = (data: GraphData) => {
            if (mounted) {
                setGraphData(data);
            }
        };

        const unsubscribe = graphDataManager.onGraphDataChange(handleGraphData);

        // Get initial data
        graphDataManager.getGraphData().then((data) => {
            if (mounted) {
                setGraphData(data);
            }
        }).catch((error) => {
            console.error('[GraphCanvas] Failed to load initial graph data:', error);
        });

        return () => {
            mounted = false;
            unsubscribe();
        };
    }, []);

    return (
        <div
            ref={containerRef}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                backgroundColor: '#000033',
                zIndex: 0
            }}
        >
            {/* Debug indicator */}
            {showStats && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    color: 'white',
                    backgroundColor: 'rgba(255, 0, 0, 0.5)',
                    padding: '5px 10px',
                    zIndex: 1000,
                    fontSize: '12px'
                }}>
                    Nodes: {graphData.nodes.length} | Edges: {graphData.edges.length} | Ready: {canvasReady ? 'Yes' : 'No'}
                </div>
            )}

            {/* Agent Polling Status Overlay */}
            <AgentPollingStatus />

            <Canvas
                camera={{
                    fov: 75,
                    near: 0.1,
                    far: 2000,
                    position: [20, 15, 20]
                }}
                onCreated={({ gl, camera, scene }) => {
                    gl.setClearColor(0x000033, 1);
                    setCanvasReady(true);
                }}
            >
                {/* Basic lighting - reduced to prevent washout */}
                <ambientLight intensity={0.15} />
                <directionalLight position={[10, 10, 10]} intensity={0.4} />

                {/* Holographic Data Sphere with minimal opacity - scaled 50x - only render if enabled */}
                {enableHologram && (
                  <HologramContent
                    opacity={0.1}
                    layer={2}
                    renderOrder={-1}
                    includeSwarm={false}
                    enableDepthFade={true}
                    fadeStart={2000}
                    fadeEnd={5000}
                  />
                )}

                {/* Graph Manager - only render when we have data and canvas is ready */}
                {canvasReady && graphData.nodes.length > 0 && (
                    <GraphManager graphData={graphData} />
                )}

                {/* Fallback cube removed - was showing when graph data was loading */}

                {/* BotsVisualization for agent graph */}
                <BotsVisualization />

                {/* Camera controls with SpacePilot integration */}
                <OrbitControls
                    ref={orbitControlsRef}
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                    zoomSpeed={0.8}
                    panSpeed={0.8}
                    rotateSpeed={0.8}
                />
                {/* Using the simpler SpacePilot integration that works with useFrame */}
                <SpacePilotSimpleIntegration orbitControlsRef={orbitControlsRef} />

                {/* Add the Head-Tracked Parallax Controller */}
                <HeadTrackedParallaxController />

                {/* XR Support - causes graph to disappear */}
                {/* {xrEnabled && <XRController />} */}
                {/* {xrEnabled && <XRVisualisationConnector />} */}

                {/* Post-processing effects - using modern R3F selective bloom */}
                <SelectiveBloom enabled={enableBloom || enableGlow} />

                {/* Performance stats */}
                {showStats && <Stats />}
            </Canvas>
        </div>
    );
};

export default GraphCanvas;
```

This implementation provides a complete, end-to-end solution for adding a highly immersive head-tracked parallax effect to your application. It's fully integrated with your existing settings store, control panel, and rendering pipeline.