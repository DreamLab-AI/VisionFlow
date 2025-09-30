# XR API Reference

## Core Classes

### BabylonScene

Main scene management class for the immersive experience.

#### Constructor

```typescript
new BabylonScene(canvas: HTMLCanvasElement)
```

Creates a new Babylon.js scene with XR capabilities.

**Parameters:**
- `canvas` - HTML canvas element for rendering

#### Methods

##### `updateGraph(graphData: GraphData, nodePositions?: Float32Array): void`

Updates the graph visualization with new data.

**Parameters:**
- `graphData` - Graph structure with nodes and edges
- `nodePositions` - Optional Float32Array of node positions (x,y,z triplets)

**Example:**
```typescript
const graphData = {
  nodes: [
    { id: '1', label: 'Node 1', type: 'agent' },
    { id: '2', label: 'Node 2', type: 'document' }
  ],
  edges: [
    { source: '1', target: '2', weight: 1.0 }
  ]
};

babylonScene.updateGraph(graphData, positionsArray);
```

##### `setBotsData(data: BotsData): void`

Updates bot/agent visualization data.

**Parameters:**
- `data` - Bot data including positions and states

##### `setSettings(settings: Settings): void`

Updates scene settings from the settings store.

**Parameters:**
- `settings` - Configuration object with graph, visualization, and performance settings

##### `getScene(): BABYLON.Scene`

Returns the underlying Babylon.js scene instance.

**Returns:** `BABYLON.Scene` - The scene object

##### `getEngine(): BABYLON.Engine`

Returns the Babylon.js engine instance.

**Returns:** `BABYLON.Engine` - The engine object

##### `run(): void`

Starts the render loop.

##### `dispose(): void`

Cleans up all resources and stops rendering.

---

### GraphRenderer

Handles rendering of graph nodes, edges, and labels.

#### Constructor

```typescript
new GraphRenderer(scene: BABYLON.Scene)
```

Creates a new graph renderer.

**Parameters:**
- `scene` - Babylon.js scene to render into

#### Methods

##### `updateNodes(nodes: Node[], positions?: Float32Array): void`

Updates node instances with new data.

**Parameters:**
- `nodes` - Array of node objects
- `positions` - Optional positions array from physics engine

**Node Object Structure:**
```typescript
interface Node {
  id: string;
  label?: string;
  type?: 'agent' | 'document' | 'entity' | 'default';
  position?: { x: number; y: number; z: number };
  colour?: string;
  size?: number;
}
```

##### `updateEdges(edges: Edge[], nodePositions?: Float32Array): void`

Updates edge connections between nodes.

**Parameters:**
- `edges` - Array of edge objects
- `nodePositions` - Optional positions for edge endpoint calculation

**Edge Object Structure:**
```typescript
interface Edge {
  source: string | number;
  target: string | number;
  weight?: number;
  colour?: string;
  opacity?: number;
}
```

##### `updateLabels(nodes: Node[]): void`

Updates text labels for nodes.

**Parameters:**
- `nodes` - Array of nodes with label information

##### `dispose(): void`

Cleans up renderer resources.

---

### XRManager

Manages WebXR session and input handling.

#### Constructor

```typescript
new XRManager(scene: BABYLON.Scene, camera: BABYLON.UniversalCamera)
```

Creates XR manager for scene.

**Parameters:**
- `scene` - Babylon.js scene
- `camera` - Main camera for non-XR fallback

#### Methods

##### `enterXR(): Promise<void>`

Attempts to start an immersive XR session.

**Returns:** Promise that resolves when XR session starts

**Example:**
```typescript
await xrManager.enterXR();
console.log('Entered XR mode');
```

##### `exitXR(): void`

Exits the current XR session.

##### `isInXR(): boolean`

Checks if currently in an XR session.

**Returns:** `boolean` - True if in XR mode

##### `onNodeSelected: Observable<NodeSelectionEvent>`

Observable for node selection events.

**Event Structure:**
```typescript
interface NodeSelectionEvent {
  nodeId: string;
  position: BABYLON.Vector3;
  inputSource: 'controller' | 'hand';
  hand?: 'left' | 'right';
}
```

**Example:**
```typescript
xrManager.onNodeSelected.add((event) => {
  console.log(`Selected node: ${event.nodeId}`);
});
```

##### `dispose(): void`

Cleans up XR resources.

---

### XRUI

Manages 3D user interface in XR space.

#### Constructor

```typescript
new XRUI(scene: BABYLON.Scene)
```

Creates XR UI manager.

**Parameters:**
- `scene` - Babylon.js scene

#### Methods

##### `setSettingsChangeCallback(callback: SettingsCallback): void`

Sets callback for settings changes.

**Parameters:**
- `callback` - Function called when UI settings change

**Callback Signature:**
```typescript
type SettingsCallback = (path: string, value: any) => void;
```

##### `updateSettings(settings: UISettings): void`

Updates UI controls with new settings values.

**Parameters:**
- `settings` - Settings object

**Settings Structure:**
```typescript
interface UISettings {
  graph?: {
    nodeSize?: number;      // 0.05 - 0.5
    showLabels?: boolean;
    edgeOpacity?: number;   // 0 - 1
  };
  visualization?: {
    showBots?: boolean;
    showEdges?: boolean;
  };
  performance?: {
    maxNodes?: number;      // 100 - 5000
    enablePhysics?: boolean;
  };
}
```

##### `toggleVisibility(): void`

Toggles UI panel visibility.

##### `setPosition(position: BABYLON.Vector3): void`

Sets UI panel position in 3D space.

**Parameters:**
- `position` - New position vector

##### `attachToController(controller: WebXRInputSource): void`

Attaches UI panel to follow a controller.

**Parameters:**
- `controller` - WebXR controller input source

##### `dispose(): void`

Cleans up UI resources.

---

## Hook APIs

### useImmersiveData

React hook for bridging data to the immersive scene.

```typescript
function useImmersiveData(): ImmersiveData
```

**Returns:**
```typescript
interface ImmersiveData {
  graphData: GraphData | null;
  nodePositions: Float32Array | null;
  botsData: BotsData | null;
  settings: Settings;
  isLoading: boolean;
  error: Error | null;
}
```

**Usage Example:**
```typescript
function ImmersiveApp() {
  const { graphData, nodePositions, settings } = useImmersiveData();

  useEffect(() => {
    if (graphData && babylonScene) {
      babylonScene.updateGraph(graphData, nodePositions);
    }
  }, [graphData, nodePositions]);

  return <canvas ref={canvasRef} />;
}
```

---

## Service APIs

### quest3AutoDetector

Service for detecting Quest 3 devices.

#### Methods

##### `isQuest3Browser(): boolean`

Detects if running on Quest 3 browser.

**Returns:** `boolean` - True if Quest 3 detected

##### `hasWebXRSupport(): Promise<boolean>`

Checks for WebXR API support.

**Returns:** `Promise<boolean>` - Resolves to true if WebXR available

##### `canEnterAR(): Promise<boolean>`

Checks if AR mode is supported.

**Returns:** `Promise<boolean>` - True if immersive-ar supported

##### `shouldAutoStart(): boolean`

Determines if XR should start automatically.

**Returns:** `boolean` - Based on device and settings

**Example:**
```typescript
import { quest3AutoDetector } from '@/services/quest3AutoDetector';

const isQuest3 = quest3AutoDetector.isQuest3Browser();
const hasXR = await quest3AutoDetector.hasWebXRSupport();

if (isQuest3 && hasXR) {
  // Auto-start immersive mode
}
```

---

## Event System

### Scene Events

The Babylon scene emits various events for XR interactions:

#### Node Events

```typescript
scene.onNodeSelectedObservable.add((nodeData) => {
  // nodeData: { nodeId: string, mesh: BABYLON.Mesh }
});

scene.onNodeHoverObservable.add((nodeData) => {
  // nodeData: { nodeId: string, mesh: BABYLON.Mesh }
});

scene.onNodeReleasedObservable.add((nodeData) => {
  // nodeData: { nodeId: string, mesh: BABYLON.Mesh }
});
```

#### XR Session Events

```typescript
xrHelper.onStateChangedObservable.add((state) => {
  // state: WebXRState enum
  // ENTERING_XR, IN_XR, EXITING_XR, NOT_IN_XR
});

xrHelper.onInitialXRPoseSetObservable.add((xrCamera) => {
  // Initial XR camera pose set
});
```

#### Controller Events

```typescript
xrInput.onControllerAddedObservable.add((controller) => {
  // controller: WebXRInputSource
  controller.onMotionControllerInitObservable.add((motionController) => {
    // Access buttons, axes, haptics
  });
});

xrInput.onControllerRemovedObservable.add((controller) => {
  // Controller disconnected
});
```

#### Hand Tracking Events

```typescript
handTracking.onHandAddedObservable.add((hand) => {
  // hand: WebXRHand
  const joints = hand.getJointMesh(WebXRHandJoint.INDEX_FINGER_TIP);
});

handTracking.onHandRemovedObservable.add((hand) => {
  // Hand tracking lost
});
```

---

## Types and Interfaces

### Core Types

```typescript
// Graph data structure
interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: {
    totalNodes: number;
    totalEdges: number;
    lastUpdated: Date;
  };
}

// Node definition
interface Node {
  id: string;
  label?: string;
  type?: NodeType;
  position?: Vector3;
  colour?: string;
  size?: number;
  metadata?: any;
}

// Edge definition
interface Edge {
  source: string | number;
  target: string | number;
  weight?: number;
  directed?: boolean;
  colour?: string;
  opacity?: number;
}

// Node types
type NodeType = 'agent' | 'document' | 'entity' | 'default';

// Vector3 compatible
interface Vector3 {
  x: number;
  y: number;
  z: number;
}

// Bot/Agent data
interface BotsData {
  bots: Bot[];
  connections: Connection[];
}

interface Bot {
  id: string;
  name: string;
  position: Vector3;
  state: 'active' | 'idle' | 'processing';
  type: string;
}
```

### XR Types

```typescript
// XR session configuration
interface XRConfig {
  sessionMode: 'immersive-ar' | 'immersive-vr' | 'inline';
  referenceSpaceType: 'local-floor' | 'bounded-floor' | 'unbounded';
  optionalFeatures: string[];
  requiredFeatures: string[];
}

// XR input source
interface XRInputConfig {
  handTracking: boolean;
  controllerProfiles: string[];
  hapticActuators: boolean;
}

// XR interaction event
interface XRInteractionEvent {
  source: WebXRInputSource;
  ray: BABYLON.Ray;
  hit: BABYLON.PickingInfo;
  timestamp: number;
}
```

### Settings Types

```typescript
// Complete settings structure
interface Settings {
  xr: {
    enabled: boolean;
    autoStart: boolean;
    mode: 'immersive-ar' | 'immersive-vr';
    handTracking: boolean;
    controllerSupport: boolean;
  };
  graph: {
    nodeSize: number;
    edgeOpacity: number;
    showLabels: boolean;
    layout: 'force' | 'hierarchical' | 'circular';
  };
  visualization: {
    showBots: boolean;
    showEdges: boolean;
    animationSpeed: number;
    particleEffects: boolean;
  };
  performance: {
    maxNodes: number;
    enablePhysics: boolean;
    targetFPS: number;
    adaptiveQuality: boolean;
  };
}
```

---

## Utility Functions

### Helpers

```typescript
// Convert node ID to array index
function nodeIdToIndex(nodeId: string | number): number {
  return typeof nodeId === 'string' ? parseInt(nodeId, 10) : nodeId;
}

// Get node position from Float32Array
function getNodePosition(
  nodeId: string | number,
  positions: Float32Array
): BABYLON.Vector3 | null {
  const index = nodeIdToIndex(nodeId);
  const offset = index * 3;

  if (offset + 2 < positions.length) {
    return new BABYLON.Vector3(
      positions[offset],
      positions[offset + 1],
      positions[offset + 2]
    );
  }
  return null;
}

// Check XR availability
async function checkXRAvailable(): Promise<boolean> {
  if (!navigator.xr) return false;

  try {
    const supported = await navigator.xr.isSessionSupported('immersive-ar');
    return supported;
  } catch {
    return false;
  }
}

// Format settings path
function formatSettingsPath(category: string, key: string): string {
  return `${category}.${key}`;
}
```

---

## Constants

```typescript
// XR Constants
export const XR_CONSTANTS = {
  DEFAULT_SESSION_MODE: 'immersive-ar',
  DEFAULT_REFERENCE_SPACE: 'local-floor',
  CONTROLLER_PROFILES: [
    'oculus-touch-v3',
    'oculus-touch',
    'generic-trigger'
  ],
  HAND_JOINT_COUNT: 25,
  TARGET_FRAMERATE: 90,
  MAX_RENDER_DISTANCE: 100,
  UI_PANEL_DISTANCE: 2
};

// Material presets
export const MATERIAL_PRESETS = {
  NODE_DEFAULT: {
    diffuseColor: new BABYLON.Color3(0.3, 0.5, 1.0),
    emissiveColor: new BABYLON.Color3(0.1, 0.2, 0.5),
    specularColor: new BABYLON.Color3(0.2, 0.3, 0.5)
  },
  EDGE_DEFAULT: {
    diffuseColor: new BABYLON.Color3(0.7, 0.7, 0.8),
    emissiveColor: new BABYLON.Color3(0.3, 0.3, 0.4),
    alpha: 0.8
  },
  UI_BACKGROUND: {
    diffuseColor: new BABYLON.Color3(0.15, 0.15, 0.2),
    emissiveColor: new BABYLON.Color3(0.1, 0.1, 0.15),
    specularColor: new BABYLON.Color3(0, 0, 0)
  }
};

// Performance thresholds
export const PERFORMANCE_THRESHOLDS = {
  LOW_FPS: 30,
  TARGET_FPS: 72,
  HIGH_FPS: 90,
  MAX_NODES_LOW: 500,
  MAX_NODES_MEDIUM: 1000,
  MAX_NODES_HIGH: 2000
};
```

---

## Error Handling

### XR Errors

```typescript
class XRNotSupportedError extends Error {
  constructor(message = 'WebXR not supported on this device') {
    super(message);
    this.name = 'XRNotSupportedError';
  }
}

class XRSessionError extends Error {
  constructor(message = 'Failed to start XR session') {
    super(message);
    this.name = 'XRSessionError';
  }
}

// Usage
try {
  await xrManager.enterXR();
} catch (error) {
  if (error instanceof XRNotSupportedError) {
    // Show fallback UI
  } else if (error instanceof XRSessionError) {
    // Retry or show error message
  }
}
```

### Error Recovery

```typescript
// Automatic XR session recovery
xrHelper.onStateChangedObservable.add((state) => {
  if (state === WebXRState.NOT_IN_XR) {
    // Session ended unexpectedly
    setTimeout(() => {
      xrManager.enterXR().catch(console.error);
    }, 1000);
  }
});
```

---

## Examples

### Complete Integration Example

```typescript
import { BabylonScene } from '@/immersive/babylon/BabylonScene';
import { useImmersiveData } from '@/immersive/hooks/useImmersiveData';

function ImmersiveApp() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const babylonSceneRef = useRef<BabylonScene | null>(null);
  const { graphData, nodePositions, settings } = useImmersiveData();

  useEffect(() => {
    if (canvasRef.current && !babylonSceneRef.current) {
      // Initialize Babylon scene
      babylonSceneRef.current = new BabylonScene(canvasRef.current);
      babylonSceneRef.current.run();

      // Setup XR button
      const xrButton = document.getElementById('xr-button');
      xrButton?.addEventListener('click', async () => {
        await babylonSceneRef.current?.xrManager.enterXR();
      });
    }

    return () => {
      babylonSceneRef.current?.dispose();
      babylonSceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    // Update scene with new data
    if (babylonSceneRef.current && graphData) {
      babylonSceneRef.current.updateGraph(graphData, nodePositions);
      babylonSceneRef.current.setSettings(settings);
    }
  }, [graphData, nodePositions, settings]);

  return (
    <div className="immersive-container">
      <canvas ref={canvasRef} />
      <button id="xr-button">Enter AR</button>
    </div>
  );
}
```

### Custom Node Interaction

```typescript
// Add custom interaction to nodes
graphRenderer.onNodeCreated.add((nodeData) => {
  const mesh = nodeData.mesh;

  // Add action manager
  mesh.actionManager = new BABYLON.ActionManager(scene);

  // On pointer over
  mesh.actionManager.registerAction(
    new BABYLON.ExecuteCodeAction(
      BABYLON.ActionManager.OnPointerOverTrigger,
      () => {
        mesh.scaling = new BABYLON.Vector3(1.2, 1.2, 1.2);
      }
    )
  );

  // On pointer out
  mesh.actionManager.registerAction(
    new BABYLON.ExecuteCodeAction(
      BABYLON.ActionManager.OnPointerOutTrigger,
      () => {
        mesh.scaling = new BABYLON.Vector3(1, 1, 1);
      }
    )
  );
});
```

### Dynamic Lighting

```typescript
// Adjust lighting based on environment
function adjustLightingForEnvironment(scene: BABYLON.Scene, brightness: number) {
  const lights = scene.lights;

  lights.forEach(light => {
    if (light instanceof BABYLON.HemisphericLight) {
      light.intensity = brightness * 1.2;
      light.groundColor = BABYLON.Color3.Lerp(
        new BABYLON.Color3(0.1, 0.1, 0.1),
        new BABYLON.Color3(0.3, 0.3, 0.4),
        brightness
      );
    }
  });

  // Update ambient
  scene.ambientColor = new BABYLON.Color3(
    brightness * 0.3,
    brightness * 0.3,
    brightness * 0.4
  );
}
```