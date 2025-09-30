# XR Immersive System Architecture

## Overview

The XR Immersive System provides a complete WebXR-based augmented reality experience for Quest 3 and other XR devices. Built with Babylon.js, it offers full 3D graph visualization, hand tracking, controller input, and immersive UI controls.

## System Architecture

### Core Components

```
/src/immersive/
├── components/
│   └── ImmersiveApp.tsx      # React entry point for immersive mode
├── babylon/
│   ├── BabylonScene.ts       # Scene management and lighting
│   ├── GraphRenderer.ts      # Graph visualization with nodes/edges
│   ├── XRManager.ts          # WebXR session and interactions
│   └── XRUI.ts               # 3D UI controls and panels
└── hooks/
    └── useImmersiveData.ts   # Data bridge from React to Babylon
```

### Component Responsibilities

#### BabylonScene.ts
- **Purpose**: Core scene management and coordination
- **Features**:
  - Babylon.js engine initialization
  - Multi-light setup for XR visibility
  - Transparent background for AR passthrough
  - Component orchestration

#### GraphRenderer.ts
- **Purpose**: High-performance graph visualization
- **Features**:
  - Instanced mesh rendering for nodes
  - Line system for edges
  - Emissive materials for XR visibility
  - Dynamic label management
  - Real-time position updates from physics engine

#### XRManager.ts
- **Purpose**: WebXR session management
- **Features**:
  - Immersive AR mode support
  - 25-joint hand tracking system
  - Controller input handling
  - Ray casting for interactions
  - Node selection and manipulation

#### XRUI.ts
- **Purpose**: 3D user interface in XR space
- **Features**:
  - Floating control panels
  - Settings synchronization
  - Sliders, checkboxes, buttons
  - Semi-transparent backgrounds for visibility

## Lighting System

The XR environment uses a multi-light setup optimised for AR/VR visibility:

### Light Sources

1. **Hemispheric Light**
   - Intensity: 1.2
   - Ground Color: RGB(0.2, 0.2, 0.3)
   - Provides ambient illumination

2. **Directional Light**
   - Direction: (-1, -2, -1)
   - Position: (3, 9, 3)
   - Intensity: 0.8
   - Creates depth and shadows

3. **Ambient Light**
   - Color: RGB(0.3, 0.3, 0.4)
   - Ensures minimum visibility in all conditions

### Material System

All materials use emissive properties for enhanced XR visibility:

- **Nodes**: Emissive blue glow (0.1, 0.2, 0.5)
- **Edges**: Emissive gray glow (0.3, 0.3, 0.4)
- **UI Panels**: Slight emissive background (0.1, 0.1, 0.15)

## Data Flow

### Integration with React Application

```typescript
// Data flow pipeline
React App → graphDataManager → useImmersiveData → BabylonScene → GraphRenderer
         → botsDataContext  ↗                   ↘
         → settingsStore                         → XRUI
```

### Real-time Updates

1. **Graph Data**: Subscribed through `graphDataManager.onGraphDataChange`
2. **Node Positions**: Float32Array from physics worker
3. **Settings**: Bidirectional sync with `settingsStore`
4. **Bot Data**: Live updates from `BotsDataContext`

## WebXR Features

### Supported Modes

- **immersive-ar**: Quest 3 passthrough AR
- **immersive-vr**: Full VR mode (fallback)
- **inline**: Non-immersive preview

### Input Methods

#### Hand Tracking
- 25 joints per hand
- Index finger tip for pointing
- Pinch gestures for selection
- Palm orientation for UI panels

#### Controller Input
- Trigger: Select/interact
- Squeeze: Toggle UI panel
- Thumbstick: Navigation
- Buttons: Context actions

### Interaction System

```typescript
// Ray casting for node selection
const ray = new Ray(
  controllerPosition,
  controllerDirection
);
const hit = scene.pickWithRay(ray);
if (hit.pickedMesh?.metadata?.nodeId) {
  // Handle node interaction
}
```

## Quest 3 Integration

### Auto-Detection

The system automatically detects Quest 3 devices using:

1. User agent string matching
2. WebXR capability checking
3. AR mode availability

### Manual Activation

Users can force immersive mode via:
- URL parameter: `?immersive=true`
- URL parameter: `?force=quest3`
- UI button: "Enter AR"

### Entry Points

```typescript
// App.tsx switching logic
if (shouldUseImmersiveClient()) {
  return <ImmersiveApp />;
} else {
  return <MainLayout />;
}
```

## Performance Optimizations

### Rendering

- **Instanced Meshes**: Single draw call for thousands of nodes
- **Line Systems**: Efficient edge rendering
- **Texture Atlasing**: Combined UI textures
- **LOD System**: Distance-based detail reduction

### Memory Management

- **Object Pooling**: Reused mesh instances
- **Lazy Loading**: Components loaded on demand
- **Dispose Patterns**: Proper cleanup on unmount

### Update Strategies

- **Batch Updates**: Grouped position changes
- **Dirty Flagging**: Only update changed elements
- **Frame Skipping**: Adaptive render rates

## API Reference

### BabylonScene

```typescript
class BabylonScene {
  constructor(canvas: HTMLCanvasElement)
  updateGraph(graphData: any, nodePositions?: Float32Array): void
  setBotsData(data: any): void
  setSettings(settings: any): void
  getScene(): BABYLON.Scene
  getEngine(): BABYLON.Engine
  run(): void
  dispose(): void
}
```

### GraphRenderer

```typescript
class GraphRenderer {
  constructor(scene: Scene)
  updateNodes(nodes: any[], positions?: Float32Array): void
  updateEdges(edges: any[], nodePositions?: Float32Array): void
  updateLabels(nodes: any[]): void
  dispose(): void
}
```

### XRManager

```typescript
class XRManager {
  constructor(scene: Scene, camera: UniversalCamera)
  enterXR(): Promise<void>
  exitXR(): void
  dispose(): void
}
```

### XRUI

```typescript
class XRUI {
  constructor(scene: Scene)
  setSettingsChangeCallback(callback: Function): void
  updateSettings(settings: Settings): void
  toggleVisibility(): void
  dispose(): void
}
```

## Configuration

### Settings Structure

```yaml
xr:
  enabled: true
  autoStart: true
  mode: immersive-ar
  handTracking: true
  controllerSupport: true

graph:
  nodeSize: 0.1
  edgeOpacity: 0.8
  showLabels: true

visualization:
  showBots: true
  showEdges: true

performance:
  maxNodes: 1000
  enablePhysics: true
```

## Testing

### Local Development

1. Use Chrome/Edge with WebXR emulator extension
2. Enable `chrome://flags/#webxr` flags
3. Use Meta Quest Link for desktop testing

### Device Testing

1. Connect Quest 3 via USB
2. Enable developer mode
3. Use adb reverse for localhost access
4. Open in Oculus Browser

### Debug Tools

- Babylon Inspector: `scene.debugLayer.show()`
- WebXR API Emulator: Chrome extension
- Remote debugging: `chrome://inspect`

## Troubleshooting

### Common Issues

1. **Black screen in XR**
   - Check lighting configuration
   - Verify emissive materials
   - Ensure transparent clear colour

2. **No hand tracking**
   - Enable in Quest settings
   - Check HTTPS requirement
   - Verify feature detection

3. **Poor performance**
   - Reduce node count
   - Disable shadows
   - Lower texture resolution

## Future Enhancements

### Planned Features

- Multi-user collaboration
- Spatial audio integration
- Gesture recognition
- Voice commands
- Cloud rendering
- Physics-based interactions
- Haptic feedback

### Experimental Features

- Neural rendering
- Eye tracking
- Facial expression mapping
- Full body tracking
- Environment mapping
- Persistent spatial anchors

## Migration Notes

### From @react-three/xr

The system was completely migrated from React Three Fiber's XR implementation to Babylon.js for better performance and WebXR compliance. Key changes:

1. **Renderer**: Three.js → Babylon.js
2. **XR**: @react-three/xr → Native WebXR
3. **UI**: React components → Babylon GUI
4. **State**: React state → Direct scene updates

### Breaking Changes

- Removed all `@react-three/xr` dependencies
- Deleted `/src/features/xr/` directory
- New API structure
- Different event handling

## Resources

### Documentation

- [Babylon.js WebXR](https://doc.babylonjs.com/features/featuresDeepDive/webXR/introToWebXR)
- [WebXR Device API](https://immersive-web.github.io/webxr/)
- [Quest 3 Development](https://developer.oculus.com/documentation/)

### Tools

- [Babylon Playground](https://playground.babylonjs.com/)
- [WebXR Samples](https://immersive-web.github.io/webxr-samples/)
- [Oculus Developer Hub](https://developer.oculus.com/downloads/package/oculus-developer-hub-win/)