# Quest 3 XR Setup Guide

## Overview

This guide walks you through setting up and using the XR immersive features with Meta Quest 3 or other WebXR-compatible devices.

## Prerequisites

### Hardware Requirements

- **Meta Quest 3** (recommended) or Quest 2/Pro
- Alternative: Any WebXR-compatible headset
- PC with GPU for development (optional)

### Software Requirements

- Quest browser (Oculus Browser)
- Developer mode enabled on Quest
- Node.js 18+ for development
- Chrome/Edge for desktop testing

## Quick Start

### 1. Accessing Immersive Mode

#### Automatic Detection
The system automatically detects Quest 3 devices and switches to immersive mode.

#### Manual Activation
Add one of these URL parameters:
- `https://your-app.com?immersive=true`
- `https://your-app.com?force=quest3`

#### UI Button
Click the "Enter AR" button in the immersive interface.

### 2. Basic Controls

#### Hand Tracking

| Gesture | Action |
|---------|--------|
| Index Point | Hover/highlight nodes |
| Pinch | Select/grab nodes |
| Palm Up | Open UI panel |
| Fist | Close UI panel |

#### Controller Input

| Button | Action |
|--------|--------|
| Trigger | Select/interact |
| Squeeze | Toggle UI panel |
| A/X | Reset view |
| B/Y | Toggle labels |
| Thumbstick | Move/rotate view |

## Development Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-org/your-repo.git
cd your-repo

# Install dependencies
npm install

# Start development server
npm run dev
```

### 2. Configure for XR Development

#### Enable HTTPS (Required for WebXR)

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update vite.config.ts
export default {
  server: {
    https: {
      key: fs.readFileSync('./key.pem'),
      cert: fs.readFileSync('./cert.pem')
    }
  }
}
```

#### Quest Development Mode

1. Open Quest Settings
2. Go to System → Developer
3. Enable Developer Mode
4. Enable USB Debugging

### 3. Connect Quest to PC

#### USB Connection

```bash
# Enable port forwarding
adb reverse tcp:3000 tcp:3000

# Verify connection
adb devices
```

#### Wi-Fi Connection

```bash
# Get Quest IP address
adb shell ip addr show wlan0

# Connect via IP
adb connect <quest-ip>:5555
```

## Testing Workflow

### Desktop Testing with WebXR Emulator

1. Install [WebXR API Emulator](https://chrome.google.com/webstore/detail/webxr-api-emulator/mjddjgeghkdijejnciaefnkjmkafnnje)
2. Open Chrome DevTools
3. Select WebXR tab
4. Choose device preset (Quest 3)
5. Click "Enter AR" in your app

### Quest Browser Testing

1. Open Oculus Browser on Quest
2. Navigate to `https://localhost:3000`
3. Accept certificate warning
4. App should auto-detect and enter immersive mode

### Remote Debugging

1. Connect Quest via USB
2. Open `chrome://inspect` on PC
3. Select Quest device
4. Inspect browser tabs

## Features Guide

### Graph Visualization

The 3D graph renders with:
- **Nodes**: Glowing spheres representing entities
- **Edges**: Illuminated lines showing connections
- **Labels**: Text overlays with entity names

#### Node Types and Colors

| Type | Color | Description |
|------|-------|-------------|
| Agent | Blue | AI agents |
| Document | Green | Knowledge documents |
| Entity | Red | Data entities |
| Default | White | Unclassified nodes |

### Interaction Features

#### Node Selection
- Point at a node to highlight
- Pinch/trigger to select
- Selected nodes glow brighter

#### Node Manipulation
- Grab and drag to reposition
- Two-handed scaling (pinch with both hands)
- Shake to reset position

#### Graph Navigation
- Teleport with thumbstick
- Rotate view with controller
- Zoom with two-handed gesture

### UI Panel Controls

The 3D UI panel provides:

| Control | Function | Range |
|---------|----------|-------|
| Node Size | Adjust node scale | 0.05 - 0.5 |
| Edge Opacity | Edge visibility | 0 - 1 |
| Show Labels | Toggle text labels | On/Off |
| Show Bots | Display AI agents | On/Off |
| Physics | Enable dynamics | On/Off |
| Max Nodes | Performance limit | 100 - 5000 |

### Settings Synchronization

Settings automatically sync between:
- Desktop client
- Immersive XR view
- Persistent storage

## Performance Optimization

### Quest 3 Settings

```javascript
// Optimal settings for Quest 3
{
  xr: {
    targetFramerate: 90,
    foveatedRendering: 2,
    dynamicResolution: true
  },
  graph: {
    maxNodes: 1000,
    lodDistance: 10,
    instancedRendering: true
  },
  performance: {
    shadowsEnabled: false,
    antialiasing: 'FXAA',
    textureQuality: 'medium'
  }
}
```

### Performance Tips

1. **Reduce Node Count**: Limit to 1000 nodes for smooth performance
2. **Disable Shadows**: Turn off shadows in XR mode
3. **Use LOD**: Enable level-of-detail for distant nodes
4. **Optimize Physics**: Reduce simulation rate to 30Hz
5. **Texture Compression**: Use KTX2 compressed textures

## Troubleshooting

### Issue: Black Screen in XR

**Solution**: The lighting system has been optimised with:
- Multiple light sources for better visibility
- Emissive materials on all objects
- Transparent background for AR passthrough
- Minimum ambient lighting

### Issue: No Hand Tracking

**Solutions**:
1. Enable hand tracking in Quest Settings
2. Ensure good lighting conditions
3. Remove rings/jewelry
4. Check for latest Quest software

### Issue: Controllers Not Working

**Solutions**:
1. Check battery levels
2. Re-pair controllers in Settings
3. Restart Quest device
4. Update controller firmware

### Issue: Poor Performance

**Solutions**:
1. Reduce graph complexity (`maxNodes: 500`)
2. Disable real-time physics
3. Lower texture resolution
4. Close other Quest apps
5. Enable fixed foveated rendering

### Issue: Cannot Enter XR Mode

**Solutions**:
1. Verify HTTPS is enabled
2. Check WebXR browser support
3. Update Quest browser
4. Clear browser cache
5. Check developer mode is enabled

## Advanced Configuration

### Custom Lighting

```typescript
// Adjust lighting for your environment
const scene = babylonScene.getScene();

// Modify hemispheric light
const hemiLight = scene.getLightByName('hemisphericLight');
hemiLight.intensity = 1.5; // Brighter
hemiLight.groundColor = new Color3(0.3, 0.3, 0.4);

// Add point lights for specific areas
const pointLight = new PointLight('focus', new Vector3(0, 2, 0), scene);
pointLight.intensity = 2;
pointLight.range = 5;
```

### Custom Materials

```typescript
// Create custom emissive material
const customMaterial = new StandardMaterial('custom', scene);
customMaterial.diffuseColor = new Color3(1, 0.5, 0);
customMaterial.emissiveColor = new Color3(0.5, 0.2, 0);
customMaterial.specularPower = 64;
```

### Custom Interactions

```typescript
// Add custom gesture handler
xrManager.onGestureDetected.add((gesture) => {
  if (gesture.type === 'swipe-up') {
    // Custom action
  }
});
```

## Best Practices

### User Comfort

1. **Avoid sudden movements**: Smooth transitions only
2. **Provide visual anchors**: Keep UI elements stable
3. **Respect personal space**: Keep content at arm's length
4. **Offer comfort options**: Teleport vs smooth locomotion
5. **Include boundaries**: Visual safety boundaries

### Accessibility

1. **Multiple input methods**: Support both hands and controllers
2. **Adjustable UI scale**: Allow text size changes
3. **Color blind modes**: Provide alternative colour schemes
4. **Voice commands**: Future enhancement
5. **Seated mode**: Support seated experiences

### Performance

1. **Progressive loading**: Load content as needed
2. **Culling**: Hide non-visible objects
3. **Batching**: Combine similar meshes
4. **Texture atlasing**: Reduce draw calls
5. **Async operations**: Don't block render thread

## Platform-Specific Notes

### Quest 3

- Supports colour passthrough
- 90Hz/120Hz refresh rates
- Enhanced hand tracking
- Eye tracking (future)

### Quest 2

- Grayscale passthrough only
- 72Hz/90Hz refresh rates
- Basic hand tracking
- Lower resolution

### Quest Pro

- Color passthrough
- Eye and face tracking
- 90Hz refresh rate
- Higher resolution

## Debugging Tools

### Babylon Inspector

```javascript
// Enable in development
if (process.env.NODE_ENV === 'development') {
  scene.debugLayer.show();
}
```

### Performance Monitor

```javascript
// Show FPS counter
const perfMonitor = new PerformanceMonitor();
perfMonitor.sampleRate = 30;
perfMonitor.enable();
```

### WebXR Debugger

```javascript
// Log XR events
xrHelper.onStateChangedObservable.add((state) => {
  console.log('XR State:', state);
});
```

## Resources

### Documentation
- [Babylon.js WebXR Guide](https://doc.babylonjs.com/features/featuresDeepDive/webXR)
- [Meta Quest Developer Docs](https://developer.oculus.com/documentation/)
- [WebXR W3C Specification](https://www.w3.org/TR/webxr/)

### Communities
- [Babylon.js Forum](https://forum.babylonjs.com/)
- [WebXR Discord](https://discord.gg/webxr)
- [Oculus Developer Forum](https://forums.oculusvr.com/)

### Tools
- [Babylon Playground](https://playground.babylonjs.com/)
- [A-Frame Inspector](https://aframe.io/aframe-inspector/)
- [Three.js XR Examples](https://threejs.org/examples/?q=webxr)