---
title: WebXR Setup Guide - VisionFlow Implementation
description: > **📋 Role**: Technical implementation reference for developers building WebXR features into VisionFlow. > For setting up your XR development environment with Turbo Flow Claude, see [XR Setup - Dev...
category: guide
tags:
  - tutorial
  - api
  - api
  - backend
updated-date: 2025-12-18
difficulty-level: intermediate
---


# WebXR Setup Guide - VisionFlow Implementation

> **📋 Role**: Technical implementation reference for developers building WebXR features into VisionFlow.
> For setting up your XR development environment with Turbo Flow Claude, see [XR Setup - Development Environment](./user/xr-setup.md).

[← Documentation Home](../../../README.md) > XR Setup

## Overview

This comprehensive guide covers setting up and using WebXR immersive features across multiple devices and browsers. Meta Quest 3 is the primary tested platform and serves as the reference implementation, but these instructions apply to any WebXR-compatible headset.

## Prerequisites

### Hardware Requirements

**Recommended (Primary Platform)**
- **Meta Quest 3** - Full colour passthrough AR, 90Hz/120Hz refresh, enhanced hand tracking
- PC with GPU for development (optional)

**Supported Alternatives**
- **Quest 2** - Grayscale passthrough, 72Hz/90Hz refresh, basic hand tracking
- **Quest Pro** - Colour passthrough, eye/face tracking, 90Hz refresh
- **Any WebXR-compatible headset** - HTC Vive, Valve Index, Pico, etc.

### Software Requirements

**Quest Devices**
- Quest Browser (Oculus Browser) - Built-in WebXR support
- Developer mode enabled (for testing)
- Latest Quest OS version

**Development Environment**
- Node.js 18+
- Chrome/Edge (with WebXR emulator for desktop testing)
- Firefox Nightly (WebXR support in development)

### Browser Compatibility

| Browser | Platform | WebXR Support | Notes |
|---------|----------|---------------|-------|
| Quest Browser | Quest 2/3/Pro | ✅ Full | Native support, recommended |
| Chrome/Edge | Desktop | ⚠️ Emulator only | Requires WebXR API Emulator extension |
| Firefox Nightly | Desktop | 🔬 Experimental | Enable `dom.vr.enabled` in about:config |
| Safari | iOS/macOS | ❌ Not supported | No WebXR implementation |

**Browser-Specific Notes:**

**Chrome/Edge (Desktop)**
1. Install [WebXR API Emulator](https://chrome.google.com/webstore/detail/webxr-api-emulator/mjddjgeghkdijejnciaefnkjmkafnnje)
2. Enable flags at `chrome://flags`:
   - `#webxr-incubations`
   - `#webxr-hand-input`
3. Open DevTools → WebXR tab → Select device preset

**Firefox Nightly**
1. Navigate to `about:config`
2. Set `dom.vr.enabled` to `true`
3. Set `dom.vr.webxr.enabled` to `true`
4. Restart browser
5. Note: Hand tracking support is limited

## Quick Start

### 1. Accessing Immersive Mode

#### Automatic Detection
The system automatically detects Quest devices via user agent and switches to immersive mode.

#### Manual Activation
Force immersive mode using URL parameters:
```
https://your-app.com?immersive=true
https://your-app.com?force=quest3
```

#### UI Button
Click the "Enter AR" button that appears in the bottom-right corner of the immersive interface.

### 2. Basic Controls

#### Hand Tracking

Hand tracking is available on Quest 2/3/Pro and must be enabled in Quest Settings → Hands and Controllers.

| Gesture | Action |
|---------|--------|
| Index Point | Hover/highlight nodes in the graph |
| Pinch | Select/grab nodes |
| Palm Up | Open 3D UI control panel |
| Fist | Close UI panel |
| Two-hand Pinch | Scale nodes |
| Shake | Reset node position |

**Hand Tracking Tips:**
- Ensure good lighting conditions
- Keep hands within camera field of view
- Remove rings and jewellery that may interfere
- Hold hands at chest level for best tracking

#### Controller Input

All Quest controllers supported (Touch, Touch Plus, Touch Pro).

| Button | Action |
|--------|--------|
| Trigger | Select/interact with nodes |
| Squeeze | Toggle UI panel visibility |
| A/X | Reset view to centre |
| B/Y | Toggle node labels |
| Thumbstick | Move/rotate view (teleport) |
| Thumbstick Click | Recenter position |

## Development Setup

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/your-org/your-repo.git
cd your-repo

# Install dependencies
npm install

# Start development server
npm run dev
```

### 2. Configure for WebXR Development

#### Enable HTTPS (Required)

WebXR requires HTTPS in production and for device testing. For local development:

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update vite.config.ts
import fs from 'fs';

export default {
  server: {
    https: {
      key: fs.readFileSync('./key.pem'),
      cert: fs.readFileSync('./cert.pem')
    },
    host: '0.0.0.0', // Allow network access
    port: 3000
  }
}
```

#### Quest Development Mode

1. Enable Developer Mode:
   - Open Meta Quest mobile app
   - Go to Menu → Devices → Select your headset
   - Developer Mode → Enable
   - Accept warnings
2. Enable USB Debugging (Settings → System → Developer)
3. Connect via USB and accept debugging prompt

### 3. Connect Quest to Development PC

#### USB Connection (Recommended)

```bash
# Ensure ADB is installed (part of Android SDK)
adb devices  # Should show your Quest

# Enable port forwarding
adb reverse tcp:3000 tcp:3000

# Verify connection
adb shell ip addr show wlan0
```

Now access `https://localhost:3000` directly from Quest Browser.

#### Wi-Fi Connection

```bash
# Get Quest IP address
adb shell ip addr show wlan0 | grep inet

# Connect wirelessly
adb connect <quest-ip>:5555

# Forward ports
adb reverse tcp:3000 tcp:3000
```

Access via `https://<your-pc-ip>:3000` from Quest Browser. Accept certificate warning.

## Testing Workflow

### Desktop Testing with Emulator

1. Install [WebXR API Emulator](https://chrome.google.com/webstore/detail/webxr-api-emulator/mjddjgeghkdijejnciaefnkjmkafnnje) for Chrome/Edge
2. Open application in browser
3. Open DevTools (F12)
4. Navigate to WebXR tab
5. Select device preset (Quest 3 recommended)
6. Enable "Stereo Effects" for realistic view
7. Click "Enter AR" button in application
8. Use emulator controls to simulate hand/controller input

**Emulator Limitations:**
- No actual passthrough (uses transparent background)
- Simplified hand tracking
- Performance may differ from device

### Quest Browser Testing

1. Put on Quest headset
2. Open Browser application
3. Navigate to `https://localhost:3000` or `https://<pc-ip>:3000`
4. Accept certificate warning (self-signed cert)
5. Application should auto-detect Quest and show immersive interface
6. Click "Enter AR" to start WebXR session

### Remote Debugging

Debug Quest Browser from your PC:

1. Connect Quest via USB
2. Open Chrome on PC
3. Navigate to `chrome://inspect`
4. Select Quest device under "Remote Target"
5. Click "Inspect" on browser tab
6. Use DevTools normally (console, network, performance)

**Remote Debugging Tips:**
- Keep USB connection stable
- Use Console for runtime errors
- Performance tab shows frame rates
- Network tab monitors WebSocket traffic

## Features Guide

### Graph Visualisation

The 3D knowledge graph renders in XR space with:
- **Nodes**: Glowing spheres representing entities
- **Edges**: Illuminated lines showing relationships
- **Labels**: Billboard text overlays with entity names
- **Physics**: Optional force-directed layout simulation

#### Node Types and Colours

| Type | Colour | Description |
|------|--------|-------------|
| Agent | Blue | AI agents and bots |
| Document | Green | Knowledge documents |
| Entity | Red | Data entities |
| Default | White | Unclassified nodes |

Colours use emissive materials for visibility in AR passthrough.

### Interaction Features

#### Node Selection
- Point at node to highlight (outline glow)
- Pinch or trigger to select
- Selected nodes glow brighter with pulse effect
- Selection state syncs across desktop and XR views

#### Node Manipulation
- Grab and drag nodes to reposition
- Two-handed pinch and pull/push to scale
- Shake node to reset to physics position
- Long-press to pin node in place

#### Graph Navigation
- **Teleport**: Use thumbstick to teleport to location
- **Rotate View**: Twist controller or two-hand gesture
- **Zoom**: Two-handed pinch-to-zoom gesture
- **Recenter**: Click thumbstick to reset position

### 3D UI Panel Controls

The floating control panel provides real-time settings adjustment:

| Control | Function | Range | Default |
|---------|----------|-------|---------|
| Node Size | Adjust node scale | 0.05 - 0.5 | 0.1 |
| Edge Opacity | Edge line visibility | 0 - 1 | 0.8 |
| Show Labels | Toggle text labels | On/Off | On |
| Show Bots | Display AI agent nodes | On/Off | On |
| Physics | Enable force simulation | On/Off | On |
| Max Nodes | Performance limit | 100 - 5000 | 1000 |

**UI Panel Access:**
- Palm-up gesture (hand tracking)
- Squeeze button (controller)
- Automatically appears on first entry

**Settings Synchronisation:**
Settings automatically sync bidirectionally between:
- Desktop client interface
- Immersive XR control panel
- Browser localStorage (persistence)

## Hand Tracking Setup

### Enable Hand Tracking

**Quest Settings:**
1. Settings → Hands and Controllers
2. Enable "Hand Tracking"
3. Select "Hands Only" or "Auto Switch"

**Application Check:**
```javascript
// Verify hand tracking availability
const handTracking = xrHelper.featuresManager.enableFeature(
  BABYLON.WebXRHandTracking,
  'latest'
);
console.log('Hand tracking available:', !!handTracking);
```

### Hand Tracking Gestures

The system recognises 25 joints per hand for precise tracking:

**Pointing (Index Finger)**
- Extended index, other fingers curled
- Used for ray-based selection
- Hover over nodes to highlight

**Pinch (Thumb + Index)**
- Thumb and index fingertips touching
- Used for selection and grabbing
- Provides haptic feedback on Quest 3

**Palm Gestures**
- Open palm facing up: Opens UI panel
- Closed fist: Closes UI panel
- Palm orientation tracked for UI positioning

**Two-Hand Gestures**
- Both hands pinching simultaneously: Scale mode
- Pull hands apart to enlarge
- Push together to shrink

### Hand Tracking Troubleshooting

**Issue: Hands not tracked**
- Ensure good lighting (not too dim or bright)
- Remove gloves, rings, long sleeves covering hands
- Update Quest OS to latest version
- Check Settings → Hands and Controllers → Hand Tracking enabled

**Issue: Tracking stutters**
- Reduce graph complexity (fewer nodes)
- Keep hands within 60° cone in front of headset
- Avoid rapid hand movements
- Check CPU usage isn't at 100%

## Controller Setup

### Controller Pairing

**Quest 2/3/Pro:**
1. Settings → Devices → Controllers
2. Hold Oculus button + B/Y for 3 seconds
3. Follow on-screen pairing instructions
4. Check battery level (Controllers → Battery Status)

### Controller Input Mapping

The application uses WebXR's standard input profiles:

```javascript
// Controller button mapping
inputSource.gamepad.buttons[0] // Trigger
inputSource.gamepad.buttons[1] // Squeeze
inputSource.gamepad.buttons[3] // A/X button
inputSource.gamepad.buttons[4] // B/Y button
inputSource.gamepad.axes[2]    // Thumbstick X
inputSource.gamepad.axes[3]    // Thumbstick Y
```

### Controller Troubleshooting

**Issue: Controllers not detected**
- Check battery level (replace if low)
- Re-pair controllers (hold Oculus + B/Y)
- Restart Quest device
- Check for firmware updates (Settings → Devices → Controllers)

**Issue: Input lag or drift**
- Replace batteries with fresh ones
- Clear Guardian boundary and redo setup
- Check for wireless interference (2.4GHz devices)
- Update controller firmware

## Spatial Audio

Spatial audio is implemented for multi-user sessions via Vircadia integration, providing 3D positional audio based on user locations in the virtual space.

### Architecture

**Web Audio API + WebRTC:**
- Web Audio API: 3D panning and distance attenuation
- WebRTC: Peer-to-peer voice transmission
- HRTF: Head-related transfer function for realistic positioning

### Configuration

```javascript
const spatialAudioConfig = {
  maxDistance: 20,        // Max audible distance (metres)
  rolloffFactor: 1,       // Distance attenuation rate
  refDistance: 1,         // Reference distance for attenuation
  panningModel: 'HRTF',   // Use HRTF for binaural audio
  distanceModel: 'inverse' // Inverse distance attenuation
};
```

### Features

**Position-Based Audio:**
- Voice volume decreases with distance
- Stereo panning based on relative position
- Orientation affects perceived direction

**Quality Settings:**
```javascript
const audioConstraints = {
  echoCancellation: true,    // Remove echo feedback
  noiseSuppression: true,    // Filter background noise
  autoGainControl: true,     // Normalise volume levels
  sampleRate: 48000          // High-quality audio (48kHz)
};
```

### Usage

```javascript
// Update listener position (local user)
spatialAudioManager.updateListenerPosition(
  cameraPosition,
  forwardVector,
  upVector
);

// Update remote peer position
spatialAudioManager.updatePeerPosition(
  agentId,
  peerPosition
);

// Mute/unmute microphone
spatialAudioManager.setMuted(true);
```

### Status

⚠️ **Currently Available in Vircadia Multi-User Mode Only**

Spatial audio is implemented and ready for use when Vircadia multi-user features are enabled. See Vircadia Integration documentation for activation instructions.

## Vircadia Multi-User Integration

### Overview

Vircadia is an open-source metaverse platform that enables collaborative multi-user XR experiences. The application includes full integration support for:

- **Multi-user graph exploration** - Multiple users in same 3D space
- **Avatars and presence** - 3D avatars with nameplates
- **Spatial audio** - Position-based voice communication
- **Collaborative interactions** - Shared selections and annotations
- **Session persistence** - Resume sessions across connections

### Architecture

```
Single-User XR (Current)          Multi-User XR (Vircadia)
├── Babylon.js Scene              ├── Babylon.js Scene
├── WebXR Session                 ├── WebXR Session
└── Local State                   ├── Vircadia Client
                                  ├── Entity Sync Manager
                                  ├── Avatar Manager
                                  └── Spatial Audio Manager
```

### Configuration

Enable Vircadia in environment variables:

```bash
# .env
VITE-VIRCADIA-ENABLED=false           # Feature flag
VITE-VIRCADIA-SERVER-URL=wss://vircadia.example.com
VITE-VIRCADIA-DOMAIN-ID=visionflow-graph

# Avatar settings
VITE-VIRCADIA-AVATAR-MODEL=/assets/avatars/default.glb
VITE-VIRCADIA-SHOW-NAMEPLATES=true

# Communication
VITE-VIRCADIA-SPATIAL-AUDIO=true
VITE-VIRCADIA-AUDIO-RANGE=20.0
```

### Features

**Collaborative Exploration:**
- See other users' avatars in real-time
- Shared node selections highlighted
- Collaborative filtering and analysis
- Real-time graph state synchronisation

**Avatar System:**
- Customisable 3D avatar models (GLTF/GLB)
- Real-time position and orientation tracking
- User nameplates with status indicators
- Distance-based level of detail (LOD)

**Spatial Communication:**
- 3D positional audio based on avatar location
- Voice quality attenuates with distance
- Private channels and broadcast modes
- Mute/unmute controls

**Session Management:**
- Persistent multi-user sessions
- Join/leave notifications
- Session-specific graph views
- Cross-session state synchronisation

### Status

⚠️ **Planned Feature - Not Yet Enabled**

Vircadia integration is fully implemented in code but disabled by default. The feature is currently in beta testing.

## Performance Optimisation

### Quest 3 Optimal Settings

```javascript
// Recommended settings for Quest 3
const quest3Config = {
  xr: {
    targetFramerate: 90,           // 90Hz for smooth experience
    foveatedRendering: 2,          // Aggressive foveation
    dynamicResolution: true        // Auto-adjust resolution
  },
  graph: {
    maxNodes: 1000,                // Limit complexity
    lodDistance: 10,               // LOD threshold (metres)
    instancedRendering: true       // Use GPU instancing
  },
  performance: {
    shadowsEnabled: false,         // Disable shadows in XR
    antialiasing: 'FXAA',          // Fast antialiasing
    textureQuality: 'medium'       // Balance quality/performance
  }
};
```

### Performance Tips by Priority

**High Impact:**
1. **Reduce Node Count**: Keep under 1000 nodes for 90fps
2. **Disable Shadows**: Significant performance gain in XR
3. **Enable GPU Instancing**: Single draw call for all nodes
4. **Fixed Foveated Rendering**: Reduces pixel load by 30-40%

**Medium Impact:**
5. **LOD System**: Simplify distant nodes automatically
6. **Disable Physics**: Static layout uses less CPU
7. **Texture Compression**: Use KTX2 format for textures
8. **Reduce Label Count**: Hide labels beyond 5 metres

**Low Impact:**
9. **Lower Antialiasing**: FXAA instead of MSAA
10. **Reduce Edge Opacity**: Fewer transparent overdraw calls

### Platform-Specific Optimisations

#### Quest 2
- Target 72Hz (lower refresh rate)
- Max 500 nodes recommended
- Disable antialiasing
- Reduce texture resolution to 512×512

#### Quest 3
- Target 90Hz (default)
- Max 1000 nodes recommended
- FXAA antialiasing acceptable
- Medium texture quality (1024×1024)

#### Quest Pro
- Target 90Hz
- Max 1200 nodes (better GPU)
- Can enable eye tracking (future feature)
- Higher texture quality acceptable

## Troubleshooting

### Common Issues

#### Black Screen in XR Mode

**Symptoms**: Enter XR but see only black screen, no graph visible.

**Solutions**:
1. Check lighting configuration:
```javascript
// Verify lights exist
const lights = scene.lights;
console.log('Light count:', lights.length);
lights.forEach(light => {
  console.log(`${light.name}: intensity ${light.intensity}`);
});
```

2. Verify emissive materials:
```javascript
// Ensure nodes have emissive property
nodeMaterial.emissiveColor = new BABYLON.Color3(0.1, 0.2, 0.5);
```

3. Check transparent background:
```javascript
// Scene must be transparent for AR passthrough
scene.clearColor = new BABYLON.Color4(0, 0, 0, 0);
```

#### No Hand Tracking

**Symptoms**: Hand tracking doesn't work, controllers required.

**Solutions**:
1. Enable in Quest settings: Settings → Hands and Controllers → Hand Tracking
2. Verify HTTPS: Hand tracking requires secure context
3. Check lighting conditions (not too dark or bright)
4. Remove gloves, rings, long sleeves covering hands
5. Update Quest OS: Settings → System → Software Update

#### Controllers Not Working

**Symptoms**: Controllers detected but buttons/triggers don't respond.

**Solutions**:
1. Check battery levels: Settings → Devices → Controllers
2. Re-pair controllers: Hold Oculus button + B/Y for 3 seconds
3. Restart Quest device completely
4. Update controller firmware: Settings → Devices → Controllers → Update
5. Check for physical damage or button sticking

#### Poor Performance / Stuttering

**Symptoms**: Frame rate drops, stuttering, or judder in XR mode.

**Solutions**:
1. Reduce graph complexity:
```javascript
settingsStore.setMaxNodes(500);  // Limit to 500 nodes
```

2. Disable real-time physics:
```javascript
settingsStore.setPhysicsEnabled(false);
```

3. Lower texture resolution:
```javascript
// Reduce texture size
const nodeTexture = new BABYLON.Texture(url, scene);
nodeTexture.updateSamplingMode(BABYLON.Texture.BILINEAR-SAMPLINGMODE);
```

4. Close other Quest apps running in background
5. Enable fixed foveated rendering:
```javascript
xrHelper.baseExperience.sessionManager.updateRenderState({
  depthNear: 0.1,
  depthFar: 100,
  foveation: 2  // 0=off, 1=low, 2=medium, 3=high
});
```

6. Check Quest temperature (thermal throttling if too hot)

#### Cannot Enter XR Mode

**Symptoms**: "Enter AR" button missing or clicking does nothing.

**Solutions**:
1. Verify HTTPS enabled:
```javascript
console.log('Protocol:', window.location.protocol); // Must be https:
```

2. Check WebXR browser support:
```javascript
if ('xr' in navigator) {
  navigator.xr.isSessionSupported('immersive-ar').then(supported => {
    console.log('AR supported:', supported);
  });
} else {
  console.error('WebXR not available');
}
```

3. Update Quest browser: Settings → Apps → Browser → Update
4. Clear browser cache: Browser menu → Settings → Clear Browsing Data
5. Enable developer mode: Settings → System → Developer
6. Try forcing immersive mode: Add `?immersive=true` to URL

#### Audio Issues (Vircadia Mode)

**Symptoms**: Cannot hear other users or microphone not working.

**Solutions**:
1. Grant microphone permission when prompted
2. Check Quest audio settings: Settings → Sound
3. Verify microphone not muted in app
4. Check peer connection status:
```javascript
spatialAudioManager.getPeerCount(); // Should be > 0
```

5. Test microphone independently: Use Quest's voice commands
6. Check network connectivity (WebRTC requires good connection)

### Browser-Specific Issues

#### Chrome/Edge Desktop

**Issue**: WebXR emulator shows blank screen

**Solutions**:
- Enable flags: `chrome://flags/#webxr-incubations`
- Update emulator extension to latest version
- Refresh page after enabling emulator
- Check console for WebXR errors

#### Firefox Nightly

**Issue**: "WebXR not supported" error

**Solutions**:
- Verify `dom.vr.webxr.enabled` set to `true` in `about:config`
- Restart browser after config changes
- Use Firefox Nightly (not stable release)
- Note: Hand tracking support limited in Firefox

#### Quest Browser

**Issue**: Certificate warnings block access

**Solutions**:
- Click "Advanced" → "Proceed to localhost (unsafe)"
- For production, use valid SSL certificate (Let's Encrypt)
- Self-signed certs require manual acceptance each session

## Advanced Configuration

### Custom Lighting

Adjust lighting for different environments:

```typescript
// Access Babylon scene
const scene = babylonScene.getScene();

// Modify hemispheric light intensity
const hemiLight = scene.getLightByName('hemisphericLight');
if (hemiLight) {
  hemiLight.intensity = 1.5;  // Brighter for dark rooms
  hemiLight.groundColor = new BABYLON.Color3(0.3, 0.3, 0.4);
}

// Add point lights for specific areas
const focusLight = new BABYLON.PointLight('focus', new BABYLON.Vector3(0, 2, 0), scene);
focusLight.intensity = 2;
focusLight.range = 5;
focusLight.diffuse = new BABYLON.Color3(1, 1, 0.8); // Warm white
```

### Custom Materials

Create custom node appearances:

```typescript
// Custom emissive material for special nodes
const customMaterial = new BABYLON.StandardMaterial('custom', scene);
customMaterial.diffuseColor = new BABYLON.Color3(1, 0.5, 0);      // Orange
customMaterial.emissiveColor = new BABYLON.Color3(0.5, 0.2, 0);   // Emissive orange
customMaterial.specularPower = 64;  // Shiny surface
customMaterial.alpha = 0.9;         // Slightly transparent

// Apply to node mesh
nodeMesh.material = customMaterial;
```

### Custom Interactions

Add custom gesture handlers:

```typescript
// Register custom gesture handler
xrManager.onGestureDetected.add((gesture) => {
  if (gesture.type === 'swipe-up') {
    // Custom action: Open menu
    console.log('Swipe up detected');
    showContextMenu();
  }

  if (gesture.type === 'double-pinch') {
    // Custom action: Quick select multiple
    console.log('Double pinch detected');
    multiSelect();
  }
});

// Add custom controller button mapping
xrInputSource.onTriggerStateChangedObservable.add((state) => {
  if (state.value > 0.9) {  // Fully pressed
    console.log('Strong trigger press');
    performStrongAction();
  }
});
```

### Custom UI Panels

Create additional 3D UI panels:

```typescript
// Create custom panel
const customPanel = new BABYLON.GUI.StackPanel3D();
customPanel.position = new BABYLON.Vector3(2, 1.5, -2);
customPanel.isVisible = true;

// Add custom buttons
const helpButton = new BABYLON.GUI.Button3D('help');
helpButton.text = 'Help';
helpButton.onPointerClickObservable.add(() => {
  showHelpOverlay();
});
customPanel.addControl(helpButton);

// Add to scene
xrUIManager.addPanel(customPanel);
```

## Best Practices

### User Comfort

1. **Avoid Sudden Movements**: Use smooth transitions for camera changes
2. **Provide Visual Anchors**: Keep UI elements stable in world space
3. **Respect Personal Space**: Position content at comfortable arm's length (0.5-2m)
4. **Offer Comfort Options**: Provide both teleport and smooth locomotion
5. **Include Boundaries**: Visual indicators for guardian bounds
6. **Limit Session Duration**: Remind users to take breaks every 30 minutes

### Accessibility

1. **Multiple Input Methods**: Support both hand tracking and controllers
2. **Adjustable UI Scale**: Allow text size and UI panel size changes
3. **Colour-Blind Modes**: Provide alternative colour schemes for nodes
4. **Voice Commands**: Future enhancement for hands-free control
5. **Seated Mode**: Support fully seated experiences for accessibility
6. **High Contrast**: Ensure UI elements visible in all lighting conditions

### Performance

1. **Progressive Loading**: Load graph content as needed (pagination)
2. **Frustum Culling**: Hide objects outside camera view automatically
3. **Batching**: Combine similar meshes to reduce draw calls
4. **Texture Atlasing**: Combine multiple textures into single atlas
5. **Async Operations**: Never block render thread (use Web Workers)
6. **Monitor Frame Time**: Target 11ms per frame for 90fps

## Platform-Specific Notes

### Quest 3 Features

✅ **Available:**
- Colour passthrough AR (full RGB)
- 90Hz/120Hz refresh rates (experimental mode)
- Enhanced hand tracking with improved accuracy
- High-resolution displays (2064×2208 per eye)

🔬 **Experimental:**
- Eye tracking (requires opt-in)
- Face tracking (limited API access)
- 120Hz mode (battery intensive)

### Quest 2 Limitations

⚠️ **Limitations:**
- Grayscale passthrough only (no colour)
- 72Hz/90Hz refresh rates (no 120Hz)
- Basic hand tracking (lower accuracy)
- Lower resolution (1832×1920 per eye)

**Recommendations:**
- Use 72Hz for better battery life
- Limit to 500 nodes for smooth performance
- Controller input more reliable than hand tracking

### Quest Pro Features

✅ **Available:**
- Colour passthrough AR
- Eye tracking (requires permission)
- Face tracking (social avatars)
- 90Hz refresh rate
- Higher resolution (1800×1920 per eye)
- Improved comfort for long sessions

## Debugging Tools

### Babylon Inspector

Enable Babylon's built-in debug tools:

```javascript
// Enable inspector in development
if (import.meta.env.DEV) {
  scene.debugLayer.show({
    embedMode: true,
    globalRoot: document.body
  });
}

// Keyboard shortcuts
// F9: Toggle inspector
// F10: Toggle frame rate display
```

**Inspector Features:**
- Scene graph hierarchy
- Material property editor
- Mesh inspector with vertex data
- Performance profiler

### Performance Monitor

Track performance metrics:

```javascript
// Enable FPS counter
const perfMonitor = new BABYLON.PerformanceMonitor();
perfMonitor.sampleRate = 30;  // Samples per second
perfMonitor.enable();

// Log performance data
perfMonitor.onFrameTimeChanged = (frameTime) => {
  if (frameTime > 11) {  // Exceeds 90fps target
    console.warn(`Slow frame: ${frameTime.toFixed(2)}ms`);
  }
};
```

### WebXR Debugger

Log WebXR events for troubleshooting:

```javascript
// Log XR state changes
xrHelper.baseExperience.sessionManager.onXRSessionInit.add((session) => {
  console.log('XR Session Started:', session);
  console.log('Session mode:', session.mode);
  console.log('Render state:', session.renderState);
});

xrHelper.baseExperience.sessionManager.onXRSessionEnded.add(() => {
  console.log('XR Session Ended');
});

// Log feature availability
console.log('XR Features Available:');
xrHelper.featuresManager.getEnabledFeatures().forEach(feature => {
  console.log(`- ${feature}`);
});
```

## Resources

### Documentation

**WebXR Standards:**
- [WebXR W3C Specification](https://www.w3.org/TR/webxr/) - Official WebXR API spec
- [WebXR Device API](https://immersive-web.github.io/webxr/) - Living specification
- [WebXR Samples](https://immersive-web.github.io/webxr-samples/) - Official examples

**Babylon.js:**
- [Babylon.js WebXR Guide](https://doc.babylonjs.com/features/featuresDeepDive/webXR) - Complete WebXR documentation
- [Babylon.js Playground](https://playground.babylonjs.com/) - Interactive examples
- [Babylon.js API Reference](https://doc.babylonjs.com/typedoc/) - Full API docs

**Meta Quest:**
- [Meta Quest Developer Docs](https://developer.oculus.com/documentation/) - Official Quest development
- [Quest Browser WebXR Guide](https://developer.oculus.com/documentation/web/browser-intro/) - Browser-specific docs
- [Hand Tracking Guidelines](https://developer.oculus.com/documentation/native/android/mobile-hand-tracking/) - Best practices

**Vircadia:**
- [Vircadia Web SDK](https://github.com/vircadia/vircadia-web-sdk) - Open-source SDK
- [Vircadia World Server](https://github.com/vircadia/vircadia-world) - Multi-user backend
- [Vircadia Documentation](https://docs.vircadia.com) - Platform documentation

### VisionFlow XR Documentation

**Architecture:**
- XR Immersive System Architecture (TODO: Document to be created) - Technical deep-dive
- Vircadia Integration Architecture (TODO: Document to be created) - Multi-user design

**Guides:**
- Quest 3 Setup - Quest 3 specific instructions (see XR documentation)
- Vircadia Multi-User Setup - Multi-user configuration (planned)

**Reference:**
- XR API Reference - Complete API documentation (see project reference docs)

### Communities

**General WebXR:**
- [WebXR Discord](https://discord.gg/webxr) - WebXR community chat
- [Babylon.js Forum](https://forum.babylonjs.com/) - Official Babylon.js forum
- [Reddit r/WebXR](https://reddit.com/r/WebXR) - WebXR discussion

**Meta Quest:**
- [Oculus Developer Forum](https://forums.oculusvr.com/) - Official Quest developer forum
- [Quest Developer Discord](https://discord.gg/oculus) - Developer community

**Vircadia:**
- [Vircadia Community](https://vircadia.com) - Official community hub
- [Vircadia Discord](https://discord.gg/vircadia) - Real-time community chat

### Tools

**Development:**
- [WebXR API Emulator](https://chrome.google.com/webstore/detail/webxr-api-emulator/mjddjgeghkdijejnciaefnkjmkafnnje) - Chrome/Edge extension
- [Oculus Developer Hub](https://developer.oculus.com/downloads/package/oculus-developer-hub-win/) - Quest development tools
- [ADB (Android Debug Bridge)](https://developer.android.com/studio/command-line/adb) - Device debugging

**Testing:**
- [WebXR Test API](https://github.com/immersive-web/webxr-test-api) - Automated testing
- [BrowserStack](https://www.browserstack.com/) - Cross-browser testing
- [Three.js XR Examples](https://threejs.org/examples/?q=webxr) - Reference implementations

---

**Last Updated**: 2025-10-03
**Maintained By**: VisionFlow XR Team
**Feedback**: Open an issue on GitHub or join our Discord
