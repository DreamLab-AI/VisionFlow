---
title: XR Setup Guide - Development Environment
description: Documentation for xr-setup
category: tutorial
tags:
  - architecture
  - design
  - api
  - api
  - http
related-docs:
  - archive/docs/guides/xr-setup.md
  - archive/docs/guides/user/working-with-agents.md
  - README.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Node.js runtime
---

---
title: XR Setup Guide
description: Setting up extended reality (XR) environments for immersive development with Turbo Flow Claude
category: user-guide
tags: [xr, vr, ar, setup, immersive, spatial-computing]
difficulty: intermediate
last-updated: 2025-10-27
related:
  - ./working-with-agents.md
  - ../index.md
  - ../../reference/readme.md
---

# XR Setup Guide - Development Environment

> **📋 Role**: Step-by-step guide for setting up XR development environment with Turbo Flow Claude.
> For technical WebXR implementation details in VisionFlow, see [WebXR Setup - VisionFlow Implementation](../xr-setup.md).

> **⚠️ Work in Progress**: This guide is currently under development. Content will be expanded in future updates.

## Overview

This guide covers setting up Extended Reality (XR) environments for immersive development with Turbo Flow Claude. Learn how to configure VR, AR, and spatial computing interfaces for enhanced agent collaboration and code visualization.

## What is XR Development?

### Extended Reality (XR) Overview

**XR encompasses:**
- **VR (Virtual Reality)**: Fully immersive digital environments
- **AR (Augmented Reality)**: Digital overlays on physical world
- **MR (Mixed Reality)**: Blend of physical and digital worlds
- **Spatial Computing**: 3D interaction with digital content

**Benefits for development:**
- Immersive code visualization
- 3D architecture exploration
- Spatial agent coordination
- Enhanced collaboration
- Multi-dimensional debugging

## System Requirements

### Hardware Requirements

**Minimum specifications:**
- **Headset**: Meta Quest 2, Quest 3, Apple Vision Pro, or PCVR headset
- **PC** (for PCVR):
  - CPU: Intel i5-9400F / AMD Ryzen 5 3600
  - GPU: NVIDIA GTX 1060 / AMD RX 580
  - RAM: 16GB
  - USB 3.0 port
- **Network**: 5GHz WiFi or Ethernet (for wireless streaming)
- **Space**: 2m x 2m clear area minimum

**Recommended specifications:**
- **Headset**: Meta Quest 3, Apple Vision Pro, Valve Index
- **PC**:
  - CPU: Intel i7-12700K / AMD Ryzen 7 5800X
  - GPU: NVIDIA RTX 3070 / AMD RX 6800
  - RAM: 32GB
  - WiFi 6E router
- **Space**: 3m x 3m clear area

### Software Requirements

**Operating Systems:**
- Windows 10/11 (for PCVR)
- macOS 12+ (for Vision Pro)
- Linux (experimental, limited support)

**Required software:**
- SteamVR (for PCVR headsets)
- Oculus PC app (for Meta headsets)
- Virtual Desktop or Air Link (for wireless)
- Web browser with WebXR support
- Node.js 18+ (for development server)

## Supported XR Platforms

### Meta Quest (2/3/Pro)

**Standalone mode:**
- No PC required
- Built-in processing
- Wireless freedom
- Lower performance

**PCVR mode (Quest Link):**
- PC-powered graphics
- Higher performance
- Wired or wireless
- Best for development

**Setup Quest Link:**
```bash
# Install Oculus PC app
# Download from: https://www.oculus.com/setup/

# Enable developer mode on headset
# Connect via USB-C or Air Link

# Test connection
adb devices
```

### Apple Vision Pro

**visionOS development:**
- Spatial computing native
- High-resolution displays
- Hand tracking
- Eye tracking
- Requires macOS for development

**Setup Vision Pro:**
```bash
# Install Xcode 15+
xcode-select --install

# Install visionOS SDK
# Available through Xcode

# Enable developer mode
# Settings > Privacy & Security > Developer Mode
```

### PCVR Headsets (Valve Index, HTC Vive, etc.)

**SteamVR setup:**
```bash
# Install SteamVR from Steam
# Launch SteamVR
# Complete room setup

# Test tracking
# SteamVR > Room Setup > Quick Calibration
```

## Installing XR Development Environment

### 1. Install Turbo Flow Claude XR Module

```bash
# Navigate to project directory
cd /home/devuser/workspace/project

# Install XR dependencies
npm install --save \
  three \
  @react-three/fiber \
  @react-three/xr \
  @react-three/drei

# Install development tools
npm install --save-dev \
  @types/three
```

### 2. Configure XR Server

**Create XR configuration file:**

```bash
# Create config directory if not exists
mkdir -p config

# Create XR config
cat > config/xr.config.js << 'EOF'
module.exports = {
  server: {
    port: 8081,
    host: '0.0.0.0',
    https: true, // Required for WebXR
  },
  xr: {
    mode: 'immersive-vr', // or 'immersive-ar'
    features: ['local-floor', 'hand-tracking'],
    framerate: 90,
  },
  rendering: {
    antialias: true,
    shadows: true,
    foveatedRendering: true,
  }
};
EOF
```

### 3. Generate SSL Certificates (Required for WebXR)

```bash
# WebXR requires HTTPS
# Generate self-signed certificate for development

openssl req -x509 -newkey rsa:4096 \
  -keyout config/key.pem \
  -out config/cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"

# Trust certificate (development only)
# Chrome: chrome://flags/#allow-insecure-localhost
# Firefox: Accept certificate warning
```

### 4. Start XR Development Server

```bash
# Start with XR support
npm run dev:xr

# Or manually
node scripts/xr-server.js
```

**Access XR environment:**
- PC browser: `https://localhost:8081`
- Headset browser: `https://[PC-IP]:8081`

## XR Interface Configuration

### Spatial Code Visualization

**Enable 3D code view:**

```javascript
// In headset browser or spatial app
const codeSpace = {
  layout: 'hierarchical', // file tree in 3D
  spacing: 2.0, // meters between nodes
  height: 1.5, // eye level
  curved: true, // curved display panels
};

// Launch spatial code viewer
TurboFlowXR.initCodeSpace(codeSpace);
```

**Features:**
- 3D file tree navigation
- Floating code panels
- Gesture-based editing
- Voice commands
- Agent avatars

### Agent Visualization

**Visualize agents in 3D space:**

```javascript
// Spawn agents with spatial representation
Task("Backend Developer", "Build API", "backend-dev", {
  xr: {
    position: [2, 1.5, -3], // x, y, z in meters
    avatar: 'robot-blue',
    workspace: 'panel-1'
  }
});

Task("Frontend Developer", "Build UI", "coder", {
  xr: {
    position: [-2, 1.5, -3],
    avatar: 'robot-green',
    workspace: 'panel-2'
  }
});
```

**Agent features:**
- Visual avatars
- Status indicators
- Progress animations
- Communication lines
- Shared workspaces

### Spatial Layouts

**Common XR layouts:**

**Theater mode:**
```javascript
// Large central display, agents on sides
TurboFlowXR.setLayout('theater', {
  mainDisplay: { width: 5, height: 3, distance: 4 },
  agentPanels: { count: 4, arc: 180 }
});
```

**Workspace mode:**
```javascript
// Surrounding panels for multi-tasking
TurboFlowXR.setLayout('workspace', {
  panels: 6,
  radius: 3,
  height: 1.5,
  curve: 120 // degrees
});
```

**Collaboration mode:**
```javascript
// Shared central space with agent positions
TurboFlowXR.setLayout('collaboration', {
  centerTable: { size: 2 },
  seats: 4,
  sharedDisplays: 2
});
```

## Interaction Methods

### Hand Tracking

**Enable hand tracking:**
```javascript
TurboFlowXR.enableHandTracking({
  gestures: {
    pinch: 'select',
    grab: 'move',
    point: 'navigate',
    swipe: 'scroll'
  }
});
```

**Common gestures:**
- **Pinch**: Select code/files
- **Grab**: Move panels
- **Point**: Navigate file tree
- **Swipe**: Scroll code
- **Air tap**: Execute commands

### Voice Commands

**Enable voice control:**
```javascript
TurboFlowXR.enableVoiceCommands({
  language: 'en-US',
  keywords: [
    'spawn agent',
    'show file',
    'run tests',
    'commit changes',
    'agent status'
  ]
});
```

**Example commands:**
- "Spawn coder agent for authentication"
- "Show file src/auth/login.ts"
- "Run unit tests"
- "Show agent status"
- "Commit changes with message..."

### Controller Input

**Map controller buttons:**
```javascript
TurboFlowXR.configureControllers({
  left: {
    trigger: 'select',
    grip: 'grab',
    thumbstick: 'navigate',
    buttonA: 'menu',
    buttonB: 'back'
  },
  right: {
    trigger: 'execute',
    grip: 'grab',
    thumbstick: 'scroll',
    buttonA: 'confirm',
    buttonB: 'cancel'
  }
});
```

## XR Development Workflows

### Immersive Code Review

**Review code in VR:**

1. Enter XR environment
2. Navigate to file tree
3. Open files as floating panels
4. Spawn reviewer agent
5. Watch real-time annotations
6. Approve/request changes via gesture

```javascript
// Spawn code review swarm in XR
Task("Code Review Swarm",
  "Review PR #123 with spatial annotations",
  "code-review-swarm",
  { xr: { enableAnnotations: true } }
);
```

### Spatial Architecture Design

**Design system architecture in 3D:**

1. Spawn system architect agent
2. View architecture as 3D graph
3. Rearrange components spatially
4. Connect dependencies with lines
5. Annotate with voice/gestures
6. Export to diagrams

```javascript
Task("System Architect",
  "Design microservices architecture in 3D space",
  "system-architect",
  { xr: { mode: '3d-graph', interactive: true } }
);
```

### Multi-Agent Collaboration

**Visualize agent swarm:**

1. Initialize swarm in XR mode
2. Watch agents spawn with avatars
3. See communication as light flows
4. Monitor progress on spatial displays
5. Intervene with gestures/voice

```javascript
// Initialize XR-enabled swarm
TurboFlowXR.initSwarm({
  topology: 'mesh',
  visualization: 'network-graph',
  agents: [
    { type: 'researcher', position: [0, 2, -3] },
    { type: 'coder', position: [-2, 1.5, -3] },
    { type: 'tester', position: [2, 1.5, -3] }
  ]
});
```

## Performance Optimization

### Rendering Performance

**Optimize for headset:**

```javascript
TurboFlowXR.configurePerformance({
  // Reduce rendering load
  foveatedRendering: true, // Render center sharply
  dynamicResolution: true, // Adjust resolution dynamically

  // Limit updates
  codeUpdateRate: 30, // Hz
  agentUpdateRate: 15, // Hz

  // LOD (Level of Detail)
  filePanelLOD: {
    near: 'high',
    medium: 'medium',
    far: 'low'
  }
});
```

### Network Optimization

**For wireless streaming:**

```javascript
TurboFlowXR.configureNetwork({
  // Compress data
  compression: 'h265',
  bitrate: 50, // Mbps

  // Reduce latency
  predictiveTracking: true,
  lowLatencyMode: true,

  // Bandwidth management
  adaptiveBitrate: true
});
```

## Troubleshooting

### Common XR Issues

**Headset not detected:**
```bash
# Check USB connection
lsusb | grep -i oculus

# Restart Oculus service
sudo systemctl restart oculusd

# Check SteamVR status
# SteamVR > Settings > Developer > Restart Headset
```

**WebXR not working:**
```bash
# Verify HTTPS is enabled
curl -k https://localhost:8081

# Check browser support
# chrome://flags/#webxr
# Enable all WebXR flags

# Accept self-signed certificate
# Navigate to https://localhost:8081 and accept warning
```

**Poor performance:**
```javascript
// Reduce rendering quality
TurboFlowXR.configurePerformance({
  shadows: false,
  antialias: false,
  maxAgentAvatars: 4,
  panelResolution: 'medium'
});
```

**Tracking issues:**
```bash
# Recalibrate room
# SteamVR > Room Setup > Run Room Setup

# Quest: Settings > Guardian > Set Up Guardian

# Clean sensors/cameras on headset
```

## Safety & Best Practices

### Physical Safety
- Clear play area of obstacles
- Use guardian/boundary system
- Take breaks every 30 minutes
- Stay hydrated
- Stop if feeling discomfort

### Development Safety
- Use version control before XR sessions
- Save work frequently
- Test gestures in safe environment
- Start with simple interactions
- Gradually increase complexity

### Ergonomics
- Adjust headset for comfort
- Proper IPD (interpupillary distance)
- Eye-level content placement
- Avoid neck strain (content height)
- Use seated mode for long sessions

## Advanced Features

### Multi-User XR Collaboration

**Connect multiple developers:**

```javascript
// Enable collaborative XR
TurboFlowXR.enableCollaboration({
  mode: 'multi-user',
  maxUsers: 4,
  voiceChat: true,
  sharedWorkspace: true,
  userAvatars: true
});

// Join session
TurboFlowXR.joinSession('session-id-123');
```

### Custom XR Components

**Build custom XR interfaces:**

```javascript
// Create custom code viewer
const CustomCodeViewer = () => {
  return (
    <XR>
      <CodePanel
        position={[0, 1.5, -2]}
        file="src/app.ts"
        syntax="typescript"
        curved={true}
      />
      <AgentAvatar
        position={[-1, 1, -3]}
        agent="coder"
        animated={true}
      />
    </XR>
  );
};
```

### AI-Powered XR Assistance

**Voice-activated AI helper:**

```javascript
// Enable XR AI assistant
TurboFlowXR.enableAIAssistant({
  voice: true,
  spatialAudio: true,
  avatar: 'floating-orb',
  capabilities: [
    'code-explanation',
    'navigation-help',
    'agent-summoning',
    'workspace-management'
  ]
});
```

## Future XR Features

**Coming soon:**
- Haptic feedback for code errors
- Eye-tracking for natural navigation
- Neural interface integration
- AR code overlay on physical screens
- Holographic agent projections

## Next Steps

After setting up XR:
1. Practice basic gestures and navigation
2. Try [Working with Agents in XR](./working-with-agents.md)
3. Review development workflow documentation
4. Join XR development community

## Related Resources

- [Working with Agents](./working-with-agents.md)
- [Documentation Home](../../../../README.md)
- [XR Setup Guide](../xr-setup.md)
- WebXR Spec: https://www.w3.org/TR/webxr/
- Three.js XR: https://threejs.org/docs/#manual/en/introduction/How-to-create-VR-content

---

---

## Related Documentation

- [Link Analysis Summary](../../../reports/consolidation/link-analysis-summary-2025-12.md)
- [Upstream Turbo-Flow-Claude Analysis](../../../../multi-agent-docker/upstream-analysis.md)

## Support

- GitHub Issues: https://github.com/ruvnet/claude-flow/issues
- XR Community: https://github.com/ruvnet/claude-flow/discussions
- WebXR Discord: https://discord.gg/webxr
- Meta Quest Support: https://www.oculus.com/support/
