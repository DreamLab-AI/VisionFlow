---
title: XR/VR Setup Guide
description: Setting up extended reality features in VisionFlow with Vircadia integration, WebXR fallback, Quest headset detection, and HRTF spatial audio via LiveKit.
category: how-to
tags:
  - xr
  - vircadia
  - webxr
  - livekit
  - spatial-audio
updated-date: 2026-02-12
difficulty-level: intermediate
---

# XR/VR Setup Guide

## Overview

VisionFlow supports immersive spatial presence through Vircadia XR integration.
Users can explore the knowledge graph in virtual reality, with automatic fallback
to browser-based WebXR when a native Vircadia client is unavailable. This guide
covers headset detection, display tuning for Quest devices, and HRTF spatial
audio powered by LiveKit.

## Prerequisites

- VisionFlow server running (see deployment guide)
- Vircadia World Server accessible on `ws://<host>:3020/world/ws`
- A WebXR-capable browser (Chrome 113+, Edge, or Quest Browser)
- Optional: Meta Quest 2 / Quest 3 headset

## Vircadia Integration

Vircadia provides the multi-user world server that synchronises entity state
across all connected clients. VisionFlow registers itself as a world script:

```bash
# Start VisionFlow with XR profile
docker compose -f docker-compose.yml \
  -f docker-compose.vircadia.yml --profile xr up -d
```

Once connected, the 3D graph scene is projected into Vircadia world-space.
Nodes become selectable entities; edges render as spatial links between them.

### Connection Settings

| Setting               | Default                                    | Notes                          |
|-----------------------|--------------------------------------------|--------------------------------|
| Vircadia Server URL   | `ws://vircadia-world-server:3020/world/ws` | Docker-internal address        |
| Auto-Connect          | `true`                                     | Reconnects on page load        |
| Entity Sync Interval  | `100 ms`                                   | Lower values increase traffic  |

## WebXR Fallback

When no Vircadia native client is detected, VisionFlow falls back to the
WebXR Device API. The client checks `navigator.xr` on load:

```typescript
const xrSupported = await navigator.xr?.isSessionSupported('immersive-vr');
if (xrSupported) {
  session = await navigator.xr.requestSession('immersive-vr', {
    requiredFeatures: ['local-floor'],
    optionalFeatures: ['hand-tracking'],
  });
}
```

If neither immersive-vr nor Vircadia is available, the graph renders in
standard desktop 3D (Babylon.js) with mouse and keyboard controls.

## Quest Headset Detection and DPR Capping

Meta Quest devices report a high `devicePixelRatio` that can overwhelm the
GPU. VisionFlow detects Quest user-agents and caps DPR at **1.0** to maintain
a stable frame rate:

```typescript
function getEffectiveDpr(): number {
  const isQuest = /Quest/i.test(navigator.userAgent);
  if (isQuest) {
    return Math.min(window.devicePixelRatio, 1.0);
  }
  return window.devicePixelRatio;
}
```

This prevents the render target from exceeding the panel resolution and keeps
the physics tick budget under 11 ms on Quest 3 hardware.

## HRTF Spatial Audio via LiveKit

VisionFlow uses LiveKit for real-time voice with head-related transfer function
(HRTF) spatialization. Each participant's audio source is positioned at their
avatar location in the 3D graph space:

1. **Room join** -- the client connects to a LiveKit room using a signed JWT.
2. **Track subscription** -- remote audio tracks are routed through an
   `AudioContext` with a `PannerNode` configured for HRTF.
3. **Position update** -- on every frame, each remote panner is moved to the
   corresponding avatar's world-space coordinates.

```typescript
const panner = audioCtx.createPanner();
panner.panningModel = 'HRTF';
panner.distanceModel = 'inverse';
panner.refDistance = 1;
panner.maxDistance = 50;
panner.rolloffFactor = 1.5;
```

Audio attenuates naturally with distance, enabling proximity-based
conversations inside the graph.

## Troubleshooting

| Symptom                        | Likely Cause                     | Fix                                      |
|--------------------------------|----------------------------------|------------------------------------------|
| Black screen in headset        | DPR too high                     | Verify DPR cap is active (check console) |
| No spatial audio               | Microphone permission denied     | Grant mic access; click page to start AudioContext |
| WebXR session fails            | Browser lacks WebXR support      | Use Quest Browser or Chrome 113+         |
| Vircadia entities not visible  | World server unreachable         | Check `docker logs vircadia-world-server` |

## See Also

- [Vircadia Multi-User Guide](vircadia-multi-user-guide.md) -- collaborative editing in XR
- [Complete Vircadia XR Integration Guide](vircadia-xr-complete-guide.md) -- architecture deep-dive
- [Voice Integration](voice-integration.md) -- LiveKit voice configuration
- [VR Development](vr-development.md) -- developer workflow for XR features
