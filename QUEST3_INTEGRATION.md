# Quest 3 Direct AR Integration

## Overview

This document describes the complete integration flow for the Quest 3 Direct AR mode with GPU-accelerated visual analytics.

## Architecture Components

### 1. GPU Visual Analytics Pipeline

#### Kernels (PTX Files)
- **visual_analytics_core.ptx**: Main temporal-spatial force web computation
- **advanced_gpu_algorithms.ptx**: UMAP, spectral clustering, GNN algorithms

#### Data Flow
```
TSNode (GPU) -> BinaryNodeData (Wire) -> Quest 3 AR Rendering
```

### 2. Server-Side Components

#### GPU Compute Actor
- Manages visual analytics GPU context
- Converts between TSNode and BinaryNodeData formats
- Executes kernels based on graph complexity:
  - < 1000 nodes: Legacy kernel
  - 1000-10000 nodes: Advanced kernel
  - > 10000 nodes or isolation layers: Visual Analytics kernel

#### REST API Endpoints
- `/api/quest3/defaults`: Quest 3 optimized settings
- `/api/quest3/calibrate`: User-specific calibration
- `/api/analytics/params`: Visual analytics parameters
- `/api/analytics/constraints`: Graph constraints
- `/api/analytics/focus`: Focus node/region
- `/api/analytics/stats`: Performance metrics

### 3. Binary WebSocket Protocol

#### Message Format (28 bytes per node)
```
[0-3]   Node ID (u32, with high bit for agent flag)
[4-15]  Position (3x f32: x, y, z)
[16-27] Velocity (3x f32: vx, vy, vz)
```

#### Protocol Rules
- WebSocket ONLY for bidirectional position/velocity streaming
- All control through REST API
- No WebSocket control frames
- Binary encoding via `binary_protocol::encode_node_data()`

### 4. Client-Side Quest 3 Components

#### Quest3DirectAR Component
- Auto-detects Quest 3 browser
- Immediately requests AR session
- Loads server defaults via REST
- Connects WebSocket for binary streaming
- Renders using instanced meshes with LOD

#### Detection Logic
```typescript
const isQuest3 = userAgent.includes('quest 3') || 
                 userAgent.includes('meta quest 3') ||
                 (userAgent.includes('oculus') && userAgent.includes('quest'));
```

#### Force Direct AR Mode
- URL parameter: `?force=quest3` or `?directar=true`
- Bypasses browser detection

## Data Flow Sequence

### 1. Quest 3 Connection
```
Quest 3 Browser -> Detect -> Load /api/quest3/defaults -> Enter AR
```

### 2. Visual Analytics Processing
```
Graph Data -> GPU Visual Analytics -> TSNode Processing -> 
Binary Conversion -> WebSocket Stream -> Quest 3 Render
```

### 3. Update Cycle (60 FPS)
```
1. GPU computes forces with visual analytics kernel
2. Results converted to BinaryNodeData format
3. GraphActor broadcasts via ClientManager
4. WebSocket sends binary data to Quest 3
5. Quest 3 renders with LOD optimization
```

## Performance Optimizations

### Server-Side
- GPU kernel selection based on graph complexity
- Visual analytics for complex graphs (>10k nodes)
- Isolation layers for focus+context
- Streaming optimization with LOD hints

### Quest 3 Client
- Immediate AR mode entry
- No UI chrome
- Instanced mesh rendering
- Distance-based LOD
- Binary protocol (no JSON parsing)
- 90Hz refresh rate optimized

## Configuration

### Quest 3 Default Settings
```rust
Quest3Settings {
    xr: {
        display_mode: "immersive-ar",
        space_type: "local-floor",
        enable_hand_tracking: true,
        refresh_rate: 90,
    },
    visualisation: {
        rendering_context: "quest3-ar",
        lod_enabled: true,
        particle_count: 1000,
        culling_distance: 50.0,
    },
    performance: {
        target_framerate: 90,
        adaptive_quality: true,
        gpu_priority: "balanced",
    }
}
```

## Testing

### Local Testing (Without Quest 3)
```bash
# Force Quest 3 mode in browser
http://localhost:3000?force=quest3

# Test REST endpoints
curl http://localhost:8080/api/quest3/defaults
curl http://localhost:8080/api/analytics/params
```

### Quest 3 Device Testing
1. Ensure server is accessible from Quest 3 network
2. Navigate to server URL in Quest 3 browser
3. Should immediately prompt for AR permissions
4. Enters AR mode with server defaults
5. Graph renders with GPU-accelerated physics

## Troubleshooting

### Quest 3 Not Detected
- Check user agent string
- Use `?force=quest3` parameter
- Verify WebXR API availability

### No AR Mode
- Check WebXR permissions
- Verify HTTPS (required for WebXR)
- Check browser console for errors

### Performance Issues
- Check `/api/analytics/stats` for GPU metrics
- Verify LOD is enabled
- Check network latency
- Reduce particle count if needed

## Future Enhancements

1. **Hand Gesture Controls**: Use Quest 3 hand tracking for graph manipulation
2. **Spatial Anchors**: Pin graph regions in physical space
3. **Multi-User AR**: Shared graph exploration
4. **Voice Commands**: Natural language graph queries
5. **Haptic Feedback**: Touch feedback for node selection