# VisionFlow Control Panel Testing Guide

## Overview
This guide documents the control panel functionality and API endpoints for testing the VisionFlow visualization settings.

## Control Panel Settings Structure

### 1. Physics Controls
These settings control the physics simulation for graph visualization:

```javascript
// Physics settings paths
visualisation.graphs.logseq.physics.damping       // Range: 0.0 - 1.0 (default: 0.95)
visualisation.graphs.logseq.physics.gravity       // Range: -1.0 - 1.0 (default: 0.0)
visualisation.graphs.logseq.physics.springStrength // Range: 0.0 - 1.0 (default: 0.01)
visualisation.graphs.logseq.physics.springLength  // Range: 0 - 500 (default: 100)
visualisation.graphs.logseq.physics.repulsion     // Range: 0 - 1000 (default: 100)
visualisation.graphs.logseq.physics.centralForce  // Range: 0.0 - 1.0 (default: 0.001)
```

### 2. Visual Effects Controls
Settings for visual appearance and post-processing effects:

```javascript
// Glow settings
visualisation.glow.nodeGlowStrength    // Range: 0.0 - 1.0
visualisation.glow.edgeGlowStrength    // Range: 0.0 - 1.0
visualisation.glow.baseColor           // Color: #RRGGBB format

// Bloom effect
visualisation.bloom.enabled            // Boolean: true/false
visualisation.bloom.intensity          // Range: 0.0 - 2.0
visualisation.bloom.threshold          // Range: 0.0 - 1.0
visualisation.bloom.radius            // Range: 0.0 - 1.0

// Node appearance
visualisation.nodes.baseColor          // Color: #RRGGBB
visualisation.nodes.highlightColor     // Color: #RRGGBB
visualisation.nodes.defaultSize        // Range: 1 - 20

// Edge appearance
visualisation.edges.defaultColor       // Color: #RRGGBB
visualisation.edges.highlightColor     // Color: #RRGGBB
visualisation.edges.thickness          // Range: 0.1 - 5.0
```

### 3. Debug Controls
Developer and debugging settings:

```javascript
system.debug.enabled                   // Boolean: Enable debug mode
system.debug.enableDataDebug          // Boolean: Log data operations
system.debug.enablePerformanceDebug   // Boolean: Show performance metrics
system.debug.consoleLogging           // Boolean: Enable console output
```

## API Endpoints for Control Panel

### Get Single Setting
```http
GET /api/settings/path?path=<setting-path>

Response:
{
  "path": "visualisation.glow.nodeGlowStrength",
  "value": 0.5,
  "success": true
}
```

### Update Single Setting
```http
PUT /api/settings/path
Content-Type: application/json

{
  "path": "visualisation.glow.nodeGlowStrength",
  "value": 0.7
}

Response:
{
  "success": true,
  "path": "visualisation.glow.nodeGlowStrength",
  "value": 0.7
}
```

### Batch Get Settings
```http
POST /api/settings/batch
Content-Type: application/json

{
  "paths": [
    "visualisation.glow.nodeGlowStrength",
    "visualisation.glow.edgeGlowStrength",
    "visualisation.bloom.enabled"
  ]
}

Response:
{
  "success": true,
  "values": [
    {"path": "visualisation.glow.nodeGlowStrength", "value": 0.5},
    {"path": "visualisation.glow.edgeGlowStrength", "value": 0.3},
    {"path": "visualisation.bloom.enabled", "value": true}
  ]
}
```

### Batch Update Settings
```http
PUT /api/settings/batch
Content-Type: application/json

{
  "updates": [
    {"path": "visualisation.glow.nodeGlowStrength", "value": 0.7},
    {"path": "visualisation.glow.edgeGlowStrength", "value": 0.5},
    {"path": "visualisation.bloom.intensity", "value": 1.2}
  ]
}

Response:
{
  "success": true,
  "results": [
    {"path": "visualisation.glow.nodeGlowStrength", "success": true},
    {"path": "visualisation.glow.edgeGlowStrength", "success": true},
    {"path": "visualisation.bloom.intensity", "success": true}
  ]
}
```

## Testing Scenarios

### Scenario 1: Physics Tuning
Test adjusting physics parameters to see how they affect graph layout:

1. **Increase Damping** (0.95 → 0.99)
   - Expected: Graph movement becomes more viscous, settles faster

2. **Add Gravity** (0.0 → 0.5)
   - Expected: Nodes drift downward

3. **Increase Spring Strength** (0.01 → 0.1)
   - Expected: Connected nodes pull together more strongly

4. **Adjust Repulsion** (100 → 500)
   - Expected: Nodes push apart more, graph expands

### Scenario 2: Visual Effects
Test visual enhancement settings:

1. **Toggle Bloom Effect**
   - Enable bloom and adjust intensity
   - Expected: Glowing halo effect around bright elements

2. **Adjust Node Glow**
   - Increase nodeGlowStrength from 0.5 to 0.9
   - Expected: Nodes become more luminous

3. **Change Color Scheme**
   - Update baseColor and highlightColor
   - Expected: Immediate color changes in visualization

### Scenario 3: Performance Testing
Test debug and performance settings:

1. **Enable Debug Mode**
   - Set system.debug.enabled to true
   - Expected: Additional debug information in console

2. **Enable Performance Metrics**
   - Set system.debug.enablePerformanceDebug to true
   - Expected: FPS counter and performance stats visible

## Control Panel UI Elements

### Expected UI Components

1. **Physics Section**
   - Sliders for damping, gravity, spring settings
   - Reset button for default values

2. **Visual Effects Section**
   - Toggle switches for bloom, glow effects
   - Color pickers for node/edge colors
   - Intensity sliders

3. **Debug Section**
   - Checkbox toggles for debug options
   - Console output viewer

4. **Presets**
   - Save current settings
   - Load preset configurations
   - Reset to defaults

## Testing with cURL

### Example: Adjust Physics Damping
```bash
# Get current value
curl -X GET "http://localhost:5173/api/settings/path?path=visualisation.graphs.logseq.physics.damping"

# Update value
curl -X PUT "http://localhost:5173/api/settings/path" \
  -H "Content-Type: application/json" \
  -d '{"path":"visualisation.graphs.logseq.physics.damping","value":0.98}'
```

### Example: Batch Update Visual Settings
```bash
curl -X PUT "http://localhost:5173/api/settings/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {"path": "visualisation.glow.nodeGlowStrength", "value": 0.8},
      {"path": "visualisation.bloom.enabled", "value": true},
      {"path": "visualisation.bloom.intensity", "value": 1.5}
    ]
  }'
```

## Known Working Endpoints

- ✅ `POST /api/settings/batch` - Batch read settings
- ✅ `PUT /api/settings/batch` - Batch update settings
- ✅ `GET /api/settings/path` - Get single setting
- ✅ `PUT /api/settings/path` - Update single setting
- ✅ `GET /api/graph/data` - Get graph data
- ✅ `POST /api/bots/spawn-agent-hybrid` - Spawn hybrid agents

## Notes

- All settings changes are debounced on the client side (50ms)
- Critical updates (physics parameters) are processed immediately
- Settings are persisted to localStorage and server
- WebSocket updates notify all connected clients of changes
- The backend has duplicate route definitions that need to be resolved