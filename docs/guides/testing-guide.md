# VisionFlow Testing Guide

**Last Updated**: 2025-10-03
**Purpose**: Comprehensive manual testing guide for VisionFlow control panel functionality and API endpoints
**Testing Approach**: Manual testing only (automated tests removed for security)

## Overview

This guide provides manual testing procedures for the VisionFlow visualization settings control panel and associated API endpoints. The control panel enables real-time adjustment of physics parameters, visual effects, and debug settings.

### Testing Strategy

**⚠️ Important**: VisionFlow uses **manual testing only**. Automated testing infrastructure was removed in October 2025 due to supply chain security concerns (see [ADR 003](../decisions/003-code-pruning-2025-10.md)).

**Testing Approach**:
- Manual functional testing via UI
- API endpoint testing via curl/Postman
- Visual verification of graph behavior
- Performance monitoring via browser DevTools

For details on why automated tests were removed, see [security-alert.md](../archive/legacy-docs-2025-10/troubleshooting/security-alert.md).

## Control Panel Settings Structure

### 1. Physics Controls

Physics settings control the graph visualization simulation:

```javascript
// Physics settings paths and ranges
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

## API Testing Endpoints

### Single Setting Operations

#### Get Single Setting
```http
GET /api/settings/path?path=<setting-path>

Example Response:
{
  "path": "visualisation.glow.nodeGlowStrength",
  "value": 0.5,
  "success": true
}
```

#### Update Single Setting
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

### Batch Operations

#### Batch Get Settings
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

#### Batch Update Settings
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

### Scenario 1: Physics Parameter Tuning

Test adjusting physics parameters and observe effects on graph layout:

1. **Increase Damping** (0.95 → 0.99)
   - **Expected Result**: Graph movement becomes more viscous, settles faster
   - **Test Command**: `curl -X PUT "http://localhost:5173/api/settings/path" -H "Content-Type: application/json" -d '{"path":"visualisation.graphs.logseq.physics.damping","value":0.99}'`

2. **Add Gravity** (0.0 → 0.5)
   - **Expected Result**: Nodes drift downward
   - **Test Command**: Update gravity setting and observe node movement

3. **Increase Spring Strength** (0.01 → 0.1)
   - **Expected Result**: Connected nodes pull together more strongly
   - **Validation**: Measure distance between connected nodes

4. **Adjust Repulsion** (100 → 500)
   - **Expected Result**: Nodes push apart more, graph expands
   - **Validation**: Measure overall graph bounding box

### Scenario 2: Visual Effects Testing

Test visual enhancement settings:

1. **Toggle Bloom Effect**
   - Enable bloom and adjust intensity (0.5 → 1.5)
   - **Expected Result**: Glowing halo effect around bright elements
   - **Validation**: Visual inspection of node rendering

2. **Adjust Node Glow**
   - Increase nodeGlowStrength from 0.5 to 0.9
   - **Expected Result**: Nodes become more luminous
   - **Test**: Compare before/after screenshots

3. **Change Color Scheme**
   - Update baseColor and highlightColor
   - **Expected Result**: Immediate colour changes in visualisation
   - **Validation**: Color picker validation

### Scenario 3: Performance Testing

Test debug and performance settings:

1. **Enable Debug Mode**
   - Set system.debug.enabled to true
   - **Expected Result**: Additional debug information in console
   - **Validation**: Check browser console for debug output

2. **Enable Performance Metrics**
   - Set system.debug.enablePerformanceDebug to true
   - **Expected Result**: FPS counter and performance stats visible
   - **Validation**: Verify performance overlay appears

## cURL Testing Commands

### Physics Parameter Testing
```bash
# Get current damping value
curl -X GET "http://localhost:5173/api/settings/path?path=visualisation.graphs.logseq.physics.damping"

# Update damping value
curl -X PUT "http://localhost:5173/api/settings/path" \
  -H "Content-Type: application/json" \
  -d '{"path":"visualisation.graphs.logseq.physics.damping","value":0.98}'

# Test gravity setting
curl -X PUT "http://localhost:5173/api/settings/path" \
  -H "Content-Type: application/json" \
  -d '{"path":"visualisation.graphs.logseq.physics.gravity","value":0.3}'
```

### Visual Effects Batch Update
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

### Debug Settings Test
```bash
curl -X PUT "http://localhost:5173/api/settings/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {"path": "system.debug.enabled", "value": true},
      {"path": "system.debug.enablePerformanceDebug", "value": true},
      {"path": "system.debug.consoleLogging", "value": true}
    ]
  }'
```

## Expected UI Components

The control panel should include:

### 1. Physics Section
- Sliders for damping, gravity, spring settings
- Reset button for default values
- Real-time value display
- Range validation

### 2. Visual Effects Section
- Toggle switches for bloom, glow effects
- Colour pickers for node/edge colours
- Intensity sliders with live preview
- Effect preview area

### 3. Debug Section
- Checkbox toggles for debug options
- Console output viewer
- Performance metrics display
- Export debug data button

### 4. Presets Management
- Save current settings as preset
- Load preset configurations
- Reset to system defaults
- Import/export settings JSON

## Known Working Endpoints

✅ **Verified Operational**:
- `POST /api/settings/batch` - Batch read settings
- `PUT /api/settings/batch` - Batch update settings
- `GET /api/settings/path` - Get single setting
- `PUT /api/settings/path` - Update single setting
- `GET /api/graph/data` - Get graph data
- `POST /api/bots/spawn-agent-hybrid` - Spawn hybrid agents

## Testing Notes

- All settings changes are debounced on the client side (50ms)
- Critical updates (physics parameters) are processed immediately
- Settings are persisted to localStorage and server simultaneously
- WebSocket updates notify all connected clients of changes
- The backend has resolved duplicate route definitions

## Validation Checklist

### API Response Validation
- [ ] HTTP status codes are correct (200 for success, 400 for validation errors)
- [ ] Response format matches expected JSON structure
- [ ] Field names use camelCase in responses
- [ ] Error messages are descriptive and actionable
- [ ] Timestamps are included in responses

### Visual Validation
- [ ] Physics changes affect graph layout immediately
- [ ] Color changes are reflected in real-time
- [ ] Bloom effects render correctly
- [ ] Debug information displays properly
- [ ] Performance metrics update continuously

### Performance Validation
- [ ] Settings updates complete within latency targets (<50ms)
- [ ] Batch operations are more efficient than individual requests
- [ ] WebSocket notifications are sent to all connected clients
- [ ] No memory leaks during extended testing
- [ ] CPU usage remains stable during parameter adjustments

## Troubleshooting Guide

### Common Issues

1. **404 Errors on API Calls**
   - **Check**: Vite proxy configuration in `vite.config.ts`
   - **Solution**: Ensure proxy is always enabled, not conditionally

2. **Settings Not Persisting**
   - **Check**: AutoSaveManager debouncing settings
   - **Solution**: Wait for batch save to complete (500ms delay)

3. **Visual Effects Not Appearing**
   - **Check**: WebGL support in browser
   - **Solution**: Test in browser with hardware acceleration enabled

4. **Physics Parameters Not Responding**
   - **Check**: Graph simulation is running
   - **Solution**: Verify physics actor is active and receiving updates

### Debug Steps

1. **Enable Debug Mode**: Set `system.debug.enabled = true`
2. **Check Console Logs**: Look for API request/response details
3. **Verify WebSocket Connection**: Check `/wss` endpoint status
4. **Test Individual Endpoints**: Use cURL commands to isolate issues
5. **Monitor Network Traffic**: Use browser DevTools Network tab

---

**Testing Status**: All major endpoints verified operational. Control panel functionality requires runtime validation in browser with WebGL support.