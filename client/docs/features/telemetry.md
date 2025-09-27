# Agent Telemetry System

A comprehensive client-side telemetry and debugging system for the hive mind visualization.

## Features

- 🤖 **Agent Spawn Tracking**: Monitor agent creation, state changes, and lifecycle
- 📡 **WebSocket Message Flow**: Log all incoming/outgoing messages with size and metadata
- 🎮 **Three.js Debugging**: Track mesh positions, monitor rendering performance
- ⚡ **Performance Monitoring**: Frame rate analysis, render time tracking, memory usage
- 🔍 **Real-time Debug Overlay**: Interactive debug panel with live telemetry (Ctrl+Shift+D)
- 💾 **Offline Logging**: localStorage backup when upload fails
- 📤 **Backend Upload**: Automatic telemetry upload every 30 seconds

## Quick Start

```typescript
import {
  initializeTelemetry,
  useTelemetry,
  useThreeJSTelemetry,
  DebugOverlay,
  useDebugOverlay
} from '../telemetry';

// 1. Initialize in your main app component
function App() {
  useEffect(() => {
    initializeTelemetry();
  }, []);

  return <YourApp />;
}

// 2. Add to React components
function MyComponent() {
  const telemetry = useTelemetry('MyComponent');

  const handleClick = () => {
    telemetry.logInteraction('button_click', { buttonId: 'submit' });
  };

  return <button onClick={handleClick}>Submit</button>;
}

// 3. Add to Three.js components
function MyThreeJSObject({ objectId }) {
  const threeTelemetry = useThreeJSTelemetry(objectId);

  useFrame(() => {
    threeTelemetry.logAnimationFrame(position, rotation);
  });

  const handlePositionUpdate = (newPosition) => {
    threeTelemetry.logPositionUpdate(newPosition, {
      reason: 'user_interaction'
    });
  };
}

// 4. Add debug overlay
function App() {
  const debugOverlay = useDebugOverlay();

  return (
    <>
      <YourApp />
      <DebugOverlay
        visible={debugOverlay.visible}
        onToggle={debugOverlay.toggle}
      />
    </>
  );
}
```

## Debug Overlay

Press **Ctrl+Shift+D** to toggle the debug overlay:

- **📊 Metrics**: Live performance metrics and system stats
- **🤖 Agents**: Recent agent activities and state changes
- **📡 WebSocket**: Message flow and connection status
- **🎮 Three.js**: Rendering operations and position tracking

### Position Tracking

The system logs all position updates for debugging:

- **Telemetry data**: Position tracking with timestamps
- **Debug logs**: Detailed position information for manual analysis

## Manual Logging

```typescript
import { agentTelemetry } from '../telemetry';

// Log agent actions
agentTelemetry.logAgentSpawn('agent-123', 'coder', {
  capabilities: ['typescript', 'react']
});

// Log WebSocket messages
agentTelemetry.logWebSocketMessage('agent-update', 'incoming', agentData);

// Log Three.js operations
agentTelemetry.logThreeJSOperation('position_update', 'mesh-456',
  { x: 10, y: 5, z: 0 }, // position
  { x: 0, y: 1.5, z: 0 }, // rotation
  { reason: 'physics_simulation' }
);

// Log performance
agentTelemetry.logRenderCycle(frameTime);
```

## Console Output

All telemetry creates organized console groups:

```
🤖 Agent Spawned: coder:agent-123
├─ Agent Type: coder
├─ Agent ID: agent-123
├─ Metadata: { capabilities: ['typescript', 'react'] }
└─ Total Spawned: 5

📥 WebSocket Message: botsGraphUpdate
├─ Type: botsGraphUpdate
├─ Direction: incoming
├─ Size: 2.1KB
└─ Data: { nodes: [...], edges: [...] }

🔴 THREE.JS CLUSTERING DETECTED: Object mesh-456 at origin
├─ Position: { x: 0.001, y: -0.002, z: 0.000 }
├─ Distance: 0.0022
└─ Metadata: { agentType: 'coder', status: 'active' }
```

## Data Upload

Telemetry automatically uploads to `/api/telemetry/upload` every 30 seconds:

```json
{
  "sessionId": "session_1726570123456_abc123def",
  "timestamp": "2025-09-17T09:25:00.000Z",
  "metrics": {
    "agentSpawns": 12,
    "webSocketMessages": 156,
    "threeJSOperations": 2341,
    "renderCycles": 1842,
    "averageFrameTime": 16.7,
    "memoryUsage": 45678901,
    "errorCount": 0
  },
  "systemInfo": {
    "userAgent": "Mozilla/5.0...",
    "viewport": { "width": 1920, "height": 1080 },
    "pixelRatio": 2,
    "webglRenderer": "ANGLE (Apple, Apple M1, OpenGL 4.1)"
  }
}
```

## Files Created

- `/ext/client/src/telemetry/AgentTelemetry.ts` - Main telemetry service
- `/ext/client/src/telemetry/useTelemetry.ts` - React hooks
- `/ext/client/src/telemetry/DebugOverlay.tsx` - Debug overlay component
- `/ext/client/src/telemetry/index.ts` - Exports and utilities
- `/ext/client/src/utils/logger.ts` - Enhanced logger (updated)

## Integration Points

The telemetry system is integrated into:

- `BotsVisualizationFixed.tsx` - Three.js position and rendering telemetry
- `BotsWebSocketIntegration.ts` - WebSocket message flow logging
- `useBotsWebSocketIntegration.ts` - Connection status and data flow

## Debugging Position Issues

The system logs all position data for manual debugging:

1. **Position logging**: Tracks all position updates with metadata
2. **Timestamps**: Precise timing for each update
3. **Manual inspection**: Check logs to identify position problems

This comprehensive telemetry system provides full visibility into the hive mind visualization, helping debug issues and monitor performance in real-time.