# VisionFlow Client Documentation

The VisionFlow client is a modern React application built with TypeScript, React Three Fiber for 3D visualisation, and an advanced state management system. It provides immersive graph visualisation with real-time updates, WebXR support, and comprehensive AI integration.

## Architecture Overview

### Technology Stack
- **React 18**: Modern React with hooks and concurrent features
- **TypeScript**: Type-safe development with strict mode
- **React Three Fiber**: 3D rendering with declarative Three.js
- **Zustand**: Lightweight, performant state management
- **WebXR**: Virtual and augmented reality support via `@react-three/xr`
- **Tailwind CSS**: Utility-first styling framework
- **Vite**: Fast build tool and development server

### Key Features
- **Dual Graph Visualisation**: Parallel rendering of Logseq and VisionFlow graphs
- **Real-time Binary Protocol**: 60fps position updates via WebSocket
- **GPU-Accelerated Physics**: Smooth spring physics animation with CUDA support
- **Command Palette**: Keyboard-driven interface for power users
- **Modular Component System**: Detachable panels and customisable layout
- **Multi-platform XR**: Quest 3, Vision Pro, and WebXR device support
- **Advanced Authentication**: Nostr-based decentralised identity

## Core Documentation

### Foundation
- **[State Management](state-management.md)** - Zustand stores, contexts, and data flow patterns
- **[Graph System](graph-system.md)** - Graph rendering, physics, and visualisation engine
- **[WebXR Integration](xr-integration.md)** - VR/AR capabilities and Quest 3 support

## Project Structure

```
client/src/
├── app/                  # Application root and layout components
│   ├── App.tsx          # Main application component
│   ├── AppInitializer.tsx # Startup logic and service initialization
│   ├── MainLayout.tsx   # Desktop/mobile layout
│   └── Quest3AR.tsx     # Quest 3 AR-specific layout
├── components/          # Shared UI components
│   ├── AuthGatedVoiceButton.tsx
│   ├── ConnectionWarning.tsx
│   └── performance/     # Performance monitoring components
├── features/            # Feature-based modular architecture
│   ├── auth/           # Authentication system
│   ├── bots/           # AI agent visualisation
│   ├── command-palette/ # Quick actions and search
│   ├── design-system/  # Reusable UI components
│   ├── graph/          # Graph rendering and management
│   ├── settings/       # Settings UI and configuration
│   ├── visualisation/  # 3D rendering components
│   └── xr/             # WebXR and VR/AR features
├── hooks/              # Custom React hooks
├── services/           # API and external service integrations
├── store/              # Zustand state stores
├── types/              # TypeScript type definitions
└── utils/              # Utility functions and helpers
```

## Quick Start

### Development Setup
```bash
# Install dependencies
cd client
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Type checking
npm run typecheck

# Linting
npm run lint
```

### Environment Configuration
```typescript
// Essential environment variables
VITE_BACKEND_URL=http://localhost:3001
VITE_WEBSOCKET_URL=ws://localhost:3001/wss
VITE_XR_ENABLED=true
VITE_DEBUG_MODE=false
```

## Component Examples

### Basic Graph Visualisation
```tsx
import { GraphCanvas } from '@/features/graph/components/GraphCanvas';
import { useGraphData } from '@/features/graph/managers/graphDataManager';

function MyGraphComponent() {
  const { nodes, edges, metadata } = useGraphData();
  
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <GraphCanvas
        nodes={nodes}
        edges={edges}
        metadata={metadata}
        physics={{ 
          springStrength: 0.3,
          iterations: 10 
        }}
      />
    </div>
  );
}
```

### Settings Integration
```tsx
import { useSettingsStore } from '@/store/settingsStore';

function SettingsComponent() {
  // Subscribe to specific settings with selectors
  const nodeSize = useSettingsStore(s => 
    s.settings.visualisation.graphs.logseq.nodes.nodeSize
  );
  const updateSettings = useSettingsStore(s => s.updateSettings);
  
  const handleNodeSizeChange = (newSize: number) => {
    updateSettings({
      visualisation: {
        graphs: {
          logseq: {
            nodes: { nodeSize: newSize }
          }
        }
      }
    });
  };

  return (
    <input
      type="range"
      min={0.1}
      max={5.0}
      step={0.1}
      value={nodeSize}
      onChange={(e) => handleNodeSizeChange(parseFloat(e.target.value))}
    />
  );
}
```

### WebXR Integration
```tsx
import { XRController } from '@/features/xr/components/XRController';
import { useXRCore } from '@/features/xr/providers/XRCoreProvider';

function XRComponent() {
  const { isSessionActive, sessionType, enterXR, exitXR } = useXRCore();
  
  return (
    <div>
      {!isSessionActive ? (
        <button onClick={() => enterXR('immersive-vr')}>
          Enter VR Mode
        </button>
      ) : (
        <button onClick={exitXR}>
          Exit XR Mode
        </button>
      )}
      
      <XRController />
    </div>
  );
}
```

## Performance Optimisation

### Rendering Performance
- **Instanced Rendering**: Batch rendering for thousands of nodes/edges
- **Level of Detail (LOD)**: Dynamic quality based on camera distance
- **Frustum Culling**: Skip rendering off-screen objects
- **GPU Physics**: Offload calculations to CUDA-enabled GPUs
- **Memory Management**: Efficient cleanup and garbage collection

### React Optimisation
```tsx
// Use React.memo for expensive components
const ExpensiveGraphComponent = React.memo(({ nodes, edges }) => {
  // Expensive rendering logic
}, (prevProps, nextProps) => {
  // Custom comparison for optimal re-renders
  return prevProps.nodes.length === nextProps.nodes.length &&
         prevProps.edges.length === nextProps.edges.length;
});

// Optimise with useMemo for computed values
function GraphMetrics({ nodes }) {
  const nodeStats = useMemo(() => ({
    total: nodes.length,
    connected: nodes.filter(n => n.edges.length > 0).length,
    isolated: nodes.filter(n => n.edges.length === 0).length
  }), [nodes]);
  
  return <div>{/* Display stats */}</div>;
}
```

### Web Workers
```typescript
// Physics calculations in web worker
// client/src/features/graph/workers/graph.worker.ts
self.onmessage = function(e) {
  const { nodes, edges, settings } = e.data;
  
  // Perform expensive physics calculations
  const updatedPositions = calculatePhysics(nodes, edges, settings);
  
  self.postMessage({ positions: updatedPositions });
};
```

## Multi-Graph Architecture

VisionFlow supports simultaneous visualisation of multiple graph types:

```typescript
// Independent graph settings
interface GraphSettings {
  logseq: {
    nodes: { baseColor: '#4B5EFF' }, // Blue theme
    edges: { color: '#6B73FF' }
  },
  visionflow: {
    nodes: { baseColor: '#10B981' }, // Green theme  
    edges: { color: '#34D399' }
  }
}

// Accessing graph-specific settings
const logseqNodeSize = useSettingsStore(s => 
  s.settings.visualisation.graphs.logseq.nodes.nodeSize
);
```

## Authentication & Security

### Nostr Authentication
```typescript
import { nostrAuthService } from '@/services/nostrAuthService';

// Authenticate with Nostr
const authenticate = async () => {
  try {
    const result = await nostrAuthService.authenticate();
    if (result.success) {
      console.log('Authenticated:', result.publicKey);
    }
  } catch (error) {
    console.error('Authentication failed:', error);
  }
};
```

### Feature Access Control
```typescript
// Settings-based feature flags
const xrEnabled = useSettingsStore(s => s.settings.xr?.enabled);
const aiEnabled = useSettingsStore(s => s.settings.ai?.enabled);

// Conditional rendering based on features
{xrEnabled && <XRComponents />}
{aiEnabled && <AIFeatures />}
```

## Troubleshooting

### Common Issues

1. **WebGL Context Lost**
   ```javascript
   // Monitor WebGL context
   canvas.addEventListener('webglcontextlost', (event) => {
     event.preventDefault();
     console.warn('WebGL context lost, attempting recovery...');
   });
   
   canvas.addEventListener('webglcontextrestored', () => {
     console.info('WebGL context restored');
     // Reinitialise resources
   });
   ```

2. **WebSocket Connection Issues**
   ```typescript
   // Debug WebSocket connectivity
   const wsService = WebSocketService.getInstance();
   
   wsService.onConnectionStatusChange((status) => {
     if (!status.connected) {
       console.error('WebSocket disconnected:', status.error);
       // Implement retry logic
     }
   });
   ```

3. **XR Session Failures**
   ```typescript
   // Check XR support
   if (!navigator.xr) {
     console.warn('WebXR not supported');
     return;
   }
   
   const supported = await navigator.xr.isSessionSupported('immersive-vr');
   if (!supported) {
     console.warn('VR mode not supported');
   }
   ```

### Debug Tools

```typescript
// Enable comprehensive debugging
localStorage.setItem('debug', 'app:*');

// Performance monitoring
import { Stats } from '@react-three/drei';
<Stats showPanel={0} className="stats-overlay" />;

// Memory usage tracking
const monitor = () => {
  if (performance.memory) {
    console.log({
      used: Math.round(performance.memory.usedJSHeapSize / 1048576) + 'MB',
      total: Math.round(performance.memory.totalJSHeapSize / 1048576) + 'MB',
      limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576) + 'MB'
    });
  }
};
```

## Browser Compatibility

### Supported Browsers
- **Chrome 90+**: Full WebXR and WebGL 2.0 support
- **Firefox 85+**: Limited WebXR, full WebGL 2.0
- **Safari 14+**: WebGL 2.0, no WebXR support
- **Edge 90+**: Full feature support

### Mobile Support
- **Quest Browser**: Native WebXR optimisations
- **iOS Safari**: Progressive Web App capabilities
- **Android Chrome**: WebXR support (device dependent)

## Next Steps

1. **Read the Core Documentation**:
   - [State Management](state-management.md) for data flow patterns
   - [Graph System](graph-system.md) for rendering architecture
   - [XR Integration](xr-integration.md) for immersive features

2. **Explore the Codebase**:
   - Start with `src/app/App.tsx` for application structure
   - Review `src/features/` for modular architecture
   - Check `src/store/settingsStore.ts` for state management

3. **Build Your First Feature**:
   - Create a new feature module in `src/features/`
   - Integrate with the existing state management
   - Add appropriate TypeScript types

The VisionFlow client provides a robust foundation for building immersive graph visualisation experiences with modern web technologies and best practices.