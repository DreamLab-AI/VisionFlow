# Client Documentation

The VisionFlow frontend is a modern React application built with TypeScript, Three.js for 3D visualisation, and Material-UI for the interface.

## Architecture Overview

### Technology Stack
- **React 18**: Modern React with hooks and concurrent features
- **TypeScript**: Type-safe development with strict mode
- **Three.js**: 3D rendering with React Three Fiber
- **Material-UI**: Component library for consistent UI
- **Zustand**: Lightweight state management
- **WebXR**: Virtual and augmented reality support

### Key Features
- **Dual Graph Visualization**: Parallel rendering of Logseq and AI agent graphs
- **Real-time Updates**: Binary WebSocket protocol for 60 FPS performance
- **GPU Physics**: Smooth spring physics animation
- **Command Palette**: Keyboard-driven interface
- **Modular UI**: Detachable panels and customizable layout

## Core Architecture

### Foundation
- **[System Architecture](architecture.md)** - Component hierarchy and data flow
- **[State Management](state.md)** - Zustand stores and context providers
- **[Type System](types.md)** - TypeScript interfaces and types
- **[Core Utilities](core.md)** - Helper functions and hooks

### Component Library
- **[React Components](components.md)** - Reusable UI components
- **[UI Components](ui-components.md)** - Material-UI based design system
- **[User Controls](user-controls-summary.md)** - Interactive controls reference

## Visualization Engine

### 3D Rendering
- **[Rendering System](rendering.md)** - Three.js rendering pipeline
- **[Graph Visualization](visualisation.md)** - Force-directed graph layout
- **[Parallel Graphs](parallel-graphs.md)** - Multi-graph rendering system

### Real-time Communication
- **[WebSocket Integration](websocket.md)** - Binary protocol implementation
- **[Connection Management](websocket-readiness.md)** - State and reconnection
- **[Binary Protocol](../api/binary-protocol.md)** - Position update format

## User Interface

### Settings & Configuration
- **[Settings Panel](settings-panel-redesign.md)** - Advanced configuration UI
- **[Settings Migration](settings-migration.md)** - Upgrade handling

### User Experience
- **[Command Palette](command-palette.md)** - Quick actions and search
- **[Help System](help-system.md)** - Interactive help and tooltips
- **[Onboarding](onboarding.md)** - New user guidance

### Extended Reality
- **[WebXR Integration](xr.md)** - VR/AR capabilities
- **[Hand Tracking](xr.md#hand-tracking)** - Meta Quest 3 support

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
```

### Project Structure
```
client/src/
├── app/              # Application root
├── components/       # Shared components
├── features/         # Feature modules
│   ├── bots/        # AI agent visualization
│   ├── graph/       # Graph rendering
│   ├── settings/    # Settings UI
│   └── xr/          # WebXR features
├── hooks/            # Custom React hooks
├── services/         # API and WebSocket
├── store/            # Zustand stores
├── types/            # TypeScript types
└── utils/            # Utilities
```

## Component Examples

### Using the Graph Visualization
```tsx
import { GraphCanvas } from '@/features/graph/components/GraphCanvas';
import { useGraphData } from '@/hooks/useGraphData';

function MyGraph() {
  const { nodes, edges } = useGraphData();
  
  return (
    <GraphCanvas
      nodes={nodes}
      edges={edges}
      physics={{ springStrength: 0.3 }}
    />
  );
}
```

### Accessing Settings
```tsx
import { useSettingsStore } from '@/store/settingsStore';

function MyComponent() {
  const nodeSize = useSettingsStore(s => s.settings.visualisation.nodes.nodeSize);
  const updateSettings = useSettingsStore(s => s.updateSettings);
  
  const handleSizeChange = (newSize: number) => {
    updateSettings({
      visualisation: {
        nodes: { nodeSize: newSize }
      }
    });
  };
}
```

## Performance Optimization

### Rendering Performance
- **Instanced Rendering**: Batch rendering for nodes/edges
- **LOD System**: Level of detail based on distance
- **Frustum Culling**: Skip off-screen objects
- **GPU Physics**: Offload calculations to GPU

### React Optimization
- **Memoization**: Use React.memo and useMemo
- **Virtualization**: Virtual scrolling for lists
- **Code Splitting**: Lazy load heavy components
- **Web Workers**: Offload physics calculations

## Troubleshooting

### Common Issues

1. **WebGL Context Lost**
   - Check GPU memory usage
   - Reduce max visible nodes
   - Enable performance mode

2. **WebSocket Disconnections**
   - Check network stability
   - Monitor console for errors
   - Verify server is running

3. **Performance Issues**
   - Enable GPU acceleration
   - Reduce physics quality
   - Lower update frequency

### Debug Tools
```typescript
// Enable debug mode
localStorage.setItem('debug', 'app:*');

// Monitor performance
import { Stats } from '@react-three/drei';
<Stats showPanel={0} />;

// Log WebSocket traffic
window.DEBUG_WEBSOCKET = true;
```