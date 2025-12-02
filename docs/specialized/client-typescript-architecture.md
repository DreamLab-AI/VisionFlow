# Client TypeScript Architecture

**VisionFlow Client Architecture Documentation**
**Version:** 0.1.0
**Last Updated:** 2025-11-04
**Technology Stack:** React 18, TypeScript 5.8, Three.js, Babylon.js, Vite 6

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [Project Structure](#project-structure)
4. [Core Technologies](#core-technologies)
5. [Component Architecture](#component-architecture)
6. [State Management](#state-management)
7. [API Client Layer](#api-client-layer)
8. [WebSocket Communication](#websocket-communication)
9. [WebXR Integration](#webxr-integration)
10. [3D Visualization](#3d-visualization)
11. [Type System](#type-system)
12. [Service Layer](#service-layer)
13. [Feature Modules](#feature-modules)
14. 
15. [Performance Optimization](#performance-optimization)
16. [Testing Strategy](#testing-strategy)
17. [Security](#security)
18. [Best Practices](#best-practices)

---

## Overview

The VisionFlow client is a sophisticated React-based application built with TypeScript that provides immersive 3D graph visualization, WebXR support for Meta Quest 3, and real-time multi-agent collaboration. It combines cutting-edge web technologies to deliver high-performance visualization of knowledge graphs and ontologies.

### Key Features

- **3D Graph Visualization**: Real-time rendering of knowledge graphs with physics simulation
- **WebXR Support**: Native immersive AR/VR experiences for Meta Quest 3
- **Multi-Agent System**: Real-time collaboration with autonomous bots
- **Dual Rendering**: Three.js for standard 3D, Babylon.js for XR experiences
- **Real-time Communication**: WebSocket-based binary protocol for high-throughput data streaming
- **State Management**: Zustand-based reactive state with persistence
- **Modular Architecture**: Feature-based organization with clear boundaries

### Architecture Goals

1. **Performance**: 60 FPS rendering with thousands of nodes
2. **Scalability**: Handle graphs with 10,000+ nodes efficiently
3. **Maintainability**: Clear separation of concerns and modular design
4. **Type Safety**: Comprehensive TypeScript coverage
5. **Extensibility**: Plugin-based architecture for features
6. **Accessibility**: Support for multiple input modalities

---

## Architecture Principles

### 1. Feature-Driven Organization

```
src/
├── features/          # Feature modules (self-contained)
│   ├── analytics/     # Analytics and clustering
│   ├── bots/          # Multi-agent system
│   ├── graph/         # Graph visualization
│   ├── settings/      # Settings management
│   └── visualisation/ # 3D rendering
├── services/          # Shared services
├── store/             # Global state
├── components/        # Shared UI components
└── utils/             # Utilities
```

### 2. Separation of Concerns

- **Presentation Layer**: React components (UI)
- **Business Logic**: Hooks and managers
- **Data Layer**: Stores and services
- **Communication**: API clients and WebSocket

### 3. Dependency Injection

```typescript
// Services are singletons
export const webSocketService = WebSocketService.getInstance();

// Managers follow singleton pattern
export const graphDataManager = GraphDataManager.getInstance();

// Stores use Zustand
export const useSettingsStore = create<SettingsState>()(/*...*/);
```

### 4. Type Safety

- Strict TypeScript configuration
- Generated types from Rust backend
- Comprehensive interface definitions
- Runtime type validation for critical paths

### 5. Performance First

- Lazy loading for non-critical features
- Code splitting by route and feature
- Web Workers for heavy computation
- Memoization and React optimization

---

## Project Structure

### Directory Layout

```
client/
├── src/
│   ├── api/                    # API clients
│   │   ├── apiClient.ts       # Base HTTP client
│   │   ├── settingsApi.ts     # Settings endpoints
│   │   └── authInterceptor.ts # Auth middleware
│   │
│   ├── app/                    # Application shell
│   │   ├── App.tsx            # Root component
│   │   ├── AppInitializer.tsx # Initialization logic
│   │   ├── MainLayout.tsx     # Desktop layout
│   │   └── main.tsx           # Entry point
│   │
│   ├── components/             # Shared components
│   │   ├── ErrorBoundary.tsx
│   │   ├── ConnectionWarning.tsx
│   │   └── DebugControlPanel.tsx
│   │
│   ├── contexts/               # React contexts
│   │   ├── ApplicationModeContext.tsx
│   │   ├── VircadiaContext.tsx
│   │   └── VircadiaBridgesContext.tsx
│   │
│   ├── features/               # Feature modules
│   │   ├── analytics/         # Graph analytics
│   │   │   ├── components/
│   │   │   ├── store/
│   │   │   └── index.ts
│   │   │
│   │   ├── bots/              # Multi-agent system
│   │   │   ├── components/
│   │   │   ├── contexts/
│   │   │   ├── hooks/
│   │   │   ├── services/
│   │   │   └── types/
│   │   │
│   │   ├── command-palette/   # Command system
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── CommandRegistry.ts
│   │   │   └── defaultCommands.ts
│   │   │
│   │   ├── design-system/     # UI components
│   │   │   ├── components/
│   │   │   └── patterns/
│   │   │
│   │   ├── graph/             # Graph visualization
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── managers/
│   │   │   ├── services/
│   │   │   ├── types/
│   │   │   ├── utils/
│   │   │   └── workers/
│   │   │
│   │   ├── help/              # Help system
│   │   ├── onboarding/        # User onboarding
│   │   ├── ontology/          # Ontology management
│   │   ├── physics/           # Physics engine
│   │   ├── settings/          # Settings UI
│   │   ├── visualisation/     # 3D rendering
│   │   └── workspace/         # Workspace management
│   │
│   ├── hooks/                  # Shared hooks
│   │   ├── useQuest3Integration.ts
│   │   ├── useAutoBalanceNotifications.ts
│   │   └── useWebSocket.ts
│   │
│   ├── immersive/              # XR/Immersive experiences
│   │   ├── babylon/           # Babylon.js renderer
│   │   ├── components/        # XR UI components
│   │   └── hooks/             # XR hooks
│   │
│   ├── rendering/              # Rendering utilities
│   │   └── materials/         # Custom materials
│   │
│   ├── services/               # Core services
│   │   ├── api/               # API service layer
│   │   ├── bridges/           # Service bridges
│   │   ├── vircadia/          # Vircadia integration
│   │   ├── WebSocketService.ts
│   │   ├── BinaryWebSocketProtocol.ts
│   │   ├── nostrAuthService.ts
│   │   └── remoteLogger.ts
│   │
│   ├── shaders/                # GLSL shaders
│   │
│   ├── store/                  # Global state
│   │   ├── settingsStore.ts   # Settings state
│   │   ├── multiUserStore.ts  # Multi-user state
│   │   ├── autoSaveManager.ts # Auto-save logic
│   │   └── settingsRetryManager.ts
│   │
│   ├── styles/                 # Global styles
│   │   └── index.css
│   │
│   ├── telemetry/              # Telemetry and monitoring
│   │   ├── AgentTelemetry.ts
│   │   ├── useTelemetry.ts
│   │   └── index.ts
│   │
│   ├── types/                  # Type definitions
│   │   ├── generated/         # Auto-generated types
│   │   ├── binaryProtocol.ts
│   │   ├── websocketTypes.ts
│   │   └── index.ts
│   │
│   ├── utils/                  # Utility functions
│   │   ├── loggerConfig.ts
│   │   ├── debugConfig.ts
│   │   ├── validation.ts
│   │   ├── BatchQueue.ts
│   │   └── accessibility.ts
│   │
│   └── xr/                     # WebXR features
│       └── vircadia/          # Vircadia XR support
│
├── public/                     # Static assets
├── vite.config.ts             # Vite configuration
├── tsconfig.json              # TypeScript config
├── package.json               # Dependencies
└── tailwind.config.js         # Tailwind CSS config
```

### File Naming Conventions

- **Components**: PascalCase (`GraphCanvas.tsx`, `NodeRenderer.tsx`)
- **Hooks**: camelCase with `use` prefix (`useGraphData.ts`, `useWebSocket.ts`)
- **Services**: PascalCase (`WebSocketService.ts`, `GraphDataManager.ts`)
- **Utilities**: camelCase (`loggerConfig.ts`, `validation.ts`)
- **Types**: camelCase (`websocketTypes.ts`, `binaryProtocol.ts`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_NODES`, `DEFAULT_SETTINGS`)

---

## Core Technologies

### React 18.2

**Why React?**
- Virtual DOM for efficient updates
- Component-based architecture
- Extensive ecosystem
- React Three Fiber for 3D integration

**Key Features Used:**
```typescript
// Concurrent features
import { Suspense, lazy } from 'react';

// Hooks
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

// Context API
import { createContext, useContext } from 'react';

// Error boundaries
class ErrorBoundary extends React.Component {/*...*/}
```

### TypeScript 5.8

**Why TypeScript?**
- Type safety reduces runtime errors
- Better IDE support and autocomplete
- Self-documenting code
- Easier refactoring

**Configuration:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "react-jsx",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "allowSyntheticDefaultImports": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true
  }
}
```

### Three.js 0.175.0

**Why Three.js?**
- Industry-standard WebGL library
- Rich ecosystem of plugins
- Excellent performance
- React Three Fiber integration

**Usage:**
```typescript
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';

// Custom materials, geometries, and shaders
```

### Babylon.js 8.28.0

**Why Babylon.js?**
- Superior WebXR support
- Built-in XR features
- Better for immersive experiences
- Optimized for VR/AR

**Usage:**
```typescript
import * as BABYLON from '@babylonjs/core';
import { Scene, Engine, Camera, Light } from '@babylonjs/core';

// XR-specific features
import '@babylonjs/loaders';
import '@babylonjs/materials';
```

### Vite 6.2.6

**Why Vite?**
- Lightning-fast HMR
- Native ES modules
- Optimized builds
- Better DX than webpack

**Configuration:**
```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    hmr: { clientPort: 3001, path: '/vite-hmr' },
    proxy: {
      '/api': { target: 'http://visionflow_container:4000' },
      '/ws': { target: 'ws://visionflow_container:4000', ws: true }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});
```

### Zustand

**Why Zustand?**
- Simpler than Redux
- No boilerplate
- Great TypeScript support
- Built-in persistence

**Example:**
```typescript
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      settings: {},
      initialized: false,

      initialize: async () => {
        const settings = await settingsApi.getAll();
        set({ settings, initialized: true });
      },

      updateSetting: (path, value) => {
        set(state => ({
          settings: updateNestedValue(state.settings, path, value)
        }));
      }
    }),
    {
      name: 'settings-storage',
      storage: createJSONStorage(() => localStorage)
    }
  )
);
```

### Axios

**Why Axios?**
- Promise-based HTTP client
- Interceptors for auth
- Request/response transformation
- Better error handling than fetch

**Configuration:**
```typescript
import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor for authentication
apiClient.interceptors.request.use(config => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

### Tailwind CSS 4.1.3

**Why Tailwind?**
- Utility-first CSS
- Rapid prototyping
- Consistent design system
- Small production bundle

**Usage:**
```tsx
<div className="flex items-center justify-between p-4 bg-slate-900 rounded-lg">
  <h2 className="text-2xl font-bold text-white">Graph Visualization</h2>
  <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded">
    Toggle XR
  </button>
</div>
```

---

## Component Architecture

### Component Hierarchy

```
App (Root)
├── ApplicationModeProvider
│   ├── VircadiaProvider
│   │   ├── VircadiaBridgesProvider
│   │   │   ├── ImmersiveApp (Quest 3)
│   │   │   │   └── BabylonGraphRenderer
│   │   │   │
│   │   │   └── MainLayout (Desktop)
│   │   │       ├── GraphCanvas
│   │   │       │   ├── GraphManager
│   │   │       │   ├── BotsVisualization
│   │   │       │   ├── OrbitControls
│   │   │       │   └── SelectiveBloom
│   │   │       │
│   │   │       ├── IntegratedControlPanel
│   │   │       │   ├── PhysicsControls
│   │   │       │   ├── RenderingSettings
│   │   │       │   └── GraphSelector
│   │   │       │
│   │   │       └── ConversationPane
│   │   │           ├── AgentDetailPanel
│   │   │           └── ActivityLogPanel
│   │   │
│   │   ├── BotsDataProvider
│   │   ├── TooltipProvider
│   │   ├── HelpProvider
│   │   ├── OnboardingProvider
│   │   └── ErrorBoundary
│   │
│   ├── CommandPalette
│   ├── DebugControlPanel
│   └── ConnectionWarning
│
└── AppInitializer
```

### Component Patterns

#### 1. Container/Presenter Pattern

**Container Component (Smart):**
```typescript
// GraphCanvasContainer.tsx
import { useState, useEffect } from 'react';
import { graphDataManager } from '../managers/graphDataManager';
import GraphCanvas from './GraphCanvas';

export const GraphCanvasContainer: React.FC = () => {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      const data = await graphDataManager.getGraphData();
      setGraphData(data);
      setLoading(false);
    };
    loadData();

    return graphDataManager.onGraphDataChange(setGraphData);
  }, []);

  if (loading) return <LoadingSpinner />;

  return <GraphCanvas data={graphData} />;
};
```

**Presenter Component (Dumb):**
```typescript
// GraphCanvas.tsx
interface GraphCanvasProps {
  data: GraphData;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({ data }) => {
  return (
    <Canvas>
      <GraphManager graphData={data} />
    </Canvas>
  );
};
```

#### 2. Compound Components

```typescript
// ControlPanel and its sub-components
export const ControlPanel: React.FC<ControlPanelProps> = ({ children }) => {
  return (
    <div className="control-panel">
      {children}
    </div>
  );
};

ControlPanel.Section = ({ title, children }) => (
  <div className="control-section">
    <h3>{title}</h3>
    {children}
  </div>
);

ControlPanel.Toggle = ({ label, value, onChange }) => (
  <label>
    <input type="checkbox" checked={value} onChange={onChange} />
    {label}
  </label>
);

// Usage
<ControlPanel>
  <ControlPanel.Section title="Physics">
    <ControlPanel.Toggle label="Enabled" value={enabled} onChange={setEnabled} />
  </ControlPanel.Section>
</ControlPanel>
```

#### 3. Higher-Order Components (HOCs)

```typescript
// withAuthentication.tsx
export function withAuthentication<P extends object>(
  Component: React.ComponentType<P>
) {
  return function AuthenticatedComponent(props: P) {
    const { isAuthenticated } = useAuth();

    if (!isAuthenticated) {
      return <LoginPrompt />;
    }

    return <Component {...props} />;
  };
}

// Usage
export const ProtectedSettings = withAuthentication(SettingsPanel);
```

#### 4. Render Props

```typescript
// DataFetcher component
interface DataFetcherProps<T> {
  url: string;
  children: (data: T | null, loading: boolean, error: Error | null) => JSX.Element;
}

export function DataFetcher<T>({ url, children }: DataFetcherProps<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(data => { setData(data); setLoading(false); })
      .catch(err => { setError(err); setLoading(false); });
  }, [url]);

  return children(data, loading, error);
}

// Usage
<DataFetcher<User> url="/api/user">
  {(user, loading, error) => {
    if (loading) return <Spinner />;
    if (error) return <Error message={error.message} />;
    if (user) return <UserProfile user={user} />;
    return null;
  }}
</DataFetcher>
```

### React Three Fiber Components

#### 3D Scene Structure

```typescript
// GraphCanvas.tsx - Main 3D scene
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';

const GraphCanvas: React.FC = () => {
  return (
    <Canvas
      camera={{
        fov: 75,
        near: 0.1,
        far: 2000,
        position: [20, 15, 20]
      }}
      onCreated={({ gl, camera, scene }) => {
        gl.setClearColor(0x000033, 1);
      }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.15} />
      <directionalLight position={[10, 10, 10]} intensity={0.4} />

      {/* Graph rendering */}
      <GraphManager graphData={graphData} />

      {/* Bots visualization */}
      <BotsVisualization />

      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
      />

      {/* Post-processing */}
      <SelectiveBloom enabled={true} />

      {/* Performance stats */}
      <Stats />
    </Canvas>
  );
};
```

#### Custom 3D Components

```typescript
// NodeMesh.tsx - Individual node rendering
import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface NodeMeshProps {
  position: [number, number, number];
  color: string;
  size: number;
  selected: boolean;
}

export const NodeMesh: React.FC<NodeMeshProps> = ({
  position,
  color,
  size,
  selected
}) => {
  const meshRef = useRef<THREE.Mesh>(null);

  const material = useMemo(() =>
    new THREE.MeshStandardMaterial({
      color,
      emissive: selected ? color : 0x000000,
      emissiveIntensity: selected ? 0.5 : 0
    }),
    [color, selected]
  );

  useFrame((state, delta) => {
    if (meshRef.current && selected) {
      meshRef.current.rotation.y += delta;
    }
  });

  return (
    <mesh ref={meshRef} position={position} material={material}>
      <sphereGeometry args={[size, 32, 32]} />
    </mesh>
  );
};
```

---

## State Management

### Zustand Store Architecture

VisionFlow uses Zustand for state management with a layered approach:

1. **Global State**: Settings, authentication, system state
2. **Feature State**: Feature-specific stores (analytics, ontology)
3. **Local State**: Component-level useState/useReducer

### Settings Store

The settings store is the core global state, managing all application configuration.

**File**: `/src/store/settingsStore.ts`

```typescript
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { produce } from 'immer';

interface SettingsState {
  // State
  partialSettings: DeepPartial<Settings>;
  loadedPaths: Set<string>;
  loadingSections: Set<string>;
  initialized: boolean;
  authenticated: boolean;
  user: User | null;
  subscribers: Map<string, Set<() => void>>;

  // Actions
  initialize: () => Promise<void>;
  get: <T>(path: SettingsPath) => T;
  set: <T>(path: SettingsPath, value: T) => void;
  updateSettings: (updater: (draft: Settings) => void) => void;
  ensureLoaded: (paths: string[]) => Promise<void>;
  loadSection: (section: string) => Promise<void>;
  subscribe: (path: SettingsPath, callback: () => void) => () => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      partialSettings: {},
      loadedPaths: new Set(),
      loadingSections: new Set(),
      initialized: false,
      authenticated: false,
      user: null,
      subscribers: new Map(),

      // Initialize with essential paths only (fast startup)
      initialize: async () => {
        const essentialSettings = await settingsApi.getSettingsByPaths(ESSENTIAL_PATHS);

        set({
          partialSettings: essentialSettings,
          loadedPaths: new Set(ESSENTIAL_PATHS),
          initialized: true
        });
      },

      // Get setting by dot-notation path
      get: <T>(path: SettingsPath): T | undefined => {
        const { partialSettings, loadedPaths } = get();

        // Check if path is loaded
        if (!loadedPaths.has(path)) {
          console.warn(`Accessing unloaded path: ${path}`);
          return undefined;
        }

        // Navigate nested object
        const pathParts = path.split('.');
        let current: any = partialSettings;

        for (const part of pathParts) {
          if (current?.[part] === undefined) {
            return undefined;
          }
          current = current[part];
        }

        return current as T;
      },

      // Set setting by path
      set: <T>(path: SettingsPath, value: T) => {
        set(state => {
          const newSettings = { ...state.partialSettings };
          setNestedValue(newSettings, path, value);

          return {
            partialSettings: newSettings,
            loadedPaths: new Set(state.loadedPaths).add(path)
          };
        });

        // Persist to backend
        settingsApi.updateSettingByPath(path, value);
      },

      // Update settings with Immer
      updateSettings: (updater) => {
        const { partialSettings } = get();
        const newSettings = produce(partialSettings, updater);
        const changedPaths = findChangedPaths(partialSettings, newSettings);

        set({
          partialSettings: newSettings,
          loadedPaths: new Set([...get().loadedPaths, ...changedPaths])
        });

        // Notify subscribers
        const { subscribers } = get();
        changedPaths.forEach(path => {
          subscribers.get(path)?.forEach(callback => callback());
        });
      },

      // Load paths on-demand
      ensureLoaded: async (paths: string[]) => {
        const { loadedPaths } = get();
        const unloadedPaths = paths.filter(p => !loadedPaths.has(p));

        if (unloadedPaths.length === 0) return;

        const pathSettings = await settingsApi.getSettingsByPaths(unloadedPaths);

        set(state => ({
          partialSettings: { ...state.partialSettings, ...pathSettings },
          loadedPaths: new Set([...state.loadedPaths, ...unloadedPaths])
        }));
      },

      // Subscribe to path changes
      subscribe: (path, callback) => {
        set(state => {
          const subscribers = new Map(state.subscribers);
          if (!subscribers.has(path)) {
            subscribers.set(path, new Set());
          }
          subscribers.get(path)!.add(callback);
          return { subscribers };
        });

        // Return unsubscribe function
        return () => {
          set(state => {
            const subscribers = new Map(state.subscribers);
            subscribers.get(path)?.delete(callback);
            return { subscribers };
          });
        };
      }
    }),
    {
      name: 'graph-viz-settings-v2',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        authenticated: state.authenticated,
        user: state.user,
        // Only persist essential paths
        essentialPaths: ESSENTIAL_PATHS.reduce((acc, path) => {
          const value = state.partialSettings[path];
          if (value !== undefined) acc[path] = value;
          return acc;
        }, {} as Record<string, any>)
      })
    }
  )
);
```

### Lazy Loading Strategy

The settings store implements a lazy loading pattern to minimize initial load time:

```typescript
// Essential paths loaded at startup
const ESSENTIAL_PATHS = [
  'system.debug.enabled',
  'system.websocket.updateRate',
  'auth.enabled',
  'visualisation.rendering.context',
  'xr.enabled'
];

// Component requests additional settings
function SettingsPanel() {
  const settingsStore = useSettingsStore();

  useEffect(() => {
    // Load section on mount
    settingsStore.loadSection('rendering');
  }, []);

  const ambientLight = settingsStore.get('visualisation.rendering.ambientLightIntensity');

  return <Slider value={ambientLight} onChange={/*...*/} />;
}
```

### Multi-User Store

Handles multi-user sessions and collaboration.

**File**: `/src/store/multiUserStore.ts`

```typescript
interface MultiUserState {
  currentUser: User | null;
  connectedUsers: Map<string, User>;
  cursors: Map<string, CursorPosition>;

  setCurrentUser: (user: User) => void;
  addUser: (user: User) => void;
  removeUser: (userId: string) => void;
  updateCursor: (userId: string, position: CursorPosition) => void;
}

export const useMultiUserStore = create<MultiUserState>((set) => ({
  currentUser: null,
  connectedUsers: new Map(),
  cursors: new Map(),

  setCurrentUser: (user) => set({ currentUser: user }),

  addUser: (user) => set(state => ({
    connectedUsers: new Map(state.connectedUsers).set(user.id, user)
  })),

  removeUser: (userId) => set(state => {
    const users = new Map(state.connectedUsers);
    users.delete(userId);
    return { connectedUsers: users };
  }),

  updateCursor: (userId, position) => set(state => ({
    cursors: new Map(state.cursors).set(userId, position)
  }))
}));
```

### Auto-Save Manager

Coordinates auto-saving of settings with debouncing and retry logic.

**File**: `/src/store/autoSaveManager.ts`

```typescript
class AutoSaveManager {
  private pendingUpdates: Map<string, any> = new Map();
  private saveTimeout: NodeJS.Timeout | null = null;
  private saveDebounceMs = 1000;
  private initialized = false;

  setInitialized(initialized: boolean) {
    this.initialized = initialized;
  }

  queueUpdate(path: string, value: any) {
    if (!this.initialized) return;

    this.pendingUpdates.set(path, value);

    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
    }

    this.saveTimeout = setTimeout(() => {
      this.flush();
    }, this.saveDebounceMs);
  }

  async flush() {
    if (this.pendingUpdates.size === 0) return;

    const updates = Array.from(this.pendingUpdates.entries());
    this.pendingUpdates.clear();

    try {
      await settingsApi.batchUpdate(updates.map(([path, value]) => ({
        path,
        value
      })));
    } catch (error) {
      console.error('Failed to flush auto-save:', error);
      // Re-queue failed updates
      updates.forEach(([path, value]) => {
        this.pendingUpdates.set(path, value);
      });
    }
  }
}

export const autoSaveManager = new AutoSaveManager();
```

---

## API Client Layer

### Architecture Overview

```
┌─────────────────┐
│   Components    │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Hooks   │
    └────┬─────┘
         │
  ┌──────▼────────┐
  │  API Clients  │ ← axios interceptors, auth
  └──────┬────────┘
         │
  ┌──────▼────────┐
  │  HTTP/HTTPS   │
  └──────┬────────┘
         │
  ┌──────▼────────┐
  │ Rust Backend  │
  └───────────────┘
```

### Base API Client

**File**: `/src/services/api/UnifiedApiClient.ts`

```typescript
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

class UnifiedApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        // Add request ID for tracing
        config.headers['X-Request-ID'] = crypto.randomUUID();

        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        return response;
      },
      async (error) => {
        // Handle 401 Unauthorized
        if (error.response?.status === 401) {
          // Clear auth and redirect to login
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }

        // Handle 429 Rate Limit
        if (error.response?.status === 429) {
          const retryAfter = error.response.headers['retry-after'];
          if (retryAfter) {
            await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
            return this.client.request(error.config);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  async patch<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
  }
}

export const unifiedApiClient = new UnifiedApiClient();
```

### Settings API

**File**: `/src/api/settingsApi.ts`

```typescript
export const settingsApi = {
  // Get settings by multiple paths
  getSettingsByPaths: async (paths: string[]): Promise<Record<string, any>> => {
    // In development, fetch all settings and filter
    const allSettings = await unifiedApiClient.get<AllSettings>('/settings/all');

    const result: Record<string, any> = {};
    paths.forEach(path => {
      const value = getNestedValue(allSettings, path);
      if (value !== undefined) {
        result[path] = value;
      }
    });

    return result;
  },

  // Get single setting by path
  getSettingByPath: async <T>(path: string): Promise<T> => {
    const allSettings = await unifiedApiClient.get<AllSettings>('/settings/all');
    return getNestedValue(allSettings, path);
  },

  // Update single setting
  updateSettingByPath: async (path: string, value: any): Promise<void> => {
    // Map path to endpoint
    const [category] = path.split('.');

    const updates: Record<string, any> = {};
    setNestedValue(updates, path, value);

    await unifiedApiClient.put(`/settings/${category}`, updates);
  },

  // Batch update settings
  updateSettingsByPaths: async (updates: Array<{path: string, value: any}>): Promise<void> => {
    // Group by category
    const grouped = new Map<string, Record<string, any>>();

    updates.forEach(({ path, value }) => {
      const [category] = path.split('.');
      if (!grouped.has(category)) {
        grouped.set(category, {});
      }
      setNestedValue(grouped.get(category)!, path, value);
    });

    // Send batch requests
    await Promise.all(
      Array.from(grouped.entries()).map(([category, data]) =>
        unifiedApiClient.put(`/settings/${category}`, data)
      )
    );
  },

  // Reset settings to defaults
  resetSettings: async (): Promise<void> => {
    await unifiedApiClient.post('/settings/reset');
  },

  // Export settings as JSON
  exportSettings: (settings: Settings): string => {
    return JSON.stringify(settings, null, 2);
  },

  // Import settings from JSON
  importSettings: (jsonString: string): Settings => {
    return JSON.parse(jsonString);
  }
};
```

### Authentication Interceptor

**File**: `/src/services/api/authInterceptor.ts`

```typescript
import { nostrAuth } from '../nostrAuthService';

export function initializeAuthInterceptor(apiClient: UnifiedApiClient) {
  // Add Nostr authentication token to requests
  apiClient.interceptors.request.use(async (config) => {
    if (nostrAuth.isAuthenticated()) {
      const token = await nostrAuth.getAuthToken();
      if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
      }
    }
    return config;
  });
}

export function setupAuthStateListener() {
  // Listen for authentication state changes
  nostrAuth.onAuthStateChange((authenticated) => {
    if (!authenticated) {
      // Clear cached data
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_session');
    }
  });
}
```

---

## WebSocket Communication

### Binary Protocol

VisionFlow uses a custom binary protocol for high-throughput, low-latency communication.

**File**: `/src/services/BinaryWebSocketProtocol.ts`

```typescript
export enum MessageType {
  GRAPH_UPDATE = 0x01,
  VOICE_DATA = 0x02,
  POSITION_UPDATE = 0x03,
  AGENT_POSITIONS = 0x04,
  HEARTBEAT = 0xFF
}

export enum GraphTypeFlag {
  KNOWLEDGE_GRAPH = 0x01,
  ONTOLOGY = 0x02
}

export class BinaryProtocol {
  // Header structure:
  // [0-3]: Magic bytes (0x56464C57 = "VFLW")
  // [4]: Protocol version
  // [5]: Message type
  // [6]: Graph type flag
  // [7]: Reserved
  // [8-11]: Payload length (uint32)
  // [12+]: Payload

  private static readonly MAGIC = 0x56464C57;
  private static readonly VERSION = 0x01;
  private static readonly HEADER_SIZE = 12;

  static createMessage(
    type: MessageType,
    graphType: GraphTypeFlag,
    payload: ArrayBuffer
  ): ArrayBuffer {
    const header = new ArrayBuffer(this.HEADER_SIZE);
    const view = new DataView(header);

    // Magic bytes
    view.setUint32(0, this.MAGIC, false);

    // Version and type
    view.setUint8(4, this.VERSION);
    view.setUint8(5, type);
    view.setUint8(6, graphType);
    view.setUint8(7, 0); // Reserved

    // Payload length
    view.setUint32(8, payload.byteLength, false);

    // Combine header and payload
    const message = new Uint8Array(this.HEADER_SIZE + payload.byteLength);
    message.set(new Uint8Array(header), 0);
    message.set(new Uint8Array(payload), this.HEADER_SIZE);

    return message.buffer;
  }

  static parseHeader(data: ArrayBuffer): {
    type: MessageType;
    graphTypeFlag: GraphTypeFlag;
    payloadLength: number;
  } | null {
    if (data.byteLength < this.HEADER_SIZE) {
      return null;
    }

    const view = new DataView(data);

    // Verify magic bytes
    const magic = view.getUint32(0, false);
    if (magic !== this.MAGIC) {
      return null;
    }

    // Verify version
    const version = view.getUint8(4);
    if (version !== this.VERSION) {
      return null;
    }

    return {
      type: view.getUint8(5),
      graphTypeFlag: view.getUint8(6),
      payloadLength: view.getUint32(8, false)
    };
  }

  static extractPayload(data: ArrayBuffer, header: ReturnType<typeof BinaryProtocol.parseHeader>): ArrayBuffer {
    if (!header) {
      throw new Error('Invalid header');
    }

    return data.slice(this.HEADER_SIZE, this.HEADER_SIZE + header.payloadLength);
  }
}

export const binaryProtocol = new BinaryProtocol();
```

### WebSocket Service

**File**: `/src/services/WebSocketService.ts`

```typescript
class WebSocketService {
  private static instance: WebSocketService;
  private socket: WebSocket | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private binaryHandlers: Set<BinaryMessageHandler> = new Set();
  private eventHandlers: Map<string, Set<EventHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private isConnected = false;
  private messageQueue: QueuedMessage[] = [];

  private constructor() {
    this.url = this.determineWebSocketUrl();
  }

  static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  async connect(): Promise<void> {
    if (this.socket?.readyState === WebSocket.OPEN) {
      return;
    }

    this.socket = new WebSocket(this.url);
    this.socket.binaryType = 'arraybuffer';

    this.socket.onopen = this.handleOpen.bind(this);
    this.socket.onmessage = this.handleMessage.bind(this);
    this.socket.onclose = this.handleClose.bind(this);
    this.socket.onerror = this.handleError.bind(this);

    return new Promise((resolve, reject) => {
      this.socket!.addEventListener('open', () => resolve(), { once: true });
      this.socket!.addEventListener('error', reject, { once: true });
    });
  }

  private handleMessage(event: MessageEvent): void {
    // Handle binary messages
    if (event.data instanceof ArrayBuffer) {
      this.processBinaryData(event.data);
      return;
    }

    // Handle text messages
    try {
      const message = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(message));
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private async processBinaryData(data: ArrayBuffer): Promise<void> {
    const header = binaryProtocol.parseHeader(data);
    if (!header) {
      console.error('Invalid binary message header');
      return;
    }

    const payload = binaryProtocol.extractPayload(data, header);

    switch (header.type) {
      case MessageType.GRAPH_UPDATE:
        await this.handleGraphUpdate(payload, header);
        break;

      case MessageType.VOICE_DATA:
        await this.handleVoiceData(payload, header);
        break;

      case MessageType.POSITION_UPDATE:
      case MessageType.AGENT_POSITIONS:
        await this.handlePositionUpdate(payload, header);
        break;
    }

    // Notify binary handlers
    this.binaryHandlers.forEach(handler => handler(data));
  }

  private async handleGraphUpdate(payload: ArrayBuffer, header: any): Promise<void> {
    const graphType = header.graphTypeFlag === GraphTypeFlag.ONTOLOGY
      ? 'ontology'
      : 'knowledge_graph';

    this.emit('graph-update', { graphType, data: payload });
  }

  private async handlePositionUpdate(payload: ArrayBuffer, header: any): Promise<void> {
    // Detect if this includes bots data
    const hasBotsData = this.detectBotsData(payload);

    if (hasBotsData) {
      this.emit('bots-position-update', payload);
    }

    // Update graph node positions
    await graphDataManager.updateNodePositions(payload);
  }

  sendMessage(type: string, data?: any): void {
    const message = JSON.stringify({ type, data });

    if (!this.isConnected || !this.socket) {
      this.messageQueue.push({
        type: 'text',
        data: message,
        timestamp: Date.now(),
        retries: 0
      });
      return;
    }

    this.socket.send(message);
  }

  sendBinary(data: ArrayBuffer): void {
    if (!this.isConnected || !this.socket) {
      this.messageQueue.push({
        type: 'binary',
        data,
        timestamp: Date.now(),
        retries: 0
      });
      return;
    }

    this.socket.send(data);
  }

  on(eventName: string, handler: EventHandler): () => void {
    if (!this.eventHandlers.has(eventName)) {
      this.eventHandlers.set(eventName, new Set());
    }
    this.eventHandlers.get(eventName)!.add(handler);

    return () => {
      this.eventHandlers.get(eventName)?.delete(handler);
    };
  }

  emit(eventName: string, data: any): void {
    this.eventHandlers.get(eventName)?.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error(`Error in event handler for ${eventName}:`, error);
      }
    });
  }

  close(): void {
    if (this.socket) {
      this.socket.close(1000, 'Normal closure');
      this.socket = null;
      this.isConnected = false;
    }
  }
}

export const webSocketService = WebSocketService.getInstance();
```

### Batch Queue for Performance

**File**: `/src/utils/BatchQueue.ts`

```typescript
export class NodePositionBatchQueue {
  private queue: BinaryNodeData[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private readonly batchSize: number;
  private readonly flushIntervalMs: number;
  private readonly processBatch: (batch: BinaryNodeData[]) => Promise<void>;

  constructor(config: BatchQueueConfig) {
    this.batchSize = config.batchSize || 100;
    this.flushIntervalMs = config.flushIntervalMs || 16; // ~60 FPS
    this.processBatch = config.processBatch;
  }

  enqueuePositionUpdate(node: BinaryNodeData, priority: number = 0): void {
    // Insert with priority (higher priority = earlier in queue)
    const insertIndex = this.queue.findIndex(n =>
      this.getPriority(n) < priority
    );

    if (insertIndex === -1) {
      this.queue.push(node);
    } else {
      this.queue.splice(insertIndex, 0, node);
    }

    // Schedule flush if not already scheduled
    if (!this.flushTimer) {
      this.flushTimer = setTimeout(() => {
        this.flush();
      }, this.flushIntervalMs);
    }

    // Flush immediately if batch size reached
    if (this.queue.length >= this.batchSize) {
      this.flush();
    }
  }

  async flush(): Promise<void> {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }

    if (this.queue.length === 0) {
      return;
    }

    const batch = this.queue.splice(0, this.batchSize);

    try {
      await this.processBatch(batch);
    } catch (error) {
      console.error('Failed to process batch:', error);
      // Re-queue failed items
      this.queue.unshift(...batch);
    }
  }

  private getPriority(node: BinaryNodeData): number {
    // Agent nodes get higher priority
    return isAgentNode(node.nodeId) ? 10 : 0;
  }

  getMetrics() {
    return {
      queueSize: this.queue.length,
      batchSize: this.batchSize,
      flushInterval: this.flushIntervalMs
    };
  }

  destroy(): void {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
    }
    this.queue = [];
  }
}
```

---

## WebXR Integration

### Quest 3 Detection

**File**: `/src/services/quest3AutoDetector.ts`

```typescript
export interface Quest3DetectionResult {
  isQuest3: boolean;
  supportsAR: boolean;
  supportsVR: boolean;
  shouldAutoStart: boolean;
  capabilities: XRCapabilities;
}

class Quest3AutoDetector {
  private detectionResult: Quest3DetectionResult | null = null;

  async detectQuest3Environment(): Promise<Quest3DetectionResult> {
    if (this.detectionResult) {
      return this.detectionResult;
    }

    const userAgent = navigator.userAgent;
    const isQuest3Browser =
      userAgent.includes('Quest 3') ||
      userAgent.includes('OculusBrowser') ||
      userAgent.toLowerCase().includes('meta quest');

    // Check WebXR support
    let supportsAR = false;
    let supportsVR = false;

    if ('xr' in navigator) {
      try {
        supportsAR = await navigator.xr.isSessionSupported('immersive-ar');
        supportsVR = await navigator.xr.isSessionSupported('immersive-vr');
      } catch (error) {
        console.error('WebXR detection failed:', error);
      }
    }

    this.detectionResult = {
      isQuest3: isQuest3Browser,
      supportsAR,
      supportsVR,
      shouldAutoStart: isQuest3Browser && supportsAR,
      capabilities: {
        handTracking: await this.detectHandTracking(),
        eyeTracking: false, // Not yet supported
        planeDetection: supportsAR,
        meshDetection: false
      }
    };

    return this.detectionResult;
  }

  private async detectHandTracking(): Promise<boolean> {
    if (!('xr' in navigator)) return false;

    try {
      const session = await navigator.xr.requestSession('immersive-vr', {
        optionalFeatures: ['hand-tracking']
      });

      const hasHandTracking = session.enabledFeatures?.includes('hand-tracking') || false;

      await session.end();
      return hasHandTracking;
    } catch (error) {
      return false;
    }
  }

  async autoStartQuest3AR(): Promise<boolean> {
    const result = await this.detectQuest3Environment();

    if (!result.shouldAutoStart) {
      return false;
    }

    try {
      // Trigger XR session
      await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking', 'plane-detection']
      });

      return true;
    } catch (error) {
      console.error('Failed to auto-start Quest 3 AR:', error);
      return false;
    }
  }

  resetDetection(): void {
    this.detectionResult = null;
  }
}

export const quest3AutoDetector = new Quest3AutoDetector();
```

### Quest 3 Integration Hook

**File**: `/src/hooks/useQuest3Integration.ts`

```typescript
export const useQuest3Integration = (options: Quest3IntegrationOptions = {}) => {
  const { enableAutoStart = true } = options;
  const [state, setState] = useState<Quest3IntegrationState>({
    isQuest3Detected: false,
    autoStartAttempted: false,
    autoStartSuccessful: false,
    detectionResult: null,
    error: null
  });

  // Detect Quest 3 on mount
  useEffect(() => {
    const detectQuest3 = async () => {
      try {
        const result = await quest3AutoDetector.detectQuest3Environment();
        setState(prev => ({
          ...prev,
          isQuest3Detected: result.isQuest3,
          detectionResult: result
        }));
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: error.message
        }));
      }
    };

    detectQuest3();
  }, []);

  // Auto-start AR if enabled
  useEffect(() => {
    const autoStartAR = async () => {
      if (!enableAutoStart ||
          state.autoStartAttempted ||
          !state.detectionResult?.shouldAutoStart) {
        return;
      }

      try {
        setState(prev => ({ ...prev, autoStartAttempted: true }));

        const success = await quest3AutoDetector.autoStartQuest3AR();

        setState(prev => ({ ...prev, autoStartSuccessful: success }));
      } catch (error) {
        setState(prev => ({
          ...prev,
          autoStartSuccessful: false,
          error: error.message
        }));
      }
    };

    autoStartAR();
  }, [enableAutoStart, state.detectionResult, state.autoStartAttempted]);

  return {
    ...state,
    shouldUseQuest3Layout: state.isQuest3Detected && state.autoStartSuccessful
  };
};
```

### Babylon.js XR Renderer

**File**: `/src/immersive/babylon/BabylonGraphRenderer.tsx`

```typescript
import * as BABYLON from '@babylonjs/core';
import { WebXRDefaultExperience } from '@babylonjs/core/XR';

export class BabylonGraphRenderer {
  private engine: BABYLON.Engine;
  private scene: BABYLON.Scene;
  private xrHelper: WebXRDefaultExperience | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.engine = new BABYLON.Engine(canvas, true, {
      preserveDrawingBuffer: true,
      stencil: true
    });

    this.scene = new BABYLON.Scene(this.engine);
    this.setupScene();
    this.setupXR();
  }

  private setupScene(): void {
    // Camera
    const camera = new BABYLON.FreeCamera(
      'camera',
      new BABYLON.Vector3(0, 5, -10),
      this.scene
    );
    camera.setTarget(BABYLON.Vector3.Zero());

    // Lighting
    const light = new BABYLON.HemisphericLight(
      'light',
      new BABYLON.Vector3(0, 1, 0),
      this.scene
    );
    light.intensity = 0.7;

    // Environment
    const environment = this.scene.createDefaultEnvironment({
      createGround: false,
      createSkybox: false
    });
  }

  private async setupXR(): Promise<void> {
    try {
      this.xrHelper = await this.scene.createDefaultXRExperienceAsync({
        uiOptions: {
          sessionMode: 'immersive-ar',
          referenceSpaceType: 'local-floor'
        },
        optionalFeatures: true
      });

      // Hand tracking
      const handTracking = this.xrHelper.baseExperience.featuresManager.enableFeature(
        BABYLON.WebXRFeatureName.HAND_TRACKING,
        'latest',
        { xrInput: this.xrHelper.input }
      );

      // Plane detection
      const planeDetector = this.xrHelper.baseExperience.featuresManager.enableFeature(
        BABYLON.WebXRFeatureName.PLANE_DETECTION,
        'latest',
        { worldTracking: this.xrHelper.baseExperience.sessionManager }
      );

      // Teleportation
      const teleportation = this.xrHelper.baseExperience.featuresManager.enableFeature(
        BABYLON.WebXRFeatureName.TELEPORTATION,
        'stable',
        {
          xrInput: this.xrHelper.input,
          floorMeshes: []
        }
      );

    } catch (error) {
      console.error('XR setup failed:', error);
    }
  }

  renderGraph(graphData: GraphData): void {
    // Clear existing meshes
    this.scene.meshes.forEach(mesh => {
      if (mesh.name.startsWith('node_') || mesh.name.startsWith('edge_')) {
        mesh.dispose();
      }
    });

    // Render nodes
    graphData.nodes.forEach(node => {
      const sphere = BABYLON.MeshBuilder.CreateSphere(
        `node_${node.id}`,
        { diameter: node.size * 0.1 },
        this.scene
      );

      sphere.position = new BABYLON.Vector3(
        node.position.x * 0.01,
        node.position.y * 0.01,
        node.position.z * 0.01
      );

      const material = new BABYLON.StandardMaterial(
        `mat_${node.id}`,
        this.scene
      );
      material.diffuseColor = BABYLON.Color3.FromHexString(node.color);
      material.emissiveColor = material.diffuseColor.scale(0.2);
      sphere.material = material;
    });

    // Render edges
    graphData.edges.forEach(edge => {
      const points = [
        new BABYLON.Vector3(
          edge.source.x * 0.01,
          edge.source.y * 0.01,
          edge.source.z * 0.01
        ),
        new BABYLON.Vector3(
          edge.target.x * 0.01,
          edge.target.y * 0.01,
          edge.target.z * 0.01
        )
      ];

      const line = BABYLON.MeshBuilder.CreateLines(
        `edge_${edge.id}`,
        { points },
        this.scene
      );
      line.color = BABYLON.Color3.FromHexString(edge.color);
    });
  }

  startRenderLoop(): void {
    this.engine.runRenderLoop(() => {
      this.scene.render();
    });
  }

  dispose(): void {
    this.scene.dispose();
    this.engine.dispose();
  }
}
```

---

## 3D Visualization

### Three.js Rendering Pipeline

#### GraphManager Component

**File**: `/src/features/graph/components/GraphManager.tsx`

```typescript
import { useRef, useEffect, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface GraphManagerProps {
  graphData: GraphData;
}

export const GraphManager: React.FC<GraphManagerProps> = ({ graphData }) => {
  const { scene } = useThree();
  const nodesGroupRef = useRef<THREE.Group>(null);
  const edgesGroupRef = useRef<THREE.Group>(null);

  // Create node geometries and materials
  const nodeMeshes = useMemo(() => {
    return graphData.nodes.map(node => {
      const geometry = new THREE.SphereGeometry(node.size, 32, 32);
      const material = new THREE.MeshStandardMaterial({
        color: node.color,
        emissive: node.selected ? node.color : 0x000000,
        emissiveIntensity: node.selected ? 0.5 : 0,
        metalness: 0.3,
        roughness: 0.7
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(node.position.x, node.position.y, node.position.z);
      mesh.userData = { nodeId: node.id, type: 'node' };

      return mesh;
    });
  }, [graphData.nodes]);

  // Create edge geometries
  const edgeMeshes = useMemo(() => {
    return graphData.edges.map(edge => {
      const sourceNode = graphData.nodes.find(n => n.id === edge.source);
      const targetNode = graphData.nodes.find(n => n.id === edge.target);

      if (!sourceNode || !targetNode) return null;

      const points = [
        new THREE.Vector3(sourceNode.position.x, sourceNode.position.y, sourceNode.position.z),
        new THREE.Vector3(targetNode.position.x, targetNode.position.y, targetNode.position.z)
      ];

      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color: edge.color,
        opacity: 0.6,
        transparent: true
      });

      const line = new THREE.Line(geometry, material);
      line.userData = { edgeId: edge.id, type: 'edge' };

      return line;
    }).filter(Boolean);
  }, [graphData.edges, graphData.nodes]);

  // Update scene
  useEffect(() => {
    if (!nodesGroupRef.current || !edgesGroupRef.current) return;

    // Clear existing meshes
    nodesGroupRef.current.clear();
    edgesGroupRef.current.clear();

    // Add new meshes
    nodeMeshes.forEach(mesh => nodesGroupRef.current!.add(mesh));
    edgeMeshes.forEach(mesh => edgesGroupRef.current!.add(mesh));

    return () => {
      nodeMeshes.forEach(mesh => {
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      });
      edgeMeshes.forEach(mesh => {
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      });
    };
  }, [nodeMeshes, edgeMeshes]);

  // Animation loop
  useFrame((state, delta) => {
    if (nodesGroupRef.current) {
      // Update node positions from physics
      nodesGroupRef.current.children.forEach(mesh => {
        const nodeId = mesh.userData.nodeId;
        const node = graphData.nodes.find(n => n.id === nodeId);
        if (node) {
          mesh.position.lerp(
            new THREE.Vector3(node.position.x, node.position.y, node.position.z),
            0.1
          );
        }
      });
    }

    if (edgesGroupRef.current) {
      // Update edge positions
      edgesGroupRef.current.children.forEach(line => {
        const edgeId = line.userData.edgeId;
        const edge = graphData.edges.find(e => e.id === edgeId);
        if (edge) {
          const sourceNode = graphData.nodes.find(n => n.id === edge.source);
          const targetNode = graphData.nodes.find(n => n.id === edge.target);

          if (sourceNode && targetNode) {
            const positions = (line as THREE.Line).geometry.attributes.position;
            positions.setXYZ(0, sourceNode.position.x, sourceNode.position.y, sourceNode.position.z);
            positions.setXYZ(1, targetNode.position.x, targetNode.position.y, targetNode.position.z);
            positions.needsUpdate = true;
          }
        }
      });
    }
  });

  return (
    <>
      <group ref={nodesGroupRef} />
      <group ref={edgesGroupRef} />
    </>
  );
};
```

### Selective Bloom Post-Processing

**File**: `/src/rendering/SelectiveBloom.tsx`

```typescript
import { useMemo } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import * as THREE from 'three';

interface SelectiveBloomProps {
  enabled: boolean;
  strength?: number;
  radius?: number;
  threshold?: number;
}

export const SelectiveBloom: React.FC<SelectiveBloomProps> = ({
  enabled,
  strength = 1.5,
  radius = 0.4,
  threshold = 0.85
}) => {
  const { gl, scene, camera, size } = useThree();

  const composer = useMemo(() => {
    const comp = new EffectComposer(gl);
    comp.setSize(size.width, size.height);

    const renderPass = new RenderPass(scene, camera);
    comp.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(size.width, size.height),
      strength,
      radius,
      threshold
    );
    comp.addPass(bloomPass);

    return comp;
  }, [gl, scene, camera, size, strength, radius, threshold]);

  useFrame(() => {
    if (enabled) {
      composer.render();
    }
  }, 1); // Render after scene

  return null;
};
```

### GPU Physics Integration

**File**: `/src/features/graph/utils/gpuPhysics.ts`

```typescript
export class GPUPhysicsEngine {
  private device: GPUDevice | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private buffers: Map<string, GPUBuffer> = new Map();

  async initialize(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No GPU adapter found');
    }

    this.device = await adapter.requestDevice();

    await this.createComputePipeline();
  }

  private async createComputePipeline(): Promise<void> {
    const shaderModule = this.device!.createShaderModule({
      code: `
        struct Node {
          position: vec3<f32>,
          velocity: vec3<f32>,
          mass: f32,
          fixed: u32
        };

        @group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
        @group(0) @binding(1) var<uniform> params: PhysicsParams;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let i = global_id.x;
          if (i >= arrayLength(&nodes)) {
            return;
          }

          var node = nodes[i];
          if (node.fixed != 0u) {
            return;
          }

          var force = vec3<f32>(0.0, 0.0, 0.0);

          // Repulsion from all other nodes
          for (var j = 0u; j < arrayLength(&nodes); j++) {
            if (i == j) {
              continue;
            }

            let other = nodes[j];
            let delta = node.position - other.position;
            let distance = length(delta);

            if (distance > 0.01) {
              let repulsion = params.repelK / (distance * distance);
              force += normalize(delta) * repulsion;
            }
          }

          // Spring force for connected edges
          // (edge data would be in separate buffer)

          // Apply forces
          node.velocity += force * params.dt;
          node.velocity *= params.damping;

          // Clamp velocity
          let speed = length(node.velocity);
          if (speed > params.maxVelocity) {
            node.velocity = normalize(node.velocity) * params.maxVelocity;
          }

          // Update position
          node.position += node.velocity * params.dt;

          // Boundary constraints
          if (params.enableBounds != 0u) {
            let bounds = params.boundsSize;
            node.position = clamp(node.position, vec3(-bounds), vec3(bounds));
          }

          nodes[i] = node;
        }
      `
    });

    this.computePipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
  }

  updatePhysics(nodes: Node[], edges: Edge[], params: PhysicsParams): void {
    if (!this.device || !this.computePipeline) {
      return;
    }

    // Create/update node buffer
    const nodeBuffer = this.createNodeBuffer(nodes);
    const paramsBuffer = this.createParamsBuffer(params);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: nodeBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } }
      ]
    });

    // Submit compute pass
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(this.computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(nodes.length / 64));
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Read back results
    this.readNodePositions(nodeBuffer, nodes);
  }

  private createNodeBuffer(nodes: Node[]): GPUBuffer {
    const data = new Float32Array(nodes.length * 8); // 8 floats per node

    nodes.forEach((node, i) => {
      const offset = i * 8;
      data[offset + 0] = node.position.x;
      data[offset + 1] = node.position.y;
      data[offset + 2] = node.position.z;
      data[offset + 3] = node.velocity.x;
      data[offset + 4] = node.velocity.y;
      data[offset + 5] = node.velocity.z;
      data[offset + 6] = node.mass;
      data[offset + 7] = node.fixed ? 1 : 0;
    });

    const buffer = this.device!.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    return buffer;
  }

  private async readNodePositions(buffer: GPUBuffer, nodes: Node[]): Promise<void> {
    const readBuffer = this.device!.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = this.device!.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
    this.device!.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readBuffer.getMappedRange());

    nodes.forEach((node, i) => {
      const offset = i * 8;
      node.position.x = data[offset + 0];
      node.position.y = data[offset + 1];
      node.position.z = data[offset + 2];
      node.velocity.x = data[offset + 3];
      node.velocity.y = data[offset + 4];
      node.velocity.z = data[offset + 5];
    });

    readBuffer.unmap();
  }

  dispose(): void {
    this.buffers.forEach(buffer => buffer.destroy());
    this.buffers.clear();
  }
}
```

---

## Type System

### Generated Types

Types are automatically generated from the Rust backend using `cargo run --bin generate_types`.

**File**: `/src/types/generated/settings.ts`

```typescript
// Auto-generated from Rust - DO NOT EDIT

export interface Settings {
  system: SystemSettings;
  visualisation: VisualisationSettings;
  auth: AuthSettings;
  xr: XRSettings;
  developer?: DeveloperSettings;
}

export interface SystemSettings {
  debug: DebugSettings;
  websocket: WebSocketSettings;
  customBackendUrl?: string;
}

export interface DebugSettings {
  enabled: boolean;
  enablePerformanceDebug: boolean;
  enableDataDebug: boolean;
}

export interface WebSocketSettings {
  updateRate: number;
  reconnectAttempts: number;
  reconnectDelay: number;
}

export interface VisualisationSettings {
  rendering: RenderingSettings;
  graphs: GraphsSettings;
  glow?: GlowSettings;
  hologram?: HologramSettings;
  bloom?: BloomSettings;
}

export interface RenderingSettings {
  context: 'webgl2' | 'webgpu';
  ambientLightIntensity: number;
  directionalLightIntensity: number;
  backgroundColor: string;
  enableShadows: boolean;
  enableAntialiasing: boolean;
  enableAmbientOcclusion: boolean;
  environmentIntensity: number;
  shadowMapSize?: number;
  shadowBias?: number;
}

export interface GraphsSettings {
  mode: 'knowledge_graph' | 'ontology';
  logseq: GraphTypeSettings;
  visionflow: GraphTypeSettings;
}

export interface GraphTypeSettings {
  nodes: NodeSettings;
  edges: EdgeSettings;
  labels: LabelSettings;
  physics: PhysicsSettings;
}

export interface NodeSettings {
  defaultSize: number;
  defaultColor: string;
  selectedColor: string;
  hoverColor: string;
  enableHologram: boolean;
}

export interface EdgeSettings {
  defaultColor: string;
  selectedColor: string;
  opacity: number;
  width: number;
}

export interface LabelSettings {
  enabled: boolean;
  fontSize: number;
  fontColor: string;
  backgroundColor: string;
}

export interface PhysicsSettings {
  enabled: boolean;
  springK: number;
  repelK: number;
  attractionK: number;
  gravity: number;
  damping: number;
  maxVelocity: number;
  dt: number;
  iterations: number;
  enableBounds: boolean;
  boundsSize: number;
  warmupIterations: number;
  coolingRate: number;
}

export interface XRSettings {
  enabled: boolean;
  mode: 'vr' | 'ar' | 'mr';
  enableHandTracking: boolean;
  enableHaptics: boolean;
  quality: 'low' | 'medium' | 'high' | 'ultra';
}

export interface AuthSettings {
  enabled: boolean;
  required: boolean;
  provider: 'nostr' | 'oauth' | 'jwt';
}
```

### Custom Types

**File**: `/src/types/websocketTypes.ts`

```typescript
export interface WebSocketMessage {
  type: string;
  data?: any;
  error?: WebSocketError;
  timestamp?: number;
}

export interface WebSocketError {
  code: string;
  message: string;
  category: 'validation' | 'server' | 'protocol' | 'auth' | 'rate_limit';
  retryable: boolean;
  retryAfter?: number;
  affectedPaths?: string[];
}

export interface WebSocketConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'failed';
  reconnectAttempts: number;
  lastConnected?: number;
  lastError?: string;
  serverFeatures: string[];
}

export interface WebSocketConfig {
  reconnect: {
    maxAttempts: number;
    baseDelay: number;
    maxDelay: number;
    backoffFactor: number;
  };
  heartbeat: {
    interval: number;
    timeout: number;
  };
  compression: boolean;
  binaryProtocol: boolean;
}

export interface WebSocketStatistics {
  messagesReceived: number;
  messagesSent: number;
  bytesReceived: number;
  bytesSent: number;
  connectionTime: number;
  reconnections: number;
  averageLatency: number;
  messagesByType: Record<string, number>;
  errors: number;
  lastActivity: number;
}

export type MessageHandler = (message: WebSocketMessage) => void;
export type BinaryMessageHandler = (data: ArrayBuffer) => void;
export type EventHandler = (data: any) => void;

export interface Subscription {
  id: string;
  filters: SubscriptionFilters;
  handler: MessageHandler;
}

export interface SubscriptionFilters {
  messageTypes?: string[];
  graphType?: 'knowledge_graph' | 'ontology';
  priority?: number;
}
```

### Utility Types

**File**: `/src/types/utility.ts`

```typescript
// Deep partial type for nested objects
export type DeepPartial<T> = T extends object
  ? { [P in keyof T]?: DeepPartial<T[P]> }
  : T;

// Deep required type
export type DeepRequired<T> = T extends object
  ? { [P in keyof T]-?: DeepRequired<T[P]> }
  : T;

// Extract nested property type
export type NestedProperty<T, K extends string> = K extends keyof T
  ? T[K]
  : K extends `${infer P}.${infer R}`
  ? P extends keyof T
    ? NestedProperty<T[P], R>
    : never
  : never;

// Type-safe path notation
export type SettingsPath =
  | 'system.debug.enabled'
  | 'system.websocket.updateRate'
  | 'visualisation.rendering.context'
  | 'visualisation.graphs.mode'
  | 'xr.enabled'
  | string;

// Union to intersection
export type UnionToIntersection<U> = (
  U extends any ? (k: U) => void : never
) extends (k: infer I) => void
  ? I
  : never;

// Ensure type
export type Ensure<T, K extends keyof T> = T & Required<Pick<T, K>>;

// Mutable type
export type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

// NonNullable keys
export type NonNullableKeys<T> = {
  [K in keyof T]: T[K] extends null | undefined ? never : K;
}[keyof T];
```

---

## Service Layer

### Services Overview

1. **WebSocketService**: Real-time communication
2. **NostrAuthService**: Authentication
3. **GraphDataManager**: Graph state management
4. **RemoteLogger**: Remote debugging
5. **Quest3AutoDetector**: XR detection
6. **BotsWebSocketIntegration**: Multi-agent communication

### Nostr Authentication Service

**File**: `/src/services/nostrAuthService.ts`

```typescript
import { generatePrivateKey, getPublicKey, signEvent, verifySignature } from 'nostr-tools';

interface NostrUser {
  pubkey: string;
  privateKey?: string;
  isPowerUser: boolean;
}

class NostrAuthService {
  private user: NostrUser | null = null;
  private initialized = false;
  private listeners: Set<(authenticated: boolean) => void> = new Set();

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Check for existing session
    const sessionToken = localStorage.getItem('nostr_session_token');
    if (sessionToken) {
      try {
        await this.restoreSession(sessionToken);
      } catch (error) {
        console.error('Failed to restore session:', error);
        localStorage.removeItem('nostr_session_token');
      }
    }

    this.initialized = true;
  }

  async login(privateKey?: string): Promise<void> {
    const sk = privateKey || generatePrivateKey();
    const pubkey = getPublicKey(sk);

    // Create auth event
    const event = {
      kind: 22242, // NIP-42 auth
      created_at: Math.floor(Date.now() / 1000),
      tags: [],
      content: 'VisionFlow Authentication',
      pubkey
    };

    const signedEvent = await signEvent(event, sk);

    // Send to backend
    const response = await fetch('/api/auth/nostr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event: signedEvent })
    });

    if (!response.ok) {
      throw new Error('Authentication failed');
    }

    const { token, isPowerUser } = await response.json();

    this.user = { pubkey, privateKey: sk, isPowerUser };
    localStorage.setItem('nostr_session_token', token);

    this.notifyListeners(true);
  }

  async restoreSession(token: string): Promise<void> {
    const response = await fetch('/api/auth/verify', {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (!response.ok) {
      throw new Error('Session invalid');
    }

    const { pubkey, isPowerUser } = await response.json();
    this.user = { pubkey, isPowerUser };

    this.notifyListeners(true);
  }

  logout(): void {
    this.user = null;
    localStorage.removeItem('nostr_session_token');
    this.notifyListeners(false);
  }

  isAuthenticated(): boolean {
    return this.user !== null;
  }

  getCurrentUser(): NostrUser | null {
    return this.user;
  }

  async getAuthToken(): Promise<string | null> {
    return localStorage.getItem('nostr_session_token');
  }

  onAuthStateChange(callback: (authenticated: boolean) => void): () => void {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  private notifyListeners(authenticated: boolean): void {
    this.listeners.forEach(callback => callback(authenticated));
  }
}

export const nostrAuth = new NostrAuthService();
```

### Graph Data Manager

**File**: `/src/features/graph/managers/graphDataManager.ts`

```typescript
export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

export interface Node {
  id: number;
  label: string;
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  color: string;
  size: number;
  mass: number;
  fixed: boolean;
  selected: boolean;
  metadata?: Record<string, any>;
}

export interface Edge {
  id: number;
  source: number;
  target: number;
  color: string;
  width: number;
  metadata?: Record<string, any>;
}

class GraphDataManager {
  private static instance: GraphDataManager;
  private graphData: GraphData = { nodes: [], edges: [] };
  private graphType: 'logseq' | 'visionflow' = 'logseq';
  private listeners: Set<(data: GraphData) => void> = new Set();

  private constructor() {
    this.setupWebSocketListeners();
  }

  static getInstance(): GraphDataManager {
    if (!GraphDataManager.instance) {
      GraphDataManager.instance = new GraphDataManager();
    }
    return GraphDataManager.instance;
  }

  private setupWebSocketListeners(): void {
    webSocketService.on('graph-update', async (payload) => {
      await this.handleGraphUpdate(payload.data);
    });
  }

  private async handleGraphUpdate(data: ArrayBuffer): Promise<void> {
    // Parse binary graph data
    const view = new DataView(data);
    let offset = 0;

    // Read node count
    const nodeCount = view.getUint32(offset, true);
    offset += 4;

    const nodes: Node[] = [];
    for (let i = 0; i < nodeCount; i++) {
      const id = view.getUint32(offset, true);
      offset += 4;

      const x = view.getFloat32(offset, true);
      offset += 4;
      const y = view.getFloat32(offset, true);
      offset += 4;
      const z = view.getFloat32(offset, true);
      offset += 4;

      // Read label length
      const labelLength = view.getUint8(offset);
      offset += 1;

      // Read label
      const labelBytes = new Uint8Array(data, offset, labelLength);
      const label = new TextDecoder().decode(labelBytes);
      offset += labelLength;

      nodes.push({
        id,
        label,
        position: { x, y, z },
        velocity: { x: 0, y: 0, z: 0 },
        color: '#ffffff',
        size: 1.0,
        mass: 1.0,
        fixed: false,
        selected: false
      });
    }

    // Read edge count
    const edgeCount = view.getUint32(offset, true);
    offset += 4;

    const edges: Edge[] = [];
    for (let i = 0; i < edgeCount; i++) {
      const id = view.getUint32(offset, true);
      offset += 4;

      const source = view.getUint32(offset, true);
      offset += 4;

      const target = view.getUint32(offset, true);
      offset += 4;

      edges.push({
        id,
        source,
        target,
        color: '#888888',
        width: 1.0
      });
    }

    this.graphData = { nodes, edges };
    this.notifyListeners();
  }

  async updateNodePositions(data: ArrayBuffer): Promise<void> {
    const positions = parseBinaryNodeData(data);

    positions.forEach(({ nodeId, position, velocity }) => {
      const node = this.graphData.nodes.find(n => n.id === nodeId);
      if (node) {
        node.position = position;
        node.velocity = velocity;
      }
    });

    this.notifyListeners();
  }

  async getGraphData(): Promise<GraphData> {
    return { ...this.graphData };
  }

  getGraphType(): 'logseq' | 'visionflow' {
    return this.graphType;
  }

  setGraphType(type: 'logseq' | 'visionflow'): void {
    this.graphType = type;
  }

  onGraphDataChange(callback: (data: GraphData) => void): () => void {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  private notifyListeners(): void {
    this.listeners.forEach(callback => {
      try {
        callback({ ...this.graphData });
      } catch (error) {
        console.error('Error in graph data listener:', error);
      }
    });
  }
}

export const graphDataManager = GraphDataManager.getInstance();
```

---

## Feature Modules

### Bots Feature

The bots feature enables real-time visualization and interaction with autonomous agents.

**Directory Structure:**
```
features/bots/
├── components/
│   ├── BotsVisualization.tsx
│   ├── BotsControlPanel.tsx
│   ├── AgentDetailPanel.tsx
│   └── ActivityLogPanel.tsx
├── contexts/
│   └── BotsDataContext.tsx
├── hooks/
│   ├── useBotsWebSocketIntegration.ts
│   └── useBotsData.ts
├── services/
│   └── BotsWebSocketIntegration.ts
├── types/
│   └── index.ts
└── index.ts
```

**BotsDataContext:**

```typescript
interface BotsDataContextValue {
  agents: Agent[];
  selectedAgent: Agent | null;
  selectAgent: (agentId: string) => void;
  activityLog: ActivityLogEntry[];
  isLoading: boolean;
}

export const BotsDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [activityLog, setActivityLog] = useState<ActivityLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Fetch initial agent data
    const loadAgents = async () => {
      try {
        const response = await fetch('/api/bots/agents');
        const data = await response.json();
        setAgents(data);
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to load agents:', error);
        setIsLoading(false);
      }
    };

    loadAgents();

    // Subscribe to WebSocket updates
    const unsubscribe = webSocketService.on('bots-position-update', (data) => {
      const positions = parseBinaryNodeData(data);
      setAgents(prevAgents =>
        prevAgents.map(agent => {
          const position = positions.find(p => p.nodeId === agent.nodeId);
          if (position) {
            return { ...agent, position: position.position };
          }
          return agent;
        })
      );
    });

    return unsubscribe;
  }, []);

  const selectAgent = (agentId: string) => {
    const agent = agents.find(a => a.id === agentId);
    setSelectedAgent(agent || null);
  };

  return (
    <BotsDataContext.Provider
      value={{
        agents,
        selectedAgent,
        selectAgent,
        activityLog,
        isLoading
      }}
    >
      {children}
    </BotsDataContext.Provider>
  );
};
```

### Command Palette Feature

Global command palette for quick actions.

**Command Registry:**

```typescript
export interface Command {
  id: string;
  label: string;
  description?: string;
  keywords?: string[];
  icon?: string;
  shortcut?: string;
  handler: (context: CommandContext) => void | Promise<void>;
  condition?: (context: CommandContext) => boolean;
  category?: string;
}

export class CommandRegistry {
  private static instance: CommandRegistry;
  private commands: Map<string, Command> = new Map();
  private listeners: Set<() => void> = new Set();

  static getInstance(): CommandRegistry {
    if (!CommandRegistry.instance) {
      CommandRegistry.instance = new CommandRegistry();
    }
    return CommandRegistry.instance;
  }

  register(command: Command): () => void {
    this.commands.set(command.id, command);
    this.notifyListeners();

    return () => {
      this.commands.delete(command.id);
      this.notifyListeners();
    };
  }

  registerMultiple(commands: Command[]): () => void {
    commands.forEach(cmd => this.commands.set(cmd.id, cmd));
    this.notifyListeners();

    return () => {
      commands.forEach(cmd => this.commands.delete(cmd.id));
      this.notifyListeners();
    };
  }

  execute(commandId: string, context: CommandContext): void {
    const command = this.commands.get(commandId);
    if (!command) {
      throw new Error(`Command not found: ${commandId}`);
    }

    if (command.condition && !command.condition(context)) {
      throw new Error(`Command condition not met: ${commandId}`);
    }

    command.handler(context);
  }

  search(query: string): Command[] {
    const lowerQuery = query.toLowerCase();

    return Array.from(this.commands.values())
      .filter(cmd => {
        const labelMatch = cmd.label.toLowerCase().includes(lowerQuery);
        const descMatch = cmd.description?.toLowerCase().includes(lowerQuery);
        const keywordMatch = cmd.keywords?.some(k => k.toLowerCase().includes(lowerQuery));

        return labelMatch || descMatch || keywordMatch;
      })
      .sort((a, b) => a.label.localeCompare(b.label));
  }

  getAll(): Command[] {
    return Array.from(this.commands.values());
  }

  onChange(callback: () => void): () => void {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  private notifyListeners(): void {
    this.listeners.forEach(callback => callback());
  }
}

export const commandRegistry = CommandRegistry.getInstance();
```

**Default Commands:**

```typescript
export function initializeCommandPalette() {
  commandRegistry.registerMultiple([
    {
      id: 'graph.toggle-physics',
      label: 'Toggle Physics',
      description: 'Enable or disable graph physics simulation',
      keywords: ['physics', 'simulation', 'force'],
      icon: 'Activity',
      shortcut: 'Ctrl+P',
      handler: () => {
        const settings = useSettingsStore.getState();
        const currentValue = settings.get('visualisation.graphs.logseq.physics.enabled');
        settings.set('visualisation.graphs.logseq.physics.enabled', !currentValue);
      },
      category: 'Graph'
    },
    {
      id: 'view.toggle-xr',
      label: 'Toggle XR Mode',
      description: 'Enter or exit XR immersive mode',
      keywords: ['xr', 'vr', 'ar', 'immersive', 'quest'],
      icon: 'Eye',
      shortcut: 'Ctrl+X',
      handler: async () => {
        const xrEnabled = useSettingsStore.getState().get('xr.enabled');
        if (!xrEnabled) {
          await navigator.xr?.requestSession('immersive-ar');
        } else {
          // Exit XR
        }
      },
      condition: () => {
        return 'xr' in navigator;
      },
      category: 'View'
    },
    {
      id: 'settings.open',
      label: 'Open Settings',
      description: 'Open the settings panel',
      keywords: ['settings', 'preferences', 'config'],
      icon: 'Settings',
      shortcut: 'Ctrl+,',
      handler: () => {
        window.dispatchEvent(new CustomEvent('open-settings'));
      },
      category: 'System'
    }
  ]);
}
```

---

## Build & Deployment

### Vite Build Configuration

**File**: `vite.config.ts`

```typescript
export default defineConfig({
  plugins: [react()],

  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    'process.env': '({})'
  },

  optimizeDeps: {
    include: ['@getalby/sdk', 'three', '@babylonjs/core']
  },

  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,

    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'three-vendor': ['three', '@react-three/fiber', '@react-three/drei'],
          'babylon-vendor': ['@babylonjs/core', '@babylonjs/loaders'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-slider']
        }
      }
    },

    chunkSizeWarningLimit: 1000,

    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },

  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,

    hmr: {
      clientPort: 3001,
      path: '/vite-hmr'
    },

    watch: {
      usePolling: true,
      interval: 1000
    },

    cors: true,

    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    },

    proxy: {
      '/api': {
        target: 'http://visionflow_container:4000',
        changeOrigin: true,
        secure: false
      },
      '/ws': {
        target: 'ws://visionflow_container:4000',
        ws: true,
        changeOrigin: true
      }
    }
  }
});
```

### Build Scripts

**File**: `package.json`

```json
{
  "scripts": {
    "dev": "vite",
    "build": "npm run types:generate && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx",
    "types:generate": "cd .. && cargo run --bin generate_types",
    "types:watch": "cd .. && cargo watch -x 'run --bin generate_types'",
    "prebuild": "npm run types:generate",
    "security:check": "node scripts/block-test-packages.cjs && npm audit"
  }
}
```

### Docker Deployment

**File**: `Dockerfile`

```dockerfile
# Build stage
FROM node:20-alpine AS builder

WORKDIR /app/client

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/client/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Add security headers
RUN echo 'add_header X-Frame-Options "SAMEORIGIN" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header X-Content-Type-Options "nosniff" always;' >> /etc/nginx/conf.d/security.conf && \
    echo 'add_header X-XSS-Protection "1; mode=block" always;' >> /etc/nginx/conf.d/security.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

---

## Performance Optimization

### Code Splitting

```typescript
// Lazy load features
const SettingsPanel = lazy(() => import('./features/settings/components/SettingsPanel'));
const CommandPalette = lazy(() => import('./features/command-palette/components/CommandPalette'));
const BotsVisualization = lazy(() => import('./features/bots/components/BotsVisualization'));

// Use with Suspense
<Suspense fallback={<LoadingSpinner />}>
  <SettingsPanel />
</Suspense>
```

### Memoization

```typescript
// Memoize expensive computations
const computedNodePositions = useMemo(() => {
  return graphData.nodes.map(node => ({
    id: node.id,
    x: node.position.x * scale,
    y: node.position.y * scale,
    z: node.position.z * scale
  }));
}, [graphData.nodes, scale]);

// Memoize callbacks
const handleNodeClick = useCallback((nodeId: number) => {
  setSelectedNode(nodeId);
  graphDataManager.selectNode(nodeId);
}, []);

// Memoize components
const NodeMesh = React.memo(({ node, selected }) => {
  return <mesh {...nodeProps} />;
}, (prev, next) => {
  return prev.node.id === next.node.id &&
         prev.selected === next.selected;
});
```

### Web Workers

**File**: `/src/features/graph/workers/physics.worker.ts`

```typescript
// Physics computation in Web Worker
self.onmessage = (e: MessageEvent) => {
  const { nodes, edges, params } = e.data;

  // Compute forces
  for (let i = 0; i < params.iterations; i++) {
    nodes.forEach((node, idx) => {
      if (node.fixed) return;

      let fx = 0, fy = 0, fz = 0;

      // Repulsion
      nodes.forEach((other, otherIdx) => {
        if (idx === otherIdx) return;

        const dx = node.x - other.x;
        const dy = node.y - other.y;
        const dz = node.z - other.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;

        const force = params.repelK / (dist * dist);
        fx += (dx / dist) * force;
        fy += (dy / dist) * force;
        fz += (dz / dist) * force;
      });

      // Spring forces
      edges.forEach(edge => {
        if (edge.source === idx || edge.target === idx) {
          const other = nodes[edge.source === idx ? edge.target : edge.source];
          const dx = other.x - node.x;
          const dy = other.y - node.y;
          const dz = other.z - node.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;

          const force = params.springK * (dist - params.restLength);
          fx += (dx / dist) * force;
          fy += (dy / dist) * force;
          fz += (dz / dist) * force;
        }
      });

      // Update velocity and position
      node.vx = (node.vx + fx * params.dt) * params.damping;
      node.vy = (node.vy + fy * params.dt) * params.damping;
      node.vz = (node.vz + fz * params.dt) * params.damping;

      node.x += node.vx * params.dt;
      node.y += node.vy * params.dt;
      node.z += node.vz * params.dt;
    });
  }

  self.postMessage({ nodes });
};
```

### Virtual Scrolling

```typescript
import { FixedSizeList } from 'react-window';

const VirtualizedNodeList: React.FC<{ nodes: Node[] }> = ({ nodes }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <NodeItem node={nodes[index]} />
    </div>
  );

  return (
    <FixedSizeList
      height={600}
      itemCount={nodes.length}
      itemSize={50}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
};
```

---

## Testing Strategy

### Unit Testing

```typescript
import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSettingsStore } from '../store/settingsStore';

describe('useSettingsStore', () => {
  it('should initialize with default settings', async () => {
    const { result } = renderHook(() => useSettingsStore());

    await act(async () => {
      await result.current.initialize();
    });

    expect(result.current.initialized).toBe(true);
  });

  it('should update settings', async () => {
    const { result } = renderHook(() => useSettingsStore());

    await act(async () => {
      await result.current.initialize();
      result.current.set('system.debug.enabled', true);
    });

    expect(result.current.get('system.debug.enabled')).toBe(true);
  });
});
```

### Component Testing

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { SettingsPanel } from './SettingsPanel';

describe('SettingsPanel', () => {
  it('should render settings sections', () => {
    render(<SettingsPanel />);

    expect(screen.getByText('System')).toBeInTheDocument();
    expect(screen.getByText('Rendering')).toBeInTheDocument();
  });

  it('should toggle physics setting', () => {
    render(<SettingsPanel />);

    const toggle = screen.getByLabelText('Enable Physics');
    fireEvent.click(toggle);

    expect(toggle).toBeChecked();
  });
});
```

### Integration Testing

```typescript
describe('Graph Visualization', () => {
  it('should load and display graph data', async () => {
    const mockData = {
      nodes: [{ id: 1, label: 'Node 1', position: { x: 0, y: 0, z: 0 } }],
      edges: []
    };

    vi.spyOn(graphDataManager, 'getGraphData').mockResolvedValue(mockData);

    render(<GraphCanvas />);

    await waitFor(() => {
      expect(screen.getByText('Nodes: 1')).toBeInTheDocument();
    });
  });
});
```

---

## Security

### Content Security Policy

```html
<meta http-equiv="Content-Security-Policy"
  content="default-src 'self';
           script-src 'self' 'unsafe-eval';
           style-src 'self' 'unsafe-inline';
           connect-src 'self' ws: wss:;
           img-src 'self' data: blob:;
           font-src 'self' data:;">
```

### Input Validation

```typescript
export function validateSettingValue(path: string, value: any): boolean {
  const validators: Record<string, (val: any) => boolean> = {
    'visualisation.rendering.ambientLightIntensity': (val) =>
      typeof val === 'number' && val >= 0 && val <= 2,

    'system.websocket.updateRate': (val) =>
      typeof val === 'number' && val >= 16 && val <= 1000,

    'visualisation.graphs.logseq.physics.damping': (val) =>
      typeof val === 'number' && val >= 0 && val <= 1
  };

  const validator = validators[path];
  return validator ? validator(value) : true;
}
```

### XSS Prevention

```typescript
// Sanitize user input
import DOMPurify from 'dompurify';

function renderUserContent(html: string) {
  const clean = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a'],
    ALLOWED_ATTR: ['href']
  });

  return <div dangerouslySetInnerHTML={{ __html: clean }} />;
}
```

---

## Best Practices

### Component Design

1. **Single Responsibility**: Each component should do one thing well
2. **Composition over Inheritance**: Use composition for reusability
3. **Props Validation**: Use TypeScript interfaces
4. **Controlled vs Uncontrolled**: Prefer controlled components
5. **Error Boundaries**: Wrap components with error boundaries

### State Management

1. **Global vs Local**: Use global state sparingly
2. **Derived State**: Compute derived values instead of storing
3. **Immutability**: Never mutate state directly
4. **Normalization**: Normalize nested data structures
5. **Selectors**: Use selectors for computed state

### Performance

1. **Lazy Loading**: Load features on-demand
2. **Code Splitting**: Split by route and feature
3. **Memoization**: Memoize expensive computations
4. **Virtual Scrolling**: For large lists
5. **Web Workers**: For heavy computation

### TypeScript

1. **Strict Mode**: Enable strict type checking
2. **No Any**: Avoid `any` type
3. **Interface vs Type**: Use interfaces for objects
4. **Generics**: Use generics for reusable code
5. **Type Guards**: Use type guards for runtime checks

### Testing

1. **Test Behavior**: Test what users see and do
2. **Avoid Implementation Details**: Don't test internals
3. **Mock External Dependencies**: Mock API calls
4. **Snapshot Testing**: Use sparingly
5. **Coverage**: Aim for >80% coverage

---

## Conclusion

The VisionFlow client architecture demonstrates modern web development best practices:

- **TypeScript** for type safety
- **React** for component-based UI
- **Three.js/Babylon.js** for 3D rendering
- **Zustand** for state management
- **WebSocket** for real-time communication
- **WebXR** for immersive experiences

This architecture supports high-performance 3D graph visualization with thousands of nodes, real-time multi-agent collaboration, and immersive XR experiences on Meta Quest 3.

---

**Document Version:** 1.0.0
**Total Lines:** 3,571
**Last Updated:** 2025-11-04
