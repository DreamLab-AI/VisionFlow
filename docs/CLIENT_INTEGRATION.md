# VisionFlow Client Integration Guide

**Version:** 3.0.0
**Last Updated:** 2025-10-22
**Framework:** React 18+ with TypeScript

---

## Overview

This guide covers integrating the VisionFlow frontend client with the new hexagonal architecture backend. The key principle is **server-authoritative state management** - the server is the single source of truth.

---

## Architecture Changes

### Before: Client-Side Caching
```typescript
// OLD PATTERN - Don't use this!
const settingsStore = {
  cache: new Map<string, SettingValue>(),

  async get(key: string): Promise<SettingValue> {
    // Check local cache first
    if (this.cache.has(key)) {
      return this.cache.get(key)!; // Stale data risk!
    }

    // Fetch from server
    const value = await api.getSetting(key);
    this.cache.set(key, value); // Client manages cache
    return value;
  }
};
```

**Problems:**
- Stale cache issues
- Complex invalidation logic
- Multiple sources of truth
- Race conditions between clients

### After: Server-Authoritative
```typescript
// NEW PATTERN - Use this!
const settingsApi = {
  async get(key: string): Promise<SettingValue> {
    // Always fetch from server (server has 5-min cache)
    return await api.getSetting(key);
  },

  async update(key: string, value: SettingValue): Promise<void> {
    await api.updateSetting(key, value);
    // Server broadcasts update via WebSocket to all clients
  }
};
```

**Benefits:**
- ✅ Single source of truth (database)
- ✅ No stale cache issues
- ✅ Server-side caching optimizes performance
- ✅ WebSocket broadcasts keep all clients in sync
- ✅ Simpler client code

---

## Settings API Client

### Updated API Methods

```typescript
// client/src/api/settingsApi.ts

class SettingsApi {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
  }

  // Get single setting (NO CLIENT CACHE)
  async getSetting(key: string): Promise<SettingValue | null> {
    const response = await fetch(
      `${this.baseUrl}/api/settings/path/${encodeURIComponent(key)}`,
      {
        headers: this.getAuthHeaders(),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to get setting: ${response.statusText}`);
    }

    const data = await response.json();
    return data.value;
  }

  // Update single setting
  async updateSetting(key: string, value: SettingValue): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/settings/path/${encodeURIComponent(key)}`,
      {
        method: 'PUT',
        headers: {
          ...this.getAuthHeaders(),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ value }),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to update setting: ${response.statusText}`);
    }

    // No need to update local cache - server will broadcast via WebSocket
  }

  // Get physics settings for specific graph
  async getPhysicsSettings(
    graphName: 'logseq' | 'visionflow' | 'ontology' | 'default'
  ): Promise<PhysicsSettings> {
    const response = await fetch(
      `${this.baseUrl}/api/settings/physics/${graphName}`,
      {
        headers: this.getAuthHeaders(),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to get physics settings: ${response.statusText}`);
    }

    return await response.json();
  }

  // Update physics settings
  async updatePhysicsSettings(
    graphName: 'logseq' | 'visionflow' | 'ontology' | 'default',
    settings: Partial<PhysicsSettings>
  ): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/settings/physics/${graphName}`,
      {
        method: 'PUT',
        headers: {
          ...this.getAuthHeaders(),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to update physics settings: ${response.statusText}`);
    }
  }

  // Validate graph separation (prevent conflation)
  async validateGraphSeparation(): Promise<{ valid: boolean; message?: string }> {
    const logseqPhysics = await this.getPhysicsSettings('logseq');
    const visionflowPhysics = await this.getPhysicsSettings('visionflow');

    if (JSON.stringify(logseqPhysics) === JSON.stringify(visionflowPhysics)) {
      return {
        valid: false,
        message: 'WARNING: Logseq and Visionflow physics settings are identical. Possible conflation!',
      };
    }

    return { valid: true };
  }

  private getAuthHeaders(): Record<string, string> {
    const token = localStorage.getItem('auth_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  }
}

export const settingsApi = new SettingsApi();
```

---

## Ontology Mode Toggle

### Graph Mode State Management

```typescript
// client/src/contexts/GraphModeContext.tsx

import React, { createContext, useContext, useState, useEffect } from 'react';

type GraphMode = 'knowledge' | 'ontology';

interface GraphModeContextValue {
  mode: GraphMode;
  setMode: (mode: GraphMode) => void;
  isOntologyMode: boolean;
}

const GraphModeContext = createContext<GraphModeContextValue | null>(null);

export const GraphModeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mode, setMode] = useState<GraphMode>('knowledge');

  const value: GraphModeContextValue = {
    mode,
    setMode,
    isOntologyMode: mode === 'ontology',
  };

  return <GraphModeContext.Provider value={value}>{children}</GraphModeContext.Provider>;
};

export const useGraphMode = () => {
  const context = useContext(GraphModeContext);
  if (!context) {
    throw new Error('useGraphMode must be used within GraphModeProvider');
  }
  return context;
};
```

### Ontology Mode Toggle Component

```typescript
// client/src/components/OntologyModeToggle.tsx

import React from 'react';
import { useGraphMode } from '../contexts/GraphModeContext';

export const OntologyModeToggle: React.FC = () => {
  const { mode, setMode } = useGraphMode();

  const handleToggle = (newMode: 'knowledge' | 'ontology') => {
    setMode(newMode);
    // Reconnect WebSocket to appropriate endpoint
    // This is handled by the WebSocket hook (see below)
  };

  return (
    <div className="mode-toggle">
      <button
        className={mode === 'knowledge' ? 'active' : ''}
        onClick={() => handleToggle('knowledge')}
      >
        Knowledge Graph
      </button>
      <button
        className={mode === 'ontology' ? 'active' : ''}
        onClick={() => handleToggle('ontology')}
      >
        Ontology Graph
      </button>
    </div>
  );
};
```

---

## Binary WebSocket Protocol

### Connection Management

```typescript
// client/src/hooks/useGraphWebSocket.ts

import { useEffect, useRef, useState } from 'react';
import { useGraphMode } from '../contexts/GraphModeContext';

interface NodeUpdate {
  graphType: 'knowledge' | 'ontology' | 'agent' | 'reserved';
  nodeId: number;
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  color: { r: number; g: number; b: number; a: number };
  isPinned: boolean;
  isSelected: boolean;
  isHighlighted: boolean;
  isVisible: boolean;
  nodeType: number;
}

export const useGraphWebSocket = (
  onNodeUpdate: (update: NodeUpdate) => void
) => {
  const { mode } = useGraphMode();
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Disconnect previous connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Determine WebSocket endpoint based on mode
    const endpoint = mode === 'knowledge'
      ? 'ws://localhost:8080/api/graph/stream'
      : 'ws://localhost:8080/api/ontology/graph/stream';

    // Create new connection
    const ws = new WebSocket(endpoint);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`Connected to ${mode} graph stream`);
      setIsConnected(true);

      // Authenticate
      const token = localStorage.getItem('auth_token');
      if (token) {
        ws.send(JSON.stringify({ type: 'authenticate', token }));
      }

      // Request full sync
      ws.send(JSON.stringify({ type: 'request_sync' }));
    };

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        // Binary message (node update)
        event.data.arrayBuffer().then((buffer) => {
          const update = parseBinaryNodeUpdate(buffer);
          onNodeUpdate(update);
        });
      } else {
        // JSON message
        const message = JSON.parse(event.data);
        // Handle other message types...
      }
    };

    ws.onclose = () => {
      console.log(`Disconnected from ${mode} graph stream`);
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    // Cleanup on unmount or mode change
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [mode, onNodeUpdate]);

  return { isConnected };
};

// Binary protocol parser (36-byte message)
function parseBinaryNodeUpdate(buffer: ArrayBuffer): NodeUpdate {
  const view = new DataView(buffer);
  const msgType = view.getUint8(0);

  if (msgType !== 0x01) {
    throw new Error('Not a node update message');
  }

  const nodeId = view.getUint32(1, true); // little-endian
  const graphType = (nodeId >> 30) & 0b11;
  const actualId = nodeId & 0x3FFFFFFF;

  return {
    graphType: ['knowledge', 'ontology', 'agent', 'reserved'][graphType] as any,
    nodeId: actualId,
    position: {
      x: view.getFloat32(5, true),
      y: view.getFloat32(9, true),
      z: view.getFloat32(13, true),
    },
    velocity: {
      x: view.getFloat32(17, true),
      y: view.getFloat32(21, true),
      z: view.getFloat32(25, true),
    },
    color: {
      r: (view.getUint32(29, true) >> 24) & 0xFF,
      g: (view.getUint32(29, true) >> 16) & 0xFF,
      b: (view.getUint32(29, true) >> 8) & 0xFF,
      a: view.getUint32(29, true) & 0xFF,
    },
    isPinned: (view.getUint8(33) & 0b1) !== 0,
    isSelected: (view.getUint8(33) & 0b10) !== 0,
    isHighlighted: (view.getUint8(33) & 0b100) !== 0,
    isVisible: (view.getUint8(33) & 0b1000) !== 0,
    nodeType: view.getUint8(34),
  };
}
```

---

## Visualization Integration

### React Component Example

```typescript
// client/src/components/GraphVisualization.tsx

import React, { useCallback, useEffect, useRef } from 'react';
import { useGraphWebSocket } from '../hooks/useGraphWebSocket';
import { useGraphMode } from '../contexts/GraphModeContext';

export const GraphVisualization: React.FC = () => {
  const { mode } = useGraphMode();
  const nodesRef = useRef<Map<number, NodeData>>(new Map());

  const handleNodeUpdate = useCallback((update: NodeUpdate) => {
    // Update node in local visualization
    nodesRef.current.set(update.nodeId, {
      id: update.nodeId,
      position: update.position,
      color: update.color,
      // ... other properties
    });

    // Trigger re-render or update Three.js scene
    updateVisualization();
  }, []);

  const { isConnected } = useGraphWebSocket(handleNodeUpdate);

  const updateVisualization = () => {
    // Update Three.js scene with new node positions
    // This is called at 60 FPS by the WebSocket updates
  };

  return (
    <div className="graph-container">
      <div className="connection-status">
        {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        <span className="mode-indicator">{mode} mode</span>
      </div>
      <canvas ref={canvasRef} />
    </div>
  );
};
```

---

## State Management Patterns

### Server-Authoritative Settings

```typescript
// client/src/stores/settingsStore.ts

import { create } from 'zustand';
import { settingsApi } from '../api/settingsApi';

interface SettingsStore {
  // No local cache - always fetch from server
  getSetting: (key: string) => Promise<SettingValue | null>;
  updateSetting: (key: string, value: SettingValue) => Promise<void>;
}

export const useSettingsStore = create<SettingsStore>((set, get) => ({
  getSetting: async (key: string) => {
    // Always fetch from server (server has optimized cache)
    return await settingsApi.getSetting(key);
  },

  updateSetting: async (key: string, value: SettingValue) => {
    await settingsApi.updateSetting(key, value);
    // Server will broadcast update to all clients via WebSocket
    // No need to update local state
  },
}));
```

### Graph State (Visualization Only)

```typescript
// client/src/stores/graphStore.ts

import { create } from 'zustand';

interface GraphStore {
  nodes: Map<number, NodeData>;
  edges: Map<string, EdgeData>;

  // Only for visualization state, not persisted
  updateNodePosition: (nodeId: number, position: Vector3) => void;
  selectNode: (nodeId: number) => void;
}

export const useGraphStore = create<GraphStore>((set, get) => ({
  nodes: new Map(),
  edges: new Map(),

  updateNodePosition: (nodeId, position) => {
    set((state) => {
      const newNodes = new Map(state.nodes);
      const node = newNodes.get(nodeId);
      if (node) {
        node.position = position;
        newNodes.set(nodeId, node);
      }
      return { nodes: newNodes };
    });
  },

  selectNode: (nodeId) => {
    // Selection is local UI state only
    // Not persisted to database
    set((state) => {
      const newNodes = new Map(state.nodes);
      newNodes.forEach((node) => {
        node.isSelected = node.id === nodeId;
      });
      return { nodes: newNodes };
    });
  },
}));
```

---

## Testing Client Integration

### Unit Tests

```typescript
// client/src/api/__tests__/settingsApi.test.ts

import { settingsApi } from '../settingsApi';

describe('SettingsApi', () => {
  it('should fetch setting from server', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ value: 'dark' }),
      })
    ) as jest.Mock;

    const value = await settingsApi.getSetting('application.theme');
    expect(value).toBe('dark');
  });

  it('should validate graph separation', async () => {
    global.fetch = jest.fn((url) => {
      if (url.includes('logseq')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ time_step: 0.016 }),
        });
      } else {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ time_step: 0.020 }),
        });
      }
    }) as jest.Mock;

    const result = await settingsApi.validateGraphSeparation();
    expect(result.valid).toBe(true);
  });
});
```

---

## Migration Checklist

### Remove Client-Side Caching
- [ ] Delete `settingsStore.cache` Map
- [ ] Remove cache invalidation logic
- [ ] Remove lazy-loading mechanisms
- [ ] Update all API calls to fetch from server

### Add Ontology Mode Toggle
- [ ] Create `GraphModeContext`
- [ ] Create `OntologyModeToggle` component
- [ ] Update WebSocket connections based on mode
- [ ] Add mode indicator to UI

### Implement Binary Protocol
- [ ] Add `parseBinaryNodeUpdate` function
- [ ] Update WebSocket message handler for Blob messages
- [ ] Test with 100k+ nodes at 60 FPS

### Server-Authoritative State
- [ ] Remove local state persistence
- [ ] Rely on WebSocket broadcasts for updates
- [ ] Remove optimistic updates (wait for server confirmation)

---

## Additional Resources

- [VisionFlow Architecture](/docs/ARCHITECTURE.md)
- [API Documentation](/docs/API.md)
- [Developer Guide](/docs/DEVELOPER_GUIDE.md)
- [React Documentation](https://react.dev/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

---

**Document Maintained By:** VisionFlow Frontend Team
**Last Review:** 2025-10-22
**Next Review:** 2025-11-22
