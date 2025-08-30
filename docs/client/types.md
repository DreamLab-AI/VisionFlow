# Client TypeScript Types

This document provides a summary of the key data structures used in the client application. While this document is a helpful reference, **the TypeScript source files are the single source of truth.**

## 1. Settings

The main `Settings` interface is the most critical data structure, defining all configurable aspects of the application.

-   **Authoritative Source**: [`client/src/features/settings/config/settings.ts`](../../client/src/features/settings/config/settings.ts)

### Multi-Graph Architecture

To support the rendering of multiple, distinct graphs (e.g., the user's knowledge graph and the VisionFlow agent graph), the settings structure has been updated to be nested.

-   **Old Structure (Deprecated)**: `settings.visualisation.nodes`, `settings.visualisation.edges`
-   **New Structure**: `settings.visualisation.graphs.logseq`, `settings.visualisation.graphs.visionflow`

Each graph type under the `graphs` object has its own complete set of `nodes`, `edges`, `labels`, and `physics` settings, allowing for independent visual configuration.

**Simplified Structure:**

```typescript
// Source: client/src/features/settings/config/settings.ts

export interface Settings {
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings;
  // ... other settings modules
}

export interface VisualisationSettings {
  // NEW: Namespace for individual graph configurations
  graphs: {
    logseq: GraphSettings;
    visionflow: GraphSettings;
  };
  
  // Global settings that apply to the entire scene
  rendering: RenderingSettings;
  animations: AnimationSettings;
  glow: GlowSettings;
  // ... other global settings
}

// Container for a single graph's complete visual settings
export interface GraphSettings {
  nodes: NodeSettings;
  edges: EdgeSettings;
  labels: LabelSettings;
  physics: PhysicsSettings;
}

// Example of accessing the new nested settings:
// const logseqNodecolour = settings.visualisation.graphs.logseq.nodes.basecolour;
// const visionflowEdgeWidth = settings.visualisation.graphs.visionflow.edges.baseWidth;
```

## 2. Binary Protocol

The binary protocol is used for high-performance, real-time communication of graph physics data over WebSockets.

-   **Authoritative Source**: [`client/src/types/binaryProtocol.ts`](../../client/src/types/binaryProtocol.ts)

### `BinaryNodeData`

This is the core data structure for a single node's physics state. A single WebSocket message can contain an array of these structures, packed together in an `ArrayBuffer`.

```typescript
// Source: client/src/types/binaryProtocol.ts

export interface BinaryNodeData {
  nodeId: number;   // u32 integer ID with type flags
  position: { x: number; y: number; z: number }; // 3 x f32
  velocity: { x: number; y: number; z: number }; // 3 x f32
}
```

**Key Implementation Details:**

-   **Total Size**: Each `BinaryNodeData` object is **28 bytes**.
-   **Type Flags**: The `nodeId` is a `u32` integer where the highest two bits are used as flags to distinguish between node types (e.g., `AGENT_NODE_FLAG`, `KNOWLEDGE_NODE_FLAG`). The actual ID is extracted using a bitmask (`NODE_ID_MASK`). This allows the client to handle different types of nodes from the same binary stream.
-   **Parsing**: The file provides `parseBinaryNodeData` and `createBinaryNodeData` functions for serializing and deserializing this data.