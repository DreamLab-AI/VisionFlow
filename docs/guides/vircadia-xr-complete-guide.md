# Complete Vircadia XR Integration Guide

**Multi-User Extended Reality with Force-Directed Graph Visualization**

*Version: 3.0.0*
*Last Updated: 2025-10-27*
*Status: Production*

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Vircadia Platform](#vircadia-platform)
4. [Force-Directed Graph System](#force-directed-graph-system)
5. [Real-Time Synchronization](#real-time-synchronization)
6. [Quest 3 Optimization](#quest-3-optimization)
7. [Deployment](#deployment)
8. [API Reference](#api-reference)

---

## Overview

### What is Vircadia?

**Vircadia** is an open-source metaverse platform that enables collaborative, multi-user virtual experiences. Originally forked from High Fidelity, Vircadia provides:

- **Multi-user sessions** with up to 50+ concurrent users
- **Spatial audio** with 3D positional sound
- **Avatar systems** with full-body tracking support
- **Real-time entity synchronization** via WebSocket
- **Persistent virtual worlds** with server-side state management

**Official Documentation**: [https://docs.vircadia.com](https://docs.vircadia.com)
**GitHub**: [https://github.com/vircadia/vircadia-world](https://github.com/vircadia/vircadia-world)

### Our Custom Integration

We've extended Vircadia to support:

1. **Force-Directed Graph Visualization** - Physics-based graph layout in 3D space
2. **Knowledge Graph Multi-User Exploration** - Collaborative data exploration
3. **Quest 3 Native Support** - Optimized WebXR for Meta Quest 3
4. **Real-time Graph Synchronization** - Sub-100ms latency for graph updates
5. **Spatial Graph Navigation** - Fly through knowledge graphs in VR

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **XR Client** | Babylon.js 8.28+ + WebXR | 3D rendering and XR sessions |
| **Graph Physics** | Custom Force Engine | Force-directed layout simulation |
| **Multi-User Backend** | Vircadia World Server (Bun + TypeScript) | Real-time state sync |
| **Database** | PostgreSQL 17.5 with Row-Level Security | Entity persistence |
| **Protocol** | SQL-over-WebSocket (Binary) | Efficient network transport |
| **Container** | Docker Compose | Isolated deployment |

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  Quest 3 AR  │ │ Desktop VR   │ │  Web Viewer  │        │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘        │
│         └────────────────┴────────────────┘                 │
│                          │                                    │
│              ┌───────────▼──────────┐                        │
│              │ Babylon.js XR Engine │                        │
│              └───────────┬──────────┘                        │
│         ┌────────────────┴────────────────┐                 │
│    ┌────▼────┐               ┌────────────▼──────┐          │
│    │  Graph  │               │  Force-Directed   │          │
│    │ Renderer│               │  Physics Engine   │          │
│    └────┬────┘               └────────────┬──────┘          │
│         └────────────────┬────────────────┘                 │
│                          │                                    │
└──────────────────────────┼────────────────────────────────┘
                           │ WebSocket
┌──────────────────────────▼────────────────────────────────┐
│              Vircadia World Server (Docker)                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │     World API Manager (Port 3000)                   │  │
│  │  • WebSocket Server (SQL-over-WS Protocol)          │  │
│  │  • JWT Authentication & Session Management           │  │
│  │  • Entity Query Processor                            │  │
│  │  • Real-Time Event Broadcasting                      │  │
│  └──────────────────┬──────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐  │
│  │    World State Manager (Port 3001)                   │  │
│  │  • Tick Processing @ 60 TPS (16ms intervals)         │  │
│  │  • Entity Delta Compression                          │  │
│  │  • Sync Group Management (public.NORMAL)             │  │
│  │  • Spatial Interest Management                       │  │
│  └──────────────────┬──────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐  │
│  │         PostgreSQL 17.5 (Port 5432)                  │  │
│  │  • entity.entities (Graph node/edge storage)         │  │
│  │  • entity.sync_groups (Multi-user coordination)      │  │
│  │  • auth.agent_sessions (JWT session management)      │  │
│  │  • Row-Level Security (RLS) Policies                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────┘
```

### Data Flow: User Interaction to Multi-User Sync

```
User Grabs Node (VR Controller)
  ↓
Client: Local Prediction (Instant Visual Feedback)
  ↓
WebSocket: QUERY_REQUEST (UPDATE entity.entities SET meta__data->position)
  ↓
Server: PostgreSQL Transaction Commit
  ↓
Server: Mark Entity for Next Tick Broadcast
  ↓
[16ms Tick Interval @ 60 TPS]
  ↓
Server: SELECT Changed Entities
  ↓
Server: Binary SYNC_GROUP_UPDATES Message
  ↓
Broadcast to ALL Connected Clients (Including Original)
  ↓
Clients: Parse Binary Update, Update Mesh Positions
  ↓
Original Client: Reconcile Prediction with Server State
  ↓
All Users See Synchronized State (<100ms total latency)
```

---

## Vircadia Platform

### Core Concepts

#### 1. Entities

Everything in Vircadia is an **Entity** - a persistent object with:

```typescript
interface VircadiaEntity {
  general__entity_name: string;          // Unique identifier
  group__sync: string;                   // Sync group (e.g., "public.NORMAL")
  general__entity_type: string;          // "MODEL", "SHAPE", "LIGHT", etc.
  spatial__transform: {                  // Position, rotation, scale
    translation: {x: number, y: number, z: number};
    rotation: {x: number, y: number, z: number, w: number};  // Quaternion
    scaling: {x: number, y: number, z: number};
  };
  meta__data: Record<string, any>;       // Custom metadata (JSON)
  spatial__velocity: {x, y, z};          // Physics velocity
  spatial__angular_velocity: {x, y, z};  // Rotation velocity
  general__created_at: Date;
  general__updated_at: Date;
}
```

**Official Reference**: [Vircadia Entity System](https://docs.vircadia.com/explore/entities)

#### 2. Sync Groups

Sync Groups define **who can see and modify** which entities:

```sql
-- Sync group configuration
CREATE TYPE entity.sync_group_visibility AS ENUM (
  'EVERYONE',          -- All users can see
  'FRIENDS_ONLY',      -- Only friends
  'PRIVATE'            -- Only creator
);

CREATE TYPE entity.sync_group_synchronization_mode AS ENUM (
  'NORMAL',            -- Standard sync (our default)
  'LOCAL_ONLY',        -- No network sync
  'SERVER_AUTHORITATIVE'  -- Server validates all changes
);
```

**Our Configuration**: `public.NORMAL`
- Read: All authenticated users
- Write: Authenticated users
- Tick Rate: 60 TPS (16ms)
- Max Buffer: 10 ticks
- Render Delay: 50ms (interpolation)

**Official Reference**: [Vircadia Sync Groups](https://docs.vircadia.com/api/sync-groups)

#### 3. SQL-over-WebSocket Protocol

Vircadia uses a unique approach: **SQL queries over WebSocket**. Instead of REST APIs, clients send SQL directly:

```typescript
// Connect to Vircadia World Server
const ws = new WebSocket('ws://localhost:3000/world/ws?token=<JWT>&provider=system');

// Execute SQL query via WebSocket
ws.send(JSON.stringify({
  type: 'QUERY_REQUEST',
  requestId: crypto.randomUUID(),
  timestamp: Date.now(),
  payload: {
    query: `
      INSERT INTO entity.entities (
        general__entity_name,
        group__sync,
        spatial__transform,
        meta__data
      ) VALUES ($1, $2, $3, $4)
      RETURNING *
    `,
    parameters: ['graph_node_123', 'public.NORMAL', transform, metadata]
  }
}));

// Receive response
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  if (response.type === 'QUERY_RESPONSE') {
    console.log('Entity created:', response.payload.result);
  }
};
```

**Security**: Row-Level Security (RLS) in PostgreSQL prevents unauthorized access.

**Official Reference**: [Vircadia WebSocket Protocol](https://docs.vircadia.com/api/websocket-protocol)

#### 4. Tick System

The State Manager processes entities in **ticks** (16ms @ 60 TPS):

```typescript
// Tick cycle
setInterval(async () => {
  // 1. Fetch changed entities from last tick
  const changedEntities = await db.query(`
    SELECT * FROM entity.entities
    WHERE general__updated_at > $1
      AND group__sync = 'public.NORMAL'
  `, [lastTickTime]);

  // 2. Build binary update message
  const binaryUpdate = encodeBinarySync(changedEntities);

  // 3. Broadcast to all connected clients
  for (const client of clients) {
    client.send(binaryUpdate);
  }

  // 4. Update last tick time
  lastTickTime = Date.now();
}, 16);  // 16ms = 60 TPS
```

**Official Reference**: [Vircadia State Manager](https://docs.vircadia.com/server/state-manager)

---

## Force-Directed Graph System

### Physics-Based Graph Layout

Our custom implementation uses **force-directed simulation** to spatially organize knowledge graphs in 3D.

#### Algorithm Overview

Force-directed layout simulates physical forces:

1. **Repulsion** - Nodes push each other apart (like charged particles)
2. **Attraction** - Connected nodes pull toward each other (like springs)
3. **Centering** - Weak force toward origin prevents drift
4. **Damping** - Friction slows movement to reach equilibrium

```typescript
// Core force simulation
interface GraphNode {
  id: string;
  position: {x: number, y: number, z: number};
  velocity: {x: number, y: number, z: number};
  mass: number;  // Affects inertia
}

interface GraphEdge {
  source: string;  // Node ID
  target: string;  // Node ID
  weight: number;  // Spring strength
}

class ForceDirectedEngine {
  private nodes: Map<string, GraphNode>;
  private edges: GraphEdge[];

  // Physics constants
  private readonly REPULSION_STRENGTH = 1000;
  private readonly SPRING_STRENGTH = 0.1;
  private readonly SPRING_LENGTH = 5.0;      // Ideal edge length (meters)
  private readonly DAMPING = 0.9;
  private readonly CENTERING_STRENGTH = 0.01;
  private readonly MAX_VELOCITY = 2.0;
  private readonly TIME_STEP = 0.016;        // 16ms (matches tick rate)

  /**
   * Apply repulsion force between all node pairs
   * F = k * (1 / distance^2) * direction
   */
  private applyRepulsion(): void {
    const nodes = Array.from(this.nodes.values());

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];

        const dx = b.position.x - a.position.x;
        const dy = b.position.y - a.position.y;
        const dz = b.position.z - a.position.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (distance < 0.1) continue;  // Avoid division by zero

        const force = this.REPULSION_STRENGTH / (distance * distance);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        const fz = (dz / distance) * force;

        // Newton's 3rd law: equal and opposite
        a.velocity.x -= fx / a.mass;
        a.velocity.y -= fy / a.mass;
        a.velocity.z -= fz / a.mass;

        b.velocity.x += fx / b.mass;
        b.velocity.y += fy / b.mass;
        b.velocity.z += fz / b.mass;
      }
    }
  }

  /**
   * Apply spring attraction along edges
   * F = k * (distance - restLength) * direction
   */
  private applyAttraction(): void {
    for (const edge of this.edges) {
      const source = this.nodes.get(edge.source);
      const target = this.nodes.get(edge.target);
      if (!source || !target) continue;

      const dx = target.position.x - source.position.x;
      const dy = target.position.y - source.position.y;
      const dz = target.position.z - source.position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      const displacement = distance - this.SPRING_LENGTH;
      const force = this.SPRING_STRENGTH * edge.weight * displacement;

      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      const fz = (dz / distance) * force;

      source.velocity.x += fx / source.mass;
      source.velocity.y += fy / source.mass;
      source.velocity.z += fz / source.mass;

      target.velocity.x -= fx / target.mass;
      target.velocity.y -= fy / target.mass;
      target.velocity.z -= fz / target.mass;
    }
  }

  /**
   * Apply weak centering force
   */
  private applyCentering(): void {
    for (const node of this.nodes.values()) {
      node.velocity.x -= node.position.x * this.CENTERING_STRENGTH;
      node.velocity.y -= node.position.y * this.CENTERING_STRENGTH;
      node.velocity.z -= node.position.z * this.CENTERING_STRENGTH;
    }
  }

  /**
   * Update positions based on velocities (Verlet integration)
   */
  private integrate(): void {
    for (const node of this.nodes.values()) {
      // Apply damping
      node.velocity.x *= this.DAMPING;
      node.velocity.y *= this.DAMPING;
      node.velocity.z *= this.DAMPING;

      // Clamp to max velocity
      const speed = Math.sqrt(
        node.velocity.x ** 2 +
        node.velocity.y ** 2 +
        node.velocity.z ** 2
      );
      if (speed > this.MAX_VELOCITY) {
        const scale = this.MAX_VELOCITY / speed;
        node.velocity.x *= scale;
        node.velocity.y *= scale;
        node.velocity.z *= scale;
      }

      // Update positions
      node.position.x += node.velocity.x * this.TIME_STEP;
      node.position.y += node.velocity.y * this.TIME_STEP;
      node.position.z += node.velocity.z * this.TIME_STEP;
    }
  }

  /**
   * Run one simulation step
   */
  public tick(): void {
    this.applyRepulsion();
    this.applyAttraction();
    this.applyCentering();
    this.integrate();
  }

  /**
   * Get node positions as Float32Array for efficient GPU upload
   */
  public getPositionsArray(): Float32Array {
    const positions = new Float32Array(this.nodes.size * 3);
    let index = 0;
    for (const node of this.nodes.values()) {
      positions[index++] = node.position.x;
      positions[index++] = node.position.y;
      positions[index++] = node.position.z;
    }
    return positions;
  }
}
```

#### Integration with Vircadia

Map graph nodes to Vircadia entities:

```typescript
class VircadiaGraphMapper {
  private client: VircadiaClient;
  private forceEngine: ForceDirectedEngine;

  /**
   * Sync graph node to Vircadia entity
   */
  async syncNodeToEntity(node: GraphNode): Promise<void> {
    // Create or update Vircadia entity
    await this.client.query(`
      INSERT INTO entity.entities (
        general__entity_name,
        group__sync,
        general__entity_type,
        spatial__transform,
        meta__data
      ) VALUES ($1, $2, $3, $4, $5)
      ON CONFLICT (general__entity_name)
      DO UPDATE SET
        spatial__transform = EXCLUDED.spatial__transform,
        general__updated_at = NOW()
    `, [
      node.id,                       // Entity name
      'public.NORMAL',               // Sync group
      'SHAPE_SPHERE',                // Entity type
      {
        translation: node.position,
        rotation: {x: 0, y: 0, z: 0, w: 1},  // Identity quaternion
        scaling: {x: 0.15, y: 0.15, z: 0.15}  // 15cm radius
      },
      {
        entityType: 'graph_node',
        graphId: node.id,
        label: node.label,
        color: node.color || '#4CAF50',
        nodeType: node.type || 'default'
      }
    ]);
  }

  /**
   * Sync graph edge to Vircadia entity (line)
   */
  async syncEdgeToEntity(edge: GraphEdge): Promise<void> {
    const sourceNode = this.forceEngine.getNode(edge.source);
    const targetNode = this.forceEngine.getNode(edge.target);

    if (!sourceNode || !targetNode) return;

    await this.client.query(`
      INSERT INTO entity.entities (
        general__entity_name,
        group__sync,
        general__entity_type,
        meta__data
      ) VALUES ($1, $2, $3, $4)
      ON CONFLICT (general__entity_name)
      DO UPDATE SET
        meta__data = EXCLUDED.meta__data,
        general__updated_at = NOW()
    `, [
      `edge_${edge.source}_${edge.target}`,
      'public.NORMAL',
      'LINE',
      {
        entityType: 'graph_edge',
        source: edge.source,
        target: edge.target,
        weight: edge.weight,
        points: [sourceNode.position, targetNode.position],
        color: edge.color || '#757575',
        opacity: edge.opacity || 0.6
      }
    ]);
  }

  /**
   * Run simulation and sync to Vircadia (called every tick)
   */
  async tick(): Promise<void> {
    // 1. Run force simulation
    this.forceEngine.tick();

    // 2. Batch update all node positions
    const updates: Promise<void>[] = [];
    for (const node of this.forceEngine.getNodes()) {
      updates.push(this.syncNodeToEntity(node));
    }
    for (const edge of this.forceEngine.getEdges()) {
      updates.push(this.syncEdgeToEntity(edge));
    }

    // 3. Wait for all updates (parallel SQL queries)
    await Promise.all(updates);
  }
}
```

### Performance Optimizations

```typescript
// Web Worker for physics simulation (offload from main thread)
const physicsWorker = new Worker('force-simulation-worker.js');

// Send graph data to worker
physicsWorker.postMessage({
  type: 'INIT',
  nodes: graphData.nodes,
  edges: graphData.edges
});

// Receive position updates every frame
physicsWorker.onmessage = (event) => {
  if (event.data.type === 'POSITIONS_UPDATE') {
    // Float32Array of positions [x1, y1, z1, x2, y2, z2, ...]
    const positions = event.data.positions;

    // Update Babylon.js mesh instances (GPU-efficient)
    graphRenderer.updateNodePositions(positions);
  }
};

// Request simulation at 60 FPS
setInterval(() => {
  physicsWorker.postMessage({ type: 'TICK' });
}, 16);
```

---

## Real-Time Synchronization

### Client-Side Prediction

Achieve <50ms perceived latency with **client-side prediction**:

```typescript
class PredictiveGraphSync {
  private pendingUpdates: Map<string, NodeUpdate> = new Map();

  /**
   * User drags node (immediate visual update)
   */
  onUserDragNode(nodeId: string, newPosition: Vector3): void {
    // 1. Immediate local update (no network delay)
    graphRenderer.updateNodePosition(nodeId, newPosition);

    // 2. Store as pending prediction
    this.pendingUpdates.set(nodeId, {
      predictedPosition: newPosition,
      timestamp: Date.now()
    });

    // 3. Send to server (async)
    this.sendPositionUpdate(nodeId, newPosition);
  }

  /**
   * Server confirms update
   */
  onServerConfirmation(nodeId: string, serverPosition: Vector3): void {
    const pending = this.pendingUpdates.get(nodeId);
    if (!pending) return;

    // Calculate prediction error
    const error = Vector3.Distance(pending.predictedPosition, serverPosition);

    if (error > 0.1) {  // 10cm threshold
      // Reconcile: smooth interpolate to server position
      graphRenderer.interpolateNodePosition(
        nodeId,
        pending.predictedPosition,
        serverPosition,
        100  // 100ms interpolation
      );
    }

    this.pendingUpdates.delete(nodeId);
  }

  /**
   * Remote user's update
   */
  onRemoteUpdate(nodeId: string, remotePosition: Vector3): void {
    // Check if we have a conflicting local prediction
    if (this.pendingUpdates.has(nodeId)) {
      // Our prediction takes precedence (we're actively manipulating)
      return;
    }

    // Apply remote update with interpolation
    graphRenderer.interpolateNodePosition(
      nodeId,
      graphRenderer.getNodePosition(nodeId),
      remotePosition,
      50  // 50ms interpolation (matches render delay)
    );
  }
}
```

---

## Quest 3 Optimization

### Rendering Performance

```typescript
// Quest 3 specific optimizations
class Quest3Optimizer {
  private scene: BABYLON.Scene;

  applyOptimizations(): void {
    const engine = this.scene.getEngine();

    // 1. Enable foveated rendering (90 FPS target)
    if (this.scene.activeCamera instanceof BABYLON.WebXRCamera) {
      const xrHelper = this.scene.activeCamera.rigCameras[0].getScene().metadata.xrHelper;
      xrHelper.baseExperience.featuresManager.enableFeature(
        BABYLON.WebXRFeatureName.FOVEATED_RENDERING,
        'latest',
        { foveationLevel: 2 }  // High (most aggressive)
      );
    }

    // 2. Hardware scaling (0.9x for performance headroom)
    engine.setHardwareScalingLevel(0.9);

    // 3. Disable shadows in VR (expensive)
    this.scene.lights.forEach(light => {
      if (light instanceof BABYLON.DirectionalLight) {
        light.shadowEnabled = false;
      }
    });

    // 4. Reduce physics update rate (30 Hz sufficient)
    this.scene.onBeforeRenderObservable.add(() => {
      if (this.scene.getFrameId() % 3 === 0) {  // Every 3rd frame
        forceEngine.tick();
      }
    });

    // 5. Instance rendering for nodes
    this.setupInstancedMeshes();

    // 6. Spatial culling (50m radius)
    const camera = this.scene.activeCamera;
    this.scene.meshes.forEach(mesh => {
      if (BABYLON.Vector3.Distance(mesh.position, camera.position) > 50) {
        mesh.isVisible = false;
      }
    });

    // 7. Texture compression
    engine.texturesSupported = engine.texturesSupported.filter(format =>
      format === '.ktx2'  // Quest 3 native format
    );
  }

  private setupInstancedMeshes(): void {
    // Create single source mesh
    const sourceMesh = BABYLON.MeshBuilder.CreateSphere(
      'nodeSource',
      { diameter: 0.3, segments: 16 },  // Low poly
      this.scene
    );
    sourceMesh.isVisible = false;

    // Create instanced mesh for all nodes
    const instancedMesh = sourceMesh.createInstance('nodeInstance');

    // Set instance buffer (positions updated from force engine)
    instancedMesh.registerInstancedBuffer('position', 3);  // x, y, z
    instancedMesh.registerInstancedBuffer('color', 4);     // r, g, b, a

    // Update buffer every frame (GPU-efficient)
    this.scene.onBeforeRenderObservable.add(() => {
      const positions = forceEngine.getPositionsArray();
      instancedMesh.instancedBuffers.position = positions;
    });
  }
}
```

**Vircadia Network Optimization for Quest 3**:

```typescript
// Bandwidth throttling (Quest 3 Wi-Fi 6E: ~5-10 Mbps)
class NetworkOptimizer {
  private readonly TARGET_BANDWIDTH = 5 * 1024 * 1024 / 8;  // 5 Mbps in bytes/sec
  private currentBandwidth = 0;

  shouldSendUpdate(entityId: string): boolean {
    // Spatial interest management (only sync nearby entities)
    const entity = entityManager.get(entityId);
    const camera = scene.activeCamera;
    const distance = Vector3.Distance(entity.position, camera.position);

    if (distance > 50) return false;  // Beyond interest radius

    // Bandwidth check
    const updateSize = this.estimateUpdateSize(entity);
    if (this.currentBandwidth + updateSize > this.TARGET_BANDWIDTH) {
      return false;  // Skip update this frame
    }

    this.currentBandwidth += updateSize;
    return true;
  }

  resetBandwidth(): void {
    this.currentBandwidth = 0;  // Reset every second
  }
}

setInterval(() => optimizer.resetBandwidth(), 1000);
```

---

## Deployment

### Docker Compose Setup

```yaml
# server.docker.compose.yml
name: vircadia-world-server

networks:
  vircadia_internal_network:
    driver: bridge
    internal: true
  vircadia_public_network:
    driver: bridge

volumes:
  postgres_data:
    name: "vircadia_world_server_postgres_data"

services:
  # PostgreSQL 17.5
  vircadia_world_postgres:
    image: postgres:17.5-alpine3.21
    container_name: vircadia_world_postgres
    user: "70:70"
    restart: always
    environment:
      POSTGRES_DB: vircadia_world
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD}
    command: [
      "postgres",
      "-c", "wal_level=logical",
      "-c", "max_wal_size=4GB",
      "-c", "checkpoint_timeout=5min"
    ]
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d vircadia_world"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - vircadia_internal_network
      - vircadia_public_network

  # World API Manager (WebSocket Server)
  vircadia_world_api_manager:
    image: oven/bun:1.2.17-alpine
    container_name: vircadia_world_api_manager
    user: "1000:1000"
    restart: always
    ports:
      - "0.0.0.0:3000:3000"
    volumes:
      - ./api/volume/app:/app
    working_dir: /app
    command: ["bun", "run", "dist/world.api.manager.js"]
    environment:
      VRCA_SERVER_SERVICE_POSTGRES_HOST_CONTAINER_BIND_EXTERNAL: vircadia_world_postgres
      VRCA_SERVER_SERVICE_POSTGRES_PORT_CONTAINER_BIND_EXTERNAL: 5432
      VRCA_SERVER_SERVICE_POSTGRES_DATABASE: vircadia_world
      VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD: ${VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD}
    depends_on:
      vircadia_world_postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "wget --spider http://127.0.0.1:3000/stats"]
      interval: 10s
    networks:
      - vircadia_internal_network
      - vircadia_public_network

  # World State Manager (Tick Processor)
  vircadia_world_state_manager:
    image: oven/bun:1.2.17-alpine
    container_name: vircadia_world_state_manager
    user: "1000:1000"
    restart: always
    ports:
      - "0.0.0.0:3001:3001"
    volumes:
      - ./state/volume/app:/app
    working_dir: /app
    command: ["bun", "run", "dist/world.state.manager.js"]
    environment:
      VRCA_SERVER_SERVICE_POSTGRES_HOST_CONTAINER_BIND_EXTERNAL: vircadia_world_postgres
      VRCA_SERVER_SERVICE_POSTGRES_PORT_CONTAINER_BIND_EXTERNAL: 5432
      VRCA_SERVER_SERVICE_POSTGRES_DATABASE: vircadia_world
      VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD: ${VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD}
    depends_on:
      vircadia_world_postgres:
        condition: service_healthy
    networks:
      - vircadia_internal_network
      - vircadia_public_network
```

### Environment Configuration

```.env
# Container naming
VRCA_SERVER_CONTAINER_NAME=vircadia-world-server

# PostgreSQL
VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD=vircadia_dev_password_2025
VRCA_SERVER_SERVICE_POSTGRES_AGENT_PROXY_USER_PASSWORD=agent_proxy_password_2025

# Network ports
# API Manager (WebSocket)
PORT_API_MANAGER=3000
# State Manager (Tick System)
PORT_STATE_MANAGER=3001
```

### Startup Commands

```bash
# Navigate to server directory
cd vircadia-world/server/service

# Start all services
docker-compose -f server.docker.compose.yml up -d

# Check status
docker-compose -f server.docker.compose.yml ps

# View logs
docker logs -f vircadia_world_api_manager
docker logs -f vircadia_world_state_manager

# Stop services
docker-compose -f server.docker.compose.yml down
```

---

## API Reference

### WebSocket Connection

```typescript
const ws = new WebSocket(`ws://localhost:3000/world/ws?token=${jwt}&provider=system`);

ws.onopen = () => {
  console.log('Connected to Vircadia World Server');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'SESSION_INFO_RESPONSE':
      const { agentId, sessionId } = message.payload;
      console.log(`Session started: ${sessionId}, Agent: ${agentId}`);
      break;

    case 'SYNC_GROUP_UPDATES':
      // Binary entity updates (60 TPS)
      handleEntityUpdates(message.payload);
      break;

    case 'QUERY_RESPONSE':
      handleQueryResult(message.payload);
      break;
  }
};
```

### Entity Operations

```typescript
// Create graph node entity
function createGraphNode(id: string, position: {x, y, z}, metadata: any) {
  ws.send(JSON.stringify({
    type: 'QUERY_REQUEST',
    requestId: crypto.randomUUID(),
    timestamp: Date.now(),
    payload: {
      query: `
        INSERT INTO entity.entities (
          general__entity_name,
          group__sync,
          general__entity_type,
          spatial__transform,
          meta__data
        ) VALUES ($1, $2, $3, $4, $5)
        RETURNING *
      `,
      parameters: [
        id,
        'public.NORMAL',
        'SHAPE_SPHERE',
        {
          translation: position,
          rotation: {x: 0, y: 0, z: 0, w: 1},
          scaling: {x: 0.15, y: 0.15, z: 0.15}
        },
        {
          entityType: 'graph_node',
          ...metadata
        }
      ]
    }
  }));
}

// Update node position
function updateNodePosition(nodeId: string, newPosition: {x, y, z}) {
  ws.send(JSON.stringify({
    type: 'QUERY_REQUEST',
    requestId: crypto.randomUUID(),
    timestamp: Date.now(),
    payload: {
      query: `
        UPDATE entity.entities
        SET
          spatial__transform = jsonb_set(
            spatial__transform,
            '{translation}',
            $2::jsonb
          ),
          general__updated_at = NOW()
        WHERE general__entity_name = $1
      `,
      parameters: [nodeId, newPosition]
    }
  }));
}

// Delete entity
function deleteEntity(entityId: string) {
  ws.send(JSON.stringify({
    type: 'QUERY_REQUEST',
    requestId: crypto.randomUUID(),
    timestamp: Date.now(),
    payload: {
      query: 'DELETE FROM entity.entities WHERE general__entity_name = $1',
      parameters: [entityId]
    }
  }));
}
```

---

## Resources

### Official Vircadia Documentation

- **Main Documentation**: [https://docs.vircadia.com](https://docs.vircadia.com)
- **Entity System**: [https://docs.vircadia.com/explore/entities](https://docs.vircadia.com/explore/entities)
- **WebSocket API**: [https://docs.vircadia.com/api/websocket-protocol](https://docs.vircadia.com/api/websocket-protocol)
- **Sync Groups**: [https://docs.vircadia.com/api/sync-groups](https://docs.vircadia.com/api/sync-groups)
- **Server Architecture**: [https://docs.vircadia.com/server/architecture](https://docs.vircadia.com/server/architecture)

### GitHub Repositories

- **Vircadia World Server**: [https://github.com/vircadia/vircadia-world](https://github.com/vircadia/vircadia-world)
- **Vircadia Web SDK**: [https://github.com/vircadia/vircadia-web-sdk](https://github.com/vircadia/vircadia-web-sdk)

### Community

- **Discord**: [https://discord.gg/vircadia](https://discord.gg/vircadia)
- **Forums**: [https://forums.vircadia.com](https://forums.vircadia.com)

---

**Last Updated**: 2025-10-27
**Maintainer**: VisionFlow Engineering Team
**License**: Apache 2.0
