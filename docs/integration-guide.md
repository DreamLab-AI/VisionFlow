# Integration Guide

This guide walks through integrating VisionFlow with the Vircadia immersive experience platform. Each section covers a specific integration point with configuration details and code examples.

## Prerequisites

- A running Vircadia World Server (see [Docker Setup](#starting-the-vircadia-world-server))
- A VisionFlow client build environment (React 19, TypeScript, Three.js)
- Environment variables configured for Vircadia features

## Starting the Vircadia World Server

Launch the server alongside the existing VisionFlow stack:

```bash
docker-compose -f docker-compose.yml -f docker-compose.vircadia.yml --profile dev up -d
```

This starts:
- **Vircadia World Server** on port `3020` (WebSocket) and `3021` (HTTP health)
- **PostgreSQL 15** on port `5432`

Verify the server is healthy:

```bash
curl http://localhost:3021/health
```

## Connecting to a Vircadia World Server

### 1. Configure the Client

Set the required environment variables in your `.env` file:

```env
VITE_VIRCADIA_ENABLED=true
VITE_VIRCADIA_SERVER_URL=ws://localhost:3020/world/ws
VITE_VIRCADIA_AUTH_TOKEN=your-auth-token
VITE_VIRCADIA_AUTH_PROVIDER=system
```

### 2. Create a ClientCore Instance

```typescript
import { ClientCore } from './services/vircadia/VircadiaClientCore';

const client = new ClientCore({
    serverUrl: import.meta.env.VITE_VIRCADIA_SERVER_URL,
    authToken: import.meta.env.VITE_VIRCADIA_AUTH_TOKEN,
    authProvider: import.meta.env.VITE_VIRCADIA_AUTH_PROVIDER,
    reconnectAttempts: 5,
    reconnectDelay: 5000,
    debug: true
});
```

### 3. Connect and Listen for Events

```typescript
// Connect with a timeout
const connectionInfo = await client.Utilities.Connection.connect({ timeoutMs: 10000 });
console.log('Connected:', connectionInfo.agentId, connectionInfo.sessionId);

// Listen for connection state changes
client.Utilities.Connection.addEventListener('statusChange', () => {
    const info = client.Utilities.Connection.getConnectionInfo();
    console.log('Connection status:', info.status);
});

// Listen for sync updates from the server
client.Utilities.Connection.addEventListener('syncUpdate', () => {
    console.log('Entities updated on server');
});
```

### 4. Execute SQL Queries

All data access uses parameterized SQL over WebSocket:

```typescript
// Read entities
const result = await client.Utilities.Connection.query<{ result: any[] }>({
    query: 'SELECT * FROM entity.entities WHERE group__sync = $1 LIMIT $2',
    parameters: ['public.NORMAL', 50],
    timeoutMs: 5000
});

// Write entities
await client.Utilities.Connection.query({
    query: `INSERT INTO entity.entities (general__entity_name, general__semantic_version, group__sync, meta__data)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (general__entity_name) DO UPDATE SET meta__data = EXCLUDED.meta__data`,
    parameters: ['my_entity', '1.0.0', 'public.NORMAL', JSON.stringify({ type: 'custom', data: {} })],
    timeoutMs: 5000
});
```

### 5. Disconnect

```typescript
client.Utilities.Connection.disconnect();
// or
client.dispose();
```

---

## Setting Up Avatars with Three.js

### 1. Initialize the Avatar Renderer

```typescript
import * as THREE from 'three';
import { ThreeJSAvatarRenderer } from './services/vircadia/ThreeJSAvatarRenderer';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

const avatarRenderer = new ThreeJSAvatarRenderer(scene, client, camera, {
    modelUrl: '/assets/avatars/default-avatar.glb',
    scale: 1.0,
    showNameplate: true,
    nameplateDistance: 15.0,
    enableAnimations: true
});
```

### 2. Create a Local Avatar

After connecting, create the local user's avatar:

```typescript
client.Utilities.Connection.addEventListener('statusChange', () => {
    const info = client.Utilities.Connection.getConnectionInfo();
    if (info.isConnected && info.agentId) {
        avatarRenderer.createLocalAvatar('PlayerOne');
    }
});
```

The local avatar automatically broadcasts its position to the server every 100ms, using the camera's current transform.

### 3. Update the Render Loop

Call `update()` each frame to advance avatar animations:

```typescript
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    avatarRenderer.update(delta);
    renderer.render(scene, camera);
}
animate();
```

### 4. Clean Up

```typescript
avatarRenderer.dispose();
```

---

## Enabling Spatial Audio

### 1. Initialize the Spatial Audio Manager

```typescript
import { SpatialAudioManager } from './services/vircadia/SpatialAudioManager';

const audioManager = new SpatialAudioManager(client, scene, {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
    ],
    maxDistance: 20,
    rolloffFactor: 1,
    refDistance: 1
});

await audioManager.initialize();
```

The `initialize()` call requests microphone access and creates the Web Audio context.

### 2. Connect to Peers

When a remote user is detected, establish a peer connection:

```typescript
// Connect to a specific peer
await audioManager.connectToPeer(remoteAgentId, remoteUsername);
```

### 3. Update Listener and Peer Positions

Update positions each frame to maintain spatial audio accuracy:

```typescript
function animate() {
    requestAnimationFrame(animate);

    // Update listener (local user) position
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    audioManager.updateListenerPosition(camera.position, forward, camera.up);

    // Update remote peer positions
    for (const avatar of avatarRenderer.getAvatars()) {
        if (avatar.agentId !== localAgentId) {
            audioManager.updatePeerPosition(avatar.agentId, avatar.position);
        }
    }
}
```

### 4. Mute Controls

```typescript
audioManager.setMuted(true);     // Mute
audioManager.setMuted(false);    // Unmute
const isMuted = audioManager.toggleMute();  // Toggle
```

---

## Using Collaborative Graph Editing

### 1. Initialize Collaborative Sync

```typescript
import { CollaborativeGraphSync } from './services/vircadia/CollaborativeGraphSync';

const collab = new CollaborativeGraphSync(scene, client, {
    enableAnnotations: true,
    enableFiltering: true,
    enableVRPresence: true
});

await collab.initialize();
```

### 2. Share Node Selections

When a user selects graph nodes, broadcast the selection:

```typescript
// Select nodes
await collab.selectNodes(['node_123', 'node_456']);

// View other users' selections
const selections = collab.getActiveSelections();
for (const sel of selections) {
    console.log(`${sel.username} selected ${sel.nodeIds.length} nodes`);
}
```

### 3. Create Annotations

Attach text annotations to graph nodes:

```typescript
const nodePosition = new THREE.Vector3(1, 2, 3);
await collab.createAnnotation('node_123', 'Important discovery', nodePosition);

// Query annotations
const annotations = collab.getNodeAnnotations('node_123');

// Delete an annotation
await collab.deleteAnnotation(annotations[0].id);
```

### 4. Share Filter State

Synchronize graph filters across users:

```typescript
await collab.updateFilterState({
    searchQuery: 'machine learning',
    categoryFilter: ['research', 'paper'],
    timeRange: { start: Date.now() - 86400000, end: Date.now() }
});
```

### 5. Broadcast User Presence

For non-VR clients, broadcast cursor position:

```typescript
collab.broadcastUserPosition(camera.position, camera.quaternion);
```

For VR clients with hand tracking:

```typescript
collab.broadcastVRPresence(
    headPosition, headRotation,
    leftHandPosition, leftHandRotation,
    rightHandPosition, rightHandRotation
);
```

---

## Synchronizing Graph Entities

### 1. Initialize the Entity Sync Manager

```typescript
import { EntitySyncManager } from './services/vircadia/EntitySyncManager';

const syncManager = new EntitySyncManager(client, {
    syncGroup: 'public.NORMAL',
    batchSize: 100,
    syncIntervalMs: 100,
    enableRealTimePositions: true
});
```

### 2. Push a Graph to Vircadia

```typescript
const graphData = {
    nodes: [
        { id: 'n1', label: 'Node A', x: 0, y: 1, z: 0 },
        { id: 'n2', label: 'Node B', x: 2, y: 0, z: 1 }
    ],
    edges: [
        { id: 'e1', source: 'n1', target: 'n2', label: 'relates_to' }
    ]
};

await syncManager.pushGraphToVircadia(graphData);
```

### 3. Pull a Graph from Vircadia

```typescript
const graph = await syncManager.pullGraphFromVircadia();
console.log(`Loaded ${graph.nodes.length} nodes, ${graph.edges.length} edges`);
```

### 4. Stream Position Updates

Queue position updates that are batched and flushed automatically:

```typescript
syncManager.updateNodePosition('n1', { x: 1.5, y: 2.0, z: 0.5 });
```

### 5. Subscribe to Entity Changes

```typescript
const unsubscribe = syncManager.onEntityUpdate((entities) => {
    console.log(`Received ${entities.length} updated entities`);
});

// Later: unsubscribe when done
unsubscribe();
```

---

## XR / Quest 3 Setup

### 1. Check Feature Flags

```typescript
import { featureFlags } from './services/vircadia/FeatureFlags';

if (!featureFlags.isVircadiaEnabled()) {
    console.log('Vircadia features are disabled');
    return;
}

if (featureFlags.isHandTrackingEnabled()) {
    console.log('Hand tracking is available');
}
```

### 2. Initialize the Quest 3 Optimizer

```typescript
import { Quest3Optimizer } from './services/vircadia/Quest3Optimizer';

const quest3 = new Quest3Optimizer(scene, renderer, client, {
    targetFrameRate: 90,
    enableHandTracking: true,
    enableControllers: true,
    foveatedRenderingLevel: 2,
    dynamicResolutionScale: true,
    minResolutionScale: 0.5,
    maxResolutionScale: 1.0
});
```

### 3. Start an XR Session

```typescript
const xrSession = await navigator.xr.requestSession('immersive-vr', {
    requiredFeatures: ['local-floor'],
    optionalFeatures: ['hand-tracking', 'layers']
});

await quest3.initialize(xrSession);
```

### 4. Monitor Performance

```typescript
const metrics = quest3.getPerformanceMetrics();
console.log(`FPS: ${metrics.fps}/${metrics.targetFPS}, Pixel Ratio: ${metrics.pixelRatio}`);
```

The Quest3Optimizer automatically adjusts resolution to maintain the target frame rate.

---

## Feature Flags Configuration

### Environment Variables

Feature flags default based on environment variables:

| Variable | Feature |
|:---------|:--------|
| `VITE_VIRCADIA_ENABLED` | Master Vircadia toggle |
| `VITE_VIRCADIA_ENABLE_MULTI_USER` | Multi-user mode |
| `VITE_VIRCADIA_ENABLE_SPATIAL_AUDIO` | Spatial audio |
| `VITE_QUEST3_ENABLE_HAND_TRACKING` | Quest 3 hand tracking |
| `VITE_INSTANCED_RENDERING` | Instanced rendering optimization |

### Runtime Configuration

Override flags at runtime:

```typescript
import { featureFlags } from './services/vircadia/FeatureFlags';

// Enable specific features
featureFlags.updateConfig({
    vircadiaEnabled: true,
    multiUserEnabled: true,
    spatialAudioEnabled: true,
    deltaCompressionEnabled: true
});

// Staged rollout to 25% of users
featureFlags.setRolloutPercentage(25);

// Allow-list specific users
featureFlags.enableForUsers(['user-abc', 'user-def']);

// Enable everything
featureFlags.enableAll();

// Reset to defaults
featureFlags.reset();

// Inspect current state
console.log(featureFlags.getStatusReport());
```

Feature flags persist to `localStorage` under the key `vircadia_feature_flags`.

---

## Network Optimization

### Initialize the Network Optimizer

```typescript
import { NetworkOptimizer } from './services/vircadia/NetworkOptimizer';

const optimizer = new NetworkOptimizer(client, {
    batchIntervalMs: 100,
    maxBatchSize: 100,
    compressionEnabled: true,
    adaptiveQuality: true,
    bandwidthTargetKbps: 5000
});
```

### Queue Position Updates

```typescript
// Instead of sending individual SQL updates, queue them
optimizer.queuePositionUpdate('node_123', 1.5, 2.0, 0.5);
optimizer.queuePositionUpdate('node_456', 3.0, 1.0, 2.0);
// Updates are batched and flushed automatically
```

### Monitor Network Health

```typescript
const stats = optimizer.getStats();
console.log(`Bandwidth: ${stats.currentBandwidthKbps.toFixed(1)} kbps`);
console.log(`Compression: ${stats.compressionRatio.toFixed(2)}x`);
console.log(`Latency: ${stats.averageLatency.toFixed(0)}ms`);
```

When `adaptiveQuality` is enabled, the optimizer automatically adjusts the batch interval to stay within the configured bandwidth target.
