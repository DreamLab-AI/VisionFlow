---
layout: default
title: Phase 6 Integration Guide
description: Multi-user sync integration guide for collaborative graph editing
nav_exclude: true
---

# Phase 6: Multi-User Sync - Integration Guide

## Quick Start

### 1. Client-Side Integration

#### Basic Usage (Desktop)
```typescript
import { CollaborativeGraphSync } from './services/vircadia/CollaborativeGraphSync';

// Initialize sync
const sync = new CollaborativeGraphSync(scene, vircadiaClient);
await sync.initialize();

// Broadcast user position (call in render loop)
scene.onBeforeRenderObservable.add(() => {
    sync.broadcastUserPosition(camera.position, camera.absoluteRotation);
});

// Select nodes
await sync.selectNodes(['node1', 'node2', 'node3']);

// Create annotation
await sync.createAnnotation('node1', 'Important note!', new BABYLON.Vector3(0, 1, 0));

// Get active users
const presence = sync.getUserPresence(); // Returns UserPresence[]
console.log(`${presence.length} users online`);
```

#### VR Integration (Quest 3)
```typescript
// In WebXR session
const xrSession = await navigator.xr.requestSession('immersive-vr');

xrSession.requestAnimationFrame(function onXRFrame(time, frame) {
    const pose = frame.getViewerPose(referenceSpace);
    const inputSources = xrSession.inputSources;

    // Get VR tracking data
    const headPos = pose.transform.position;
    const headRot = pose.transform.orientation;

    const leftHand = inputSources.find(s => s.handedness === 'left');
    const rightHand = inputSources.find(s => s.handedness === 'right');

    // Broadcast VR presence
    sync.broadcastVRPresence(
        new BABYLON.Vector3(headPos.x, headPos.y, headPos.z),
        new BABYLON.Quaternion(headRot.x, headRot.y, headRot.z, headRot.w),
        leftHand ? new BABYLON.Vector3(...leftHand.gripSpace.position) : undefined,
        leftHand ? new BABYLON.Quaternion(...leftHand.gripSpace.orientation) : undefined,
        rightHand ? new BABYLON.Vector3(...rightHand.gripSpace.position) : undefined,
        rightHand ? new BABYLON.Quaternion(...rightHand.gripSpace.orientation) : undefined
    );

    xrSession.requestAnimationFrame(onXRFrame);
});
```

### 2. Server-Side Integration

#### Add Route (Rust)
```rust
// In src/main.rs or routing configuration
use crate::handlers::collaborative_sync_handler::{
    CollaborativeSyncActor, SyncManager
};

// Create global sync manager
let sync_manager = web::Data::new(Arc::new(Mutex::new(SyncManager::new())));

HttpServer::new(move || {
    App::new()
        .app_data(sync_manager.clone())
        .route("/ws/sync/{user_id}", web::get().to(websocket_sync_handler))
})
```

#### WebSocket Handler
```rust
async fn websocket_sync_handler(
    req: HttpRequest,
    stream: web::Payload,
    path: web::Path<String>,
    sync_manager: web::Data<Arc<Mutex<SyncManager>>>,
) -> Result<HttpResponse, Error> {
    let user_id = path.into_inner();
    let actor = CollaborativeSyncActor::new(user_id.clone());

    let resp = ws::start(actor, &req, stream)?;

    // Register connection in sync manager
    let mut manager = sync_manager.lock().unwrap();
    manager.add_connection(user_id, resp.clone());

    Ok(resp)
}
```

### 3. Configuration Options

```typescript
const config: Partial<CollaborativeConfig> = {
    // Visual settings
    highlightColor: new BABYLON.Color3(0.2, 0.8, 0.3), // Green
    annotationColor: new BABYLON.Color3(1.0, 0.8, 0.2), // Yellow

    // Behavior
    selectionTimeout: 30000, // 30 seconds
    enableAnnotations: true,
    enableFiltering: true,
    enableVRPresence: true, // Enable VR hand tracking
};

const sync = new CollaborativeGraphSync(scene, client, config);
```

## API Reference

### CollaborativeGraphSync

#### Methods

**`initialize(): Promise<void>`**
- Initializes WebSocket connection
- Subscribes to sync channels
- Loads existing annotations

**`selectNodes(nodeIds: string[]): Promise<void>`**
- Broadcasts node selection to other users
- Updates local selection state
- Triggers highlight rendering

**`updateFilterState(filterState: FilterState): Promise<void>`**
- Syncs filter settings across users
- Updates local filter state

**`createAnnotation(nodeId: string, text: string, position: BABYLON.Vector3): Promise<void>`**
- Creates and broadcasts annotation
- Renders annotation mesh
- Persists to sync state

**`deleteAnnotation(annotationId: string): Promise<void>`**
- Deletes annotation (owner only)
- Broadcasts deletion
- Removes mesh

**`broadcastUserPosition(position: BABYLON.Vector3, rotation: BABYLON.Quaternion): void`**
- Sends user cursor position
- Call in render loop (auto-throttled)

**`broadcastVRPresence(...): void`**
- Sends VR head + hand positions
- Call in XR render loop
- Supports both hands independently

**`getActiveSelections(): UserSelection[]`**
- Returns all user selections

**`getAnnotations(): GraphAnnotation[]`**
- Returns all annotations

**`getNodeAnnotations(nodeId: string): GraphAnnotation[]`**
- Returns annotations for specific node

**`getUserPresence(): UserPresence[]`**
- Returns all active users

**`dispose(): void`**
- Cleans up resources
- Removes event listeners
- Disposes meshes

### Interfaces

#### UserPresence
```typescript
interface UserPresence {
    userId: string;
    username: string;
    position: BABYLON.Vector3;
    rotation: BABYLON.Quaternion;

    // VR tracking (optional)
    headPosition?: BABYLON.Vector3;
    headRotation?: BABYLON.Quaternion;
    leftHandPosition?: BABYLON.Vector3;
    leftHandRotation?: BABYLON.Quaternion;
    rightHandPosition?: BABYLON.Vector3;
    rightHandRotation?: BABYLON.Quaternion;

    lastUpdate: number;
}
```

#### GraphOperation
```typescript
interface GraphOperation {
    id: string;
    type: 'node_move' | 'node_add' | 'node_delete' | 'edge_add' | 'edge_delete';
    userId: string;
    nodeId?: string;
    position?: { x: number; y: number; z: number };
    timestamp: number;
    version: number;
}
```

#### GraphAnnotation
```typescript
interface GraphAnnotation {
    id: string;
    agentId: string;
    username: string;
    nodeId: string;
    text: string;
    position: { x: number; y: number; z: number };
    timestamp: number;
}
```

## Message Protocol

### Binary Message Format

All messages follow this structure:
```
[1 byte: Message Type][N bytes: Payload]
```

### Message Types

| Type | Hex | Name | Payload |
|------|-----|------|---------|
| 0x50 | 80 | SYNC_UPDATE | Graph operation |
| 0x51 | 81 | ANNOTATION_UPDATE | JSON annotation |
| 0x52 | 82 | SELECTION_UPDATE | JSON selection |
| 0x53 | 83 | USER_POSITION | 32 bytes (pos + rot + id) |
| 0x54 | 84 | VR_PRESENCE | 87 bytes (head + hands) |

### SYNC_UPDATE Payload
```
[1 byte: op_type]
[36 bytes: user_id (UUID)]
[2 bytes: node_id_length (u16 LE)]
[N bytes: node_id (UTF-8)]
[12 bytes: position (optional, 3x f32 LE)]
```

### VR_PRESENCE Payload (87 bytes)
```
[12 bytes: head_position (3x f32 LE)]
[16 bytes: head_rotation (4x f32 LE quaternion)]
[1 byte: hand_flags (bit 0=left, bit 1=right)]
[29 bytes: left_hand (12 pos + 16 rot + 1 padding)]
[29 bytes: right_hand (12 pos + 16 rot)]
```

## Performance Tips

### 1. Throttling
Position updates are auto-throttled to 60 FPS:
```typescript
// Called every frame, but sent max 60 times/second
scene.onBeforeRenderObservable.add(() => {
    sync.broadcastUserPosition(camera.position, camera.absoluteRotation);
});
```

### 2. Batching
Operations within 16ms are batched automatically:
```typescript
// These will be combined into one message
await sync.selectNodes(['node1']);
await sync.selectNodes(['node2']);
await sync.selectNodes(['node3']);
```

### 3. Selective Sync
Subscribe only to needed channels:
```typescript
// Minimal setup (no annotations or filters)
const sync = new CollaborativeGraphSync(scene, client, {
    enableAnnotations: false,
    enableFiltering: false,
    enableVRPresence: false
});
```

## Troubleshooting

### No sync updates received
1. Check WebSocket connection:
   ```typescript
   const ws = client.Utilities.Connection.getWebSocket();
   console.log('WebSocket state:', ws?.readyState);
   // Should be 1 (OPEN)
   ```

2. Verify subscriptions:
   ```typescript
   // Should see debug logs:
   // "Subscribed to channel: graph_sync"
   // "Subscribed to channel: user_presence"
   ```

### Conflict resolution not working
1. Check operation timestamps:
   ```typescript
   // Operations must have unique timestamps or userIds
   const op = { timestamp: Date.now(), userId: 'unique-id' };
   ```

2. Verify pending queue:
   ```typescript
   // Operations within 1 second window are checked for conflicts
   ```

### VR hands not visible
1. Enable VR presence:
   ```typescript
   const sync = new CollaborativeGraphSync(scene, client, {
       enableVRPresence: true
   });
   ```

2. Check hand tracking:
   ```typescript
   const leftHand = inputSources.find(s => s.handedness === 'left');
   if (!leftHand?.gripSpace) {
       console.warn('Left hand not tracked');
   }
   ```

## Migration from Polling

### Before (Polling)
```typescript
// Old code with 1-second polling
setInterval(async () => {
    await fetchRemoteSelections();
    await fetchAnnotations();
}, 1000);
```

### After (WebSocket)
```typescript
// New code with real-time sync
const sync = new CollaborativeGraphSync(scene, client);
await sync.initialize();
// That's it! Updates are automatic via WebSocket
```

### Compatibility
The new system is backward compatible:
- Same public API
- No breaking changes
- Works alongside existing VircadiaClientCore

## Examples

### Example 1: Multi-User Node Editor
```typescript
// User A moves node
const nodeId = 'node_123';
const newPos = new BABYLON.Vector3(10, 20, 30);

// Update local
nodeMesh.position = newPos;

// Sync to others
sync.selectNodes([nodeId]);

// User B sees update instantly via WebSocket
```

### Example 2: VR Collaboration
```typescript
// User A in VR headset
sync.broadcastVRPresence(headPos, headRot, leftHandPos, leftHandRot, rightHandPos, rightHandRot);

// User B on desktop sees:
// - User A's head as colored sphere
// - User A's left hand as red sphere
// - User A's right hand as blue sphere
```

### Example 3: Annotation System
```typescript
// User A creates annotation
await sync.createAnnotation(
    'node_456',
    'This needs review',
    new BABYLON.Vector3(5, 5, 5)
);

// User B sees annotation immediately
const annotations = sync.getNodeAnnotations('node_456');
console.log(annotations[0].text); // "This needs review"
```

---

**Last Updated**: 2025-12-25
**Version**: 1.0.0
**Status**: Production Ready ✅
