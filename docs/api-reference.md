# API Reference

This document provides a complete reference for the public APIs of each service in the VisionFlow Vircadia integration.

## VircadiaClientCore

The foundation service that manages the WebSocket connection to a Vircadia World Server and provides a SQL query interface.

### Types

```typescript
type ClientCoreConnectionState =
    | "connected"
    | "connecting"
    | "reconnecting"
    | "disconnected";

interface ClientCoreConfig {
    serverUrl: string;           // WebSocket URL (e.g., "wss://host:3020/world/ws")
    authToken: string;           // Authentication token
    authProvider: string;        // Auth provider name (e.g., "system", "nostr")
    reconnectAttempts?: number;  // Max reconnect attempts (default: 5)
    reconnectDelay?: number;     // Delay between reconnects in ms (default: 5000)
    debug?: boolean;             // Enable debug logging
    suppress?: boolean;          // Suppress all logging
}

interface QueryOptions {
    query: string;               // SQL query string with $1, $2, ... parameters
    parameters?: unknown[];      // Parameterized values
    timeoutMs?: number;          // Query timeout in ms (default: 10000)
}

interface QueryResult<T = unknown> {
    success: boolean;
    result?: T;
    errorMessage?: string;
    timestamp: number;
}

interface ClientCoreConnectionInfo {
    status: ClientCoreConnectionState;
    isConnected: boolean;
    isConnecting: boolean;
    isReconnecting: boolean;
    connectionDuration?: number;
    reconnectAttempts: number;
    pendingRequests: Array<{ requestId: string; elapsedMs: number }>;
    agentId: string | null;
    sessionId: string | null;
}
```

### ClientCore

```typescript
class ClientCore {
    constructor(config: ClientCoreConfig);

    // Access connection utilities
    get Utilities(): {
        Connection: {
            connect(options?: { timeoutMs?: number }): Promise<ClientCoreConnectionInfo>;
            disconnect(): void;
            query<T = unknown>(options: QueryOptions): Promise<QueryResult<T>>;
            getConnectionInfo(): ClientCoreConnectionInfo;
            addEventListener(event: string, listener: () => void): void;
            removeEventListener(event: string, listener: () => void): void;
        };
    };

    // Clean up all resources
    dispose(): void;
}
```

### Events

| Event | Description |
|:------|:------------|
| `statusChange` | Fired when connection state changes |
| `syncUpdate` | Fired when server sends `SYNC_GROUP_UPDATES_RESPONSE` |
| `tick` | Fired when server sends `TICK_NOTIFICATION_RESPONSE` |
| `error` | Fired when server sends `GENERAL_ERROR_RESPONSE` |

---

## ThreeJSAvatarRenderer

Manages 3D avatar rendering, GLTF model loading, position broadcasting, and nameplate display.

### Types

```typescript
interface AvatarConfig {
    modelUrl: string;            // Path to GLTF/GLB avatar model
    scale: number;               // Avatar scale factor (default: 1.0)
    showNameplate: boolean;      // Display username nameplates (default: true)
    nameplateDistance: number;    // Max nameplate visibility distance (default: 10.0)
    enableAnimations: boolean;   // Enable animation playback (default: true)
}

interface UserAvatar {
    agentId: string;
    username: string;
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    mesh?: THREE.Object3D;
    nameplate?: THREE.Sprite;
    mixer?: THREE.AnimationMixer;
    animations?: THREE.AnimationClip[];
}
```

### ThreeJSAvatarRenderer

```typescript
class ThreeJSAvatarRenderer {
    constructor(
        scene: THREE.Scene,
        client: ClientCore,
        camera: THREE.Camera,
        config?: Partial<AvatarConfig>
    );

    // Create the local user's avatar tied to the camera
    createLocalAvatar(username: string): Promise<void>;

    // Load a remote user's GLTF avatar into the scene
    loadRemoteAvatar(agentId: string, username: string): Promise<void>;

    // Update an avatar's 3D transform
    updateAvatarPosition(
        agentId: string,
        position: THREE.Vector3,
        rotation?: THREE.Quaternion
    ): void;

    // Advance animation mixers (call per frame)
    update(deltaTime: number): void;

    // Remove an avatar from the scene and dispose GPU resources
    removeAvatar(agentId: string): void;

    // Query all tracked avatars
    getAvatars(): UserAvatar[];
    getAvatarCount(): number;

    // Clean up all avatars and event listeners
    dispose(): void;
}
```

---

## SpatialAudioManager

Handles WebRTC peer connections for voice communication with HRTF-based spatial audio positioning.

### Types

```typescript
interface SpatialAudioConfig {
    iceServers: RTCIceServer[];          // ICE/STUN/TURN servers
    audioConstraints: MediaStreamConstraints['audio'];  // Microphone settings
    maxDistance: number;                  // Audio cutoff distance (default: 20)
    rolloffFactor: number;               // Distance attenuation (default: 1)
    refDistance: number;                  // Reference distance (default: 1)
}
```

### SpatialAudioManager

```typescript
class SpatialAudioManager {
    constructor(
        client: ClientCore,
        scene: THREE.Scene,
        config?: Partial<SpatialAudioConfig>
    );

    // Initialize audio context and acquire local microphone
    initialize(): Promise<void>;

    // Establish a WebRTC peer connection with a remote user
    connectToPeer(agentId: string, username: string): Promise<void>;

    // Update the listener position (typically from the camera)
    updateListenerPosition(
        position: THREE.Vector3,
        forward: THREE.Vector3,
        up: THREE.Vector3
    ): void;

    // Update a remote peer's 3D audio source position
    updatePeerPosition(agentId: string, position: THREE.Vector3): void;

    // Mute/unmute the local microphone
    setMuted(muted: boolean): void;
    toggleMute(): boolean;

    // Query peer count
    getPeerCount(): number;

    // Clean up streams, peer connections, and audio context
    dispose(): void;
}
```

### Signaling Flow

```mermaid
sequenceDiagram
    participant A as Caller
    participant SAM as SpatialAudioManager
    participant DB as Vircadia Entity Store
    participant B as Callee

    A->>SAM: connectToPeer(agentId, username)
    SAM->>SAM: RTCPeerConnection.createOffer()
    SAM->>DB: INSERT webrtc_offer entity
    B->>DB: Poll for signaling messages
    DB-->>B: Offer SDP
    B->>B: setRemoteDescription + createAnswer
    B->>DB: INSERT webrtc_answer entity
    A->>DB: Poll for signaling messages
    DB-->>A: Answer SDP
    A->>A: setRemoteDescription
    Note over A,B: ICE candidates exchanged similarly
    A<-->B: Direct P2P audio stream (HRTF spatialized)
```

---

## CollaborativeGraphSync

Provides real-time multi-user collaboration features: node selection sharing, graph annotations, user presence visualization, and operational transform conflict resolution.

### Types

```typescript
interface CollaborativeConfig {
    highlightColor: THREE.Color;     // Selection highlight color
    annotationColor: THREE.Color;    // Annotation text color
    selectionTimeout: number;        // Selection expiry in ms (default: 30000)
    enableAnnotations: boolean;      // Enable annotations (default: true)
    enableFiltering: boolean;        // Enable filter sharing (default: true)
    enableVRPresence: boolean;       // Enable VR hand tracking presence (default: true)
}

interface UserSelection {
    agentId: string;
    username: string;
    nodeIds: string[];
    timestamp: number;
    filterState?: FilterState;
}

interface FilterState {
    searchQuery?: string;
    categoryFilter?: string[];
    timeRange?: { start: number; end: number };
    customFilters?: Record<string, any>;
}

interface GraphAnnotation {
    id: string;
    agentId: string;
    username: string;
    nodeId: string;
    text: string;
    position: { x: number; y: number; z: number };
    timestamp: number;
}

interface UserPresence {
    userId: string;
    username: string;
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    headPosition?: THREE.Vector3;
    headRotation?: THREE.Quaternion;
    leftHandPosition?: THREE.Vector3;
    leftHandRotation?: THREE.Quaternion;
    rightHandPosition?: THREE.Vector3;
    rightHandRotation?: THREE.Quaternion;
    lastUpdate: number;
}

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

### CollaborativeGraphSync

```typescript
class CollaborativeGraphSync {
    constructor(
        scene: THREE.Scene,
        client: ClientCore,
        config?: Partial<CollaborativeConfig>
    );

    // Initialize WebSocket sync channels and load annotations
    initialize(): Promise<void>;

    // Broadcast local node selection to all peers
    selectNodes(nodeIds: string[]): Promise<void>;

    // Share filter state with all peers
    updateFilterState(filterState: FilterState): Promise<void>;

    // Create and broadcast a text annotation on a graph node
    createAnnotation(
        nodeId: string,
        text: string,
        position: THREE.Vector3
    ): Promise<void>;

    // Delete an annotation (only own annotations)
    deleteAnnotation(annotationId: string): Promise<void>;

    // Broadcast user position for presence visualization
    broadcastUserPosition(
        position: THREE.Vector3,
        rotation: THREE.Quaternion
    ): void;

    // Broadcast VR head and hand tracking data
    broadcastVRPresence(
        headPos: THREE.Vector3,
        headRot: THREE.Quaternion,
        leftHandPos?: THREE.Vector3,
        leftHandRot?: THREE.Quaternion,
        rightHandPos?: THREE.Vector3,
        rightHandRot?: THREE.Quaternion
    ): void;

    // Query state
    getActiveSelections(): UserSelection[];
    getAnnotations(): GraphAnnotation[];
    getNodeAnnotations(nodeId: string): GraphAnnotation[];
    getUserPresence(): UserPresence[];

    // Clean up all meshes and WebSocket listeners
    dispose(): void;
}
```

---

## EntitySyncManager

Manages bidirectional synchronization between VisionFlow graph data and Vircadia entity storage.

### Types

```typescript
interface SyncConfig {
    syncGroup: string;                 // Sync group name (default: "public.NORMAL")
    batchSize: number;                 // Insert batch size (default: 100)
    syncIntervalMs: number;            // Position flush interval (default: 100)
    enableRealTimePositions: boolean;  // Enable position streaming (default: true)
}

interface SyncStats {
    totalEntities: number;
    syncedNodes: number;
    syncedEdges: number;
    lastSyncTime: number;
    pendingUpdates: number;
    errors: number;
}
```

### EntitySyncManager

```typescript
class EntitySyncManager {
    constructor(client: ClientCore, config?: Partial<SyncConfig>);

    // Push a complete graph to Vircadia entity storage
    pushGraphToVircadia(graphData: GraphData): Promise<void>;

    // Pull graph data from Vircadia entity storage
    pullGraphFromVircadia(): Promise<GraphData>;

    // Queue a node position update for batched flush
    updateNodePosition(
        nodeId: string,
        position: { x: number; y: number; z: number }
    ): void;

    // Subscribe to entity changes (returns unsubscribe function)
    onEntityUpdate(
        callback: (entities: VircadiaEntity[]) => void
    ): () => void;

    // Delete a single entity by name
    deleteEntity(entityName: string): Promise<void>;

    // Clear all graph entities from Vircadia
    clearGraph(): Promise<void>;

    // Query sync statistics
    getStats(): SyncStats;

    // Clean up timers and pending updates
    dispose(): void;
}
```

---

## NetworkOptimizer

Provides delta compression, binary batching, and adaptive quality control for position updates.

### Types

```typescript
interface NetworkOptimizerConfig {
    batchIntervalMs: number;       // Batch flush interval (default: 100)
    maxBatchSize: number;          // Max updates per batch (default: 100)
    compressionEnabled: boolean;   // Enable delta compression (default: true)
    adaptiveQuality: boolean;      // Auto-adjust update rate (default: true)
    bandwidthTargetKbps: number;   // Bandwidth target in kbps (default: 5000)
}

interface PositionUpdate {
    entityName: string;
    x: number;
    y: number;
    z: number;
    timestamp: number;
}

interface DeltaCompressedUpdate {
    entityName: string;
    dx: number;    // Delta from last known position
    dy: number;
    dz: number;
    timestamp: number;
}

interface NetworkStats {
    bytesSent: number;
    bytesReceived: number;
    messagesSent: number;
    messagesReceived: number;
    compressionRatio: number;
    averageLatency: number;
    currentBandwidthKbps: number;
}
```

### NetworkOptimizer

```typescript
class NetworkOptimizer {
    constructor(client: ClientCore, config?: Partial<NetworkOptimizerConfig>);

    // Queue a position update for batched delivery
    queuePositionUpdate(entityName: string, x: number, y: number, z: number): void;

    // Decode a binary position batch
    decodePositionsFromBinary(buffer: ArrayBuffer): DeltaCompressedUpdate[];

    // Query network statistics
    getStats(): NetworkStats;

    // Reset all statistics
    resetStats(): void;

    // Clean up timers and pending updates
    dispose(): void;
}
```

---

## Quest3Optimizer

Provides Meta Quest 3 XR optimizations including foveated rendering, dynamic resolution scaling, hand tracking, and controller broadcasting.

### Types

```typescript
interface Quest3Config {
    targetFrameRate: 90 | 120;         // Target FPS (default: 90)
    enableHandTracking: boolean;       // Enable hand joint tracking (default: true)
    enableControllers: boolean;        // Enable controller input (default: true)
    foveatedRenderingLevel: 0|1|2|3;   // Fixed foveation level (default: 2)
    dynamicResolutionScale: boolean;   // Auto pixel ratio (default: true)
    minResolutionScale: number;        // Min pixel ratio (default: 0.5)
    maxResolutionScale: number;        // Max pixel ratio (default: 1.0)
}

interface HandJoint {
    name: string;
    position: THREE.Vector3;
    orientation: THREE.Quaternion;
}

interface HandTrackingData {
    agentId: string;
    hand: 'left' | 'right';
    joints: HandJoint[];
    timestamp: number;
}

interface ControllerState {
    agentId: string;
    controllerId: 'left' | 'right';
    position: THREE.Vector3;
    orientation: THREE.Quaternion;
    buttons: Record<string, boolean>;
    axes: Record<string, number>;
    timestamp: number;
}
```

### Quest3Optimizer

```typescript
class Quest3Optimizer {
    constructor(
        scene: THREE.Scene,
        renderer: THREE.WebGLRenderer,
        client: ClientCore,
        config?: Partial<Quest3Config>
    );

    // Initialize all XR optimizations for an active session
    initialize(xrSession: XRSession): Promise<void>;

    // Update remote hand tracking visualization
    updateRemoteHandTracking(
        agentId: string,
        hand: 'left' | 'right',
        joints: HandJoint[]
    ): void;

    // Update remote controller visualization
    updateRemoteController(agentId: string, state: ControllerState): void;

    // Get current performance metrics
    getPerformanceMetrics(): {
        fps: number;
        targetFPS: number;
        pixelRatio: number;
        foveationLevel: number;
        handTrackingActive: boolean;
        controllersActive: number;
    };

    // Clean up XR resources
    dispose(): void;
}
```

---

## FeatureFlags

Singleton service for runtime feature gating with rollout percentage support and persistent storage.

### Types

```typescript
interface FeatureFlagConfig {
    // Core features
    vircadiaEnabled: boolean;
    multiUserEnabled: boolean;
    spatialAudioEnabled: boolean;

    // Extended features
    handTrackingEnabled: boolean;
    collaborativeGraphEnabled: boolean;
    annotationsEnabled: boolean;

    // Performance features
    deltaCompressionEnabled: boolean;
    instancedRenderingEnabled: boolean;
    dynamicResolutionEnabled: boolean;
    foveatedRenderingEnabled: boolean;

    // Rollout control
    rolloutPercentage: number;       // 0-100 percentage
    allowedUserIds?: string[];
    allowedAgentIds?: string[];
}
```

### FeatureFlags

```typescript
class FeatureFlags {
    // Singleton access
    static getInstance(): FeatureFlags;

    // Configuration management
    updateConfig(updates: Partial<FeatureFlagConfig>): void;
    getConfig(): FeatureFlagConfig;

    // Feature checks
    isVircadiaEnabled(userId?: string, agentId?: string): boolean;
    isMultiUserEnabled(): boolean;
    isSpatialAudioEnabled(): boolean;
    isHandTrackingEnabled(): boolean;
    isCollaborativeGraphEnabled(): boolean;
    isAnnotationsEnabled(): boolean;
    isDeltaCompressionEnabled(): boolean;
    isInstancedRenderingEnabled(): boolean;
    isDynamicResolutionEnabled(): boolean;
    isFoveatedRenderingEnabled(): boolean;

    // Rollout management
    enableForUsers(userIds: string[]): void;
    enableForAgents(agentIds: string[]): void;
    setRolloutPercentage(percentage: number): void;

    // Bulk operations
    enableAll(): void;
    disableAll(): void;
    reset(): void;

    // Diagnostics
    getStatusReport(): string;
}

// Pre-instantiated singleton
const featureFlags: FeatureFlags;
```

---

## BinaryWebSocketProtocol

Singleton service for encoding and decoding compact binary messages over WebSocket, supporting Protocol V2 and V3.

### Constants

| Constant | Value | Description |
|:---------|:------|:------------|
| `PROTOCOL_V2` | `2` | Supported protocol version |
| `PROTOCOL_V3` | `3` | Current protocol version (analytics extension) |
| `MESSAGE_HEADER_SIZE` | `4` bytes | Standard message header |
| `GRAPH_UPDATE_HEADER_SIZE` | `5` bytes | Graph update header (includes graph type flag) |
| `AGENT_POSITION_SIZE` | `21` bytes | Per-agent position record |
| `AGENT_STATE_SIZE` | `49` bytes | Per-agent full state record |
| `VOICE_HEADER_SIZE` | `7` bytes | Voice chunk header |

### MessageType Enum

| Type | Code | Description |
|:-----|:-----|:------------|
| `GRAPH_UPDATE` | `0x01` | Graph node/edge data |
| `VOICE_DATA` | `0x02` | Voice audio data |
| `POSITION_UPDATE` | `0x10` | Agent position updates |
| `AGENT_POSITIONS` | `0x11` | Batch agent positions |
| `AGENT_STATE_FULL` | `0x20` | Full agent state |
| `AGENT_STATE_DELTA` | `0x21` | Delta agent state |
| `AGENT_ACTION` | `0x23` | Agent-to-data action |
| `CONTROL_BITS` | `0x30` | Client control flags |
| `HANDSHAKE` | `0x32` | Protocol handshake |
| `HEARTBEAT` | `0x33` | Keep-alive |
| `BROADCAST_ACK` | `0x34` | Backpressure flow control |
| `SYNC_UPDATE` | `0x50` | Graph operation sync |
| `ANNOTATION_UPDATE` | `0x51` | Annotation sync |
| `SELECTION_UPDATE` | `0x52` | Selection sync |
| `USER_POSITION` | `0x53` | User cursor position |
| `VR_PRESENCE` | `0x54` | VR head + hand tracking |
| `ERROR` | `0xFF` | Error message |

### BinaryWebSocketProtocol

```typescript
class BinaryWebSocketProtocol {
    static getInstance(): BinaryWebSocketProtocol;

    // Message creation and parsing
    createMessage(type: MessageType, payload: ArrayBuffer, graphTypeFlag?: GraphTypeFlag): ArrayBuffer;
    parseHeader(buffer: ArrayBuffer): MessageHeader | null;
    extractPayload(buffer: ArrayBuffer, header?: MessageHeader): ArrayBuffer;
    validateMessage(buffer: ArrayBuffer): boolean;

    // Position updates
    encodePositionUpdates(updates: AgentPositionUpdate[]): ArrayBuffer | null;
    decodePositionUpdates(payload: ArrayBuffer): AgentPositionUpdate[];

    // Agent state
    encodeAgentState(agents: AgentStateData[]): ArrayBuffer;
    decodeAgentState(payload: ArrayBuffer): AgentStateData[];

    // SSSP pathfinding data
    encodeSSSPData(nodes: SSSPData[]): ArrayBuffer;
    decodeSSSPData(payload: ArrayBuffer): SSSPData[];

    // Control bits
    encodeControlBits(flags: ControlFlags): ArrayBuffer;
    decodeControlBits(payload: ArrayBuffer): ControlFlags;

    // Voice chunks
    encodeVoiceChunk(chunk: VoiceChunk): ArrayBuffer;
    decodeVoiceChunk(payload: ArrayBuffer): VoiceChunk | null;

    // Backpressure flow control
    createBroadcastAck(sequenceId: number, nodesReceived: number): ArrayBuffer;
    decodeBroadcastAck(payload: ArrayBuffer): BroadcastAckData | null;

    // Agent action events
    encodeAgentAction(event: AgentActionEvent): ArrayBuffer;
    decodeAgentAction(payload: ArrayBuffer): AgentActionEvent | null;
    decodeAgentActions(payload: ArrayBuffer): AgentActionEvent[];

    // Configuration
    setUserInteracting(interacting: boolean): void;
    configureThrottling(positionMs: number, metadataMs: number): void;
    setVoiceEnabled(enabled: boolean): void;

    // Bandwidth estimation
    calculateBandwidth(agentCount: number, updateRateHz: number): {
        positionOnly: number;
        fullState: number;
        withVoice: number;
    };
}
```

---

## GraphEntityMapper

Utility class for converting between VisionFlow graph structures and Vircadia entity storage format.

### Types

```typescript
interface GraphNode {
    id: string;
    label: string;
    type?: string;
    color?: string;
    size?: number;
    x?: number;
    y?: number;
    z?: number;
    metadata?: Record<string, unknown>;
}

interface GraphEdge {
    id: string;
    source: string;
    target: string;
    label?: string;
    color?: string;
    weight?: number;
    metadata?: Record<string, unknown>;
}

interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
}

interface VircadiaEntity {
    general__entity_name: string;
    general__semantic_version: string;
    general__created_by?: string;
    general__updated_by?: string;
    group__sync: string;
    group__load_priority: number;
    meta__data?: Record<string, unknown>;
}
```

### GraphEntityMapper

```typescript
class GraphEntityMapper {
    constructor(options?: Partial<EntitySyncOptions>);

    // Map graph data to entities
    mapNodeToEntity(node: GraphNode): VircadiaEntity;
    mapEdgeToEntity(edge: GraphEdge, nodePositions: Map<string, Vec3>): VircadiaEntity;
    mapGraphToEntities(graphData: GraphData): VircadiaEntity[];

    // SQL generation (parameterized)
    generateEntityInsertSQL(entity: VircadiaEntity): { query: string; parameters: unknown[] };
    generateBatchInsertSQL(entities: VircadiaEntity[]): { queries: { query: string; parameters: unknown[] }[] };
    generatePositionUpdateSQL(entityName: string, position: Vec3): { query: string; parameters: unknown[] };

    // Entity position update
    updateEntityPosition(entity: VircadiaEntity, position: Vec3): VircadiaEntity;

    // Static conversion utilities
    static extractMetadata(entity: VircadiaEntity): VircadiaEntityMetadata | null;
    static entityToGraphNode(entity: VircadiaEntity): GraphNode | null;
    static entityToGraphEdge(entity: VircadiaEntity): GraphEdge | null;
    static entitiesToGraph(entities: VircadiaEntity[]): GraphData;
}
```
