# WebSocket Communication

*[Client](../index.md)*

This document describes the WebSocket communication system, which is the primary channel for real-time data transfer between the client and server, most notably for graph physics updates.

## Architecture & Patterns

The system is built around three core design patterns: the Singleton, the Readiness Protocol, and the Adapter Pattern.

```mermaid
graph TB
    subgraph ClientApplication ["Client Application"]
        Appinitialiser["Appinitialiser.tsx"]
        GDM["GraphDataManager.ts"]
    end

    subgraph SingletonService ["Singleton Service"]
        WSS["WebSocketService.ts"]
    end

    Appinitialiser -- "1. Gets Instance" --> WSS
    Appinitialiser -- "2. Creates Adapter" --> WSS
    Appinitialiser -- "3. Passes Adapter to" --> GDM
    GDM -- "4. Uses Adapter" --> WSS

    style WSS fill:#cde4ff,stroke:#333,stroke-width:2px
```

### 1. Singleton Pattern (`WebSocketService.ts`)

The `WebSocketService` is implemented as a singleton to ensure there is only one instance managing the WebSocket connection for the entire application.

-   **Source**: [`client/src/services/WebSocketService.ts`](../../client/src/services/WebSocketService.ts)
-   **Access**: The single instance is accessed statically via `WebSocketService.getInstance()`.
-   **Responsibilities**:
    -   Establishing and maintaining the WebSocket connection.
    -   Automatic reconnection logic with exponential backoff.
    -   Handling incoming JSON and binary messages.
    -   Managing the readiness state of the connection.

### 2. Readiness Protocol

To prevent the application from attempting to send data over a connection that isn't fully operational, a strict readiness protocol is enforced. A connection is only considered "ready" when two conditions are met:

1.  The browser's WebSocket `onopen` event has fired.
2.  The server has sent a `{"type": "connection_established"}` message back to the client.

This two-step handshake ensures that both the client and server are fully prepared for communication.

-   **Implementation**:
    -   `WebSocketService` maintains two internal flags: `isConnected` (for the `onopen` event) and `isServerReady` (for the server message).
    -   The public method `isReady(): boolean` returns `true` only when both flags are true.

```mermaid
stateDiagram-v2
    [*] --> Disconnected
    Disconnected --> Connecting: connect()
    Connecting --> Connected: onopen event fires
    Connected --> Ready: connection_established message received
    Ready --> Disconnected: onclose event
    Connected --> Disconnected: onclose event
```

### 3. Adapter Pattern

To decouple the `GraphDataManager` from the concrete implementation of the `WebSocketService`, an adapter is used. This is crucial for modularity and testing.

-   **Creation**: In [`Appinitialiser.tsx`](../../client/src/app/Appinitialiser.tsx), a `wsAdapter` object is created. This object exposes a clean, minimal interface that is specifically tailored to the needs of the `GraphDataManager`.
-   **Interface**: The adapter provides simple methods like `send(data)` and `isReady()`.
-   **Dependency Injection**: The `Appinitialiser` passes this adapter to the `GraphDataManager` instance using `graphDataManager.setWebSocketService(wsAdapter)`.

This pattern means `GraphDataManager` doesn't need to know about the `WebSocketService` singleton directly. It only interacts with the simple adapter interface it was given.

**Conceptual Implementation in `Appinitialiser.tsx`**:

```typescript
// client/src/app/Appinitialiser.tsx

const websocketService = WebSocketService.getInstance();

// ... setup websocketService event handlers ...

// Create the adapter with the methods GraphDataManager needs
const wsAdapter = {
    send: (data: ArrayBuffer) => {
        websocketService.sendRawBinaryData(data);
    },
    isReady: () => websocketService.isReady()
};

// Pass the adapter to the data manager
graphDataManager.setWebSocketService(wsAdapter);

// Now, GraphDataManager can use the service without being tightly coupled
// e.g., inside GraphDataManager:
// if (this.wsServiceAdapter?.isReady()) { ... }
```

## Data Flow

The primary use of the WebSocket is to receive real-time binary position updates for the graph nodes from the server's physics simulation.

1.  **Connection**: The `Appinitialiser` initiates the connection.
2.  **Readiness Check**: The `GraphDataManager` periodically checks `wsAdapter.isReady()`.
3.  **Subscription**: Once ready, the client sends a `subscribe_position_updates` message to the server.
4.  **Binary Data Stream**: The server begins streaming `ArrayBuffer` messages containing node position data.
5.  **Processing**: The `onBinaryMessage` handler in `WebSocketService` passes the data to `graphDataManager.updateNodePositions(data)`.
6.  **Worker Thread**: The `GraphDataManager` forwards the binary data to the `graph.worker.ts` for efficient off-main-thread processing and position smoothing.
7.  **Rendering**: The main thread receives smoothed positions back from the worker and updates the graph visualisation.

## Related Topics

- [Binary Protocol Specification](../binary-protocol.md)
- [Client Architecture](../client/architecture.md)
- [Client Core Utilities and Hooks](../client/core.md)
- [Client Rendering System](../client/rendering.md)
- [Client TypeScript Types](../client/types.md)
- [Client side DCO](../archive/legacy/old_markdown/Client side DCO.md)
- [Client-Side visualisation Concepts](../client/visualization.md)
- [Command Palette](../client/command-palette.md)
- [GPU-Accelerated Analytics](../client/features/gpu-analytics.md)
- [Graph System](../client/graph-system.md)
- [Help System](../client/help-system.md)
- [MCP WebSocket Relay Architecture](../architecture/mcp-websocket-relay.md)
- [Onboarding System](../client/onboarding.md)
- [Parallel Graphs Feature](../client/parallel-graphs.md)
- [RGB and Client Side Validation](../archive/legacy/old_markdown/RGB and Client Side Validation.md)
- [Settings Panel](../client/settings-panel.md)
- [State Management](../client/state-management.md)
- [UI Component Library](../client/ui-components.md)
- [User Controls Summary - Settings Panel](../client/user-controls-summary.md)
- [VisionFlow Client Documentation](../client/index.md)
- [VisionFlow WebSocket API Documentation](../api/websocket/index.md)
- [WebSocket API Reference](../api/websocket.md)
- [WebSocket Protocols](../api/websocket-protocols.md)
- [WebXR Integration](../client/xr-integration.md)
