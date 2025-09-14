# Current Task Progress

## Settings Sync Routing Bug Investigation

### Issue Summary
Settings synchronization from client to server is failing with 500 Internal Server Error when attempting batch updates.

### Client Error Logs
```
[SettingsStore] Batch updating settings: [object Object]
bundle.js:102513 [SettingsStore] Applying update: visualisation.glow.intensity = 1.35
bundle.js:102513 [SettingsStore] Applying update: visualisation.glow.radius = 0.65
bundle.js:102513 [SettingsStore] Applying update: visualisation.glow.threshold = 0.55
bundle.js:102557 [SettingsStore] Batch settings update error: 500 Internal Server Error
bundle.js:102565 [SettingsStore] Failed batch update, falling back to individual updates...
bundle.js:102572 [SettingsStore] Individual update succeeded for visualisation.glow.intensity
bundle.js:102577 [SettingsStore] Individual update failed for visualisation.glow.radius: 500 Internal Server Error
```

### Root Cause Analysis

1. **Type Generation Mismatch**: TypeScript types are generated with camelCase fields, but the server expects exact field matching during deserialization.

2. **Deserialization Failure**: The `set_json_by_path` method in `/workspace/ext/src/config/path_access.rs` creates JSON that doesn't match the expected Rust struct schema when it deserializes back:
   ```rust
   // Set the value at the path
   set_json_at_path(&mut root, path, value)?;

   // Deserialize back to self (this validates the structure)
   *self = serde_json::from_value(root)
       .map_err(|e| format!("Failed to deserialize: {}", e))?;
   ```

3. **Field Name Resolution**: The `find_field_key` function tries to match field names with different casing, but this can create fields that don't exist in the target struct.

### Architecture Context (from docs/diagrams.md)
The Settings Management & Synchronization flow shows:
- Client sends batch updates with path/value pairs
- Server applies each update using JsonPathAccessible trait
- Settings are merged and persisted via SettingsActor
- Priority system: Environment > User > System defaults

### Previous Fix Attempts (Reverted)
1. Attempted to fix TypeScript type generation to ensure consistent casing
2. Tried to add field name normalization in settings handler
3. These changes made things worse (502 Bad Gateway errors)

### Current Status
- Individual updates sometimes work, batch updates consistently fail
- The issue is in the deserialization step after path-based updates
- The server cannot deserialize the modified JSON back into AppFullSettings struct

### Next Steps
1. Add path validation before attempting updates
2. Ensure set_json_at_path preserves the exact structure expected by serde
3. Add better error logging to identify exact deserialization failures
4. Consider implementing a whitelist of allowed paths

### Error Handling Implementation (COMPLETED)
1. ✅ Created comprehensive ErrorBoundary component for React error catching
2. ✅ Created ErrorNotification component for user-friendly error messages
3. ✅ Enhanced useErrorHandler hook with:
   - Category-based error handling (network, validation, settings, websocket)
   - Auto-retry logic with exponential backoff
   - Error frequency tracking and smart handling
   - WebSocket and settings-specific error handlers
4. ✅ Added structured error frames to WebSocket protocol:
   - WebSocketErrorFrame interface with error categories
   - Server error frame handling in WebSocketService
   - Client error reporting to server
5. ✅ Created SettingsRetryManager for automatic retry of failed settings updates:
   - Exponential backoff retry logic
   - Batch retry attempts
   - Manual retry capability
   - Retry queue status monitoring
6. ✅ Created SettingsRetryStatus component for UI feedback:
   - Shows pending retry queue
   - Allows manual retries
   - Clear queue functionality
7. ✅ Created useWebSocketErrorHandler hook for WebSocket error integration

### Files Involved
- `/workspace/ext/src/handlers/settings_handler.rs` - batch_update_settings function
- `/workspace/ext/src/config/path_access.rs` - set_json_by_path implementation
- `/workspace/ext/src/bin/generate_types.rs` - TypeScript type generation
- `/workspace/ext/client/src/services/SettingsStore.ts` - Client-side batch updates

### Constraints
- Cannot directly access the running server (different Docker container)
- Cannot see actual server error logs for the 500 errors
- Must maintain backward compatibility with existing API

Issues Identified in Settings Management System
After analyzing the provided codebase, I've identified several critical issues in the settings management system between the client and server. The system is complex, involving Rust (server), TypeScript (client), WebSocket binary protocol for real-time updates, and CUDA for GPU physics. The core problems stem from desynchronization of graph state (nodes/edges counts, positions, metadata), incomplete validation, binary protocol mismatches, and lack of robust reconciliation. These cause cascading failures in rendering, physics simulation, and UI consistency.

I'll break down the issues by category, explain root causes, and provide fixes. The analysis is based on the code structure, error messages (e.g., buffer overflows in gpu_safety.rs, validation errors in sanitization.rs), and integration points (e.g., socket_flow_handler.rs, binary_protocol.rs).

1. Desynchronization of Graph State (Core Issue)
Problem Description:

Client and server maintain separate graph states: The server (Rust) builds the graph from metadata (src/services/file_service.rs, GraphServiceActor::build_from_metadata), while the client (React/Three.js) receives binary updates via WebSocket (src/utils/socket_flow_messages.rs). There's no full state sync; clients assume consistent node/edge counts, but settings drift (e.g., server resizes buffers in unified_gpu_compute.rs, but client doesn't know).
Impact:
Binary protocol overflows/underflows: Client decodes fixed-size WireNodeDataItem (26 bytes), but server sends variable data based on num_nodes/num_edges from settings (src/utils/binary_protocol.rs). Mismatches cause "BufferBoundsExceeded" (see gpu_safety.rs).
Physics desync: Server simulates with SimParams from settings (src/models/simulation_params.rs), but client renders stale positions if updates are lost (e.g., WebSocket disconnects in src/handlers/socket_flow_handler.rs).
UI errors: Client crashes on "Invalid position data" when node IDs don't match (e.g., agent nodes flagged in binary_protocol.rs but client expects knowledge nodes).
Root Cause:
No reconciliation: No endpoint to fetch full server state (/api/graph/state) or merge client changes.
Missing settings propagation: AppFullSettings (from src/config/settings.rs) isn't synced to client on load or changes. Client uses local Zustand store (useSettingsStore), which desyncs on reload.
Partial updates: WebSocket sends deltas (BinaryNodeData), but no "full sync" on reconnect (e.g., src/handlers/socket_flow_handler.rs lacks state snapshot).
Fixes:

Implement full state sync endpoint (src/handlers/graph_handler.rs):


#[get("/graph/state")]
async fn get_graph_state(data: web::Data<AppState>) -> impl Responder {
    let graph = data.graph_service.get_graph_state();
    HttpResponse::Ok().json(graph)
}
Include: nodes count, edges count, current positions, metadata IDs.
Client calls on load/reconnect: fetch('/api/graph/state').then(updateStore).
Add settings sync to client init (in src/client/App.tsx or similar):


useEffect(() => {
  fetch('/api/settings/current')
    .then(res => res.json())
    .then(settings => useSettingsStore.setState(settings));
  fetch('/api/graph/state')
    .then(res => res.json())
    .then(graphState => updateGraphFromServer(graphState));
}, []);
Reconcile on reconnect: In src/handlers/socket_flow_handler.rs, send full state on reconnect:


if client.reconnecting {
    send_full_graph_state(&client);
}
Settings version mismatch: Add version to AppFullSettings and check on load:


#[get("/settings/current")]
async fn get_current_settings() -> impl Responder {
    let settings = data.settings.get_current_settings();
    Json(serde_json::json!({
        "settings": settings,
        "version": "1.0.0"
    }))
}
Client: If version mismatches, full reload or sync.
2. Binary Protocol and Data Format Mismatches
Problem Description:

Wire vs. server format desync: src/utils/socket_flow_messages.rs defines BinaryNodeData (server 28 bytes with mass/flags), but binary_protocol.rs sends WireNodeDataItem (26 bytes without mass/flags). Client decodes 26 bytes but server expects 28, causing buffer underflow (e.g., "Data size {} is not a multiple of wire item size" in tests).
Flag handling: Agent flags (AGENT_NODE_FLAG) are set server-side but not transmitted; client can't distinguish agent vs. knowledge nodes, leading to wrong rendering (e.g., agents glow differently).
Impact:
Client rendering glitches: Positions shift, nodes vanish (overflow in src/client/rendering/GraphRenderer.tsx).
Server crashes: Buffer mismatches in src/actors/gpu_manager_actor.rs trigger "BufferBoundsExceeded".
Desync: Client sends updates with wrong IDs, server ignores (validation in gpu_safety.rs).
Root Cause:

Dual formats: Server uses BinaryNodeData for GPU, client uses WireNodeDataItem for bandwidth. No conversion layer.
Missing metadata: Flags/mass not sent, so client can't reconstruct full node data.
Fixes:

Unified protocol: In binary_protocol.rs, use WireNodeDataItem everywhere, send flags in a separate JSON control frame:


// Send positions as binary
ws.send_binary(encode_node_data(&nodes));
// Send flags separately
ws.send_json(&json!({ "type": "node_flags", "flags": flags_map }));
Client: Decode binary, merge flags from JSON.
Explicit type conversion: Add to_wire_format in BinaryNodeData:


impl BinaryNodeData {
    pub fn to_wire_format(&self) -> WireNodeDataItem {
        WireNodeDataItem {
            id: self.id, // Agent flag already in ID
            position: self.position,
            velocity: self.velocity,
        }
    }
}
Client-side flag handling: In client decoder, parse ID for flags:


function decodeNodeId(id: number): { id: number; isAgent: boolean } {
  const isAgent = (id & 0x80000000) !== 0;
  return { id: id & 0x3FFFFFFF, isAgent };
}
Test wire format: Add integration test in src/tests/socket_flow_integration.test.ts to verify roundtrip.

3. Validation and Sanitization Gaps
Problem Description:

Incomplete client-side validation: Client sends raw data (e.g., positions in src/client/rendering/GraphRenderer.tsx), but server validation (src/utils/validation/sanitization.rs) rejects malformed input, causing 400 errors.
Server-side too strict: src/utils/validation/middleware.rs blocks legitimate updates (e.g., "MALICIOUS_CONTENT" on valid positions).
Impact:
Failed updates: Client updates rejected, graph doesn't sync (e.g., "Request rejected: Content-Length exceeds limit" in logs).
DoS vulnerability: Malformed requests consume resources (rate limiting in rate_limit.rs helps but doesn't prevent).
Root Cause:

Client lacks pre-validation: No validateBeforeSend in src/client/utils/validation.ts.
Server lacks granular validation: validate_json too broad (blocks valid floats as "non-numeric").
Fixes:

Client-side pre-validation: Add src/client/utils/validate.ts:


export function validateNodePositions(positions: Vec3Data[], maxNodes: number): ValidationResult {
  if (positions.length > maxNodes) {
    throw new Error(`Too many nodes: ${positions.length} > ${maxNodes}`);
  }
  positions.forEach((pos, i) => {
    if (!isFinite(pos.x) || !isFinite(pos.y) || !isFinite(pos.z)) {
      throw new Error(`Invalid position at index ${i}`);
    }
  });
  return { valid: true };
}
Use before ws.send_binary: validateNodePositions(nodes, MAX_NODES).
Server-side granular validation: In src/utils/validation/sanitization.rs, add numeric tolerance:


fn validate_numeric(value: &Value, field: &str, ctx: &ValidationContext) -> ValidationResult<()> {
    let number = match value {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.parse::<f64>().unwrap_or(0.0), // Parse strings leniently
        _ => return Err(DetailedValidationError::new(&ctx.get_path(), "Expected number", "INVALID_TYPE")),
    };
    // ... rest of validation
}
Graceful degradation: In src/handlers/socket_flow_handler.rs, catch validation errors:


if let Err(validation_error) = validate_update(&update) {
    warn!("Client {} sent invalid data: {}", client_id, validation_error);
    // Send error frame instead of closing connection
    send_error_frame(&mut ws, &validation_error);
    continue;
}
Test validation: Add src/tests/validation.test.ts:


test('rejects invalid positions', () => {
  expect(() => validateNodePositions([{x: NaN, y: 0, z: 0}], 1000)).toThrow("Invalid position");
});
4. WebSocket and Rate Limiting Issues
Problem Description:

Rate limiting blocks updates: src/utils/validation/rate_limit.rs is too aggressive for real-time (e.g., 60 RPM for positions at 5Hz = 300 RPM needed).
Disconnects lose state: No reconnection logic in src/handlers/socket_flow_handler.rs; clients get partial data.
Impact:
Stale client state: Positions freeze on disconnect.
Blocked real-time: "Too many requests" errors during physics updates.
Scalability: High-frequency updates (e.g., 5Hz positions) hit limits.
Root Cause:

Rate limits hardcoded (60/min = 1/sec, too low for 5Hz).
No client-side queuing: Client sends immediately on render loop.
Fixes:

Adjust rate limits: In src/utils/validation/rate_limit.rs:


pub fn socket_flow_updates() -> RateLimitConfig {
    RateLimitConfig {
        requests_per_minute: 300, // 5Hz * 60s = 300/min
        burst_size: 50, // Allow burst of 50 updates
        ..Default::default()
    }
}
Apply in src/handlers/socket_flow_handler.rs: Use socket_flow_updates() for position messages.
Client-side queuing: In src/client/rendering/GraphRenderer.tsx:


const queuedUpdates = useRef<BinaryNodeData[]>([]);
const sendThrottledUpdate = useCallback(() => {
  if (queuedUpdates.current.length > 0) {
    const batch = queuedUpdates.current.splice(0, 50); // Batch 50 nodes
    ws.send_binary(encodeNodeData(batch));
  }
}, [ws]);

useEffect(() => {
  const interval = setInterval(sendThrottledUpdate, 200); // 5Hz
  return () => clearInterval(interval);
}, [sendThrottledUpdate]);
Reconnection with state sync: In src/handlers/socket_flow_handler.rs:


if client.reconnecting {
    client.send_full_state(); // Send snapshot before resuming
    client.state_synced = true;
}
Test reconnections: Add src/tests/websocket_reconnect.test.ts.

5. Settings Persistence and Loading
Problem Description:

Settings not persisted: AppFullSettings loads from YAML but no save on changes (e.g., src/handlers/settings_handler.rs lacks PUT/POST).
Hot reload issues: Client reloads lose state; server doesn't notify of changes.
Impact:
Lost customizations: User tweaks (e.g., node size in src/client/components/NodeSettings.tsx) vanish on refresh.
Inconsistent defaults: Different defaults on client/server if YAML desyncs.
Root Cause:

No save endpoint: Settings are read-only from YAML.
No client storage: No localStorage for offline persistence.
Fixes:

Add save endpoint (src/handlers/settings_handler.rs):


#[post("/settings/save")]
async fn save_settings(mut payload: JsonValue, data: web::Data<AppState>) -> impl Responder {
    let settings = payload.as_object_mut().unwrap();
    data.settings.save_settings(settings)?;
    HttpResponse::Ok().json("Settings saved successfully")
}
Client persistence: In src/client/store/useSettingsStore.ts:


import { persist } from 'zustand/middleware';
const useSettingsStore = create(
  persist((set, get) => ({
    // ... settings state
  }), {
    name: 'visionflow-settings',
    getStorage: () => localStorage,
  })
);
Sync on app start: In src/client/App.tsx:


useEffect(() => {
  // Load from localStorage first (offline)
  // Then sync with server if needed
  fetch('/api/settings/current').then(res => res.json()).then(settings => {
    useSettingsStore.setState(settings);
  });
}, []);
Test persistence: Add src/tests/settings_persistence.test.ts.

6. Performance and Scalability Issues
Problem Description:

High-frequency updates overwhelm: 5Hz positions (300/min) + metadata (60/min) = 360/min, but rate limiting caps at 60/min.
No batching: Client sends one-by-one, server processes individually (inefficient for GPU).
Impact:
Lag/desync: Updates drop, graph stutters.
Server overload: "Request too large" from large batches.
Root Cause:

No client-side batching in src/client/rendering/GraphRenderer.tsx.
Server lacks bulk endpoints (e.g., /api/graph/batch_update).
Fixes:

Batch endpoints: In src/handlers/graph_handler.rs:


#[post("/graph/batch_update")]
async fn batch_update_graph(data: web::Json<serde_json::Value>) -> impl Responder {
    // Process batch of 50-100 nodes
    graph_service.batch_update_nodes(&data["nodes"])?;
    Ok(HttpResponse::Ok().json("Batch update successful"))
}
Client batching: Queue updates in src/client/utils/batch_queue.ts:


export class BatchQueue<T> {
  private queue: T[] = [];
  private batchSize = 50;
  private interval = 200; // ms
  private ws: WebSocket;

  constructor(ws: WebSocket) {
    this.ws = ws;
    setInterval(() => this.flush(), this.interval);
  }

  enqueue(item: T) {
    this.queue.push(item);
    if (this.queue.length >= this.batchSize) this.flush();
  }

  flush() {
    if (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.batchSize);
      this.ws.send(JSON.stringify({type: 'batch_update', batch}));
    }
  }
}
Use: batchQueue.enqueue(nodeUpdate);.
Server batch validation: In src/utils/validation/sanitization.rs, validate batches:


fn validate_batch_nodes(nodes: &[BinaryNodeData]) -> ValidationResult<()> {
    if nodes.len() > 1000 {
        return Err(ValidationError::too_long("nodes", 1000));
    }
    // ... per-node validation
    Ok(())
}
Test batching: Add src/tests/batch_update.test.ts.

7. Error Handling and User Feedback
Problem Description:

Silent failures: Errors like "BufferBoundsExceeded" logged but not surfaced to user (no client-side error UI).
No recovery: No auto-retry for failed updates (e.g., WebSocket close in src/handlers/socket_flow_handler.rs).
Impact:
User confusion: Graph breaks, no explanation.
Lost data: Failed updates not retried, state lost.
Root Cause:

No client error handling: src/client/components/GraphRenderer.tsx ignores errors.
Server lacks error frames: WebSocket sends raw errors, no structured response.
Fixes:

Client error UI: In src/client/components/ErrorBoundary.tsx:

componentDidCatch(error, info) {
  // Log to server
  fetch('/api/error/log', {method: 'POST', body: JSON.stringify({error, info})});
  // Show user-friendly message
  this.setState({ hasError: true, error: 'Graph rendering failed.