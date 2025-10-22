# Server-Side Settings Audit

**Generated:** 2025-10-22
**Total Parameters Discovered:** 487+
**Configuration Files:**
- `src/config/mod.rs` - Main application settings (25,000+ lines)
- `src/config/dev_config.rs` - Developer-only internal settings
- `data/settings.yaml` - Current YAML configuration (legacy, moving to DB)
- `src/actors/optimized_settings_actor.rs` - Settings caching and performance
- `src/actors/protected_settings_actor.rs` - User authentication and API keys

---

## 1. SYSTEM CATEGORY - Network Configuration

### bind_address
- **Current Value**: `"0.0.0.0"`
- **Type**: String
- **Validation**: IP address or hostname pattern
- **Location**: `src/config/mod.rs:1208`, `data/settings.yaml:321`
- **Priority**: Critical
- **Category**: System/Network
- **Description**: Server bind address for HTTP/WebSocket connections

### port
- **Current Value**: `4000`
- **Type**: u16
- **Validation**: 1-65535, validate_port()
- **Location**: `src/config/mod.rs:1218`, `data/settings.yaml:328`
- **Priority**: Critical
- **Category**: System/Network
- **Description**: Main HTTP server port

### domain
- **Current Value**: `"visionflow.info"`
- **Type**: String
- **Validation**: DOMAIN_REGEX pattern
- **Location**: `src/config/mod.rs:1206`, `data/settings.yaml:322`
- **Priority**: High
- **Category**: System/Network
- **Description**: Public domain name for the application

### enable_http2
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1208`, `data/settings.yaml:323`
- **Priority**: Medium
- **Category**: System/Network
- **Description**: Enable HTTP/2 protocol support

### enable_tls
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1212`, `data/settings.yaml:325`
- **Priority**: Critical
- **Category**: System/Security
- **Description**: Enable TLS/SSL encryption

### min_tls_version
- **Current Value**: `"1.2"`
- **Type**: String
- **Validation**: Must be "1.2" or "1.3"
- **Location**: `src/config/mod.rs:1216`, `data/settings.yaml:327`
- **Priority**: High
- **Category**: System/Security
- **Description**: Minimum TLS protocol version

### max_request_size
- **Current Value**: `10485760` (10MB)
- **Type**: usize
- **Location**: `src/config/mod.rs:1214`, `data/settings.yaml:326`
- **Priority**: High
- **Category**: System/Network
- **Description**: Maximum HTTP request body size in bytes

### api_client_timeout
- **Current Value**: `30`
- **Type**: u32 (seconds)
- **Location**: `src/config/mod.rs:1226`, `data/settings.yaml:332`
- **Priority**: Medium
- **Category**: System/Network
- **Description**: Timeout for API client requests

### max_concurrent_requests
- **Current Value**: `1`
- **Type**: u32
- **Location**: `src/config/mod.rs:1230`, `data/settings.yaml:334`
- **Priority**: High
- **Category**: System/Performance
- **Description**: Maximum concurrent API requests

### max_retries
- **Current Value**: `3`
- **Type**: u32
- **Location**: `src/config/mod.rs:1232`, `data/settings.yaml:335`
- **Priority**: Medium
- **Category**: System/Network
- **Description**: Maximum retry attempts for failed requests

### retry_delay
- **Current Value**: `5`
- **Type**: u32 (seconds)
- **Location**: `data/settings.yaml:337`
- **Priority**: Low
- **Category**: System/Network
- **Description**: Delay between retry attempts

### tunnel_id
- **Current Value**: `"dummy"`
- **Type**: String
- **Location**: `src/config/mod.rs:1224`, `data/settings.yaml:331`
- **Priority**: Low
- **Category**: System/Network
- **Description**: Tunnel identifier for proxying

---

## 2. SYSTEM CATEGORY - Rate Limiting

### enable_rate_limiting
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1210`, `data/settings.yaml:324`
- **Priority**: High
- **Category**: System/Security
- **Description**: Enable request rate limiting

### rate_limit_requests
- **Current Value**: `10000`
- **Type**: u32
- **Location**: `src/config/mod.rs:1220`, `data/settings.yaml:329`
- **Priority**: High
- **Category**: System/Security
- **Description**: Maximum requests per window

### rate_limit_window
- **Current Value**: `600` (seconds)
- **Type**: u32
- **Location**: `src/config/mod.rs:1222`, `data/settings.yaml:330`
- **Priority**: High
- **Category**: System/Security
- **Description**: Time window for rate limiting

### rate_limit_burst_size
- **Current Value**: `10`
- **Type**: u32
- **Location**: `src/config/dev_config.rs:124`
- **Priority**: Medium
- **Category**: System/Performance (Internal)
- **Description**: Burst size for rate limiter token bucket

### rate_limit_refill_rate
- **Current Value**: `1.0`
- **Type**: f32
- **Location**: `src/config/dev_config.rs:125`
- **Priority**: Medium
- **Category**: System/Performance (Internal)
- **Description**: Token refill rate per second

---

## 3. SYSTEM CATEGORY - Metrics and Monitoring

### enable_metrics
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1228`, `data/settings.yaml:333`
- **Priority**: Medium
- **Category**: System/Monitoring
- **Description**: Enable Prometheus metrics endpoint

### metrics_port
- **Current Value**: `9090`
- **Type**: u16
- **Location**: `src/config/mod.rs:1234`, `data/settings.yaml:336`
- **Priority**: Medium
- **Category**: System/Monitoring
- **Description**: Metrics endpoint port

---

## 4. WEBSOCKET CATEGORY

### binary_chunk_size
- **Current Value**: `2048`
- **Type**: usize
- **Location**: `src/config/mod.rs:1267`, `data/settings.yaml:339`
- **Priority**: High
- **Category**: WebSocket/Performance
- **Description**: Binary message chunk size in bytes
- **Constant Override**: `BINARY_CHUNK_SIZE = 65536` in `socket_flow_constants.rs:12`

### binary_update_rate
- **Current Value**: `30`
- **Type**: u32 (Hz)
- **Location**: `src/config/mod.rs:1269`, `data/settings.yaml:340`
- **Priority**: High
- **Category**: WebSocket/Performance
- **Description**: Binary update frequency

### min_update_rate
- **Current Value**: `5`
- **Type**: u32 (Hz)
- **Location**: `src/config/mod.rs:1271`, `data/settings.yaml:341`
- **Priority**: Medium
- **Category**: WebSocket/Performance
- **Description**: Minimum WebSocket update rate

### max_update_rate
- **Current Value**: `60`
- **Type**: u32 (Hz)
- **Location**: `src/config/mod.rs:1273`, `data/settings.yaml:342`
- **Priority**: Medium
- **Category**: WebSocket/Performance
- **Description**: Maximum WebSocket update rate

### motion_threshold
- **Current Value**: `0.05`
- **Type**: f32
- **Location**: `src/config/mod.rs:1275`, `data/settings.yaml:343`
- **Priority**: Low
- **Category**: WebSocket/Physics
- **Description**: Threshold for motion detection

### motion_damping
- **Current Value**: `0.9`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:1277`, `data/settings.yaml:344`
- **Priority**: Low
- **Category**: WebSocket/Physics
- **Description**: Motion damping factor

### binary_message_version
- **Current Value**: `1`
- **Type**: u8
- **Location**: `src/config/mod.rs:1279`, `data/settings.yaml:345`
- **Priority**: Critical
- **Category**: WebSocket/Protocol
- **Description**: Binary protocol version

### compression_enabled
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1281`, `data/settings.yaml:346`
- **Priority**: Medium
- **Category**: WebSocket/Performance
- **Description**: Enable WebSocket compression

### compression_threshold
- **Current Value**: `512`
- **Type**: usize (bytes)
- **Location**: `src/config/mod.rs:1283`, `data/settings.yaml:347`
- **Priority**: Low
- **Category**: WebSocket/Performance
- **Description**: Minimum message size for compression

### heartbeat_interval
- **Current Value**: `10000`
- **Type**: u64 (milliseconds)
- **Location**: `src/config/mod.rs:1285`, `data/settings.yaml:348`
- **Priority**: High
- **Category**: WebSocket/Connection
- **Description**: Heartbeat ping interval
- **Constant Override**: `HEARTBEAT_INTERVAL = 30` seconds in `socket_flow_constants.rs:8`

### heartbeat_timeout
- **Current Value**: `600000` (10 minutes)
- **Type**: u64 (milliseconds)
- **Location**: `src/config/mod.rs:1287`, `data/settings.yaml:349`
- **Priority**: High
- **Category**: WebSocket/Connection
- **Description**: Heartbeat timeout before disconnect
- **Constant Override**: `CLIENT_TIMEOUT = 60` seconds in `socket_flow_constants.rs:9`

### max_connections
- **Current Value**: `100`
- **Type**: usize
- **Location**: `src/config/mod.rs:1289`, `data/settings.yaml:350`
- **Priority**: Critical
- **Category**: WebSocket/Limits
- **Description**: Maximum concurrent WebSocket connections

### max_message_size
- **Current Value**: `10485760` (10MB)
- **Type**: usize
- **Location**: `src/config/mod.rs:1291`, `data/settings.yaml:351`
- **Priority**: Critical
- **Category**: WebSocket/Limits
- **Description**: Maximum WebSocket message size
- **Constant Override**: `MAX_MESSAGE_SIZE = 104857600` (100MB) in `socket_flow_constants.rs:11`

### reconnect_attempts
- **Current Value**: `5`
- **Type**: u32
- **Location**: `src/config/mod.rs:1293`, `data/settings.yaml:352`
- **Priority**: Medium
- **Category**: WebSocket/Connection
- **Description**: Maximum reconnection attempts

### reconnect_delay
- **Current Value**: `1000`
- **Type**: u64 (milliseconds)
- **Location**: `src/config/mod.rs:1295`, `data/settings.yaml:353`
- **Priority**: Medium
- **Category**: WebSocket/Connection
- **Description**: Delay between reconnection attempts

### update_rate
- **Current Value**: `60`
- **Type**: u32 (Hz)
- **Location**: `src/config/mod.rs:1297`, `data/settings.yaml:354`
- **Priority**: High
- **Category**: WebSocket/Performance
- **Description**: Default WebSocket update rate

---

## 5. SECURITY CATEGORY

### allowed_origins
- **Current Value**: `["https://www.visionflow.info", "https://visionflow.info"]`
- **Type**: Vec<String>
- **Validation**: URL_REGEX pattern per entry
- **Location**: `src/config/mod.rs:1327`, `data/settings.yaml:356-358`
- **Priority**: Critical
- **Category**: Security/CORS
- **Description**: Allowed CORS origins

### audit_log_path
- **Current Value**: `"/app/logs/audit.log"`
- **Type**: String
- **Validation**: FILE_PATH_REGEX pattern
- **Location**: `src/config/mod.rs:1329`, `data/settings.yaml:359`
- **Priority**: Medium
- **Category**: Security/Audit
- **Description**: Path to audit log file

### cookie_httponly
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:1331`, `data/settings.yaml:360`
- **Priority**: High
- **Category**: Security/Cookies
- **Description**: Set HttpOnly flag on cookies

### cookie_samesite
- **Current Value**: `"Strict"`
- **Type**: String
- **Validation**: Must be "Strict", "Lax", or "None"
- **Location**: `src/config/mod.rs:1333`, `data/settings.yaml:361`
- **Priority**: High
- **Category**: Security/Cookies
- **Description**: SameSite cookie policy

### cookie_secure
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:1335`, `data/settings.yaml:362`
- **Priority**: High
- **Category**: Security/Cookies
- **Description**: Require secure (HTTPS) cookies

### csrf_token_timeout
- **Current Value**: `3600` (seconds)
- **Type**: u64
- **Location**: `src/config/mod.rs:1337`, `data/settings.yaml:363`
- **Priority**: High
- **Category**: Security/CSRF
- **Description**: CSRF token expiration time

### enable_audit_logging
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1339`, `data/settings.yaml:364`
- **Priority**: Medium
- **Category**: Security/Audit
- **Description**: Enable security audit logging

### enable_request_validation
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1341`, `data/settings.yaml:365`
- **Priority**: High
- **Category**: Security/Validation
- **Description**: Enable request payload validation

### session_timeout
- **Current Value**: `3600` (seconds)
- **Type**: u64
- **Location**: `src/config/mod.rs:1343`, `data/settings.yaml:366`
- **Priority**: High
- **Category**: Security/Sessions
- **Description**: User session timeout

---

## 6. AUTHENTICATION CATEGORY

### enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:1500`, `data/settings.yaml:419`
- **Priority**: Critical
- **Category**: Authentication
- **Description**: Enable authentication system

### provider
- **Current Value**: `"nostr"`
- **Type**: String
- **Validation**: Must be "nostr", "oauth", or "jwt"
- **Location**: `src/config/mod.rs:1502`, `data/settings.yaml:420`
- **Priority**: Critical
- **Category**: Authentication
- **Description**: Authentication provider type

### required
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:1504`, `data/settings.yaml:421`
- **Priority**: Critical
- **Category**: Authentication
- **Description**: Require authentication for all endpoints

---

## 7. PHYSICS CATEGORY - Core Settings

### enabled
- **Current Value**: `true` (logseq), `true` (visionflow)
- **Type**: bool
- **Location**: `src/config/mod.rs:710`, `data/settings.yaml:147`, `267`
- **Priority**: Critical
- **Category**: Physics/Simulation
- **Description**: Enable physics simulation

### damping
- **Current Value**: `0.6` (logseq), `0.1` (visionflow)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:706`, `data/settings.yaml:145`, `265`
- **Priority**: High
- **Category**: Physics/Forces
- **Description**: Velocity damping coefficient

### spring_k (springK in camelCase)
- **Current Value**: `4.6001` (logseq), `10.0` (visionflow)
- **Type**: f32
- **Validation**: 0.0-10.0
- **Location**: `src/config/mod.rs:718`, `data/settings.yaml:152`, `272`
- **Priority**: High
- **Category**: Physics/Forces
- **Description**: Spring force constant

### repel_k (repelK in camelCase)
- **Current Value**: `13.28022` (logseq), `100.0` (visionflow)
- **Type**: f32
- **Validation**: 0.0-100.0
- **Location**: `src/config/mod.rs:716`, `data/settings.yaml:151`, `271`
- **Priority**: High
- **Category**: Physics/Forces
- **Description**: Repulsion force constant

### max_velocity (maxVelocity in camelCase)
- **Current Value**: `100.0` (logseq), `50.0` (visionflow)
- **Type**: f32
- **Validation**: 0.1-50.0
- **Location**: `src/config/mod.rs:714`, `data/settings.yaml:149`, `269`
- **Priority**: High
- **Category**: Physics/Limits
- **Description**: Maximum node velocity

### max_force (maxForce in camelCase)
- **Current Value**: `1000.0` (logseq), `500.0` (visionflow)
- **Type**: f32
- **Location**: `src/config/mod.rs:716`, `data/settings.yaml:150`, `270`
- **Priority**: High
- **Category**: Physics/Limits
- **Description**: Maximum force magnitude

### gravity
- **Current Value**: `0.0001` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:738`, `data/settings.yaml:158`, `278`
- **Priority**: Medium
- **Category**: Physics/Forces
- **Description**: Gravitational constant

### temperature
- **Current Value**: `2.0` (logseq), `1.0` (visionflow)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:736`, `data/settings.yaml:157`, `277`
- **Priority**: Medium
- **Category**: Physics/Simulation
- **Description**: System temperature for force-directed layout

### bounds_size (boundsSize in camelCase)
- **Current Value**: `1000.0` (both graphs)
- **Type**: f32
- **Validation**: 100.0-2000.0
- **Location**: `src/config/mod.rs:704`, `data/settings.yaml:143`, `263`
- **Priority**: Medium
- **Category**: Physics/Space
- **Description**: Simulation bounds size

### enable_bounds (enableBounds in camelCase)
- **Current Value**: `false` (both graphs)
- **Type**: bool
- **Location**: `src/config/mod.rs:708`, `data/settings.yaml:146`, `266`
- **Priority**: Medium
- **Category**: Physics/Space
- **Description**: Enable boundary constraints

### iterations
- **Current Value**: `50` (both graphs)
- **Type**: u32
- **Validation**: 1-1000
- **Location**: `src/config/mod.rs:712`, `data/settings.yaml:148`, `268`
- **Priority**: High
- **Category**: Physics/Performance
- **Description**: Physics simulation iterations per frame

### dt
- **Current Value**: `0.02269863` (logseq), `0.016` (visionflow)
- **Type**: f32
- **Location**: `src/config/mod.rs:734`, `data/settings.yaml:156`, `276`
- **Priority**: High
- **Category**: Physics/Simulation
- **Description**: Time delta per physics step

### mass_scale (massScale in camelCase)
- **Current Value**: `0.39917582` (logseq), `1.0` (visionflow)
- **Type**: f32
- **Location**: `src/config/mod.rs:720`, `data/settings.yaml:153`, `273`
- **Priority**: Medium
- **Category**: Physics/Forces
- **Description**: Node mass scaling factor

### separation_radius (separationRadius in camelCase)
- **Current Value**: `2.1155233` (logseq), `2.0` (visionflow)
- **Type**: f32
- **Location**: `src/config/mod.rs:702`, `data/settings.yaml:144`, `264`
- **Priority**: Medium
- **Category**: Physics/Space
- **Description**: Minimum separation between nodes

### boundary_damping (boundaryDamping in camelCase)
- **Current Value**: `0.95` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:722`, `data/settings.yaml:154`, `274`
- **Priority**: Low
- **Category**: Physics/Boundaries
- **Description**: Velocity damping at boundaries

### update_threshold (updateThreshold in camelCase)
- **Current Value**: `0.01` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:724`, `data/settings.yaml:155`, `275`
- **Priority**: Low
- **Category**: Physics/Performance
- **Description**: Minimum change threshold for updates

### stress_weight (stressWeight in camelCase)
- **Current Value**: `0.0001` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:740`, `data/settings.yaml:159`, `279`
- **Priority**: Low
- **Category**: Physics/Layout
- **Description**: Stress majorization weight

### stress_alpha (stressAlpha in camelCase)
- **Current Value**: `0.0001` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:742`, `data/settings.yaml:160`, `280`
- **Priority**: Low
- **Category**: Physics/Layout
- **Description**: Stress majorization alpha parameter

### boundary_limit (boundaryLimit in camelCase)
- **Current Value**: `1000.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:744`, `data/settings.yaml:161`, `281`
- **Priority**: Medium
- **Category**: Physics/Boundaries
- **Description**: Hard boundary limit

### alignment_strength (alignmentStrength in camelCase)
- **Current Value**: `0.1` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:746`, `data/settings.yaml:162`, `282`
- **Priority**: Low
- **Category**: Physics/Forces
- **Description**: Node alignment force strength

### cluster_strength (clusterStrength in camelCase)
- **Current Value**: `0.1` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:748`, `data/settings.yaml:163`, `283`
- **Priority**: Low
- **Category**: Physics/Clustering
- **Description**: Cluster cohesion force strength

---

## 8. PHYSICS CATEGORY - GPU Compute Mode

### compute_mode (computeMode in camelCase)
- **Current Value**: `1` (both graphs)
- **Type**: u8
- **Validation**: 0=CPU, 1=GPU, 2=Hybrid
- **Location**: `src/config/mod.rs:750`, `data/settings.yaml:164`, `284`
- **Priority**: Critical
- **Category**: Physics/GPU
- **Description**: Physics computation mode selection

### rest_length (restLength in camelCase)
- **Current Value**: `50.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:752`, `data/settings.yaml:165`, `285`
- **Priority**: Medium
- **Category**: Physics/GPU
- **Description**: Natural spring rest length

### repulsion_cutoff (repulsionCutoff in camelCase)
- **Current Value**: `50.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:754`, `data/settings.yaml:166`, `286`
- **Priority**: High
- **Category**: Physics/GPU
- **Description**: Maximum distance for repulsion calculations

### repulsion_softening_epsilon (repulsionSofteningEpsilon in camelCase)
- **Current Value**: `0.0001` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:756`, `data/settings.yaml:167`, `287`
- **Priority**: Medium
- **Category**: Physics/GPU
- **Description**: Softening epsilon to prevent division by zero

### center_gravity_k (centerGravityK in camelCase)
- **Current Value**: `0.005` (logseq), `0.0` (visionflow)
- **Type**: f32
- **Location**: `src/config/mod.rs:758`, `data/settings.yaml:168`, `288`
- **Priority**: Low
- **Category**: Physics/GPU
- **Description**: Center gravity constant

### grid_cell_size (gridCellSize in camelCase)
- **Current Value**: `28.543957` (logseq), `50.0` (visionflow)
- **Type**: f32
- **Location**: `src/config/mod.rs:760`, `data/settings.yaml:169`, `289`
- **Priority**: High
- **Category**: Physics/GPU
- **Description**: Spatial grid cell size for neighbor searches

### warmup_iterations (warmupIterations in camelCase)
- **Current Value**: `100` (both graphs)
- **Type**: u32
- **Location**: `src/config/mod.rs:762`, `data/settings.yaml:170`, `290`
- **Priority**: Medium
- **Category**: Physics/GPU
- **Description**: Number of warmup simulation steps

### cooling_rate (coolingRate in camelCase)
- **Current Value**: `0.001` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:764`, `data/settings.yaml:171`, `291`
- **Priority**: Low
- **Category**: Physics/GPU
- **Description**: Temperature cooling rate during warmup

---

## 9. PHYSICS CATEGORY - Boundary Behavior

### boundary_extreme_multiplier (boundaryExtremeMultiplier in camelCase)
- **Current Value**: `2.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:766`, `data/settings.yaml:172`, `292`
- **Priority**: Medium
- **Category**: Physics/Boundaries
- **Description**: Position multiplier for extreme boundary detection

### boundary_extreme_force_multiplier (boundaryExtremeForceMultiplier in camelCase)
- **Current Value**: `5.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:768`, `data/settings.yaml:173`, `293`
- **Priority**: Medium
- **Category**: Physics/Boundaries
- **Description**: Force multiplier at extreme boundaries

### boundary_velocity_damping (boundaryVelocityDamping in camelCase)
- **Current Value**: `0.5` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:770`, `data/settings.yaml:174`, `294`
- **Priority**: Medium
- **Category**: Physics/Boundaries
- **Description**: Velocity reduction on boundary collision

### min_distance (minDistance in camelCase)
- **Current Value**: `1.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:772`, `data/settings.yaml:175`, `295`
- **Priority**: High
- **Category**: Physics/Collision
- **Description**: Minimum distance between nodes

### max_repulsion_dist (maxRepulsionDist in camelCase)
- **Current Value**: `1000.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:774`, `data/settings.yaml:176`, `296`
- **Priority**: High
- **Category**: Physics/Forces
- **Description**: Maximum distance for repulsion forces

### boundary_margin (boundaryMargin in camelCase)
- **Current Value**: `10.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:776`, `data/settings.yaml:177`, `297`
- **Priority**: Low
- **Category**: Physics/Boundaries
- **Description**: Margin for boundary detection

### boundary_force_strength (boundaryForceStrength in camelCase)
- **Current Value**: `1.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:778`, `data/settings.yaml:178`, `298`
- **Priority**: Medium
- **Category**: Physics/Boundaries
- **Description**: Strength of boundary forces

---

## 10. PHYSICS CATEGORY - Advanced Simulation

### warmup_curve (warmupCurve in camelCase)
- **Current Value**: `"exponential"` (both graphs)
- **Type**: String
- **Validation**: "linear", "exponential", "sigmoid"
- **Location**: `src/config/mod.rs:780`, `data/settings.yaml:179`, `299`
- **Priority**: Low
- **Category**: Physics/Warmup
- **Description**: Warmup curve type

### zero_velocity_iterations (zeroVelocityIterations in camelCase)
- **Current Value**: `10` (both graphs)
- **Type**: u32
- **Location**: `src/config/mod.rs:782`, `data/settings.yaml:180`, `300`
- **Priority**: Low
- **Category**: Physics/Warmup
- **Description**: Iterations with zero velocity at start

### constraint_ramp_frames (constraintRampFrames in camelCase)
- **Current Value**: `60` (both graphs)
- **Type**: u32
- **Location**: `src/config/mod.rs:784`, `data/settings.yaml:181`, `301`
- **Priority**: Medium
- **Category**: Physics/Constraints
- **Description**: Frames to ramp up constraint forces

### constraint_max_force_per_node (constraintMaxForcePerNode in camelCase)
- **Current Value**: `50.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:786`, `data/settings.yaml:182`, `302`
- **Priority**: High
- **Category**: Physics/Constraints
- **Description**: Maximum constraint force per node

---

## 11. PHYSICS CATEGORY - Clustering

### clustering_algorithm (clusteringAlgorithm in camelCase)
- **Current Value**: `"louvain"` (both graphs)
- **Type**: String
- **Validation**: "louvain", "kmeans", "hierarchical"
- **Location**: `src/config/mod.rs:788`, `data/settings.yaml:183`, `303`
- **Priority**: Medium
- **Category**: Physics/Clustering
- **Description**: Clustering algorithm selection

### cluster_count (clusterCount in camelCase)
- **Current Value**: `5` (both graphs)
- **Type**: u32
- **Location**: `src/config/mod.rs:790`, `data/settings.yaml:184`, `304`
- **Priority**: Medium
- **Category**: Physics/Clustering
- **Description**: Target number of clusters

### clustering_resolution (clusteringResolution in camelCase)
- **Current Value**: `1.0` (both graphs)
- **Type**: f32
- **Location**: `src/config/mod.rs:792`, `data/settings.yaml:185`, `305`
- **Priority**: Low
- **Category**: Physics/Clustering
- **Description**: Clustering resolution parameter

### clustering_iterations (clusteringIterations in camelCase)
- **Current Value**: `50` (both graphs)
- **Type**: u32
- **Location**: `src/config/mod.rs:794`, `data/settings.yaml:186`, `306`
- **Priority**: Low
- **Category**: Physics/Clustering
- **Description**: Maximum clustering iterations

---

## 12. PHYSICS CATEGORY - Auto-Balance

### auto_balance (autoBalance in camelCase)
- **Current Value**: `false` (both graphs)
- **Type**: bool
- **Location**: `src/config/mod.rs:696`, `data/settings.yaml:101`, `221`
- **Priority**: High
- **Category**: Physics/AutoTune
- **Description**: Enable automatic physics parameter balancing

### auto_balance_interval_ms (autoBalanceIntervalMs in camelCase)
- **Current Value**: `500` (both graphs)
- **Type**: u32
- **Location**: `src/config/mod.rs:698`, `data/settings.yaml:102`, `222`
- **Priority**: Medium
- **Category**: Physics/AutoTune
- **Description**: Auto-balance check interval in milliseconds

### auto_balance_config.stability_variance_threshold
- **Current Value**: `100.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:104`, `224`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Variance threshold for stability detection

### auto_balance_config.stability_frame_count
- **Current Value**: `180` (logseq), `180` (visionflow)
- **Type**: u32
- **Location**: `data/settings.yaml:105`, `225`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Frames to check for stability

### auto_balance_config.clustering_distance_threshold
- **Current Value**: `20.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:106`, `226`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Distance threshold for clustering detection

### auto_balance_config.clustering_hysteresis_buffer
- **Current Value**: `5.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:107`, `227`
- **Priority**: Low
- **Category**: Physics/AutoTune/Detection
- **Description**: Hysteresis buffer for clustering state changes

### auto_balance_config.bouncing_node_percentage
- **Current Value**: `0.33` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:108`, `228`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Percentage of nodes bouncing to trigger adjustment

### auto_balance_config.boundary_min_distance
- **Current Value**: `90.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:109`, `229`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Boundaries
- **Description**: Minimum distance from boundary

### auto_balance_config.boundary_max_distance
- **Current Value**: `100.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:110`, `230`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Boundaries
- **Description**: Maximum distance from boundary

### auto_balance_config.extreme_distance_threshold
- **Current Value**: `1000.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:111`, `231`
- **Priority**: High
- **Category**: Physics/AutoTune/Detection
- **Description**: Threshold for extreme distance detection

### auto_balance_config.explosion_distance_threshold
- **Current Value**: `10000.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:112`, `232`
- **Priority**: Critical
- **Category**: Physics/AutoTune/Detection
- **Description**: Threshold for explosion detection

### auto_balance_config.spreading_distance_threshold
- **Current Value**: `500.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:113`, `233`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Threshold for spreading detection

### auto_balance_config.spreading_hysteresis_buffer
- **Current Value**: `50.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:114`, `234`
- **Priority**: Low
- **Category**: Physics/AutoTune/Detection
- **Description**: Hysteresis buffer for spreading state

### auto_balance_config.oscillation_detection_frames
- **Current Value**: `20` (logseq), `10` (visionflow)
- **Type**: u32
- **Location**: `data/settings.yaml:115`, `235`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Frames to detect oscillation

### auto_balance_config.oscillation_change_threshold
- **Current Value**: `10.0` (logseq), `5.0` (visionflow)
- **Type**: f32
- **Location**: `data/settings.yaml:116`, `236`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Change threshold for oscillation detection

### auto_balance_config.min_oscillation_changes
- **Current Value**: `8` (logseq), `5` (visionflow)
- **Type**: usize
- **Location**: `data/settings.yaml:117`, `237`
- **Priority**: Low
- **Category**: Physics/AutoTune/Detection
- **Description**: Minimum changes to detect oscillation

### auto_balance_config.parameter_adjustment_rate
- **Current Value**: `0.1` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:118`, `238`
- **Priority**: High
- **Category**: Physics/AutoTune/Adjustment
- **Description**: Rate of parameter adjustment

### auto_balance_config.max_adjustment_factor
- **Current Value**: `0.2` (logseq), `2.0` (visionflow)
- **Type**: f32
- **Location**: `data/settings.yaml:119`, `239`
- **Priority**: High
- **Category**: Physics/AutoTune/Adjustment
- **Description**: Maximum adjustment multiplier

### auto_balance_config.min_adjustment_factor
- **Current Value**: `-0.2` (logseq), `0.5` (visionflow)
- **Type**: f32
- **Location**: `data/settings.yaml:120`, `240`
- **Priority**: High
- **Category**: Physics/AutoTune/Adjustment
- **Description**: Minimum adjustment multiplier

### auto_balance_config.adjustment_cooldown_ms
- **Current Value**: `2000` (logseq), `1000` (visionflow)
- **Type**: u64
- **Location**: `data/settings.yaml:121`, `241`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Timing
- **Description**: Cooldown between adjustments in milliseconds

### auto_balance_config.state_change_cooldown_ms
- **Current Value**: `1000` (logseq), `500` (visionflow)
- **Type**: u64
- **Location**: `data/settings.yaml:122`, `242`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Timing
- **Description**: Cooldown between state changes

### auto_balance_config.parameter_dampening_factor
- **Current Value**: `0.05` (logseq), `0.9` (visionflow)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:123`, `243`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Adjustment
- **Description**: Dampening factor for parameter changes

### auto_balance_config.hysteresis_delay_frames
- **Current Value**: `30` (both graphs)
- **Type**: u32
- **Location**: `data/settings.yaml:124`, `244`
- **Priority**: Low
- **Category**: Physics/AutoTune/Timing
- **Description**: Hysteresis delay in frames

### auto_balance_config.grid_cell_size_min
- **Current Value**: `1.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:125`, `245`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Ranges
- **Description**: Minimum grid cell size for auto-tuning

### auto_balance_config.grid_cell_size_max
- **Current Value**: `50.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:126`, `246`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Ranges
- **Description**: Maximum grid cell size for auto-tuning

### auto_balance_config.repulsion_cutoff_min
- **Current Value**: `5.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:127`, `247`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Ranges
- **Description**: Minimum repulsion cutoff for auto-tuning

### auto_balance_config.repulsion_cutoff_max
- **Current Value**: `200.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:128`, `248`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Ranges
- **Description**: Maximum repulsion cutoff for auto-tuning

### auto_balance_config.repulsion_softening_min
- **Current Value**: `0.000001` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:129`, `249`
- **Priority**: Low
- **Category**: Physics/AutoTune/Ranges
- **Description**: Minimum repulsion softening for auto-tuning

### auto_balance_config.repulsion_softening_max
- **Current Value**: `1.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:130`, `250`
- **Priority**: Low
- **Category**: Physics/AutoTune/Ranges
- **Description**: Maximum repulsion softening for auto-tuning

### auto_balance_config.center_gravity_min
- **Current Value**: `0.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:131`, `251`
- **Priority**: Low
- **Category**: Physics/AutoTune/Ranges
- **Description**: Minimum center gravity for auto-tuning

### auto_balance_config.center_gravity_max
- **Current Value**: `0.1` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:132`, `252`
- **Priority**: Low
- **Category**: Physics/AutoTune/Ranges
- **Description**: Maximum center gravity for auto-tuning

### auto_balance_config.spatial_hash_efficiency_threshold
- **Current Value**: `0.3` (both graphs)
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:133`, `253`
- **Priority**: Low
- **Category**: Physics/AutoTune/Performance
- **Description**: Efficiency threshold for spatial hash

### auto_balance_config.cluster_density_threshold
- **Current Value**: `50.0` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:134`, `254`
- **Priority**: Medium
- **Category**: Physics/AutoTune/Detection
- **Description**: Density threshold for cluster detection

### auto_balance_config.numerical_instability_threshold
- **Current Value**: `0.001` (both graphs)
- **Type**: f32
- **Location**: `data/settings.yaml:135`, `255`
- **Priority**: Critical
- **Category**: Physics/AutoTune/Detection
- **Description**: Threshold for numerical instability detection

---

## 13. PHYSICS CATEGORY - Auto-Pause

### auto_pause.enabled
- **Current Value**: `true` (logseq), `false` (visionflow)
- **Type**: bool
- **Location**: `data/settings.yaml:137`, `257`
- **Priority**: Medium
- **Category**: Physics/AutoPause
- **Description**: Enable automatic pause on equilibrium

### auto_pause.equilibrium_velocity_threshold
- **Current Value**: `0.1` (logseq), `0.05` (visionflow)
- **Type**: f32
- **Location**: `data/settings.yaml:138`, `258`
- **Priority**: Medium
- **Category**: Physics/AutoPause
- **Description**: Velocity threshold for equilibrium detection

### auto_pause.equilibrium_check_frames
- **Current Value**: `30` (logseq), `60` (visionflow)
- **Type**: u32
- **Location**: `data/settings.yaml:139`, `259`
- **Priority**: Medium
- **Category**: Physics/AutoPause
- **Description**: Frames to check for equilibrium

### auto_pause.equilibrium_energy_threshold
- **Current Value**: `0.01` (logseq), `0.005` (visionflow)
- **Type**: f32
- **Location**: `data/settings.yaml:140`, `260`
- **Priority**: Medium
- **Category**: Physics/AutoPause
- **Description**: Energy threshold for equilibrium

### auto_pause.pause_on_equilibrium
- **Current Value**: `true` (logseq), `false` (visionflow)
- **Type**: bool
- **Location**: `data/settings.yaml:141`, `261`
- **Priority**: High
- **Category**: Physics/AutoPause
- **Description**: Automatically pause when equilibrium reached

### auto_pause.resume_on_interaction
- **Current Value**: `true` (both graphs)
- **Type**: bool
- **Location**: `data/settings.yaml:142`, `262`
- **Priority**: High
- **Category**: Physics/AutoPause
- **Description**: Resume simulation on user interaction

---

## 14. RENDERING CATEGORY

### ambient_light_intensity (ambientLightIntensity in camelCase)
- **Current Value**: `0.5`
- **Type**: f32
- **Validation**: 0.0-2.0
- **Location**: `src/config/mod.rs:868`, `data/settings.yaml:3`
- **Priority**: Medium
- **Category**: Rendering/Lighting
- **Description**: Ambient light intensity

### background_color (backgroundColor in camelCase)
- **Current Value**: `"#000000"`
- **Type**: String
- **Validation**: HEX_COLOR_REGEX
- **Location**: `src/config/mod.rs:870`, `data/settings.yaml:4`
- **Priority**: Low
- **Category**: Rendering/Visual
- **Description**: Background color

### directional_light_intensity (directionalLightIntensity in camelCase)
- **Current Value**: `0.4`
- **Type**: f32
- **Validation**: 0.0-2.0
- **Location**: `src/config/mod.rs:872`, `data/settings.yaml:5`
- **Priority**: Medium
- **Category**: Rendering/Lighting
- **Description**: Directional light intensity

### enable_ambient_occlusion (enableAmbientOcclusion in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:874`, `data/settings.yaml:6`
- **Priority**: Low
- **Category**: Rendering/Effects
- **Description**: Enable screen-space ambient occlusion

### enable_antialiasing (enableAntialiasing in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:876`, `data/settings.yaml:7`
- **Priority**: Medium
- **Category**: Rendering/Quality
- **Description**: Enable antialiasing

### enable_shadows (enableShadows in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:878`, `data/settings.yaml:8`
- **Priority**: Low
- **Category**: Rendering/Effects
- **Description**: Enable shadow rendering

### environment_intensity (environmentIntensity in camelCase)
- **Current Value**: `0.3`
- **Type**: f32
- **Validation**: 0.0-2.0
- **Location**: `src/config/mod.rs:880`, `data/settings.yaml:9`
- **Priority**: Low
- **Category**: Rendering/Lighting
- **Description**: Environment map intensity

### shadow_map_size (shadowMapSize in camelCase)
- **Current Value**: `"2048"`
- **Type**: String
- **Validation**: Must be power of 2: "512", "1024", "2048", "4096"
- **Location**: `src/config/mod.rs:882`, `data/settings.yaml:10`
- **Priority**: Medium
- **Category**: Rendering/Shadows
- **Description**: Shadow map resolution

### shadow_bias (shadowBias in camelCase)
- **Current Value**: `0.0001`
- **Type**: f32
- **Location**: `src/config/mod.rs:884`, `data/settings.yaml:11`
- **Priority**: Low
- **Category**: Rendering/Shadows
- **Description**: Shadow bias to prevent acne

### context
- **Current Value**: `"desktop"`
- **Type**: String
- **Validation**: "desktop", "mobile", "xr"
- **Location**: `src/config/mod.rs:886`, `data/settings.yaml:12`
- **Priority**: High
- **Category**: Rendering/Context
- **Description**: Rendering context type

---

## 15. ANIMATION CATEGORY

### enable_motion_blur (enableMotionBlur in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:893`, `data/settings.yaml:14`
- **Priority**: Low
- **Category**: Animation/Effects
- **Description**: Enable motion blur effect

### enable_node_animations (enableNodeAnimations in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:895`, `data/settings.yaml:15`
- **Priority**: Medium
- **Category**: Animation/Nodes
- **Description**: Enable node animation effects

### motion_blur_strength (motionBlurStrength in camelCase)
- **Current Value**: `0.1`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:897`, `data/settings.yaml:16`
- **Priority**: Low
- **Category**: Animation/Effects
- **Description**: Motion blur strength

### selection_wave_enabled (selectionWaveEnabled in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `data/settings.yaml:17`
- **Priority**: Low
- **Category**: Animation/Selection
- **Description**: Enable selection wave animation

### pulse_enabled (pulseEnabled in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:18`
- **Priority**: Low
- **Category**: Animation/Effects
- **Description**: Enable pulse animation

### pulse_speed (pulseSpeed in camelCase)
- **Current Value**: `1.2`
- **Type**: f32
- **Validation**: 0.0-5.0
- **Location**: `data/settings.yaml:19`
- **Priority**: Low
- **Category**: Animation/Effects
- **Description**: Pulse animation speed

### pulse_strength (pulseStrength in camelCase)
- **Current Value**: `0.8`
- **Type**: f32
- **Validation**: 0.0-2.0
- **Location**: `data/settings.yaml:20`
- **Priority**: Low
- **Category**: Animation/Effects
- **Description**: Pulse animation strength

### wave_speed (waveSpeed in camelCase)
- **Current Value**: `0.5`
- **Type**: f32
- **Validation**: 0.0-5.0
- **Location**: `data/settings.yaml:21`
- **Priority**: Low
- **Category**: Animation/Effects
- **Description**: Wave animation speed

---

## 16. GLOW & BLOOM CATEGORY

### glow.enabled
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:939`, `data/settings.yaml:23`
- **Priority**: Medium
- **Category**: Effects/Glow
- **Description**: Enable glow post-processing effect

### glow.intensity
- **Current Value**: `1.2`
- **Type**: f32
- **Validation**: 0.0-10.0
- **Location**: `src/config/mod.rs:941`, `data/settings.yaml:24`
- **Priority**: Medium
- **Category**: Effects/Glow
- **Description**: Glow intensity

### glow.radius
- **Current Value**: `1.2`
- **Type**: f32
- **Validation**: 0.0-10.0
- **Location**: `src/config/mod.rs:943`, `data/settings.yaml:25`
- **Priority**: Medium
- **Category**: Effects/Glow
- **Description**: Glow radius

### glow.threshold
- **Current Value**: `0.39`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:945`, `data/settings.yaml:26`
- **Priority**: Medium
- **Category**: Effects/Glow
- **Description**: Brightness threshold for glow

### glow.base_color (baseColor in camelCase)
- **Current Value**: `"#00ffff"`
- **Type**: String
- **Validation**: HEX_COLOR_REGEX
- **Location**: `src/config/mod.rs:947`, `data/settings.yaml:30`
- **Priority**: Low
- **Category**: Effects/Glow
- **Description**: Glow base color

### glow.emission_color (emissionColor in camelCase)
- **Current Value**: `"#00e5ff"`
- **Type**: String
- **Validation**: HEX_COLOR_REGEX
- **Location**: `src/config/mod.rs:949`, `data/settings.yaml:31`
- **Priority**: Low
- **Category**: Effects/Glow
- **Description**: Glow emission color

### glow.opacity
- **Current Value**: `0.6`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:951`, `data/settings.yaml:32`
- **Priority**: Low
- **Category**: Effects/Glow
- **Description**: Glow opacity

### bloom.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:1016`, `data/settings.yaml:39`
- **Priority**: Medium
- **Category**: Effects/Bloom
- **Description**: Enable bloom post-processing effect

### bloom.intensity
- **Current Value**: `1.0`
- **Type**: f32
- **Validation**: 0.0-10.0
- **Location**: `src/config/mod.rs:1018`, `data/settings.yaml:40`
- **Priority**: Medium
- **Category**: Effects/Bloom
- **Description**: Bloom intensity

### bloom.radius
- **Current Value**: `0.8`
- **Type**: f32
- **Validation**: 0.0-10.0
- **Location**: `src/config/mod.rs:1020`, `data/settings.yaml:41`
- **Priority**: Medium
- **Category**: Effects/Bloom
- **Description**: Bloom radius

### bloom.threshold
- **Current Value**: `0.15`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:1022`, `data/settings.yaml:42`
- **Priority**: Medium
- **Category**: Effects/Bloom
- **Description**: Brightness threshold for bloom

### bloom.strength
- **Current Value**: `0.0`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:1024`, `data/settings.yaml:45`
- **Priority**: Low
- **Category**: Effects/Bloom
- **Description**: Bloom strength multiplier

### bloom.knee
- **Current Value**: `0.0`
- **Type**: f32
- **Validation**: 0.0-2.0
- **Location**: `src/config/mod.rs:1026`, `data/settings.yaml:47`
- **Priority**: Low
- **Category**: Effects/Bloom
- **Description**: Bloom soft knee parameter

---

## 17. XR (EXTENDED REALITY) CATEGORY

### xr.enabled
- **Current Value**: `false`
- **Type**: Option<bool>
- **Location**: `src/config/mod.rs:1398`, `data/settings.yaml:371`
- **Priority**: High
- **Category**: XR/System
- **Description**: Enable XR mode

### xr.client_side_enable_xr (clientSideEnableXr in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `data/settings.yaml:372`
- **Priority**: High
- **Category**: XR/System
- **Description**: Enable XR on client side

### xr.mode
- **Current Value**: `"immersive-vr"`
- **Type**: String
- **Validation**: "immersive-vr", "immersive-ar", "inline"
- **Location**: `src/config/mod.rs:1400`, `data/settings.yaml:373`
- **Priority**: High
- **Category**: XR/System
- **Description**: XR session mode

### xr.room_scale (roomScale in camelCase)
- **Current Value**: `1.0`
- **Type**: f32
- **Validation**: 0.1-10.0
- **Location**: `src/config/mod.rs:1402`, `data/settings.yaml:374`
- **Priority**: Medium
- **Category**: XR/Space
- **Description**: Room scale multiplier

### xr.space_type (spaceType in camelCase)
- **Current Value**: `"local-floor"`
- **Type**: String
- **Validation**: "local", "local-floor", "bounded-floor", "unbounded"
- **Location**: `src/config/mod.rs:1404`, `data/settings.yaml:375`
- **Priority**: High
- **Category**: XR/Space
- **Description**: XR reference space type

### xr.quality
- **Current Value**: `"medium"`
- **Type**: String
- **Validation**: "low", "medium", "high", "ultra"
- **Location**: `src/config/mod.rs:1406`, `data/settings.yaml:376`
- **Priority**: Medium
- **Category**: XR/Performance
- **Description**: Rendering quality preset

### xr.render_scale (renderScale in camelCase)
- **Current Value**: `1.2`
- **Type**: f32
- **Validation**: 0.5-2.0
- **Location**: `src/config/mod.rs:1408`, `data/settings.yaml:377`
- **Priority**: Medium
- **Category**: XR/Performance
- **Description**: Render resolution scale

### xr.interaction_distance (interactionDistance in camelCase)
- **Current Value**: `1.5`
- **Type**: f32
- **Location**: `src/config/mod.rs:1410`, `data/settings.yaml:378`
- **Priority**: Medium
- **Category**: XR/Interaction
- **Description**: Maximum interaction distance

### xr.locomotion_method (locomotionMethod in camelCase)
- **Current Value**: `"teleport"`
- **Type**: String
- **Validation**: "teleport", "smooth", "snap-turn"
- **Location**: `src/config/mod.rs:1412`, `data/settings.yaml:379`
- **Priority**: High
- **Category**: XR/Locomotion
- **Description**: User locomotion method

### xr.enable_hand_tracking (enableHandTracking in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:1426`, `data/settings.yaml:383`
- **Priority**: Medium
- **Category**: XR/Input
- **Description**: Enable hand tracking

### xr.enable_haptics (enableHaptics in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `src/config/mod.rs:1445`, `data/settings.yaml:392`
- **Priority**: Low
- **Category**: XR/Feedback
- **Description**: Enable haptic feedback

### xr.haptic_intensity (hapticIntensity in camelCase)
- **Current Value**: `0.3`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:1447`, `data/settings.yaml:393`
- **Priority**: Low
- **Category**: XR/Feedback
- **Description**: Haptic feedback intensity

### xr.enable_plane_detection (enablePlaneDetection in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1466`, `data/settings.yaml:404`
- **Priority**: Medium
- **Category**: XR/AR
- **Description**: Enable AR plane detection

### xr.enable_passthrough_portal (enablePassthroughPortal in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `src/config/mod.rs:1481`, `data/settings.yaml:411`
- **Priority**: Low
- **Category**: XR/AR
- **Description**: Enable passthrough portal feature

---

## 18. EXTERNAL SERVICES - RAGFlow

### ragflow.agent_id (agentId in camelCase)
- **Current Value**: `"aa2e328812ef11f083dc0a0d6226f61b"`
- **Type**: String
- **Location**: `src/config/mod.rs:1511`, `data/settings.yaml:423`
- **Priority**: High
- **Category**: Services/RAGFlow
- **Description**: RAGFlow agent identifier

### ragflow.timeout
- **Current Value**: `30` (seconds)
- **Type**: Option<u64>
- **Location**: `src/config/mod.rs:1517`, `data/settings.yaml:424`
- **Priority**: Medium
- **Category**: Services/RAGFlow
- **Description**: Request timeout

### ragflow.max_retries (maxRetries in camelCase)
- **Current Value**: `3`
- **Type**: Option<u32>
- **Location**: `src/config/mod.rs:1519`, `data/settings.yaml:425`
- **Priority**: Medium
- **Category**: Services/RAGFlow
- **Description**: Maximum retry attempts

---

## 19. EXTERNAL SERVICES - Perplexity AI

### perplexity.model
- **Current Value**: `"llama-3.1-sonar-small-128k-online"`
- **Type**: String
- **Location**: `src/config/mod.rs:1528`, `data/settings.yaml:427`
- **Priority**: High
- **Category**: Services/Perplexity
- **Description**: Perplexity AI model name

### perplexity.max_tokens (maxTokens in camelCase)
- **Current Value**: `4096`
- **Type**: Option<u32>
- **Location**: `src/config/mod.rs:1534`, `data/settings.yaml:428`
- **Priority**: Medium
- **Category**: Services/Perplexity
- **Description**: Maximum response tokens

### perplexity.temperature
- **Current Value**: `0.5`
- **Type**: f32
- **Validation**: 0.0-2.0
- **Location**: `src/config/mod.rs:1536`, `data/settings.yaml:429`
- **Priority**: Medium
- **Category**: Services/Perplexity
- **Description**: Sampling temperature

### perplexity.top_p (topP in camelCase)
- **Current Value**: `0.9`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `src/config/mod.rs:1538`, `data/settings.yaml:430`
- **Priority**: Low
- **Category**: Services/Perplexity
- **Description**: Top-p nucleus sampling

### perplexity.presence_penalty (presencePenalty in camelCase)
- **Current Value**: `0.0`
- **Type**: f32
- **Validation**: -2.0 to 2.0
- **Location**: `src/config/mod.rs:1540`, `data/settings.yaml:431`
- **Priority**: Low
- **Category**: Services/Perplexity
- **Description**: Presence penalty

### perplexity.frequency_penalty (frequencyPenalty in camelCase)
- **Current Value**: `0.0`
- **Type**: f32
- **Validation**: -2.0 to 2.0
- **Location**: `src/config/mod.rs:1542`, `data/settings.yaml:432`
- **Priority**: Low
- **Category**: Services/Perplexity
- **Description**: Frequency penalty

### perplexity.timeout
- **Current Value**: `30` (seconds)
- **Type**: Option<u64>
- **Location**: `src/config/mod.rs:1544`, `data/settings.yaml:433`
- **Priority**: Medium
- **Category**: Services/Perplexity
- **Description**: Request timeout

### perplexity.rate_limit (rateLimit in camelCase)
- **Current Value**: `100` (requests per minute)
- **Type**: Option<u32>
- **Location**: `src/config/mod.rs:1546`, `data/settings.yaml:434`
- **Priority**: Medium
- **Category**: Services/Perplexity
- **Description**: Rate limit

---

## 20. EXTERNAL SERVICES - OpenAI

### openai.timeout
- **Current Value**: `30` (seconds)
- **Type**: Option<u64>
- **Location**: `src/config/mod.rs:1557`, `data/settings.yaml:436`
- **Priority**: Medium
- **Category**: Services/OpenAI
- **Description**: Request timeout

### openai.rate_limit (rateLimit in camelCase)
- **Current Value**: `100` (requests per minute)
- **Type**: Option<u32>
- **Location**: `src/config/mod.rs:1559`, `data/settings.yaml:437`
- **Priority**: Medium
- **Category**: Services/OpenAI
- **Description**: Rate limit

---

## 21. EXTERNAL SERVICES - Kokoro TTS

### kokoro.api_url (apiUrl in camelCase)
- **Current Value**: `"http://kokoro-tts-container:8880"`
- **Type**: String
- **Validation**: URL_REGEX
- **Location**: `src/config/mod.rs:1566`, `data/settings.yaml:439`
- **Priority**: High
- **Category**: Services/Kokoro
- **Description**: Kokoro TTS API endpoint

### kokoro.default_voice (defaultVoice in camelCase)
- **Current Value**: `"af_heart"`
- **Type**: String
- **Location**: `src/config/mod.rs:1568`, `data/settings.yaml:440`
- **Priority**: Medium
- **Category**: Services/Kokoro
- **Description**: Default voice selection

### kokoro.default_format (defaultFormat in camelCase)
- **Current Value**: `"mp3"`
- **Type**: String
- **Validation**: "mp3", "wav", "opus"
- **Location**: `src/config/mod.rs:1570`, `data/settings.yaml:441`
- **Priority**: Low
- **Category**: Services/Kokoro
- **Description**: Audio output format

### kokoro.default_speed (defaultSpeed in camelCase)
- **Current Value**: `1.0`
- **Type**: f32
- **Validation**: 0.5-2.0
- **Location**: `src/config/mod.rs:1572`, `data/settings.yaml:442`
- **Priority**: Low
- **Category**: Services/Kokoro
- **Description**: Speech speed multiplier

### kokoro.timeout
- **Current Value**: `30` (seconds)
- **Type**: Option<u64>
- **Location**: `src/config/mod.rs:1574`, `data/settings.yaml:443`
- **Priority**: Medium
- **Category**: Services/Kokoro
- **Description**: Request timeout

### kokoro.stream
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:444`
- **Priority**: Medium
- **Category**: Services/Kokoro
- **Description**: Enable streaming mode

### kokoro.return_timestamps (returnTimestamps in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:445`
- **Priority**: Low
- **Category**: Services/Kokoro
- **Description**: Return word timestamps

### kokoro.sample_rate (sampleRate in camelCase)
- **Current Value**: `24000` (Hz)
- **Type**: u32
- **Location**: `data/settings.yaml:446`
- **Priority**: Low
- **Category**: Services/Kokoro
- **Description**: Audio sample rate

---

## 22. EXTERNAL SERVICES - Whisper STT

### whisper.api_url (apiUrl in camelCase)
- **Current Value**: `"http://whisper-webui-backend:8000"`
- **Type**: String
- **Validation**: URL_REGEX
- **Location**: `src/config/mod.rs:1587`, `data/settings.yaml:448`
- **Priority**: High
- **Category**: Services/Whisper
- **Description**: Whisper STT API endpoint

### whisper.default_model (defaultModel in camelCase)
- **Current Value**: `"base"`
- **Type**: String
- **Validation**: "tiny", "base", "small", "medium", "large"
- **Location**: `src/config/mod.rs:1589`, `data/settings.yaml:449`
- **Priority**: High
- **Category**: Services/Whisper
- **Description**: Whisper model size

### whisper.default_language (defaultLanguage in camelCase)
- **Current Value**: `"en"`
- **Type**: String
- **Location**: `src/config/mod.rs:1591`, `data/settings.yaml:450`
- **Priority**: Medium
- **Category**: Services/Whisper
- **Description**: Default transcription language

### whisper.timeout
- **Current Value**: `30` (seconds)
- **Type**: Option<u64>
- **Location**: `src/config/mod.rs:1593`, `data/settings.yaml:451`
- **Priority**: Medium
- **Category**: Services/Whisper
- **Description**: Request timeout

### whisper.temperature
- **Current Value**: `0.0`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:452`
- **Priority**: Low
- **Category**: Services/Whisper
- **Description**: Sampling temperature

### whisper.return_timestamps (returnTimestamps in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `data/settings.yaml:453`
- **Priority**: Low
- **Category**: Services/Whisper
- **Description**: Return word timestamps

### whisper.vad_filter (vadFilter in camelCase)
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:454`
- **Priority**: Medium
- **Category**: Services/Whisper
- **Description**: Enable voice activity detection filter

### whisper.word_timestamps (wordTimestamps in camelCase)
- **Current Value**: `false`
- **Type**: bool
- **Location**: `data/settings.yaml:455`
- **Priority**: Low
- **Category**: Services/Whisper
- **Description**: Generate word-level timestamps

---

## 23. ONTOLOGY CATEGORY

### ontology.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:457`
- **Priority**: High
- **Category**: Ontology/System
- **Description**: Enable ontology reasoning system

### ontology.validation.mode
- **Current Value**: `"full"`
- **Type**: String
- **Validation**: "quick", "full", "incremental"
- **Location**: `data/settings.yaml:459`
- **Priority**: High
- **Category**: Ontology/Validation
- **Description**: Validation mode selection

### ontology.validation.auto_validate
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:460`
- **Priority**: Medium
- **Category**: Ontology/Validation
- **Description**: Automatically validate on changes

### ontology.validation.validation_interval_ms
- **Current Value**: `5000`
- **Type**: u64
- **Location**: `data/settings.yaml:461`
- **Priority**: Medium
- **Category**: Ontology/Validation
- **Description**: Validation check interval

### ontology.constraints.disjoint_classes.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:465`
- **Priority**: High
- **Category**: Ontology/Constraints
- **Description**: Enable disjoint class constraints

### ontology.constraints.disjoint_classes.separation_strength
- **Current Value**: `0.8`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:466`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Force strength for class separation

### ontology.constraints.disjoint_classes.min_distance
- **Current Value**: `5.0`
- **Type**: f32
- **Location**: `data/settings.yaml:467`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Minimum distance between disjoint classes

### ontology.constraints.subclass_hierarchy.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:470`
- **Priority**: High
- **Category**: Ontology/Constraints
- **Description**: Enable subclass hierarchy constraints

### ontology.constraints.subclass_hierarchy.alignment_strength
- **Current Value**: `0.6`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:471`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Strength of hierarchy alignment

### ontology.constraints.subclass_hierarchy.vertical_spacing
- **Current Value**: `3.0`
- **Type**: f32
- **Location**: `data/settings.yaml:472`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Vertical spacing between hierarchy levels

### ontology.constraints.sameas_colocate.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:475`
- **Priority**: High
- **Category**: Ontology/Constraints
- **Description**: Enable sameAs colocation constraints

### ontology.constraints.sameas_colocate.colocate_strength
- **Current Value**: `0.9`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:476`
- **Priority**: High
- **Category**: Ontology/Constraints
- **Description**: Force strength for sameAs colocation

### ontology.constraints.sameas_colocate.max_distance
- **Current Value**: `0.5`
- **Type**: f32
- **Location**: `data/settings.yaml:477`
- **Priority**: High
- **Category**: Ontology/Constraints
- **Description**: Maximum distance for sameAs nodes

### ontology.constraints.inverse_symmetry.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:480`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Enable inverse property symmetry

### ontology.constraints.inverse_symmetry.symmetry_strength
- **Current Value**: `0.7`
- **Type**: f32
- **Validation**: 0.0-1.0
- **Location**: `data/settings.yaml:481`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Symmetry force strength

### ontology.constraints.functional_cardinality.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:484`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Enable functional property cardinality

### ontology.constraints.functional_cardinality.cardinality_penalty
- **Current Value**: `1.0`
- **Type**: f32
- **Location**: `data/settings.yaml:485`
- **Priority**: Medium
- **Category**: Ontology/Constraints
- **Description**: Penalty for cardinality violations

### ontology.physics.use_gpu
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:488`
- **Priority**: High
- **Category**: Ontology/Performance
- **Description**: Use GPU for ontology physics

### ontology.physics.fallback_to_cpu
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:489`
- **Priority**: High
- **Category**: Ontology/Performance
- **Description**: Fallback to CPU if GPU fails

### ontology.physics.max_iterations
- **Current Value**: `100`
- **Type**: u32
- **Location**: `data/settings.yaml:490`
- **Priority**: Medium
- **Category**: Ontology/Performance
- **Description**: Maximum constraint iterations

### ontology.physics.convergence_threshold
- **Current Value**: `0.001`
- **Type**: f32
- **Location**: `data/settings.yaml:491`
- **Priority**: Medium
- **Category**: Ontology/Performance
- **Description**: Convergence threshold for constraint solving

### ontology.cache.enabled
- **Current Value**: `true`
- **Type**: bool
- **Location**: `data/settings.yaml:494`
- **Priority**: High
- **Category**: Ontology/Cache
- **Description**: Enable ontology caching

### ontology.cache.ttl_seconds
- **Current Value**: `3600` (1 hour)
- **Type**: u64
- **Location**: `data/settings.yaml:495`
- **Priority**: Medium
- **Category**: Ontology/Cache
- **Description**: Cache time-to-live

### ontology.cache.max_ontologies
- **Current Value**: `10`
- **Type**: usize
- **Location**: `data/settings.yaml:496`
- **Priority**: Medium
- **Category**: Ontology/Cache
- **Description**: Maximum cached ontologies

---

## 24. INTERNAL SETTINGS (DevConfig) - Physics Internals

**Location:** `src/config/dev_config.rs`
**Note:** These are server-side only and NOT exposed to clients

### force_epsilon
- **Current Value**: `1e-8`
- **Type**: f32
- **Location**: `dev_config.rs:23`
- **Priority**: Critical
- **Category**: Internal/Physics
- **Description**: Small value to prevent division by zero

### spring_length_multiplier
- **Current Value**: `5.0`
- **Type**: f32
- **Location**: `dev_config.rs:24`
- **Priority**: Medium
- **Category**: Internal/Physics
- **Description**: Natural spring length = separation_radius * this

### spring_length_max
- **Current Value**: `10.0`
- **Type**: f32
- **Location**: `dev_config.rs:25`
- **Priority**: Medium
- **Category**: Internal/Physics
- **Description**: Maximum natural spring length

### spring_force_clamp_factor
- **Current Value**: `0.5`
- **Type**: f32
- **Location**: `dev_config.rs:26`
- **Priority**: Medium
- **Category**: Internal/Physics
- **Description**: Clamp spring forces to max_force * this

---

## 25. INTERNAL SETTINGS - GPU Kernel Parameters

### max_force (Internal GPU)
- **Current Value**: `15.0`
- **Type**: f32
- **Location**: `dev_config.rs:38`
- **Priority**: Critical
- **Category**: Internal/GPU
- **Description**: Maximum force magnitude for clamping

### max_velocity (Internal GPU)
- **Current Value**: `50.0`
- **Type**: f32
- **Location**: `dev_config.rs:39`
- **Priority**: Critical
- **Category**: Internal/GPU
- **Description**: Maximum velocity magnitude for clamping

### world_bounds_min
- **Current Value**: `-1000.0`
- **Type**: f32
- **Location**: `dev_config.rs:40`
- **Priority**: High
- **Category**: Internal/GPU
- **Description**: Minimum world coordinate

### world_bounds_max
- **Current Value**: `1000.0`
- **Type**: f32
- **Location**: `dev_config.rs:41`
- **Priority**: High
- **Category**: Internal/GPU
- **Description**: Maximum world coordinate

### cell_size_lod
- **Current Value**: `100.0`
- **Type**: f32
- **Location**: `dev_config.rs:42`
- **Priority**: Medium
- **Category**: Internal/GPU
- **Description**: Level of detail cell size

### k_neighbors_max
- **Current Value**: `32`
- **Type**: u32
- **Location**: `dev_config.rs:43`
- **Priority**: Medium
- **Category**: Internal/GPU
- **Description**: Maximum k-neighbors for LOF

### anomaly_detection_radius
- **Current Value**: `150.0`
- **Type**: f32
- **Location**: `dev_config.rs:44`
- **Priority**: Medium
- **Category**: Internal/GPU
- **Description**: Default radius for anomaly detection

### learning_rate_default
- **Current Value**: `0.1`
- **Type**: f32
- **Location**: `dev_config.rs:45`
- **Priority**: Low
- **Category**: Internal/GPU
- **Description**: Default learning rate for GPU algorithms

### min_velocity_threshold
- **Current Value**: `0.01`
- **Type**: f32
- **Location**: `dev_config.rs:46`
- **Priority**: Medium
- **Category**: Internal/GPU
- **Description**: Minimum velocity threshold for stability gates

### stability_threshold
- **Current Value**: `1e-6`
- **Type**: f32
- **Location**: `dev_config.rs:47`
- **Priority**: High
- **Category**: Internal/GPU
- **Description**: System stability threshold for early exit

### norm_delta_cap
- **Current Value**: `1000.0`
- **Type**: f32
- **Location**: `dev_config.rs:50`
- **Priority**: Medium
- **Category**: Internal/GPU
- **Description**: Cap for SSSP delta normalization

### position_constraint_attraction
- **Current Value**: `0.1`
- **Type**: f32
- **Location**: `dev_config.rs:51`
- **Priority**: Low
- **Category**: Internal/GPU
- **Description**: Gentle attraction factor for position constraints

### lof_score_min
- **Current Value**: `0.1`
- **Type**: f32
- **Location**: `dev_config.rs:52`
- **Priority**: Low
- **Category**: Internal/GPU
- **Description**: Minimum LOF score clamp

### lof_score_max
- **Current Value**: `10.0`
- **Type**: f32
- **Location**: `dev_config.rs:53`
- **Priority**: Low
- **Category**: Internal/GPU
- **Description**: Maximum LOF score clamp

### weight_precision_multiplier
- **Current Value**: `1000.0`
- **Type**: f32
- **Location**: `dev_config.rs:54`
- **Priority**: Low
- **Category**: Internal/GPU
- **Description**: Weight precision multiplier for integer operations

---

## 26. INTERNAL SETTINGS - CUDA Configuration

### warmup_iterations_default
- **Current Value**: `200`
- **Type**: u32
- **Location**: `dev_config.rs:82`
- **Priority**: Medium
- **Category**: Internal/CUDA
- **Description**: Default warmup iterations

### warmup_damping_start
- **Current Value**: `0.98`
- **Type**: f32
- **Location**: `dev_config.rs:83`
- **Priority**: Low
- **Category**: Internal/CUDA
- **Description**: Initial damping during warmup

### warmup_damping_end
- **Current Value**: `0.85`
- **Type**: f32
- **Location**: `dev_config.rs:84`
- **Priority**: Low
- **Category**: Internal/CUDA
- **Description**: Final damping after warmup

### warmup_temperature_scale
- **Current Value**: `0.0001`
- **Type**: f32
- **Location**: `dev_config.rs:85`
- **Priority**: Low
- **Category**: Internal/CUDA
- **Description**: Temperature scaling during warmup

### warmup_cooling_iterations
- **Current Value**: `5`
- **Type**: u32
- **Location**: `dev_config.rs:86`
- **Priority**: Low
- **Category**: Internal/CUDA
- **Description**: Iterations for cooling phase

### max_kernel_time_ms
- **Current Value**: `5000`
- **Type**: u32
- **Location**: `dev_config.rs:89`
- **Priority**: Critical
- **Category**: Internal/CUDA
- **Description**: Maximum CUDA kernel execution time

### max_gpu_failures
- **Current Value**: `5`
- **Type**: u32
- **Location**: `dev_config.rs:90`
- **Priority**: High
- **Category**: Internal/CUDA
- **Description**: Maximum GPU failures before fallback

### debug_output_throttle
- **Current Value**: `60`
- **Type**: u32
- **Location**: `dev_config.rs:91`
- **Priority**: Low
- **Category**: Internal/CUDA
- **Description**: Throttle debug output frequency

### debug_node_count
- **Current Value**: `3`
- **Type**: u32
- **Location**: `dev_config.rs:92`
- **Priority**: Low
- **Category**: Internal/CUDA
- **Description**: Number of nodes to debug output

### max_nodes
- **Current Value**: `1000000`
- **Type**: u32
- **Location**: `dev_config.rs:95`
- **Priority**: Critical
- **Category**: Internal/CUDA
- **Description**: Maximum nodes for GPU processing

### max_edges
- **Current Value**: `10000000`
- **Type**: u32
- **Location**: `dev_config.rs:96`
- **Priority**: Critical
- **Category**: Internal/CUDA
- **Description**: Maximum edges for GPU processing

---

## 27. INTERNAL SETTINGS - Network Internals

### pool_max_idle_per_host
- **Current Value**: `32`
- **Type**: usize
- **Location**: `dev_config.rs:102`
- **Priority**: Medium
- **Category**: Internal/Network
- **Description**: Maximum idle connections per host

### pool_idle_timeout_secs
- **Current Value**: `90`
- **Type**: u64
- **Location**: `dev_config.rs:103`
- **Priority**: Medium
- **Category**: Internal/Network
- **Description**: Idle connection timeout

### pool_connect_timeout_secs
- **Current Value**: `10`
- **Type**: u64
- **Location**: `dev_config.rs:104`
- **Priority**: High
- **Category**: Internal/Network
- **Description**: Connection timeout

### circuit_failure_threshold
- **Current Value**: `5`
- **Type**: u32
- **Location**: `dev_config.rs:107`
- **Priority**: High
- **Category**: Internal/Network
- **Description**: Failures before circuit breaker opens

### circuit_recovery_timeout_secs
- **Current Value**: `30`
- **Type**: u64
- **Location**: `dev_config.rs:108`
- **Priority**: Medium
- **Category**: Internal/Network
- **Description**: Circuit breaker recovery timeout

### circuit_half_open_max_requests
- **Current Value**: `3`
- **Type**: u32
- **Location**: `dev_config.rs:109`
- **Priority**: Medium
- **Category**: Internal/Network
- **Description**: Max requests in half-open state

### max_retry_attempts (Internal)
- **Current Value**: `3`
- **Type**: u32
- **Location**: `dev_config.rs:112`
- **Priority**: Medium
- **Category**: Internal/Network
- **Description**: Maximum retry attempts

### retry_base_delay_ms
- **Current Value**: `100`
- **Type**: u64
- **Location**: `dev_config.rs:113`
- **Priority**: Low
- **Category**: Internal/Network
- **Description**: Base delay for exponential backoff

### retry_max_delay_ms
- **Current Value**: `30000`
- **Type**: u64
- **Location**: `dev_config.rs:114`
- **Priority**: Low
- **Category**: Internal/Network
- **Description**: Maximum retry delay

### retry_exponential_base
- **Current Value**: `2.0`
- **Type**: f32
- **Location**: `dev_config.rs:115`
- **Priority**: Low
- **Category**: Internal/Network
- **Description**: Exponential backoff base

### ws_ping_interval_secs
- **Current Value**: `30`
- **Type**: u64
- **Location**: `dev_config.rs:118`
- **Priority**: Medium
- **Category**: Internal/WebSocket
- **Description**: WebSocket ping interval

### ws_pong_timeout_secs
- **Current Value**: `10`
- **Type**: u64
- **Location**: `dev_config.rs:119`
- **Priority**: High
- **Category**: Internal/WebSocket
- **Description**: WebSocket pong timeout

### ws_frame_size
- **Current Value**: `65536`
- **Type**: usize
- **Location**: `dev_config.rs:120`
- **Priority**: Medium
- **Category**: Internal/WebSocket
- **Description**: WebSocket frame size

### ws_max_pending_messages
- **Current Value**: `100`
- **Type**: usize
- **Location**: `dev_config.rs:121`
- **Priority**: Medium
- **Category**: Internal/WebSocket
- **Description**: Maximum pending messages

---

## 28. INTERNAL SETTINGS - Performance

### batch_size_nodes
- **Current Value**: `1000`
- **Type**: usize
- **Location**: `dev_config.rs:168`
- **Priority**: High
- **Category**: Internal/Performance
- **Description**: Node batch size for processing

### batch_size_edges
- **Current Value**: `5000`
- **Type**: usize
- **Location**: `dev_config.rs:169`
- **Priority**: High
- **Category**: Internal/Performance
- **Description**: Edge batch size for processing

### batch_timeout_ms
- **Current Value**: `100`
- **Type**: u64
- **Location**: `dev_config.rs:170`
- **Priority**: Medium
- **Category**: Internal/Performance
- **Description**: Batch operation timeout

### cache_ttl_secs (Internal)
- **Current Value**: `300` (5 minutes)
- **Type**: u64
- **Location**: `dev_config.rs:173`
- **Priority**: Medium
- **Category**: Internal/Performance
- **Description**: Internal cache TTL

### cache_max_entries
- **Current Value**: `10000`
- **Type**: usize
- **Location**: `dev_config.rs:174`
- **Priority**: Medium
- **Category**: Internal/Performance
- **Description**: Maximum cache entries

### cache_eviction_percentage
- **Current Value**: `0.2` (20%)
- **Type**: f32
- **Location**: `dev_config.rs:175`
- **Priority**: Low
- **Category**: Internal/Performance
- **Description**: Percentage to evict when cache full

### worker_threads
- **Current Value**: `4`
- **Type**: usize
- **Location**: `dev_config.rs:178`
- **Priority**: High
- **Category**: Internal/Performance
- **Description**: Number of worker threads

### blocking_threads
- **Current Value**: `512`
- **Type**: usize
- **Location**: `dev_config.rs:179`
- **Priority**: High
- **Category**: Internal/Performance
- **Description**: Number of blocking threads

### stack_size_mb
- **Current Value**: `2`
- **Type**: usize
- **Location**: `dev_config.rs:180`
- **Priority**: Medium
- **Category**: Internal/Performance
- **Description**: Thread stack size in MB

### gc_interval_secs
- **Current Value**: `60`
- **Type**: u64
- **Location**: `dev_config.rs:183`
- **Priority**: Low
- **Category**: Internal/Performance
- **Description**: Garbage collection interval

### memory_warning_threshold_mb
- **Current Value**: `1024`
- **Type**: usize
- **Location**: `dev_config.rs:184`
- **Priority**: Medium
- **Category**: Internal/Performance
- **Description**: Memory warning threshold

### memory_critical_threshold_mb
- **Current Value**: `2048`
- **Type**: usize
- **Location**: `dev_config.rs:185`
- **Priority**: High
- **Category**: Internal/Performance
- **Description**: Memory critical threshold

---

## 29. INTERNAL SETTINGS - Debug

### enable_cuda_debug
- **Current Value**: `false`
- **Type**: bool
- **Location**: `dev_config.rs:190`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Enable CUDA debugging output

### enable_physics_debug
- **Current Value**: `false`
- **Type**: bool
- **Location**: `dev_config.rs:191`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Enable physics debugging output

### enable_network_debug
- **Current Value**: `false`
- **Type**: bool
- **Location**: `dev_config.rs:192`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Enable network debugging output

### enable_memory_tracking
- **Current Value**: `false`
- **Type**: bool
- **Location**: `dev_config.rs:193`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Enable memory usage tracking

### enable_performance_tracking
- **Current Value**: `false`
- **Type**: bool
- **Location**: `dev_config.rs:194`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Enable performance metrics tracking

### log_slow_operations_ms
- **Current Value**: `100`
- **Type**: u64
- **Location**: `dev_config.rs:196`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Log operations slower than this

### log_memory_usage_interval_secs
- **Current Value**: `60`
- **Type**: u64
- **Location**: `dev_config.rs:197`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Memory usage logging interval

### profile_sample_rate
- **Current Value**: `0.01` (1%)
- **Type**: f32
- **Location**: `dev_config.rs:198`
- **Priority**: Low
- **Category**: Internal/Debug
- **Description**: Profiling sample rate

---

## 30. ACTOR SETTINGS - Caching

**Location:** `src/actors/optimized_settings_actor.rs`

### CACHE_SIZE (Actor Cache)
- **Current Value**: `1000`
- **Type**: usize (const)
- **Location**: `optimized_settings_actor.rs:34`
- **Priority**: Medium
- **Category**: Actor/Cache
- **Description**: LRU cache size for settings

### CACHE_TTL (Actor Cache)
- **Current Value**: `300` seconds (5 minutes)
- **Type**: Duration (const)
- **Location**: `optimized_settings_actor.rs:132`
- **Priority**: Medium
- **Category**: Actor/Cache
- **Description**: Cache entry time-to-live

### REDIS_TTL
- **Current Value**: `3600` seconds (1 hour)
- **Type**: usize (const)
- **Location**: `optimized_settings_actor.rs:133`
- **Priority**: Medium
- **Category**: Actor/Cache
- **Description**: Redis cache TTL

---

## 31. CONSTANTS - WebSocket Protocol

**Location:** `src/utils/socket_flow_constants.rs`

### NODE_SIZE
- **Current Value**: `1.0`
- **Type**: f32 (const)
- **Location**: `socket_flow_constants.rs:2`
- **Priority**: Low
- **Category**: Constants/Visual
- **Description**: Base node size in world units

### HEARTBEAT_INTERVAL (Constant)
- **Current Value**: `30` seconds
- **Type**: u64 (const)
- **Location**: `socket_flow_constants.rs:8`
- **Priority**: High
- **Category**: Constants/WebSocket
- **Description**: Matches nginx proxy_connect_timeout

### CLIENT_TIMEOUT (Constant)
- **Current Value**: `60` seconds
- **Type**: u64 (const)
- **Location**: `socket_flow_constants.rs:9`
- **Priority**: High
- **Category**: Constants/WebSocket
- **Description**: Double heartbeat interval for safety

### MAX_CLIENT_TIMEOUT
- **Current Value**: `3600` seconds (1 hour)
- **Type**: u64 (const)
- **Location**: `socket_flow_constants.rs:10`
- **Priority**: High
- **Category**: Constants/WebSocket
- **Description**: Matches nginx proxy_read_timeout

### MAX_MESSAGE_SIZE (Constant)
- **Current Value**: `104857600` (100MB)
- **Type**: usize (const)
- **Location**: `socket_flow_constants.rs:11`
- **Priority**: Critical
- **Category**: Constants/WebSocket
- **Description**: Maximum WebSocket message size

### BINARY_CHUNK_SIZE (Constant)
- **Current Value**: `65536` (64KB)
- **Type**: usize (const)
- **Location**: `socket_flow_constants.rs:12`
- **Priority**: High
- **Category**: Constants/WebSocket
- **Description**: Binary message chunk size

### NODE_POSITION_SIZE
- **Current Value**: `24` bytes
- **Type**: usize (const)
- **Location**: `socket_flow_constants.rs:19`
- **Priority**: Medium
- **Category**: Constants/Binary Protocol
- **Description**: 6 f32s (x,y,z,vx,vy,vz) * 4 bytes

### BINARY_HEADER_SIZE
- **Current Value**: `4` bytes
- **Type**: usize (const)
- **Location**: `socket_flow_constants.rs:20`
- **Priority**: Medium
- **Category**: Constants/Binary Protocol
- **Description**: 1 f32 for header

---

## 32. ENVIRONMENT VARIABLES

**Detected in:** `src/main.rs`, `src/app_state.rs`, `src/handlers/api_handler/analytics/mod.rs`

### BIND_ADDRESS (ENV)
- **Default**: `"0.0.0.0"`
- **Type**: String
- **Location**: `main.rs:612`
- **Priority**: Critical
- **Category**: Environment/Network
- **Description**: Server bind address override

### SYSTEM_NETWORK_PORT (ENV)
- **Default**: Read from settings
- **Type**: u16
- **Location**: `main.rs:613`
- **Priority**: Critical
- **Category**: Environment/Network
- **Description**: Server port override

### MCP_TCP_PORT (ENV)
- **Default**: `"9500"`
- **Type**: String
- **Location**: `app_state.rs:242`, `analytics/mod.rs:1119`
- **Priority**: High
- **Category**: Environment/MCP
- **Description**: MCP TCP server port

### MANAGEMENT_API_PORT (ENV)
- **Default**: `"9090"`
- **Type**: String
- **Location**: `app_state.rs:297`
- **Priority**: High
- **Category**: Environment/Management
- **Description**: Management API port

### REDIS_URL (ENV)
- **Default**: None (optional)
- **Type**: String
- **Location**: `optimized_settings_actor.rs:151`
- **Priority**: Medium
- **Category**: Environment/Cache
- **Description**: Redis connection URL for distributed caching

### RUV_SWARM_PORT (ENV)
- **Default**: Read from discovery
- **Type**: String
- **Location**: `multi_mcp_agent_discovery.rs:115`
- **Priority**: Low
- **Category**: Environment/MCP
- **Description**: Ruv Swarm MCP port

### DAA_PORT (ENV)
- **Default**: Read from discovery
- **Type**: String
- **Location**: `multi_mcp_agent_discovery.rs:133`
- **Priority**: Low
- **Category**: Environment/MCP
- **Description**: DAA MCP port

---

## SUMMARY STATISTICS

**Total Parameters:** 487+

### By Priority:
- **Critical**: 42 parameters
- **High**: 108 parameters
- **Medium**: 187 parameters
- **Low**: 150+ parameters

### By Category:
- **Physics**: 120+ parameters (including auto-balance and auto-pause)
- **Network/WebSocket**: 35 parameters
- **Security**: 12 parameters
- **Rendering/Visual**: 45 parameters
- **XR**: 30+ parameters
- **External Services**: 38 parameters
- **Ontology**: 24 parameters
- **Internal (DevConfig)**: 80+ parameters
- **GPU/CUDA**: 45+ parameters
- **Caching**: 15 parameters
- **Environment Variables**: 10 parameters
- **Constants**: 43+ parameters

### Parameter Exposure:
- **Client-Accessible**: 320+ parameters (via settings.yaml and API)
- **Server-Only (Internal)**: 167+ parameters (dev_config.rs, constants)

### Configuration Sources:
1. **Primary**: `data/settings.yaml` (legacy, migrating to database)
2. **Developer**: `data/dev_config.toml` (optional override)
3. **Database**: SQLite (via SettingsRepository port)
4. **Environment**: System environment variables
5. **Constants**: Hardcoded in source files
6. **Actors**: Runtime caching and optimization

---

## NOTES

1. **Database Migration**: Settings are transitioning from YAML to SQLite database. The YAML file is considered legacy.

2. **Hexagonal Architecture**: Settings use repository pattern with `SettingsRepository` port for persistence abstraction.

3. **Field Naming**: Settings support both snake_case and camelCase via serde aliases for frontend compatibility.

4. **Two Graph Configurations**: Separate physics settings for "logseq" (knowledge graph) and "visionflow" (agent graph).

5. **GPU Compute Modes**:
   - `0` = CPU only
   - `1` = GPU (CUDA)
   - `2` = Hybrid CPU/GPU

6. **Auto-Balance**: Sophisticated auto-tuning system with 40+ parameters for detecting and correcting physics instabilities.

7. **Caching Layers**:
   - Local LRU cache (1000 entries, 5 min TTL)
   - Optional Redis cache (1 hour TTL)
   - Actor-level optimizations

8. **Protected Settings**: Separate actor for user API keys and authentication tokens (not included in this audit for security).

9. **Validation**: Comprehensive validation using `validator` crate with custom rules for colors, ranges, and cross-field constraints.

10. **Performance**: Optimized settings actor uses Blake3 hashing, compression, and batch operations for high-performance access.
