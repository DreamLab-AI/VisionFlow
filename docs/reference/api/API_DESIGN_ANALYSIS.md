---
layout: default
title: "API Design Analysis & Improvement Recommendations"
parent: API
grand_parent: Reference
nav_order: 99
---

# API Design Analysis & Improvement Recommendations

**Date**: 2025-12-25
**Scope**: REST and WebSocket APIs in `/src/handlers/`
**Focus**: Developer experience, consistency, and API quality

---

## Executive Summary

### Current State
- **40+ handler modules** implementing REST and WebSocket endpoints
- **Multiple transport protocols**: WebSocket (actix-ws), QUIC/WebTransport, fastwebsockets
- **4 WebSocket variants**: socket_flow, realtime, multi_mcp, speech_socket
- **Binary protocols**: Custom 36-byte format, Postcard serialization (12 GB/s)
- **Rate limiting**: Present but inconsistent implementation
- **Error handling**: Varied approaches, no standardized error contract

### Strengths
- **Exceptional performance engineering**: QUIC achieves 50-98% latency reduction, binary protocol 80% bandwidth savings
- **Advanced transport options**: WebTransport/QUIC for modern clients, fallback to WebSocket
- **Sophisticated real-time features**: Multi-channel subscriptions, delta encoding, backpressure handling

### Critical Gaps
1. **No OpenAPI specification** - Zero machine-readable API documentation
2. **Inconsistent error responses** - Each handler uses different error formats
3. **Missing versioning strategy** - No API version headers or URL versioning
4. **Undocumented rate limits** - Rate limiting exists but no documented quotas
5. **Authentication patterns vary** - Query string tokens, headers, and Nostr events all used differently

---

## 1. API Consistency Analysis

### 1.1 Naming Conventions

**Issues Found:**
```rust
// Inconsistent endpoint naming
/api/analytics/params              // RESTful (good)
/api/bots/data                     // RESTful (good)
requestInitialData                 // camelCase WebSocket message (inconsistent)
enableRandomization                // camelCase WebSocket message (inconsistent)
subscribe_position_updates         // snake_case WebSocket message (good)
```

**Recommendation:**
```yaml
REST Endpoints:
  - Use kebab-case: /api/graph-export/status/{id}
  - Resource-oriented: /api/workspaces/{id}/graphs
  - Action as verbs: POST /api/graphs/{id}/optimize

WebSocket Messages:
  - Use snake_case consistently: request_initial_data
  - Type field required: {"type": "subscribe", "data": {...}}
  - Response mirrors request: request → response, subscribe → subscription_confirmed
```

### 1.2 Response Format Standardization

**Current Chaos:**
```rust
// socket_flow_handler.rs - ad hoc JSON
{"type": "error", "message": "...", "recoverable": true}

// realtime_websocket_handler.rs - structured message
{"type": "...", "data": {...}, "timestamp": 123, "client_id": "..."}

// REST handlers - no standard envelope
Ok(HttpResponse::InternalServerError().json(format!("Error: {}", e)))
```

**Recommended Standard Envelope:**
```typescript
// REST API Response
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  meta: {
    timestamp: string;      // ISO 8601
    request_id: string;     // For tracing
    version: string;        // API version
  }
}

interface ApiError {
  code: string;            // Machine-readable: "RATE_LIMIT_EXCEEDED"
  message: string;         // Human-readable
  details?: Record<string, any>;
  retry_after?: number;    // Seconds (for rate limits)
  documentation_url?: string;
}

// WebSocket Message
interface WsMessage<T> {
  type: string;
  data: T;
  timestamp: number;       // Unix epoch millis
  client_id?: string;
  session_id?: string;
  request_id?: string;     // Echo client's request_id
}
```

---

## 2. Error Handling Deep Dive

### 2.1 Current Error Patterns

**socket_flow_handler.rs** (Line 1413-1422):
```rust
Err(e) => {
    warn!("[WebSocket] Failed to parse text message: {}", e);
    let error_msg = serde_json::json!({
        "type": "error",
        "message": format!("Failed to parse text message: {}", e)
    });
    if let Ok(msg_str) = serde_json::to_string(&error_msg) {
        ctx.text(msg_str);
    }
}
```
**Issues**:
- No error code
- No actionable guidance
- Message exposes internal error details (security risk)
- No request correlation

**Improved Version:**
```rust
Err(e) => {
    warn!("[WebSocket] Failed to parse message: {}", e);
    let error_response = WsErrorResponse {
        type_: "error".to_string(),
        error: ApiError {
            code: "INVALID_MESSAGE_FORMAT".to_string(),
            message: "Message could not be parsed. Ensure JSON is valid.".to_string(),
            details: Some(json!({
                "expected_format": "{ \"type\": \"...\", \"data\": {...} }",
                "parsing_error": e.to_string().split(':').next().unwrap_or("Unknown")
            })),
            retry_after: None,
            documentation_url: Some("https://docs.example.com/websocket#message-format".to_string()),
        },
        timestamp: current_timestamp_ms(),
        request_id: None,
    };
    if let Ok(json) = serde_json::to_string(&error_response) {
        ctx.text(json);
    }
}
```

### 2.2 Error Code Taxonomy

**Proposed Error Codes:**
```rust
pub enum ApiErrorCode {
    // Client Errors (4xx equivalent)
    InvalidRequest,           // Malformed request
    Unauthorized,             // No auth token
    Forbidden,                // Token valid but insufficient permissions
    NotFound,                 // Resource doesn't exist
    RateLimitExceeded,        // Too many requests
    InvalidMessageFormat,     // WebSocket message parse failure

    // Server Errors (5xx equivalent)
    InternalError,            // Generic server error
    ServiceUnavailable,       // Dependency down
    Timeout,                  // Operation exceeded deadline
    CircuitBreakerOpen,       // Service degraded

    // Domain-Specific
    GraphNotFound,
    SimulationFailed,
    ExportNotReady,
}

impl ApiErrorCode {
    pub fn http_status(&self) -> u16 {
        match self {
            Self::InvalidRequest | Self::InvalidMessageFormat => 400,
            Self::Unauthorized => 401,
            Self::Forbidden => 403,
            Self::NotFound | Self::GraphNotFound => 404,
            Self::RateLimitExceeded => 429,
            _ => 500,
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimitExceeded | Self::ServiceUnavailable | Self::Timeout | Self::CircuitBreakerOpen
        )
    }
}
```

---

## 3. WebSocket Protocol Efficiency

### 3.1 Current Protocols Comparison

| Protocol | Handler | Serialization | Use Case | Bandwidth (100k nodes) |
|----------|---------|---------------|----------|----------------------|
| **Binary V2** | socket_flow | 36-byte struct | Position updates | 3.6 MB/frame |
| **Postcard** | quic_transport | Postcard | QUIC datagrams | ~2.8 MB/frame (22% better) |
| **JSON** | realtime_websocket | JSON | Events/subscriptions | 18 MB/frame (legacy) |
| **Custom** | multi_mcp | Mixed | Agent visualization | Variable |

**Analysis:**
- **socket_flow_handler.rs**: Excellent binary protocol BUT no compression negotiation
- **quic_transport_handler.rs**: Superior postcard serialization, supports delta encoding (16 bytes vs 28 bytes)
- **realtime_websocket_handler.rs**: JSON acceptable for control plane, terrible for data plane

### 3.2 Missing: Per-Message Compression

**Current (socket_flow_handler.rs line 1746-1749):**
```rust
ws::WsResponseBuilder::new(ws, &req, stream)
    .protocols(&["permessage-deflate"])  // Declared but not verified
    .start()
```

**Issue**: No feedback if compression failed to negotiate.

**Improvement:**
```rust
// 1. Negotiate compression and inform client
let builder = ws::WsResponseBuilder::new(ws, &req, stream);
let supports_compression = req.headers()
    .get("Sec-WebSocket-Extensions")
    .and_then(|h| h.to_str().ok())
    .map(|s| s.contains("permessage-deflate"))
    .unwrap_or(false);

if supports_compression {
    builder.protocols(&["permessage-deflate"])
} else {
    warn!("Client does not support WebSocket compression");
    builder
}.start()?;

// 2. Send negotiated protocol to client
let protocol_msg = json!({
    "type": "connection_protocol",
    "compression": supports_compression,
    "binary_version": 2,
    "recommended_buffer_size": 65536
});
```

### 3.3 Message Batching & Backpressure

**Excellent Pattern (socket_flow_handler.rs line 1078-1104):**
```rust
// Filters nodes with delta encoding before sending
if !filtered_nodes.is_empty() {
    let binary_data = binary_protocol::encode_node_data(&filtered_nodes);

    // Track metrics
    act.total_node_count = filtered_nodes.len();
    act.bytes_sent += binary_data.len();

    ctx.binary(binary_data);
}
```

**Missing**: Backpressure handling when client can't keep up.

**Recommendation:**
```rust
// Add client-side buffer monitoring
struct WebSocketMetrics {
    send_queue_depth: AtomicUsize,
    dropped_frames: AtomicU64,
    last_ack: Instant,
}

// Before sending
if metrics.send_queue_depth.load(Ordering::Relaxed) > 10 {
    warn!("Client {} falling behind, dropping frame", client_id);
    metrics.dropped_frames.fetch_add(1, Ordering::Relaxed);
    return;
}

// Send with monitoring
ctx.binary(binary_data);
metrics.send_queue_depth.fetch_add(1, Ordering::Relaxed);

// Client acknowledges via ping/pong
Ok(ws::Message::Pong(_)) => {
    metrics.send_queue_depth.store(0, Ordering::Relaxed);
    metrics.last_ack = Instant::now();
}
```

---

## 4. Missing Endpoints for Developer Joy

### 4.1 API Introspection

**Critical Missing Endpoints:**
```rust
// 1. GET /api/schema
// Returns OpenAPI 3.1 specification
{
  "openapi": "3.1.0",
  "info": { "version": "2.0", "title": "VisionFlow API" },
  "paths": {...}
}

// 2. GET /api/health/detailed
// Beyond simple health check
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": 123456,
  "services": {
    "neo4j": {"status": "healthy", "latency_ms": 2},
    "redis": {"status": "degraded", "latency_ms": 150},
    "claude_flow_mcp": {"status": "healthy", "active_agents": 5}
  },
  "rate_limits": {
    "websocket_updates": {"limit": 60, "window": "1m"},
    "rest_api": {"limit": 1000, "window": "1h"}
  }
}

// 3. GET /api/capabilities
// Feature detection for client
{
  "websocket": {
    "compression": true,
    "binary_protocol_version": 2,
    "max_message_size": 16777216
  },
  "quic": {
    "available": true,
    "endpoint": "0.0.0.0:4433",
    "max_streams": 100
  },
  "authentication": ["nostr", "bearer_token"],
  "features": ["delta_encoding", "backpressure", "filtering"]
}
```

### 4.2 Rate Limit Transparency

**Current (socket_flow_handler.rs line 1426-1441):**
```rust
if !WEBSOCKET_RATE_LIMITER.is_allowed(&self.client_ip) {
    let error_msg = json!({
        "type": "rate_limit_warning",
        "message": "Update rate too high, some updates may be dropped",
        "retry_after": WEBSOCKET_RATE_LIMITER.reset_time(&client_ip).as_secs()
    });
    ctx.text(msg_str);
    return;  // Silently drops message
}
```

**Issues**:
- Silently drops after warning
- No quota remaining header
- No proactive limit disclosure

**Improvement:**
```rust
// 1. Add rate limit headers to REST responses
impl Responder for RateLimitedResponse {
    fn respond_to(self, req: &HttpRequest) -> HttpResponse {
        let mut resp = HttpResponse::Ok().json(self.data);

        // Standard rate limit headers (IETF draft)
        resp.headers_mut().insert(
            HeaderName::from_static("x-ratelimit-limit"),
            HeaderValue::from(self.limit)
        );
        resp.headers_mut().insert(
            HeaderName::from_static("x-ratelimit-remaining"),
            HeaderValue::from(self.remaining)
        );
        resp.headers_mut().insert(
            HeaderName::from_static("x-ratelimit-reset"),
            HeaderValue::from(self.reset_at.timestamp())
        );

        resp
    }
}

// 2. WebSocket quota messaging
ctx.text(json!({
    "type": "rate_limit_status",
    "quota": {
        "limit": 60,
        "remaining": 42,
        "reset_at": 1735142400,
        "window": "1m"
    }
}));

// 3. Proactive warnings at 80% quota
if remaining < limit * 0.2 {
    ctx.text(json!({
        "type": "rate_limit_approaching",
        "remaining": remaining,
        "limit": limit
    }));
}
```

### 4.3 WebSocket Reconnection Support

**Missing**: Stateful reconnection with position recovery.

**Recommendation:**
```rust
// Client sends reconnection token
{
    "type": "reconnect",
    "session_token": "abc123",
    "last_frame_id": 98765
}

// Server responds with catch-up data
{
    "type": "reconnect_successful",
    "missed_frames": 3,
    "catchup_data": [...],  // Delta updates from frame 98765
    "new_session_token": "def456"
}

// Implementation
struct SessionState {
    session_id: String,
    last_frame_id: u64,
    subscriptions: HashSet<String>,
    created_at: Instant,
    ttl: Duration,  // 5 minutes
}

lazy_static! {
    static ref SESSION_CACHE: Arc<RwLock<LruCache<String, SessionState>>> =
        Arc::new(RwLock::new(LruCache::new(10000)));
}
```

---

## 5. Versioning Strategy

### 5.1 Current State: No Versioning

**Evidence:**
- No `/api/v1/` or `/api/v2/` prefixes
- No `Accept-Version` header support
- Protocol version only in binary (line 36-byte format vs postcard)

### 5.2 Recommended Multi-Layer Versioning

```rust
// 1. URL-based major versions
/api/v1/graphs          // Stable, deprecated in 6 months
/api/v2/graphs          // Current
/api/v3/graphs          // Beta

// 2. Header-based minor versions
Accept-Version: v2.1    // Request specific minor version
API-Version: v2.1       // Response indicates version used

// 3. Content negotiation for format
Accept: application/vnd.visionflow.v2+json
Accept: application/vnd.visionflow.v2+msgpack

// 4. WebSocket protocol negotiation
Sec-WebSocket-Protocol: visionflow-v2-binary, visionflow-v1-json

// Implementation
pub struct ApiVersion {
    major: u8,
    minor: u8,
    format: SerializationFormat,
}

impl FromRequest for ApiVersion {
    type Error = ActixError;

    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        // Parse from URL, header, or default to latest
        let version = req
            .match_info()
            .get("version")
            .or_else(|| req.headers().get("Accept-Version").and_then(|h| h.to_str().ok()))
            .unwrap_or("v2.0");

        // Validate and deprecation warnings
        if version == "v1.0" {
            warn!("Client using deprecated API v1.0, sunset date: 2026-06-01");
        }

        // Return parsed version
    }
}
```

---

## 6. Authentication & Authorization Patterns

### 6.1 Current Inconsistencies

**Pattern 1: Query String Tokens (socket_flow_handler.rs line 1713-1722)**
```rust
let token_from_qs = req.query_string()
    .split('&')
    .find_map(|param| {
        let parts: Vec<&str> = param.split('=').collect();
        if parts.len() == 2 && parts[0] == "token" {
            Some(parts[1].to_string())
        } else { None }
    });
```
**Issue**: Tokens in URLs are logged, cached, and leaked via Referer headers.

**Pattern 2: Nostr Events (nostr_handler.rs)**
```rust
pub async fn authenticate(
    event: web::Json<NostrAuthEvent>,
    nostr_service: web::Data<NostrService>,
) -> Result<HttpResponse, actix_web::Error>
```
**Good**: Cryptographic signatures, no token storage.

**Pattern 3: Header-based (implicit in many handlers)**
```rust
let pubkey = req.headers().get("X-Nostr-Pubkey")
```

### 6.2 Recommended Unified Authentication

```rust
// 1. Primary: Bearer tokens in Authorization header
Authorization: Bearer <jwt_token>

// 2. WebSocket: Token in Sec-WebSocket-Protocol
Sec-WebSocket-Protocol: visionflow-v2, auth-<token>

// 3. QUIC: Token in ALPN or initial control message
ControlMessage::Hello {
    client_id: "...",
    protocol_version: 2,
    capabilities: vec!["auth-bearer"],
    auth_token: Some("jwt_token"),
}

// 4. Nostr: Dedicated endpoint, returns session token
POST /api/auth/nostr
{ "event": {...} }
→ { "session_token": "jwt", "expires_in": 3600 }

// Unified middleware
pub struct AuthExtractor;

impl FromRequest for AuthExtractor {
    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        // 1. Try Authorization header
        if let Some(token) = extract_bearer_token(req) {
            return validate_jwt(token);
        }

        // 2. Try Nostr pubkey + validate recent event
        if let Some(pubkey) = req.headers().get("X-Nostr-Pubkey") {
            return validate_nostr_session(pubkey);
        }

        // 3. Fallback to anonymous (rate limited)
        Ok(AuthContext::Anonymous { ip: extract_ip(req) })
    }
}
```

---

## 7. Rate Limiting & Backpressure

### 7.1 Current Implementation Analysis

**Found in:**
- `socket_flow_handler.rs` (line 28-33): Global WebSocket rate limiter
- `validation_handler.rs`: Endpoint-specific limits
- `multi_mcp_websocket_handler.rs`: Circuit breaker pattern

**Good:**
```rust
lazy_static::lazy_static! {
    static ref WEBSOCKET_RATE_LIMITER: Arc<RateLimiter> = {
        Arc::new(RateLimiter::new(EndpointRateLimits::socket_flow_updates()))
    };
}
```

**Missing:**
- No per-user quotas (only IP-based)
- No burst allowance
- No quota persistence across restarts

### 7.2 Enhanced Rate Limiting

```rust
pub struct RateLimitConfig {
    // Leaky bucket parameters
    pub requests_per_second: u32,
    pub burst_size: u32,

    // Quotas
    pub daily_quota: Option<u64>,
    pub monthly_quota: Option<u64>,

    // Tiers
    pub tier: RateLimitTier,
}

pub enum RateLimitTier {
    Anonymous { ip: IpAddr },                  // 100 req/min
    Authenticated { user_id: String },         // 1000 req/min
    PowerUser { user_id: String },             // 10000 req/min
    Enterprise { org_id: String },             // Unlimited
}

// Distributed rate limiting via Redis
pub struct DistributedRateLimiter {
    redis: Arc<RedisPool>,
}

impl DistributedRateLimiter {
    pub async fn check_and_decrement(&self, key: &str, config: &RateLimitConfig) -> RateLimitResult {
        // Lua script for atomic check + decrement
        let script = r#"
            local current = redis.call('GET', KEYS[1])
            if current and tonumber(current) >= tonumber(ARGV[1]) then
                return redis.error_reply('RATE_LIMIT_EXCEEDED')
            end
            redis.call('INCR', KEYS[1])
            redis.call('EXPIRE', KEYS[1], ARGV[2])
            return redis.call('GET', KEYS[1])
        "#;

        self.redis.eval(script, &[key], &[config.requests_per_second, 60]).await
    }
}
```

---

## 8. OpenAPI/Documentation Quality

### 8.1 Critical Finding: Zero OpenAPI Specs

**Search Results:**
```bash
$ find src/handlers -name "*.rs" -exec grep -l "openapi\|swagger\|ApiDoc" {} \;
# No results
```

**Impact:**
- No Swagger UI
- No client SDK generation
- No contract testing
- No API changelog

### 8.2 Recommended OpenAPI Implementation

```rust
// 1. Use utoipa for automatic OpenAPI generation
use utoipa::{OpenApi, ToSchema};

#[derive(OpenApi)]
#[openapi(
    paths(
        get_graph_data,
        create_workspace,
        websocket_handler,
    ),
    components(
        schemas(GraphData, Node, Edge, ApiError)
    ),
    tags(
        (name = "graphs", description = "Graph management endpoints"),
        (name = "workspaces", description = "Workspace operations")
    ),
    info(
        title = "VisionFlow API",
        version = "2.0.0",
        description = "High-performance graph visualization and knowledge management",
        contact(
            name = "API Support",
            email = "api@visionflow.dev"
        ),
        license(
            name = "MIT"
        )
    ),
    servers(
        (url = "https://api.visionflow.dev", description = "Production"),
        (url = "http://localhost:8080", description = "Local development")
    )
)]
struct ApiDoc;

// 2. Annotate handlers
#[utoipa::path(
    get,
    path = "/api/graphs/{id}",
    params(
        ("id" = u32, Path, description = "Graph database ID")
    ),
    responses(
        (status = 200, description = "Graph found", body = GraphData),
        (status = 404, description = "Graph not found", body = ApiError),
        (status = 500, description = "Internal error", body = ApiError)
    ),
    tag = "graphs",
    security(
        ("bearer_token" = [])
    )
)]
pub async fn get_graph_data(
    path: web::Path<u32>,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, actix_web::Error> {
    // Implementation
}

// 3. Serve OpenAPI spec
pub fn configure_openapi(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/api/schema")
            .route(web::get().to(|| async {
                HttpResponse::Ok().json(ApiDoc::openapi())
            }))
    );

    // Swagger UI
    cfg.service(
        SwaggerUi::new("/api/docs/{_:.*}")
            .url("/api/schema", ApiDoc::openapi())
    );
}
```

### 8.3 WebSocket Protocol Documentation

**Current:** Markdown in `/docs/reference/api/03-websocket.md` (good)

**Enhancement:** Machine-readable AsyncAPI spec
```yaml
asyncapi: 3.0.0
info:
  title: VisionFlow WebSocket API
  version: 2.0.0
  description: Real-time graph position updates

channels:
  socket_flow:
    address: /api/ws/socket-flow
    messages:
      positionUpdate:
        $ref: '#/components/messages/PositionUpdate'
      ping:
        $ref: '#/components/messages/Ping'
    bindings:
      ws:
        method: GET
        headers:
          type: object
          properties:
            Sec-WebSocket-Protocol:
              enum: [visionflow-v2-binary, visionflow-v1-json]

components:
  messages:
    PositionUpdate:
      name: position_update
      contentType: application/octet-stream
      payload:
        type: object
        description: 36-byte binary structure
        properties:
          node_id:
            type: integer
            format: uint32
          x:
            type: number
            format: float32
          # ... remaining fields
```

---

## 9. Concrete Implementation Plan

### Phase 1: Foundation (Week 1-2)
```rust
// 1.1 Create standardized error types
// File: src/api/errors.rs
pub mod errors {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ApiError {
        pub code: ApiErrorCode,
        pub message: String,
        pub details: Option<serde_json::Value>,
        pub retry_after: Option<u64>,
        pub documentation_url: Option<String>,
    }

    impl actix_web::ResponseError for ApiError {
        fn status_code(&self) -> StatusCode {
            StatusCode::from_u16(self.code.http_status()).unwrap()
        }

        fn error_response(&self) -> HttpResponse {
            HttpResponse::build(self.status_code()).json(ApiResponse {
                success: false,
                data: None::<()>,
                error: Some(self.clone()),
                meta: ResponseMeta::new(),
            })
        }
    }
}

// 1.2 Create response envelope
// File: src/api/responses.rs
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ApiError>,
    pub meta: ResponseMeta,
}

// 1.3 Add to all handlers
impl Responder for ApiResponse<GraphData> {
    type Body = BoxBody;

    fn respond_to(self, _: &HttpRequest) -> HttpResponse<Self::Body> {
        HttpResponse::Ok().json(self)
    }
}
```

### Phase 2: OpenAPI (Week 3-4)
```toml
# Cargo.toml additions
[dependencies]
utoipa = { version = "4", features = ["actix_extras", "chrono", "uuid"] }
utoipa-swagger-ui = { version = "6", features = ["actix-web"] }
```

```rust
// Add OpenAPI to 5 core handlers first:
// - graph_state_handler
// - workspace_handler
// - socket_flow_handler (describe binary protocol)
// - realtime_websocket_handler
// - quic_transport_handler
```

### Phase 3: Versioning (Week 5)
```rust
// URL-based versioning
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .configure(deprecated_routes)
    );
    cfg.service(
        web::scope("/api/v2")
            .configure(current_routes)
    );
}

// Header-based within version
pub struct ApiVersionExtractor(ApiVersion);

impl FromRequest for ApiVersionExtractor {
    // Extract from Accept-Version header
}
```

### Phase 4: Rate Limiting (Week 6)
```rust
// Redis-backed distributed rate limiting
pub struct RateLimitMiddleware {
    redis: Arc<RedisPool>,
}

impl<S, B> Transform<S, ServiceRequest> for RateLimitMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
{
    fn new_transform(&self, service: S) -> Self::Future {
        // Check rate limit before handler
        // Add X-RateLimit-* headers to response
    }
}
```

---

## 10. Priority Ranking

### Immediate (Do This Week)
1. **Standardize error responses** - Affects all 40+ handlers, breaks inconsistency
2. **Add OpenAPI to top 10 endpoints** - graph, workspace, export, websocket
3. **Document rate limits** - Add to `/api/health/detailed` response

### High Priority (This Month)
4. **Implement API versioning** - `/api/v2/` prefix for all new endpoints
5. **Add rate limit headers** - `X-RateLimit-Limit`, `X-RateLimit-Remaining`
6. **WebSocket reconnection support** - Session tokens with 5min TTL

### Medium Priority (This Quarter)
7. **AsyncAPI for WebSocket** - Machine-readable WebSocket docs
8. **Client SDK generation** - Auto-generate TypeScript, Python, Rust clients from OpenAPI
9. **Deprecation warnings** - HTTP `Sunset` header for old endpoints

### Low Priority (Nice to Have)
10. **GraphQL endpoint** - Alternative to REST for complex queries
11. **gRPC alternative** - For high-performance internal services
12. **API playground** - Interactive API tester beyond Swagger UI

---

## Appendix A: Error Code Reference

```rust
pub const ERROR_CODES: &[(&str, &str, u16)] = &[
    // Client Errors
    ("INVALID_REQUEST", "Request validation failed", 400),
    ("UNAUTHORIZED", "Authentication required", 401),
    ("FORBIDDEN", "Insufficient permissions", 403),
    ("NOT_FOUND", "Resource not found", 404),
    ("RATE_LIMIT_EXCEEDED", "Too many requests", 429),
    ("INVALID_MESSAGE_FORMAT", "WebSocket message malformed", 400),

    // Server Errors
    ("INTERNAL_ERROR", "Unexpected server error", 500),
    ("SERVICE_UNAVAILABLE", "Dependency unavailable", 503),
    ("TIMEOUT", "Operation exceeded deadline", 504),
    ("CIRCUIT_BREAKER_OPEN", "Service degraded", 503),

    // Domain-Specific
    ("GRAPH_NOT_FOUND", "Graph ID does not exist", 404),
    ("SIMULATION_FAILED", "Physics simulation error", 500),
    ("EXPORT_NOT_READY", "Export still processing", 202),
    ("WORKSPACE_LIMIT_EXCEEDED", "Maximum workspaces reached", 403),
];
```

---

## Appendix B: WebSocket Message Catalog

```typescript
// Request Messages (Client → Server)
type ClientMessage =
  | { type: "subscribe", data: { channel: string, filters?: object } }
  | { type: "unsubscribe", data: { channel: string } }
  | { type: "ping", timestamp: number }
  | { type: "request_snapshot", graphs: string[] }
  | { type: "update_filter", filter: FilterCriteria }
  | { type: "authenticate", token: string, pubkey?: string };

// Response Messages (Server → Client)
type ServerMessage =
  | { type: "subscription_confirmed", data: { channel: string } }
  | { type: "pong", timestamp: number, server_timestamp: number }
  | { type: "position_update", data: BinaryNodeData }  // Binary
  | { type: "error", error: ApiError }
  | { type: "rate_limit_status", quota: QuotaInfo };
```

---

**End of Analysis**

This analysis provides a roadmap to transform the VisionFlow API from a high-performance but inconsistent interface into a world-class developer experience. The key is incremental adoption: start with error standardization, add OpenAPI to core endpoints, then expand versioning and advanced features.
