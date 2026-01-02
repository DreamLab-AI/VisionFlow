---
layout: default
title: "API Improvement Implementation Templates"
parent: API
grand_parent: Reference
nav_order: 99
---

# API Improvement Implementation Templates

**Purpose**: Copy-paste code templates for implementing API improvements
**Companion to**: `API_DESIGN_ANALYSIS.md`

---

## Template 1: Standardized API Response Envelope

### File: `src/api/responses.rs` (Create New)

```rust
//! Standard API response envelopes and error types

use actix_web::{HttpResponse, Responder, http::StatusCode, body::BoxBody, HttpRequest};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

// ============================================================================
// RESPONSE ENVELOPE
// ============================================================================

/// Standard API response wrapper
/// All REST endpoints should return this structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ApiError>,
    pub meta: ResponseMeta,
}

impl<T> ApiResponse<T> {
    /// Create a successful response
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            meta: ResponseMeta::new(),
        }
    }

    /// Create an error response
    pub fn error(error: ApiError) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            meta: ResponseMeta::new(),
        }
    }

    /// Create an error response with data (e.g., partial success)
    pub fn partial_error(data: T, error: ApiError) -> Self {
        Self {
            success: false,
            data: Some(data),
            error: Some(error),
            meta: ResponseMeta::new(),
        }
    }
}

impl<T: Serialize> Responder for ApiResponse<T> {
    type Body = BoxBody;

    fn respond_to(self, _: &HttpRequest) -> HttpResponse<Self::Body> {
        let status = if self.success {
            StatusCode::OK
        } else if let Some(ref error) = self.error {
            StatusCode::from_u16(error.code.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };

        HttpResponse::build(status).json(self)
    }
}

// ============================================================================
// RESPONSE METADATA
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMeta {
    pub timestamp: DateTime<Utc>,
    pub request_id: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimitInfo>,
}

impl ResponseMeta {
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            request_id: Uuid::new_v4().to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            rate_limit: None,
        }
    }

    pub fn with_rate_limit(mut self, limit: RateLimitInfo) -> Self {
        self.rate_limit = Some(limit);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub limit: u32,
    pub remaining: u32,
    pub reset_at: i64,  // Unix timestamp
}

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub code: ApiErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u64>,  // Seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
}

impl ApiError {
    pub fn new(code: ApiErrorCode, message: impl Into<String>) -> Self {
        let msg = message.into();
        let docs_url = format!("https://docs.visionflow.dev/errors#{}", code.as_str());

        Self {
            code,
            message: msg,
            details: None,
            retry_after: None,
            documentation_url: Some(docs_url),
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_retry_after(mut self, seconds: u64) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    // Common error constructors
    pub fn not_found(resource: &str) -> Self {
        Self::new(
            ApiErrorCode::NotFound,
            format!("{} not found", resource),
        )
    }

    pub fn invalid_request(reason: &str) -> Self {
        Self::new(
            ApiErrorCode::InvalidRequest,
            format!("Invalid request: {}", reason),
        )
    }

    pub fn rate_limit_exceeded(retry_after: u64) -> Self {
        Self::new(
            ApiErrorCode::RateLimitExceeded,
            "Rate limit exceeded, please slow down",
        )
        .with_retry_after(retry_after)
    }

    pub fn internal_error() -> Self {
        Self::new(
            ApiErrorCode::InternalError,
            "An unexpected error occurred",
        )
    }
}

impl actix_web::ResponseError for ApiError {
    fn status_code(&self) -> StatusCode {
        StatusCode::from_u16(self.code.http_status())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
    }

    fn error_response(&self) -> HttpResponse {
        ApiResponse::<()>::error(self.clone()).respond_to(&HttpRequest::default())
    }
}

// ============================================================================
// ERROR CODES
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApiErrorCode {
    // Client Errors (4xx)
    InvalidRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    MethodNotAllowed,
    Conflict,
    Gone,
    UnprocessableEntity,
    RateLimitExceeded,
    InvalidMessageFormat,

    // Server Errors (5xx)
    InternalError,
    ServiceUnavailable,
    Timeout,
    CircuitBreakerOpen,
    DatabaseError,

    // Domain-Specific
    GraphNotFound,
    WorkspaceNotFound,
    SimulationFailed,
    ExportNotReady,
    ImportFailed,
    WorkspaceLimitExceeded,
    InvalidFilter,
    NodeNotFound,
    EdgeNotFound,
}

impl ApiErrorCode {
    pub fn http_status(&self) -> u16 {
        match self {
            Self::InvalidRequest | Self::InvalidMessageFormat | Self::InvalidFilter => 400,
            Self::Unauthorized => 401,
            Self::Forbidden | Self::WorkspaceLimitExceeded => 403,
            Self::NotFound | Self::GraphNotFound | Self::WorkspaceNotFound
            | Self::NodeNotFound | Self::EdgeNotFound => 404,
            Self::MethodNotAllowed => 405,
            Self::Conflict => 409,
            Self::Gone => 410,
            Self::UnprocessableEntity => 422,
            Self::RateLimitExceeded => 429,
            Self::InternalError | Self::DatabaseError | Self::SimulationFailed
            | Self::ImportFailed => 500,
            Self::ServiceUnavailable | Self::CircuitBreakerOpen => 503,
            Self::Timeout => 504,
            Self::ExportNotReady => 202,  // Special case: Accepted but processing
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimitExceeded
                | Self::ServiceUnavailable
                | Self::Timeout
                | Self::CircuitBreakerOpen
        )
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::Unauthorized => "unauthorized",
            Self::Forbidden => "forbidden",
            Self::NotFound => "not_found",
            Self::RateLimitExceeded => "rate_limit_exceeded",
            Self::InternalError => "internal_error",
            Self::GraphNotFound => "graph_not_found",
            // ... add remaining mappings
            _ => "unknown_error",
        }
    }
}

// ============================================================================
// CONVERSION HELPERS
// ============================================================================

impl From<neo4rs::Error> for ApiError {
    fn from(err: neo4rs::Error) -> Self {
        log::error!("Neo4j error: {}", err);
        Self::new(ApiErrorCode::DatabaseError, "Database operation failed")
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        Self::new(ApiErrorCode::InvalidRequest, format!("JSON parsing error: {}", err))
    }
}

impl From<actix_web::error::PayloadError> for ApiError {
    fn from(_: actix_web::error::PayloadError) -> Self {
        Self::new(ApiErrorCode::InvalidRequest, "Invalid request payload")
    }
}

// ============================================================================
// WEBSOCKET MESSAGES
// ============================================================================

/// Standard WebSocket message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsMessage<T> {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: T,
    pub timestamp: u64,  // Unix epoch millis
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,  // Echo client's request_id
}

impl<T> WsMessage<T> {
    pub fn new(msg_type: impl Into<String>, data: T) -> Self {
        Self {
            msg_type: msg_type.into(),
            data,
            timestamp: current_timestamp_ms(),
            client_id: None,
            session_id: None,
            request_id: None,
        }
    }

    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }
}

/// WebSocket error message
pub type WsErrorMessage = WsMessage<ApiError>;

impl WsErrorMessage {
    pub fn from_error(error: ApiError) -> Self {
        WsMessage::new("error", error)
    }

    pub fn invalid_message(reason: &str) -> Self {
        Self::from_error(ApiError::new(
            ApiErrorCode::InvalidMessageFormat,
            format!("Invalid message: {}", reason),
        ))
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
```

---

## Template 2: Migrating Existing Handler

### Before (Inconsistent Error Handling)

```rust
// OLD CODE - socket_flow_handler.rs
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

### After (Standardized)

```rust
// NEW CODE - socket_flow_handler.rs
use crate::api::responses::{WsErrorMessage, ApiError, ApiErrorCode};

Err(e) => {
    warn!("[WebSocket] Failed to parse message: {}", e);

    let error = ApiError::new(
        ApiErrorCode::InvalidMessageFormat,
        "Message could not be parsed. Ensure JSON is valid.",
    )
    .with_details(serde_json::json!({
        "expected_format": "{ \"type\": \"...\", \"data\": {...} }",
        "hint": "Check message structure against documentation"
    }));

    let error_msg = WsErrorMessage::from_error(error)
        .with_client_id(&self.client_id)
        .with_session_id(&self.session_id);

    if let Ok(json) = serde_json::to_string(&error_msg) {
        ctx.text(json);
    }
}
```

---

## Template 3: OpenAPI Handler Annotations

### File: `src/handlers/graph_state_handler.rs` (Modified)

```rust
use utoipa::{OpenApi, ToSchema};
use crate::api::responses::{ApiResponse, ApiError};

// Add schema derives
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: HashMap<String, NodeMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Node {
    #[schema(example = 42)]
    pub id: u32,

    #[schema(example = "file-abc123")]
    pub metadata_id: String,

    #[schema(example = "Project Overview")]
    pub label: String,

    pub data: Vec3Data,

    #[schema(example = "LogseqPage")]
    pub node_type: Option<String>,
}

// Annotate handler
#[utoipa::path(
    get,
    path = "/api/graph/data",
    tags = ["graphs"],
    summary = "Retrieve complete graph data",
    description = "Returns all nodes, edges, and metadata for the current graph. \
                   This is a heavy operation; for real-time updates use WebSocket.",
    responses(
        (
            status = 200,
            description = "Graph data retrieved successfully",
            body = GraphData,
            example = json!({
                "nodes": [{"id": 1, "label": "Home", "x": 0.0, "y": 0.0, "z": 0.0}],
                "edges": [{"id": "e1", "source": 1, "target": 2, "weight": 0.8}],
                "metadata": {}
            })
        ),
        (
            status = 500,
            description = "Database error",
            body = ApiError,
            example = json!({
                "code": "DATABASE_ERROR",
                "message": "Failed to query graph database",
                "documentation_url": "https://docs.visionflow.dev/errors#database_error"
            })
        )
    ),
    security(
        ("bearer_token" = [])
    )
)]
pub async fn get_graph_data_handler(
    app_state: web::Data<AppState>,
) -> Result<ApiResponse<GraphData>, ApiError> {
    // Get data from actor
    let graph_data = app_state
        .graph_service_addr
        .send(crate::actors::messages::GetGraphData)
        .await
        .map_err(|_| ApiError::new(ApiErrorCode::ServiceUnavailable, "Graph service unavailable"))?
        .map_err(|e| ApiError::from(e))?;

    Ok(ApiResponse::success(graph_data))
}

// Generate OpenAPI spec
#[derive(OpenApi)]
#[openapi(
    paths(get_graph_data_handler),
    components(schemas(GraphData, Node, Edge, Vec3Data, ApiError)),
    tags(
        (name = "graphs", description = "Graph data operations")
    )
)]
struct GraphApiDoc;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/api/graph/data")
            .route(web::get().to(get_graph_data_handler))
    );
}
```

---

## Template 4: Rate Limit Middleware

### File: `src/middleware/rate_limit.rs` (Create New)

```rust
use actix_web::{
    dev::{Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage, HttpResponse,
};
use futures::future::{ok, Ready};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::api::responses::{ApiError, ApiErrorCode, RateLimitInfo};

pub struct RateLimitMiddleware {
    limiter: Arc<dyn RateLimiter + Send + Sync>,
}

impl RateLimitMiddleware {
    pub fn new(limiter: Arc<dyn RateLimiter + Send + Sync>) -> Self {
        Self { limiter }
    }
}

impl<S, B> Transform<S, ServiceRequest> for RateLimitMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RateLimitMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(RateLimitMiddlewareService {
            service: Arc::new(service),
            limiter: self.limiter.clone(),
        })
    }
}

pub struct RateLimitMiddlewareService<S> {
    service: Arc<S>,
    limiter: Arc<dyn RateLimiter + Send + Sync>,
}

impl<S, B> Service<ServiceRequest> for RateLimitMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(&self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let limiter = self.limiter.clone();
        let service = self.service.clone();

        Box::pin(async move {
            // Extract identifier (IP or user ID)
            let identifier = req
                .extensions()
                .get::<AuthContext>()
                .map(|ctx| ctx.identifier())
                .unwrap_or_else(|| extract_ip(&req));

            // Check rate limit
            match limiter.check(&identifier).await {
                Ok(info) => {
                    // Store for adding headers later
                    req.extensions_mut().insert(info.clone());

                    // Proceed with request
                    let mut res = service.call(req).await?;

                    // Add rate limit headers
                    res.headers_mut().insert(
                        actix_web::http::header::HeaderName::from_static("x-ratelimit-limit"),
                        actix_web::http::header::HeaderValue::from(info.limit),
                    );
                    res.headers_mut().insert(
                        actix_web::http::header::HeaderName::from_static("x-ratelimit-remaining"),
                        actix_web::http::header::HeaderValue::from(info.remaining),
                    );
                    res.headers_mut().insert(
                        actix_web::http::header::HeaderName::from_static("x-ratelimit-reset"),
                        actix_web::http::header::HeaderValue::from(info.reset_at),
                    );

                    Ok(res)
                }
                Err(retry_after) => {
                    // Rate limit exceeded
                    let error = ApiError::rate_limit_exceeded(retry_after);
                    Err(actix_web::error::ErrorTooManyRequests(error))
                }
            }
        })
    }
}

#[async_trait::async_trait]
pub trait RateLimiter {
    async fn check(&self, identifier: &str) -> Result<RateLimitInfo, u64>;
}
```

---

## Template 5: API Versioning

### File: `src/api/versioning.rs` (Create New)

```rust
use actix_web::{HttpRequest, FromRequest, dev::Payload, Error as ActixError};
use std::future::{ready, Ready};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApiVersion {
    pub major: u8,
    pub minor: u8,
}

impl ApiVersion {
    pub const V1_0: Self = Self { major: 1, minor: 0 };
    pub const V2_0: Self = Self { major: 2, minor: 0 };
    pub const V2_1: Self = Self { major: 2, minor: 1 };

    pub fn is_deprecated(&self) -> bool {
        self.major < 2
    }

    pub fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "binary_protocol_v2" => self >= &Self::V2_0,
            "quic_transport" => self >= &Self::V2_1,
            "delta_encoding" => self >= &Self::V2_0,
            _ => false,
        }
    }
}

impl FromRequest for ApiVersion {
    type Error = ActixError;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        // 1. Try URL path (/api/v2/...)
        if let Some(version) = req.match_info().get("version") {
            if let Some(parsed) = Self::parse_from_str(version) {
                return ready(Ok(parsed));
            }
        }

        // 2. Try Accept-Version header
        if let Some(header) = req.headers().get("Accept-Version") {
            if let Ok(version_str) = header.to_str() {
                if let Some(parsed) = Self::parse_from_str(version_str) {
                    return ready(Ok(parsed));
                }
            }
        }

        // 3. Default to latest stable
        ready(Ok(Self::V2_0))
    }

    fn extract(req: &HttpRequest) -> Self::Future {
        Self::from_request(req, &mut Payload::None)
    }
}

impl ApiVersion {
    fn parse_from_str(s: &str) -> Option<Self> {
        let s = s.trim_start_matches("v").trim_start_matches("V");
        let parts: Vec<&str> = s.split('.').collect();

        if parts.len() >= 2 {
            let major = parts[0].parse::<u8>().ok()?;
            let minor = parts[1].parse::<u8>().ok()?;
            Some(Self { major, minor })
        } else {
            None
        }
    }
}

impl std::fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}.{}", self.major, self.minor)
    }
}

// Configure versioned routes
pub fn configure_versioned_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .wrap(DeprecationWarning)
            .configure(v1_routes)
    );

    cfg.service(
        web::scope("/api/v2")
            .configure(v2_routes)
    );
}
```

---

## Usage Examples

### Example 1: Returning Success Response

```rust
pub async fn create_workspace(
    data: web::Json<CreateWorkspaceRequest>,
    app_state: web::Data<AppState>,
) -> Result<ApiResponse<Workspace>, ApiError> {
    let workspace = app_state
        .workspace_service
        .create(data.into_inner())
        .await
        .map_err(|e| ApiError::from(e))?;

    Ok(ApiResponse::success(workspace))
}
```

### Example 2: Returning Error

```rust
pub async fn get_graph(
    path: web::Path<u32>,
    app_state: web::Data<AppState>,
) -> Result<ApiResponse<GraphData>, ApiError> {
    let graph_id = path.into_inner();

    let graph = app_state
        .graph_repository
        .find_by_id(graph_id)
        .await?
        .ok_or_else(|| ApiError::not_found("Graph"))?;

    Ok(ApiResponse::success(graph))
}
```

### Example 3: WebSocket Error

```rust
// In StreamHandler implementation
Ok(ws::Message::Text(text)) => {
    match serde_json::from_str::<WsMessage<serde_json::Value>>(&text) {
        Ok(msg) => {
            // Handle message
        }
        Err(e) => {
            let error_msg = WsErrorMessage::invalid_message(&e.to_string())
                .with_client_id(&self.client_id);

            ctx.text(serde_json::to_string(&error_msg).unwrap());
        }
    }
}
```

---

**End of Templates**

These templates provide immediate, actionable code for improving API consistency. Copy the relevant template into your codebase and adapt to specific needs.
