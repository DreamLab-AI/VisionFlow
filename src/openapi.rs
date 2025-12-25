//! OpenAPI/Swagger Documentation
//!
//! Provides automatic API documentation using utoipa.
//! Access Swagger UI at /swagger-ui/

use utoipa::OpenApi;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// VisionFlow API - WebXR Graph Visualization Server
///
/// GPU-accelerated knowledge graph visualization with:
/// - Real-time physics simulation
/// - QUIC/WebTransport for ultra-low latency
/// - Neo4j graph database backend
/// - Ontology reasoning and semantic analysis
#[derive(OpenApi)]
#[openapi(
    info(
        title = "VisionFlow API",
        version = "1.0.0",
        description = r#"
# VisionFlow API

WebXR Graph Visualization Server with GPU-accelerated physics, QUIC transport, and semantic reasoning.

## Features
- **Real-time Physics**: GPU-accelerated force-directed graph layout
- **Ultra-low Latency**: QUIC/WebTransport with 0-RTT connections
- **Semantic Reasoning**: OWL ontology integration with Whelk reasoner
- **High Throughput**: Postcard serialization (12 GB/s vs 2 GB/s JSON)

## Authentication
Most endpoints require authentication via:
- `X-API-Key` header for management operations
- WebSocket handshake for real-time connections

## WebSocket Endpoints
- `/wss` - Main graph sync (binary protocol)
- `/ws/speech` - Speech-to-text streaming
- `/ws/mcp-relay` - MCP protocol relay

## Rate Limits
- Export endpoints: 10 requests per minute per IP
- Query endpoints: 100 requests per minute per IP
"#,
        contact(
            name = "VisionFlow Team",
            url = "https://github.com/visionflow"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    ),
    servers(
        (url = "/api", description = "Main API endpoint"),
        (url = "/", description = "Root (WebSocket endpoints)")
    ),
    tags(
        (name = "graph", description = "Graph data operations - CRUD for nodes and edges"),
        (name = "physics", description = "Physics simulation control - start/stop/configure"),
        (name = "settings", description = "User and system settings management"),
        (name = "health", description = "Health checks and readiness probes"),
        (name = "ontology", description = "OWL ontology reasoning and class hierarchy"),
        (name = "semantic", description = "Semantic search and intelligent pathfinding"),
        (name = "export", description = "Graph export in JSON, GraphML, GEXF, CSV formats"),
        (name = "workspace", description = "Workspace and graph state management"),
        (name = "analytics", description = "Graph analytics, clustering, and community detection"),
        (name = "websocket", description = "Real-time WebSocket connections")
    ),
    components(
        schemas(
            ErrorResponse,
            HealthResponse,
            GraphResponse,
            NodeResponse,
            EdgeResponse,
            PhysicsState,
            PhysicsSettings,
            ExportFormat,
            ExportRequest,
            AddNodeRequest,
            UpdateNodeRequest,
            AddEdgeRequest,
            SearchRequest,
            PathfindingRequest,
            PathfindingResponse,
        )
    )
)]
pub struct ApiDoc;

// ============================================================================
// SCHEMA TYPES FOR DOCUMENTATION
// ============================================================================

/// API Error Response
#[derive(Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// HTTP status code
    pub status: u16,
    /// Optional error details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
    /// Request ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Health check response
#[derive(Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    /// Service status: "healthy", "degraded", or "unhealthy"
    pub status: String,
    /// Service version
    pub version: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Neo4j connection status
    pub neo4j_connected: bool,
    /// GPU available and initialized
    pub gpu_available: bool,
    /// Number of active WebSocket connections
    pub active_connections: u32,
    /// Current memory usage in MB
    pub memory_mb: u64,
}

/// Graph data response
#[derive(Serialize, Deserialize, ToSchema)]
pub struct GraphResponse {
    /// List of nodes
    pub nodes: Vec<NodeResponse>,
    /// List of edges
    pub edges: Vec<EdgeResponse>,
    /// Total node count
    pub node_count: usize,
    /// Total edge count
    pub edge_count: usize,
    /// Graph metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Node data
#[derive(Serialize, Deserialize, ToSchema)]
pub struct NodeResponse {
    /// Unique node ID
    pub id: u32,
    /// Node label/name
    pub label: String,
    /// X position in 3D space
    pub x: f32,
    /// Y position in 3D space
    pub y: f32,
    /// Z position in 3D space
    pub z: f32,
    /// Node size (default: 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    /// Node color (hex format, e.g., "#FF5733")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Node type/category
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Edge data
#[derive(Serialize, Deserialize, ToSchema)]
pub struct EdgeResponse {
    /// Unique edge ID
    pub id: String,
    /// Source node ID
    pub source: u32,
    /// Target node ID
    pub target: u32,
    /// Edge weight (affects spring force in physics)
    pub weight: f32,
    /// Edge type/relationship type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    /// Edge label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Physics simulation state
#[derive(Serialize, Deserialize, ToSchema)]
pub struct PhysicsState {
    /// Is simulation currently running
    pub running: bool,
    /// Current iteration count
    pub iteration: u64,
    /// Total kinetic energy in the system
    pub kinetic_energy: f64,
    /// Is simulation stable (energy below threshold)
    pub is_stable: bool,
    /// Frames per second
    pub fps: f32,
    /// GPU utilization percentage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_utilization: Option<f32>,
}

/// Physics simulation settings
#[derive(Serialize, Deserialize, ToSchema)]
pub struct PhysicsSettings {
    /// Enable physics simulation
    pub enabled: bool,
    /// Repulsion constant (higher = nodes push apart more)
    pub repel_k: f32,
    /// Spring constant (higher = edges pull nodes together more)
    pub spring_k: f32,
    /// Velocity damping (0-1, higher = more friction)
    pub damping: f32,
    /// Maximum velocity cap
    pub max_velocity: f32,
    /// Time step per iteration
    pub dt: f32,
    /// Target iterations per frame
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations_per_frame: Option<u32>,
}

/// Graph export format
#[derive(Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// JSON format (native)
    Json,
    /// GraphML XML format
    GraphML,
    /// GEXF format (Gephi)
    Gexf,
    /// CSV format (separate files for nodes and edges)
    Csv,
}

/// Export request parameters
#[derive(Serialize, Deserialize, ToSchema)]
pub struct ExportRequest {
    /// Export format
    pub format: ExportFormat,
    /// Include node metadata
    #[serde(default = "default_true")]
    pub include_metadata: bool,
    /// Include physics state (positions, velocities)
    #[serde(default = "default_true")]
    pub include_physics: bool,
    /// Filter by node type (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type_filter: Option<Vec<String>>,
}

fn default_true() -> bool { true }

/// Request to add a new node
#[derive(Serialize, Deserialize, ToSchema)]
pub struct AddNodeRequest {
    /// Node label (required)
    pub label: String,
    /// Initial X position (optional, random if not specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f32>,
    /// Initial Y position
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f32>,
    /// Initial Z position
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f32>,
    /// Node size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    /// Node color (hex)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Node type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Request to update a node
#[derive(Serialize, Deserialize, ToSchema)]
pub struct UpdateNodeRequest {
    /// New label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// New position
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f32>,
    /// New size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    /// New color
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// New type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
}

/// Request to add a new edge
#[derive(Serialize, Deserialize, ToSchema)]
pub struct AddEdgeRequest {
    /// Source node ID
    pub source: u32,
    /// Target node ID
    pub target: u32,
    /// Edge weight (default: 1.0)
    #[serde(default = "default_weight")]
    pub weight: f32,
    /// Edge type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    /// Edge label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

fn default_weight() -> f32 { 1.0 }

/// Semantic search request
#[derive(Serialize, Deserialize, ToSchema)]
pub struct SearchRequest {
    /// Search query string
    pub query: String,
    /// Maximum results to return
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Filter by node type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    /// Include semantic expansion
    #[serde(default)]
    pub semantic_expansion: bool,
}

fn default_limit() -> usize { 10 }

/// Pathfinding request
#[derive(Serialize, Deserialize, ToSchema)]
pub struct PathfindingRequest {
    /// Source node ID
    pub source_id: u32,
    /// Target node ID
    pub target_id: u32,
    /// Maximum hops (default: 10)
    #[serde(default = "default_max_hops")]
    pub max_hops: usize,
    /// Algorithm: "dijkstra", "astar", "semantic"
    #[serde(default = "default_algorithm")]
    pub algorithm: String,
    /// Consider edge weights
    #[serde(default = "default_true")]
    pub weighted: bool,
}

fn default_max_hops() -> usize { 10 }
fn default_algorithm() -> String { "dijkstra".to_string() }

/// Pathfinding response
#[derive(Serialize, Deserialize, ToSchema)]
pub struct PathfindingResponse {
    /// Path found
    pub found: bool,
    /// Ordered list of node IDs in path
    pub path: Vec<u32>,
    /// Total path cost/distance
    pub cost: f64,
    /// Number of hops
    pub hops: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}
