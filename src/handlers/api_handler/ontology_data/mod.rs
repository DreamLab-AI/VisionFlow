//! Ontology Data Exposure API
//!
//! REST endpoints and WebSocket handlers for exposing ontology data to clients.
//! Provides domain listings, class hierarchies, property schemas, entity relationships,
//! advanced query interface, real-time updates, and graph visualization integration.
//!
//! ## Endpoints
//! - GET /api/ontology/domains - List all ETSI domains
//! - GET /api/ontology/classes - List ontology classes with filters
//! - GET /api/ontology/properties - List properties with schemas
//! - GET /api/ontology/entities/:id - Get specific entity with relationships
//! - POST /api/ontology/query - Advanced query interface
//! - WebSocket /api/ontology/stream - Real-time ontology updates
//!
//! ## Features
//! - SQLite-based caching for performance
//! - Integration with graph visualization
//! - Real-time WebSocket updates
//! - Comprehensive error handling and validation

use actix::prelude::*;
use actix_web::{web, HttpRequest, HttpResponse, Responder, Error as ActixError};
use actix_web_actors::ws;
use chrono::{DateTime, Utc};
use log::{info, debug, error, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration as StdDuration;
use uuid::Uuid;

mod db;
mod cache;
mod query;

use crate::AppState;
use crate::actors::messages::{
    GetOntologyHealth, OntologyHealth, GetCachedOntologies, CachedOntologyInfo,
    ValidateOntology, ValidationMode, GetOntologyReport
};
use crate::services::owl_validator::PropertyGraph;

// Re-export submodules
pub use db::OntologyDatabase;
pub use cache::OntologyCache;
pub use query::QueryEngine;

// ============================================================================
// REQUEST/RESPONSE DTOs
// ============================================================================

/// Request to list ETSI domains
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListDomainsRequest {
    /// Filter by domain name pattern
    pub filter: Option<String>,
    /// Include domain statistics
    pub include_stats: Option<bool>,
}

/// ETSI domain information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DomainInfo {
    /// Domain identifier (e.g., "etsi-nfv", "etsi-mec")
    pub id: String,
    /// Human-readable domain name
    pub name: String,
    /// Domain description
    pub description: String,
    /// Number of classes in domain
    pub class_count: u32,
    /// Number of properties in domain
    pub property_count: u32,
    /// Domain namespace URI
    pub namespace: String,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// Response for domain listing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DomainsResponse {
    pub domains: Vec<DomainInfo>,
    pub total_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Request to list ontology classes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListClassesRequest {
    /// Filter by domain
    pub domain: Option<String>,
    /// Filter by class name pattern
    pub filter: Option<String>,
    /// Include subclasses
    pub include_subclasses: Option<bool>,
    /// Include properties
    pub include_properties: Option<bool>,
    /// Pagination offset
    pub offset: Option<u32>,
    /// Pagination limit
    pub limit: Option<u32>,
}

/// Ontology class information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassInfo {
    /// Class identifier
    pub id: String,
    /// Class name
    pub name: String,
    /// Class description
    pub description: Option<String>,
    /// Parent class IDs
    pub parent_classes: Vec<String>,
    /// Child class IDs
    pub child_classes: Vec<String>,
    /// Domain this class belongs to
    pub domain: String,
    /// Class properties
    pub properties: Vec<PropertyInfo>,
    /// Number of instances in graph
    pub instance_count: u32,
    /// Class namespace URI
    pub namespace: String,
}

/// Response for class listing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassesResponse {
    pub classes: Vec<ClassInfo>,
    pub total_count: usize,
    pub offset: u32,
    pub limit: u32,
    pub timestamp: DateTime<Utc>,
}

/// Request to list ontology properties
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListPropertiesRequest {
    /// Filter by domain
    pub domain: Option<String>,
    /// Filter by property name pattern
    pub filter: Option<String>,
    /// Filter by property type (object_property, data_property, annotation_property)
    pub property_type: Option<String>,
    /// Include domain/range constraints
    pub include_constraints: Option<bool>,
    /// Pagination offset
    pub offset: Option<u32>,
    /// Pagination limit
    pub limit: Option<u32>,
}

/// Ontology property information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PropertyInfo {
    /// Property identifier
    pub id: String,
    /// Property name
    pub name: String,
    /// Property description
    pub description: Option<String>,
    /// Property type (object_property, data_property, annotation_property)
    pub property_type: String,
    /// Domain constraint (classes this property applies to)
    pub domain_classes: Vec<String>,
    /// Range constraint (target classes or data types)
    pub range_classes: Vec<String>,
    /// Whether property is functional
    pub is_functional: bool,
    /// Whether property is inverse functional
    pub is_inverse_functional: bool,
    /// Whether property is transitive
    pub is_transitive: bool,
    /// Whether property is symmetric
    pub is_symmetric: bool,
    /// Cardinality constraints
    pub cardinality: Option<CardinalityConstraint>,
    /// Domain this property belongs to
    pub domain: String,
}

/// Cardinality constraint for properties
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CardinalityConstraint {
    /// Minimum cardinality
    pub min: Option<u32>,
    /// Maximum cardinality
    pub max: Option<u32>,
    /// Exact cardinality
    pub exact: Option<u32>,
}

/// Response for property listing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PropertiesResponse {
    pub properties: Vec<PropertyInfo>,
    pub total_count: usize,
    pub offset: u32,
    pub limit: u32,
    pub timestamp: DateTime<Utc>,
}

/// Request to get entity details
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GetEntityRequest {
    /// Include incoming relationships
    pub include_incoming: Option<bool>,
    /// Include outgoing relationships
    pub include_outgoing: Option<bool>,
    /// Include inferred relationships
    pub include_inferred: Option<bool>,
    /// Maximum relationship depth
    pub max_depth: Option<u32>,
}

/// Entity information with relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityInfo {
    /// Entity identifier
    pub id: String,
    /// Entity label
    pub label: String,
    /// Entity type (class)
    pub entity_type: String,
    /// Entity properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Incoming relationships
    pub incoming_relationships: Vec<RelationshipInfo>,
    /// Outgoing relationships
    pub outgoing_relationships: Vec<RelationshipInfo>,
    /// Inferred relationships
    pub inferred_relationships: Vec<RelationshipInfo>,
    /// Related entities (neighbors)
    pub related_entities: Vec<String>,
    /// Domain this entity belongs to
    pub domain: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// Relationship information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipInfo {
    /// Relationship identifier
    pub id: String,
    /// Source entity ID
    pub source_id: String,
    /// Target entity ID
    pub target_id: String,
    /// Relationship type (property)
    pub relationship_type: String,
    /// Relationship properties
    pub properties: HashMap<String, serde_json::Value>,
    /// Whether this is an inferred relationship
    pub is_inferred: bool,
    /// Inference confidence (0.0 - 1.0)
    pub confidence: Option<f32>,
}

/// Advanced query request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryRequest {
    /// SPARQL-like query string
    pub query: String,
    /// Query parameters
    pub parameters: Option<HashMap<String, serde_json::Value>>,
    /// Maximum results
    pub limit: Option<u32>,
    /// Result offset
    pub offset: Option<u32>,
    /// Include query execution plan
    pub explain: Option<bool>,
    /// Query timeout in seconds
    pub timeout_seconds: Option<u32>,
}

/// Query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryResult {
    /// Query execution ID
    pub query_id: String,
    /// Result rows
    pub results: Vec<HashMap<String, serde_json::Value>>,
    /// Total result count
    pub total_count: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Query execution plan (if requested)
    pub execution_plan: Option<String>,
    /// Whether query was served from cache
    pub from_cache: bool,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Graph visualization nodes and edges
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphVisualizationData {
    /// Graph nodes
    pub nodes: Vec<GraphNode>,
    /// Graph edges
    pub edges: Vec<GraphEdge>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Graph node for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub domain: String,
    pub size: f32,
    pub color: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Graph edge for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub edge_type: String,
    pub label: String,
    pub is_inferred: bool,
    pub weight: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// WebSocket update message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum OntologyUpdate {
    /// Ontology loaded
    OntologyLoaded {
        ontology_id: String,
        timestamp: DateTime<Utc>,
    },
    /// Validation started
    ValidationStarted {
        job_id: String,
        timestamp: DateTime<Utc>,
    },
    /// Validation progress
    ValidationProgress {
        job_id: String,
        progress: f32,
        current_step: String,
        timestamp: DateTime<Utc>,
    },
    /// Validation completed
    ValidationCompleted {
        job_id: String,
        report_id: String,
        violations_count: u32,
        timestamp: DateTime<Utc>,
    },
    /// Validation failed
    ValidationFailed {
        job_id: String,
        error: String,
        timestamp: DateTime<Utc>,
    },
    /// Entity added
    EntityAdded {
        entity_id: String,
        entity_type: String,
        timestamp: DateTime<Utc>,
    },
    /// Entity updated
    EntityUpdated {
        entity_id: String,
        changes: HashMap<String, serde_json::Value>,
        timestamp: DateTime<Utc>,
    },
    /// Entity removed
    EntityRemoved {
        entity_id: String,
        timestamp: DateTime<Utc>,
    },
    /// Relationship added
    RelationshipAdded {
        relationship_id: String,
        source_id: String,
        target_id: String,
        relationship_type: String,
        timestamp: DateTime<Utc>,
    },
    /// Cache cleared
    CacheCleared {
        timestamp: DateTime<Utc>,
    },
    /// Health status update
    HealthUpdate {
        health: OntologyHealthDto,
        timestamp: DateTime<Utc>,
    },
}

/// DTO for ontology health
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OntologyHealthDto {
    pub loaded_ontologies: u32,
    pub cached_reports: u32,
    pub validation_queue_size: u32,
    pub last_validation: Option<DateTime<Utc>>,
    pub cache_hit_rate: f32,
    pub avg_validation_time_ms: f32,
    pub active_jobs: u32,
    pub memory_usage_mb: f32,
}

impl From<OntologyHealth> for OntologyHealthDto {
    fn from(health: OntologyHealth) -> Self {
        OntologyHealthDto {
            loaded_ontologies: health.loaded_ontologies,
            cached_reports: health.cached_reports,
            validation_queue_size: health.validation_queue_size,
            last_validation: health.last_validation,
            cache_hit_rate: health.cache_hit_rate,
            avg_validation_time_ms: health.avg_validation_time_ms,
            active_jobs: health.active_jobs,
            memory_usage_mb: health.memory_usage_mb,
        }
    }
}

/// Error response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
    pub timestamp: DateTime<Utc>,
    pub trace_id: String,
}

impl ErrorResponse {
    pub fn new(error: &str, code: &str) -> Self {
        Self {
            error: error.to_string(),
            code: code.to_string(),
            details: None,
            timestamp: Utc::now(),
            trace_id: Uuid::new_v4().to_string(),
        }
    }

    pub fn with_details(mut self, details: HashMap<String, serde_json::Value>) -> Self {
        self.details = Some(details);
        self
    }
}

// ============================================================================
// REST ENDPOINTS
// ============================================================================

/// GET /api/ontology/domains - List all ETSI domains
pub async fn list_domains(
    state: web::Data<AppState>,
    req: web::Query<ListDomainsRequest>,
) -> impl Responder {
    info!("Listing ontology domains with filter: {:?}", req.filter);

    let db = match OntologyDatabase::new() {
        Ok(db) => db,
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            let error_response = ErrorResponse::new("Database initialization failed", "DB_ERROR");
            return HttpResponse::InternalServerError().json(error_response);
        }
    };

    match db.list_domains(req.filter.as_deref(), req.include_stats.unwrap_or(false)) {
        Ok(domains) => {
            let response = DomainsResponse {
                total_count: domains.len(),
                domains,
                timestamp: Utc::now(),
            };
            HttpResponse::Ok().json(response)
        }
        Err(e) => {
            error!("Failed to list domains: {}", e);
            let error_response = ErrorResponse::new(&e, "DOMAIN_LIST_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// GET /api/ontology/classes - List ontology classes with filters
pub async fn list_classes(
    state: web::Data<AppState>,
    req: web::Query<ListClassesRequest>,
) -> impl Responder {
    info!("Listing ontology classes: domain={:?}, filter={:?}", req.domain, req.filter);

    let db = match OntologyDatabase::new() {
        Ok(db) => db,
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            let error_response = ErrorResponse::new("Database initialization failed", "DB_ERROR");
            return HttpResponse::InternalServerError().json(error_response);
        }
    };

    let offset = req.offset.unwrap_or(0);
    let limit = req.limit.unwrap_or(50).min(500); // Cap at 500

    match db.list_classes(
        req.domain.as_deref(),
        req.filter.as_deref(),
        req.include_subclasses.unwrap_or(false),
        req.include_properties.unwrap_or(false),
        offset,
        limit,
    ) {
        Ok((classes, total_count)) => {
            let response = ClassesResponse {
                classes,
                total_count,
                offset,
                limit,
                timestamp: Utc::now(),
            };
            HttpResponse::Ok().json(response)
        }
        Err(e) => {
            error!("Failed to list classes: {}", e);
            let error_response = ErrorResponse::new(&e, "CLASS_LIST_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// GET /api/ontology/properties - List properties with schemas
pub async fn list_properties(
    state: web::Data<AppState>,
    req: web::Query<ListPropertiesRequest>,
) -> impl Responder {
    info!("Listing ontology properties: domain={:?}, type={:?}", req.domain, req.property_type);

    let db = match OntologyDatabase::new() {
        Ok(db) => db,
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            let error_response = ErrorResponse::new("Database initialization failed", "DB_ERROR");
            return HttpResponse::InternalServerError().json(error_response);
        }
    };

    let offset = req.offset.unwrap_or(0);
    let limit = req.limit.unwrap_or(50).min(500); // Cap at 500

    match db.list_properties(
        req.domain.as_deref(),
        req.filter.as_deref(),
        req.property_type.as_deref(),
        req.include_constraints.unwrap_or(false),
        offset,
        limit,
    ) {
        Ok((properties, total_count)) => {
            let response = PropertiesResponse {
                properties,
                total_count,
                offset,
                limit,
                timestamp: Utc::now(),
            };
            HttpResponse::Ok().json(response)
        }
        Err(e) => {
            error!("Failed to list properties: {}", e);
            let error_response = ErrorResponse::new(&e, "PROPERTY_LIST_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// GET /api/ontology/entities/:id - Get specific entity with relationships
pub async fn get_entity(
    state: web::Data<AppState>,
    path: web::Path<String>,
    req: web::Query<GetEntityRequest>,
) -> impl Responder {
    let entity_id = path.into_inner();
    info!("Getting entity: {}", entity_id);

    let db = match OntologyDatabase::new() {
        Ok(db) => db,
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            let error_response = ErrorResponse::new("Database initialization failed", "DB_ERROR");
            return HttpResponse::InternalServerError().json(error_response);
        }
    };

    match db.get_entity(
        &entity_id,
        req.include_incoming.unwrap_or(true),
        req.include_outgoing.unwrap_or(true),
        req.include_inferred.unwrap_or(false),
        req.max_depth.unwrap_or(1),
    ) {
        Ok(Some(entity)) => HttpResponse::Ok().json(entity),
        Ok(None) => {
            warn!("Entity not found: {}", entity_id);
            let error_response = ErrorResponse::new("Entity not found", "ENTITY_NOT_FOUND");
            HttpResponse::NotFound().json(error_response)
        }
        Err(e) => {
            error!("Failed to get entity: {}", e);
            let error_response = ErrorResponse::new(&e, "ENTITY_RETRIEVAL_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// POST /api/ontology/query - Advanced query interface
pub async fn query_ontology(
    state: web::Data<AppState>,
    req: web::Json<QueryRequest>,
) -> impl Responder {
    info!("Executing ontology query");
    debug!("Query: {}", req.query);

    let start_time = std::time::Instant::now();
    let query_id = Uuid::new_v4().to_string();

    // Check cache first
    let cache = OntologyCache::new();
    let cache_key = format!("query:{}:{:?}", req.query, req.parameters);

    if let Some(cached_result) = cache.get_query_result(&cache_key) {
        info!("Serving query from cache");
        return HttpResponse::Ok().json(cached_result);
    }

    let db = match OntologyDatabase::new() {
        Ok(db) => db,
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            let error_response = ErrorResponse::new("Database initialization failed", "DB_ERROR");
            return HttpResponse::InternalServerError().json(error_response);
        }
    };

    let query_engine = QueryEngine::new(db);

    match query_engine.execute_query(
        &req.query,
        req.parameters.clone(),
        req.limit.unwrap_or(100),
        req.offset.unwrap_or(0),
        req.timeout_seconds.unwrap_or(30),
    ) {
        Ok(results) => {
            let execution_time_ms = start_time.elapsed().as_millis() as u64;

            let result = QueryResult {
                query_id,
                results: results.clone(),
                total_count: results.len(),
                execution_time_ms,
                execution_plan: if req.explain.unwrap_or(false) {
                    Some("Query execution plan not yet implemented".to_string())
                } else {
                    None
                },
                from_cache: false,
                timestamp: Utc::now(),
            };

            // Cache the result
            cache.set_query_result(&cache_key, &result);

            HttpResponse::Ok().json(result)
        }
        Err(e) => {
            error!("Query execution failed: {}", e);
            let error_response = ErrorResponse::new(&e, "QUERY_FAILED");
            HttpResponse::BadRequest().json(error_response)
        }
    }
}

/// GET /api/ontology/graph - Get graph visualization data
pub async fn get_graph_visualization(
    state: web::Data<AppState>,
    query: web::Query<HashMap<String, String>>,
) -> impl Responder {
    info!("Getting graph visualization data");

    let domain_filter = query.get("domain").map(|s| s.as_str());
    let max_nodes = query.get("max_nodes")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(500)
        .min(2000); // Cap at 2000 nodes

    let db = match OntologyDatabase::new() {
        Ok(db) => db,
        Err(e) => {
            error!("Failed to initialize database: {}", e);
            let error_response = ErrorResponse::new("Database initialization failed", "DB_ERROR");
            return HttpResponse::InternalServerError().json(error_response);
        }
    };

    match db.get_graph_visualization(domain_filter, max_nodes) {
        Ok(graph_data) => HttpResponse::Ok().json(graph_data),
        Err(e) => {
            error!("Failed to get graph visualization: {}", e);
            let error_response = ErrorResponse::new(&e, "GRAPH_VIZ_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

// ============================================================================
// WEBSOCKET IMPLEMENTATION
// ============================================================================

/// WebSocket actor for real-time ontology updates
pub struct OntologyStreamSocket {
    /// Client ID for tracking
    client_id: String,
    /// Subscription filters
    filters: HashMap<String, String>,
    /// Last heartbeat
    last_heartbeat: std::time::Instant,
}

impl OntologyStreamSocket {
    pub fn new(client_id: String) -> Self {
        Self {
            client_id,
            filters: HashMap::new(),
            last_heartbeat: std::time::Instant::now(),
        }
    }

    fn send_update(&self, ctx: &mut ws::WebsocketContext<Self>, update: OntologyUpdate) {
        match serde_json::to_string(&update) {
            Ok(json) => ctx.text(json),
            Err(e) => error!("Failed to serialize update: {}", e),
        }
    }

    fn heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(StdDuration::from_secs(30), |act, ctx| {
            if std::time::Instant::now().duration_since(act.last_heartbeat) > StdDuration::from_secs(90) {
                warn!("Client {} heartbeat timeout", act.client_id);
                ctx.stop();
            } else {
                ctx.ping(b"");
            }
        });
    }
}

impl Actor for OntologyStreamSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket connection started for client: {}", self.client_id);

        // Start heartbeat
        self.heartbeat(ctx);

        // Send initial connection confirmation
        let msg = serde_json::json!({
            "type": "connected",
            "clientId": self.client_id,
            "timestamp": Utc::now()
        });
        ctx.text(msg.to_string());
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("WebSocket connection stopped for client: {}", self.client_id);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for OntologyStreamSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                debug!("Received WebSocket message from {}: {}", self.client_id, text);

                // Parse client commands
                if let Ok(command) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(cmd_type) = command.get("type").and_then(|v| v.as_str()) {
                        match cmd_type {
                            "subscribe" => {
                                if let Some(filter) = command.get("filter") {
                                    info!("Client {} subscribing with filter: {:?}", self.client_id, filter);
                                }
                            }
                            "unsubscribe" => {
                                info!("Client {} unsubscribing", self.client_id);
                                self.filters.clear();
                            }
                            _ => {
                                warn!("Unknown command type: {}", cmd_type);
                            }
                        }
                    }
                }
            }
            Ok(ws::Message::Ping(msg)) => {
                self.last_heartbeat = std::time::Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_heartbeat = std::time::Instant::now();
            }
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket close received from {}: {:?}", self.client_id, reason);
                ctx.close(reason);
            }
            _ => {}
        }
    }
}

/// WebSocket upgrade handler
pub async fn ontology_stream(
    req: HttpRequest,
    stream: web::Payload,
    query: web::Query<HashMap<String, String>>,
) -> Result<HttpResponse, ActixError> {
    info!("WebSocket upgrade request for ontology stream");

    let client_id = query.get("client_id")
        .cloned()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let websocket = OntologyStreamSocket::new(client_id);

    ws::start(websocket, &req, stream)
}

// ============================================================================
// ROUTE CONFIGURATION
// ============================================================================

/// Configure ontology data API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/ontology")
            // Domain endpoints
            .route("/domains", web::get().to(list_domains))
            // Class endpoints
            .route("/classes", web::get().to(list_classes))
            // Property endpoints
            .route("/properties", web::get().to(list_properties))
            // Entity endpoints
            .route("/entities/{id}", web::get().to(get_entity))
            // Query endpoint
            .route("/query", web::post().to(query_ontology))
            // Graph visualization endpoint
            .route("/graph", web::get().to(get_graph_visualization))
            // WebSocket stream
            .route("/stream", web::get().to(ontology_stream))
    );
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_info_serialization() {
        let domain = DomainInfo {
            id: "etsi-nfv".to_string(),
            name: "ETSI NFV".to_string(),
            description: "Network Functions Virtualization".to_string(),
            class_count: 150,
            property_count: 300,
            namespace: "http://example.org/etsi-nfv#".to_string(),
            updated_at: Utc::now(),
        };

        let json = serde_json::to_value(&domain).unwrap();
        assert!(json.get("id").is_some());
        assert!(json.get("classCount").is_some());
    }

    #[test]
    fn test_query_request_deserialization() {
        let json = r#"{
            "query": "SELECT ?s WHERE { ?s a ?type }",
            "limit": 100,
            "explain": true
        }"#;

        let req: QueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "SELECT ?s WHERE { ?s a ?type }");
        assert_eq!(req.limit, Some(100));
        assert_eq!(req.explain, Some(true));
    }
}
