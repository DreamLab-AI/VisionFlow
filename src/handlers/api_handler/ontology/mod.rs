//! Ontology REST and WebSocket API endpoints
//!
//! This module provides comprehensive API endpoints for ontology operations including:
//! - Loading ontology axioms from files/URLs
//! - Updating mapping configurations
//! - Running validation with different modes
//! - Real-time WebSocket updates for validation progress
//! - Applying inferences to the graph
//! - System health monitoring and cache management

use actix::Addr;
use actix_web::{web, HttpRequest, HttpResponse, Responder, Error as ActixError};
use actix_web_actors::ws;
use chrono::{DateTime, Utc};
use log::{info, debug, error, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration as StdDuration;
use uuid::Uuid;

use crate::AppState;
use crate::actors::messages::{
    LoadOntologyAxioms, UpdateOntologyMapping, ValidateOntology, ValidationMode,
    ApplyInferences, GetOntologyReport, GetOntologyHealth, ClearOntologyCaches,
    OntologyHealth
};
use crate::actors::ontology_actor::OntologyActor;
use crate::services::owl_validator::{
    ValidationConfig, RdfTriple, PropertyGraph
};
use crate::handlers::api_handler::analytics::FEATURE_FLAGS;

// ============================================================================
// REQUEST/RESPONSE DTOs
// ============================================================================

/// Request to load ontology axioms from various sources
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadAxiomsRequest {
    /// Source of the ontology (file path, URL, or direct content)
    pub source: String,
    /// Optional format specification ("turtle", "rdf-xml", "n-triples")
    pub format: Option<String>,
    /// Whether to validate immediately after loading
    pub validate_immediately: Option<bool>,
}

/// Response for successful axiom loading
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadAxiomsResponse {
    /// Generated ontology ID
    pub ontology_id: String,
    /// Timestamp when loaded
    pub loaded_at: DateTime<Utc>,
    /// Number of axioms loaded
    pub axiom_count: Option<u32>,
    /// Loading duration in milliseconds
    pub loading_time_ms: u64,
    /// Optional validation job ID if immediate validation was requested
    pub validation_job_id: Option<String>,
}

/// Request to update ontology mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MappingRequest {
    /// Mapping configuration
    pub config: ValidationConfigDto,
    /// Whether to apply to all loaded ontologies
    pub apply_to_all: Option<bool>,
}

/// DTO for validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationConfigDto {
    /// Enable OWL reasoning
    pub enable_reasoning: Option<bool>,
    /// Reasoning timeout in seconds
    pub reasoning_timeout_seconds: Option<u64>,
    /// Enable inference generation
    pub enable_inference: Option<bool>,
    /// Maximum inference depth
    pub max_inference_depth: Option<usize>,
    /// Enable caching
    pub enable_caching: Option<bool>,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: Option<u64>,
    /// Validate cardinality constraints
    pub validate_cardinality: Option<bool>,
    /// Validate domain and range constraints
    pub validate_domains_ranges: Option<bool>,
    /// Validate disjoint classes
    pub validate_disjoint_classes: Option<bool>,
}

impl From<ValidationConfigDto> for ValidationConfig {
    fn from(dto: ValidationConfigDto) -> Self {
        ValidationConfig {
            enable_reasoning: dto.enable_reasoning.unwrap_or(true),
            reasoning_timeout_seconds: dto.reasoning_timeout_seconds.unwrap_or(30),
            enable_inference: dto.enable_inference.unwrap_or(true),
            max_inference_depth: dto.max_inference_depth.unwrap_or(3),
            enable_caching: dto.enable_caching.unwrap_or(true),
            cache_ttl_seconds: dto.cache_ttl_seconds.unwrap_or(3600),
            validate_cardinality: dto.validate_cardinality.unwrap_or(true),
            validate_domains_ranges: dto.validate_domains_ranges.unwrap_or(true),
            validate_disjoint_classes: dto.validate_disjoint_classes.unwrap_or(true),
        }
    }
}

/// Request to validate ontology against graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationRequest {
    /// Ontology ID to validate against
    pub ontology_id: String,
    /// Validation mode
    pub mode: ValidationModeDto,
    /// Optional priority (1-10, higher is more urgent)
    pub priority: Option<u8>,
    /// Whether to send progress updates via WebSocket
    pub enable_websocket_updates: Option<bool>,
    /// Client ID for WebSocket updates
    pub client_id: Option<String>,
}

/// DTO for validation mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ValidationModeDto {
    Quick,
    Full,
    Incremental,
}

impl From<ValidationModeDto> for ValidationMode {
    fn from(dto: ValidationModeDto) -> Self {
        match dto {
            ValidationModeDto::Quick => ValidationMode::Quick,
            ValidationModeDto::Full => ValidationMode::Full,
            ValidationModeDto::Incremental => ValidationMode::Incremental,
        }
    }
}

/// Response for validation request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationResponse {
    /// Generated job ID for tracking
    pub job_id: String,
    /// Current status
    pub status: String,
    /// Estimated completion time
    pub estimated_completion: Option<DateTime<Utc>>,
    /// Queue position (if queued)
    pub queue_position: Option<usize>,
    /// WebSocket URL for real-time updates (if enabled)
    pub websocket_url: Option<String>,
}

/// Request to apply inferences to the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplyInferencesRequest {
    /// RDF triples to apply inferences to
    pub rdf_triples: Vec<RdfTripleDto>,
    /// Maximum inference depth
    pub max_depth: Option<usize>,
    /// Whether to update the graph immediately
    pub update_graph: Option<bool>,
}

/// DTO for RDF triple
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RdfTripleDto {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub is_literal: Option<bool>,
    pub datatype: Option<String>,
    pub language: Option<String>,
}

impl From<RdfTripleDto> for RdfTriple {
    fn from(dto: RdfTripleDto) -> Self {
        RdfTriple {
            subject: dto.subject,
            predicate: dto.predicate,
            object: dto.object,
            is_literal: dto.is_literal.unwrap_or(false),
            datatype: dto.datatype,
            language: dto.language,
        }
    }
}

impl From<RdfTriple> for RdfTripleDto {
    fn from(triple: RdfTriple) -> Self {
        RdfTripleDto {
            subject: triple.subject,
            predicate: triple.predicate,
            object: triple.object,
            is_literal: Some(triple.is_literal),
            datatype: triple.datatype,
            language: triple.language,
        }
    }
}

/// Response for inference application
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceResult {
    /// Number of input triples
    pub input_count: usize,
    /// Generated inferred triples
    pub inferred_triples: Vec<RdfTripleDto>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Whether graph was updated
    pub graph_updated: bool,
}

/// Health status response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthStatusResponse {
    /// Overall system status
    pub status: String,
    /// Detailed health metrics
    pub health: OntologyHealthDto,
    /// Feature flag status
    pub ontology_validation_enabled: bool,
    /// Last update timestamp
    pub timestamp: DateTime<Utc>,
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
// UTILITY FUNCTIONS
// ============================================================================

/// Check if ontology validation feature is enabled
async fn check_feature_enabled() -> Result<(), ErrorResponse> {
    let flags = FEATURE_FLAGS.lock().await;

    if !flags.ontology_validation {
        let mut details = HashMap::new();
        details.insert("message".to_string(), serde_json::json!("Enable the ontology_validation feature flag to use this endpoint"));

        return Err(ErrorResponse::new("Ontology validation feature is disabled", "FEATURE_DISABLED")
            .with_details(details));
    }

    Ok(())
}

/// Create a timeout for actor communication
fn actor_timeout() -> StdDuration {
    StdDuration::from_secs(30)
}

/// Convert graph data to PropertyGraph for validation
fn extract_property_graph(_state: &AppState) -> Result<PropertyGraph, ErrorResponse> {
    // This is a simplified implementation
    // In a real scenario, we would extract the current graph data
    // from the GraphServiceActor and convert it to PropertyGraph format
    Ok(PropertyGraph {
        nodes: vec![], // TODO: Extract from graph service
        edges: vec![], // TODO: Extract from graph service
        metadata: HashMap::new(), // Empty metadata for now
    })
}

// ============================================================================
// REST ENDPOINTS
// ============================================================================

/// POST /api/ontology/load-axioms - Load ontology from file/URL
pub async fn load_axioms(
    state: web::Data<AppState>,
    req: web::Json<LoadAxiomsRequest>,
) -> impl Responder {
    info!("Loading ontology axioms from source: {}", req.source);

    // Check feature flag
    if let Err(error) = check_feature_enabled().await {
        return HttpResponse::ServiceUnavailable().json(error);
    }

    let start_time = std::time::Instant::now();

    // Send message to OntologyActor
    let load_msg = LoadOntologyAxioms {
        source: req.source.clone(),
        format: req.format.clone(),
    };

    match state.ontology_actor_addr.send(load_msg).await {
        Ok(Ok(ontology_id)) => {
            let loading_time_ms = start_time.elapsed().as_millis() as u64;

            let response = LoadAxiomsResponse {
                ontology_id,
                loaded_at: Utc::now(),
                axiom_count: None, // TODO: Return actual axiom count
                loading_time_ms,
                validation_job_id: None, // TODO: Implement immediate validation
            };

            info!("Successfully loaded ontology: {}", response.ontology_id);
            HttpResponse::Ok().json(response)
        },
        Ok(Err(error)) => {
            error!("Failed to load ontology: {}", error);
            let error_response = ErrorResponse::new(&error, "LOAD_FAILED");
            HttpResponse::BadRequest().json(error_response)
        },
        Err(mailbox_error) => {
            error!("Actor communication error: {}", mailbox_error);
            let error_response = ErrorResponse::new("Internal server error", "ACTOR_ERROR");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// POST /api/ontology/mapping - Update mapping configuration
pub async fn update_mapping(
    state: web::Data<AppState>,
    req: web::Json<MappingRequest>,
) -> impl Responder {
    info!("Updating ontology mapping configuration");

    // Check feature flag
    if let Err(error) = check_feature_enabled().await {
        return HttpResponse::ServiceUnavailable().json(error);
    }

    // Convert DTO to internal config
    let config = ValidationConfig::from(req.config.clone());

    let update_msg = UpdateOntologyMapping { config };

    match state.ontology_actor_addr.send(update_msg).await {
        Ok(Ok(())) => {
            info!("Successfully updated ontology mapping");
            HttpResponse::Ok().json(serde_json::json!({
                "status": "success",
                "message": "Mapping configuration updated",
                "timestamp": Utc::now()
            }))
        },
        Ok(Err(error)) => {
            error!("Failed to update mapping: {}", error);
            let error_response = ErrorResponse::new(&error, "MAPPING_UPDATE_FAILED");
            HttpResponse::BadRequest().json(error_response)
        },
        Err(mailbox_error) => {
            error!("Actor communication error: {}", mailbox_error);
            let error_response = ErrorResponse::new("Internal server error", "ACTOR_ERROR");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// POST /api/ontology/validate - Run validation (quick/full)
pub async fn validate_ontology(
    state: web::Data<AppState>,
    req: web::Json<ValidationRequest>,
) -> impl Responder {
    info!("Starting ontology validation: {} (mode: {:?})", req.ontology_id, req.mode);

    // Check feature flag
    if let Err(error) = check_feature_enabled().await {
        return HttpResponse::ServiceUnavailable().json(error);
    }

    // Extract property graph from current graph data
    let property_graph = match extract_property_graph(&state) {
        Ok(graph) => graph,
        Err(error) => return HttpResponse::InternalServerError().json(error),
    };

    let validation_msg = ValidateOntology {
        ontology_id: req.ontology_id.clone(),
        graph_data: property_graph,
        mode: ValidationMode::from(req.mode.clone()),
    };

    match state.ontology_actor_addr.send(validation_msg).await {
        Ok(Ok(report)) => {
            // For synchronous validation, return the report immediately
            // In a real implementation, this would be asynchronous with job tracking
            let response = ValidationResponse {
                job_id: report.id.clone(),
                status: "completed".to_string(),
                estimated_completion: Some(Utc::now()),
                queue_position: None,
                websocket_url: req.client_id.as_ref().map(|id|
                    format!("/api/ontology/ws?client_id={}", id)
                ),
            };

            info!("Validation completed for {}: {} violations found",
                  req.ontology_id, report.violations.len());
            HttpResponse::Ok().json(response)
        },
        Ok(Err(error)) => {
            error!("Validation failed: {}", error);
            let error_response = ErrorResponse::new(&error, "VALIDATION_FAILED");
            HttpResponse::BadRequest().json(error_response)
        },
        Err(mailbox_error) => {
            error!("Actor communication error: {}", mailbox_error);
            let error_response = ErrorResponse::new("Internal server error", "ACTOR_ERROR");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// GET /api/ontology/report - Get latest validation report
pub async fn get_validation_report(
    state: web::Data<AppState>,
    query: web::Query<HashMap<String, String>>,
) -> impl Responder {
    let report_id = query.get("report_id").cloned();

    info!("Retrieving validation report: {:?}", report_id);

    // Check feature flag
    if let Err(error) = check_feature_enabled().await {
        return HttpResponse::ServiceUnavailable().json(error);
    }

    let report_msg = GetOntologyReport { report_id };

    match state.ontology_actor_addr.send(report_msg).await {
        Ok(Ok(Some(report))) => {
            info!("Retrieved validation report: {}", report.id);
            HttpResponse::Ok().json(report)
        },
        Ok(Ok(None)) => {
            warn!("Validation report not found");
            let error_response = ErrorResponse::new("Report not found", "REPORT_NOT_FOUND");
            HttpResponse::NotFound().json(error_response)
        },
        Ok(Err(error)) => {
            error!("Failed to retrieve report: {}", error);
            let error_response = ErrorResponse::new(&error, "REPORT_RETRIEVAL_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        },
        Err(mailbox_error) => {
            error!("Actor communication error: {}", mailbox_error);
            let error_response = ErrorResponse::new("Internal server error", "ACTOR_ERROR");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// POST /api/ontology/apply - Apply inferences to graph
pub async fn apply_inferences(
    state: web::Data<AppState>,
    req: web::Json<ApplyInferencesRequest>,
) -> impl Responder {
    info!("Applying inferences to {} triples", req.rdf_triples.len());

    // Check feature flag
    if let Err(error) = check_feature_enabled().await {
        return HttpResponse::ServiceUnavailable().json(error);
    }

    let start_time = std::time::Instant::now();

    // Convert DTOs to internal types
    let triples: Vec<RdfTriple> = req.rdf_triples.iter()
        .map(|dto| RdfTriple::from(dto.clone()))
        .collect();

    let apply_msg = ApplyInferences {
        rdf_triples: triples,
        max_depth: req.max_depth,
    };

    match state.ontology_actor_addr.send(apply_msg).await {
        Ok(Ok(inferred_triples)) => {
            let processing_time_ms = start_time.elapsed().as_millis() as u64;

            let response = InferenceResult {
                input_count: req.rdf_triples.len(),
                inferred_triples: inferred_triples.into_iter()
                    .map(RdfTripleDto::from)
                    .collect(),
                processing_time_ms,
                graph_updated: req.update_graph.unwrap_or(false),
            };

            info!("Generated {} inferred triples", response.inferred_triples.len());
            HttpResponse::Ok().json(response)
        },
        Ok(Err(error)) => {
            error!("Failed to apply inferences: {}", error);
            let error_response = ErrorResponse::new(&error, "INFERENCE_FAILED");
            HttpResponse::BadRequest().json(error_response)
        },
        Err(mailbox_error) => {
            error!("Actor communication error: {}", mailbox_error);
            let error_response = ErrorResponse::new("Internal server error", "ACTOR_ERROR");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// GET /api/ontology/health - System health status
pub async fn get_health_status(state: web::Data<AppState>) -> impl Responder {
    info!("Retrieving ontology system health");

    let health_msg = GetOntologyHealth;

    match state.ontology_actor_addr.send(health_msg).await {
        Ok(Ok(health)) => {
            let response = HealthStatusResponse {
                status: if health.validation_queue_size > 100 { "degraded" } else { "healthy" }.to_string(),
                health: OntologyHealthDto::from(health),
                ontology_validation_enabled: true, // TODO: Check feature flag
                timestamp: Utc::now(),
            };

            HttpResponse::Ok().json(response)
        },
        Ok(Err(error)) => {
            error!("Failed to retrieve health status: {}", error);
            let error_response = ErrorResponse::new(&error, "HEALTH_CHECK_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        },
        Err(mailbox_error) => {
            error!("Actor communication error: {}", mailbox_error);
            let error_response = ErrorResponse::new("Internal server error", "ACTOR_ERROR");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

/// DELETE /api/ontology/cache - Clear caches
pub async fn clear_caches(state: web::Data<AppState>) -> impl Responder {
    info!("Clearing ontology caches");

    // Check feature flag
    if let Err(error) = check_feature_enabled().await {
        return HttpResponse::ServiceUnavailable().json(error);
    }

    let clear_msg = ClearOntologyCaches;

    match state.ontology_actor_addr.send(clear_msg).await {
        Ok(Ok(())) => {
            info!("Successfully cleared ontology caches");
            HttpResponse::Ok().json(serde_json::json!({
                "status": "success",
                "message": "All caches cleared",
                "timestamp": Utc::now()
            }))
        },
        Ok(Err(error)) => {
            error!("Failed to clear caches: {}", error);
            let error_response = ErrorResponse::new(&error, "CACHE_CLEAR_FAILED");
            HttpResponse::InternalServerError().json(error_response)
        },
        Err(mailbox_error) => {
            error!("Actor communication error: {}", mailbox_error);
            let error_response = ErrorResponse::new("Internal server error", "ACTOR_ERROR");
            HttpResponse::InternalServerError().json(error_response)
        }
    }
}

// ============================================================================
// WEBSOCKET IMPLEMENTATION
// ============================================================================

/// WebSocket actor for real-time validation progress updates
pub struct OntologyWebSocket {
    /// Client ID for tracking
    client_id: String,
    /// Connection to ontology actor
    ontology_addr: Addr<OntologyActor>,
}

impl OntologyWebSocket {
    pub fn new(client_id: String, ontology_addr: Addr<OntologyActor>) -> Self {
        Self {
            client_id,
            ontology_addr,
        }
    }
}

impl actix::Actor for OntologyWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("WebSocket connection started for client: {}", self.client_id);

        // Send initial connection confirmation
        let msg = serde_json::json!({
            "type": "connection_established",
            "client_id": self.client_id,
            "timestamp": Utc::now()
        });
        ctx.text(msg.to_string());
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("WebSocket connection stopped for client: {}", self.client_id);
    }
}

impl actix::StreamHandler<Result<ws::Message, ws::ProtocolError>> for OntologyWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                debug!("Received WebSocket message from {}: {}", self.client_id, text);

                // Echo back for now - in production, this would handle client commands
                let response = serde_json::json!({
                    "type": "echo",
                    "original": &*text,
                    "timestamp": Utc::now()
                });
                ctx.text(response.to_string());
            }
            Ok(ws::Message::Ping(msg)) => {
                ctx.pong(&msg);
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
pub async fn websocket_handler(
    req: HttpRequest,
    stream: web::Payload,
    state: web::Data<AppState>,
    query: web::Query<HashMap<String, String>>,
) -> Result<HttpResponse, ActixError> {
    info!("WebSocket upgrade request for ontology updates");

    // Check feature flag
    if let Err(error) = check_feature_enabled().await {
        return Ok(HttpResponse::ServiceUnavailable().json(error));
    }

    let client_id = query.get("client_id")
        .cloned()
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let websocket = OntologyWebSocket::new(
        client_id,
        state.ontology_actor_addr.clone()
    );

    ws::start(websocket, &req, stream)
}

// ============================================================================
// ROUTE CONFIGURATION
// ============================================================================

/// Configure ontology API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/ontology")
            .route("/load-axioms", web::post().to(load_axioms))
            .route("/mapping", web::post().to(update_mapping))
            .route("/validate", web::post().to(validate_ontology))
            .route("/report", web::get().to(get_validation_report))
            .route("/apply", web::post().to(apply_inferences))
            .route("/health", web::get().to(get_health_status))
            .route("/cache", web::delete().to(clear_caches))
            .route("/ws", web::get().to(websocket_handler))
    );
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};
    use serde_json::Value;

    #[actix_web::test]
    async fn test_health_endpoint_structure() {
        // Test the structure of our DTOs
        let health = OntologyHealthDto {
            loaded_ontologies: 5,
            cached_reports: 10,
            validation_queue_size: 2,
            last_validation: Some(Utc::now()),
            cache_hit_rate: 0.85,
            avg_validation_time_ms: 1500.0,
            active_jobs: 1,
            memory_usage_mb: 256.0,
        };

        let response = HealthStatusResponse {
            status: "healthy".to_string(),
            health,
            ontology_validation_enabled: true,
            timestamp: Utc::now(),
        };

        // Ensure it serializes correctly
        let json = serde_json::to_value(&response).unwrap();
        assert!(json.get("status").is_some());
        assert!(json.get("health").is_some());
        assert!(json.get("ontologyValidationEnabled").is_some());
    }

    #[tokio::test]
    async fn test_validation_config_conversion() {
        let dto = ValidationConfigDto {
            enable_reasoning: Some(true),
            reasoning_timeout_seconds: Some(60),
            enable_inference: Some(false),
            max_inference_depth: Some(5),
            enable_caching: Some(true),
            cache_ttl_seconds: Some(7200),
            validate_cardinality: Some(true),
            validate_domains_ranges: Some(true),
            validate_disjoint_classes: Some(false),
        };

        let config = ValidationConfig::from(dto);
        assert_eq!(config.enable_reasoning, true);
        assert_eq!(config.reasoning_timeout_seconds, 60);
        assert_eq!(config.enable_inference, false);
        assert_eq!(config.max_inference_depth, 5);
    }

    #[tokio::test]
    async fn test_rdf_triple_conversion() {
        let dto = RdfTripleDto {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            is_literal: Some(false),
            datatype: Some("uri".to_string()),
            language: None,
        };

        let triple = RdfTriple::from(dto.clone());
        let back_to_dto = RdfTripleDto::from(triple);

        assert_eq!(dto.subject, back_to_dto.subject);
        assert_eq!(dto.predicate, back_to_dto.predicate);
        assert_eq!(dto.object, back_to_dto.object);
        assert_eq!(dto.is_literal, back_to_dto.is_literal);
        assert_eq!(dto.datatype, back_to_dto.datatype);
        assert_eq!(dto.language, back_to_dto.language);
    }
}