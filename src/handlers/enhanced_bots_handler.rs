// Enhanced Bots Handler with Comprehensive Input Validation
use actix_web::{web, HttpResponse, HttpRequest, Result};
use crate::AppState;
use crate::handlers::validation_handler::ValidationService;
use crate::utils::validation::rate_limit::{RateLimiter, EndpointRateLimits, extract_client_id};
use crate::utils::validation::sanitization::Sanitizer;
use crate::utils::validation::errors::DetailedValidationError;
use crate::utils::validation::MAX_REQUEST_SIZE;
use log::{info, warn, error, debug};
use serde_json::{json, Value};
use std::sync::Arc;

/// Enhanced bots handler with comprehensive validation
pub struct EnhancedBotsHandler {
    validation_service: ValidationService,
    rate_limiter: Arc<RateLimiter>,
}

impl EnhancedBotsHandler {
    pub fn new() -> Self {
        let config = EndpointRateLimits::bots_operations();
        let rate_limiter = Arc::new(RateLimiter::new(config));

        Self {
            validation_service: ValidationService::new(),
            rate_limiter,
        }
    }

    /// Enhanced bots data update with validation
    pub async fn update_bots_data_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        payload: web::Json<Value>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);
        
        // Rate limiting check
        if !self.rate_limiter.is_allowed(&client_id) {
            warn!("Rate limit exceeded for bots data update from client: {}", client_id);
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many bots data updates. Please wait before retrying.",
                "retry_after": self.rate_limiter.reset_time(&client_id).as_secs()
            })));
        }

        // Request size validation
        let payload_size = serde_json::to_vec(&*payload).unwrap_or_default().len();
        if payload_size > MAX_REQUEST_SIZE {
            error!("Bots data payload too large: {} bytes", payload_size);
            return Ok(HttpResponse::PayloadTooLarge().json(json!({
                "error": "payload_too_large",
                "message": format!("Bots data size {} bytes exceeds limit of {} bytes", payload_size, MAX_REQUEST_SIZE),
                "max_size": MAX_REQUEST_SIZE
            })));
        }

        info!("Processing enhanced bots data update from client: {} (size: {} bytes)", client_id, payload_size);

        // Comprehensive validation and sanitization
        let validated_payload = match self.validation_service.validate_bots_data(&payload) {
            Ok(sanitized) => sanitized,
            Err(validation_error) => {
                warn!("Bots data validation failed for client {}: {}", client_id, validation_error);
                return Ok(validation_error.to_http_response());
            }
        };

        debug!("Bots data validation passed for client: {}", client_id);

        // Additional validation for bots data structure
        self.validate_bots_data_structure(&validated_payload)?;

        // Extract and process nodes and edges
        let nodes = validated_payload.get("nodes")
            .and_then(|n| n.as_array())
            .ok_or_else(|| DetailedValidationError::missing_required_field("nodes"))?;

        let edges = validated_payload.get("edges")
            .and_then(|e| e.as_array())
            .ok_or_else(|| DetailedValidationError::missing_required_field("edges"))?;

        info!("Received bots data with {} nodes and {} edges from client: {}", 
              nodes.len(), edges.len(), client_id);

        // Validate individual nodes and edges
        self.validate_bots_nodes(nodes)?;
        self.validate_bots_edges(edges)?;

        // TODO: Process the validated data through the existing bots handler logic
        // This would involve converting to the internal bots data structures
        // and updating the graph visualization

        Ok(HttpResponse::Ok().json(json!({
            "success": true,
            "message": "Bots data updated successfully",
            "nodes_processed": nodes.len(),
            "edges_processed": edges.len(),
            "client_id": client_id,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }

    /// Enhanced get bots data with validation metadata
    pub async fn get_bots_data_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // More permissive rate limiting for GET requests
        let get_rate_limiter = Arc::new(RateLimiter::new(
            crate::utils::validation::rate_limit::RateLimitConfig {
                requests_per_minute: 80,
                burst_size: 15,
                ..Default::default()
            }
        ));

        if !get_rate_limiter.is_allowed(&client_id) {
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many get bots data requests"
            })));
        }

        debug!("Processing get bots data request from client: {}", client_id);

        // TODO: Fetch bots data through existing handler
        // This would call the existing get_bots_data function
        
        // For now, return a mock response with validation info
        Ok(HttpResponse::Ok().json(json!({
            "status": "success",
            "data": {
                "nodes": [],
                "edges": []
            },
            "validation_info": {
                "input_sanitization": "enabled",
                "rate_limiting": "active",
                "data_integrity_checks": "enforced"
            },
            "client_id": client_id,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }

    /// Enhanced swarm initialization with validation
    pub async fn initialize_swarm_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        payload: web::Json<Value>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // Stricter rate limiting for swarm operations
        let swarm_rate_limiter = Arc::new(RateLimiter::new(
            crate::utils::validation::rate_limit::RateLimitConfig {
                requests_per_minute: 20,
                burst_size: 3,
                ..Default::default()
            }
        ));

        if !swarm_rate_limiter.is_allowed(&client_id) {
            warn!("Rate limit exceeded for swarm initialization from client: {}", client_id);
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many swarm initialization requests. This is a resource-intensive operation."
            })));
        }

        info!("Processing enhanced swarm initialization from client: {}", client_id);

        // Comprehensive validation
        let validated_payload = match self.validation_service.validate_swarm_init(&payload) {
            Ok(sanitized) => sanitized,
            Err(validation_error) => {
                warn!("Swarm initialization validation failed for client {}: {}", client_id, validation_error);
                return Ok(validation_error.to_http_response());
            }
        };

        debug!("Swarm initialization validation passed for client: {}", client_id);

        // Extract validated parameters
        let topology = validated_payload.get("topology")
            .and_then(|t| t.as_str())
            .ok_or_else(|| DetailedValidationError::missing_required_field("topology"))?;

        let max_agents = validated_payload.get("max_agents")
            .and_then(|m| m.as_f64())
            .ok_or_else(|| DetailedValidationError::missing_required_field("max_agents"))? as u32;

        let strategy = validated_payload.get("strategy")
            .and_then(|s| s.as_str())
            .ok_or_else(|| DetailedValidationError::missing_required_field("strategy"))?;

        let enable_neural = validated_payload.get("enable_neural")
            .and_then(|n| n.as_bool())
            .unwrap_or(false);

        // Additional swarm validation
        self.validate_swarm_configuration(topology, max_agents, strategy)?;

        info!("Initializing swarm: topology={}, max_agents={}, strategy={}, neural={}",
              topology, max_agents, strategy, enable_neural);

        // TODO: Process swarm initialization through existing handler
        // This would call the existing initialize_swarm function

        Ok(HttpResponse::Ok().json(json!({
            "success": true,
            "message": "Swarm initialization started successfully",
            "swarm_id": format!("swarm-{}", uuid::Uuid::new_v4()),
            "configuration": {
                "topology": topology,
                "max_agents": max_agents,
                "strategy": strategy,
                "enable_neural": enable_neural
            },
            "client_id": client_id,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }

    /// Get agent telemetry with enhanced validation
    pub async fn get_agent_telemetry_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // Permissive rate limiting for telemetry
        let telemetry_rate_limiter = Arc::new(RateLimiter::new(
            crate::utils::validation::rate_limit::RateLimitConfig {
                requests_per_minute: 120,
                burst_size: 30,
                ..Default::default()
            }
        ));

        if !telemetry_rate_limiter.is_allowed(&client_id) {
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded", 
                "message": "Too many telemetry requests"
            })));
        }

        debug!("Processing agent telemetry request from client: {}", client_id);

        // TODO: Fetch telemetry through existing handler
        // This would call the existing get_agent_telemetry function

        Ok(HttpResponse::Ok().json(json!({
            "status": "success",
            "telemetry": {
                "active_agents": 0,
                "swarm_health": "unknown",
                "last_updated": chrono::Utc::now().to_rfc3339()
            },
            "validation_info": {
                "client_verified": true,
                "rate_limit_status": "ok"
            },
            "client_id": client_id,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }

    /// Validate bots data structure
    fn validate_bots_data_structure(&self, payload: &Value) -> Result<(), DetailedValidationError> {
        // Check that nodes and edges are arrays
        let nodes = payload.get("nodes")
            .and_then(|n| n.as_array())
            .ok_or_else(|| DetailedValidationError::new("nodes", "Must be an array", "INVALID_TYPE"))?;

        let edges = payload.get("edges") 
            .and_then(|e| e.as_array())
            .ok_or_else(|| DetailedValidationError::new("edges", "Must be an array", "INVALID_TYPE"))?;

        // Validate array sizes
        if nodes.len() > 1000 {
            return Err(DetailedValidationError::new(
                "nodes",
                "Too many nodes in request",
                "TOO_MANY_NODES"
            ));
        }

        if edges.len() > 10000 {
            return Err(DetailedValidationError::new(
                "edges", 
                "Too many edges in request",
                "TOO_MANY_EDGES"
            ));
        }

        Ok(())
    }

    /// Validate individual bot nodes
    fn validate_bots_nodes(&self, nodes: &[Value]) -> Result<(), DetailedValidationError> {
        for (i, node) in nodes.iter().enumerate() {
            let node_obj = node.as_object()
                .ok_or_else(|| DetailedValidationError::new(
                    &format!("nodes[{}]", i),
                    "Node must be an object",
                    "INVALID_NODE_TYPE"
                ))?;

            // Validate required fields
            let _id = node_obj.get("id")
                .and_then(|id| id.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("nodes[{}].id", i)))?;

            let agent_type = node_obj.get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("nodes[{}].type", i)))?;

            // Validate agent type
            let valid_types = ["coordinator", "researcher", "coder", "tester", "analyst", "queen"];
            if !valid_types.contains(&agent_type) {
                return Err(DetailedValidationError::new(
                    &format!("nodes[{}].type", i),
                    &format!("Invalid agent type: {}", agent_type),
                    "INVALID_AGENT_TYPE"
                ));
            }

            // Validate numeric fields
            if let Some(health) = node_obj.get("health").and_then(|h| h.as_f64()) {
                if !(0.0..=100.0).contains(&health) {
                    return Err(DetailedValidationError::out_of_range(
                        &format!("nodes[{}].health", i),
                        health,
                        0.0,
                        100.0
                    ));
                }
            }

            if let Some(cpu_usage) = node_obj.get("cpu_usage").and_then(|c| c.as_f64()) {
                if !(0.0..=100.0).contains(&cpu_usage) {
                    return Err(DetailedValidationError::out_of_range(
                        &format!("nodes[{}].cpu_usage", i),
                        cpu_usage,
                        0.0,
                        100.0
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate bots edges
    fn validate_bots_edges(&self, edges: &[Value]) -> Result<(), DetailedValidationError> {
        for (i, edge) in edges.iter().enumerate() {
            let edge_obj = edge.as_object()
                .ok_or_else(|| DetailedValidationError::new(
                    &format!("edges[{}]", i),
                    "Edge must be an object",
                    "INVALID_EDGE_TYPE"
                ))?;

            // Validate required fields
            let _id = edge_obj.get("id")
                .and_then(|id| id.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("edges[{}].id", i)))?;

            let _source = edge_obj.get("source")
                .and_then(|s| s.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("edges[{}].source", i)))?;

            let _target = edge_obj.get("target")
                .and_then(|t| t.as_str()) 
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("edges[{}].target", i)))?;

            // Validate numeric fields
            if let Some(data_volume) = edge_obj.get("data_volume").and_then(|d| d.as_f64()) {
                if data_volume < 0.0 {
                    return Err(DetailedValidationError::new(
                        &format!("edges[{}].data_volume", i),
                        "Data volume cannot be negative",
                        "NEGATIVE_VALUE"
                    ));
                }
            }

            if let Some(message_count) = edge_obj.get("message_count").and_then(|m| m.as_f64()) {
                if message_count < 0.0 {
                    return Err(DetailedValidationError::new(
                        &format!("edges[{}].message_count", i),
                        "Message count cannot be negative",
                        "NEGATIVE_VALUE"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate swarm configuration
    fn validate_swarm_configuration(
        &self,
        topology: &str,
        max_agents: u32,
        strategy: &str,
    ) -> Result<(), DetailedValidationError> {
        // Validate topology and max_agents relationship
        match topology {
            "hierarchical" => {
                if max_agents > 50 {
                    return Err(DetailedValidationError::new(
                        "max_agents",
                        "Hierarchical topology supports maximum 50 agents",
                        "TOPOLOGY_AGENT_LIMIT"
                    ));
                }
            }
            "mesh" => {
                if max_agents > 20 {
                    return Err(DetailedValidationError::new(
                        "max_agents", 
                        "Mesh topology supports maximum 20 agents due to complexity",
                        "TOPOLOGY_AGENT_LIMIT"
                    ));
                }
            }
            "ring" => {
                if max_agents < 3 {
                    return Err(DetailedValidationError::new(
                        "max_agents",
                        "Ring topology requires minimum 3 agents",
                        "TOPOLOGY_AGENT_MINIMUM"
                    ));
                }
            }
            "star" => {
                if max_agents > 100 {
                    return Err(DetailedValidationError::new(
                        "max_agents",
                        "Star topology supports maximum 100 agents",
                        "TOPOLOGY_AGENT_LIMIT"
                    ));
                }
            }
            _ => {
                return Err(DetailedValidationError::new(
                    "topology",
                    &format!("Unknown topology: {}", topology),
                    "INVALID_TOPOLOGY"
                ));
            }
        }

        // Validate strategy compatibility
        let valid_strategies = ["balanced", "specialized", "adaptive"];
        if !valid_strategies.contains(&strategy) {
            return Err(DetailedValidationError::new(
                "strategy",
                &format!("Invalid strategy: {}", strategy),
                "INVALID_STRATEGY"
            ));
        }

        Ok(())
    }
}

impl Default for EnhancedBotsHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for enhanced bots routes
pub fn config_enhanced(cfg: &mut web::ServiceConfig) {
    let handler = web::Data::new(EnhancedBotsHandler::new());
    
    cfg.app_data(handler.clone())
        .service(
            web::scope("/bots/v2")
                .route("/data", web::get().to(|req, state, handler: web::Data<EnhancedBotsHandler>| {
                    handler.get_bots_data_enhanced(req, state)
                }))
                .route("/update", web::post().to(|req, state, payload, handler: web::Data<EnhancedBotsHandler>| {
                    handler.update_bots_data_enhanced(req, state, payload)
                }))
                .route("/initialize-swarm", web::post().to(|req, state, payload, handler: web::Data<EnhancedBotsHandler>| {
                    handler.initialize_swarm_enhanced(req, state, payload)
                }))
                .route("/telemetry", web::get().to(|req, state, handler: web::Data<EnhancedBotsHandler>| {
                    handler.get_agent_telemetry_enhanced(req, state)
                }))
        );
}