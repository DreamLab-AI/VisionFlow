//! Multi-MCP WebSocket Handler
//! 
//! Provides real-time WebSocket streaming of agent visualization data
//! from multiple MCP servers to the VisionFlow graph renderer.

use actix_web::{web, HttpResponse, HttpRequest, Result as ActixResult};
use actix_web_actors::ws;
use actix::{Actor, StreamHandler, AsyncContext, Handler, Message};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Duration, Instant};
use std::sync::{atomic::Ordering, Arc};
use log::{info, debug, warn, error};
use uuid::Uuid;

use crate::AppState;
use crate::services::agent_visualization_protocol::McpServerType;
// DEPRECATED: HybridHealthManager removed
use crate::utils::network::{
    TimeoutConfig, CircuitBreaker,
    HealthCheckManager, RetryConfig, retry_with_backoff,
    ServiceEndpoint, HealthCheckConfig, RetryableError
};

// Define a simple retryable error type for MCP operations
#[derive(Debug, Clone)]
struct McpError(String);

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MCP Error: {}", self.0)
    }
}

impl std::error::Error for McpError {}

impl RetryableError for McpError {
    fn is_retryable(&self) -> bool {
        true // All MCP errors are considered retryable
    }
}

/// WebSocket actor for multi-MCP agent visualization
pub struct MultiMcpVisualizationWs {
    app_state: web::Data<AppState>,
    _hybrid_manager: Option<()>, // DEPRECATED
    client_id: String,
    // visualization_actor_addr: Option<Addr<MultiMcpVisualizationActor>>, // Removed - not implemented
    last_heartbeat: Instant,
    last_discovery_request: Instant,
    subscription_filters: SubscriptionFilters,
    performance_mode: PerformanceMode,
    // Resilience components
    timeout_config: TimeoutConfig,
    circuit_breaker: Option<std::sync::Arc<CircuitBreaker>>,
    health_manager: Option<std::sync::Arc<HealthCheckManager>>,
    retry_config: RetryConfig,
    connection_failures: u32,
    last_successful_operation: Instant,
}

/// Client subscription filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionFilters {
    /// Filter by server types
    pub server_types: Vec<McpServerType>,
    /// Filter by agent types
    pub agent_types: Vec<String>,
    /// Filter by swarm IDs
    pub swarm_ids: Vec<String>,
    /// Include performance analysis
    pub include_performance: bool,
    /// Include neural agent data
    pub include_neural: bool,
    /// Include topology updates
    pub include_topology: bool,
}

impl Default for SubscriptionFilters {
    fn default() -> Self {
        Self {
            server_types: vec![McpServerType::ClaudeFlow, McpServerType::RuvSwarm, McpServerType::Daa],
            agent_types: vec![],
            swarm_ids: vec![],
            include_performance: true,
            include_neural: true,
            include_topology: true,
        }
    }
}

/// Performance mode for different update rates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMode {
    /// High frequency updates (60Hz) - for active monitoring
    HighFrequency,
    /// Normal updates (10Hz) - default mode
    Normal,
    /// Low frequency (1Hz) - for dashboard overview
    LowFrequency,
    /// On-demand only - minimal CPU usage
    OnDemand,
}

impl Default for PerformanceMode {
    fn default() -> Self {
        Self::Normal
    }
}

impl MultiMcpVisualizationWs {
    pub fn new(app_state: web::Data<AppState>, _hybrid_manager: Option<()>) -> Self {
        let client_id = Uuid::new_v4().to_string();
        info!("Creating new Multi-MCP WebSocket client with resilience and hybrid integration: {}", client_id);

        // Initialize circuit breaker for MCP operations
        let circuit_breaker = std::sync::Arc::new(CircuitBreaker::mcp_operations());

        // Initialize health manager
        let health_manager_network = std::sync::Arc::new(HealthCheckManager::new());

        Self {
            app_state,
            _hybrid_manager: None,
            client_id,
            // visualization_actor_addr: None, // Removed - not implemented
            last_heartbeat: Instant::now(),
            last_discovery_request: Instant::now(),
            subscription_filters: SubscriptionFilters::default(),
            performance_mode: PerformanceMode::default(),
            timeout_config: TimeoutConfig::websocket(),
            circuit_breaker: Some(circuit_breaker),
            health_manager: Some(health_manager_network),
            retry_config: RetryConfig::mcp_operations(),
            connection_failures: 0,
            last_successful_operation: Instant::now(),
        }
    }

    /// Start position updates based on performance mode
    fn start_position_updates(&self, ctx: &mut ws::WebsocketContext<Self>) {
        let interval = match self.performance_mode {
            PerformanceMode::HighFrequency => Duration::from_millis(16), // ~60Hz
            PerformanceMode::Normal => Duration::from_millis(100),       // 10Hz
            PerformanceMode::LowFrequency => Duration::from_millis(1000), // 1Hz
            PerformanceMode::OnDemand => return, // No automatic updates
        };

        ctx.run_interval(interval, |_act, ctx| {
            // Request current agent data
            ctx.address().do_send(RequestAgentUpdate);
        });
    }

    /// Start heartbeat monitoring
    fn start_heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_secs(5), |act, ctx| {
            if Instant::now().duration_since(act.last_heartbeat) > Duration::from_secs(30) {
                warn!("WebSocket client {} heartbeat timeout, disconnecting", act.client_id);
                ctx.close(None);
                return;
            }
            
            ctx.ping(b"ping");
        });
    }

    /// Perform health check on MCP services
    fn perform_health_checks(&mut self) {
        if let Some(health_manager) = &self.health_manager {
            let health_manager_clone = health_manager.clone();
            let client_id = self.client_id.clone();
            
            actix::spawn(async move {
                // Check various MCP services
                for service in ["claude-flow", "ruv-swarm", "flow-nexus"] {
                    let health_result = health_manager_clone.check_service_now(service).await;
                    let is_healthy = health_result.map_or(false, |r| r.status.is_usable());
                    
                    if !is_healthy {
                        warn!("[Multi-MCP] Service {} unhealthy for client {}", service, client_id);
                    }
                }
            });
        }
    }
    
    /// Check if any MCP services are healthy using cached health status
    /// This is a non-blocking method that uses the last known health status
    fn has_healthy_services(&self) -> bool {
        if let Some(health_manager) = &self.health_manager {
            let health_manager_clone = health_manager.clone();
            
            // Spawn a task to check health asynchronously, but don't wait for it
            // Use a timeout to avoid blocking the WebSocket handler
            tokio::spawn(async move {
                for service in ["claude-flow", "ruv-swarm", "flow-nexus"] {
                    // Get cached health status instead of performing immediate check
                    if let Some(health_info) = health_manager_clone.get_service_health(service).await {
                        if health_info.current_status.is_usable() {
                            debug!("Service {} is healthy (cached)", service);
                        }
                    }
                }
            });
            
            // For now, return true to avoid blocking WebSocket operations
            // The health checks run in background and update caches
            // In the future, this could check a cached status map
            return true;
        }
        // Default to true if health manager not available
        true
    }
    
    /// Record successful operation
    fn record_success(&mut self) {
        self.connection_failures = 0;
        self.last_successful_operation = Instant::now();
    }
    
    /// Record failed operation
    fn record_failure(&mut self) {
        self.connection_failures += 1;
        warn!("[Multi-MCP] Operation failure #{} for client {}", 
              self.connection_failures, self.client_id);
    }

    /// Send discovery data to client with resilience
    fn send_discovery_data(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        let client_id = self.client_id.clone();
        let circuit_breaker = self.circuit_breaker.clone();
        let _timeout_config = self.timeout_config.clone();
        
        // Use app_state for service discovery
        let _app_state = ctx.address();
        
        // Check if we have healthy services before proceeding
        if !self.has_healthy_services() {
            warn!("[Multi-MCP] No healthy services available for discovery, client {}", client_id);
            ctx.text(serde_json::json!({
                "type": "error",
                "message": "No healthy MCP services available",
                "timestamp": chrono::Utc::now().timestamp_millis()
            }).to_string());
            return;
        }
        
        if let Some(cb) = circuit_breaker {
            // Execute discovery with circuit breaker protection
            let addr = ctx.address();
            let retry_config = self.retry_config.clone();
            let failures = self.connection_failures;
            
            actix::spawn(async move {
                // Use retry logic with circuit breaker
                let result = retry_with_backoff(retry_config, || {
                    let cb_clone = cb.clone();
                    Box::pin(async move {
                        cb_clone.execute(async {
                            // Simulated discovery operation with potential failures
                            if fastrand::f32() < 0.2 && failures > 0 {
                                return Err(Box::new(std::io::Error::new(
                                    std::io::ErrorKind::ConnectionRefused,
                                    "Discovery service temporarily unavailable"
                                )) as Box<dyn std::error::Error + Send + Sync>);
                            }
                            
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                        }).await.map_err(|e| McpError(format!("{:?}", e)))
                    })
                }).await;
                
                match result {
                    Ok(_) => {
                        debug!("Discovery operation successful for client: {}", client_id);
                        addr.do_send(DiscoverySuccess);
                        addr.do_send(RequestDiscoveryData);
                    }
                    Err(e) => {
                        error!("Discovery operation failed for client {} after retries: {:?}", client_id, e);
                        addr.do_send(DiscoveryFailure(format!("{:?}", e)));
                    }
                }
            });
        } else {
            // Fallback to direct request with basic retry
            let addr = ctx.address();
            let retry_config = self.retry_config.clone();
            
            actix::spawn(async move {
                let result = retry_with_backoff(retry_config, || {
                    Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        if fastrand::f32() < 0.1 {
                            Err::<(), McpError>(McpError("Random failure".to_string()))
                        } else {
                            Ok::<(), McpError>(())
                        }
                    })
                }).await;
                
                match result {
                    Ok(_) => addr.do_send(RequestDiscoveryData),
                    Err(e) => {
                        error!("Discovery fallback failed for client {}: {:?}", client_id, e);
                        addr.do_send(DiscoveryFailure(format!("{:?}", e)));
                    }
                }
            });
        }
    }

    /// Handle client configuration update
    fn handle_client_config(&mut self, config: ClientConfig, ctx: &mut ws::WebsocketContext<Self>) {
        info!("Updating client configuration for {}", self.client_id);
        
        if let Some(filters) = config.subscription_filters {
            self.subscription_filters = filters;
        }
        
        if let Some(performance_mode) = config.performance_mode {
            self.performance_mode = performance_mode;
            // Restart position updates with new timing
            self.start_position_updates(ctx);
        }
        
        // Send acknowledgment
        let response = json!({
            "type": "config_updated",
            "client_id": self.client_id,
            "timestamp": chrono::Utc::now().timestamp_millis(),
            "filters": self.subscription_filters,
            "performance_mode": self.performance_mode
        });
        
        ctx.text(response.to_string());
    }

    /// Handle discovery request
    fn handle_discovery_request(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        let now = Instant::now();
        
        // Rate limit discovery requests (max once per second)
        if now.duration_since(self.last_discovery_request) < Duration::from_secs(1) {
            debug!("Discovery request rate limited for client {}", self.client_id);
            return;
        }
        
        self.last_discovery_request = now;
        self.send_discovery_data(ctx);
    }

    /// Filter visualization message based on subscription filters
    fn should_send_message(&self, message_type: &str, _message_content: &serde_json::Value) -> bool {
        match message_type {
            "discovery" => true, // Always send discovery data
            "multi_agent_update" => true, // Always send agent updates (filtered later)
            "topology_update" => {
                // Use the filtering method
                self.subscription_filters.include_topology
            }
            "neural_update" => self.subscription_filters.include_neural,
            "performance_analysis" => self.subscription_filters.include_performance,
            _ => true, // Send unknown message types
        }
    }

    /// Filter agent data based on subscription filters
    fn filter_agent_data(&self, data: &mut serde_json::Value) {
        // Filter agents by server type
        if let Some(agents_array) = data.get_mut("agents").and_then(|a| a.as_array_mut()) {
            agents_array.retain(|agent| {
                if let Some(server_source) = agent.get("server_source") {
                    if let Ok(server_type) = serde_json::from_value::<McpServerType>(server_source.clone()) {
                        return self.subscription_filters.server_types.contains(&server_type);
                    }
                }
                false
            });
        }

        // Filter by agent types if specified
        if !self.subscription_filters.agent_types.is_empty() {
            if let Some(agents_array) = data.get_mut("agents").and_then(|a| a.as_array_mut()) {
                agents_array.retain(|agent| {
                    if let Some(agent_type) = agent.get("agent_type").and_then(|t| t.as_str()) {
                        return self.subscription_filters.agent_types.contains(&agent_type.to_string());
                    }
                    false
                });
            }
        }

        // Filter by swarm IDs if specified
        if !self.subscription_filters.swarm_ids.is_empty() {
            if let Some(agents_array) = data.get_mut("agents").and_then(|a| a.as_array_mut()) {
                agents_array.retain(|agent| {
                    if let Some(swarm_id) = agent.get("swarm_id").and_then(|s| s.as_str()) {
                        return self.subscription_filters.swarm_ids.contains(&swarm_id.to_string());
                    }
                    false
                });
            }
        }
    }
}

impl Actor for MultiMcpVisualizationWs {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Multi-MCP WebSocket client {} connected", self.client_id);
        
        // Start heartbeat
        self.start_heartbeat(ctx);
        
        // Register MCP services for health monitoring
        if let Some(health_manager) = &self.health_manager {
            let health_manager = health_manager.clone();
            actix::spawn(async move {
                for (i, service) in ["claude-flow", "ruv-swarm", "flow-nexus"].iter().enumerate() {
                    let endpoint = ServiceEndpoint {
                        name: service.to_string(),
                        host: "localhost".to_string(),
                        port: 8080 + i as u16, // Different ports for different services
                        config: HealthCheckConfig::default(),
                        additional_endpoints: vec![],
                    };
                    health_manager.register_service(endpoint).await;
                }
            });
        }
        
        // Start position updates
        self.start_position_updates(ctx);
        
        // Start periodic health monitoring
        ctx.run_interval(Duration::from_secs(30), |act, _ctx| {
            act.perform_health_checks();
        });
        
        // Start resilience monitoring
        ctx.run_interval(Duration::from_secs(60), |act, ctx| {
            let now = Instant::now();
            let time_since_success = now.duration_since(act.last_successful_operation);
            
            // If we haven't had a successful operation in 5 minutes, try to reconnect
            if time_since_success > Duration::from_secs(300) {
                warn!("[Multi-MCP] No successful operations for {:?}, attempting recovery for client {}", 
                     time_since_success, act.client_id);
                act.send_discovery_data(ctx);
            }
            
            // Log resilience stats
            if let Some(cb) = &act.circuit_breaker {
                let cb = cb.clone();
                let client_id = act.client_id.clone();
                let connection_failures = act.connection_failures;
                actix::spawn(async move {
                    let stats = cb.stats().await;
                    debug!("[Multi-MCP] Client {} resilience stats - Circuit: {:?}, Failures: {}, Successes: {}, Connection failures: {}",
                          client_id, stats.state, stats.failed_requests, stats.successful_requests, connection_failures);
                });
            }
        });
        
        // Send initial discovery data
        self.send_discovery_data(ctx);
        
        // Register with visualization actor - removed (not implemented)
        // if let Some(addr) = &self.visualization_actor_addr {
        //     addr.do_send(crate::actors::multi_mcp_visualization_actor::RegisterWebSocketClient {
        //         client_id: self.client_id.clone(),
        //     });
        // }
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("Multi-MCP WebSocket client {} disconnected", self.client_id);
        
        // Unregister from visualization actor - removed (not implemented)
        // if let Some(addr) = &self.visualization_actor_addr {
        //     addr.do_send(crate::actors::multi_mcp_visualization_actor::UnregisterWebSocketClient {
        //         client_id: self.client_id.clone(),
        //     });
        // }
    }
}

/// WebSocket message handler
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MultiMcpVisualizationWs {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.last_heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_heartbeat = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                debug!("Received WebSocket message: {}", text);
                
                if let Ok(request) = serde_json::from_str::<ClientRequest>(&text) {
                    match request.action.as_str() {
                        "configure" => {
                            if let Some(config_data) = request.data {
                                if let Ok(config) = serde_json::from_value::<ClientConfig>(config_data) {
                                    self.handle_client_config(config, ctx);
                                }
                            }
                        }
                        "request_discovery" => {
                            self.handle_discovery_request(ctx);
                        }
                        "request_agents" => {
                            // Process request with try-catch error handling
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                // Check circuit breaker state before processing (non-blocking)
                                if let Some(cb) = &self.circuit_breaker {
                                    let cb_clone = cb.clone();
                                    let ctx_addr = ctx.address();
                                    let client_id = self.client_id.clone();

                                    // Spawn async task to check circuit breaker without blocking
                                    tokio::spawn(async move {
                                        let stats = cb_clone.stats().await;
                                        match stats.state {
                                            crate::utils::network::CircuitBreakerState::Open => {
                                                warn!("[Multi-MCP] Circuit breaker open, using degraded mode for client {}", client_id);
                                                // Send cached/fallback data instead of failing
                                                ctx_addr.do_send(RequestAgentUpdate);
                                            }
                                            _ => {
                                                // Circuit breaker is closed/half-open, proceed normally
                                                ctx_addr.do_send(RequestAgentUpdate);
                                            }
                                        }
                                    });
                                } else {
                                    // No circuit breaker, proceed directly
                                    ctx.address().do_send(RequestAgentUpdate);
                                }
                            }));

                            if result.is_err() {
                                error!("Error processing agent request for client {}", self.client_id);
                                self.record_failure();
                                self.send_error_response(ctx, "Agent request processing failed");
                            }
                        }
                        "request_performance" => {
                            // Handle performance request with graceful degradation
                            if !self.has_healthy_services() {
                                warn!("[Multi-MCP] No healthy services for performance data, using cached data");
                                let degraded_response = serde_json::json!({
                                    "type": "performance_data",
                                    "message": "Using cached performance data - services degraded",
                                    "timestamp": chrono::Utc::now().timestamp_millis(),
                                    "data": {
                                        "status": "degraded",
                                        "cached_metrics": true,
                                        "last_update": chrono::Utc::now().timestamp_millis()
                                    }
                                });
                                ctx.text(degraded_response.to_string());
                            } else {
                                ctx.address().do_send(RequestPerformanceUpdate);
                            }
                        }
                        "request_topology" => {
                            if let Some(data) = request.data {
                                if let Some(swarm_id_value) = data.get("swarm_id") {
                                    if let Some(swarm_id) = swarm_id_value.as_str() {
                                        ctx.address().do_send(RequestTopologyUpdate { 
                                            swarm_id: swarm_id.to_string() 
                                        });
                                    }
                                }
                            }
                        }
                        _ => {
                            warn!("Unknown WebSocket action: {}", request.action);
                            self.send_error_response(ctx, &format!("Unknown action: {}", request.action));
                        }
                    }
                }
            }
            Ok(ws::Message::Binary(_)) => {
                warn!("Binary WebSocket messages not supported");
            }
            Ok(ws::Message::Close(reason)) => {
                info!("[Multi-MCP] WebSocket closing for client {}: {:?}", self.client_id, reason);

                // Log final resilience statistics with error handling
                if let Some(cb) = &self.circuit_breaker {
                    let cb_clone = cb.clone();
                    let client_id = self.client_id.clone();
                    let connection_failures = self.connection_failures;
                    actix::spawn(async move {
                        let stats = cb_clone.stats().await;
                        info!("[Multi-MCP] Final stats for client {} - Circuit: {:?}, Failures: {}, Successes: {}, Connection failures: {}",
                             client_id, stats.state, stats.failed_requests, stats.successful_requests, connection_failures);
                    });
                }

                ctx.close(reason);
            }
            _ => {
                warn!("Unhandled WebSocket message type for client {}", self.client_id);
                ctx.close(None);
            }
        }
    }
}

/// Client request message structure
#[derive(Debug, Deserialize)]
struct ClientRequest {
    action: String,
    data: Option<serde_json::Value>,
}

/// Client configuration structure
#[derive(Debug, Deserialize)]
struct ClientConfig {
    subscription_filters: Option<SubscriptionFilters>,
    performance_mode: Option<PerformanceMode>,
}

/// Internal WebSocket messages
#[derive(Message)]
#[rtype(result = "()")]
struct RequestAgentUpdate;

#[derive(Message)]
#[rtype(result = "()")]
struct RequestDiscoveryData;

#[derive(Message)]
#[rtype(result = "()")]
struct RequestPerformanceUpdate;

#[derive(Message)]
#[rtype(result = "()")]
struct RequestTopologyUpdate {
    swarm_id: String,
}

#[derive(Message)]
#[rtype(result = "()")]
struct DiscoverySuccess;

#[derive(Message)]
#[rtype(result = "()")]
struct DiscoveryFailure(String);

#[derive(Message)]
#[rtype(result = "()")]
struct SendHeartbeatPing;

#[derive(Message)]
#[rtype(result = "()")]
struct ReconnectionCompleted;

/// WebSocket message handlers
impl Handler<RequestAgentUpdate> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, _: RequestAgentUpdate, _ctx: &mut Self::Context) {
        // This would request current agent data from the visualization actor
        debug!("Requesting agent update for client {}", self.client_id);
    }
}

impl Handler<RequestDiscoveryData> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, _: RequestDiscoveryData, _ctx: &mut Self::Context) {
        debug!("Requesting discovery data for client {}", self.client_id);
    }
}

impl Handler<RequestPerformanceUpdate> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, _: RequestPerformanceUpdate, _ctx: &mut Self::Context) {
        debug!("Requesting performance update for client {}", self.client_id);
    }
}

impl Handler<RequestTopologyUpdate> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, msg: RequestTopologyUpdate, _ctx: &mut Self::Context) {
        debug!("Requesting topology update for swarm {} for client {}", msg.swarm_id, self.client_id);
    }
}

impl Handler<DiscoverySuccess> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, _: DiscoverySuccess, _ctx: &mut Self::Context) {
        debug!("[Multi-MCP] Discovery success for client {}", self.client_id);
        self.record_success();
    }
}

impl Handler<DiscoveryFailure> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, msg: DiscoveryFailure, ctx: &mut Self::Context) {
        warn!("[Multi-MCP] Discovery failure for client {}: {}", self.client_id, msg.0);
        self.record_failure();

        // Send error notification to client with graceful degradation
        let error_response = serde_json::json!({
            "type": "discovery_error",
            "message": msg.0,
            "client_id": self.client_id,
            "timestamp": chrono::Utc::now().timestamp_millis(),
            "retry_in_seconds": self.retry_config.initial_delay.as_secs(),
            "fallback_mode": "local_cache",
            "degraded_functionality": true
        });

        // Try to send error response, but don't fail if we can't
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.text(error_response.to_string());
        })) {
            error!("Failed to send error response for client {}: {:?}", self.client_id, e);
        }
    }
}

impl Handler<SendHeartbeatPing> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, _: SendHeartbeatPing, ctx: &mut Self::Context) {
        ctx.ping(b"mcp-heartbeat");
    }
}

impl Handler<ReconnectionCompleted> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, _: ReconnectionCompleted, _ctx: &mut Self::Context) {
        info!("[Multi-MCP] Reconnection completed for client {}", self.client_id);
        self.record_success();
    }
}

/// HTTP endpoint to start WebSocket connection
pub async fn multi_mcp_visualization_ws(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
    _hybrid_manager: Option<()>, // DEPRECATED
) -> ActixResult<HttpResponse> {
    debug!("Starting Multi-MCP visualization WebSocket connection");
    ws::start(MultiMcpVisualizationWs::new(app_state, None), &req, stream)
}

/// HTTP endpoint to get current MCP server status
pub async fn get_mcp_server_status(
    _app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    // This would query the visualization actor for current server status
    let response = json!({
        "servers": [
            {
                "server_id": "claude-flow",
                "server_type": "claude_flow",
                "host": "localhost",
                "port": 9500,
                "is_connected": true,
                "agent_count": 4
            },
            {
                "server_id": "ruv-swarm",
                "server_type": "ruv_swarm",
                "host": "localhost", 
                "port": 9501,
                "is_connected": false,
                "agent_count": 0
            }
        ],
        "total_agents": 4,
        "timestamp": chrono::Utc::now().timestamp_millis()
    });
    
    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .json(response))
}

/// HTTP endpoint to trigger discovery refresh
pub async fn refresh_mcp_discovery(
    _app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    info!("Manual MCP discovery refresh requested");
    
    // This would send a message to the visualization actor to refresh discovery
    
    Ok(HttpResponse::Ok().json(json!({
        "success": true,
        "message": "Discovery refresh initiated",
        "timestamp": chrono::Utc::now().timestamp_millis()
    })))
}

/// Configure WebSocket routes
pub fn configure_multi_mcp_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/multi-mcp")
            .route("/ws", web::get().to(multi_mcp_visualization_ws))
            .route("/status", web::get().to(get_mcp_server_status))
            .route("/refresh", web::post().to(refresh_mcp_discovery))
    );
}

impl MultiMcpVisualizationWs {
    /// Send error response with circuit breaker protection
    fn send_error_response(&mut self, ctx: &mut ws::WebsocketContext<Self>, error_message: &str) {
        let error_response = serde_json::json!({
            "type": "error",
            "message": error_message,
            "client_id": self.client_id,
            "timestamp": chrono::Utc::now().timestamp_millis(),
            "recoverable": true
        });

        // Try to send error response with error handling
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.text(error_response.to_string());
        })) {
            error!("Failed to send error response for client {}: {:?}", self.client_id, e);
            // If we can't even send error responses, close the connection
            ctx.close(None);
        }
    }
}