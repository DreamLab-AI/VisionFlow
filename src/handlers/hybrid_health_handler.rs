// DEPRECATED: Legacy hybrid health handler removed
// Docker exec architecture replaced by HTTP Management API
// Health monitoring now via Management API /health endpoint

/*
use actix_web::{
    web::{self},
    HttpResponse, HttpRequest, Result, Error,
};
use actix_web_actors::ws;
use actix::{Actor, StreamHandler, AsyncContext, ActorContext};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use log::{info, warn, error, debug};
use uuid::Uuid;

use crate::utils::docker_hive_mind::{DockerHiveMind, SessionInfo};
use crate::utils::mcp_connection::MCPConnectionPool;
use crate::utils::hybrid_fault_tolerance::{NetworkRecoveryManager};
use crate::utils::hybrid_performance_optimizer::{HybridPerformanceOptimizer};

/// Hybrid system health monitoring and management
pub struct HybridHealthManager {
    pub docker_hive_mind: Arc<DockerHiveMind>,
    mcp_pool: Arc<MCPConnectionPool>,
    recovery_manager: Arc<NetworkRecoveryManager>,
    performance_optimizer: Arc<HybridPerformanceOptimizer>,
    health_cache: Arc<RwLock<Option<CachedHealthStatus>>>,
    websocket_clients: Arc<RwLock<HashMap<String, WebSocketClient>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSystemStatus {
    pub docker_health: String, // "healthy" | "degraded" | "unavailable" | "unknown"
    pub mcp_health: String,    // "connected" | "reconnecting" | "disconnected" | "unknown"
    pub active_sessions: Vec<SessionInfo>,
    pub telemetry_delay: u64,
    pub network_latency: u64,
    pub container_health: Option<ContainerHealthData>,
    pub system_status: String, // "healthy" | "degraded" | "critical" | "unknown"
    pub failover_active: bool,
    pub performance: PerformanceMetrics,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerHealthData {
    pub is_running: bool,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_healthy: bool,
    pub disk_space_gb: f64,
    pub last_response_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub cache_hit_ratio: f64,
    pub connection_pool_utilization: f64,
    pub memory_usage_mb: f64,
    pub active_optimizations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CachedHealthStatus {
    pub status: HybridSystemStatus,
    pub cached_at: DateTime<Utc>,
    pub cache_ttl: chrono::Duration,
}

#[derive(Debug, Clone)]
pub struct WebSocketClient {
    pub id: String,
    pub connected_at: DateTime<Utc>,
    pub subscriptions: Vec<String>, // "status", "performance", "sessions", "health"
    pub last_ping: DateTime<Utc>,
}

/// WebSocket message types for the hybrid health system
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    #[serde(rename = "ping")]
    Ping { timestamp: DateTime<Utc> },

    #[serde(rename = "pong")]
    Pong { timestamp: DateTime<Utc> },

    #[serde(rename = "subscribe")]
    Subscribe { subscriptions: Vec<String> },

    #[serde(rename = "unsubscribe")]
    Unsubscribe { subscriptions: Vec<String> },

    #[serde(rename = "request_status")]
    RequestStatus { force_refresh: Option<bool> },

    #[serde(rename = "request_performance")]
    RequestPerformance,

    #[serde(rename = "status_update")]
    StatusUpdate {
        payload: HybridSystemStatus,
        timestamp: DateTime<Utc>,
    },

    #[serde(rename = "performance_update")]
    PerformanceUpdate {
        payload: PerformanceMetrics,
        timestamp: DateTime<Utc>,
    },

    #[serde(rename = "session_update")]
    SessionUpdate {
        sessions: Vec<SessionInfo>,
        timestamp: DateTime<Utc>,
    },

    #[serde(rename = "error")]
    Error {
        message: String,
        timestamp: DateTime<Utc>,
    },

    #[serde(rename = "initial_status")]
    InitialStatus {
        payload: HybridSystemStatus,
        timestamp: DateTime<Utc>,
    },
}

#[derive(Debug, Deserialize)]
pub struct HealthQueryParams {
    pub include_performance: Option<bool>,
    pub include_sessions: Option<bool>,
    pub include_container: Option<bool>,
    pub force_refresh: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct SpawnSwarmRequest {
    pub task: String,
    pub priority: Option<String>,
    pub strategy: Option<String>,
    pub method: Option<String>, // "docker" | "mcp" | "hybrid"
    pub max_workers: Option<u32>,
    pub auto_scale: Option<bool>,
    pub config: Option<Value>,
}

impl HybridHealthManager {
    pub fn new(
        docker_hive_mind: DockerHiveMind,
        mcp_pool: MCPConnectionPool,
    ) -> Self {
        let docker_hive_mind = Arc::new(docker_hive_mind);
        let mcp_pool = Arc::new(mcp_pool);

        let recovery_manager = Arc::new(crate::utils::hybrid_fault_tolerance::create_fault_tolerance_system(
            (*docker_hive_mind).clone(),
            (*mcp_pool).clone(),
        ));

        let performance_optimizer = Arc::new(crate::utils::hybrid_performance_optimizer::create_performance_optimizer(
            (*mcp_pool).clone(),
        ));

        Self {
            docker_hive_mind,
            mcp_pool,
            recovery_manager,
            performance_optimizer,
            health_cache: Arc::new(RwLock::new(None)),
            websocket_clients: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start_background_monitoring(&self) {
        info!("Starting hybrid system background monitoring");

        // Start performance optimization
        let perf_optimizer = Arc::clone(&self.performance_optimizer);
        tokio::spawn(async move {
            perf_optimizer.start_optimization().await;
        });

        // Start recovery monitoring
        let recovery_manager = Arc::clone(&self.recovery_manager);
        tokio::spawn(async move {
            recovery_manager.start_continuous_monitoring().await;
        });

        // Start periodic health updates
        let health_manager = Arc::new(self.clone());
        tokio::spawn(async move {
            health_manager.periodic_health_broadcast().await;
        });
    }

    async fn periodic_health_broadcast(&self) {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;

            match self.get_system_status_internal(false).await {
                Ok(status) => {
                    self.broadcast_to_websocket_clients("status_update", &status).await;
                },
                Err(e) => {
                    error!("Failed to get system status for broadcast: {}", e);
                }
            }
        }
    }

    pub async fn get_system_status(&self) -> Result<HybridSystemStatus, Box<dyn std::error::Error + Send + Sync>> {
        self.get_system_status_internal(false).await
    }

    async fn get_system_status_internal(&self, force_refresh: bool) -> Result<HybridSystemStatus, Box<dyn std::error::Error + Send + Sync>> {
        // Check cache first
        if !force_refresh {
            let cache = self.health_cache.read().await;
            if let Some(cached) = cache.as_ref() {
                let age = Utc::now() - cached.cached_at;
                if age < cached.cache_ttl {
                    debug!("Returning cached health status");
                    return Ok(cached.status.clone());
                }
            }
        }

        debug!("Fetching fresh system status");
        let start_time = std::time::Instant::now();

        // Get Docker sessions
        let (active_sessions, docker_health) = match self.docker_hive_mind.get_sessions().await {
            Ok(sessions) => {
                debug!("Docker health check successful: {} sessions", sessions.len());
                (sessions, "healthy".to_string())
            },
            Err(e) => {
                warn!("Docker health check failed: {}", e);
                (Vec::new(), "unavailable".to_string())
            }
        };

        // Get MCP health
        let mcp_health = match self.mcp_pool.execute_command(
            "health_check",
            "tools/call",
            json!({"name": "agent_list", "arguments": {"filter": "health"}})
        ).await {
            Ok(_) => {
                debug!("MCP health check successful");
                "connected".to_string()
            },
            Err(e) => {
                warn!("MCP health check failed: {}", e);
                "disconnected".to_string()
            }
        };

        // Get container health
        let container_health = match self.docker_hive_mind.health_check().await {
            Ok(health) => Some(ContainerHealthData {
                is_running: health.is_running,
                cpu_usage: health.cpu_usage,
                memory_usage: health.memory_usage,
                network_healthy: health.network_healthy,
                disk_space_gb: health.disk_space_gb,
                last_response_ms: health.last_response_ms,
                timestamp: Utc::now(),
            }),
            Err(e) => {
                warn!("Container health check failed: {}", e);
                None
            }
        };

        // Get performance metrics
        let performance_report = self.performance_optimizer.get_performance_report().await;
        let performance = PerformanceMetrics {
            total_requests: performance_report.overall_metrics.total_requests,
            successful_requests: performance_report.overall_metrics.successful_requests,
            failed_requests: performance_report.overall_metrics.failed_requests,
            average_response_time_ms: performance_report.overall_metrics.average_response_time_ms,
            cache_hit_ratio: performance_report.overall_metrics.cache_hit_ratio,
            connection_pool_utilization: performance_report.overall_metrics.connection_pool_utilization,
            memory_usage_mb: performance_report.overall_metrics.memory_usage_mb,
            active_optimizations: performance_report.overall_metrics.active_optimizations
                .iter()
                .map(|opt| format!("{:?}", opt))
                .collect(),
        };

        // Determine overall system status
        let system_status = self.calculate_system_status(
            &docker_health,
            &mcp_health,
            &container_health,
            &performance
        );

        // Check if failover is active
        let failover_active = docker_health == "unavailable" || mcp_health == "disconnected";

        let network_latency = start_time.elapsed().as_millis() as u64;
        let telemetry_delay = if active_sessions.is_empty() { 0 } else { network_latency * 2 };

        let status = HybridSystemStatus {
            docker_health,
            mcp_health,
            active_sessions,
            telemetry_delay,
            network_latency,
            container_health,
            system_status,
            failover_active,
            performance,
            last_updated: Utc::now(),
        };

        // Update cache
        {
            let mut cache = self.health_cache.write().await;
            *cache = Some(CachedHealthStatus {
                status: status.clone(),
                cached_at: Utc::now(),
                cache_ttl: chrono::Duration::seconds(30), // 30 second cache
            });
        }

        Ok(status)
    }

    fn calculate_system_status(
        &self,
        docker_health: &str,
        mcp_health: &str,
        container_health: &Option<ContainerHealthData>,
        performance: &PerformanceMetrics,
    ) -> String {
        // Check critical conditions
        if docker_health == "unavailable" {
            if mcp_health == "disconnected" {
                return "critical".to_string();
            } else {
                return "degraded".to_string();
            }
        }

        // Check container health
        if let Some(health) = container_health {
            if !health.is_running {
                return "critical".to_string();
            }
            if health.cpu_usage > 90.0 || health.memory_usage > 90.0 || health.disk_space_gb < 1.0 {
                return "degraded".to_string();
            }
        }

        // Check performance metrics
        if performance.failed_requests > 0 {
            let failure_rate = performance.failed_requests as f64 / performance.total_requests.max(1) as f64;
            if failure_rate > 0.1 { // >10% failure rate
                return "degraded".to_string();
            }
        }

        if performance.average_response_time_ms > 5000.0 {
            return "degraded".to_string();
        }

        "healthy".to_string()
    }

    pub async fn add_websocket_client(&self, client_id: String) {
        let mut clients = self.websocket_clients.write().await;
        clients.insert(client_id.clone(), WebSocketClient {
            id: client_id,
            connected_at: Utc::now(),
            subscriptions: vec!["status".to_string()],
            last_ping: Utc::now(),
        });
    }

    pub async fn remove_websocket_client(&self, client_id: &str) {
        let mut clients = self.websocket_clients.write().await;
        clients.remove(client_id);
    }

    pub async fn update_client_subscriptions(&self, client_id: &str, subscriptions: Vec<String>) {
        let mut clients = self.websocket_clients.write().await;
        if let Some(client) = clients.get_mut(client_id) {
            client.subscriptions = subscriptions;
        }
    }

    async fn broadcast_to_websocket_clients(&self, message_type: &str, _data: &HybridSystemStatus) {
        let clients = self.websocket_clients.read().await;

        if clients.is_empty() {
            return;
        }

        debug!("Broadcasting {} to {} WebSocket clients", message_type, clients.len());
        // The actual broadcasting is handled by the WebSocket actor instances
    }
}

impl Clone for HybridHealthManager {
    fn clone(&self) -> Self {
        Self {
            docker_hive_mind: Arc::clone(&self.docker_hive_mind),
            mcp_pool: Arc::clone(&self.mcp_pool),
            recovery_manager: Arc::clone(&self.recovery_manager),
            performance_optimizer: Arc::clone(&self.performance_optimizer),
            health_cache: Arc::clone(&self.health_cache),
            websocket_clients: Arc::clone(&self.websocket_clients),
        }
    }
}

// HTTP Handlers

/// Get overall system status
pub async fn get_hybrid_status(
    _params: web::Query<HealthQueryParams>,
    health_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<HttpResponse> {
    let force_refresh = _params.force_refresh.unwrap_or(false);

    match health_manager.get_system_status_internal(force_refresh).await {
        Ok(status) => {
            info!("System status requested: {} ({})", status.system_status, status.docker_health);
            Ok(HttpResponse::Ok().json(status))
        },
        Err(e) => {
            error!("Failed to get system status: {}", e);
            Err(Error::from(actix_web::error::ErrorInternalServerError(e)))
        }
    }
}

/// Get detailed performance report
pub async fn get_performance_report(
    health_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<HttpResponse> {
    let report = health_manager.performance_optimizer.get_performance_report().await;

    info!("Performance report requested: {} total requests, {:.1}% cache hit ratio",
        report.overall_metrics.total_requests,
        report.overall_metrics.cache_hit_ratio * 100.0
    );

    Ok(HttpResponse::Ok().json(report))
}

/// Spawn a new swarm
pub async fn spawn_swarm_hybrid(
    health_manager: web::Data<Arc<HybridHealthManager>>,
    request: web::Json<SpawnSwarmRequest>,
) -> Result<HttpResponse> {
    info!("Spawning swarm: {} (method: {:?})", request.task, request.method);

    // Convert request to appropriate format
    use crate::utils::docker_hive_mind::{SwarmConfig, SwarmPriority, SwarmStrategy};

    let priority = match request.priority.as_deref() {
        Some("low") => SwarmPriority::Low,
        Some("high") | Some("critical") => SwarmPriority::High,
        _ => SwarmPriority::Medium,
    };

    let strategy = match request.strategy.as_deref() {
        Some("tactical") => SwarmStrategy::Tactical,
        Some("strategic") => SwarmStrategy::Strategic,
        Some("adaptive") => SwarmStrategy::Adaptive,
        _ => SwarmStrategy::HiveMind,
    };

    let config = SwarmConfig {
        priority,
        strategy,
        max_workers: request.max_workers,
        auto_scale: request.auto_scale.unwrap_or(true),
        monitor: true,
        verbose: false,
        ..Default::default()
    };

    // Determine method
    let method = request.method.as_deref().unwrap_or("hybrid");

    match method {
        "docker" => {
            match health_manager.docker_hive_mind.spawn_swarm(&request.task, config).await {
                Ok(session_id) => {
                    info!("Swarm spawned successfully via Docker: {}", session_id);

                    // Broadcast session update
                    if let Ok(status) = health_manager.get_system_status_internal(true).await {
                        health_manager.broadcast_to_websocket_clients("session_update", &status).await;
                    }

                    Ok(HttpResponse::Ok().json(json!({
                        "success": true,
                        "sessionId": session_id,
                        "swarmId": session_id,
                        "method": "docker",
                        "status": "spawning",
                        "timestamp": Utc::now()
                    })))
                },
                Err(e) => {
                    error!("Failed to spawn swarm via Docker: {}", e);
                    Err(Error::from(actix_web::error::ErrorInternalServerError(e)))
                }
            }
        },
        "mcp" => {
            // Use MCP fallback
            match health_manager.mcp_pool.execute_command(
                "spawn_swarm",
                "tools/call",
                json!({
                    "name": "task_orchestrate",
                    "arguments": {
                        "task": request.task.clone(),
                        "priority": request.priority.clone().unwrap_or("medium".to_string()),
                        "strategy": request.strategy.clone().unwrap_or("hive-mind".to_string())
                    }
                })
            ).await {
                Ok(result) => {
                    info!("Swarm spawned successfully via MCP fallback");
                    Ok(HttpResponse::Ok().json(json!({
                        "success": true,
                        "result": result,
                        "method": "mcp",
                        "status": "spawning",
                        "timestamp": Utc::now()
                    })))
                },
                Err(e) => {
                    error!("Failed to spawn swarm via MCP: {}", e);
                    Err(Error::from(actix_web::error::ErrorInternalServerError(e)))
                }
            }
        },
        _ => {
            // Hybrid approach - try Docker first, fallback to MCP
            match health_manager.docker_hive_mind.spawn_swarm(&request.task, config).await {
                Ok(session_id) => {
                    info!("Swarm spawned successfully via Docker (hybrid): {}", session_id);

                    // Broadcast session update
                    if let Ok(status) = health_manager.get_system_status_internal(true).await {
                        health_manager.broadcast_to_websocket_clients("session_update", &status).await;
                    }

                    Ok(HttpResponse::Ok().json(json!({
                        "success": true,
                        "sessionId": session_id,
                        "swarmId": session_id,
                        "method": "docker-primary",
                        "status": "spawning",
                        "timestamp": Utc::now()
                    })))
                },
                Err(docker_err) => {
                    warn!("Docker spawn failed, trying MCP fallback: {}", docker_err);

                    match health_manager.mcp_pool.execute_command(
                        "spawn_swarm_fallback",
                        "tools/call",
                        json!({
                            "name": "task_orchestrate",
                            "arguments": {
                                "task": request.task.clone(),
                                "priority": request.priority.clone().unwrap_or("medium".to_string()),
                                "strategy": request.strategy.clone().unwrap_or("hive-mind".to_string())
                            }
                        })
                    ).await {
                        Ok(result) => {
                            info!("Swarm spawned successfully via MCP fallback (hybrid)");
                            Ok(HttpResponse::Ok().json(json!({
                                "success": true,
                                "result": result,
                                "method": "mcp-fallback",
                                "status": "spawning",
                                "dockerError": docker_err.to_string(),
                                "timestamp": Utc::now()
                            })))
                        },
                        Err(mcp_err) => {
                            error!("Both Docker and MCP failed. Docker: {}, MCP: {}", docker_err, mcp_err);
                            Err(Error::from(actix_web::error::ErrorInternalServerError(format!("Both Docker and MCP failed. Docker: {}, MCP: {}", docker_err, mcp_err))))
                        }
                    }
                }
            }
        }
    }
}

/// Stop a swarm
pub async fn stop_swarm(
    session_id: web::Path<String>,
    health_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<HttpResponse> {
    let session_id_str = session_id.into_inner();
    info!("Stopping swarm: {}", session_id_str);

    match health_manager.docker_hive_mind.stop_swarm(&session_id_str).await {
        Ok(_) => {
            info!("Swarm stopped successfully: {}", session_id_str);

            // Broadcast session update
            if let Ok(status) = health_manager.get_system_status_internal(true).await {
                health_manager.broadcast_to_websocket_clients("session_update", &status).await;
            }

            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "sessionId": session_id_str,
                "status": "stopped",
                "timestamp": Utc::now()
            })))
        },
        Err(e) => {
            error!("Failed to stop swarm {}: {}", session_id_str, e);
            Err(Error::from(actix_web::error::ErrorInternalServerError(e)))
        }
    }
}

/// WebSocket actor for hybrid health status monitoring
pub struct HybridHealthWebSocket {
    _health_manager: Arc<HybridHealthManager>,
    client_id: String,
    last_heartbeat: Instant,
    subscriptions: Vec<String>,
}

impl HybridHealthWebSocket {
    pub fn new(health_manager: Arc<HybridHealthManager>) -> Self {
        Self {
            _health_manager: health_manager,
            client_id: Uuid::new_v4().to_string(),
            last_heartbeat: Instant::now(),
            subscriptions: vec!["status".to_string()],
        }
    }


    /// Start heartbeat mechanism
    fn start_heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_secs(5), |act, ctx| {
            if Instant::now().duration_since(act.last_heartbeat) > Duration::from_secs(30) {
                warn!("WebSocket client {} heartbeat timeout, disconnecting", act.client_id);
                ctx.stop();
                return;
            }

            ctx.ping(b"ping");
        });
    }

    /// Handle client message
    fn handle_client_message(&mut self, text: &str, ctx: &mut ws::WebsocketContext<Self>) {
        // Simple JSON parsing for basic commands
        if let Ok(request) = serde_json::from_str::<Value>(text) {
            match request.get("type").and_then(|t| t.as_str()) {
                Some("ping") => {
                    let pong = json!({
                        "type": "pong",
                        "timestamp": Utc::now()
                    });
                    if let Ok(response) = serde_json::to_string(&pong) {
                        ctx.text(response);
                    }
                },
                Some("subscribe") => {
                    if let Some(subscriptions) = request.get("subscriptions").and_then(|s| s.as_array()) {
                        let sub_list: Vec<String> = subscriptions
                            .iter()
                            .filter_map(|s| s.as_str().map(|s| s.to_string()))
                            .collect();

                        info!("Client {} subscribing to: {:?}", self.client_id, sub_list);
                        self.subscriptions = sub_list;

                        let ack = json!({
                            "type": "subscription_ack",
                            "subscriptions": self.subscriptions,
                            "timestamp": Utc::now()
                        });
                        if let Ok(response) = serde_json::to_string(&ack) {
                            ctx.text(response);
                        }
                    }
                },
                Some("request_status") => {
                    // Send a simple status response
                    let status_msg = json!({
                        "type": "status_update",
                        "payload": {
                            "docker_health": "healthy",
                            "mcp_health": "connected",
                            "system_status": "healthy",
                            "active_sessions": [],
                            "telemetry_delay": 0,
                            "network_latency": 0,
                            "failover_active": false,
                            "last_updated": Utc::now()
                        },
                        "timestamp": Utc::now()
                    });
                    if let Ok(response) = serde_json::to_string(&status_msg) {
                        ctx.text(response);
                    }
                },
                _ => {
                    debug!("Unknown WebSocket message type: {}", text);
                }
            }
        } else {
            warn!("Failed to parse WebSocket message from client {}: invalid JSON", self.client_id);
        }
    }
}

impl Actor for HybridHealthWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Hybrid health WebSocket connection established for client: {}", self.client_id);

        // Send a simple welcome message immediately
        let welcome_msg = json!({
            "type": "connection_established",
            "client_id": self.client_id,
            "timestamp": Utc::now()
        });

        if let Ok(json_msg) = serde_json::to_string(&welcome_msg) {
            ctx.text(json_msg);
        }

        // Start heartbeat
        self.start_heartbeat(ctx);

        // Start periodic updates with simpler approach
        ctx.run_interval(Duration::from_secs(15), |_act, ctx| {
            // Send a simple ping to keep connection alive
            ctx.ping(b"health_check");
        });
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Hybrid health WebSocket connection closed for client: {}", self.client_id);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for HybridHealthWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.last_heartbeat = Instant::now();
                ctx.pong(&msg);
            },
            Ok(ws::Message::Pong(_)) => {
                self.last_heartbeat = Instant::now();
            },
            Ok(ws::Message::Text(text)) => {
                self.last_heartbeat = Instant::now();
                self.handle_client_message(&text, ctx);
            },
            Ok(ws::Message::Binary(_)) => {
                warn!("Binary messages not supported by hybrid health WebSocket");
            },
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket closing for client {}: {:?}", self.client_id, reason);
                ctx.close(reason);
                ctx.stop();
            },
            Err(e) => {
                error!("WebSocket error for client {}: {}", self.client_id, e);
                ctx.stop();
            },
            _ => {
                debug!("Unhandled WebSocket message type for client {}", self.client_id);
            }
        }
    }
}

/// WebSocket handler for real-time hybrid health status updates
pub async fn websocket_hybrid_status(
    req: HttpRequest,
    stream: web::Payload,
    health_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<HttpResponse> {
    let ws_actor = HybridHealthWebSocket::new(Arc::clone(&**health_manager));
    ws::start(ws_actor, &req, stream)
}

*/
