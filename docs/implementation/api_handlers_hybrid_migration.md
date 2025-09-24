# API Handlers Hybrid Migration Guide

## Overview

This document outlines the migration strategy for API handlers to use the hybrid Docker exec + TCP/MCP architecture. The goal is to separate control plane operations (task creation) from data plane operations (telemetry streaming) while maintaining backward compatibility.

## Current Handler Architecture

### Problematic Patterns

```rust
// mcp_relay_handler.rs - Creates isolated TCP connections per client
this.mcpProcess = spawn(mcpCommand, mcpArgs, {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: '/workspace',
    env: devEnv
});

// multi_mcp_websocket_handler.rs - Each WebSocket creates separate MCP process
self.circuit_breaker.execute(async {
    // Creates new isolated connection
    let connection_result = connect_to_mcp().await;
})
```

### Issues:
1. **Process Multiplication**: Each client spawns separate MCP processes
2. **State Isolation**: Tasks exist only within specific connections
3. **Resource Waste**: Duplicate memory and CPU usage per client
4. **Debugging Nightmare**: Multiple processes make troubleshooting difficult

## Migration Strategy by Handler

### 1. MCP Relay Handler Migration

**File**: `/workspace/ext/src/handlers/mcp_relay_handler.rs`

#### 1.1 Replace Process Spawning with Docker Exec

```rust
// OLD: Spawn separate MCP process per connection (lines 42-48)
this.mcpProcess = spawn(mcpCommand, mcpArgs, {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: '/workspace',
    env: devEnv
});

// NEW: Use shared Docker hive mind for task operations
use crate::utils::docker_hive_mind::{create_docker_hive_mind, SwarmConfig, SwarmPriority};

pub struct MCPRelayActor {
    client_id: String,
    docker_hive_mind: Arc<DockerHiveMind>,  // Shared instance
    mcp_telemetry_pool: Option<Arc<MCPConnectionPool>>,  // For telemetry only
    // ... rest of fields
}

impl MCPRelayActor {
    fn new() -> Self {
        let docker_hive_mind = Arc::new(create_docker_hive_mind());

        // Optional MCP pool for telemetry (not task creation)
        let mcp_telemetry_pool = if should_enable_telemetry() {
            Some(Arc::new(MCPConnectionPool::new("multi-agent-container".to_string(), "9500".to_string())))
        } else {
            None
        };

        Self {
            client_id: uuid::Uuid::new_v4().to_string(),
            docker_hive_mind,
            mcp_telemetry_pool,
            // ... other fields
        }
    }

    async fn handle_task_creation(&self, request: TaskRequest) -> Result<Value, RelayError> {
        info!("[MCP Relay] Creating task via Docker exec: {}", request.task);

        let config = SwarmConfig {
            priority: match request.priority.as_deref() {
                Some("high") | Some("critical") => SwarmPriority::High,
                Some("low") => SwarmPriority::Low,
                _ => SwarmPriority::Medium,
            },
            strategy: crate::utils::docker_hive_mind::SwarmStrategy::HiveMind,
            auto_scale: true,
            monitor: true,
            verbose: request.verbose.unwrap_or(false),
            ..Default::default()
        };

        match self.docker_hive_mind.spawn_swarm(&request.task, config).await {
            Ok(session_id) => {
                info!("[MCP Relay] Task created with session ID: {}", session_id);

                // Optionally get telemetry via MCP (non-blocking)
                let telemetry = if let Some(pool) = &self.mcp_telemetry_pool {
                    self.get_session_telemetry(pool, &session_id).await.unwrap_or_default()
                } else {
                    json!({})
                };

                Ok(json!({
                    "success": true,
                    "taskId": session_id,
                    "swarmId": session_id,
                    "method": "docker-exec",
                    "timestamp": chrono::Utc::now().timestamp_millis(),
                    "telemetry": telemetry
                }))
            }
            Err(e) => {
                error!("[MCP Relay] Docker task creation failed: {}", e);

                // Fallback to MCP if available
                if let Some(pool) = &self.mcp_telemetry_pool {
                    warn!("[MCP Relay] Attempting MCP fallback for task creation");
                    self.mcp_fallback_task_creation(pool, &request).await
                } else {
                    Err(RelayError::TaskCreationFailed(e.to_string()))
                }
            }
        }
    }

    async fn get_session_telemetry(&self, pool: &MCPConnectionPool, session_id: &str) -> Result<Value, RelayError> {
        let params = json!({
            "name": "task_status",
            "arguments": {
                "taskId": session_id
            }
        });

        // Non-blocking telemetry collection
        tokio::time::timeout(Duration::from_millis(500), async {
            pool.execute_command("telemetry", "tools/call", params).await
        })
        .await
        .unwrap_or_else(|_| {
            debug!("[MCP Relay] Telemetry timeout for session {}", session_id);
            Ok(json!({"status": "telemetry_unavailable"}))
        })
        .unwrap_or_default()
    }

    async fn mcp_fallback_task_creation(&self, pool: &MCPConnectionPool, request: &TaskRequest) -> Result<Value, RelayError> {
        let params = json!({
            "name": "task_orchestrate",
            "arguments": {
                "task": request.task,
                "priority": request.priority.as_deref().unwrap_or("medium"),
                "strategy": "balanced"
            }
        });

        match pool.execute_command("fallback", "tools/call", params).await {
            Ok(mut result) => {
                // Mark as fallback
                if let Some(obj) = result.as_object_mut() {
                    obj.insert("method".to_string(), json!("mcp-fallback"));
                    obj.insert("note".to_string(), json!("Docker exec failed, using MCP fallback"));
                }
                Ok(result)
            }
            Err(e) => Err(RelayError::FallbackFailed(e.to_string()))
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct TaskRequest {
    task: String,
    priority: Option<String>,
    verbose: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
enum RelayError {
    #[error("Task creation failed: {0}")]
    TaskCreationFailed(String),

    #[error("MCP fallback failed: {0}")]
    FallbackFailed(String),
}

fn should_enable_telemetry() -> bool {
    std::env::var("ENABLE_MCP_TELEMETRY")
        .map(|v| v.to_lowercase() == "true")
        .unwrap_or(true)  // Default to enabled
}
```

#### 1.2 Update WebSocket Message Handling

```rust
// Update handle method around line 270
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MCPRelayActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                debug!("[MCP Relay] Received text from client: {}", text);

                if let Ok(request) = serde_json::from_str::<serde_json::Value>(&text) {
                    // Route based on message type
                    if let Some(msg_type) = request.get("type").and_then(|t| t.as_str()) {
                        match msg_type {
                            "create_task" => {
                                if let Ok(task_req) = serde_json::from_value::<TaskRequest>(request) {
                                    let docker_hive_mind = Arc::clone(&self.docker_hive_mind);
                                    let mcp_pool = self.mcp_telemetry_pool.clone();
                                    let addr = ctx.address();

                                    // Handle task creation asynchronously
                                    tokio::spawn(async move {
                                        match Self::handle_task_creation_static(docker_hive_mind, mcp_pool, task_req).await {
                                            Ok(response) => {
                                                addr.do_send(SendResponse(response.to_string()));
                                            }
                                            Err(e) => {
                                                let error_response = json!({
                                                    "type": "error",
                                                    "message": e.to_string(),
                                                    "timestamp": chrono::Utc::now().timestamp_millis()
                                                });
                                                addr.do_send(SendResponse(error_response.to_string()));
                                            }
                                        }
                                    });
                                    return;
                                }
                            }
                            "get_status" => {
                                // Get status from Docker, enrich with MCP telemetry
                                self.handle_status_request(&request, ctx);
                                return;
                            }
                            "list_swarms" => {
                                self.handle_list_request(ctx);
                                return;
                            }
                            "ping" => {
                                ctx.text(json!({
                                    "type": "pong",
                                    "timestamp": chrono::Utc::now().timestamp_millis()
                                }).to_string());
                                return;
                            }
                            _ => {}
                        }
                    }
                }

                // Unknown message format, send error
                ctx.text(json!({
                    "type": "error",
                    "message": "Unknown message format",
                    "timestamp": chrono::Utc::now().timestamp_millis()
                }).to_string());
            }
            // ... handle other message types
        }
    }
}

// Message types for actor communication
#[derive(Message)]
#[rtype(result = "()")]
struct SendResponse(String);

impl Handler<SendResponse> for MCPRelayActor {
    type Result = ();

    fn handle(&mut self, msg: SendResponse, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}
```

### 2. Multi-MCP WebSocket Handler Migration

**File**: `/workspace/ext/src/handlers/multi_mcp_websocket_handler.rs`

#### 2.1 Replace Circuit Breaker MCP Calls with Hybrid Approach

```rust
// Update around line 516
"request_agents" => {
    // NEW: Hybrid approach - Docker primary, MCP telemetry secondary

    let docker_hive_mind = crate::utils::docker_hive_mind::create_docker_hive_mind();

    // Primary data from Docker (authoritative)
    let docker_future = async {
        match docker_hive_mind.get_sessions().await {
            Ok(sessions) => {
                json!({
                    "type": "multi_agent_update",
                    "source": "docker",
                    "swarms": sessions,
                    "timestamp": chrono::Utc::now().timestamp_millis()
                })
            }
            Err(e) => {
                warn!("Docker sessions unavailable: {}", e);
                json!({
                    "type": "error",
                    "source": "docker",
                    "message": e.to_string(),
                    "timestamp": chrono::Utc::now().timestamp_millis()
                })
            }
        }
    };

    // Secondary telemetry from MCP (optional enrichment)
    let mcp_future = async {
        if let Some(cb) = &self.circuit_breaker {
            let cb_clone = cb.clone();
            let stats = cb_clone.stats().await;

            match stats.state {
                crate::utils::network::CircuitBreakerState::Open => {
                    json!({
                        "mcp_status": "circuit_breaker_open",
                        "telemetry": null
                    })
                }
                _ => {
                    // Try to get MCP telemetry (with timeout)
                    match tokio::time::timeout(Duration::from_millis(200), async {
                        // This would be actual MCP telemetry call
                        json!({
                            "agent_metrics": {},
                            "performance": {}
                        })
                    }).await {
                        Ok(telemetry) => json!({
                            "mcp_status": "available",
                            "telemetry": telemetry
                        }),
                        Err(_) => json!({
                            "mcp_status": "timeout",
                            "telemetry": null
                        })
                    }
                }
            }
        } else {
            json!({
                "mcp_status": "disabled",
                "telemetry": null
            })
        }
    };

    // Execute both concurrently
    let ctx_addr = ctx.address();
    tokio::spawn(async move {
        let (docker_result, mcp_result) = tokio::join!(docker_future, mcp_future);

        // Merge results
        let mut response = docker_result;
        if let Some(response_obj) = response.as_object_mut() {
            response_obj.insert("mcp_telemetry".to_string(), mcp_result);
        }

        ctx_addr.do_send(BroadcastMessage(response.to_string()));
    });
}
```

#### 2.2 Optimize Discovery with Hybrid Data Sources

```rust
// Update send_discovery_data method around line 227
fn send_discovery_data(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
    let client_id = self.client_id.clone();
    let docker_hive_mind = crate::utils::docker_hive_mind::create_docker_hive_mind();

    // Check if we have healthy services before proceeding
    if !self.has_healthy_services() {
        warn!("[Multi-MCP] No healthy services available for discovery, client {}", client_id);
        ctx.text(json!({
            "type": "error",
            "message": "No healthy services available",
            "timestamp": chrono::Utc::now().timestamp_millis()
        }).to_string());
        return;
    }

    let addr = ctx.address();
    let retry_config = self.retry_config.clone();

    tokio::spawn(async move {
        // Primary discovery from Docker
        let docker_discovery = async {
            match docker_hive_mind.health_check().await {
                Ok(health) => json!({
                    "docker_health": health,
                    "container_status": "healthy"
                }),
                Err(e) => json!({
                    "docker_error": e.to_string(),
                    "container_status": "unhealthy"
                })
            }
        };

        // Secondary discovery from MCP servers
        let mcp_discovery = async {
            let mut mcp_servers = json!({
                "servers": []
            });

            // Discover MCP servers with timeout
            for server_name in ["claude-flow", "ruv-swarm", "flow-nexus"] {
                match tokio::time::timeout(Duration::from_millis(100), async {
                    // Lightweight MCP server ping
                    json!({
                        "server": server_name,
                        "status": "unknown"  // Would be actual health check
                    })
                }).await {
                    Ok(server_info) => {
                        mcp_servers["servers"].as_array_mut().unwrap().push(server_info);
                    }
                    Err(_) => {
                        mcp_servers["servers"].as_array_mut().unwrap().push(json!({
                            "server": server_name,
                            "status": "timeout"
                        }));
                    }
                }
            }

            mcp_servers
        };

        // Execute discovery concurrently
        let (docker_result, mcp_result) = tokio::join!(docker_discovery, mcp_discovery);

        let discovery_response = json!({
            "type": "discovery",
            "client_id": client_id,
            "docker": docker_result,
            "mcp": mcp_result,
            "timestamp": chrono::Utc::now().timestamp_millis()
        });

        addr.do_send(BroadcastMessage(discovery_response.to_string()));
    });
}

// New message type for broadcasting
#[derive(Message)]
#[rtype(result = "()")]
struct BroadcastMessage(String);

impl Handler<BroadcastMessage> for MultiMcpVisualizationWs {
    type Result = ();

    fn handle(&mut self, msg: BroadcastMessage, ctx: &mut Self::Context) {
        // Apply client filters before sending
        if let Ok(parsed_msg) = serde_json::from_str::<serde_json::Value>(&msg.0) {
            if let Some(msg_type) = parsed_msg.get("type").and_then(|t| t.as_str()) {
                if self.should_send_message(msg_type, &parsed_msg) {
                    let mut filtered_msg = parsed_msg.clone();
                    self.filter_agent_data(&mut filtered_msg);
                    ctx.text(filtered_msg.to_string());
                }
            }
        }
    }
}
```

### 3. Health Handler Migration

**File**: `/workspace/ext/src/handlers/mcp_health_handler.rs`

#### 3.1 Create Hybrid Health Endpoint

```rust
use crate::utils::docker_hive_mind::create_docker_hive_mind;
use actix_web::{web, HttpResponse, Result};
use serde_json::json;
use log::{info, warn};
use std::time::Duration;

pub async fn get_hybrid_health_status() -> Result<HttpResponse> {
    info!("Checking hybrid system health");

    let docker_hive_mind = create_docker_hive_mind();

    // Check Docker health (primary system)
    let docker_health_future = async {
        match docker_hive_mind.health_check().await {
            Ok(health) => json!({
                "status": "healthy",
                "details": health,
                "timestamp": chrono::Utc::now()
            }),
            Err(e) => json!({
                "status": "unhealthy",
                "error": e.to_string(),
                "timestamp": chrono::Utc::now()
            })
        }
    };

    // Check MCP services (telemetry systems)
    let mcp_health_future = async {
        let mut services = Vec::new();

        for service in ["tcp:9500", "websocket:3002", "health:9501"] {
            let service_health = match tokio::time::timeout(Duration::from_millis(500), async {
                // Would be actual service health checks
                match service {
                    "tcp:9500" => check_tcp_mcp_health().await,
                    "websocket:3002" => check_websocket_bridge_health().await,
                    "health:9501" => check_health_endpoint().await,
                    _ => false
                }
            }).await {
                Ok(healthy) => healthy,
                Err(_) => false,
            };

            services.push(json!({
                "service": service,
                "healthy": service_health,
                "timestamp": chrono::Utc::now()
            }));
        }

        json!({
            "mcp_services": services,
            "overall_mcp_health": services.iter()
                .all(|s| s.get("healthy").and_then(|h| h.as_bool()).unwrap_or(false))
        })
    };

    // Execute health checks concurrently
    let (docker_health, mcp_health) = tokio::join!(docker_health_future, mcp_health_future);

    let overall_healthy = docker_health.get("status")
        .and_then(|s| s.as_str())
        .map(|s| s == "healthy")
        .unwrap_or(false);

    let response = json!({
        "system": "hybrid_docker_mcp",
        "overall_status": if overall_healthy { "healthy" } else { "degraded" },
        "components": {
            "docker_hive_mind": docker_health,
            "mcp_telemetry": mcp_health
        },
        "note": "Docker failure is critical, MCP failure only affects telemetry",
        "timestamp": chrono::Utc::now()
    });

    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .json(response))
}

async fn check_tcp_mcp_health() -> bool {
    // Implement actual TCP MCP health check
    use tokio::net::TcpStream;
    tokio::time::timeout(Duration::from_millis(200),
        TcpStream::connect("multi-agent-container:9500")).await.is_ok()
}

async fn check_websocket_bridge_health() -> bool {
    // Implement actual WebSocket bridge health check
    true // Placeholder
}

async fn check_health_endpoint() -> bool {
    // Implement actual health endpoint check
    use reqwest::Client;
    let client = Client::new();
    tokio::time::timeout(Duration::from_millis(200),
        client.get("http://localhost:9501/health").send()).await
        .map(|result| result.map(|response| response.status().is_success()).unwrap_or(false))
        .unwrap_or(false)
}
```

## Configuration Changes

### Environment Variables

Add configuration for hybrid mode:

```bash
# Docker Hive Mind Configuration
MULTI_AGENT_CONTAINER_NAME=multi-agent-container
CLAUDE_FLOW_PATH=/app/node_modules/.bin/claude-flow

# Hybrid Mode Settings
ENABLE_MCP_TELEMETRY=true
ENABLE_DOCKER_FALLBACK=true
MCP_TELEMETRY_TIMEOUT_MS=500
DOCKER_COMMAND_TIMEOUT_MS=30000

# Circuit Breaker Settings
MCP_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
MCP_CIRCUIT_BREAKER_TIMEOUT_MS=60000
MCP_CIRCUIT_BREAKER_RETRY_DELAY_MS=1000
```

### Feature Flags

Implement gradual rollout with feature flags:

```rust
pub struct HybridConfig {
    pub enable_docker_primary: bool,
    pub enable_mcp_telemetry: bool,
    pub enable_mcp_fallback: bool,
    pub docker_timeout_ms: u64,
    pub mcp_telemetry_timeout_ms: u64,
}

impl HybridConfig {
    pub fn from_env() -> Self {
        Self {
            enable_docker_primary: std::env::var("ENABLE_DOCKER_PRIMARY")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(true),
            enable_mcp_telemetry: std::env::var("ENABLE_MCP_TELEMETRY")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(true),
            enable_mcp_fallback: std::env::var("ENABLE_MCP_FALLBACK")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(true),
            docker_timeout_ms: std::env::var("DOCKER_COMMAND_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30000),
            mcp_telemetry_timeout_ms: std::env::var("MCP_TELEMETRY_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(500),
        }
    }
}
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio_test]
    async fn test_hybrid_task_creation() {
        // Mock Docker hive mind
        let mock_hive_mind = MockDockerHiveMind::new();
        mock_hive_mind.expect_spawn_swarm()
            .returning(|_, _| Ok("test-session-123".to_string()));

        let relay = MCPRelayActor::with_mock_hive_mind(mock_hive_mind);

        let request = TaskRequest {
            task: "Test task".to_string(),
            priority: Some("high".to_string()),
            verbose: Some(false),
        };

        let result = relay.handle_task_creation(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response["method"], "docker-exec");
        assert_eq!(response["taskId"], "test-session-123");
    }

    #[tokio_test]
    async fn test_mcp_fallback_on_docker_failure() {
        let mock_hive_mind = MockDockerHiveMind::new();
        mock_hive_mind.expect_spawn_swarm()
            .returning(|_, _| Err("Docker daemon unreachable".into()));

        let mock_mcp_pool = MockMCPConnectionPool::new();
        mock_mcp_pool.expect_execute_command()
            .returning(|_, _, _| Ok(json!({"taskId": "mcp-fallback-456"})));

        let relay = MCPRelayActor::with_mocks(mock_hive_mind, Some(mock_mcp_pool));

        let request = TaskRequest {
            task: "Test task".to_string(),
            priority: Some("medium".to_string()),
            verbose: Some(false),
        };

        let result = relay.handle_task_creation(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response["method"], "mcp-fallback");
        assert_eq!(response["taskId"], "mcp-fallback-456");
    }
}
```

### Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_web::test]
    async fn test_hybrid_health_endpoint() {
        let app = test::init_service(
            App::new()
                .route("/health/hybrid", web::get().to(get_hybrid_health_status))
        ).await;

        let req = test::TestRequest::get()
            .uri("/health/hybrid")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let body: serde_json::Value = test::read_body_json(resp).await;
        assert!(body.get("system").is_some());
        assert!(body.get("overall_status").is_some());
    }
}
```

## Migration Checklist

### Pre-Migration
- [ ] Docker container is healthy and accessible
- [ ] `docker_hive_mind.rs` module is implemented and tested
- [ ] Environment variables are configured
- [ ] Feature flags are set for gradual rollout

### Handler Migration
- [ ] `mcp_relay_handler.rs` migrated to hybrid approach
- [ ] `multi_mcp_websocket_handler.rs` updated for dual data sources
- [ ] `mcp_health_handler.rs` extended with hybrid health checks
- [ ] All handlers use shared Docker hive mind instance

### Testing & Validation
- [ ] Unit tests pass for all hybrid functionality
- [ ] Integration tests verify Docker exec task creation
- [ ] Load testing confirms improved performance
- [ ] Fallback mechanisms tested with simulated failures

### Monitoring & Observability
- [ ] Metrics collection for Docker vs MCP usage
- [ ] Error rate monitoring for hybrid operations
- [ ] Performance dashboards updated for new architecture
- [ ] Alert thresholds adjusted for hybrid mode

This migration strategy ensures a smooth transition to the hybrid architecture while maintaining backward compatibility and providing robust fallback mechanisms.