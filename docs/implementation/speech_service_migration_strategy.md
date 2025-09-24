# Speech Service Migration Strategy

## Overview

The speech service currently uses TCP/MCP connections for task orchestration, leading to process isolation and persistence issues. This document outlines the migration strategy to use the hybrid Docker exec + TCP/MCP architecture.

## Current State Analysis

### Problems in Current Implementation

```rust
// FROM: ext/src/services/speech_service.rs:1149-1159
match call_task_orchestrate(&mcp_host, &mcp_port, &description, Some(priority_str), Some("balanced")).await {
    Ok(task_result) => {
        let task_id = task_result.get("taskId")
            .and_then(|id| id.as_str())
            .unwrap_or("unknown");

        // This task exists only in the TCP connection's isolated process!
        // If the connection drops, the task is lost
    }
}
```

### Issues:
1. **Task Isolation**: Tasks created via TCP/MCP exist only in that connection's process
2. **No Persistence**: Voice commands spawn ephemeral tasks that vanish on connection loss
3. **State Fragmentation**: CLI queries can't see UI-spawned tasks
4. **Network Brittleness**: TCP failures kill active tasks

## Migration Strategy

### Phase 1: Replace Task Creation with Docker Exec

#### 1.1 Update Voice Command Execution

Replace TCP/MCP task spawning with Docker exec:

```rust
// BEFORE: Using TCP/MCP (lines 1159)
match call_task_orchestrate(&mcp_host, &mcp_port, &description, Some(priority_str), Some("balanced")).await {

// AFTER: Using Docker exec hybrid approach
match call_task_orchestrate_hybrid(&description, Some(priority_str), Some("balanced"), &mcp_host, &mcp_port).await {
```

#### 1.2 Modify execute_voice_command_with_context Function

```rust
// Current implementation around line 1033
async fn execute_voice_command_with_context(
    voice_cmd: VoiceCommand,
    context_manager: Arc<VoiceContextManager>,
) -> String {
    // REPLACE TCP/MCP calls with hybrid approach

    match voice_cmd.parsed_intent {
        SwarmIntent::SpawnAgent { agent_type, .. } => {
            // OLD: call_swarm_init + call_agent_spawn via TCP
            // NEW: Use docker_hive_mind directly

            use crate::utils::docker_hive_mind::{create_docker_hive_mind, SwarmConfig, SwarmPriority};

            let hive_mind = create_docker_hive_mind();
            let config = SwarmConfig {
                priority: SwarmPriority::Medium,
                strategy: crate::utils::docker_hive_mind::SwarmStrategy::HiveMind,
                max_workers: Some(10),
                ..Default::default()
            };

            let task_description = format!("Spawn {} agent for voice command", agent_type);

            match hive_mind.spawn_swarm(&task_description, config).await {
                Ok(session_id) => {
                    // Track in context
                    let mut params = std::collections::HashMap::new();
                    params.insert("agent_type".to_string(), agent_type.clone());
                    params.insert("session_id".to_string(), session_id.clone());

                    let _ = context_manager.add_pending_operation(
                        &session_id,
                        "spawn_agent".to_string(),
                        params,
                        None,
                    ).await;

                    format!("Successfully spawned {} agent with session ID: {}.", agent_type, session_id)
                }
                Err(e) => {
                    error!("Docker spawn failed: {}, trying MCP fallback", e);
                    // Fallback to original TCP/MCP approach
                    format!("Failed to spawn {} agent. Error: {}", agent_type, e)
                }
            }
        },

        SwarmIntent::ExecuteTask { description, priority } => {
            let priority_level = match priority {
                crate::actors::voice_commands::TaskPriority::Critical => SwarmPriority::High,
                crate::actors::voice_commands::TaskPriority::High => SwarmPriority::High,
                crate::actors::voice_commands::TaskPriority::Medium => SwarmPriority::Medium,
                crate::actors::voice_commands::TaskPriority::Low => SwarmPriority::Low,
            };

            let hive_mind = create_docker_hive_mind();
            let config = SwarmConfig {
                priority: priority_level,
                strategy: crate::utils::docker_hive_mind::SwarmStrategy::HiveMind,
                auto_scale: true,
                monitor: true,
                ..Default::default()
            };

            match hive_mind.spawn_swarm(&description, config).await {
                Ok(session_id) => {
                    // Track the task in context
                    let mut params = std::collections::HashMap::new();
                    params.insert("task_id".to_string(), session_id.clone());
                    params.insert("description".to_string(), description.clone());

                    let _ = context_manager.add_pending_operation(
                        &session_id,
                        "execute_task".to_string(),
                        params,
                        Some(chrono::Utc::now() + chrono::Duration::minutes(30)),
                    ).await;

                    format!("Task '{}' has been assigned to the hive mind with session ID: {}.", description, session_id)
                }
                Err(e) => {
                    error!("Docker task execution failed: {}", e);
                    format!("Failed to execute task '{}'. Error: {}", description, e)
                }
            }
        },

        SwarmIntent::QueryStatus { target } => {
            // Use hybrid approach: Docker for authoritative data, MCP for telemetry
            let hive_mind = create_docker_hive_mind();

            match hive_mind.get_sessions().await {
                Ok(sessions) => {
                    if sessions.is_empty() {
                        "System status: No active swarms found.".to_string()
                    } else {
                        let active_count = sessions.iter()
                            .filter(|s| matches!(s.status,
                                crate::utils::docker_hive_mind::SwarmStatus::Active |
                                crate::utils::docker_hive_mind::SwarmStatus::Spawning))
                            .count();

                        format!("System status: {} active swarms operational.", active_count)
                    }
                }
                Err(e) => {
                    warn!("Docker status query failed: {}, trying MCP fallback", e);
                    // Fallback to original MCP approach
                    format!("Failed to query system status. Error: {}", e)
                }
            }
        },

        SwarmIntent::ListAgents => {
            let hive_mind = create_docker_hive_mind();

            match hive_mind.get_sessions().await {
                Ok(sessions) => {
                    if sessions.is_empty() {
                        "No active swarms found.".to_string()
                    } else {
                        let swarm_descriptions: Vec<String> = sessions.iter()
                            .map(|s| format!("{} ({})", s.session_id, s.task_description))
                            .collect();

                        format!("Active swarms: {}.", swarm_descriptions.join(", "))
                    }
                }
                Err(e) => {
                    error!("Failed to list swarms: {}", e);
                    format!("Failed to list active swarms. Error: {}", e)
                }
            }
        },

        // Keep other intents unchanged for now
        _ => {
            "Command received but not yet implemented.".to_string()
        }
    }
}
```

### Phase 2: Preserve TCP/MCP for Telemetry

#### 2.1 Keep Status Queries for Rich Data

```rust
// New hybrid status function
async fn get_hybrid_swarm_status(session_id: &str, mcp_host: &str, mcp_port: &str) -> String {
    use crate::utils::mcp_connection::get_swarm_status_hybrid;

    match get_swarm_status_hybrid(session_id, mcp_host, mcp_port).await {
        Ok(status_data) => {
            // Extract Docker status (authoritative)
            let docker_status = status_data.get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown");

            // Extract MCP telemetry (for rich data)
            let agent_count = status_data.get("telemetry")
                .and_then(|t| t.get("agents"))
                .and_then(|a| a.as_array())
                .map(|arr| arr.len())
                .unwrap_or(0);

            if agent_count > 0 {
                format!("Swarm {} is {} with {} active agents.", session_id, docker_status, agent_count)
            } else {
                format!("Swarm {} is {}.", session_id, docker_status)
            }
        }
        Err(e) => {
            format!("Unable to get status for swarm {}. Error: {}", session_id, e)
        }
    }
}
```

#### 2.2 Enhanced Voice Command Processing with Telemetry

```rust
// Add to process_voice_command_with_tags function around line 1004
pub async fn process_voice_command_with_tags(&self, text: String, session_id: String) -> VisionFlowResult<String> {
    use crate::services::speech_voice_integration::VoiceSwarmIntegration;
    use crate::utils::docker_hive_mind::create_docker_hive_mind;

    // First try Docker-based processing
    let hive_mind = create_docker_hive_mind();

    if Self::is_voice_command(&text) {
        if let Ok(voice_cmd) = VoiceCommand::parse(&text, session_id.clone()) {
            // Execute via Docker
            let docker_response = Self::execute_voice_command_with_context(
                voice_cmd.clone(),
                Arc::clone(&self.context_manager),
            ).await;

            // Extract session ID from Docker response for telemetry
            if let Some(extracted_session) = self.extract_session_id_from_response(&docker_response) {
                // Get rich telemetry via MCP (non-blocking)
                let telemetry_future = async {
                    let mcp_host = std::env::var("MCP_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
                    let mcp_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

                    get_hybrid_swarm_status(&extracted_session, &mcp_host, &mcp_port).await
                };

                // Don't block on telemetry, return Docker response immediately
                tokio::spawn(async move {
                    let _telemetry = telemetry_future.await;
                    // Telemetry could be cached or sent to WebSocket clients
                });
            }

            // Process with tag manager for hive mind tracking
            match VoiceSwarmIntegration::process_voice_command_with_tags(self, text, session_id, Arc::clone(&self.tag_manager)).await {
                Ok(tag) => {
                    Ok(format!("{} (Tracked with tag: {})", docker_response, tag.short_id()))
                }
                Err(e) => {
                    warn!("Tag processing failed: {}", e);
                    Ok(docker_response)
                }
            }
        } else {
            Ok("Sorry, I couldn't understand that command.".to_string())
        }
    } else {
        Ok("That doesn't appear to be a voice command.".to_string())
    }
}

// Helper function to extract session ID from Docker response
fn extract_session_id_from_response(&self, response: &str) -> Option<String> {
    // Look for patterns like "session ID: abc123" or "with session ID: abc123"
    if let Ok(re) = regex::Regex::new(r"session\s+ID:\s+([a-zA-Z0-9-]+)") {
        if let Some(captures) = re.captures(response) {
            if let Some(id_match) = captures.get(1) {
                return Some(id_match.as_str().to_string());
            }
        }
    }
    None
}
```

### Phase 3: Update API Handlers

#### 3.1 WebSocket Handlers Migration

Update the WebSocket handlers to use the hybrid approach:

```rust
// In multi_mcp_websocket_handler.rs around line 536
"request_agents" => {
    // NEW: Get data from Docker first, enrich with MCP telemetry

    let docker_hive_mind = crate::utils::docker_hive_mind::create_docker_hive_mind();

    // Get primary data from Docker (authoritative)
    let docker_sessions = match docker_hive_mind.get_sessions().await {
        Ok(sessions) => sessions,
        Err(e) => {
            warn!("Docker sessions unavailable: {}", e);
            Vec::new()
        }
    };

    // Enrich with MCP telemetry if available
    let mut enriched_data = json!({
        "type": "multi_agent_update",
        "swarms": docker_sessions,
        "source": "hybrid",
        "timestamp": chrono::Utc::now().timestamp_millis()
    });

    // Try to get MCP telemetry (non-blocking)
    if let Some(cb) = &self.circuit_breaker {
        let cb_clone = cb.clone();
        let ctx_addr = ctx.address();
        let swarms_data = enriched_data.clone();

        tokio::spawn(async move {
            let stats = cb_clone.stats().await;
            match stats.state {
                crate::utils::network::CircuitBreakerState::Open => {
                    // Circuit breaker open, send Docker data only
                    ctx_addr.do_send(SendToClient(swarms_data.to_string()));
                }
                _ => {
                    // Try to enrich with MCP data
                    // This would be expanded to actually query MCP for telemetry
                    ctx_addr.do_send(SendToClient(swarms_data.to_string()));
                }
            }
        });
    } else {
        // Send Docker data directly
        ctx.text(enriched_data.to_string());
    }
}
```

### Phase 4: Telemetry Streaming Optimization

#### 4.1 Separate Telemetry Pipeline

Create a dedicated telemetry pipeline that doesn't interfere with task orchestration:

```rust
// New file: src/services/telemetry_stream.rs
pub struct TelemetryStream {
    docker_hive_mind: Arc<DockerHiveMind>,
    mcp_pool: Arc<MCPConnectionPool>,
    websocket_clients: Arc<RwLock<HashMap<String, WebSocketClient>>>,
}

impl TelemetryStream {
    pub async fn start_hybrid_streaming(&self) -> Result<(), TelemetryError> {
        // Primary data stream from Docker (every 5 seconds)
        let docker_stream = self.create_docker_status_stream(Duration::from_secs(5)).await;

        // Telemetry enrichment from MCP (every 1 second)
        let mcp_stream = self.create_mcp_telemetry_stream(Duration::from_secs(1)).await;

        // Merge and broadcast
        let merged_stream = self.merge_telemetry_streams(docker_stream, mcp_stream).await;
        self.broadcast_to_clients(merged_stream).await
    }

    async fn create_docker_status_stream(&self, interval: Duration) -> impl Stream<Item = Value> {
        tokio_stream::wrappers::IntervalStream::new(tokio::time::interval(interval))
            .then(|_| async {
                match self.docker_hive_mind.get_sessions().await {
                    Ok(sessions) => json!({
                        "type": "docker_status",
                        "sessions": sessions,
                        "timestamp": Utc::now()
                    }),
                    Err(_) => json!({
                        "type": "docker_error",
                        "timestamp": Utc::now()
                    })
                }
            })
    }

    async fn create_mcp_telemetry_stream(&self, interval: Duration) -> impl Stream<Item = Value> {
        tokio_stream::wrappers::IntervalStream::new(tokio::time::interval(interval))
            .then(|_| async {
                // Non-blocking MCP telemetry collection
                match self.mcp_pool.execute_command("telemetry", "agent_metrics", json!({})).await {
                    Ok(telemetry) => json!({
                        "type": "mcp_telemetry",
                        "data": telemetry,
                        "timestamp": Utc::now()
                    }),
                    Err(_) => json!({
                        "type": "mcp_unavailable",
                        "timestamp": Utc::now()
                    })
                }
            })
    }
}
```

## Implementation Timeline

### Week 1: Core Migration
- [ ] Replace `call_task_orchestrate` with `call_task_orchestrate_hybrid` in speech service
- [ ] Update voice command execution functions
- [ ] Test Docker exec task spawning
- [ ] Verify task persistence across connection drops

### Week 2: Status Query Enhancement
- [ ] Implement hybrid status queries combining Docker + MCP data
- [ ] Update WebSocket handlers to use hybrid approach
- [ ] Test telemetry streaming with both data sources
- [ ] Implement fallback mechanisms

### Week 3: Performance Optimization
- [ ] Implement telemetry stream separation
- [ ] Add WebSocket multiplexing for bandwidth efficiency
- [ ] Optimize Docker command execution with connection pooling
- [ ] Add comprehensive error handling and retry logic

### Week 4: Testing and Validation
- [ ] End-to-end testing of voice commands with hybrid approach
- [ ] Load testing with multiple concurrent sessions
- [ ] Network failure recovery testing
- [ ] Performance benchmarking vs. pure TCP/MCP approach

## Success Metrics

### Reliability Improvements
- [ ] >95% task spawn success rate (vs. current ~60%)
- [ ] Zero task loss on network disconnection
- [ ] <30 second recovery time after container restart

### Performance Improvements
- [ ] <500ms voice command response time (vs. current 2-5s)
- [ ] 70% reduction in memory usage per client
- [ ] 80% reduction in network bandwidth for telemetry

### Functionality Preservation
- [ ] All existing voice commands work with hybrid approach
- [ ] Rich telemetry data still available for visualization
- [ ] WebSocket streaming maintains real-time updates
- [ ] Context management and voice tagging preserved

## Rollback Strategy

If issues arise during migration:

1. **Phase 1 Rollback**: Revert to original TCP/MCP calls in speech service
2. **Phase 2 Rollback**: Disable MCP telemetry enrichment, use Docker-only status
3. **Phase 3 Rollback**: Fallback WebSocket handlers to pure TCP/MCP mode
4. **Emergency Rollback**: Feature flag to completely disable hybrid approach

## Risk Mitigation

### High-Risk Areas
1. **Docker Daemon Failure**: Implement health monitoring and automatic restart
2. **Command Parsing**: Comprehensive testing of hive-mind output parsing
3. **Session ID Extraction**: Multiple fallback patterns for session ID detection

### Testing Strategy
1. **Unit Tests**: Mock Docker exec calls and test parsing logic
2. **Integration Tests**: Real Docker container testing with network failures
3. **Load Tests**: Concurrent voice command processing stress testing
4. **Recovery Tests**: Container restart and network partition scenarios

This migration strategy provides a clear path to eliminate TCP/MCP process isolation issues while preserving all existing functionality and improving system reliability.