# Hybrid Docker Exec + TCP/MCP Architecture

**Status:** ✅ **IMPLEMENTED AND OPERATIONAL**
**Last Updated:** 2025-10-06
**Implementation:** Complete with MCP Session Bridge

## Executive Summary

This document outlines the hybrid architecture for the multi-agent system that combines Docker exec for task orchestration with TCP/MCP for real-time telemetry and visualization. **This architecture is now fully implemented and operational.**

### Implementation Status

✅ **MCP Session Bridge** (`src/services/mcp_session_bridge.rs`)
✅ **Real Session Spawning** via `/bots/initialize-swarm`
✅ **UUID ↔ Swarm ID Correlation** with bidirectional mapping
✅ **Agent Streaming** via Binary Protocol V2 (36 bytes/node)
✅ **GPU Integration** for agent physics visualization
✅ **WebSocket Broadcasting** with agent/knowledge node type flags
✅ **Telemetry System** with correlation IDs and structured logging

## Current State Analysis

### Problems with Pure TCP/MCP Approach
1. **Process Isolation**: Each TCP connection spawns isolated MCP processes, preventing task persistence
2. **State Fragmentation**: Tasks exist only in their originating connection's process
3. **Network Brittleness**: TCP connections fail unpredictably in Docker environments
4. **Resource Overhead**: Multiple isolated processes consume excessive memory
5. **Debugging Complexity**: Process boundaries make debugging difficult

### Identified Assets in Current System
1. **Functional hive-mind**: Docker container has working claude-flow hive-mind
2. **Rich Telemetry**: TCP/MCP provides detailed agent visualisation data
3. **WebSocket Infrastructure**: Real-time streaming to GPU-based spring system
4. **Resilience Framework**: Circuit breakers, health monitoring, retry logic

## Architectural Principles

### 1. Separation of Concerns
- **Control Plane (Docker Exec)**: Task creation, lifecycle management, process orchestration
- **Data Plane (TCP/MCP)**: Telemetry streaming, visualisation data, performance metrics
- **Visualisation Plane (WebSocket)**: Real-time updates to spring system on GPU

### 2. Single Source of Truth
- Hive-mind in multi-agent-container is authoritative for all task state
- TCP/MCP serves as read-only data source for visualisation
- No task creation via TCP/MCP to prevent state fragmentation

### 3. Fault Isolation
- Docker exec failures don't affect telemetry streaming
- TCP/MCP failures don't affect task execution
- WebSocket failures don't affect core orchestration

## Implemented Data Flow (2025-10-06)

### Agent Spawning Flow
```
UI (MultiAgentInitializationPrompt.tsx)
  ↓ POST /bots/initialize-swarm
  ↓
Backend (bots_handler.rs::initialize_hive_mind_swarm)
  ↓ spawn_swarm_monitored()
  ↓
MCP Session Bridge (mcp_session_bridge.rs)
  ↓ spawn_and_monitor() → DockerHiveMind.spawn_swarm()
  ↓ Returns UUID immediately
  ↓ Poll for swarm_id (filesystem discovery + MCP query)
  ↓ Link UUID ↔ swarm_id in bidirectional cache
  ↓
Returns: {uuid, swarm_id, initial_agents}
```

### Agent Data Streaming Flow
```
MCP Server (multi-agent-container:9500)
  ↓ TCP connection
  ↓
Claude Flow Actor (claude_flow_actor.rs)
  ↓ Poll agent statuses (2s interval)
  ↓ MCP TCP client query_agent_list()
  ↓ Convert MultiMcpAgentStatus → AgentStatus
  ↓
Graph Service Actor (graph_actor.rs)
  ↓ Handler<UpdateBotsGraph>
  ↓ Convert Agent → Node with AGENT_NODE_FLAG (bit 31)
  ↓ Send to GPU for physics simulation
  ↓ Encode Binary Protocol V2 (36 bytes/node)
  ↓
Client (BinaryWebSocketProtocol.ts)
  ↓ Decode and visualize
```

### Key Implementation Files

| Component | File | Status |
|-----------|------|--------|
| Session Bridge | `src/services/mcp_session_bridge.rs` | ✅ Complete |
| Agent Spawning | `src/handlers/bots_handler.rs` | ✅ Refactored |
| Agent Streaming | `src/actors/claude_flow_actor.rs` | ✅ Operational |
| Graph Integration | `src/actors/graph_actor.rs` | ✅ Dual-graph support |
| Binary Protocol | `src/utils/binary_protocol.rs` | ✅ V2 (36 bytes) |
| Telemetry | `src/telemetry/agent_telemetry.rs` | ✅ Correlation IDs |
| Client Protocol | `client/src/types/binaryProtocol.ts` | ✅ V2 parser |

## Core Architecture Components

## 1. docker_hive_mind.rs Module

```rust
use tokio::process::Command;
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use chrono::{DateTime, Utc};

pub struct DockerHiveMind {
    container_name: String,
    claude_flow_path: String,
    active_swarms: Arc<RwLock<HashMap<String, SwarmMetadata>>>,
    command_history: Arc<RwLock<Vec<CommandRecord>>>,
    health_monitor: DockerHealthMonitor,
}

#[derive(Debug, Clone)]
pub struct SwarmMetadata {
    pub swarm_id: String,
    pub task_description: String,
    pub spawn_time: DateTime<Utc>,
    pub status: SwarmStatus,
    pub priority: String,
    pub docker_pid: Option<u32>,
    pub last_heartbeat: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum SwarmStatus {
    Spawning,
    Active,
    Paused,
    Completed,
    Failed(String),
}

impl DockerHiveMind {
    pub async fn spawn_task(
        &self,
        task: &str,
        priority: Option<&str>,
        strategy: Option<&str>,
    ) -> Result<SwarmMetadata, DockerHiveMindError> {
        // Build docker exec command
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec", "-d",
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "spawn",
            task,
            "--claude"
        ]);

        // Add priority mapping
        if let Some(p) = priority {
            cmd.arg("--queen-type");
            let queen_type = match p {
                "high" | "critical" => "tactical",
                "medium" => "strategic",
                "low" => "adaptive",
                _ => "strategic",
            };
            cmd.arg(queen_type);
        }

        // Execute with timeout and retry
        let output = self.execute_with_retry(cmd, 3).await?;

        // Generate swarm metadata
        let swarm_id = self.extract_swarm_id(&output).await?;
        let metadata = SwarmMetadata {
            swarm_id: swarm_id.clone(),
            task_description: task.to_string(),
            spawn_time: Utc::now(),
            status: SwarmStatus::Spawning,
            priority: priority.unwrap_or("medium").to_string(),
            docker_pid: self.extract_docker_pid(&output).await,
            last_heartbeat: Utc::now(),
        };

        // Store metadata
        self.active_swarms.write().await.insert(swarm_id.clone(), metadata.clone());

        // Record command
        self.record_command("spawn_task", task, &output).await;

        Ok(metadata)
    }

    pub async fn get_status(&self, swarm_id: Option<&str>) -> Result<Value, DockerHiveMindError> {
        let cmd = if let Some(id) = swarm_id {
            Command::new("docker")
                .args(&["exec", &self.container_name,
                       &self.claude_flow_path, "hive-mind", "status", id])
                .output()
                .await?
        } else {
            Command::new("docker")
                .args(&["exec", &self.container_name,
                       &self.claude_flow_path, "hive-mind", "sessions"])
                .output()
                .await?
        };

        self.parse_hive_mind_output(&cmd.stdout).await
    }

    pub async fn get_sessions(&self) -> Result<Vec<SwarmMetadata>, DockerHiveMindError> {
        let sessions_output = Command::new("docker")
            .args(&["exec", &self.container_name,
                   &self.claude_flow_path, "hive-mind", "sessions"])
            .output()
            .await?;

        let live_sessions = self.parse_sessions(&sessions_output.stdout).await?;

        // Merge with local metadata
        let mut active_swarms = self.active_swarms.write().await;
        for session in &live_sessions {
            if let Some(existing) = active_swarms.get_mut(&session.swarm_id) {
                existing.status = session.status.clone();
                existing.last_heartbeat = Utc::now();
            } else {
                active_swarms.insert(session.swarm_id.clone(), session.clone());
            }
        }

        Ok(live_sessions)
    }

    pub async fn get_metrics(&self) -> Result<Value, DockerHiveMindError> {
        let metrics_output = Command::new("docker")
            .args(&["exec", &self.container_name,
                   &self.claude_flow_path, "hive-mind", "metrics"])
            .output()
            .await?;

        let mut metrics = self.parse_metrics(&metrics_output.stdout).await?;

        // Add local metadata
        let active_count = self.active_swarms.read().await.len();
        metrics["local_metadata"] = json!({
            "tracked_swarms": active_count,
            "last_update": Utc::now(),
            "docker_health": self.health_monitor.get_status().await
        });

        Ok(metrics)
    }

    pub async fn cleanup_zombie_processes(&self) -> Result<u32, DockerHiveMindError> {
        // Find orphaned docker exec processes
        let ps_output = Command::new("docker")
            .args(&["exec", &self.container_name, "ps", "aux"])
            .output()
            .await?;

        let processes = String::from_utf8_lossy(&ps_output.stdout);
        let mut cleaned = 0;

        // Identify zombie claude-flow processes
        for line in processes.lines() {
            if line.contains("claude-flow") && line.contains("<defunct>") {
                if let Some(pid) = self.extract_pid_from_ps_line(line) {
                    if self.kill_process_in_container(pid).await.is_ok() {
                        cleaned += 1;
                    }
                }
            }
        }

        // Clean up local metadata for dead swarms
        self.cleanup_dead_swarms().await;

        Ok(cleaned)
    }

    async fn execute_with_retry(
        &self,
        mut cmd: Command,
        max_retries: u32,
    ) -> Result<std::process::Output, DockerHiveMindError> {
        let mut last_error = None;

        for attempt in 1..=max_retries {
            match cmd.output().await {
                Ok(output) => {
                    if output.status.success() {
                        return Ok(output);
                    } else {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        last_error = Some(DockerHiveMindError::CommandFailed {
                            attempt,
                            stderr: stderr.to_string(),
                        });
                    }
                }
                Err(e) => {
                    last_error = Some(DockerHiveMindError::DockerExecFailed {
                        attempt,
                        source: e,
                    });
                }
            }

            if attempt < max_retries {
                tokio::time::sleep(tokio::time::Duration::from_millis(500 * attempt as u64)).await;
            }
        }

        Err(last_error.unwrap())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DockerHiveMindError {
    #[error("Docker exec failed on attempt {attempt}: {source}")]
    DockerExecFailed { attempt: u32, source: std::io::Error },

    #[error("Command failed on attempt {attempt}: {stderr}")]
    CommandFailed { attempt: u32, stderr: String },

    #[error("Failed to parse hive-mind output: {reason}")]
    ParseError { reason: String },

    #[error("Container health check failed: {status}")]
    ContainerUnhealthy { status: String },

    #[error("Swarm {swarm_id} not found")]
    SwarmNotFound { swarm_id: String },
}
```

## 2. Protocol Boundaries

### Operations Using Docker Exec (Control Plane)

| Operation | Command | Justification |
|-----------|---------|---------------|
| `spawn_task` | `docker exec -d multi-agent-container /app/node_modules/.bin/claude-flow hive-mind spawn "task" --claude` | Creates persistent task in hive-mind process |
| `get_sessions` | `docker exec multi-agent-container claude-flow hive-mind sessions` | Queries authoritative swarm list |
| `get_status` | `docker exec multi-agent-container claude-flow hive-mind status [id]` | Gets real-time swarm status |
| `stop_swarm` | `docker exec multi-agent-container claude-flow hive-mind stop [id]` | Controlled shutdown of swarm |
| `cleanup_processes` | `docker exec multi-agent-container ps aux \| grep claude-flow` | Process management within container |
| `health_check` | `docker exec multi-agent-container claude-flow hive-mind status` | Container process health |

### Operations Using TCP/MCP (Data Plane)

| Operation | Justification | Bandwidth |
|-----------|---------------|-----------|
| `agent_metrics` | Rich telemetry data for visualisation | ~100KB/s |
| `performance_analysis` | GPU spring system needs continuous updates | ~50KB/s |
| `neural_patterns` | Real-time AI decision data | ~25KB/s |
| `topology_updates` | Dynamic agent relationship changes | ~10KB/s |
| `swarm_monitor` | Live dashboard data streaming | ~75KB/s |

### Crossover Points (Control → Data)

1. **Task Spawn**: Docker exec creates → TCP/MCP discovers via polling
2. **Status Updates**: Docker exec queries → TCP/MCP enriches with telemetry
3. **Process Death**: Docker exec detects → TCP/MCP marks as offline

## 3. Technical Challenge Solutions

### Broken Network States Recovery

```rust
pub struct NetworkStateRecovery {
    docker_api: Docker,
    container_monitor: ContainerMonitor,
}

impl NetworkStateRecovery {
    pub async fn detect_broken_network(&self) -> Result<bool, RecoveryError> {
        // Test docker exec connectivity
        let exec_test = Command::new("docker")
            .args(&["exec", "multi-agent-container", "echo", "test"])
            .output()
            .await;

        // Test TCP MCP connectivity
        let tcp_test = TcpStream::connect("multi-agent-container:9500")
            .await;

        match (exec_test, tcp_test) {
            (Ok(_), Err(_)) => {
                // Docker works, TCP broken - recoverable
                Ok(false)
            }
            (Err(_), _) => {
                // Docker broken - needs container restart
                Ok(true)
            }
            (Ok(_), Ok(_)) => {
                // Both working
                Ok(false)
            }
        }
    }

    pub async fn recover_network_state(&self) -> Result<(), RecoveryError> {
        // 1. Restart container networking
        self.docker_api.restart_container("multi-agent-container").await?;

        // 2. Wait for hive-mind to come online
        self.wait_for_hive_mind_ready().await?;

        // 3. Restore active swarms from backup
        self.restore_swarm_state().await?;

        Ok(())
    }
}
```

### PID Management Across Containers

```rust
pub struct ContainerPIDManager {
    container_name: String,
    pid_registry: Arc<RwLock<HashMap<String, ProcessInfo>>>,
}

#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub container_pid: u32,
    pub host_pid: Option<u32>,
    pub swarm_id: String,
    pub command_line: String,
    pub start_time: DateTime<Utc>,
    pub status: ProcessStatus,
}

impl ContainerPIDManager {
    pub async fn track_swarm_process(&self, swarm_id: &str) -> Result<ProcessInfo, PIDError> {
        // Get container PID for the swarm
        let ps_output = Command::new("docker")
            .args(&["exec", &self.container_name, "pgrep", "-f", swarm_id])
            .output()
            .await?;

        let container_pid: u32 = String::from_utf8_lossy(&ps_output.stdout)
            .trim()
            .parse()?;

        // Get host PID using docker inspect
        let inspect_output = Command::new("docker")
            .args(&["inspect", &self.container_name,
                   "--format", "{{.State.Pid}}"])
            .output()
            .await?;

        let container_host_pid: u32 = String::from_utf8_lossy(&inspect_output.stdout)
            .trim()
            .parse()?;

        // Calculate actual host PID (approximation)
        let host_pid = container_host_pid + container_pid;

        let process_info = ProcessInfo {
            container_pid,
            host_pid: Some(host_pid),
            swarm_id: swarm_id.to_string(),
            command_line: format!("claude-flow hive-mind {}", swarm_id),
            start_time: Utc::now(),
            status: ProcessStatus::Running,
        };

        self.pid_registry.write().await.insert(swarm_id.to_string(), process_info.clone());

        Ok(process_info)
    }

    pub async fn cleanup_orphaned_processes(&self) -> Result<u32, PIDError> {
        let mut cleaned = 0;
        let registry = self.pid_registry.read().await;

        for (swarm_id, process_info) in registry.iter() {
            // Check if process is still alive in container
            let check_cmd = Command::new("docker")
                .args(&["exec", &self.container_name, "kill", "-0",
                       &process_info.container_pid.to_string()])
                .output()
                .await;

            if check_cmd.is_err() {
                // Process is dead, clean up
                drop(registry);
                self.pid_registry.write().await.remove(swarm_id);
                cleaned += 1;
            }
        }

        Ok(cleaned)
    }
}
```

### Process Cleanup and Zombie Prevention

```rust
pub struct ZombieProcessCleaner {
    container_name: String,
    cleanup_interval: Duration,
}

impl ZombieProcessCleaner {
    pub fn start_background_cleanup(&self) {
        let container_name = self.container_name.clone();
        let interval = self.cleanup_interval;

        tokio::spawn(async move {
            let mut cleanup_timer = tokio::time::interval(interval);

            loop {
                cleanup_timer.tick().await;

                if let Err(e) = Self::cleanup_zombies(&container_name).await {
                    error!("Zombie cleanup failed: {}", e);
                }
            }
        });
    }

    async fn cleanup_zombies(container_name: &str) -> Result<u32, CleanupError> {
        // Find zombie processes
        let ps_output = Command::new("docker")
            .args(&["exec", container_name, "ps", "axo", "pid,ppid,stat,comm"])
            .output()
            .await?;

        let processes = String::from_utf8_lossy(&ps_output.stdout);
        let mut killed_zombies = 0;

        for line in processes.lines().skip(1) { // Skip header
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 4 && fields[2].contains('Z') {
                // Found zombie process
                let pid = fields[0];
                let ppid = fields[1];

                // Kill parent to clean up zombie
                if let Ok(_) = Command::new("docker")
                    .args(&["exec", container_name, "kill", ppid])
                    .output()
                    .await {
                    killed_zombies += 1;
                }
            }
        }

        // Force garbage collection in container
        let _ = Command::new("docker")
            .args(&["exec", container_name, "sh", "-c",
                   "echo 3 > /proc/sys/vm/drop_caches"])
            .output()
            .await;

        Ok(killed_zombies)
    }
}
```

### Container Restart Resilience

```rust
pub struct ContainerResilienceManager {
    docker_hive_mind: Arc<DockerHiveMind>,
    state_backup: Arc<SwarmStateBackup>,
    restart_policy: RestartPolicy,
}

impl ContainerResilienceManager {
    pub async fn handle_container_restart(&self) -> Result<(), ResilienceError> {
        // 1. Detect container restart
        let restart_detected = self.detect_restart().await?;

        if restart_detected {
            // 2. Backup current state before restart
            self.state_backup.create_checkpoint().await?;

            // 3. Wait for container to be healthy
            self.wait_for_container_health().await?;

            // 4. Restore active swarms
            let restored_swarms = self.restore_swarms().await?;

            // 5. Reconnect TCP/MCP data plane
            self.reconnect_data_plane().await?;

            info!("Container restart recovery complete: {} swarms restored",
                  restored_swarms.len());
        }

        Ok(())
    }

    async fn restore_swarms(&self) -> Result<Vec<SwarmMetadata>, ResilienceError> {
        let checkpoints = self.state_backup.list_recent_checkpoints(Duration::from_hours(1)).await?;
        let mut restored = Vec::new();

        for checkpoint in checkpoints {
            for swarm in checkpoint.active_swarms {
                // Try to restore swarm
                match self.docker_hive_mind.spawn_task(
                    &swarm.task_description,
                    Some(&swarm.priority),
                    None
                ).await {
                    Ok(new_metadata) => {
                        restored.push(new_metadata);
                    }
                    Err(e) => {
                        warn!("Failed to restore swarm {}: {}", swarm.swarm_id, e);
                    }
                }
            }
        }

        Ok(restored)
    }
}
```

## 4. Telemetry Optimization

### Separated Control and Data Planes

```rust
pub struct TelemetryOptimizer {
    control_plane: Arc<DockerHiveMind>,
    data_plane: Arc<MCPTelemetryStream>,
    websocket_multiplexer: Arc<WebSocketMultiplexer>,
    compression: CompressionManager,
}

impl TelemetryOptimizer {
    pub async fn start_optimized_streaming(&self) -> Result<(), TelemetryError> {
        // Control plane: Low-frequency, authoritative data
        let control_stream = self.control_plane.create_status_stream(Duration::from_secs(5)).await?;

        // Data plane: High-frequency, rich telemetry
        let data_stream = self.data_plane.create_metrics_stream(Duration::from_millis(100)).await?;

        // Merge streams with priority
        let merged_stream = self.merge_streams(control_stream, data_stream).await;

        // Apply compression
        let compressed_stream = self.compression.compress_stream(merged_stream).await;

        // Multiplex to WebSocket clients
        self.websocket_multiplexer.broadcast_stream(compressed_stream).await?;

        Ok(())
    }

    pub async fn create_batch_telemetry_update(&self) -> Result<Value, TelemetryError> {
        // Collect from control plane
        let swarm_status = self.control_plane.get_sessions().await?;

        // Enrich with data plane metrics
        let mut batch_update = json!({
            "timestamp": Utc::now(),
            "swarms": []
        });

        for swarm in swarm_status {
            let metrics = self.data_plane.get_swarm_metrics(&swarm.swarm_id).await?;

            batch_update["swarms"].as_array_mut().unwrap().push(json!({
                "swarm_id": swarm.swarm_id,
                "status": swarm.status,
                "task": swarm.task_description,
                "metrics": metrics,
                "agents": self.data_plane.get_agent_list(&swarm.swarm_id).await?
            }));
        }

        Ok(batch_update)
    }
}
```

### WebSocket Multiplexing Strategy

```rust
pub struct WebSocketMultiplexer {
    clients: Arc<RwLock<HashMap<String, WebSocketClient>>>,
    compression_settings: CompressionSettings,
    bandwidth_limiter: BandwidthLimiter,
}

#[derive(Debug, Clone)]
pub struct WebSocketClient {
    pub client_id: String,
    pub sender: mpsc::UnboundedSender<Message>,
    pub subscription_filters: SubscriptionFilters,
    pub bandwidth_quota: u64, // bytes per second
    pub last_update: Instant,
}

impl WebSocketMultiplexer {
    pub async fn broadcast_telemetry(&self, data: Value) -> Result<(), MultiplexError> {
        let clients = self.clients.read().await;
        let compressed_data = self.compression_settings.compress(&data).await?;

        // Calculate per-client data based on filters
        for (client_id, client) in clients.iter() {
            let filtered_data = self.apply_client_filters(&data, &client.subscription_filters).await?;

            // Check bandwidth limits
            if self.bandwidth_limiter.can_send(client_id, filtered_data.len()).await {
                let message = Message::Binary(self.compression_settings.compress(&filtered_data).await?);

                if let Err(e) = client.sender.send(message) {
                    warn!("Failed to send to client {}: {}", client_id, e);
                }
            } else {
                // Send summary instead of full data
                let summary = self.create_summary(&filtered_data).await;
                let message = Message::Text(summary);
                let _ = client.sender.send(message);
            }
        }

        Ok(())
    }

    async fn apply_client_filters(
        &self,
        data: &Value,
        filters: &SubscriptionFilters
    ) -> Result<Value, MultiplexError> {
        let mut filtered = data.clone();

        // Filter swarms
        if let Some(swarms) = filtered.get_mut("swarms").and_then(|s| s.as_array_mut()) {
            swarms.retain(|swarm| {
                if !filters.swarm_ids.is_empty() {
                    let swarm_id = swarm.get("swarm_id").and_then(|s| s.as_str()).unwrap_or("");
                    return filters.swarm_ids.contains(&swarm_id.to_string());
                }
                true
            });
        }

        // Filter by performance data inclusion
        if !filters.include_performance {
            if let Some(swarms) = filtered.get_mut("swarms").and_then(|s| s.as_array_mut()) {
                for swarm in swarms {
                    if let Some(obj) = swarm.as_object_mut() {
                        obj.remove("metrics");
                    }
                }
            }
        }

        Ok(filtered)
    }
}
```

## 5. Migration Strategy

### Phase 1: Create docker_hive_mind.rs Module
- Implement core Docker exec functions
- Add error handling and retry logic
- Create process tracking and cleanup
- Add comprehensive logging

### Phase 2: Update Speech Service
- Replace `call_task_orchestrate` with `call_task_orchestrate_docker`
- Remove TCP MCP spawning from voice commands
- Keep TCP MCP for agent status queries (read-only)
- Add telemetry integration points

### Phase 3: Migrate API Handlers
- Update `mcp_relay_handler.rs` to use hybrid approach
- Modify WebSocket handlers to separate control/data
- Implement bandwidth limiting and compression
- Add fallback mechanisms

### Phase 4: Optimise Telemetry Pipeline
- Implement WebSocket multiplexing
- Add compression and filtering
- Create bandwidth management
- Optimise for GPU spring system updates

### Phase 5: Add Monitoring and Recovery
- Container restart detection and recovery
- Network state monitoring
- Process cleanup automation
- Performance metrics collection

## Performance Projections

| Metric | Current (Pure TCP) | Hybrid Architecture | Improvement |
|--------|-------------------|---------------------|-------------|
| Task Spawn Latency | 2-5 seconds | 200-500ms | 4-10x faster |
| Memory Usage | 150MB per connection | 50MB base + 5MB per client | 70% reduction |
| Network Reliability | 60% success rate | 95% success rate | 58% improvement |
| Debugging Time | 2-4 hours | 15-30 minutes | 80% reduction |
| Telemetry Bandwidth | 500KB/s per client | 100KB/s per client | 80% reduction |

## Risk Assessment and Mitigation

### High-Risk Areas
1. **Docker Daemon Failure**: Container becomes unreachable
   - *Mitigation*: Health monitoring, automatic restart, state persistence

2. **Process Zombie Accumulation**: Memory leaks from abandoned processes
   - *Mitigation*: Background cleanup, PID tracking, resource limits

3. **Network Partition**: Container network isolation
   - *Mitigation*: Multiple connection methods, circuit breakers, graceful degradation

### Medium-Risk Areas
1. **Telemetry Data Loss**: Missing visualization updates
   - *Mitigation*: Buffering, compression, priority queuing

2. **State Synchronization**: Control/data plane inconsistencies
   - *Mitigation*: Single source of truth, periodic reconciliation

## Success Metrics

1. **Reliability**: >95% task spawn success rate
2. **Performance**: <500ms average task spawn latency
3. **Resource Efficiency**: <100MB total memory usage
4. **Data Quality**: <5% telemetry packet loss
5. **Recovery Time**: <30 seconds container restart recovery

## Implementation Timeline

- **Week 1**: docker_hive_mind.rs module implementation
- **Week 2**: Speech service migration and testing
- **Week 3**: API handler updates and WebSocket optimisation
- **Week 4**: Telemetry pipeline and compression
- **Week 5**: Monitoring, recovery, and performance tuning
- **Week 6**: Integration testing and documentation

This hybrid architecture provides the reliability of Docker exec for critical operations while maintaining the rich data capabilities of TCP/MCP for visualization and monitoring.