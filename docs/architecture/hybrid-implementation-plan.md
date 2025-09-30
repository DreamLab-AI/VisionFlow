# Hybrid Architecture Implementation Plan
## Strategic Planning for Docker-MCP Integration

### Executive Summary

The multi-agent system requires a hybrid architecture that leverages Docker exec for swarm lifecycle management while maintaining MCP/TCP connections for high-throughput telemetry and visualization. This plan addresses the process isolation issues while optimising for performance and reliability.

## Current Architecture Analysis

### Problems Identified
1. **Process Isolation**: Each TCP/MCP connection spawns isolated processes
2. **State Fragmentation**: Tasks exist only within their connection's process
3. **Broken Network States**: PID conflicts and connection drops
4. **Performance Overhead**: TCP/JSON-RPC layer adds latency
5. **Resource Leaks**: Orphaned processes and connections

### System Components
- **multi-agent-container** (172.18.0.10): MCP server, claude-flow hive-mind
- **gui-tools-container** (172.18.0.9): Visualization tools
- **visionflow_container**: AR/VR client application
- **Network**: docker_ragflow (172.18.0.0/16)

## Hybrid Architecture Strategy

### Core Principle: Right Tool for Right Job

**Docker Exec for:**
- ✅ Swarm initialization and lifecycle
- ✅ Task orchestration and spawning
- ✅ Command execution with persistence
- ✅ State management across sessions
- ✅ Resource cleanup and monitoring

**TCP/MCP for:**
- ✅ Real-time telemetry streaming
- ✅ High-frequency visualization updates
- ✅ GPU spring system data feeds
- ✅ Client UI status synchronization
- ✅ WebSocket bridge communication

## Implementation Plan

### Phase 1: Core Docker Interface Module (Week 1)

#### 1.1 Create docker_hive_mind.rs Module
```rust
// /workspace/ext/src/utils/docker_hive_mind.rs
pub struct DockerHiveMind {
    container_name: String,
    claude_flow_path: String,
    session_cache: Arc<RwLock<HashMap<String, SessionInfo>>>,
    health_monitor: HealthMonitor,
}

pub struct SessionInfo {
    session_id: String,
    task_description: String,
    status: SwarmStatus,
    created_at: DateTime<Utc>,
    last_activity: DateTime<Utc>,
    metrics: SwarmMetrics,
}
```

#### 1.2 Core Operations
- `spawn_swarm(task: &str, config: SwarmConfig) -> Result<SwarmId>`
- `get_sessions() -> Result<Vec<SessionInfo>>`
- `get_swarm_status(swarm_id: &str) -> Result<SwarmStatus>`
- `stop_swarm(swarm_id: &str) -> Result<()>`
- `cleanup_orphaned_processes() -> Result<u32>`

#### 1.3 Implementation Strategy
```rust
impl DockerHiveMind {
    pub async fn spawn_swarm(&self, task: &str, config: SwarmConfig) -> Result<String> {
        // 1. Validate container health
        self.health_monitor.check_container_health().await?;

        // 2. Execute docker command with proper error handling
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec", "-d",
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "spawn", task,
            "--claude", "--auto-spawn"
        ]);

        // 3. Add configuration parameters
        self.apply_config(&mut cmd, &config);

        // 4. Execute and capture session ID
        let output = cmd.output().await?;
        let session_id = self.extract_session_id(&output)?;

        // 5. Cache session info
        self.cache_session(session_id.clone(), task, config).await;

        Ok(session_id)
    }

    pub async fn get_sessions(&self) -> Result<Vec<SessionInfo>> {
        // Query hive-mind sessions via docker exec
        let output = Command::new("docker")
            .args(&["exec", &self.container_name,
                   &self.claude_flow_path, "hive-mind", "sessions"])
            .output().await?;

        self.parse_sessions_output(&output)
    }
}
```

### Phase 2: Migration Path Design (Week 2)

#### 2.1 Gradual Migration Strategy

**Step 1: Speech Service Migration**
- Replace MCP `task_orchestrate` calls with Docker exec
- Maintain existing API contracts
- Add fallback mechanisms

**Step 2: API Handler Updates**
- Update `/bots/spawn-task` endpoint
- Modify WebSocket task broadcasting
- Implement dual-mode operation

**Step 3: Client UI Adaptation**
- Update BotsControlPanel.tsx
- Modify task status polling
- Add Docker health indicators

#### 2.2 Backwards Compatibility
```rust
pub enum TaskMethod {
    Docker,     // Primary method
    MCP,        // Fallback for telemetry
    Hybrid,     // Both for comparison
}

pub struct HybridTaskOrchestrator {
    docker_client: DockerHiveMind,
    mcp_client: MCPConnectionPool,
    config: HybridConfig,
}
```

### Phase 3: Fault Tolerance Mechanisms (Week 2-3)

#### 3.1 Network Failure Recovery
```rust
pub struct NetworkRecoveryManager {
    retry_policy: ExponentialBackoff,
    circuit_breaker: CircuitBreaker,
    health_checker: ContainerHealthChecker,
}

impl NetworkRecoveryManager {
    pub async fn recover_from_failure(&self, failure: NetworkFailure) -> RecoveryAction {
        match failure {
            NetworkFailure::ContainerDown => self.restart_container().await,
            NetworkFailure::ProcessHung => self.kill_and_respawn().await,
            NetworkFailure::NetworkPartition => self.wait_and_retry().await,
            NetworkFailure::ResourceExhaustion => self.cleanup_and_restart().await,
        }
    }
}
```

#### 3.2 Container Restart Handling
- **Graceful Restart**: Save session state before restart
- **Session Recovery**: Restore active swarms after restart
- **State Validation**: Verify swarm integrity post-restart
- **Client Notification**: Update UIs about restart events

#### 3.3 State Synchronization
```rust
pub struct StateSync {
    persistent_store: PersistentStorage,
    memory_cache: Arc<RwLock<CacheStore>>,
    sync_interval: Duration,
}

// Periodic sync of Docker swarm state with MCP telemetry
impl StateSync {
    pub async fn sync_swarm_states(&self) -> Result<SyncReport> {
        let docker_sessions = self.docker_client.get_sessions().await?;
        let mcp_telemetry = self.mcp_client.get_telemetry().await?;

        self.reconcile_states(docker_sessions, mcp_telemetry).await
    }
}
```

#### 3.4 Graceful Degradation
- **Docker Unavailable**: Fall back to MCP-only mode
- **MCP Unavailable**: Operate with Docker + cached telemetry
- **Both Unavailable**: Show error state with recovery options
- **Partial Failure**: Continue with reduced functionality

### Phase 4: Performance Optimization (Week 3-4)

#### 4.1 Connection Pooling Strategies
```rust
pub struct HybridConnectionPool {
    docker_connections: Arc<Mutex<ConnectionPool<DockerConnection>>>,
    mcp_connections: Arc<RwLock<HashMap<String, Arc<PersistentMCPConnection>>>>,
    load_balancer: LoadBalancer,
}

pub struct ConnectionPolicy {
    max_docker_connections: u32,
    max_mcp_connections: u32,
    connection_timeout: Duration,
    idle_timeout: Duration,
    health_check_interval: Duration,
}
```

#### 4.2 Caching Mechanisms
- **Session Cache**: In-memory cache of active swarms
- **Telemetry Cache**: Time-windowed metrics cache
- **Health Cache**: Container and network health status
- **Response Cache**: Cache frequent API responses

#### 4.3 Lazy Loading Patterns
```rust
pub struct LazyTelemetryLoader {
    cache: Arc<RwLock<TelemetryCache>>,
    background_updater: Arc<BackgroundUpdater>,
}

impl LazyTelemetryLoader {
    pub async fn get_metrics(&self, swarm_id: &str) -> Result<SwarmMetrics> {
        // Check cache first
        if let Some(cached) = self.cache.read().await.get(swarm_id) {
            if !cached.is_stale() {
                return Ok(cached.metrics.clone());
            }
        }

        // Fetch from Docker/MCP in background
        self.background_updater.request_update(swarm_id);

        // Return cached or placeholder
        self.get_cached_or_placeholder(swarm_id).await
    }
}
```

#### 4.4 Resource Cleanup
- **Orphaned Process Detection**: Scan for zombie processes
- **Memory Leak Prevention**: Automatic cache eviction
- **Connection Cleanup**: Close idle connections
- **Temporary File Cleanup**: Remove stale session files

### Phase 5: Client UI Updates (Week 4)

#### 5.1 BotsControlPanel.tsx Updates
```typescript
// Enhanced agent spawning with Docker backend
const handleAddAgent = async (type: BotsAgent['type']) => {
    try {
        // Use new hybrid API endpoint
        const response = await apiService.post('/bots/spawn-agent-docker', {
            agentType: type,
            method: 'docker', // or 'mcp' or 'hybrid'
            config: {
                priority: 'medium',
                strategy: 'hive-mind'
            }
        });

        if (response.success) {
            setAgentCount(prev => prev + 1);
            // Start real-time monitoring
            startTelemetryStream(response.sessionId);
        }
    } catch (error) {
        console.error(`Error spawning ${type} agent:`, error);
        // Try MCP fallback
        await handleAddAgentFallback(type);
    }
};
```

#### 5.2 Real-time Status Updates
```typescript
interface HybridStatus {
    dockerHealth: 'healthy' | 'degraded' | 'unavailable';
    mcpHealth: 'connected' | 'reconnecting' | 'disconnected';
    activeSessions: SessionInfo[];
    telemetryDelay: number;
    networkLatency: number;
}

const useHybridStatus = () => {
    const [status, setStatus] = useState<HybridStatus>();

    useEffect(() => {
        const ws = new WebSocket('/ws/hybrid-status');
        ws.onmessage = (event) => {
            const status = JSON.parse(event.data) as HybridStatus;
            setStatus(status);
        };

        return () => ws.close();
    }, []);

    return status;
};
```

### Phase 6: Documentation and Cleanup (Week 5)

#### 6.1 API Documentation Updates
- Document new hybrid endpoints
- Update client integration guides
- Create troubleshooting guides
- Add performance tuning recommendations

#### 6.2 Migration Guide
```markdown
# Migration from TCP-Only to Hybrid Architecture

## Before Migration
- All tasks created via TCP/MCP connections
- Process isolation issues
- No persistence across connections

## After Migration
- Task lifecycle via Docker exec
- Telemetry via MCP connections
- Persistent state across sessions
- Fault tolerance and recovery
```

## Implementation Tasks with Dependencies

### High Priority (Week 1)
1. **Create docker_hive_mind.rs module** (No dependencies)
   - Core swarm lifecycle functions
   - Error handling and validation
   - Session caching system

2. **Update mcp_connection.rs** (Depends on #1)
   - Add hybrid mode support
   - Implement fallback mechanisms
   - Update connection pooling

3. **Modify speech service** (Depends on #1, #2)
   - Replace MCP task spawning
   - Add Docker exec calls
   - Maintain API compatibility

### Medium Priority (Week 2-3)
4. **Implement fault tolerance** (Depends on #1-#3)
   - Network failure recovery
   - Container restart handling
   - State synchronization

5. **Add performance optimizations** (Depends on #1-#4)
   - Connection pooling
   - Caching mechanisms
   - Resource cleanup

6. **Update API handlers** (Depends on #1-#5)
   - Hybrid endpoint creation
   - WebSocket integration
   - Error handling

### Lower Priority (Week 4-5)
7. **Update client UI** (Depends on #1-#6)
   - BotsControlPanel modifications
   - Real-time status displays
   - Health indicators

8. **Documentation updates** (Depends on #1-#7)
   - API documentation
   - Migration guides
   - Troubleshooting guides

9. **Testing and validation** (Depends on #1-#8)
   - Integration tests
   - Performance benchmarks
   - Stress testing

## Success Metrics

### Reliability Metrics
- ✅ 99.9% task persistence across connections
- ✅ < 5 second recovery time from failures
- ✅ Zero orphaned processes after operations
- ✅ < 1% task execution failure rate

### Performance Metrics
- ✅ < 100ms Docker exec task spawning
- ✅ < 50ms MCP telemetry response time
- ✅ > 1000 concurrent swarm sessions
- ✅ < 10MB memory usage per session

### User Experience Metrics
- ✅ Real-time UI updates (< 200ms delay)
- ✅ Graceful degradation in failure modes
- ✅ Clear error messages and recovery options
- ✅ Consistent API response formats

## Risk Mitigation

### Technical Risks
- **Docker Daemon Failure**: Implement health monitoring and auto-restart
- **Network Partitions**: Use circuit breakers and retry policies
- **Memory Leaks**: Implement automatic cleanup and monitoring
- **Performance Degradation**: Use lazy loading and caching

### Operational Risks
- **Migration Complexity**: Implement gradual rollout with feature flags
- **Backwards Compatibility**: Maintain dual-mode operation during transition
- **Monitoring Gaps**: Add comprehensive logging and metrics
- **Documentation Debt**: Update docs in parallel with implementation

## Conclusion

The hybrid architecture leverages Docker exec for reliable swarm lifecycle management while maintaining MCP/TCP for high-throughput telemetry. This approach resolves process isolation issues while optimising for performance and fault tolerance.

The phased implementation ensures minimal disruption while providing clear migration paths and comprehensive error handling. Success metrics focus on reliability, performance, and user experience improvements.