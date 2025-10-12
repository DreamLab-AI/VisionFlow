// DEPRECATED: Legacy hybrid fault tolerance removed
// Docker exec architecture replaced by HTTP Management API
// Use TaskOrchestratorActor with retry logic instead

/*
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Mutex};
use tokio::time::{sleep, timeout, Instant};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use chrono::{DateTime, Utc};
use log::{info, warn, error, debug};
use std::collections::HashMap;

use crate::utils::docker_hive_mind::{DockerHiveMind, SessionInfo, ContainerHealth};
use crate::utils::mcp_connection::MCPConnectionPool;

/// Network failure types that can occur in the hybrid system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkFailure {
    ContainerDown,
    ProcessHung,
    NetworkPartition,
    ResourceExhaustion,
    MCPConnectionLost,
    DockerDaemonUnreachable,
    HighLatency(u64),
    MemoryExhaustion,
    DiskSpaceLow,
}

/// Recovery actions that can be taken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    RestartContainer,
    RestartDocker,
    KillAndRespawn,
    WaitAndRetry(Duration),
    CleanupAndRestart,
    FallbackToMCP,
    FallbackToDocker,
    GracefulDegradation,
    EmergencyShutdown,
    NoAction,
}

/// Circuit breaker states for managing failing services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,     // Normal operation
    Open,       // Failing - block requests
    HalfOpen,   // Testing if service recovered
}

/// Exponential backoff configuration
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
    current_delay: Duration,
    max_retries: u32,
    current_attempt: u32,
}

impl ExponentialBackoff {
    pub fn new(initial_delay: Duration, max_delay: Duration, multiplier: f64, max_retries: u32) -> Self {
        Self {
            initial_delay,
            max_delay,
            multiplier,
            current_delay: initial_delay,
            max_retries,
            current_attempt: 0,
        }
    }

    pub fn next_delay(&mut self) -> Option<Duration> {
        if self.current_attempt >= self.max_retries {
            return None;
        }

        let delay = self.current_delay;
        self.current_attempt += 1;

        // Calculate next delay with exponential backoff
        let next_delay_ms = (self.current_delay.as_millis() as f64 * self.multiplier) as u64;
        self.current_delay = Duration::from_millis(next_delay_ms.min(self.max_delay.as_millis() as u64));

        Some(delay)
    }

    pub fn reset(&mut self) {
        self.current_delay = self.initial_delay;
        self.current_attempt = 0;
    }
}

/// Circuit breaker for protecting against cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<Mutex<u32>>,
    failure_threshold: u32,
    recovery_timeout: Duration,
    last_failure: Arc<RwLock<Option<Instant>>>,
    success_threshold: u32,  // Half-open -> closed
    half_open_success_count: Arc<Mutex<u32>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_timeout: Duration, success_threshold: u32) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(Mutex::new(0)),
            failure_threshold,
            recovery_timeout,
            last_failure: Arc::new(RwLock::new(None)),
            success_threshold,
            half_open_success_count: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn can_execute(&self) -> bool {
        let state = self.state.read().await;
        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                if let Some(last_failure) = *self.last_failure.read().await {
                    if last_failure.elapsed() >= self.recovery_timeout {
                        drop(state);
                        let mut state = self.state.write().await;
                        *state = CircuitState::HalfOpen;
                        let mut success_count = self.half_open_success_count.lock().await;
                        *success_count = 0;
                        return true;
                    }
                }
                false
            },
            CircuitState::HalfOpen => true,
        }
    }

    pub async fn record_success(&self) {
        let state = self.state.read().await;
        match *state {
            CircuitState::Closed => {
                // Reset failure count
                let mut failure_count = self.failure_count.lock().await;
                *failure_count = 0;
            },
            CircuitState::HalfOpen => {
                let mut success_count = self.half_open_success_count.lock().await;
                *success_count += 1;

                if *success_count >= self.success_threshold {
                    drop(state);
                    let mut state = self.state.write().await;
                    *state = CircuitState::Closed;
                    let mut failure_count = self.failure_count.lock().await;
                    *failure_count = 0;
                }
            },
            _ => {}
        }
    }

    pub async fn record_failure(&self) {
        let mut failure_count = self.failure_count.lock().await;
        *failure_count += 1;

        let mut last_failure = self.last_failure.write().await;
        *last_failure = Some(Instant::now());

        if *failure_count >= self.failure_threshold {
            drop(failure_count);
            let mut state = self.state.write().await;
            *state = CircuitState::Open;
        }
    }

    pub async fn get_state(&self) -> CircuitState {
        self.state.read().await.clone()
    }
}

/// Container health checker with automatic recovery
#[derive(Debug)]
pub struct ContainerHealthChecker {
    container_name: String,
    health_check_interval: Duration,
    unhealthy_threshold: u32,
    recovery_strategy: RecoveryStrategy,
}

#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    RestartContainer,
    RestartDocker,
    RecreateContainer,
    FailoverToSecondary,
}

impl ContainerHealthChecker {
    pub fn new(container_name: String, health_check_interval: Duration, unhealthy_threshold: u32) -> Self {
        Self {
            container_name,
            health_check_interval,
            unhealthy_threshold,
            recovery_strategy: RecoveryStrategy::RestartContainer,
        }
    }

    pub async fn continuous_health_monitoring(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut consecutive_failures = 0u32;
        let mut backoff = ExponentialBackoff::new(
            Duration::from_secs(1),
            Duration::from_secs(60),
            2.0,
            10
        );

        loop {
            match self.check_container_health().await {
                Ok(health) => {
                    if health.is_healthy() {
                        consecutive_failures = 0;
                        backoff.reset();
                        debug!("Container {} is healthy", self.container_name);
                    } else {
                        consecutive_failures += 1;
                        warn!("Container {} unhealthy (attempt {})", self.container_name, consecutive_failures);

                        if consecutive_failures >= self.unhealthy_threshold {
                            error!("Container {} critically unhealthy, attempting recovery", self.container_name);
                            if let Err(e) = self.attempt_recovery().await {
                                error!("Recovery failed for container {}: {}", self.container_name, e);
                            }
                            consecutive_failures = 0; // Reset after recovery attempt
                        }
                    }
                },
                Err(e) => {
                    consecutive_failures += 1;
                    error!("Health check failed for container {}: {}", self.container_name, e);

                    if consecutive_failures >= self.unhealthy_threshold {
                        if let Err(recovery_err) = self.attempt_recovery().await {
                            error!("Emergency recovery failed: {}", recovery_err);
                        }
                        consecutive_failures = 0;
                    }
                }
            }

            // Wait before next check (with backoff if failing)
            let delay = if consecutive_failures > 0 {
                backoff.next_delay().unwrap_or(self.health_check_interval)
            } else {
                self.health_check_interval
            };

            sleep(delay).await;
        }
    }

    async fn check_container_health(&self) -> Result<ExtendedContainerHealth, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::process::Command;

        let start_time = Instant::now();

        // Check container status
        let mut cmd = Command::new("docker");
        cmd.args(&["inspect", &self.container_name, "--format", "{{json .State}}"]);

        let output = timeout(Duration::from_secs(10), cmd.output()).await??;
        let state_json = String::from_utf8_lossy(&output.stdout);

        let container_state: serde_json::Value = serde_json::from_str(&state_json)?;
        let is_running = container_state["Running"].as_bool().unwrap_or(false);
        let exit_code = container_state["ExitCode"].as_i64().unwrap_or(-1);

        // Get resource usage
        let (cpu_usage, memory_usage) = self.get_resource_usage().await?;

        // Check disk space
        let disk_space = self.check_disk_space().await?;

        // Test network connectivity
        let network_latency = self.test_network_connectivity().await;

        Ok(ExtendedContainerHealth {
            is_running,
            exit_code: exit_code as i32,
            cpu_usage,
            memory_usage,
            disk_space_gb: disk_space,
            network_latency_ms: network_latency,
            response_time_ms: start_time.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }

    async fn get_resource_usage(&self) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
        use tokio::process::Command;

        let mut cmd = Command::new("docker");
        cmd.args(&["stats", &self.container_name, "--no-stream", "--format", "{{.CPUPerc}},{{.MemPerc}}"]);

        let output = timeout(Duration::from_secs(5), cmd.output()).await??;
        let stats = String::from_utf8_lossy(&output.stdout);

        let line = stats.lines().next().unwrap_or("0%,0%");
        let parts: Vec<&str> = line.split(',').collect();

        let cpu = parts[0].trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
        let memory = parts.get(1).unwrap_or(&"0%").trim_end_matches('%').parse::<f64>().unwrap_or(0.0);

        Ok((cpu, memory))
    }

    async fn check_disk_space(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::process::Command;

        let mut cmd = Command::new("docker");
        cmd.args(&["exec", &self.container_name, "df", "-BG", "--output=avail", "/"]);

        let output = timeout(Duration::from_secs(5), cmd.output()).await??;
        let df_output = String::from_utf8_lossy(&output.stdout);

        if let Some(line) = df_output.lines().nth(1) {
            let available = line.trim().trim_end_matches('G').parse::<f64>().unwrap_or(0.0);
            Ok(available)
        } else {
            Ok(0.0)
        }
    }

    async fn test_network_connectivity(&self) -> Option<u64> {
        use tokio::process::Command;

        let start = Instant::now();
        let mut cmd = Command::new("docker");
        cmd.args(&["exec", &self.container_name, "ping", "-c", "1", "-W", "2", "1.1.1.1"]);

        if timeout(Duration::from_secs(5), cmd.output()).await.is_ok() {
            Some(start.elapsed().as_millis() as u64)
        } else {
            None
        }
    }

    async fn attempt_recovery(&self) -> Result<RecoveryAction, Box<dyn std::error::Error + Send + Sync>> {
        match self.recovery_strategy {
            RecoveryStrategy::RestartContainer => {
                info!("Attempting to restart container: {}", self.container_name);
                self.restart_container().await
            },
            RecoveryStrategy::RestartDocker => {
                warn!("Attempting to restart Docker daemon");
                self.restart_docker_daemon().await
            },
            RecoveryStrategy::RecreateContainer => {
                warn!("Attempting to recreate container: {}", self.container_name);
                self.recreate_container().await
            },
            RecoveryStrategy::FailoverToSecondary => {
                error!("Failing over to secondary container");
                self.failover_to_secondary().await
            },
        }
    }

    async fn restart_container(&self) -> Result<RecoveryAction, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::process::Command;

        // Stop container
        let mut stop_cmd = Command::new("docker");
        stop_cmd.args(&["stop", &self.container_name]);
        let _ = timeout(Duration::from_secs(30), stop_cmd.output()).await;

        // Start container
        let mut start_cmd = Command::new("docker");
        start_cmd.args(&["start", &self.container_name]);

        let output = timeout(Duration::from_secs(30), start_cmd.output()).await??;

        if output.status.success() {
            info!("Container {} restarted successfully", self.container_name);
            Ok(RecoveryAction::RestartContainer)
        } else {
            Err(format!("Failed to restart container: {}", String::from_utf8_lossy(&output.stderr)).into())
        }
    }

    async fn restart_docker_daemon(&self) -> Result<RecoveryAction, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::process::Command;

        let mut cmd = Command::new("systemctl");
        cmd.args(&["restart", "docker"]);

        let output = timeout(Duration::from_secs(60), cmd.output()).await??;

        if output.status.success() {
            // Wait for Docker to stabilize
            sleep(Duration::from_secs(10)).await;

            // Restart our container
            self.restart_container().await?;

            info!("Docker daemon restarted successfully");
            Ok(RecoveryAction::RestartDocker)
        } else {
            Err("Failed to restart Docker daemon".into())
        }
    }

    async fn recreate_container(&self) -> Result<RecoveryAction, Box<dyn std::error::Error + Send + Sync>> {
        // This would require knowledge of the container's original run parameters
        // For now, just restart
        self.restart_container().await
    }

    async fn failover_to_secondary(&self) -> Result<RecoveryAction, Box<dyn std::error::Error + Send + Sync>> {
        // This would require a secondary container setup
        Err("Secondary container failover not implemented".into())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ExtendedContainerHealth {
    pub is_running: bool,
    pub exit_code: i32,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_space_gb: f64,
    pub network_latency_ms: Option<u64>,
    pub response_time_ms: u64,
    pub timestamp: DateTime<Utc>,
}

impl ExtendedContainerHealth {
    pub fn is_healthy(&self) -> bool {
        self.is_running &&
        self.exit_code == 0 &&
        self.cpu_usage < 90.0 &&
        self.memory_usage < 90.0 &&
        self.disk_space_gb > 1.0 &&
        self.network_latency_ms.is_some() &&
        self.response_time_ms < 10000
    }
}

/// Main network recovery manager
pub struct NetworkRecoveryManager {
    docker_circuit_breaker: CircuitBreaker,
    mcp_circuit_breaker: CircuitBreaker,
    docker_hive_mind: Arc<DockerHiveMind>,
    mcp_pool: Arc<MCPConnectionPool>,
    health_checker: ContainerHealthChecker,
    recovery_config: RecoveryConfig,
}

#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    pub docker_retry_policy: ExponentialBackoff,
    pub mcp_retry_policy: ExponentialBackoff,
    pub health_check_interval: Duration,
    pub failover_timeout: Duration,
    pub graceful_degradation_threshold: u32,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            docker_retry_policy: ExponentialBackoff::new(
                Duration::from_secs(1),
                Duration::from_secs(60),
                2.0,
                5
            ),
            mcp_retry_policy: ExponentialBackoff::new(
                Duration::from_millis(500),
                Duration::from_secs(30),
                1.5,
                10
            ),
            health_check_interval: Duration::from_secs(30),
            failover_timeout: Duration::from_secs(120),
            graceful_degradation_threshold: 3,
        }
    }
}

impl NetworkRecoveryManager {
    pub fn new(
        docker_hive_mind: DockerHiveMind,
        mcp_pool: MCPConnectionPool,
        container_name: String,
    ) -> Self {
        let recovery_config = RecoveryConfig::default();

        Self {
            docker_circuit_breaker: CircuitBreaker::new(5, Duration::from_secs(60), 3),
            mcp_circuit_breaker: CircuitBreaker::new(10, Duration::from_secs(30), 2),
            docker_hive_mind: Arc::new(docker_hive_mind),
            mcp_pool: Arc::new(mcp_pool),
            health_checker: ContainerHealthChecker::new(
                container_name,
                recovery_config.health_check_interval,
                3
            ),
            recovery_config,
        }
    }

    pub async fn recover_from_failure(&self, failure: NetworkFailure) -> RecoveryAction {
        info!("Attempting recovery from failure: {:?}", failure);

        match failure {
            NetworkFailure::ContainerDown => {
                match self.health_checker.attempt_recovery().await {
                    Ok(action) => action,
                    Err(e) => {
                        error!("Container recovery failed: {}", e);
                        RecoveryAction::EmergencyShutdown
                    }
                }
            },
            NetworkFailure::ProcessHung => {
                self.kill_and_respawn().await
            },
            NetworkFailure::NetworkPartition => {
                self.wait_and_retry(Duration::from_secs(30)).await
            },
            NetworkFailure::ResourceExhaustion => {
                self.cleanup_and_restart().await
            },
            NetworkFailure::MCPConnectionLost => {
                RecoveryAction::FallbackToDocker
            },
            NetworkFailure::DockerDaemonUnreachable => {
                RecoveryAction::FallbackToMCP
            },
            NetworkFailure::HighLatency(latency_ms) => {
                if latency_ms > 5000 {
                    RecoveryAction::GracefulDegradation
                } else {
                    RecoveryAction::NoAction
                }
            },
            NetworkFailure::MemoryExhaustion => {
                self.cleanup_and_restart().await
            },
            NetworkFailure::DiskSpaceLow => {
                self.cleanup_disk_space().await
            },
        }
    }

    async fn kill_and_respawn(&self) -> RecoveryAction {
        info!("Killing hung processes and respawning");

        match self.docker_hive_mind.cleanup_orphaned_processes().await {
            Ok(killed_count) => {
                info!("Killed {} orphaned processes", killed_count);
                RecoveryAction::KillAndRespawn
            },
            Err(e) => {
                error!("Failed to cleanup processes: {}", e);
                RecoveryAction::RestartContainer
            }
        }
    }

    async fn wait_and_retry(&self, delay: Duration) -> RecoveryAction {
        info!("Waiting {:?} before retry due to network partition", delay);
        sleep(delay).await;
        RecoveryAction::WaitAndRetry(delay)
    }

    async fn cleanup_and_restart(&self) -> RecoveryAction {
        info!("Performing cleanup and restart");

        // Cleanup orphaned processes
        let _ = self.docker_hive_mind.cleanup_orphaned_processes().await;

        // Restart container
        match self.health_checker.restart_container().await {
            Ok(_) => RecoveryAction::CleanupAndRestart,
            Err(e) => {
                error!("Cleanup and restart failed: {}", e);
                RecoveryAction::EmergencyShutdown
            }
        }
    }

    async fn cleanup_disk_space(&self) -> RecoveryAction {
        use tokio::process::Command;

        info!("Cleaning up disk space");

        // Clean Docker system
        let mut cleanup_cmd = Command::new("docker");
        cleanup_cmd.args(&["system", "prune", "-f"]);
        let _ = timeout(Duration::from_secs(60), cleanup_cmd.output()).await;

        // Clean logs in container
        let mut log_cleanup_cmd = Command::new("docker");
        log_cleanup_cmd.args(&["exec", "multi-agent-container", "find", "/tmp", "-name", "*.log", "-delete"]);
        let _ = timeout(Duration::from_secs(30), log_cleanup_cmd.output()).await;

        RecoveryAction::CleanupAndRestart
    }

    pub async fn execute_with_circuit_breaker<F, T>(&self,
        service: &str,
        operation: F
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
    where
        F: std::future::Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
    {
        let circuit_breaker = match service {
            "docker" => &self.docker_circuit_breaker,
            "mcp" => &self.mcp_circuit_breaker,
            _ => return Err("Unknown service".into()),
        };

        if !circuit_breaker.can_execute().await {
            return Err(format!("Circuit breaker open for service: {}", service).into());
        }

        match operation.await {
            Ok(result) => {
                circuit_breaker.record_success().await;
                Ok(result)
            },
            Err(e) => {
                circuit_breaker.record_failure().await;
                Err(e)
            }
        }
    }

    pub async fn get_system_health(&self) -> SystemHealthReport {
        let docker_state = self.docker_circuit_breaker.get_state().await;
        let mcp_state = self.mcp_circuit_breaker.get_state().await;

        let container_health = self.health_checker.check_container_health().await.ok();

        SystemHealthReport {
            docker_circuit_state: docker_state.clone(),
            mcp_circuit_state: mcp_state.clone(),
            container_health: container_health.clone(),
            timestamp: Utc::now(),
            overall_status: self.calculate_overall_status(&docker_state, &mcp_state, &container_health),
        }
    }

    fn calculate_overall_status(
        &self,
        docker_state: &CircuitState,
        mcp_state: &CircuitState,
        container_health: &Option<ExtendedContainerHealth>
    ) -> SystemStatus {
        // Check container health first
        if let Some(health) = container_health {
            if !health.is_healthy() {
                return SystemStatus::Critical;
            }
        } else {
            return SystemStatus::Critical;
        }

        // Check circuit states
        match (docker_state, mcp_state) {
            (CircuitState::Closed, CircuitState::Closed) => SystemStatus::Healthy,
            (CircuitState::Closed, CircuitState::HalfOpen) |
            (CircuitState::HalfOpen, CircuitState::Closed) => SystemStatus::Degraded,
            (CircuitState::Open, CircuitState::Closed) |
            (CircuitState::Closed, CircuitState::Open) => SystemStatus::Degraded,
            (CircuitState::Open, CircuitState::Open) => SystemStatus::Critical,
            _ => SystemStatus::Degraded,
        }
    }

    pub async fn start_continuous_monitoring(&self) {
        info!("Starting continuous health monitoring");

        let health_checker = ContainerHealthChecker::new(
            "multi-agent-container".to_string(),
            self.recovery_config.health_check_interval,
            self.recovery_config.graceful_degradation_threshold,
        );

        // This would typically be spawned as a background task
        if let Err(e) = health_checker.continuous_health_monitoring().await {
            error!("Health monitoring failed: {}", e);
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemHealthReport {
    pub docker_circuit_state: CircuitState,
    pub mcp_circuit_state: CircuitState,
    pub container_health: Option<ExtendedContainerHealth>,
    pub timestamp: DateTime<Utc>,
    pub overall_status: SystemStatus,
}

#[derive(Debug, Clone, Serialize)]
pub enum SystemStatus {
    Healthy,
    Degraded,
    Critical,
}

/// State synchronization between Docker and MCP systems
pub struct StateSync {
    docker_hive_mind: Arc<DockerHiveMind>,
    mcp_pool: Arc<MCPConnectionPool>,
    sync_interval: Duration,
    last_sync: Arc<RwLock<DateTime<Utc>>>,
    state_cache: Arc<RwLock<HashMap<String, SyncedState>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncedState {
    pub docker_sessions: Vec<SessionInfo>,
    pub mcp_agents: Option<Value>,
    pub inconsistencies: Vec<StateInconsistency>,
    pub last_sync: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StateInconsistency {
    pub description: String,
    pub severity: InconsistencySeverity,
    pub suggested_action: String,
}

#[derive(Debug, Clone, Serialize)]
pub enum InconsistencySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl StateSync {
    pub fn new(docker_hive_mind: DockerHiveMind, mcp_pool: MCPConnectionPool) -> Self {
        Self {
            docker_hive_mind: Arc::new(docker_hive_mind),
            mcp_pool: Arc::new(mcp_pool),
            sync_interval: Duration::from_secs(30),
            last_sync: Arc::new(RwLock::new(Utc::now() - chrono::Duration::hours(1))),
            state_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn sync_swarm_states(&self) -> Result<SyncReport, Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting state synchronization");

        // Get Docker sessions
        let docker_sessions = match self.docker_hive_mind.get_sessions().await {
            Ok(sessions) => sessions,
            Err(e) => {
                warn!("Failed to get Docker sessions: {}", e);
                Vec::new()
            }
        };

        // Get MCP agents
        let mcp_agents = match self.mcp_pool.execute_command("sync", "tools/call", json!({
            "name": "agent_list",
            "arguments": { "filter": "all" }
        })).await {
            Ok(agents) => Some(agents),
            Err(e) => {
                debug!("MCP agents unavailable: {}", e);
                None
            }
        };

        // Detect inconsistencies
        let inconsistencies = self.detect_inconsistencies(&docker_sessions, &mcp_agents).await;

        // Update cache
        let synced_state = SyncedState {
            docker_sessions: docker_sessions.clone(),
            mcp_agents: mcp_agents.clone(),
            inconsistencies: inconsistencies.clone(),
            last_sync: Utc::now(),
        };

        {
            let mut cache = self.state_cache.write().await;
            cache.insert("global".to_string(), synced_state);
        }

        {
            let mut last_sync = self.last_sync.write().await;
            *last_sync = Utc::now();
        }

        Ok(SyncReport {
            docker_session_count: docker_sessions.len(),
            mcp_available: mcp_agents.is_some(),
            inconsistencies_found: inconsistencies.len(),
            sync_time: Utc::now(),
            inconsistencies,
        })
    }

    async fn detect_inconsistencies(
        &self,
        docker_sessions: &[SessionInfo],
        mcp_agents: &Option<Value>,
    ) -> Vec<StateInconsistency> {
        let mut inconsistencies = Vec::new();

        // Check if sessions exist in Docker but not in MCP telemetry
        if let Some(agents) = mcp_agents {
            // This is a simplified check - in reality would need more complex comparison
            if docker_sessions.len() > 0 && agents.as_array().map_or(true, |arr| arr.is_empty()) {
                inconsistencies.push(StateInconsistency {
                    description: "Docker sessions exist but MCP shows no agents".to_string(),
                    severity: InconsistencySeverity::Medium,
                    suggested_action: "Restart MCP telemetry collection".to_string(),
                });
            }
        } else {
            if !docker_sessions.is_empty() {
                inconsistencies.push(StateInconsistency {
                    description: "Docker sessions active but MCP unavailable".to_string(),
                    severity: InconsistencySeverity::High,
                    suggested_action: "Check MCP connection and restart if needed".to_string(),
                });
            }
        }

        inconsistencies
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncReport {
    pub docker_session_count: usize,
    pub mcp_available: bool,
    pub inconsistencies_found: usize,
    pub sync_time: DateTime<Utc>,
    pub inconsistencies: Vec<StateInconsistency>,
}

/// Create a complete fault tolerance system for the hybrid architecture
pub fn create_fault_tolerance_system(
    docker_hive_mind: DockerHiveMind,
    mcp_pool: MCPConnectionPool,
) -> NetworkRecoveryManager {
    NetworkRecoveryManager::new(
        docker_hive_mind,
        mcp_pool,
        "multi-agent-container".to_string(),
    )
}*/
