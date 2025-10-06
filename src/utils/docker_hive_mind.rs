use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::path::PathBuf;
use tokio::sync::{RwLock, Mutex};
use tokio::process::Command;
use tokio::time::{timeout, sleep};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use chrono::{DateTime, Utc};
use log::{info, warn, error, debug};
use uuid::Uuid;

/// Docker-based hive mind orchestration for multi-agent systems
/// This module provides direct Docker exec access to the claude-flow hive-mind
/// system, bypassing TCP/MCP process isolation issues.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub priority: SwarmPriority,
    pub strategy: SwarmStrategy,
    pub max_workers: Option<u32>,
    pub consensus_type: Option<String>,
    pub memory_size_mb: Option<u32>,
    pub auto_scale: bool,
    pub encryption: bool,
    pub monitor: bool,
    pub verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmStrategy {
    Strategic,  // Adaptive queen
    Tactical,   // High priority
    Adaptive,   // Dynamic adjustment
    HiveMind,   // Default
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,  // UUID from session manager
    pub task_description: String,
    pub status: SwarmStatus,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub metrics: SwarmMetrics,
    pub config: SwarmConfig,
    pub workers: Vec<WorkerAgent>,
    // Session manager paths
    pub working_dir: std::path::PathBuf,
    pub output_dir: std::path::PathBuf,
    pub database_path: std::path::PathBuf,
    pub log_file: std::path::PathBuf,
    // Optional swarm ID (when available from claude-flow)
    pub swarm_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmStatus {
    Spawning,
    Active,
    Paused,
    Completing,
    Completed,
    Failed,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
    pub active_workers: u32,
    pub completed_tasks: u32,
    pub failed_tasks: u32,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub network_io_mb: f64,
    pub uptime_seconds: u64,
    pub consensus_decisions: u32,
    pub last_activity: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerAgent {
    pub agent_id: String,
    pub agent_type: String,
    pub status: String,
    pub tasks_completed: u32,
    pub last_activity: DateTime<Utc>,
}

/// Session manager metadata response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagerMetadata {
    pub session_id: String,
    pub task: String,
    pub priority: String,
    pub created: DateTime<Utc>,
    pub updated: Option<DateTime<Utc>>,
    pub status: String,
    pub working_dir: PathBuf,
    pub output_dir: PathBuf,
    pub database: PathBuf,
    pub log_file: PathBuf,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// Session registry from session manager list command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRegistry {
    pub sessions: HashMap<String, SessionRegistryEntry>,
    pub created: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRegistryEntry {
    pub task: String,
    pub status: String,
    pub created: DateTime<Utc>,
    pub updated: DateTime<Utc>,
    pub dir: String,
}

#[derive(Debug, Clone)]
pub struct HealthMonitor {
    container_name: String,
    health_cache: Arc<RwLock<Option<ContainerHealth>>>,
    last_check: Arc<RwLock<DateTime<Utc>>>,
    check_interval: Duration,
}

#[derive(Debug, Clone)]
pub struct ContainerHealth {
    pub is_running: bool,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_healthy: bool,
    pub disk_space_gb: f64,
    pub last_response_ms: u64,
}

/// Main Docker hive mind orchestrator
#[derive(Clone, Debug)]
pub struct DockerHiveMind {
    container_name: String,
    session_manager_script: String,
    session_cache: Arc<RwLock<HashMap<String, SessionInfo>>>,
    health_monitor: HealthMonitor,
    config: DockerHiveMindConfig,
}

#[derive(Debug, Clone)]
pub struct DockerHiveMindConfig {
    pub command_timeout: Duration,
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub health_check_interval: Duration,
    pub session_cache_ttl: Duration,
    pub cleanup_interval: Duration,
}

impl Default for DockerHiveMindConfig {
    fn default() -> Self {
        Self {
            command_timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(1000),
            health_check_interval: Duration::from_secs(30),
            session_cache_ttl: Duration::from_secs(300),
            cleanup_interval: Duration::from_secs(600),
        }
    }
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            priority: SwarmPriority::Medium,
            strategy: SwarmStrategy::HiveMind,
            max_workers: Some(8),
            consensus_type: Some("majority".to_string()),
            memory_size_mb: Some(100),
            auto_scale: true,
            encryption: false,
            monitor: true,
            verbose: false,
        }
    }
}

impl DockerHiveMind {
    /// Create a new Docker hive mind orchestrator
    pub fn new(container_name: String) -> Self {
        let config = DockerHiveMindConfig::default();
        let health_monitor = HealthMonitor::new(container_name.clone());

        Self {
            container_name: container_name.clone(),
            session_manager_script: "/app/scripts/hive-session-manager.sh".to_string(),
            session_cache: Arc::new(RwLock::new(HashMap::new())),
            health_monitor,
            config,
        }
    }

    /// Create with custom configuration
    pub fn with_config(container_name: String, config: DockerHiveMindConfig) -> Self {
        let health_monitor = HealthMonitor::new(container_name.clone());

        Self {
            container_name: container_name.clone(),
            session_manager_script: "/app/scripts/hive-session-manager.sh".to_string(),
            session_cache: Arc::new(RwLock::new(HashMap::new())),
            health_monitor,
            config,
        }
    }

    /// Create a new session via session manager
    pub async fn create_session(
        &self,
        task: &str,
        priority: &str,
        metadata: Option<Value>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        info!("Creating session for task: {}", task);

        // 1. Check container health
        self.health_monitor.check_container_health().await?;

        // 2. Prepare metadata JSON
        let metadata_json = metadata
            .map(|m| serde_json::to_string(&m).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());

        // 3. Build Docker command to create session
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.session_manager_script,
            "create",
            task,
            priority,
            &metadata_json,
        ]);

        // 4. Execute with retry
        let output = self.execute_with_retries(&mut cmd).await?;

        // 5. Extract UUID from stdout
        let uuid = String::from_utf8_lossy(&output.stdout).trim().to_string();

        if uuid.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to create session: {}", stderr).into());
        }

        info!("Session created with UUID: {}", uuid);
        Ok(uuid)
    }

    /// Start a session via session manager (spawns hive-mind in background)
    pub async fn start_session(&self, uuid: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting session: {}", uuid);

        // Build Docker command to start session in detached mode
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec", "-d",
            &self.container_name,
            &self.session_manager_script,
            "start",
            uuid,
        ]);

        // Execute
        let output = self.execute_with_retries(&mut cmd).await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to start session {}: {}", uuid, stderr).into());
        }

        info!("Session {} started successfully", uuid);
        Ok(())
    }

    /// Spawn a new swarm with the given task description (combines create + start)
    pub async fn spawn_swarm(&self, task: &str, config: SwarmConfig) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        info!("Spawning swarm for task: {}", task);

        // Map priority to string
        let priority = match config.priority {
            SwarmPriority::Critical => "high",
            SwarmPriority::High => "high",
            SwarmPriority::Medium => "medium",
            SwarmPriority::Low => "low",
        };

        // Build metadata
        let metadata = json!({
            "strategy": format!("{:?}", config.strategy),
            "max_workers": config.max_workers,
            "consensus_type": config.consensus_type,
            "memory_size_mb": config.memory_size_mb,
            "auto_scale": config.auto_scale,
            "encryption": config.encryption,
            "monitor": config.monitor,
            "verbose": config.verbose,
        });

        // 1. Create session
        let uuid = self.create_session(task, priority, Some(metadata)).await?;

        // 2. Start session
        self.start_session(&uuid).await?;

        // 3. Get session metadata to populate cache
        let session_metadata = self.get_session_metadata(&uuid).await?;

        // 4. Cache session info
        let session_info = SessionInfo {
            session_id: uuid.clone(),
            task_description: task.to_string(),
            status: SwarmStatus::Spawning,
            created_at: session_metadata.created,
            last_activity: session_metadata.updated.unwrap_or(Utc::now()),
            metrics: SwarmMetrics::default(),
            config: config.clone(),
            workers: Vec::new(),
            working_dir: session_metadata.working_dir,
            output_dir: session_metadata.output_dir,
            database_path: session_metadata.database,
            log_file: session_metadata.log_file,
            swarm_id: None, // Will be populated later via MCP
        };

        {
            let mut cache = self.session_cache.write().await;
            cache.insert(uuid.clone(), session_info);
        }

        info!("Swarm spawned successfully with UUID: {}", uuid);
        Ok(uuid)
    }

    /// Get session metadata from session manager
    pub async fn get_session_metadata(&self, uuid: &str) -> Result<SessionManagerMetadata, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Getting metadata for session: {}", uuid);

        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.session_manager_script,
            "get",
            uuid,
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to get session metadata for {}: {}", uuid, stderr).into());
        }

        let json = String::from_utf8_lossy(&output.stdout);
        let metadata: SessionManagerMetadata = serde_json::from_str(&json)?;

        Ok(metadata)
    }

    /// Get session output directory
    pub async fn get_session_output_dir(&self, uuid: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.session_manager_script,
            "output-dir",
            uuid,
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            return Err(format!("Failed to get output dir for session {}", uuid).into());
        }

        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(std::path::PathBuf::from(path))
    }

    /// Get session log file path
    pub async fn get_session_log_file(&self, uuid: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.session_manager_script,
            "log",
            uuid,
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            return Err(format!("Failed to get log file for session {}", uuid).into());
        }

        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(std::path::PathBuf::from(path))
    }

    /// Get all active sessions from session manager
    pub async fn get_sessions(&self) -> Result<Vec<SessionInfo>, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Fetching all sessions from session manager");

        // 1. Query session manager for list
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.session_manager_script,
            "list",
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to get sessions: {}", stderr).into());
        }

        // 2. Parse session registry JSON
        let json = String::from_utf8_lossy(&output.stdout);
        let registry: SessionRegistry = serde_json::from_str(&json)?;

        // 3. Convert to SessionInfo vec
        let mut sessions = Vec::new();
        for (uuid, session_data) in registry.sessions {
            // Get from cache if available, otherwise create basic info
            let cached = {
                let cache = self.session_cache.read().await;
                cache.get(&uuid).cloned()
            };

            let session_info = if let Some(mut cached_session) = cached {
                // Update status from registry
                cached_session.status = self.parse_status_string(&session_data.status);
                cached_session.last_activity = session_data.updated;
                cached_session
            } else {
                // Create basic session info from registry
                SessionInfo {
                    session_id: uuid.clone(),
                    task_description: session_data.task,
                    status: self.parse_status_string(&session_data.status),
                    created_at: session_data.created,
                    last_activity: session_data.updated,
                    metrics: SwarmMetrics::default(),
                    config: SwarmConfig::default(),
                    workers: Vec::new(),
                    working_dir: std::path::PathBuf::from(&session_data.dir),
                    output_dir: std::path::PathBuf::from(format!("/workspace/ext/hive-sessions/{}", uuid)),
                    database_path: std::path::PathBuf::from(format!("{}/.swarm/memory.db", session_data.dir)),
                    log_file: std::path::PathBuf::from(format!("/var/log/multi-agent/hive-{}.log", uuid)),
                    swarm_id: None,
                }
            };

            sessions.push(session_info);
        }

        // 4. Update cache
        self.update_session_cache(&sessions).await;

        Ok(sessions)
    }

    /// Get status of a specific session by UUID
    pub async fn get_session_status(&self, uuid: &str) -> Result<SwarmStatus, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Getting status for session: {}", uuid);

        // 1. Check cache first
        {
            let cache = self.session_cache.read().await;
            if let Some(session) = cache.get(uuid) {
                if session.last_activity > Utc::now() - chrono::Duration::seconds(30) {
                    return Ok(session.status.clone());
                }
            }
        }

        // 2. Query session manager status
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.session_manager_script,
            "status",
            uuid,
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            return Ok(SwarmStatus::Unknown);
        }

        // 3. Parse status string
        let status_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let status = self.parse_status_string(&status_str);

        // 4. Update cache
        {
            let mut cache = self.session_cache.write().await;
            if let Some(session) = cache.get_mut(uuid) {
                session.status = status.clone();
                session.last_activity = Utc::now();
            }
        }

        Ok(status)
    }

    /// Get status of a specific swarm (alias for session status, maintained for compatibility)
    pub async fn get_swarm_status(&self, swarm_id_or_uuid: &str) -> Result<SwarmStatus, Box<dyn std::error::Error + Send + Sync>> {
        self.get_session_status(swarm_id_or_uuid).await
    }

    /// Parse status string from session manager into SwarmStatus enum
    fn parse_status_string(&self, status: &str) -> SwarmStatus {
        match status.to_lowercase().as_str() {
            "created" => SwarmStatus::Spawning,
            "starting" => SwarmStatus::Spawning,
            "running" => SwarmStatus::Active,
            "paused" => SwarmStatus::Paused,
            "completed" => SwarmStatus::Completed,
            "failed" => SwarmStatus::Failed,
            "completing" => SwarmStatus::Completing,
            _ => SwarmStatus::Unknown,
        }
    }

    /// Stop a session (UUID-based)
    pub async fn stop_session(&self, uuid: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Stopping session: {}", uuid);

        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.session_manager_script,
            "stop",
            uuid,
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to stop session {}: {}", uuid, stderr).into());
        }

        // Update cache
        {
            let mut cache = self.session_cache.write().await;
            if let Some(session) = cache.get_mut(uuid) {
                session.status = SwarmStatus::Completing;
                session.last_activity = Utc::now();
            }
        }

        info!("Session {} stopped successfully", uuid);
        Ok(())
    }

    /// Stop a swarm (alias for stop_session, maintained for compatibility)
    pub async fn stop_swarm(&self, swarm_id_or_uuid: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.stop_session(swarm_id_or_uuid).await
    }

    /// Pause swarm (stub - session manager doesn't support pause yet)
    pub async fn pause_swarm(&self, _swarm_id_or_uuid: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        warn!("pause_swarm is not implemented for session manager");
        Ok(())
    }

    /// Resume swarm (stub - session manager doesn't support resume yet)
    pub async fn resume_swarm(&self, _swarm_id_or_uuid: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        warn!("resume_swarm is not implemented for session manager");
        Ok(())
    }

    /// Get swarm metrics (stub - should use MCP bridge instead)
    pub async fn get_swarm_metrics(&self, _swarm_id: &str) -> Result<SwarmMetrics, Box<dyn std::error::Error + Send + Sync>> {
        warn!("get_swarm_metrics is deprecated - use McpSessionBridge instead");
        Ok(SwarmMetrics::default())
    }

    /// Cleanup orphaned processes and stale sessions
    pub async fn cleanup_orphaned_processes(&self) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        info!("Cleaning up orphaned processes");

        let mut cleaned = 0u32;

        // 1. Get list of all processes in container
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            "ps", "aux"
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            return Ok(0);
        }

        let processes = String::from_utf8_lossy(&output.stdout);

        // 2. Find zombie claude-flow processes
        for line in processes.lines() {
            if line.contains("claude-flow") && line.contains("<defunct>") {
                if let Some(pid) = self.extract_pid_from_line(line) {
                    if self.kill_process_in_container(pid).await.is_ok() {
                        cleaned += 1;
                    }
                }
            }
        }

        // 3. Clean up stale cache entries
        self.cleanup_stale_cache().await;

        info!("Cleaned up {} orphaned processes", cleaned);
        Ok(cleaned)
    }

    /// Check container and network health
    pub async fn health_check(&self) -> Result<ContainerHealth, Box<dyn std::error::Error + Send + Sync>> {
        self.health_monitor.check_container_health().await
    }

    // Private helper methods

    async fn execute_with_retries(&self, cmd: &mut Command) -> Result<std::process::Output, Box<dyn std::error::Error + Send + Sync>> {
        let mut last_error = None;

        for attempt in 1..=self.config.max_retries {
            debug!("Execute attempt {}/{}", attempt, self.config.max_retries);

            match timeout(self.config.command_timeout, cmd.output()).await {
                Ok(Ok(output)) => {
                    if output.status.success() {
                        return Ok(output);
                    } else {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        last_error = Some(format!("Command failed: {}", stderr));
                    }
                },
                Ok(Err(e)) => {
                    last_error = Some(format!("IO error: {}", e));
                },
                Err(_) => {
                    last_error = Some("Command timeout".to_string());
                }
            }

            if attempt < self.config.max_retries {
                sleep(self.config.retry_delay).await;
            }
        }

        Err(last_error.unwrap_or("Unknown error".to_string()).into())
    }

    async fn update_session_cache(&self, sessions: &[SessionInfo]) {
        let mut cache = self.session_cache.write().await;

        // Update existing sessions
        for session in sessions {
            cache.insert(session.session_id.clone(), session.clone());
        }
    }

    async fn cleanup_stale_cache(&self) {
        let mut cache = self.session_cache.write().await;
        let cutoff = Utc::now() - chrono::Duration::from_std(self.config.session_cache_ttl).unwrap();

        cache.retain(|_, session| session.last_activity > cutoff);
    }

    fn extract_pid_from_line(&self, line: &str) -> Option<u32> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            parts[1].parse::<u32>().ok()
        } else {
            None
        }
    }

    async fn kill_process_in_container(&self, pid: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            "kill", "-9", &pid.to_string()
        ]);

        timeout(Duration::from_secs(5), cmd.output()).await??;
        Ok(())
    }

    /// Terminate swarm (alias for stop_swarm for compatibility)
    pub async fn terminate_swarm(&self, swarm_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.stop_swarm(swarm_id).await
    }
}

impl Default for SwarmMetrics {
    fn default() -> Self {
        Self {
            active_workers: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            network_io_mb: 0.0,
            uptime_seconds: 0,
            consensus_decisions: 0,
            last_activity: Utc::now(),
        }
    }
}

impl HealthMonitor {
    pub fn new(container_name: String) -> Self {
        Self {
            container_name,
            health_cache: Arc::new(RwLock::new(None)),
            last_check: Arc::new(RwLock::new(Utc::now() - chrono::Duration::hours(1))),
            check_interval: Duration::from_secs(30),
        }
    }

    pub async fn check_container_health(&self) -> Result<ContainerHealth, Box<dyn std::error::Error + Send + Sync>> {
        // Check if we need to refresh health data
        let should_check = {
            let last_check = self.last_check.read().await;
            Utc::now() - *last_check > chrono::Duration::from_std(self.check_interval).unwrap()
        };

        if !should_check {
            if let Some(cached) = self.health_cache.read().await.as_ref() {
                return Ok(cached.clone());
            }
        }

        // Perform health check
        let health = self.perform_health_check().await?;

        // Update cache
        {
            let mut cache = self.health_cache.write().await;
            *cache = Some(health.clone());
        }
        {
            let mut last_check = self.last_check.write().await;
            *last_check = Utc::now();
        }

        Ok(health)
    }

    async fn perform_health_check(&self) -> Result<ContainerHealth, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();

        // Check if container is running
        let mut cmd = Command::new("docker");
        cmd.args(&["inspect", &self.container_name, "--format", "{{.State.Running}}"]);

        let output = timeout(Duration::from_secs(5), cmd.output()).await??;
        let is_running = String::from_utf8_lossy(&output.stdout).trim() == "true";

        if !is_running {
            return Ok(ContainerHealth {
                is_running: false,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                network_healthy: false,
                disk_space_gb: 0.0,
                last_response_ms: start_time.elapsed().as_millis() as u64,
            });
        }

        // Get container stats
        let mut stats_cmd = Command::new("docker");
        stats_cmd.args(&["stats", &self.container_name, "--no-stream", "--format", "table {{.CPUPerc}},{{.MemUsage}}"]);

        let stats_output = timeout(Duration::from_secs(5), stats_cmd.output()).await??;
        let stats = String::from_utf8_lossy(&stats_output.stdout);

        let (cpu_usage, memory_usage) = self.parse_stats(&stats);

        // Check network connectivity with ping
        let network_healthy = self.check_network_connectivity().await;

        // Check disk space
        let disk_space = self.check_disk_space().await.unwrap_or(0.0);

        Ok(ContainerHealth {
            is_running: true,
            cpu_usage,
            memory_usage,
            network_healthy,
            disk_space_gb: disk_space,
            last_response_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    fn parse_stats(&self, stats: &str) -> (f64, f64) {
        // Parse Docker stats output
        // Format: "0.50%,100MiB / 8GiB"

        for line in stats.lines().skip(1) { // Skip header
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let cpu = parts[0].trim_end_matches('%').parse::<f64>().unwrap_or(0.0);

                // Parse memory (e.g., "100MiB / 8GiB")
                let memory_part = parts[1].split(" / ").next().unwrap_or("0");
                let memory = self.parse_memory_value(memory_part);

                return (cpu, memory);
            }
        }

        (0.0, 0.0)
    }

    fn parse_memory_value(&self, value: &str) -> f64 {
        let value = value.trim();
        if value.ends_with("GiB") {
            value.trim_end_matches("GiB").parse::<f64>().unwrap_or(0.0) * 1024.0
        } else if value.ends_with("MiB") {
            value.trim_end_matches("MiB").parse::<f64>().unwrap_or(0.0)
        } else if value.ends_with("KiB") {
            value.trim_end_matches("KiB").parse::<f64>().unwrap_or(0.0) / 1024.0
        } else {
            0.0
        }
    }

    async fn check_network_connectivity(&self) -> bool {
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec", &self.container_name,
            "ping", "-c", "1", "-W", "2", "1.1.1.1"
        ]);

        timeout(Duration::from_secs(5), cmd.output())
            .await
            .map(|result| result.map(|output| output.status.success()).unwrap_or(false))
            .unwrap_or(false)
    }

    async fn check_disk_space(&self) -> Option<f64> {
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec", &self.container_name,
            "df", "-BG", "/"
        ]);

        if let Ok(output) = timeout(Duration::from_secs(5), cmd.output()).await.unwrap_or_else(|_| Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout"))) {
            let output_str = String::from_utf8_lossy(&output.stdout);
            // Parse df output for available space
            for line in output_str.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(available) = parts[3].trim_end_matches('G').parse::<f64>() {
                        return Some(available);
                    }
                }
            }
        }

        None
    }
}

/// Convenience function to create a docker hive mind instance
pub fn create_docker_hive_mind() -> DockerHiveMind {
    DockerHiveMind::new("multi-agent-container".to_string())
}

/// Convenience function with multi-agent container
pub async fn spawn_task_docker(
    task: &str,
    priority: Option<SwarmPriority>,
    strategy: Option<SwarmStrategy>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let hive_mind = create_docker_hive_mind();

    let config = SwarmConfig {
        priority: priority.unwrap_or(SwarmPriority::Medium),
        strategy: strategy.unwrap_or(SwarmStrategy::HiveMind),
        ..Default::default()
    };

    hive_mind.spawn_swarm(task, config).await
}

/// Get all active swarms
pub async fn get_active_swarms() -> Result<Vec<SessionInfo>, Box<dyn std::error::Error + Send + Sync>> {
    let hive_mind = create_docker_hive_mind();
    hive_mind.get_sessions().await
}

/// Check container health
pub async fn check_system_health() -> Result<ContainerHealth, Box<dyn std::error::Error + Send + Sync>> {
    let hive_mind = create_docker_hive_mind();
    hive_mind.health_check().await
}