use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
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
    pub session_id: String,
    pub task_description: String,
    pub status: SwarmStatus,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub metrics: SwarmMetrics,
    pub config: SwarmConfig,
    pub workers: Vec<WorkerAgent>,
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
    claude_flow_path: String,
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
            claude_flow_path: "/app/node_modules/.bin/claude-flow".to_string(),
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
            claude_flow_path: "/app/node_modules/.bin/claude-flow".to_string(),
            session_cache: Arc::new(RwLock::new(HashMap::new())),
            health_monitor,
            config,
        }
    }

    /// Spawn a new swarm with the given task description
    pub async fn spawn_swarm(&self, task: &str, config: SwarmConfig) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        info!("Spawning swarm for task: {}", task);

        // 1. Check container health
        self.health_monitor.check_container_health().await?;

        // 2. Generate session ID
        let session_id = format!("swarm-{}-{}",
            Utc::now().timestamp_millis(),
            &Uuid::new_v4().to_string()[..8]
        );

        // 3. Build Docker command
        let mut cmd = self.build_spawn_command(task, &config);

        // 4. Execute with retry logic
        let output = self.execute_with_retries(&mut cmd).await?;

        // 5. Parse response and extract actual session ID if available
        let actual_session_id = self.extract_session_id(&output)
            .unwrap_or_else(|_| session_id.clone());

        // 6. Cache session info
        let session_info = SessionInfo {
            session_id: actual_session_id.clone(),
            task_description: task.to_string(),
            status: SwarmStatus::Spawning,
            created_at: Utc::now(),
            last_activity: Utc::now(),
            metrics: SwarmMetrics::default(),
            config: config.clone(),
            workers: Vec::new(),
        };

        {
            let mut cache = self.session_cache.write().await;
            cache.insert(actual_session_id.clone(), session_info);
        }

        info!("Swarm spawned successfully with ID: {}", actual_session_id);
        Ok(actual_session_id)
    }

    /// Get all active sessions
    pub async fn get_sessions(&self) -> Result<Vec<SessionInfo>, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Fetching all hive-mind sessions");

        // 1. Query Docker container for live sessions
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "sessions"
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to get sessions: {}", stderr).into());
        }

        // 2. Parse output and update cache
        let sessions = self.parse_sessions_output(&output).await?;

        // 3. Update cache with live data
        self.update_session_cache(&sessions).await;

        // 4. Return combined cache + live data
        Ok(sessions)
    }

    /// Get status of a specific swarm
    pub async fn get_swarm_status(&self, swarm_id: &str) -> Result<SwarmStatus, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Getting status for swarm: {}", swarm_id);

        // 1. Check cache first
        {
            let cache = self.session_cache.read().await;
            if let Some(session) = cache.get(swarm_id) {
                if session.last_activity > Utc::now() - chrono::Duration::seconds(30) {
                    return Ok(session.status.clone());
                }
            }
        }

        // 2. Query live status from container
        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "status", swarm_id
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            return Ok(SwarmStatus::Unknown);
        }

        // 3. Parse status from output
        let status = self.parse_swarm_status(&output)?;

        // 4. Update cache
        {
            let mut cache = self.session_cache.write().await;
            if let Some(session) = cache.get_mut(swarm_id) {
                session.status = status.clone();
                session.last_activity = Utc::now();
            }
        }

        Ok(status)
    }

    /// Stop a swarm
    pub async fn stop_swarm(&self, swarm_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Stopping swarm: {}", swarm_id);

        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "stop", swarm_id
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to stop swarm {}: {}", swarm_id, stderr).into());
        }

        // Update cache
        {
            let mut cache = self.session_cache.write().await;
            if let Some(session) = cache.get_mut(swarm_id) {
                session.status = SwarmStatus::Completing;
                session.last_activity = Utc::now();
            }
        }

        info!("Swarm {} stopped successfully", swarm_id);
        Ok(())
    }

    /// Resume a paused swarm
    pub async fn resume_swarm(&self, swarm_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Resuming swarm: {}", swarm_id);

        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "resume", swarm_id
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to resume swarm {}: {}", swarm_id, stderr).into());
        }

        // Update cache
        {
            let mut cache = self.session_cache.write().await;
            if let Some(session) = cache.get_mut(swarm_id) {
                session.status = SwarmStatus::Active;
                session.last_activity = Utc::now();
            }
        }

        info!("Swarm {} resumed successfully", swarm_id);
        Ok(())
    }

    /// Get swarm metrics
    pub async fn get_swarm_metrics(&self, swarm_id: &str) -> Result<SwarmMetrics, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Getting metrics for swarm: {}", swarm_id);

        let mut cmd = Command::new("docker");
        cmd.args(&[
            "exec",
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "metrics", swarm_id
        ]);

        let output = timeout(self.config.command_timeout, cmd.output()).await??;

        if !output.status.success() {
            return Ok(SwarmMetrics::default());
        }

        let metrics = self.parse_swarm_metrics(&output)?;

        // Update cache
        {
            let mut cache = self.session_cache.write().await;
            if let Some(session) = cache.get_mut(swarm_id) {
                session.metrics = metrics.clone();
                session.last_activity = Utc::now();
            }
        }

        Ok(metrics)
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

    fn build_spawn_command(&self, task: &str, config: &SwarmConfig) -> Command {
        let mut cmd = Command::new("docker");

        cmd.args(&[
            "exec", "-d",  // Detached to avoid blocking
            &self.container_name,
            &self.claude_flow_path,
            "hive-mind", "spawn",
            task,
            "--claude", "--auto-spawn"
        ]);

        // Add queen type based on priority
        match config.priority {
            SwarmPriority::High | SwarmPriority::Critical => {
                cmd.args(&["--queen-type", "tactical"]);
            },
            SwarmPriority::Medium => {
                cmd.args(&["--queen-type", "strategic"]);
            },
            SwarmPriority::Low => {
                cmd.args(&["--queen-type", "adaptive"]);
            },
        }

        // Add max workers
        if let Some(workers) = config.max_workers {
            cmd.args(&["--max-workers", &workers.to_string()]);
        }

        // Add consensus type
        if let Some(consensus) = &config.consensus_type {
            cmd.args(&["--consensus", consensus]);
        }

        // Add memory size
        if let Some(memory) = config.memory_size_mb {
            cmd.args(&["--memory-size", &memory.to_string()]);
        }

        // Add flags
        if config.auto_scale {
            cmd.arg("--auto-scale");
        }

        if config.encryption {
            cmd.arg("--encryption");
        }

        if config.monitor {
            cmd.arg("--monitor");
        }

        if config.verbose {
            cmd.arg("--verbose");
        }

        cmd
    }

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

    fn extract_session_id(&self, output: &std::process::Output) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{}\n{}", stdout, stderr);

        // Look for session ID patterns
        if let Some(captures) = regex::Regex::new(r"session[_-]([a-zA-Z0-9-]+)")
            .unwrap()
            .captures(&combined)
        {
            if let Some(id) = captures.get(1) {
                return Ok(format!("session-{}", id.as_str()));
            }
        }

        // Fallback: generate temporary ID
        Ok(format!("temp-{}", Uuid::new_v4().to_string()[..8].to_string()))
    }

    async fn parse_sessions_output(&self, output: &std::process::Output) -> Result<Vec<SessionInfo>, Box<dyn std::error::Error + Send + Sync>> {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut sessions = Vec::new();

        // Try to parse as JSON first
        if let Ok(json_value) = serde_json::from_str::<Value>(&stdout) {
            if let Some(sessions_array) = json_value.get("sessions").and_then(|s| s.as_array()) {
                for session_value in sessions_array {
                    if let Ok(session) = serde_json::from_value::<SessionInfo>(session_value.clone()) {
                        sessions.push(session);
                    }
                }
            }
        } else {
            // Parse as plain text output
            sessions = self.parse_text_sessions(&stdout);
        }

        // Merge with cache for additional context
        let cache = self.session_cache.read().await;
        for session in &mut sessions {
            if let Some(cached) = cache.get(&session.session_id) {
                session.config = cached.config.clone();
                session.task_description = cached.task_description.clone();
            }
        }

        Ok(sessions)
    }

    fn parse_text_sessions(&self, text: &str) -> Vec<SessionInfo> {
        let mut sessions = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        for line in lines {
            if let Some(session) = self.parse_session_line(line) {
                sessions.push(session);
            }
        }

        sessions
    }

    fn parse_session_line(&self, line: &str) -> Option<SessionInfo> {
        // Simple text parsing for session lines
        // Format: "session-123456 | Status: Active | Task: Build REST API"

        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 2 {
            return None;
        }

        let session_id = parts[0].trim().to_string();
        let status_part = parts.get(1)?.trim();
        let task_part = parts.get(2).unwrap_or(&"Unknown task").trim();

        let status = match status_part.to_lowercase() {
            s if s.contains("active") => SwarmStatus::Active,
            s if s.contains("paused") => SwarmStatus::Paused,
            s if s.contains("completed") => SwarmStatus::Completed,
            s if s.contains("failed") => SwarmStatus::Failed,
            s if s.contains("spawning") => SwarmStatus::Spawning,
            _ => SwarmStatus::Unknown,
        };

        Some(SessionInfo {
            session_id,
            task_description: task_part.replace("Task: ", "").to_string(),
            status,
            created_at: Utc::now() - chrono::Duration::hours(1), // Estimate
            last_activity: Utc::now(),
            metrics: SwarmMetrics::default(),
            config: SwarmConfig::default(),
            workers: Vec::new(),
        })
    }

    fn parse_swarm_status(&self, output: &std::process::Output) -> Result<SwarmStatus, Box<dyn std::error::Error + Send + Sync>> {
        let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();

        let status = if stdout.contains("active") || stdout.contains("running") {
            SwarmStatus::Active
        } else if stdout.contains("paused") || stdout.contains("suspended") {
            SwarmStatus::Paused
        } else if stdout.contains("completed") || stdout.contains("finished") {
            SwarmStatus::Completed
        } else if stdout.contains("failed") || stdout.contains("error") {
            SwarmStatus::Failed
        } else if stdout.contains("spawning") || stdout.contains("starting") {
            SwarmStatus::Spawning
        } else if stdout.contains("completing") || stdout.contains("stopping") {
            SwarmStatus::Completing
        } else {
            SwarmStatus::Unknown
        };

        Ok(status)
    }

    fn parse_swarm_metrics(&self, output: &std::process::Output) -> Result<SwarmMetrics, Box<dyn std::error::Error + Send + Sync>> {
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Try JSON first
        if let Ok(json_value) = serde_json::from_str::<Value>(&stdout) {
            if let Ok(metrics) = serde_json::from_value::<SwarmMetrics>(json_value) {
                return Ok(metrics);
            }
        }

        // Fallback to default
        Ok(SwarmMetrics::default())
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