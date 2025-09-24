use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::process::Command;
use tokio::time::{timeout, sleep};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use chrono::{DateTime, Utc};
use log::{info, warn, error, debug};
use uuid::Uuid;

/// Consolidated Docker execution service for all container operations
/// Removes duplication between DockerHiveMind and other Docker services
#[derive(Clone, Debug)]
pub struct ConsolidatedDockerService {
    container_name: String,
    base_command_path: String,
    command_cache: Arc<RwLock<HashMap<String, CachedCommandResult>>>,
    health_cache: Arc<RwLock<Option<ContainerHealth>>>,
    config: DockerServiceConfig,
}

#[derive(Debug, Clone)]
pub struct DockerServiceConfig {
    pub command_timeout: Duration,
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub health_check_interval: Duration,
    pub cache_ttl: Duration,
}

#[derive(Debug, Clone)]
pub struct CachedCommandResult {
    pub result: CommandResult,
    pub cached_at: DateTime<Utc>,
    pub ttl: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub execution_time_ms: u64,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerHealth {
    pub is_running: bool,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_healthy: bool,
    pub disk_space_gb: f64,
    pub last_response_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub working_dir: Option<String>,
    pub env_vars: HashMap<String, String>,
    pub timeout_secs: Option<u64>,
    pub retry_on_failure: bool,
    pub cache_result: bool,
}

impl Default for DockerServiceConfig {
    fn default() -> Self {
        Self {
            command_timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(1000),
            health_check_interval: Duration::from_secs(30),
            cache_ttl: Duration::from_secs(60),
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            working_dir: Some("/app".to_string()),
            env_vars: HashMap::new(),
            timeout_secs: Some(30),
            retry_on_failure: true,
            cache_result: false,
        }
    }
}

impl ConsolidatedDockerService {
    /// Create a new consolidated Docker service
    pub fn new(container_name: String) -> Self {
        Self {
            container_name: container_name.clone(),
            base_command_path: "/app/node_modules/.bin/claude-flow".to_string(),
            command_cache: Arc::new(RwLock::new(HashMap::new())),
            health_cache: Arc::new(RwLock::new(None)),
            config: DockerServiceConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(container_name: String, config: DockerServiceConfig) -> Self {
        Self {
            container_name: container_name.clone(),
            base_command_path: "/app/node_modules/.bin/claude-flow".to_string(),
            command_cache: Arc::new(RwLock::new(HashMap::new())),
            health_cache: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Execute a command in the Docker container with full control
    pub async fn execute_command(
        &self,
        command: &str,
        args: &[&str],
        context: ExecutionContext,
    ) -> Result<CommandResult, Box<dyn std::error::Error + Send + Sync>> {
        let cache_key = if context.cache_result {
            Some(format!("{}_{}", command, args.join("_")))
        } else {
            None
        };

        // Check cache first
        if let Some(ref key) = cache_key {
            if let Some(cached) = self.get_cached_result(key).await {
                debug!("Using cached result for command: {}", command);
                return Ok(cached.result);
            }
        }

        let start_time = std::time::Instant::now();
        let full_command = format!("{} {}", command, args.join(" "));

        let mut retries = if context.retry_on_failure { self.config.max_retries } else { 1 };

        for attempt in 1..=retries {
            debug!("Executing Docker command (attempt {}/{}): {}", attempt, retries, full_command);

            let mut docker_cmd = Command::new("docker");
            docker_cmd.args(["exec", &self.container_name]);

            // Set working directory if specified
            if let Some(ref work_dir) = context.working_dir {
                docker_cmd.args(["-w", work_dir]);
            }

            // Add environment variables
            for (key, value) in &context.env_vars {
                docker_cmd.args(["-e", &format!("{}={}", key, value)]);
            }

            // Add the actual command
            docker_cmd.arg(command);
            docker_cmd.args(args);

            let timeout_duration = Duration::from_secs(
                context.timeout_secs.unwrap_or(self.config.command_timeout.as_secs())
            );

            match timeout(timeout_duration, docker_cmd.output()).await {
                Ok(Ok(output)) => {
                    let execution_time = start_time.elapsed().as_millis() as u64;
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                    let result = CommandResult {
                        success: output.status.success(),
                        stdout: stdout.clone(),
                        stderr: stderr.clone(),
                        exit_code: output.status.code(),
                        execution_time_ms: execution_time,
                        command: full_command.clone(),
                    };

                    if result.success {
                        info!("Docker command executed successfully: {} ({}ms)", full_command, execution_time);

                        // Cache successful results if requested
                        if let Some(key) = cache_key {
                            self.cache_result(key, result.clone()).await;
                        }

                        return Ok(result);
                    } else {
                        warn!("Docker command failed (attempt {}/{}): {} - stderr: {}",
                              attempt, retries, full_command, stderr);

                        if attempt == retries {
                            return Ok(result); // Return the failed result on last attempt
                        }
                    }
                }
                Ok(Err(e)) => {
                    error!("Failed to execute Docker command (attempt {}/{}): {} - error: {}",
                           attempt, retries, full_command, e);

                    if attempt == retries {
                        return Err(Box::new(e));
                    }
                }
                Err(_) => {
                    error!("Docker command timeout (attempt {}/{}): {}", attempt, retries, full_command);

                    if attempt == retries {
                        return Err(format!("Command timeout: {}", full_command).into());
                    }
                }
            }

            // Wait before retry
            if attempt < retries {
                sleep(self.config.retry_delay).await;
            }
        }

        Err(format!("Command failed after {} retries: {}", retries, full_command).into())
    }

    /// Execute a claude-flow specific command
    pub async fn execute_claude_flow(
        &self,
        subcommand: &str,
        args: &[&str],
    ) -> Result<CommandResult, Box<dyn std::error::Error + Send + Sync>> {
        let context = ExecutionContext {
            working_dir: Some("/app".to_string()),
            cache_result: true,
            ..Default::default()
        };

        self.execute_command(&self.base_command_path, &[vec![subcommand], args.to_vec()].concat(), context).await
    }

    /// Execute an npm command
    pub async fn execute_npm(
        &self,
        args: &[&str],
    ) -> Result<CommandResult, Box<dyn std::error::Error + Send + Sync>> {
        let context = ExecutionContext {
            working_dir: Some("/app".to_string()),
            ..Default::default()
        };

        self.execute_command("npm", args, context).await
    }

    /// Execute a simple shell command
    pub async fn execute_shell(
        &self,
        command: &str,
    ) -> Result<CommandResult, Box<dyn std::error::Error + Send + Sync>> {
        let context = ExecutionContext {
            working_dir: Some("/app".to_string()),
            ..Default::default()
        };

        self.execute_command("sh", &["-c", command], context).await
    }

    /// Check container health with caching
    pub async fn check_health(&self) -> Result<ContainerHealth, Box<dyn std::error::Error + Send + Sync>> {
        // Check cache first
        {
            let health_cache = self.health_cache.read().await;
            if let Some(ref cached_health) = *health_cache {
                // TODO: Add timestamp check for cache validity
                return Ok(cached_health.clone());
            }
        }

        debug!("Checking Docker container health: {}", self.container_name);

        // Check if container is running
        let inspect_result = self.execute_command(
            "docker",
            &["inspect", &self.container_name, "--format", "{{.State.Running}}"],
            ExecutionContext::default()
        ).await;

        let is_running = match inspect_result {
            Ok(result) => result.stdout.trim() == "true",
            Err(_) => false,
        };

        if !is_running {
            let health = ContainerHealth {
                is_running: false,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                network_healthy: false,
                disk_space_gb: 0.0,
                last_response_ms: 0,
            };

            // Cache the result
            {
                let mut health_cache = self.health_cache.write().await;
                *health_cache = Some(health.clone());
            }

            return Ok(health);
        }

        // Get detailed stats
        let stats_result = self.execute_command(
            "docker",
            &["stats", &self.container_name, "--no-stream", "--format", "table {{.CPUPerc}}\t{{.MemPerc}}"],
            ExecutionContext::default()
        ).await;

        let (cpu_usage, memory_usage) = match stats_result {
            Ok(result) => {
                if let Some(line) = result.stdout.lines().nth(1) {
                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() >= 2 {
                        let cpu = parts[0].trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                        let mem = parts[1].trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                        (cpu, mem)
                    } else {
                        (0.0, 0.0)
                    }
                } else {
                    (0.0, 0.0)
                }
            }
            Err(_) => (0.0, 0.0),
        };

        // Check disk space
        let disk_result = self.execute_command(
            "docker",
            &["exec", &self.container_name, "df", "-h", "/app", "--output=avail"],
            ExecutionContext::default()
        ).await;

        let disk_space_gb = match disk_result {
            Ok(result) => {
                if let Some(line) = result.stdout.lines().nth(1) {
                    // Parse disk space (could be in various formats like "10G", "1024M", etc.)
                    parse_disk_space(line.trim())
                } else {
                    0.0
                }
            }
            Err(_) => 0.0,
        };

        let health = ContainerHealth {
            is_running: true,
            cpu_usage,
            memory_usage,
            network_healthy: true, // TODO: Add actual network check
            disk_space_gb,
            last_response_ms: 0, // TODO: Add response time measurement
        };

        // Cache the result
        {
            let mut health_cache = self.health_cache.write().await;
            *health_cache = Some(health.clone());
        }

        Ok(health)
    }

    /// Start background health monitoring
    pub async fn start_health_monitoring(&self) {
        let self_clone = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self_clone.config.health_check_interval);

            loop {
                interval.tick().await;

                if let Err(e) = self_clone.check_health().await {
                    error!("Health check failed: {}", e);
                }
            }
        });

        info!("Started Docker health monitoring for container: {}", self.container_name);
    }

    /// Get cached command result
    async fn get_cached_result(&self, key: &str) -> Option<CachedCommandResult> {
        let cache = self.command_cache.read().await;
        if let Some(cached) = cache.get(key) {
            let age = Utc::now() - cached.cached_at;
            if age.to_std().unwrap_or(Duration::MAX) < cached.ttl {
                return Some(cached.clone());
            }
        }
        None
    }

    /// Cache a command result
    async fn cache_result(&self, key: String, result: CommandResult) {
        let mut cache = self.command_cache.write().await;
        cache.insert(key, CachedCommandResult {
            result,
            cached_at: Utc::now(),
            ttl: self.config.cache_ttl,
        });
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut cache = self.command_cache.write().await;
        cache.clear();
        info!("Docker service cache cleared");
    }
}

/// Parse disk space string (e.g., "10G", "1024M") to GB
fn parse_disk_space(space_str: &str) -> f64 {
    let space_str = space_str.trim();
    if space_str.is_empty() {
        return 0.0;
    }

    let (number_str, unit) = if space_str.ends_with('G') {
        (&space_str[..space_str.len()-1], "G")
    } else if space_str.ends_with('M') {
        (&space_str[..space_str.len()-1], "M")
    } else if space_str.ends_with('K') {
        (&space_str[..space_str.len()-1], "K")
    } else {
        (space_str, "B")
    };

    if let Ok(number) = number_str.parse::<f64>() {
        match unit {
            "G" => number,
            "M" => number / 1024.0,
            "K" => number / (1024.0 * 1024.0),
            _ => number / (1024.0 * 1024.0 * 1024.0),
        }
    } else {
        0.0
    }
}

/// Factory functions for common Docker services
impl ConsolidatedDockerService {
    /// Create a service for claude-flow operations
    pub fn claude_flow_service(container_name: String) -> Self {
        let mut service = Self::new(container_name);
        service.base_command_path = "/app/node_modules/.bin/claude-flow".to_string();
        service
    }

    /// Create a service for MCP operations
    pub fn mcp_service(container_name: String) -> Self {
        let mut service = Self::new(container_name);
        service.base_command_path = "npx".to_string();
        service
    }

    /// Create a service for general shell operations
    pub fn shell_service(container_name: String) -> Self {
        let mut service = Self::new(container_name);
        service.base_command_path = "sh".to_string();
        service
    }
}