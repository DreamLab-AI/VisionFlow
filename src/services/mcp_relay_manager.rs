use crate::telemetry::agent_telemetry::{
    get_telemetry_logger, CorrelationId, LogLevel, TelemetryEvent,
};
use crate::utils::network::{
    CircuitBreaker, CircuitBreakerConfig, HealthCheckManager, RetryableError, TimeoutConfig,
};
use log::{debug, error, info, warn};
use serde_json;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

/// Manages the MCP WebSocket relay in the multi-agent-container with resilience patterns
pub struct McpRelayManager {
    circuit_breaker: Arc<CircuitBreaker>,
    health_manager: Arc<HealthCheckManager>,
    timeout_config: TimeoutConfig,
}

/// Custom error type for MCP relay operations
#[derive(Debug, thiserror::Error)]
pub enum McpRelayError {
    #[error("Docker command failed: {0}")]
    DockerCommandFailed(String),
    #[error("Container not found: {0}")]
    ContainerNotFound(String),
    #[error("Service health check failed")]
    HealthCheckFailed,
    #[error("Operation timeout")]
    Timeout,
}

impl RetryableError for McpRelayError {
    fn is_retryable(&self) -> bool {
        match self {
            McpRelayError::DockerCommandFailed(_) => true,
            McpRelayError::ContainerNotFound(_) => false, // Don't retry if container doesn't exist
            McpRelayError::HealthCheckFailed => true,
            McpRelayError::Timeout => true,
        }
    }
}

impl McpRelayManager {
    /// Create a new MCP relay manager with resilience patterns
    pub fn new() -> Self {
        let circuit_breaker = Arc::new(CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            failure_rate_threshold: 0.5,
            time_window: std::time::Duration::from_secs(60),
            recovery_timeout: std::time::Duration::from_secs(30),
            success_threshold: 2,
            half_open_max_requests: 3,
            minimum_request_threshold: 5,
        }));

        let health_manager = Arc::new(HealthCheckManager::new());

        Self {
            circuit_breaker,
            health_manager,
            timeout_config: TimeoutConfig::default(),
        }
    }

    /// Check if the MCP relay is running in the multi-agent-container with resilience
    pub async fn check_relay_status(&self) -> Result<bool, McpRelayError> {
        let operation = || {
            Box::pin(async {
                Self::check_relay_status_internal()
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            })
        };

        match self.circuit_breaker.execute(operation()).await {
            Ok(result) => Ok(result),
            Err(e) => {
                error!("Circuit breaker failed for MCP relay status check: {:?}", e);
                Err(McpRelayError::HealthCheckFailed)
            }
        }
    }

    /// Internal method for checking relay status
    async fn check_relay_status_internal() -> Result<bool, McpRelayError> {
        let start_time = Instant::now();
        let correlation_id = CorrelationId::new();

        info!("Checking MCP relay status in multi-agent-container...");

        // Log MCP bridge operation start
        if let Some(logger) = get_telemetry_logger() {
            logger.log_mcp_message("status_check", "outbound", 0, "initiated");
        }

        let output = Command::new("docker")
            .args(&["exec", "multi-agent-container", "pgrep", "-f", "mcp-server"])
            .output();

        let duration_ms = start_time.elapsed().as_millis() as f64;

        match output {
            Ok(result) => {
                let is_running = result.status.success();
                let status = if is_running { "running" } else { "stopped" };

                if is_running {
                    info!("MCP relay is running in multi-agent-container");
                } else {
                    warn!("MCP relay is not running in multi-agent-container");
                }

                // Enhanced telemetry logging for MCP bridge status
                if let Some(logger) = get_telemetry_logger() {
                    let event = TelemetryEvent::new(
                        correlation_id,
                        if is_running {
                            LogLevel::INFO
                        } else {
                            LogLevel::WARN
                        },
                        "mcp_bridge",
                        "status_check_result",
                        &format!("MCP relay status check completed: {}", status),
                        "mcp_relay_manager",
                    )
                    .with_duration(duration_ms)
                    .with_metadata("container_status", serde_json::json!(status))
                    .with_metadata("container_name", serde_json::json!("multi-agent-container"))
                    .with_metadata("check_method", serde_json::json!("docker_exec_pgrep"));

                    logger.log_event(event);

                    // Also log as MCP message flow
                    logger.log_mcp_message("status_check", "inbound", result.stdout.len(), status);
                }

                Ok(is_running)
            }
            Err(e) => {
                error!("Failed to check MCP relay status: {}", e);

                // Log MCP bridge error
                if let Some(logger) = get_telemetry_logger() {
                    let event = TelemetryEvent::new(
                        correlation_id,
                        LogLevel::ERROR,
                        "mcp_bridge",
                        "status_check_error",
                        &format!("MCP relay status check failed: {}", e),
                        "mcp_relay_manager",
                    )
                    .with_duration(duration_ms)
                    .with_metadata("error_type", serde_json::json!("docker_command_failed"))
                    .with_metadata("error_message", serde_json::json!(e.to_string()));

                    logger.log_event(event);

                    logger.log_mcp_message("status_check", "error", 0, "failed");
                }

                Ok(false)
            }
        }
    }

    /// Start the MCP relay in the multi-agent-container if not already running
    pub async fn ensure_relay_running(&self) -> Result<(), String> {
        // Check service health before proceeding
        if let Some(health_result) = self.health_manager.check_service_now("mcp-relay").await {
            match health_result.status {
                crate::utils::network::HealthStatus::Healthy => {
                    info!("MCP relay health check passed");
                }
                _ => {
                    warn!("Health check failed for MCP relay: {:?}", health_result);
                }
            }
        } else {
            warn!("No health check configuration found for MCP relay");
        }

        if Self::check_relay_status_internal().await.unwrap_or(false) {
            info!("MCP relay already running, no action needed");
            return Ok(());
        }

        info!("Starting MCP relay in multi-agent-container...");

        // Start the relay in the background
        let output = Command::new("docker")
            .args(&[
                "exec",
                "-d",
                "multi-agent-container",
                "bash",
                "-c",
                "cd /app && npm run mcp:start > /tmp/mcp-server.log 2>&1",
            ])
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    info!("Successfully started MCP relay in multi-agent-container");

                    // Give it a moment to start
                    std::thread::sleep(std::time::Duration::from_secs(2));

                    // Verify it's running (asynchronous check)
                    if Self::check_relay_status_internal().await.unwrap_or(false) {
                        Ok(())
                    } else {
                        Err("MCP relay started but not running".to_string())
                    }
                } else {
                    let stderr = String::from_utf8_lossy(&result.stderr);
                    Err(format!("Failed to start MCP relay: {}", stderr))
                }
            }
            Err(e) => Err(format!("Failed to execute docker command: {}", e)),
        }
    }

    /// Get the logs from the MCP relay
    pub fn get_relay_logs(lines: usize) -> Result<String, String> {
        let output = Command::new("docker")
            .args(&[
                "exec",
                "multi-agent-container",
                "tail",
                "-n",
                &lines.to_string(),
                "/tmp/mcp-server.log",
            ])
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    Ok(String::from_utf8_lossy(&result.stdout).to_string())
                } else {
                    Err(format!(
                        "Failed to get logs: {}",
                        String::from_utf8_lossy(&result.stderr)
                    ))
                }
            }
            Err(e) => Err(format!("Failed to execute docker command: {}", e)),
        }
    }

    /// Implement continuous health monitoring for the MCP relay
    pub async fn start_health_monitoring(&self) {
        let health_manager = self.health_manager.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                interval.tick().await;

                if let Some(health_result) = health_manager.check_service_now("mcp-relay").await {
                    match health_result.status {
                        crate::utils::network::HealthStatus::Healthy => {
                            debug!("MCP relay health check passed");
                        }
                        _ => {
                            warn!("MCP relay health check failed: {:?}", health_result);
                            // Could trigger automatic restart here if needed
                        }
                    }
                } else {
                    warn!("No health check configuration found for MCP relay");
                }
            }
        });
    }

    /// Check if multi-agent-container is running
    pub fn check_mcp_container() -> bool {
        let output = Command::new("docker")
            .args(&["ps", "-q", "-f", "name=multi-agent-container"])
            .output();

        match output {
            Ok(result) => !result.stdout.is_empty(),
            Err(_) => false,
        }
    }
}

/// Ensure MCP relay is available before starting ClaudeFlowActor
pub async fn ensure_mcp_ready() -> Result<(), String> {
    // First check if multi-agent-container exists
    if !McpRelayManager::check_mcp_container() {
        return Err("multi-agent-container is not running".to_string());
    }

    // Create manager instance to use health monitoring
    let manager = McpRelayManager::new();

    // Try to ensure relay is running
    manager.ensure_relay_running().await?;

    // Additional wait for relay to be fully ready
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    Ok(())
}
