//! Network resilience utilities for VisionFlow
//! 
//! This module provides comprehensive network resilience patterns including:
//! - Exponential backoff retry logic
//! - Circuit breaker patterns
//! - Connection pooling and management
//! - Health check systems
//! - Timeout management

pub mod retry;
pub mod circuit_breaker;
pub mod connection_pool;
pub mod health_check;
pub mod timeout;
pub mod graceful_degradation;

pub use retry::{
    RetryConfig, RetryError, RetryResult, RetryableError,
    retry_with_backoff, retry_network_operation, retry_tcp_connection,
    retry_websocket_operation, retry_mcp_operation, retry_with_timeout,
};

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, CircuitBreakerStats,
    CircuitBreakerError, CircuitBreakerRegistry, RequestOutcome,
};

pub use connection_pool::{
    ConnectionPool, ConnectionPoolConfig, ConnectionPoolStats, ConnectionPoolError,
    ConnectionPoolRegistry, PooledConnection,
};

pub use health_check::{
    HealthCheckManager, HealthCheckConfig, HealthStatus, HealthCheckResult,
    ServiceHealthInfo, ServiceEndpoint, SystemHealthSummary,
};

pub use timeout::{
    TimeoutConfig, TimeoutResult, TimeoutError, TimeoutType, TimeoutGuard,
    BatchTimeoutManager, AdaptiveTimeout,
    with_timeout, with_config_timeout, connect_with_timeout, request_with_timeout,
    read_with_timeout, write_with_timeout,
};

pub use graceful_degradation::{
    GracefulDegradationManager, GracefulDegradationConfig, DegradationLevel, DegradationStrategy,
};

use std::sync::Arc;
use log::info;
use serde::{Deserialize, Serialize};

/// Comprehensive network resilience manager that coordinates all resilience patterns
pub struct NetworkResilienceManager {
    circuit_breaker_registry: CircuitBreakerRegistry,
    connection_pool_registry: ConnectionPoolRegistry,
    health_check_manager: HealthCheckManager,
    default_retry_config: RetryConfig,
    default_timeout_config: TimeoutConfig,
}

impl NetworkResilienceManager {
    /// Create a new network resilience manager with default configurations
    pub fn new() -> Self {
        Self {
            circuit_breaker_registry: CircuitBreakerRegistry::new(),
            connection_pool_registry: ConnectionPoolRegistry::new(),
            health_check_manager: HealthCheckManager::new(),
            default_retry_config: RetryConfig::network(),
            default_timeout_config: TimeoutConfig::default(),
        }
    }

    /// Create a manager optimized for high-performance scenarios
    pub fn high_performance() -> Self {
        Self {
            circuit_breaker_registry: CircuitBreakerRegistry::new(),
            connection_pool_registry: ConnectionPoolRegistry::new(),
            health_check_manager: HealthCheckManager::new(),
            default_retry_config: RetryConfig {
                max_attempts: 2,
                initial_delay: std::time::Duration::from_millis(50),
                max_delay: std::time::Duration::from_secs(2),
                backoff_multiplier: 1.5,
                jitter_factor: 0.05,
                preserve_original_error: false,
            },
            default_timeout_config: TimeoutConfig::low_latency(),
        }
    }

    /// Register a service with comprehensive resilience patterns
    pub async fn register_service(&self, service_config: ServiceResilienceConfig) -> Result<(), String> {
        let service_name = &service_config.service_name;
        info!("Registering service with resilience patterns: {}", service_name);

        // Register circuit breaker
        self.circuit_breaker_registry
            .get_or_create(service_name, service_config.circuit_breaker_config.clone())
            .await;

        // Register connection pool
        if let Some(pool_config) = service_config.connection_pool_config {
            self.connection_pool_registry
                .get_or_create_pool(service_name, pool_config)
                .await;
        }

        // Register health checks
        if let Some(endpoint) = service_config.health_check_endpoint {
            self.health_check_manager.register_service(endpoint).await;
        }

        info!("Service {} registered with all resilience patterns", service_name);
        Ok(())
    }

    /// Unregister a service from all resilience patterns
    pub async fn unregister_service(&self, service_name: &str) {
        info!("Unregistering service from resilience patterns: {}", service_name);
        
        // Health checks cleanup
        self.health_check_manager.unregister_service(service_name).await;
        
        info!("Service {} unregistered from all resilience patterns", service_name);
    }

    /// Execute an operation with full resilience patterns
    pub async fn execute_with_resilience<F, T, E>(
        &self,
        service_name: &str,
        operation: F,
    ) -> Result<T, ResilienceError<E>>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>> + Send + Clone + 'static,
        E: RetryableError + std::fmt::Debug + Clone + Send + Sync + 'static,
        T: Send,
    {
        // Get circuit breaker
        let circuit_breaker = self.circuit_breaker_registry
            .get_or_create(service_name, CircuitBreakerConfig::network())
            .await;

        // Execute with circuit breaker and retry
        let retry_operation = {
            let circuit_breaker = circuit_breaker.clone();
            let operation = operation.clone();
            move || {
                let circuit_breaker = circuit_breaker.clone();
                let operation = operation.clone();
                Box::pin(async move {
                    circuit_breaker.execute(operation()).await
                        .map_err(|e| match e {
                            CircuitBreakerError::CircuitOpen => 
                                std::sync::Arc::new(std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "Circuit breaker open")) as std::sync::Arc<dyn std::error::Error + Send + Sync>,
                            CircuitBreakerError::OperationFailed(original_error) => 
                                std::sync::Arc::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Operation failed: {:?}", original_error))) as std::sync::Arc<dyn std::error::Error + Send + Sync>,
                        })
                })
            }
        };

        match retry_with_backoff(self.default_retry_config.clone(), retry_operation).await {
            Ok(result) => Ok(result),
            Err(RetryError::AllAttemptsFailed(_)) => Err(ResilienceError::AllRetriesFailed),
            Err(RetryError::Cancelled) => Err(ResilienceError::OperationCancelled),
            Err(RetryError::ConfigError(msg)) => Err(ResilienceError::ConfigurationError(msg)),
            Err(RetryError::ResourceExhaustion(msg)) => Err(ResilienceError::ResourceExhausted(msg)),
        }
    }

    /// Get comprehensive resilience statistics for all services
    pub async fn get_resilience_stats(&self) -> ResilienceStats {
        let circuit_breaker_stats = self.circuit_breaker_registry.get_all_stats().await;
        let connection_pool_stats = self.connection_pool_registry.get_all_stats().await;
        let health_stats = self.health_check_manager.get_all_health().await;
        let system_health = self.health_check_manager.get_system_health_summary().await;

        ResilienceStats {
            circuit_breaker_stats,
            connection_pool_stats,
            health_stats,
            system_health,
        }
    }

    /// Shutdown all resilience components
    pub async fn shutdown(&self) {
        info!("Shutting down network resilience manager");
        
        self.circuit_breaker_registry.reset_all().await;
        self.connection_pool_registry.shutdown_all().await;
        self.health_check_manager.shutdown().await;
        
        info!("Network resilience manager shutdown complete");
    }

    /// Get circuit breaker for a service
    pub async fn get_circuit_breaker(&self, service_name: &str) -> Arc<CircuitBreaker> {
        self.circuit_breaker_registry
            .get_or_create(service_name, CircuitBreakerConfig::network())
            .await
    }

    /// Get connection pool for a service
    pub async fn get_connection_pool(&self, service_name: &str) -> Arc<tokio::sync::Mutex<ConnectionPool>> {
        self.connection_pool_registry
            .get_or_create_pool(service_name, ConnectionPoolConfig::default())
            .await
    }

    /// Check health of a specific service
    pub async fn check_service_health(&self, service_name: &str) -> Option<ServiceHealthInfo> {
        self.health_check_manager.get_service_health(service_name).await
    }
}

impl Default for NetworkResilienceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for service resilience patterns
#[derive(Debug, Clone)]
pub struct ServiceResilienceConfig {
    pub service_name: String,
    pub circuit_breaker_config: CircuitBreakerConfig,
    pub connection_pool_config: Option<ConnectionPoolConfig>,
    pub health_check_endpoint: Option<ServiceEndpoint>,
    pub retry_config: Option<RetryConfig>,
    pub timeout_config: Option<TimeoutConfig>,
}

impl ServiceResilienceConfig {
    /// Create a basic configuration for a service
    pub fn new(service_name: String, host: String, port: u16) -> Self {
        let endpoint = ServiceEndpoint::new(service_name.clone(), host, port);
        
        Self {
            service_name,
            circuit_breaker_config: CircuitBreakerConfig::network(),
            connection_pool_config: Some(ConnectionPoolConfig::default()),
            health_check_endpoint: Some(endpoint),
            retry_config: Some(RetryConfig::network()),
            timeout_config: Some(TimeoutConfig::default()),
        }
    }

    /// Create configuration for a critical service
    pub fn critical_service(service_name: String, host: String, port: u16) -> Self {
        let endpoint = ServiceEndpoint::new(service_name.clone(), host, port)
            .with_config(HealthCheckConfig::critical_service());
        
        Self {
            service_name,
            circuit_breaker_config: CircuitBreakerConfig::network(),
            connection_pool_config: Some(ConnectionPoolConfig::default()),
            health_check_endpoint: Some(endpoint),
            retry_config: Some(RetryConfig::network()),
            timeout_config: Some(TimeoutConfig::low_latency()),
        }
    }

    /// Create configuration for a background service
    pub fn background_service(service_name: String, host: String, port: u16) -> Self {
        let endpoint = ServiceEndpoint::new(service_name.clone(), host, port)
            .with_config(HealthCheckConfig::background_service());
        
        Self {
            service_name,
            circuit_breaker_config: CircuitBreakerConfig::network(),
            connection_pool_config: Some(ConnectionPoolConfig::high_throughput()),
            health_check_endpoint: Some(endpoint),
            retry_config: Some(RetryConfig::network()),
            timeout_config: Some(TimeoutConfig::high_throughput()),
        }
    }
}

/// Comprehensive resilience statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceStats {
    pub circuit_breaker_stats: std::collections::HashMap<String, CircuitBreakerStats>,
    pub connection_pool_stats: std::collections::HashMap<String, ConnectionPoolStats>,
    pub health_stats: std::collections::HashMap<String, ServiceHealthInfo>,
    pub system_health: SystemHealthSummary,
}

/// Errors that can occur with resilience operations
#[derive(Debug, thiserror::Error)]
pub enum ResilienceError<E> {
    #[error("All retry attempts failed")]
    AllRetriesFailed,
    #[error("Operation was cancelled")]
    OperationCancelled,
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Service unavailable")]
    ServiceUnavailable,
    #[error("Circuit breaker is open")]
    CircuitBreakerOpen,
    #[error("Connection pool exhausted")]
    ConnectionPoolExhausted,
    #[error("Health check failed")]
    HealthCheckFailed,
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    #[error("Timeout exceeded")]
    TimeoutExceeded,
    #[error("Original error: {0:?}")]
    OriginalError(E),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_resilience_manager_creation() {
        let manager = NetworkResilienceManager::new();
        let stats = manager.get_resilience_stats().await;
        
        assert_eq!(stats.circuit_breaker_stats.len(), 0);
        assert_eq!(stats.connection_pool_stats.len(), 0);
    }

    #[tokio::test]
    async fn test_service_registration() {
        let manager = NetworkResilienceManager::new();
        
        let config = ServiceResilienceConfig::new(
            "test-service".to_string(),
            "localhost".to_string(),
            8080,
        );
        
        let result = manager.register_service(config).await;
        assert!(result.is_ok());
        
        let stats = manager.get_resilience_stats().await;
        assert!(stats.circuit_breaker_stats.contains_key("test-service"));
        assert!(stats.connection_pool_stats.contains_key("test-service"));
        
        manager.shutdown().await;
    }

    #[test]
    fn test_service_config_creation() {
        let config = ServiceResilienceConfig::critical_service(
            "critical".to_string(),
            "localhost".to_string(),
            9000,
        );
        
        assert_eq!(config.service_name, "critical");
        assert!(config.connection_pool_config.is_some());
        assert!(config.health_check_endpoint.is_some());
    }
}