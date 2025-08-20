use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::{timeout, Instant, Sleep};
use log::{debug, warn};
use serde::{Deserialize, Serialize};

/// Configuration for different types of network timeouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Connection establishment timeout
    pub connect_timeout: Duration,
    /// Request/response timeout
    pub request_timeout: Duration,
    /// Read timeout for individual operations
    pub read_timeout: Duration,
    /// Write timeout for individual operations
    pub write_timeout: Duration,
    /// Keep-alive timeout for idle connections
    pub keepalive_timeout: Duration,
    /// Total operation timeout (overall cap)
    pub total_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(15),
            write_timeout: Duration::from_secs(10),
            keepalive_timeout: Duration::from_secs(60),
            total_timeout: Duration::from_secs(120),
        }
    }
}

impl TimeoutConfig {
    /// Configuration for low-latency operations
    pub fn low_latency() -> Self {
        Self {
            connect_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(3),
            write_timeout: Duration::from_secs(2),
            keepalive_timeout: Duration::from_secs(30),
            total_timeout: Duration::from_secs(10),
        }
    }

    /// Configuration for high-throughput operations
    pub fn high_throughput() -> Self {
        Self {
            connect_timeout: Duration::from_secs(15),
            request_timeout: Duration::from_secs(60),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(20),
            keepalive_timeout: Duration::from_secs(300),
            total_timeout: Duration::from_secs(300),
        }
    }

    /// Configuration for TCP connections
    pub fn tcp_connection() -> Self {
        Self {
            connect_timeout: Duration::from_secs(8),
            request_timeout: Duration::from_secs(25),
            read_timeout: Duration::from_secs(12),
            write_timeout: Duration::from_secs(8),
            keepalive_timeout: Duration::from_secs(120),
            total_timeout: Duration::from_secs(60),
        }
    }

    /// Configuration for WebSocket operations
    pub fn websocket() -> Self {
        Self {
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(20),
            read_timeout: Duration::from_secs(30), // WebSockets may have longer reads
            write_timeout: Duration::from_secs(5),
            keepalive_timeout: Duration::from_secs(60),
            total_timeout: Duration::from_secs(120),
        }
    }

    /// Configuration for MCP operations
    pub fn mcp_operations() -> Self {
        Self {
            connect_timeout: Duration::from_secs(3),
            request_timeout: Duration::from_secs(15),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(5),
            keepalive_timeout: Duration::from_secs(45),
            total_timeout: Duration::from_secs(30),
        }
    }
}

/// Result of a timeout operation
#[derive(Debug)]
pub enum TimeoutResult<T> {
    /// Operation completed successfully within timeout
    Success(T),
    /// Operation timed out
    Timeout,
    /// Operation failed with an error
    Error(Box<dyn std::error::Error + Send + Sync>),
}

impl<T> TimeoutResult<T> {
    /// Check if the operation was successful
    pub fn is_success(&self) -> bool {
        matches!(self, TimeoutResult::Success(_))
    }

    /// Check if the operation timed out
    pub fn is_timeout(&self) -> bool {
        matches!(self, TimeoutResult::Timeout)
    }

    /// Check if the operation failed with an error
    pub fn is_error(&self) -> bool {
        matches!(self, TimeoutResult::Error(_))
    }

    /// Convert to a standard Result, treating timeout as an error
    pub fn into_result(self) -> Result<T, TimeoutError> {
        match self {
            TimeoutResult::Success(value) => Ok(value),
            TimeoutResult::Timeout => Err(TimeoutError::Timeout),
            TimeoutResult::Error(err) => Err(TimeoutError::OperationFailed(err.to_string())),
        }
    }

    /// Extract the success value, or return None
    pub fn success(self) -> Option<T> {
        match self {
            TimeoutResult::Success(value) => Some(value),
            _ => None,
        }
    }
}

/// Errors that can occur with timeout operations
#[derive(Debug, thiserror::Error)]
pub enum TimeoutError {
    #[error("Operation timed out")]
    Timeout,
    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

/// Execute a future with a timeout, returning a TimeoutResult
pub async fn with_timeout<F, T>(
    duration: Duration,
    future: F,
) -> TimeoutResult<T>
where
    F: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
{
    match timeout(duration, future).await {
        Ok(Ok(value)) => TimeoutResult::Success(value),
        Ok(Err(err)) => TimeoutResult::Error(err),
        Err(_) => TimeoutResult::Timeout,
    }
}

/// Execute a future with timeout configuration, automatically selecting appropriate timeout
pub async fn with_config_timeout<F, T>(
    config: &TimeoutConfig,
    operation_type: TimeoutType,
    future: F,
) -> TimeoutResult<T>
where
    F: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
{
    let timeout_duration = match operation_type {
        TimeoutType::Connect => config.connect_timeout,
        TimeoutType::Request => config.request_timeout,
        TimeoutType::Read => config.read_timeout,
        TimeoutType::Write => config.write_timeout,
        TimeoutType::Keepalive => config.keepalive_timeout,
        TimeoutType::Total => config.total_timeout,
    };

    debug!("Setting {} timeout to {:?}", operation_type.as_str(), timeout_duration);
    with_timeout(timeout_duration, future).await
}

/// Types of network operations for timeout selection
#[derive(Debug, Clone, Copy)]
pub enum TimeoutType {
    Connect,
    Request,
    Read,
    Write,
    Keepalive,
    Total,
}

impl TimeoutType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeoutType::Connect => "connect",
            TimeoutType::Request => "request",
            TimeoutType::Read => "read",
            TimeoutType::Write => "write",
            TimeoutType::Keepalive => "keepalive",
            TimeoutType::Total => "total",
        }
    }
}

/// A timeout guard that can be used to implement multi-stage timeouts
pub struct TimeoutGuard {
    total_timeout: Duration,
    started_at: Instant,
    config: TimeoutConfig,
}

impl TimeoutGuard {
    /// Create a new timeout guard with the given configuration
    pub fn new(config: TimeoutConfig) -> Self {
        Self {
            total_timeout: config.total_timeout,
            started_at: Instant::now(),
            config,
        }
    }

    /// Get remaining time for the total timeout
    pub fn remaining_time(&self) -> Option<Duration> {
        let elapsed = self.started_at.elapsed();
        self.total_timeout.checked_sub(elapsed)
    }

    /// Get timeout for a specific operation type, considering remaining total time
    pub fn timeout_for(&self, operation_type: TimeoutType) -> Option<Duration> {
        let operation_timeout = match operation_type {
            TimeoutType::Connect => self.config.connect_timeout,
            TimeoutType::Request => self.config.request_timeout,
            TimeoutType::Read => self.config.read_timeout,
            TimeoutType::Write => self.config.write_timeout,
            TimeoutType::Keepalive => self.config.keepalive_timeout,
            TimeoutType::Total => return self.remaining_time(),
        };

        // Return the smaller of operation timeout and remaining total time
        match self.remaining_time() {
            Some(remaining) => Some(operation_timeout.min(remaining)),
            None => None, // Total timeout already exceeded
        }
    }

    /// Check if the total timeout has been exceeded
    pub fn is_expired(&self) -> bool {
        self.remaining_time().is_none()
    }

    /// Execute a future with the appropriate timeout for the operation type
    pub async fn execute<F, T>(
        &self,
        operation_type: TimeoutType,
        future: F,
    ) -> TimeoutResult<T>
    where
        F: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
    {
        if self.is_expired() {
            warn!("TimeoutGuard: Total timeout already exceeded");
            return TimeoutResult::Timeout;
        }

        match self.timeout_for(operation_type) {
            Some(timeout_duration) => {
                debug!(
                    "TimeoutGuard: Executing {} operation with {:?} timeout",
                    operation_type.as_str(),
                    timeout_duration
                );
                with_timeout(timeout_duration, future).await
            }
            None => {
                warn!("TimeoutGuard: No time remaining for {} operation", operation_type.as_str());
                TimeoutResult::Timeout
            }
        }
    }
}

/// Convenience functions for common timeout operations

/// Execute a connection operation with timeout
pub async fn connect_with_timeout<F, T>(
    config: &TimeoutConfig,
    future: F,
) -> Result<T, TimeoutError>
where
    F: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
{
    with_config_timeout(config, TimeoutType::Connect, future)
        .await
        .into_result()
}

/// Execute a request operation with timeout
pub async fn request_with_timeout<F, T>(
    config: &TimeoutConfig,
    future: F,
) -> Result<T, TimeoutError>
where
    F: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
{
    with_config_timeout(config, TimeoutType::Request, future)
        .await
        .into_result()
}

/// Execute a read operation with timeout
pub async fn read_with_timeout<F, T>(
    config: &TimeoutConfig,
    future: F,
) -> Result<T, TimeoutError>
where
    F: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
{
    with_config_timeout(config, TimeoutType::Read, future)
        .await
        .into_result()
}

/// Execute a write operation with timeout
pub async fn write_with_timeout<F, T>(
    config: &TimeoutConfig,
    future: F,
) -> Result<T, TimeoutError>
where
    F: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
{
    with_config_timeout(config, TimeoutType::Write, future)
        .await
        .into_result()
}

/// A future that wraps another future with adaptive timeout
pub struct AdaptiveTimeout<F> {
    future: Pin<Box<F>>,
    sleep: Pin<Box<Sleep>>,
    timeout_duration: Duration,
    started_at: Instant,
}

impl<F> AdaptiveTimeout<F>
where
    F: Future,
{
    /// Create a new adaptive timeout future
    pub fn new(future: F, initial_timeout: Duration) -> Self {
        let sleep = Box::pin(tokio::time::sleep(initial_timeout));
        Self {
            future: Box::pin(future),
            sleep,
            timeout_duration: initial_timeout,
            started_at: Instant::now(),
        }
    }

    /// Extend the timeout by a certain duration
    pub fn extend_timeout(&mut self, additional_time: Duration) {
        let new_timeout = self.timeout_duration + additional_time;
        let elapsed = self.started_at.elapsed();
        
        if elapsed < new_timeout {
            let remaining = new_timeout - elapsed;
            self.sleep = Box::pin(tokio::time::sleep(remaining));
            self.timeout_duration = new_timeout;
            debug!("Extended timeout by {:?}, new timeout: {:?}", additional_time, new_timeout);
        }
    }
}

impl<F> Future for AdaptiveTimeout<F>
where
    F: Future,
{
    type Output = Result<F::Output, TimeoutError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Poll the main future first
        if let Poll::Ready(output) = self.future.as_mut().poll(cx) {
            return Poll::Ready(Ok(output));
        }

        // Poll the timeout
        if let Poll::Ready(_) = self.sleep.as_mut().poll(cx) {
            return Poll::Ready(Err(TimeoutError::Timeout));
        }

        Poll::Pending
    }
}

/// Batch timeout manager for handling multiple operations with shared timeout budget
pub struct BatchTimeoutManager {
    total_timeout: Duration,
    started_at: Instant,
    operation_timeouts: Vec<Duration>,
    completed_operations: usize,
}

impl BatchTimeoutManager {
    /// Create a new batch timeout manager
    pub fn new(total_timeout: Duration, expected_operations: usize) -> Self {
        // Distribute timeout evenly across operations initially
        let per_operation_timeout = total_timeout / expected_operations as u32;
        let operation_timeouts = vec![per_operation_timeout; expected_operations];

        Self {
            total_timeout,
            started_at: Instant::now(),
            operation_timeouts,
            completed_operations: 0,
        }
    }

    /// Get timeout for the next operation
    pub fn next_operation_timeout(&mut self) -> Option<Duration> {
        if self.completed_operations >= self.operation_timeouts.len() {
            return None;
        }

        let remaining_total = self.total_timeout.checked_sub(self.started_at.elapsed())?;
        let remaining_operations = self.operation_timeouts.len() - self.completed_operations;
        
        if remaining_operations == 0 {
            return Some(remaining_total);
        }

        // Redistribute remaining time across remaining operations
        let per_operation = remaining_total / remaining_operations as u32;
        let planned_timeout = self.operation_timeouts[self.completed_operations];
        
        Some(per_operation.min(planned_timeout))
    }

    /// Mark an operation as completed
    pub fn mark_operation_completed(&mut self) {
        self.completed_operations += 1;
    }

    /// Check if there's time remaining for more operations
    pub fn has_time_remaining(&self) -> bool {
        self.started_at.elapsed() < self.total_timeout
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_timeout_result_success() {
        let result = with_timeout(Duration::from_secs(1), async {
            Ok::<i32, Box<dyn std::error::Error + Send + Sync>>(42)
        }).await;

        assert!(result.is_success());
        assert_eq!(result.success().unwrap(), 42);
    }

    #[tokio::test]
    async fn test_timeout_result_timeout() {
        let result = with_timeout(Duration::from_millis(10), async {
            sleep(Duration::from_millis(50)).await;
            Ok::<i32, Box<dyn std::error::Error + Send + Sync>>(42)
        }).await;

        assert!(result.is_timeout());
        assert!(result.success().is_none());
    }

    #[tokio::test]
    async fn test_timeout_guard() {
        let config = TimeoutConfig {
            total_timeout: Duration::from_millis(100),
            connect_timeout: Duration::from_millis(50),
            ..Default::default()
        };

        let guard = TimeoutGuard::new(config);
        
        // Should have time remaining initially
        assert!(!guard.is_expired());
        assert!(guard.remaining_time().is_some());
        
        // Should get appropriate timeout for connect operation
        let connect_timeout = guard.timeout_for(TimeoutType::Connect);
        assert!(connect_timeout.is_some());
    }

    #[tokio::test]
    async fn test_timeout_guard_expiry() {
        let config = TimeoutConfig {
            total_timeout: Duration::from_millis(10),
            ..Default::default()
        };

        let guard = TimeoutGuard::new(config);
        
        // Wait for timeout to expire
        sleep(Duration::from_millis(20)).await;
        
        assert!(guard.is_expired());
        assert!(guard.remaining_time().is_none());
    }

    #[tokio::test]
    async fn test_batch_timeout_manager() {
        let mut manager = BatchTimeoutManager::new(Duration::from_millis(100), 3);
        
        assert!(manager.has_time_remaining());
        
        let timeout1 = manager.next_operation_timeout();
        assert!(timeout1.is_some());
        
        manager.mark_operation_completed();
        
        let timeout2 = manager.next_operation_timeout();
        assert!(timeout2.is_some());
    }

    #[test]
    fn test_timeout_config_presets() {
        let low_latency = TimeoutConfig::low_latency();
        assert!(low_latency.connect_timeout < Duration::from_secs(5));
        
        let high_throughput = TimeoutConfig::high_throughput();
        assert!(high_throughput.total_timeout > Duration::from_secs(60));
        
        let tcp = TimeoutConfig::tcp_connection();
        assert!(tcp.connect_timeout < Duration::from_secs(15));
    }

    #[tokio::test]
    async fn test_adaptive_timeout() {
        let mut adaptive = AdaptiveTimeout::new(
            async {
                sleep(Duration::from_millis(30)).await;
                42
            },
            Duration::from_millis(10)
        );

        // Should timeout initially
        let result = tokio::time::timeout(Duration::from_millis(20), &mut adaptive).await;
        assert!(result.is_err()); // Timeout

        // Extend timeout and try again
        adaptive.extend_timeout(Duration::from_millis(50));
        let result = adaptive.await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
}