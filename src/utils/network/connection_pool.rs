use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use log::{debug, warn, error, info};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Configuration for connection pooling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    /// Maximum number of connections per endpoint
    pub max_connections_per_endpoint: usize,
    /// Maximum total connections across all endpoints
    pub max_total_connections: usize,
    /// How long to wait for an available connection
    pub connection_timeout: Duration,
    /// How long a connection can be idle before being closed
    pub idle_timeout: Duration,
    /// How long a connection can exist before being recycled
    pub max_connection_lifetime: Duration,
    /// How often to run cleanup of idle/expired connections
    pub cleanup_interval: Duration,
    /// Enable connection validation before reuse
    pub validate_on_borrow: bool,
    /// Enable connection validation while idle
    pub validate_while_idle: bool,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_endpoint: 10,
            max_total_connections: 100,
            connection_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            max_connection_lifetime: Duration::from_secs(3600), // 1 hour
            cleanup_interval: Duration::from_secs(60),
            validate_on_borrow: true,
            validate_while_idle: false,
        }
    }
}

impl ConnectionPoolConfig {
    /// Configuration optimized for high-throughput scenarios
    pub fn high_throughput() -> Self {
        Self {
            max_connections_per_endpoint: 20,
            max_total_connections: 200,
            connection_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(600), // 10 minutes
            max_connection_lifetime: Duration::from_secs(7200), // 2 hours
            cleanup_interval: Duration::from_secs(30),
            validate_on_borrow: false, // Skip validation for performance
            validate_while_idle: true,
        }
    }

    /// Configuration for low-latency scenarios
    pub fn low_latency() -> Self {
        Self {
            max_connections_per_endpoint: 5,
            max_total_connections: 50,
            connection_timeout: Duration::from_secs(2),
            idle_timeout: Duration::from_secs(120), // 2 minutes
            max_connection_lifetime: Duration::from_secs(1800), // 30 minutes
            cleanup_interval: Duration::from_secs(15),
            validate_on_borrow: true,
            validate_while_idle: false,
        }
    }
}

/// Represents a pooled TCP connection
#[derive(Debug)]
pub struct PooledConnection {
    pub id: String,
    pub stream: TcpStream,
    pub created_at: Instant,
    pub last_used: Instant,
    pub endpoint: String,
    pub usage_count: usize,
}

impl PooledConnection {
    pub fn new(stream: TcpStream, endpoint: String) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4().to_string(),
            stream,
            created_at: now,
            last_used: now,
            endpoint,
            usage_count: 0,
        }
    }

    /// Check if connection is expired based on age
    pub fn is_expired(&self, max_lifetime: Duration) -> bool {
        self.created_at.elapsed() > max_lifetime
    }

    /// Check if connection is idle for too long
    pub fn is_idle(&self, idle_timeout: Duration) -> bool {
        self.last_used.elapsed() > idle_timeout
    }

    /// Update last used timestamp and increment usage count
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.usage_count += 1;
    }

    /// Validate connection by attempting a simple operation
    pub async fn validate(&self) -> bool {
        // For TCP streams, we can check if the socket is still connected
        // This is a basic check - in practice you might want to send a ping
        match self.stream.peer_addr() {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}

/// Statistics for the connection pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub connections_per_endpoint: HashMap<String, usize>,
    pub total_requests: u64,
    pub successful_borrows: u64,
    pub failed_borrows: u64,
    pub timeouts: u64,
    pub connection_creates: u64,
    pub connection_closes: u64,
    pub validations_passed: u64,
    pub validations_failed: u64,
}

/// Connection pool for managing TCP connections
pub struct ConnectionPool {
    config: ConnectionPoolConfig,
    connections: Arc<RwLock<HashMap<String, Vec<PooledConnection>>>>,
    connection_semaphore: Arc<Semaphore>,
    stats: Arc<Mutex<ConnectionPoolStats>>,
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ConnectionPool {
    /// Create a new connection pool with the given configuration
    pub fn new(config: ConnectionPoolConfig) -> Self {
        let pool = Self {
            connection_semaphore: Arc::new(Semaphore::new(config.max_total_connections)),
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(ConnectionPoolStats {
                total_connections: 0,
                active_connections: 0,
                idle_connections: 0,
                connections_per_endpoint: HashMap::new(),
                total_requests: 0,
                successful_borrows: 0,
                failed_borrows: 0,
                timeouts: 0,
                connection_creates: 0,
                connection_closes: 0,
                validations_passed: 0,
                validations_failed: 0,
            })),
            cleanup_handle: None,
        };

        pool
    }

    /// Start the cleanup task
    pub fn start_cleanup_task(&mut self) {
        let connections = self.connections.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.cleanup_interval);
            
            loop {
                interval.tick().await;
                Self::cleanup_connections(&connections, &stats, &config).await;
            }
        });

        self.cleanup_handle = Some(handle);
    }

    /// Get a connection from the pool or create a new one
    pub async fn get_connection(&self, endpoint: &str) -> Result<PooledConnection, ConnectionPoolError> {
        let start_time = Instant::now();
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.total_requests += 1;
        }

        // Try to acquire a semaphore permit
        let permit = match tokio::time::timeout(
            self.config.connection_timeout,
            self.connection_semaphore.acquire(),
        ).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                self.record_timeout().await;
                return Err(ConnectionPoolError::SemaphoreAcquisitionFailed);
            }
            Err(_) => {
                self.record_timeout().await;
                return Err(ConnectionPoolError::Timeout);
            }
        };

        // Try to get an existing connection first
        if let Some(connection) = self.borrow_existing_connection(endpoint).await? {
            permit.forget(); // Keep the permit as we're using a connection
            return Ok(connection);
        }

        // Create a new connection
        match self.create_new_connection(endpoint).await {
            Ok(connection) => {
                permit.forget(); // Keep the permit as we're using a connection
                self.record_successful_borrow().await;
                Ok(connection)
            }
            Err(err) => {
                drop(permit); // Release the permit as we failed to create a connection
                self.record_failed_borrow().await;
                Err(err)
            }
        }
    }

    /// Return a connection to the pool
    pub async fn return_connection(&self, mut connection: PooledConnection) -> Result<(), ConnectionPoolError> {
        connection.mark_used();

        // Validate connection if configured
        if self.config.validate_on_borrow && !connection.validate().await {
            self.record_validation_failed().await;
            self.connection_semaphore.add_permits(1);
            return Ok(()); // Don't return invalid connections to pool
        }

        // Check if connection is expired
        if connection.is_expired(self.config.max_connection_lifetime) {
            debug!("Connection {} expired, not returning to pool", connection.id);
            self.connection_semaphore.add_permits(1);
            return Ok(());
        }

        // Check connection limit for this endpoint
        let endpoint = connection.endpoint.clone();
        let mut connections = self.connections.write().await;
        
        let endpoint_connections = connections.entry(endpoint.clone()).or_insert_with(Vec::new);
        
        if endpoint_connections.len() >= self.config.max_connections_per_endpoint {
            debug!("Endpoint {} at connection limit, not returning connection", endpoint);
            self.connection_semaphore.add_permits(1);
            return Ok(());
        }

        endpoint_connections.push(connection);
        self.update_connection_stats().await;
        
        debug!("Returned connection to pool for endpoint: {}", endpoint);
        Ok(())
    }

    /// Get current pool statistics
    pub async fn stats(&self) -> ConnectionPoolStats {
        self.stats.lock().await.clone()
    }

    /// Close all connections and shutdown the pool
    pub async fn shutdown(&mut self) {
        info!("Shutting down connection pool");
        
        // Cancel cleanup task
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }

        // Close all connections
        let mut connections = self.connections.write().await;
        for (endpoint, conn_list) in connections.drain() {
            debug!("Closing {} connections for endpoint: {}", conn_list.len(), endpoint);
        }

        info!("Connection pool shutdown complete");
    }

    // Private methods

    async fn borrow_existing_connection(&self, endpoint: &str) -> Result<Option<PooledConnection>, ConnectionPoolError> {
        let mut connections = self.connections.write().await;
        
        if let Some(endpoint_connections) = connections.get_mut(endpoint) {
            // Try to find a valid connection
            while let Some(mut connection) = endpoint_connections.pop() {
                // Check if connection is still valid
                if self.config.validate_on_borrow && !connection.validate().await {
                    self.record_validation_failed().await;
                    continue;
                }

                // Check if connection is expired
                if connection.is_expired(self.config.max_connection_lifetime) {
                    debug!("Connection {} expired, creating new one", connection.id);
                    continue;
                }

                // Check if connection is idle for too long
                if connection.is_idle(self.config.idle_timeout) {
                    debug!("Connection {} idle for too long, creating new one", connection.id);
                    continue;
                }

                connection.mark_used();
                self.record_validation_passed().await;
                self.record_successful_borrow().await;
                self.update_connection_stats().await;
                
                debug!("Reusing existing connection {} for endpoint: {}", connection.id, endpoint);
                return Ok(Some(connection));
            }
        }

        Ok(None)
    }

    async fn create_new_connection(&self, endpoint: &str) -> Result<PooledConnection, ConnectionPoolError> {
        debug!("Creating new connection to: {}", endpoint);
        
        let stream = match tokio::time::timeout(
            self.config.connection_timeout,
            TcpStream::connect(endpoint),
        ).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(err)) => {
                error!("Failed to connect to {}: {}", endpoint, err);
                return Err(ConnectionPoolError::ConnectionFailed(err.to_string()));
            }
            Err(_) => {
                warn!("Connection timeout to: {}", endpoint);
                return Err(ConnectionPoolError::Timeout);
            }
        };

        // Set TCP options for optimal performance
        if let Err(err) = stream.set_nodelay(true) {
            warn!("Failed to set TCP_NODELAY: {}", err);
        }

        let connection = PooledConnection::new(stream, endpoint.to_string());
        
        self.record_connection_created().await;
        self.update_connection_stats().await;
        
        debug!("Created new connection {} to: {}", connection.id, endpoint);
        Ok(connection)
    }

    async fn cleanup_connections(
        connections: &Arc<RwLock<HashMap<String, Vec<PooledConnection>>>>,
        stats: &Arc<Mutex<ConnectionPoolStats>>,
        config: &ConnectionPoolConfig,
    ) {
        let mut connections_guard = connections.write().await;
        let mut closed_count = 0;

        for (endpoint, endpoint_connections) in connections_guard.iter_mut() {
            let original_len = endpoint_connections.len();
            
            endpoint_connections.retain(|conn| {
                let should_keep = !conn.is_expired(config.max_connection_lifetime) 
                    && !conn.is_idle(config.idle_timeout);
                
                if !should_keep {
                    closed_count += 1;
                    debug!("Cleaning up connection {} for endpoint {}", conn.id, endpoint);
                }
                
                should_keep
            });

            if original_len != endpoint_connections.len() {
                debug!(
                    "Cleaned up {} connections for endpoint: {}",
                    original_len - endpoint_connections.len(),
                    endpoint
                );
            }
        }

        // Remove empty endpoint entries
        connections_guard.retain(|_, conns| !conns.is_empty());

        if closed_count > 0 {
            let mut stats_guard = stats.lock().await;
            stats_guard.connection_closes += closed_count;
            debug!("Cleanup closed {} connections", closed_count);
        }
    }

    // Statistics methods
    async fn record_successful_borrow(&self) {
        let mut stats = self.stats.lock().await;
        stats.successful_borrows += 1;
    }

    async fn record_failed_borrow(&self) {
        let mut stats = self.stats.lock().await;
        stats.failed_borrows += 1;
    }

    async fn record_timeout(&self) {
        let mut stats = self.stats.lock().await;
        stats.timeouts += 1;
    }

    async fn record_connection_created(&self) {
        let mut stats = self.stats.lock().await;
        stats.connection_creates += 1;
    }

    async fn record_validation_passed(&self) {
        let mut stats = self.stats.lock().await;
        stats.validations_passed += 1;
    }

    async fn record_validation_failed(&self) {
        let mut stats = self.stats.lock().await;
        stats.validations_failed += 1;
    }

    async fn update_connection_stats(&self) {
        let connections = self.connections.read().await;
        let mut stats = self.stats.lock().await;
        
        stats.total_connections = connections.values().map(|v| v.len()).sum();
        stats.idle_connections = stats.total_connections; // All pooled connections are idle
        stats.active_connections = 0; // Active connections are not in the pool
        
        stats.connections_per_endpoint.clear();
        for (endpoint, conns) in connections.iter() {
            stats.connections_per_endpoint.insert(endpoint.clone(), conns.len());
        }
    }
}

impl Drop for ConnectionPool {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }
    }
}

/// Errors that can occur with connection pool operations
#[derive(Debug, thiserror::Error)]
pub enum ConnectionPoolError {
    #[error("Connection timeout")]
    Timeout,
    #[error("Failed to acquire semaphore permit")]
    SemaphoreAcquisitionFailed,
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Connection validation failed")]
    ValidationFailed,
    #[error("Pool is shutting down")]
    PoolShuttingDown,
}

/// Global connection pool registry
pub struct ConnectionPoolRegistry {
    pools: Arc<RwLock<HashMap<String, Arc<Mutex<ConnectionPool>>>>>,
}

impl ConnectionPoolRegistry {
    pub fn new() -> Self {
        Self {
            pools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a connection pool for a specific service
    pub async fn get_or_create_pool(&self, service_name: &str, config: ConnectionPoolConfig) -> Arc<Mutex<ConnectionPool>> {
        let mut pools = self.pools.write().await;
        
        if let Some(pool) = pools.get(service_name) {
            pool.clone()
        } else {
            let mut pool = ConnectionPool::new(config);
            pool.start_cleanup_task();
            let pool_arc = Arc::new(Mutex::new(pool));
            pools.insert(service_name.to_string(), pool_arc.clone());
            info!("Created new connection pool for service: {}", service_name);
            pool_arc
        }
    }

    /// Get statistics for all pools
    pub async fn get_all_stats(&self) -> HashMap<String, ConnectionPoolStats> {
        let pools = self.pools.read().await;
        let mut all_stats = HashMap::new();
        
        for (name, pool) in pools.iter() {
            let pool_guard = pool.lock().await;
            all_stats.insert(name.clone(), pool_guard.stats().await);
        }
        
        all_stats
    }

    /// Shutdown all pools
    pub async fn shutdown_all(&self) {
        let mut pools = self.pools.write().await;
        for (name, pool) in pools.drain() {
            info!("Shutting down pool: {}", name);
            let mut pool_guard = pool.lock().await;
            pool_guard.shutdown().await;
        }
    }
}

impl Default for ConnectionPoolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let config = ConnectionPoolConfig::default();
        let mut pool = ConnectionPool::new(config);
        pool.start_cleanup_task();
        
        let stats = pool.stats().await;
        assert_eq!(stats.total_connections, 0);
        
        pool.shutdown().await;
    }

    #[tokio::test]
    async fn test_pooled_connection_validation() {
        // This test requires a real connection, so we'll skip it in unit tests
        // In integration tests, you would create a real TcpStream and test validation
    }

    #[tokio::test]
    async fn test_connection_expiry() {
        let stream = tokio::net::TcpStream::connect("127.0.0.1:80").await;
        
        // Skip test if we can't connect (no server running)
        if stream.is_err() {
            return;
        }
        
        let mut connection = PooledConnection::new(stream.unwrap(), "127.0.0.1:80".to_string());
        
        // Test that new connection is not expired
        assert!(!connection.is_expired(Duration::from_secs(60)));
        
        // Test that connection becomes idle
        sleep(Duration::from_millis(10)).await;
        assert!(connection.is_idle(Duration::from_millis(5)));
        
        // Mark as used and check it's no longer idle
        connection.mark_used();
        assert!(!connection.is_idle(Duration::from_millis(5)));
    }

    #[tokio::test]
    async fn test_connection_pool_registry() {
        let registry = ConnectionPoolRegistry::new();
        
        let pool1 = registry.get_or_create_pool("service1", ConnectionPoolConfig::default()).await;
        let pool2 = registry.get_or_create_pool("service1", ConnectionPoolConfig::default()).await;
        
        // Should return the same pool instance
        assert!(Arc::ptr_eq(&pool1, &pool2));
        
        registry.shutdown_all().await;
    }
}