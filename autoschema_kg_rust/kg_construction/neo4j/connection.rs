//! Neo4j connection management with connection pooling

use crate::neo4j::error::{Neo4jError, Result};
use deadpool::managed::{Manager, Pool, PoolConfig};
use neo4rs::{Config, Graph};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use url::Url;

/// Configuration for Neo4j connection
#[derive(Debug, Clone)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: Option<String>,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub fetch_size: Option<usize>,
    pub max_connection_lifetime: Option<Duration>,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: "password".to_string(),
            database: None,
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            fetch_size: Some(1000),
            max_connection_lifetime: Some(Duration::from_secs(3600)),
        }
    }
}

impl Neo4jConfig {
    pub fn new<S: Into<String>>(uri: S, username: S, password: S) -> Self {
        Self {
            uri: uri.into(),
            username: username.into(),
            password: password.into(),
            ..Default::default()
        }
    }
    
    pub fn with_database<S: Into<String>>(mut self, database: S) -> Self {
        self.database = Some(database.into());
        self
    }
    
    pub fn with_max_connections(mut self, max_connections: usize) -> Self {
        self.max_connections = max_connections;
        self
    }
    
    pub fn with_connection_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = timeout;
        self
    }
    
    pub fn with_fetch_size(mut self, fetch_size: usize) -> Self {
        self.fetch_size = Some(fetch_size);
        self
    }
    
    pub fn with_max_connection_lifetime(mut self, lifetime: Duration) -> Self {
        self.max_connection_lifetime = Some(lifetime);
        self
    }
    
    pub fn validate(&self) -> Result<()> {
        // Validate URI format
        let _url = Url::parse(&self.uri)
            .map_err(|e| Neo4jError::ConfigError(format!("Invalid URI: {}", e)))?;
        
        // Validate username and password are not empty
        if self.username.is_empty() {
            return Err(Neo4jError::ConfigError("Username cannot be empty".to_string()));
        }
        
        if self.password.is_empty() {
            return Err(Neo4jError::ConfigError("Password cannot be empty".to_string()));
        }
        
        // Validate max_connections
        if self.max_connections == 0 {
            return Err(Neo4jError::ConfigError("Max connections must be greater than 0".to_string()));
        }
        
        Ok(())
    }
}

/// Connection manager for Neo4j
struct Neo4jConnectionManager {
    config: Neo4jConfig,
}

impl Neo4jConnectionManager {
    fn new(config: Neo4jConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Manager for Neo4jConnectionManager {
    type Type = Graph;
    type Error = Neo4jError;
    
    async fn create(&self) -> Result<Self::Type> {
        let mut neo4j_config = Config::new()
            .uri(&self.config.uri)
            .user(&self.config.username)
            .password(&self.config.password);
        
        if let Some(ref database) = self.config.database {
            neo4j_config = neo4j_config.db(database);
        }
        
        if let Some(fetch_size) = self.config.fetch_size {
            neo4j_config = neo4j_config.fetch_size(fetch_size);
        }
        
        let graph = Graph::connect(neo4j_config).await?;
        
        // Test the connection
        let mut result = graph.execute(neo4rs::query("RETURN 1 as test")).await?;
        if result.next().await?.is_none() {
            return Err(Neo4jError::ConnectionError(
                neo4rs::Error::UnexpectedMessage("Connection test failed".to_string())
            ));
        }
        
        Ok(graph)
    }
    
    async fn recycle(&self, conn: &mut Self::Type) -> deadpool::managed::RecycleResult<Self::Error> {
        // Test if connection is still alive
        match conn.execute(neo4rs::query("RETURN 1 as test")).await {
            Ok(mut result) => {
                match result.next().await {
                    Ok(Some(_)) => Ok(()),
                    Ok(None) => Err(deadpool::managed::RecycleError::Message(
                        "Connection test returned no results".to_string()
                    )),
                    Err(e) => Err(deadpool::managed::RecycleError::Backend(Neo4jError::from(e))),
                }
            },
            Err(e) => Err(deadpool::managed::RecycleError::Backend(Neo4jError::from(e))),
        }
    }
}

/// Neo4j connection pool manager
#[derive(Clone)]
pub struct Neo4jConnectionPool {
    pool: Pool<Neo4jConnectionManager>,
    config: Neo4jConfig,
    stats: Arc<RwLock<ConnectionStats>>,
}

/// Connection statistics
#[derive(Debug, Default, Clone)]
pub struct ConnectionStats {
    pub total_connections_created: u64,
    pub active_connections: u64,
    pub failed_connections: u64,
    pub queries_executed: u64,
    pub total_query_time: Duration,
}

impl Neo4jConnectionPool {
    /// Create a new connection pool
    pub async fn new(config: Neo4jConfig) -> Result<Self> {
        config.validate()?;
        
        let manager = Neo4jConnectionManager::new(config.clone());
        
        let pool_config = PoolConfig {
            max_size: config.max_connections,
            timeouts: deadpool::managed::Timeouts {
                wait: Some(config.connection_timeout),
                create: Some(config.connection_timeout),
                recycle: Some(Duration::from_secs(10)),
            },
            ..Default::default()
        };
        
        let pool = Pool::builder(manager)
            .config(pool_config)
            .build()
            .map_err(|e| Neo4jError::PoolError(format!("Failed to create pool: {}", e)))?;
        
        let stats = Arc::new(RwLock::new(ConnectionStats::default()));
        
        // Test the pool by getting a connection
        {
            let _conn = pool.get().await
                .map_err(|e| Neo4jError::PoolError(format!("Failed to get test connection: {}", e)))?;
            
            let mut stats_guard = stats.write().await;
            stats_guard.total_connections_created += 1;
        }
        
        Ok(Self {
            pool,
            config,
            stats,
        })
    }
    
    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<deadpool::managed::Object<Neo4jConnectionManager>> {
        let start_time = std::time::Instant::now();
        
        let connection = self.pool.get().await
            .map_err(|e| Neo4jError::PoolError(format!("Failed to get connection: {}", e)))?;
        
        let mut stats = self.stats.write().await;
        stats.active_connections += 1;
        stats.total_query_time += start_time.elapsed();
        
        Ok(connection)
    }
    
    /// Execute a query with automatic connection management
    pub async fn execute_query(&self, query: neo4rs::Query) -> Result<neo4rs::RowStream> {
        let start_time = std::time::Instant::now();
        
        let conn = self.get_connection().await?;
        let result = conn.execute(query).await.map_err(|e| {
            log::error!("Query execution failed: {}", e);
            Neo4jError::from(e)
        });
        
        let mut stats = self.stats.write().await;
        stats.queries_executed += 1;
        stats.total_query_time += start_time.elapsed();
        
        if result.is_err() {
            stats.failed_connections += 1;
        }
        
        result
    }
    
    /// Get pool statistics
    pub async fn get_stats(&self) -> ConnectionStats {
        self.stats.read().await.clone()
    }
    
    /// Get pool status
    pub fn status(&self) -> deadpool::Status {
        self.pool.status()
    }
    
    /// Close the connection pool
    pub async fn close(&self) {
        self.pool.close();
        log::info!("Neo4j connection pool closed");
    }
    
    /// Health check for the connection pool
    pub async fn health_check(&self) -> Result<bool> {
        match self.get_connection().await {
            Ok(conn) => {
                match conn.execute(neo4rs::query("RETURN 1 as health_check")).await {
                    Ok(mut result) => {
                        match result.next().await {
                            Ok(Some(_)) => Ok(true),
                            _ => Ok(false),
                        }
                    },
                    Err(_) => Ok(false),
                }
            },
            Err(_) => Ok(false),
        }
    }
    
    /// Get configuration
    pub fn config(&self) -> &Neo4jConfig {
        &self.config
    }
}

/// Main connection manager that handles the pool
#[derive(Clone)]
pub struct Neo4jConnectionManager {
    pool: Neo4jConnectionPool,
}

impl Neo4jConnectionManager {
    /// Create a new connection manager
    pub async fn new(config: Neo4jConfig) -> Result<Self> {
        let pool = Neo4jConnectionPool::new(config).await?;
        Ok(Self { pool })
    }
    
    /// Create with default configuration
    pub async fn with_defaults() -> Result<Self> {
        Self::new(Neo4jConfig::default()).await
    }
    
    /// Create from environment variables
    pub async fn from_env() -> Result<Self> {
        let uri = std::env::var("NEO4J_URI")
            .unwrap_or_else(|_| "bolt://localhost:7687".to_string());
        let username = std::env::var("NEO4J_USERNAME")
            .unwrap_or_else(|_| "neo4j".to_string());
        let password = std::env::var("NEO4J_PASSWORD")
            .map_err(|_| Neo4jError::ConfigError("NEO4J_PASSWORD environment variable is required".to_string()))?;
        
        let mut config = Neo4jConfig::new(uri, username, password);
        
        if let Ok(database) = std::env::var("NEO4J_DATABASE") {
            config = config.with_database(database);
        }
        
        if let Ok(max_conn_str) = std::env::var("NEO4J_MAX_CONNECTIONS") {
            if let Ok(max_conn) = max_conn_str.parse::<usize>() {
                config = config.with_max_connections(max_conn);
            }
        }
        
        Self::new(config).await
    }
    
    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<deadpool::managed::Object<Neo4jConnectionManager>> {
        self.pool.get_connection().await
    }
    
    /// Execute a query
    pub async fn execute_query(&self, query: neo4rs::Query) -> Result<neo4rs::RowStream> {
        self.pool.execute_query(query).await
    }
    
    /// Get pool statistics
    pub async fn get_stats(&self) -> ConnectionStats {
        self.pool.get_stats().await
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<bool> {
        self.pool.health_check().await
    }
    
    /// Close connections
    pub async fn close(&self) {
        self.pool.close().await
    }
    
    /// Get configuration
    pub fn config(&self) -> &Neo4jConfig {
        self.pool.config()
    }
}

// Add async_trait to Cargo.toml dependencies if not already present
