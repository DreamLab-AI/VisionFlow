// DEPRECATED: Legacy hybrid performance optimizer removed
// Docker exec architecture replaced by HTTP Management API
// Connection pooling now handled by reqwest HTTP client

/*
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex, Semaphore};
use tokio::time::{sleep, timeout};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use chrono::{DateTime, Utc};
use log::{info, warn, error, debug};

use crate::utils::docker_hive_mind::{DockerHiveMind, SessionInfo, SwarmMetrics};
use crate::utils::mcp_connection::{MCPConnectionPool, PersistentMCPConnection};

/// Performance optimization strategies for the hybrid architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    ConnectionPooling,
    LazyLoading,
    Caching,
    LoadBalancing,
    ResourceCleanup,
    BatchOperations,
}

/// Connection pool manager for Docker operations
pub struct DockerConnectionPool {
    connections: Arc<Mutex<VecDeque<DockerConnection>>>,
    active_connections: Arc<RwLock<HashMap<String, DockerConnection>>>,
    semaphore: Arc<Semaphore>,
    pool_config: PoolConfig,
    metrics: Arc<RwLock<PoolMetrics>>,
}

#[derive(Debug, Clone)]
pub struct DockerConnection {
    id: String,
    created_at: Instant,
    last_used: Instant,
    usage_count: u64,
    is_healthy: bool,
    hive_mind: Arc<DockerHiveMind>,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    max_connections: u32,
    min_connections: u32,
    connection_timeout: Duration,
    idle_timeout: Duration,
    health_check_interval: Duration,
    cleanup_interval: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 2,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            health_check_interval: Duration::from_secs(60),
            cleanup_interval: Duration::from_secs(120),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PoolMetrics {
    total_connections: u32,
    active_connections: u32,
    idle_connections: u32,
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    average_response_time_ms: f64,
    peak_connections: u32,
    last_cleanup: DateTime<Utc>,
}

impl Default for PoolMetrics {
    fn default() -> Self {
        Self {
            total_connections: 0,
            active_connections: 0,
            idle_connections: 0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_response_time_ms: 0.0,
            peak_connections: 0,
            last_cleanup: Utc::now(),
        }
    }
}

impl DockerConnectionPool {
    pub fn new(config: PoolConfig) -> Self {
        let max_connections = config.max_connections as usize;

        Self {
            connections: Arc::new(Mutex::new(VecDeque::new())),
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(max_connections)),
            pool_config: config,
            metrics: Arc::new(RwLock::new(PoolMetrics::default())),
        }
    }

    pub async fn get_connection(&self) -> Result<DockerConnection, Box<dyn std::error::Error + Send + Sync>> {
        // Acquire semaphore permit
        let _permit = self.semaphore.clone().acquire_owned().await
            .map_err(|_| "Failed to acquire connection permit")?;

        let start_time = Instant::now();

        // Try to get existing idle connection
        {
            let mut connections = self.connections.lock().await;
            if let Some(mut conn) = connections.pop_front() {
                if conn.is_healthy && start_time.duration_since(conn.last_used) < self.pool_config.idle_timeout {
                    conn.last_used = start_time;
                    conn.usage_count += 1;

                    // Move to active connections
                    let mut active = self.active_connections.write().await;
                    active.insert(conn.id.clone(), conn.clone());

                    self.update_metrics_on_acquire().await;
                    return Ok(conn);
                }
            }
        }

        // Create new connection
        let connection_id = uuid::Uuid::new_v4().to_string();
        let docker_connection = DockerConnection {
            id: connection_id.clone(),
            created_at: start_time,
            last_used: start_time,
            usage_count: 1,
            is_healthy: true,
            hive_mind: Arc::new(crate::utils::docker_hive_mind::create_docker_hive_mind()),
        };

        // Test connection health
        match timeout(
            self.pool_config.connection_timeout,
            docker_connection.hive_mind.health_check()
        ).await {
            Ok(Ok(_)) => {
                // Add to active connections
                let mut active = self.active_connections.write().await;
                active.insert(connection_id, docker_connection.clone());

                self.update_metrics_on_acquire().await;
                info!("Created new Docker connection: {}", docker_connection.id);
                Ok(docker_connection)
            },
            Ok(Err(e)) => {
                error!("Docker connection health check failed: {}", e);
                Err(format!("Connection health check failed: {}", e).into())
            },
            Err(_) => {
                error!("Docker connection timeout");
                Err("Connection timeout".into())
            }
        }
    }

    pub async fn return_connection(&self, mut connection: DockerConnection) {
        connection.last_used = Instant::now();

        // Remove from active connections
        {
            let mut active = self.active_connections.write().await;
            active.remove(&connection.id);
        }

        // Check if connection is still healthy and not too old
        let connection_age = connection.last_used.duration_since(connection.created_at);
        if connection.is_healthy && connection_age < Duration::from_secs(3600) {
            // Return to idle pool
            let mut connections = self.connections.lock().await;
            connections.push_back(connection);
        } else {
            debug!("Discarding unhealthy/old connection: {}", connection.id);
        }

        self.update_metrics_on_return().await;
    }

    pub async fn cleanup_idle_connections(&self) {
        let mut cleaned = 0u32;
        let now = Instant::now();

        {
            let mut connections = self.connections.lock().await;
            connections.retain(|conn| {
                let is_fresh = now.duration_since(conn.last_used) < self.pool_config.idle_timeout;
                if !is_fresh {
                    cleaned += 1;
                }
                is_fresh && conn.is_healthy
            });
        }

        if cleaned > 0 {
            info!("Cleaned up {} idle Docker connections", cleaned);
        }

        {
            let mut metrics = self.metrics.write().await;
            metrics.last_cleanup = Utc::now();
        }
    }

    pub async fn get_metrics(&self) -> PoolMetrics {
        let active_count = self.active_connections.read().await.len() as u32;
        let idle_count = self.connections.lock().await.len() as u32;

        let mut metrics = self.metrics.write().await;
        metrics.active_connections = active_count;
        metrics.idle_connections = idle_count;
        metrics.total_connections = active_count + idle_count;

        metrics.clone()
    }

    async fn update_metrics_on_acquire(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        metrics.successful_requests += 1;

        let current_active = self.active_connections.read().await.len() as u32;
        if current_active > metrics.peak_connections {
            metrics.peak_connections = current_active;
        }
    }

    async fn update_metrics_on_return(&self) {
        // Metrics updated in return_connection if needed
    }

    pub async fn start_maintenance(&self) {
        info!("Starting Docker connection pool maintenance");

        loop {
            sleep(self.pool_config.cleanup_interval).await;
            self.cleanup_idle_connections().await;
            self.health_check_connections().await;
        }
    }

    async fn health_check_connections(&self) {
        let mut unhealthy_connections = Vec::new();

        {
            let active = self.active_connections.read().await;
            for (id, conn) in active.iter() {
                // Quick health check (non-blocking)
                if !conn.is_healthy || conn.last_used.elapsed() > Duration::from_secs(2 * 3600) {
                    unhealthy_connections.push(id.clone());
                }
            }
        }

        if !unhealthy_connections.is_empty() {
            let mut active = self.active_connections.write().await;
            for id in unhealthy_connections {
                active.remove(&id);
                debug!("Removed unhealthy connection: {}", id);
            }
        }
    }
}

impl DockerConnection {
    pub async fn execute_operation<F, T>(&self, operation: F) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
    where
        F: std::future::Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
    {
        let start_time = Instant::now();

        match operation.await {
            Ok(result) => {
                debug!("Docker operation completed in {:?}", start_time.elapsed());
                Ok(result)
            },
            Err(e) => {
                error!("Docker operation failed after {:?}: {}", start_time.elapsed(), e);
                Err(e)
            }
        }
    }
}

/// Caching mechanisms for improved performance
pub struct HybridCache {
    session_cache: Arc<RwLock<HashMap<String, CachedSession>>>,
    telemetry_cache: Arc<RwLock<HashMap<String, CachedTelemetry>>>,
    health_cache: Arc<RwLock<Option<CachedHealth>>>,
    response_cache: Arc<RwLock<HashMap<String, CachedResponse>>>,
    cache_config: CacheConfig,
    cache_metrics: Arc<RwLock<CacheMetrics>>,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    session_ttl: Duration,
    telemetry_ttl: Duration,
    health_ttl: Duration,
    response_ttl: Duration,
    max_entries: usize,
    cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            session_ttl: Duration::from_secs(60),      // 1 minute
            telemetry_ttl: Duration::from_secs(10),    // 10 seconds
            health_ttl: Duration::from_secs(30),       // 30 seconds
            response_ttl: Duration::from_secs(120),    // 2 minutes
            max_entries: 1000,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CachedSession {
    pub data: SessionInfo,
    pub cached_at: DateTime<Utc>,
    pub access_count: u64,
    pub last_access: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CachedTelemetry {
    pub data: Value,
    pub cached_at: DateTime<Utc>,
    pub source: String, // "docker" or "mcp"
    pub access_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CachedHealth {
    pub data: Value,
    pub cached_at: DateTime<Utc>,
    pub is_healthy: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CachedResponse {
    pub data: Value,
    pub cached_at: DateTime<Utc>,
    pub request_hash: String,
    pub response_time_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheMetrics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_ratio: f64,
    pub total_entries: usize,
    pub memory_usage_mb: f64,
    pub last_cleanup: DateTime<Utc>,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            cache_hit_ratio: 0.0,
            total_entries: 0,
            memory_usage_mb: 0.0,
            last_cleanup: Utc::now(),
        }
    }
}

impl HybridCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            session_cache: Arc::new(RwLock::new(HashMap::new())),
            telemetry_cache: Arc::new(RwLock::new(HashMap::new())),
            health_cache: Arc::new(RwLock::new(None)),
            response_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_config: config,
            cache_metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    pub async fn get_session(&self, session_id: &str) -> Option<SessionInfo> {
        let mut metrics = self.cache_metrics.write().await;
        metrics.total_requests += 1;

        let session_cache = self.session_cache.read().await;
        if let Some(cached) = session_cache.get(session_id) {
            let age = Utc::now() - cached.cached_at;
            if age < chrono::Duration::from_std(self.cache_config.session_ttl).unwrap() {
                metrics.cache_hits += 1;
                metrics.cache_hit_ratio = metrics.cache_hits as f64 / metrics.total_requests as f64;

                // Update access stats (would need mutable access in real implementation)
                return Some(cached.data.clone());
            }
        }

        metrics.cache_misses += 1;
        metrics.cache_hit_ratio = metrics.cache_hits as f64 / metrics.total_requests as f64;
        None
    }

    pub async fn cache_session(&self, session_id: String, session: SessionInfo) {
        let cached_session = CachedSession {
            data: session,
            cached_at: Utc::now(),
            access_count: 0,
            last_access: Utc::now(),
        };

        let mut session_cache = self.session_cache.write().await;
        session_cache.insert(session_id, cached_session);

        // Implement LRU eviction if needed
        if session_cache.len() > self.cache_config.max_entries {
            self.evict_lru_sessions(&mut session_cache).await;
        }
    }

    pub async fn get_telemetry(&self, key: &str) -> Option<Value> {
        let telemetry_cache = self.telemetry_cache.read().await;
        if let Some(cached) = telemetry_cache.get(key) {
            let age = Utc::now() - cached.cached_at;
            if age < chrono::Duration::from_std(self.cache_config.telemetry_ttl).unwrap() {
                return Some(cached.data.clone());
            }
        }
        None
    }

    pub async fn cache_telemetry(&self, key: String, data: Value, source: String) {
        let cached_telemetry = CachedTelemetry {
            data,
            cached_at: Utc::now(),
            source,
            access_count: 0,
        };

        let mut telemetry_cache = self.telemetry_cache.write().await;
        telemetry_cache.insert(key, cached_telemetry);
    }

    pub async fn get_health(&self) -> Option<Value> {
        let health_cache = self.health_cache.read().await;
        if let Some(cached) = health_cache.as_ref() {
            let age = Utc::now() - cached.cached_at;
            if age < chrono::Duration::from_std(self.cache_config.health_ttl).unwrap() {
                return Some(cached.data.clone());
            }
        }
        None
    }

    pub async fn cache_health(&self, data: Value, is_healthy: bool) {
        let cached_health = CachedHealth {
            data,
            cached_at: Utc::now(),
            is_healthy,
        };

        let mut health_cache = self.health_cache.write().await;
        *health_cache = Some(cached_health);
    }

    pub async fn cleanup_expired_entries(&self) {
        let now = Utc::now();
        let mut cleaned = 0usize;

        // Cleanup sessions
        {
            let mut session_cache = self.session_cache.write().await;
            let session_ttl = chrono::Duration::from_std(self.cache_config.session_ttl).unwrap();
            session_cache.retain(|_, cached| {
                let is_fresh = now - cached.cached_at < session_ttl;
                if !is_fresh {
                    cleaned += 1;
                }
                is_fresh
            });
        }

        // Cleanup telemetry
        {
            let mut telemetry_cache = self.telemetry_cache.write().await;
            let telemetry_ttl = chrono::Duration::from_std(self.cache_config.telemetry_ttl).unwrap();
            telemetry_cache.retain(|_, cached| {
                let is_fresh = now - cached.cached_at < telemetry_ttl;
                if !is_fresh {
                    cleaned += 1;
                }
                is_fresh
            });
        }

        // Cleanup health
        {
            let mut health_cache = self.health_cache.write().await;
            if let Some(cached) = health_cache.as_ref() {
                let health_ttl = chrono::Duration::from_std(self.cache_config.health_ttl).unwrap();
                if now - cached.cached_at >= health_ttl {
                    *health_cache = None;
                    cleaned += 1;
                }
            }
        }

        // Cleanup responses
        {
            let mut response_cache = self.response_cache.write().await;
            let response_ttl = chrono::Duration::from_std(self.cache_config.response_ttl).unwrap();
            response_cache.retain(|_, cached| {
                let is_fresh = now - cached.cached_at < response_ttl;
                if !is_fresh {
                    cleaned += 1;
                }
                is_fresh
            });
        }

        if cleaned > 0 {
            info!("Cleaned up {} expired cache entries", cleaned);
        }

        // Update metrics
        {
            let mut metrics = self.cache_metrics.write().await;
            metrics.last_cleanup = now;
            metrics.total_entries = self.count_total_entries().await;
        }
    }

    async fn count_total_entries(&self) -> usize {
        let session_count = self.session_cache.read().await.len();
        let telemetry_count = self.telemetry_cache.read().await.len();
        let health_count = if self.health_cache.read().await.is_some() { 1 } else { 0 };
        let response_count = self.response_cache.read().await.len();

        session_count + telemetry_count + health_count + response_count
    }

    async fn evict_lru_sessions(&self, session_cache: &mut HashMap<String, CachedSession>) {
        // Simple LRU eviction - remove oldest entries until under limit
        let target_size = self.cache_config.max_entries * 80 / 100; // Remove 20%

        let mut entries: Vec<_> = session_cache.iter()
            .map(|(k, v)| (k.clone(), v.last_access))
            .collect();

        entries.sort_by(|a, b| a.1.cmp(&b.1));

        let to_remove = entries.len().saturating_sub(target_size);
        for (key, _) in entries.into_iter().take(to_remove) {
            session_cache.remove(&key);
        }

        info!("Evicted {} LRU cache entries", to_remove);
    }

    pub async fn get_metrics(&self) -> CacheMetrics {
        let mut metrics = self.cache_metrics.write().await;
        metrics.total_entries = self.count_total_entries().await;

        // Estimate memory usage (rough calculation)
        metrics.memory_usage_mb = (metrics.total_entries * 1024) as f64 / (1024.0 * 1024.0);

        metrics.clone()
    }

    pub async fn start_maintenance(&self) {
        info!("Starting cache maintenance");

        loop {
            sleep(self.cache_config.cleanup_interval).await;
            self.cleanup_expired_entries().await;
        }
    }
}

/// Lazy telemetry loader for on-demand data fetching
pub struct LazyTelemetryLoader {
    cache: Arc<HybridCache>,
    docker_pool: Arc<DockerConnectionPool>,
    mcp_pool: Arc<MCPConnectionPool>,
    background_updater: Arc<BackgroundUpdater>,
    loader_config: LoaderConfig,
}

#[derive(Debug, Clone)]
pub struct LoaderConfig {
    prefetch_threshold: Duration,
    background_update_interval: Duration,
    max_concurrent_updates: usize,
    priority_refresh_keys: Vec<String>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            prefetch_threshold: Duration::from_secs(30),
            background_update_interval: Duration::from_secs(15),
            max_concurrent_updates: 5,
            priority_refresh_keys: vec![
                "system_health".to_string(),
                "active_swarms".to_string(),
                "resource_usage".to_string(),
            ],
        }
    }
}

pub struct BackgroundUpdater {
    update_queue: Arc<RwLock<VecDeque<UpdateTask>>>,
    semaphore: Arc<Semaphore>,
    is_running: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone)]
pub struct UpdateTask {
    key: String,
    priority: UpdatePriority,
    requested_at: DateTime<Utc>,
    retry_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl LazyTelemetryLoader {
    pub fn new(
        cache: Arc<HybridCache>,
        docker_pool: Arc<DockerConnectionPool>,
        mcp_pool: Arc<MCPConnectionPool>,
    ) -> Self {
        let loader_config = LoaderConfig::default();
        let background_updater = Arc::new(BackgroundUpdater {
            update_queue: Arc::new(RwLock::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(loader_config.max_concurrent_updates)),
            is_running: Arc::new(RwLock::new(false)),
        });

        Self {
            cache,
            docker_pool,
            mcp_pool,
            background_updater,
            loader_config,
        }
    }

    pub async fn get_metrics(&self, swarm_id: &str) -> Result<SwarmMetrics, Box<dyn std::error::Error + Send + Sync>> {
        // Check cache first
        if let Some(cached) = self.cache.get_telemetry(&format!("metrics:{}", swarm_id)).await {
            if let Ok(metrics) = serde_json::from_value::<SwarmMetrics>(cached) {
                debug!("Returning cached metrics for swarm: {}", swarm_id);
                return Ok(metrics);
            }
        }

        // Cache miss - fetch from Docker first, then trigger background MCP update
        let docker_conn = self.docker_pool.get_connection().await?;

        match docker_conn.hive_mind.get_swarm_metrics(swarm_id).await {
            Ok(metrics) => {
                // Cache the result
                self.cache.cache_telemetry(
                    format!("metrics:{}", swarm_id),
                    serde_json::to_value(&metrics)?,
                    "docker".to_string(),
                ).await;

                // Request background MCP update for enrichment
                self.background_updater.request_update(
                    &format!("mcp_metrics:{}", swarm_id),
                    UpdatePriority::Medium,
                ).await;

                self.docker_pool.return_connection(docker_conn).await;
                Ok(metrics)
            },
            Err(e) => {
                self.docker_pool.return_connection(docker_conn).await;
                error!("Failed to get Docker metrics for {}: {}", swarm_id, e);

                // Try to get cached data as fallback
                self.get_cached_or_placeholder_metrics(swarm_id).await
            }
        }
    }

    async fn get_cached_or_placeholder_metrics(&self, swarm_id: &str) -> Result<SwarmMetrics, Box<dyn std::error::Error + Send + Sync>> {
        // Try to get real metrics from system first
        let hive_mind = crate::utils::docker_hive_mind::create_docker_hive_mind();

        // Attempt to get session info for this swarm
        match hive_mind.get_sessions().await {
            Ok(sessions) => {
                // Find session that matches our swarm_id
                for session in sessions {
                    if session.session_id.contains(swarm_id) {
                        // Extract real metrics from session
                        return Ok(SwarmMetrics {
                            active_workers: session.workers.len() as u32,
                            completed_tasks: session.metrics.completed_tasks,
                            failed_tasks: session.metrics.failed_tasks,
                            memory_usage_mb: session.metrics.memory_usage_mb,
                            cpu_usage_percent: session.metrics.cpu_usage_percent,
                            network_io_mb: session.metrics.network_io_mb,
                            uptime_seconds: (Utc::now() - session.created_at).num_seconds() as u64,
                            consensus_decisions: session.metrics.consensus_decisions,
                            last_activity: session.last_activity,
                        });
                    }
                }
            }
            Err(e) => {
                warn!("Failed to get real swarm metrics: {}", e);
            }
        }

        // Fallback to placeholder metrics if no real data available
        Ok(SwarmMetrics {
            active_workers: 1,
            completed_tasks: 0,
            failed_tasks: 0,
            memory_usage_mb: 128.0,
            cpu_usage_percent: 25.0,
            network_io_mb: 1.0,
            uptime_seconds: 0,
            consensus_decisions: 0,
            last_activity: Utc::now(),
        })
    }

    pub async fn start_background_updates(&self) {
        {
            let mut is_running = self.background_updater.is_running.write().await;
            if *is_running {
                return; // Already running
            }
            *is_running = true;
        }

        info!("Starting background telemetry updates");

        loop {
            // Process update queue
            self.process_update_queue().await;

            // Schedule priority refreshes
            self.schedule_priority_refreshes().await;

            sleep(self.loader_config.background_update_interval).await;
        }
    }

    async fn process_update_queue(&self) {
        let tasks_to_process = {
            let mut queue = self.background_updater.update_queue.write().await;
            let mut tasks = Vec::new();

            // Take up to max_concurrent_updates tasks
            for _ in 0..self.loader_config.max_concurrent_updates {
                if let Some(task) = queue.pop_front() {
                    tasks.push(task);
                } else {
                    break;
                }
            }

            tasks
        };

        // Process tasks concurrently
        let mut handles = Vec::new();

        for task in tasks_to_process {
            let cache = Arc::clone(&self.cache);
            let docker_pool = Arc::clone(&self.docker_pool);
            let mcp_pool = Arc::clone(&self.mcp_pool);
            let semaphore = Arc::clone(&self.background_updater.semaphore);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                match Self::update_data(cache, docker_pool, mcp_pool, &task).await {
                    Ok(_) => debug!("Background update completed for key: {}", task.key),
                    Err(e) => warn!("Background update failed for key {}: {}", task.key, e),
                }
            });

            handles.push(handle);
        }

        // Wait for all updates to complete
        for handle in handles {
            let _ = handle.await;
        }
    }

    async fn update_data(
        cache: Arc<HybridCache>,
        docker_pool: Arc<DockerConnectionPool>,
        mcp_pool: Arc<MCPConnectionPool>,
        task: &UpdateTask,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if task.key.starts_with("metrics:") {
            let swarm_id = &task.key[8..]; // Remove "metrics:" prefix

            if let Ok(docker_conn) = docker_pool.get_connection().await {
                if let Ok(metrics) = docker_conn.hive_mind.get_swarm_metrics(swarm_id).await {
                    cache.cache_telemetry(
                        task.key.clone(),
                        serde_json::to_value(&metrics)?,
                        "docker".to_string(),
                    ).await;
                }
                docker_pool.return_connection(docker_conn).await;
            }
        } else if task.key.starts_with("mcp_metrics:") {
            // Fetch complementary MCP telemetry data
            if let Ok(mcp_data) = mcp_pool.execute_command(
                "background_update",
                "tools/call",
                json!({
                    "name": "task_status",
                    "arguments": { "taskId": &task.key[12..] }
                }),
            ).await {
                cache.cache_telemetry(
                    task.key.clone(),
                    mcp_data,
                    "mcp".to_string(),
                ).await;
            }
        }

        Ok(())
    }

    async fn schedule_priority_refreshes(&self) {
        for key in &self.loader_config.priority_refresh_keys {
            self.background_updater.request_update(key, UpdatePriority::High).await;
        }
    }
}

impl BackgroundUpdater {
    pub async fn request_update(&self, key: &str, priority: UpdatePriority) {
        let task = UpdateTask {
            key: key.to_string(),
            priority,
            requested_at: Utc::now(),
            retry_count: 0,
        };

        let mut queue = self.update_queue.write().await;
        queue.push_back(task);

        // Sort by priority (higher priority first)
        let mut tasks: Vec<_> = queue.drain(..).collect();
        tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        queue.extend(tasks);
    }
}

/// Main performance optimizer that coordinates all optimization strategies
pub struct HybridPerformanceOptimizer {
    docker_pool: Arc<DockerConnectionPool>,
    cache: Arc<HybridCache>,
    telemetry_loader: Arc<LazyTelemetryLoader>,
    optimizer_config: OptimizerConfig,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    optimization_strategies: Vec<OptimizationStrategy>,
    performance_monitoring_interval: Duration,
    auto_optimization: bool,
    resource_cleanup_interval: Duration,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimization_strategies: vec![
                OptimizationStrategy::ConnectionPooling,
                OptimizationStrategy::Caching,
                OptimizationStrategy::LazyLoading,
                OptimizationStrategy::ResourceCleanup,
            ],
            performance_monitoring_interval: Duration::from_secs(60),
            auto_optimization: true,
            resource_cleanup_interval: Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub cache_hit_ratio: f64,
    pub connection_pool_utilization: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub active_optimizations: Vec<OptimizationStrategy>,
    pub last_updated: DateTime<Utc>,
}

impl HybridPerformanceOptimizer {
    pub fn new(mcp_pool: Arc<MCPConnectionPool>) -> Self {
        let optimizer_config = OptimizerConfig::default();
        let docker_pool = Arc::new(DockerConnectionPool::new(PoolConfig::default()));
        let cache = Arc::new(HybridCache::new(CacheConfig::default()));
        let telemetry_loader = Arc::new(LazyTelemetryLoader::new(
            Arc::clone(&cache),
            Arc::clone(&docker_pool),
            mcp_pool,
        ));

        Self {
            docker_pool,
            cache,
            telemetry_loader,
            optimizer_config: optimizer_config.clone(),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_response_time_ms: 0.0,
                p95_response_time_ms: 0.0,
                cache_hit_ratio: 0.0,
                connection_pool_utilization: 0.0,
                memory_usage_mb: 0.0,
                cpu_usage_percent: 0.0,
                active_optimizations: optimizer_config.optimization_strategies.clone(),
                last_updated: Utc::now(),
            })),
        }
    }

    pub async fn start_optimization(&self) {
        info!("Starting hybrid performance optimization");

        // Start background tasks
        let docker_pool = Arc::clone(&self.docker_pool);
        tokio::spawn(async move {
            docker_pool.start_maintenance().await;
        });

        let cache = Arc::clone(&self.cache);
        tokio::spawn(async move {
            cache.start_maintenance().await;
        });

        let telemetry_loader = Arc::clone(&self.telemetry_loader);
        tokio::spawn(async move {
            telemetry_loader.start_background_updates().await;
        });

        // Start performance monitoring
        self.start_performance_monitoring().await;
    }

    async fn start_performance_monitoring(&self) {
        info!("Starting performance monitoring");

        loop {
            self.collect_performance_metrics().await;

            if self.optimizer_config.auto_optimization {
                self.auto_optimize().await;
            }

            sleep(self.optimizer_config.performance_monitoring_interval).await;
        }
    }

    async fn collect_performance_metrics(&self) {
        let pool_metrics = self.docker_pool.get_metrics().await;
        let cache_metrics = self.cache.get_metrics().await;

        let mut performance_metrics = self.performance_metrics.write().await;

        performance_metrics.total_requests = pool_metrics.total_requests + cache_metrics.total_requests;
        performance_metrics.successful_requests = pool_metrics.successful_requests;
        performance_metrics.failed_requests = pool_metrics.failed_requests;
        performance_metrics.average_response_time_ms = pool_metrics.average_response_time_ms;
        performance_metrics.cache_hit_ratio = cache_metrics.cache_hit_ratio;
        performance_metrics.connection_pool_utilization =
            pool_metrics.active_connections as f64 / pool_metrics.total_connections.max(1) as f64;
        performance_metrics.memory_usage_mb = cache_metrics.memory_usage_mb;
        performance_metrics.last_updated = Utc::now();
    }

    async fn auto_optimize(&self) {
        let metrics = self.performance_metrics.read().await.clone();

        // Implement auto-optimization logic based on metrics
        if metrics.cache_hit_ratio < 0.5 {
            // Low cache hit ratio - increase cache TTL or prefetch more aggressively
            info!("Low cache hit ratio detected ({}), optimizing caching strategy", metrics.cache_hit_ratio);
        }

        if metrics.connection_pool_utilization > 0.8 {
            // High connection pool utilization - consider increasing pool size
            info!("High connection pool utilization ({}), considering pool expansion", metrics.connection_pool_utilization);
        }

        if metrics.average_response_time_ms > 1000.0 {
            // High response times - enable more aggressive optimizations
            info!("High response times detected ({}ms), enabling aggressive optimizations", metrics.average_response_time_ms);
        }
    }

    pub async fn get_performance_report(&self) -> PerformanceReport {
        let metrics = self.performance_metrics.read().await.clone();
        let pool_metrics = self.docker_pool.get_metrics().await;
        let cache_metrics = self.cache.get_metrics().await;

        PerformanceReport {
            overall_metrics: metrics,
            connection_pool_metrics: pool_metrics.clone(),
            cache_metrics: cache_metrics.clone(),
            recommendations: self.generate_recommendations(&cache_metrics, &pool_metrics).await,
            timestamp: Utc::now(),
        }
    }

    async fn generate_recommendations(
        &self,
        cache_metrics: &CacheMetrics,
        pool_metrics: &PoolMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        if cache_metrics.cache_hit_ratio < 0.6 {
            recommendations.push(OptimizationRecommendation {
                category: "Caching".to_string(),
                description: "Low cache hit ratio detected".to_string(),
                action: "Consider increasing cache TTL or implementing smarter prefetching".to_string(),
                priority: "Medium".to_string(),
                estimated_impact: "10-20% response time improvement".to_string(),
            });
        }

        if pool_metrics.total_connections < 2 {
            recommendations.push(OptimizationRecommendation {
                category: "Connection Pooling".to_string(),
                description: "Very few connections in pool".to_string(),
                action: "Increase minimum connection pool size".to_string(),
                priority: "Low".to_string(),
                estimated_impact: "5-10% throughput improvement".to_string(),
            });
        }

        if pool_metrics.failed_requests > pool_metrics.successful_requests / 10 {
            recommendations.push(OptimizationRecommendation {
                category: "Reliability".to_string(),
                description: "High failure rate detected".to_string(),
                action: "Implement circuit breakers and retry mechanisms".to_string(),
                priority: "High".to_string(),
                estimated_impact: "Significant reliability improvement".to_string(),
            });
        }

        recommendations
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceReport {
    pub overall_metrics: PerformanceMetrics,
    pub connection_pool_metrics: PoolMetrics,
    pub cache_metrics: CacheMetrics,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationRecommendation {
    pub category: String,
    pub description: String,
    pub action: String,
    pub priority: String,
    pub estimated_impact: String,
}

/// Create a complete performance optimization system
pub fn create_performance_optimizer(mcp_pool: MCPConnectionPool) -> HybridPerformanceOptimizer {
    HybridPerformanceOptimizer::new(Arc::new(mcp_pool))
}*/
