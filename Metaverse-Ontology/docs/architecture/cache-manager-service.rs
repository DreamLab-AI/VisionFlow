// OntologyCacheManager - Reference Implementation
// Multi-level caching strategy for ontology extraction results

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use lru::LruCache;
use crate::models::AnnotatedOntology;
use crate::services::OwlExtractorService;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Cache miss for path: {0}")]
    CacheMiss(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Redis error: {0}")]
    RedisError(String),

    #[error("File system error: {0}")]
    FileSystemError(String),

    #[error("Cache eviction error: {0}")]
    EvictionError(String),
}

/// Cache entry with metadata
#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub ontology: AnnotatedOntology,
    pub cached_at: SystemTime,
    pub file_modified_at: SystemTime,
    pub size_bytes: usize,
    pub hit_count: usize,
}

impl CacheEntry {
    pub fn new(ontology: AnnotatedOntology, file_modified_at: SystemTime) -> Self {
        Self {
            size_bytes: Self::estimate_size(&ontology),
            ontology,
            cached_at: SystemTime::now(),
            file_modified_at,
            hit_count: 0,
        }
    }

    /// Estimate memory size of ontology (rough approximation)
    fn estimate_size(ontology: &AnnotatedOntology) -> usize {
        let class_size = ontology.classes.len() * 200; // ~200 bytes per class
        let property_size = ontology.properties.len() * 150;
        let individual_size = ontology.individuals.len() * 100;
        let axiom_size = ontology.axioms.len() * 80;

        class_size + property_size + individual_size + axiom_size
    }

    pub fn is_stale(&self, current_modified: SystemTime, ttl: Duration) -> bool {
        // Check if file was modified after caching
        if current_modified > self.file_modified_at {
            return true;
        }

        // Check TTL
        if let Ok(elapsed) = self.cached_at.elapsed() {
            return elapsed > ttl;
        }

        false
    }

    pub fn increment_hits(&mut self) {
        self.hit_count += 1;
    }
}

/// Cache invalidation strategy
#[derive(Debug, Clone, Copy)]
pub enum CacheInvalidationStrategy {
    /// Invalidate on file modification (watch file system)
    FileModification,
    /// Invalidate on explicit API call
    Explicit,
    /// Time-based TTL
    TTL(Duration),
    /// Hybrid (file modification + TTL)
    Hybrid(Duration),
}

impl Default for CacheInvalidationStrategy {
    fn default() -> Self {
        Self::Hybrid(Duration::from_secs(3600)) // 1 hour TTL
    }
}

/// Configuration for cache manager
#[derive(Clone, Debug)]
pub struct CacheManagerConfig {
    /// L1 cache capacity (number of ontologies)
    pub l1_capacity: usize,

    /// L1 cache maximum memory (bytes)
    pub l1_max_memory_bytes: usize,

    /// Enable L2 cache (Redis)
    pub l2_enabled: bool,

    /// L2 cache TTL
    pub l2_ttl_seconds: u64,

    /// Redis connection string
    pub redis_url: Option<String>,

    /// Invalidation strategy
    pub invalidation_strategy: CacheInvalidationStrategy,

    /// Enable cache metrics
    pub enable_metrics: bool,
}

impl Default for CacheManagerConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 100,
            l1_max_memory_bytes: 1_000_000_000, // 1GB
            l2_enabled: false,
            l2_ttl_seconds: 3600,
            redis_url: None,
            invalidation_strategy: CacheInvalidationStrategy::default(),
            enable_metrics: true,
        }
    }
}

/// Cache performance metrics
#[derive(Default, Clone, Debug)]
pub struct CacheMetrics {
    pub l1_hits: usize,
    pub l1_misses: usize,
    pub l2_hits: usize,
    pub l2_misses: usize,
    pub total_evictions: usize,
    pub total_memory_bytes: usize,
}

impl CacheMetrics {
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total == 0 {
            return 0.0;
        }
        self.l1_hits as f64 / total as f64
    }

    pub fn l2_hit_rate(&self) -> f64 {
        let total = self.l2_hits + self.l2_misses;
        if total == 0 {
            return 0.0;
        }
        self.l2_hits as f64 / total as f64
    }

    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits;
        let total_requests = total_hits + self.l1_misses;
        if total_requests == 0 {
            return 0.0;
        }
        total_hits as f64 / total_requests as f64
    }
}

/// Multi-level cache manager for ontologies
pub struct OntologyCacheManager {
    config: CacheManagerConfig,

    /// L1: In-memory LRU cache
    l1_cache: Arc<RwLock<LruCache<PathBuf, CacheEntry>>>,

    /// L2: Redis cache (optional)
    #[cfg(feature = "redis")]
    l2_cache: Option<Arc<redis::Client>>,

    /// Performance metrics
    metrics: Arc<RwLock<CacheMetrics>>,

    /// File watcher (for invalidation)
    #[cfg(feature = "file-watch")]
    file_watcher: Arc<RwLock<Option<notify::RecommendedWatcher>>>,
}

impl OntologyCacheManager {
    /// Create new cache manager
    pub fn new(config: CacheManagerConfig) -> Result<Self, CacheError> {
        let l1_cache = Arc::new(RwLock::new(
            LruCache::new(std::num::NonZeroUsize::new(config.l1_capacity).unwrap())
        ));

        #[cfg(feature = "redis")]
        let l2_cache = if config.l2_enabled {
            if let Some(redis_url) = &config.redis_url {
                let client = redis::Client::open(redis_url.as_str())
                    .map_err(|e| CacheError::RedisError(e.to_string()))?;
                Some(Arc::new(client))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            l1_cache,
            #[cfg(feature = "redis")]
            l2_cache,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
            #[cfg(feature = "file-watch")]
            file_watcher: Arc::new(RwLock::new(None)),
        })
    }

    /// Get ontology from cache or extract from file
    pub async fn get_or_extract(
        &self,
        file_path: &Path,
        extractor: &OwlExtractorService,
    ) -> Result<AnnotatedOntology, CacheError> {
        let file_path_buf = file_path.to_path_buf();

        // Try L1 cache
        if let Some(ontology) = self.l1_get(&file_path_buf).await? {
            return Ok(ontology);
        }

        // Try L2 cache (Redis)
        #[cfg(feature = "redis")]
        if self.config.l2_enabled {
            if let Some(ontology) = self.l2_get(&file_path_buf).await? {
                // Populate L1 cache
                self.l1_put(&file_path_buf, ontology.clone()).await?;
                return Ok(ontology);
            }
        }

        // L3: Extract from file
        let ontology = extractor
            .build_complete_ontology(&file_path_buf)
            .await
            .map_err(|e| CacheError::FileSystemError(e.to_string()))?;

        // Populate caches
        self.l1_put(&file_path_buf, ontology.clone()).await?;

        #[cfg(feature = "redis")]
        if self.config.l2_enabled {
            let _ = self.l2_put(&file_path_buf, &ontology).await; // Ignore L2 errors
        }

        Ok(ontology)
    }

    /// Get from L1 cache
    async fn l1_get(&self, path: &PathBuf) -> Result<Option<AnnotatedOntology>, CacheError> {
        let mut cache = self.l1_cache.write().await;

        if let Some(entry) = cache.get_mut(path) {
            // Check if entry is stale
            let file_modified = self.get_file_modified_time(path)?;
            let ttl = self.get_ttl();

            if entry.is_stale(file_modified, ttl) {
                // Remove stale entry
                cache.pop(path);

                // Update metrics
                if self.config.enable_metrics {
                    let mut metrics = self.metrics.write().await;
                    metrics.l1_misses += 1;
                }

                return Ok(None);
            }

            // Update hit count
            entry.increment_hits();

            // Update metrics
            if self.config.enable_metrics {
                let mut metrics = self.metrics.write().await;
                metrics.l1_hits += 1;
            }

            return Ok(Some(entry.ontology.clone()));
        }

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.l1_misses += 1;
        }

        Ok(None)
    }

    /// Put into L1 cache
    async fn l1_put(&self, path: &PathBuf, ontology: AnnotatedOntology) -> Result<(), CacheError> {
        let file_modified = self.get_file_modified_time(path)?;
        let entry = CacheEntry::new(ontology, file_modified);

        let mut cache = self.l1_cache.write().await;

        // Check memory limit before inserting
        if self.config.enable_metrics {
            let metrics = self.metrics.read().await;
            if metrics.total_memory_bytes + entry.size_bytes > self.config.l1_max_memory_bytes {
                // Evict entries until we have space
                self.evict_until_space(&mut cache, entry.size_bytes).await?;
            }
        }

        cache.put(path.clone(), entry);

        Ok(())
    }

    /// Get from L2 cache (Redis)
    #[cfg(feature = "redis")]
    async fn l2_get(&self, path: &PathBuf) -> Result<Option<AnnotatedOntology>, CacheError> {
        if let Some(client) = &self.l2_cache {
            let key = format!("ontology:{}", path.display());

            let mut conn = client
                .get_async_connection()
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;

            let serialized: Option<Vec<u8>> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;

            if let Some(data) = serialized {
                let ontology: AnnotatedOntology = bincode::deserialize(&data)
                    .map_err(|e| CacheError::SerializationError(e.to_string()))?;

                // Update metrics
                if self.config.enable_metrics {
                    let mut metrics = self.metrics.write().await;
                    metrics.l2_hits += 1;
                }

                return Ok(Some(ontology));
            }
        }

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.l2_misses += 1;
        }

        Ok(None)
    }

    /// Put into L2 cache (Redis)
    #[cfg(feature = "redis")]
    async fn l2_put(&self, path: &PathBuf, ontology: &AnnotatedOntology) -> Result<(), CacheError> {
        if let Some(client) = &self.l2_cache {
            let key = format!("ontology:{}", path.display());

            let serialized = bincode::serialize(ontology)
                .map_err(|e| CacheError::SerializationError(e.to_string()))?;

            let mut conn = client
                .get_async_connection()
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;

            redis::cmd("SETEX")
                .arg(&key)
                .arg(self.config.l2_ttl_seconds)
                .arg(serialized)
                .query_async(&mut conn)
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;
        }

        Ok(())
    }

    /// Invalidate cache entry
    pub async fn invalidate(&self, path: &Path) -> Result<(), CacheError> {
        let path_buf = path.to_path_buf();

        // Remove from L1
        let mut cache = self.l1_cache.write().await;
        cache.pop(&path_buf);

        // Remove from L2 (Redis)
        #[cfg(feature = "redis")]
        if let Some(client) = &self.l2_cache {
            let key = format!("ontology:{}", path.display());
            let mut conn = client
                .get_async_connection()
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;

            redis::cmd("DEL")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;
        }

        Ok(())
    }

    /// Clear all caches
    pub async fn clear_all(&self) -> Result<(), CacheError> {
        // Clear L1
        let mut cache = self.l1_cache.write().await;
        cache.clear();

        // Clear L2 (Redis) - flush all ontology keys
        #[cfg(feature = "redis")]
        if let Some(client) = &self.l2_cache {
            let mut conn = client
                .get_async_connection()
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;

            // Delete all keys matching pattern
            let keys: Vec<String> = redis::cmd("KEYS")
                .arg("ontology:*")
                .query_async(&mut conn)
                .await
                .map_err(|e| CacheError::RedisError(e.to_string()))?;

            for key in keys {
                redis::cmd("DEL")
                    .arg(&key)
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| CacheError::RedisError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Get cache metrics
    pub async fn get_metrics(&self) -> CacheMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = CacheMetrics::default();
    }

    // Helper methods

    fn get_file_modified_time(&self, path: &Path) -> Result<SystemTime, CacheError> {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .map_err(|e| CacheError::FileSystemError(e.to_string()))
    }

    fn get_ttl(&self) -> Duration {
        match self.config.invalidation_strategy {
            CacheInvalidationStrategy::TTL(duration) => duration,
            CacheInvalidationStrategy::Hybrid(duration) => duration,
            _ => Duration::from_secs(3600), // Default 1 hour
        }
    }

    async fn evict_until_space(
        &self,
        cache: &mut LruCache<PathBuf, CacheEntry>,
        required_bytes: usize,
    ) -> Result<(), CacheError> {
        let mut freed_bytes = 0;

        while freed_bytes < required_bytes && !cache.is_empty() {
            if let Some((_, entry)) = cache.pop_lru() {
                freed_bytes += entry.size_bytes;

                // Update metrics
                if self.config.enable_metrics {
                    let mut metrics = self.metrics.write().await;
                    metrics.total_evictions += 1;
                    metrics.total_memory_bytes -= entry.size_bytes;
                }
            } else {
                break;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_l1_cache() {
        let config = CacheManagerConfig::default();
        let manager = OntologyCacheManager::new(config).unwrap();

        // Test cache miss
        let path = PathBuf::from("/tmp/test.owl");
        let result = manager.l1_get(&path).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test metrics
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.l1_misses, 1);
        assert_eq!(metrics.l1_hits, 0);
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let config = CacheManagerConfig::default();
        let manager = OntologyCacheManager::new(config).unwrap();

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.l1_hit_rate(), 0.0);

        // Reset metrics
        manager.reset_metrics().await;
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.l1_hits, 0);
    }
}
