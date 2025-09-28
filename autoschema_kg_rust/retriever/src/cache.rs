//! High-performance caching layer for retrieval optimization
//!
//! Multi-level caching with intelligent eviction and warming strategies

use crate::error::{Result, RetrieverError};
use crate::config::{CacheConfig, EvictionPolicy, WarmupStrategy};
use crate::search::SearchResult;
use crate::query::ProcessedQuery;
use dashmap::DashMap;
use moka::future::Cache as MokaCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// Cached data
    pub data: T,

    /// Creation timestamp
    pub created_at: i64,

    /// Last access timestamp
    pub last_accessed: i64,

    /// Access count
    pub access_count: u64,

    /// Entry size in bytes (estimate)
    pub size_bytes: usize,

    /// Cache hit reason
    pub hit_reason: String,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total requests
    pub total_requests: u64,

    /// Cache hits
    pub hits: u64,

    /// Cache misses
    pub misses: u64,

    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,

    /// Total entries
    pub total_entries: u64,

    /// Memory usage in bytes
    pub memory_usage_bytes: u64,

    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,

    /// Cache performance by layer
    pub layer_stats: HashMap<String, LayerStats>,
}

/// Statistics for individual cache layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: u64,
    pub memory_bytes: u64,
    pub avg_size_bytes: u64,
}

/// Cache key for different types of cached data
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum CacheKey {
    Query(String),
    Embedding(String),
    SearchResults(String),
    GraphTraversal(String),
    ContextWindow(String),
    Custom(String),
}

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheKey::Query(q) => write!(f, "query:{}", q),
            CacheKey::Embedding(e) => write!(f, "embedding:{}", e),
            CacheKey::SearchResults(s) => write!(f, "search:{}", s),
            CacheKey::GraphTraversal(g) => write!(f, "graph:{}", g),
            CacheKey::ContextWindow(c) => write!(f, "context:{}", c),
            CacheKey::Custom(c) => write!(f, "custom:{}", c),
        }
    }
}

/// Multi-level cache manager
pub struct CacheManager {
    /// L1 Cache: Fast in-memory cache for frequent queries
    l1_cache: MokaCache<String, Vec<u8>>,

    /// L2 Cache: Larger capacity with different eviction policy
    l2_cache: Arc<DashMap<String, CacheEntry<Vec<u8>>>>,

    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,

    /// Configuration
    config: CacheConfig,

    /// Warmup manager
    warmup_manager: Arc<CacheWarmupManager>,
}

impl CacheManager {
    /// Create new cache manager
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let l1_cache = MokaCache::builder()
            .max_capacity(config.max_size as u64 / 2) // Half capacity for L1
            .time_to_live(config.ttl)
            .build();

        let l2_cache = Arc::new(DashMap::new());

        let stats = Arc::new(RwLock::new(CacheStats {
            total_requests: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            total_entries: 0,
            memory_usage_bytes: 0,
            avg_response_time_ms: 0.0,
            layer_stats: HashMap::new(),
        }));

        let warmup_manager = Arc::new(CacheWarmupManager::new(config.warming.clone()));

        let manager = Self {
            l1_cache,
            l2_cache,
            stats,
            config,
            warmup_manager,
        };

        // Start warmup if configured
        if config.warming.enabled && config.warming.on_startup {
            manager.warmup_manager.start_warmup().await?;
        }

        Ok(manager)
    }

    /// Get value from cache
    pub async fn get<T>(&self, key: &CacheKey) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let start_time = Instant::now();
        let key_str = key.to_string();

        // Update request stats
        self.update_request_stats().await;

        // Try L1 cache first
        if let Some(data) = self.l1_cache.get(&key_str).await {
            self.update_hit_stats("L1", start_time).await;
            let deserialized: T = bincode::deserialize(&data)
                .map_err(|e| RetrieverError::cache(format!("Deserialization failed: {}", e)))?;
            return Ok(Some(deserialized));
        }

        // Try L2 cache
        if let Some(entry) = self.l2_cache.get(&key_str) {
            let mut entry = entry.clone();
            entry.last_accessed = chrono::Utc::now().timestamp();
            entry.access_count += 1;

            // Promote to L1 if frequently accessed
            if entry.access_count > 5 {
                self.l1_cache.insert(key_str.clone(), entry.data.clone()).await;
            }

            self.l2_cache.insert(key_str, entry.clone());
            self.update_hit_stats("L2", start_time).await;

            let deserialized: T = bincode::deserialize(&entry.data)
                .map_err(|e| RetrieverError::cache(format!("Deserialization failed: {}", e)))?;
            return Ok(Some(deserialized));
        }

        // Cache miss
        self.update_miss_stats(start_time).await;
        Ok(None)
    }

    /// Put value into cache
    pub async fn put<T>(&self, key: &CacheKey, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        let key_str = key.to_string();
        let serialized = bincode::serialize(value)
            .map_err(|e| RetrieverError::cache(format!("Serialization failed: {}", e)))?;

        let size_bytes = serialized.len();
        let now = chrono::Utc::now().timestamp();

        // Create cache entry
        let entry = CacheEntry {
            data: serialized.clone(),
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            hit_reason: "initial_insert".to_string(),
        };

        // Insert into both caches
        self.l1_cache.insert(key_str.clone(), serialized).await;
        self.l2_cache.insert(key_str, entry);

        // Update size stats
        self.update_size_stats(size_bytes as u64).await;

        // Trigger eviction if needed
        self.maybe_evict().await?;

        Ok(())
    }

    /// Remove value from cache
    pub async fn remove(&self, key: &CacheKey) -> Result<bool> {
        let key_str = key.to_string();

        self.l1_cache.invalidate(&key_str).await;
        let removed = self.l2_cache.remove(&key_str).is_some();

        Ok(removed)
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        self.l1_cache.invalidate_all();
        self.l2_cache.clear();

        // Reset stats
        let mut stats = self.stats.write().await;
        *stats = CacheStats::default();

        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();

        // Update current metrics
        stats.total_entries = self.l1_cache.entry_count() + self.l2_cache.len() as u64;
        stats.hit_rate = if stats.total_requests > 0 {
            stats.hits as f64 / stats.total_requests as f64
        } else {
            0.0
        };

        // Calculate L1 stats
        let l1_stats = LayerStats {
            hits: 0, // Would need separate tracking
            misses: 0,
            entries: self.l1_cache.entry_count(),
            memory_bytes: self.estimate_l1_memory(),
            avg_size_bytes: if self.l1_cache.entry_count() > 0 {
                self.estimate_l1_memory() / self.l1_cache.entry_count()
            } else {
                0
            },
        };

        // Calculate L2 stats
        let l2_total_size: usize = self.l2_cache.iter().map(|entry| entry.size_bytes).sum();
        let l2_stats = LayerStats {
            hits: 0, // Would need separate tracking
            misses: 0,
            entries: self.l2_cache.len() as u64,
            memory_bytes: l2_total_size as u64,
            avg_size_bytes: if self.l2_cache.len() > 0 {
                l2_total_size as u64 / self.l2_cache.len() as u64
            } else {
                0
            },
        };

        stats.layer_stats.insert("L1".to_string(), l1_stats);
        stats.layer_stats.insert("L2".to_string(), l2_stats);
        stats.memory_usage_bytes = l2_total_size as u64 + self.estimate_l1_memory();

        stats
    }

    /// Warm cache with popular queries
    pub async fn warm_cache(&self, queries: Vec<String>) -> Result<()> {
        self.warmup_manager.warm_with_queries(queries).await
    }

    /// Check if cache contains key
    pub async fn contains_key(&self, key: &CacheKey) -> bool {
        let key_str = key.to_string();
        self.l1_cache.contains_key(&key_str) || self.l2_cache.contains_key(&key_str)
    }

    /// Get cache size in bytes
    pub async fn size_bytes(&self) -> u64 {
        let l1_size = self.estimate_l1_memory();
        let l2_size: usize = self.l2_cache.iter().map(|entry| entry.size_bytes).sum();
        l1_size + l2_size as u64
    }

    /// Trigger cache eviction based on policy
    async fn maybe_evict(&self) -> Result<()> {
        let current_size = self.size_bytes().await;
        let max_size = self.config.max_size as u64 * 1024 * 1024; // Convert MB to bytes

        if current_size > max_size {
            self.evict_entries().await?;
        }

        Ok(())
    }

    async fn evict_entries(&self) -> Result<()> {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru().await,
            EvictionPolicy::LFU => self.evict_lfu().await,
            EvictionPolicy::FIFO => self.evict_fifo().await,
            EvictionPolicy::TTL => self.evict_expired().await,
            EvictionPolicy::Adaptive => self.evict_adaptive().await,
        }
    }

    async fn evict_lru(&self) -> Result<()> {
        // L1 cache handles its own LRU eviction
        // For L2, find and remove least recently used entries
        let mut entries: Vec<_> = self.l2_cache
            .iter()
            .map(|entry| (entry.key().clone(), entry.last_accessed))
            .collect();

        entries.sort_by_key(|(_, last_accessed)| *last_accessed);

        // Remove oldest 25% of entries
        let to_remove = entries.len() / 4;
        for (key, _) in entries.into_iter().take(to_remove) {
            self.l2_cache.remove(&key);
        }

        Ok(())
    }

    async fn evict_lfu(&self) -> Result<()> {
        let mut entries: Vec<_> = self.l2_cache
            .iter()
            .map(|entry| (entry.key().clone(), entry.access_count))
            .collect();

        entries.sort_by_key(|(_, access_count)| *access_count);

        let to_remove = entries.len() / 4;
        for (key, _) in entries.into_iter().take(to_remove) {
            self.l2_cache.remove(&key);
        }

        Ok(())
    }

    async fn evict_fifo(&self) -> Result<()> {
        let mut entries: Vec<_> = self.l2_cache
            .iter()
            .map(|entry| (entry.key().clone(), entry.created_at))
            .collect();

        entries.sort_by_key(|(_, created_at)| *created_at);

        let to_remove = entries.len() / 4;
        for (key, _) in entries.into_iter().take(to_remove) {
            self.l2_cache.remove(&key);
        }

        Ok(())
    }

    async fn evict_expired(&self) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        let ttl_seconds = self.config.ttl.as_secs() as i64;

        let expired_keys: Vec<_> = self.l2_cache
            .iter()
            .filter(|entry| now - entry.created_at > ttl_seconds)
            .map(|entry| entry.key().clone())
            .collect();

        for key in expired_keys {
            self.l2_cache.remove(&key);
        }

        Ok(())
    }

    async fn evict_adaptive(&self) -> Result<()> {
        // Combine LRU and LFU strategies
        let mut scored_entries: Vec<_> = self.l2_cache
            .iter()
            .map(|entry| {
                let age_score = chrono::Utc::now().timestamp() - entry.last_accessed;
                let frequency_score = 1.0 / (entry.access_count as f32 + 1.0);
                let combined_score = age_score as f32 + frequency_score;
                (entry.key().clone(), combined_score)
            })
            .collect();

        scored_entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_remove = scored_entries.len() / 4;
        for (key, _) in scored_entries.into_iter().take(to_remove) {
            self.l2_cache.remove(&key);
        }

        Ok(())
    }

    async fn update_request_stats(&self) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
    }

    async fn update_hit_stats(&self, layer: &str, start_time: Instant) {
        let mut stats = self.stats.write().await;
        stats.hits += 1;

        let response_time = start_time.elapsed().as_millis() as f64;
        stats.avg_response_time_ms = (stats.avg_response_time_ms * (stats.hits - 1) as f64 + response_time) / stats.hits as f64;

        let layer_stats = stats.layer_stats.entry(layer.to_string()).or_insert(LayerStats::default());
        layer_stats.hits += 1;
    }

    async fn update_miss_stats(&self, start_time: Instant) {
        let mut stats = self.stats.write().await;
        stats.misses += 1;

        let response_time = start_time.elapsed().as_millis() as f64;
        let total_responses = stats.hits + stats.misses;
        stats.avg_response_time_ms = (stats.avg_response_time_ms * (total_responses - 1) as f64 + response_time) / total_responses as f64;
    }

    async fn update_size_stats(&self, size_bytes: u64) {
        let mut stats = self.stats.write().await;
        stats.memory_usage_bytes += size_bytes;
    }

    fn estimate_l1_memory(&self) -> u64 {
        // Rough estimation based on entry count
        self.l1_cache.entry_count() * 1024 // Assume 1KB per entry average
    }
}

impl Default for LayerStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            entries: 0,
            memory_bytes: 0,
            avg_size_bytes: 0,
        }
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            total_entries: 0,
            memory_usage_bytes: 0,
            avg_response_time_ms: 0.0,
            layer_stats: HashMap::new(),
        }
    }
}

/// Cache warmup manager
pub struct CacheWarmupManager {
    config: crate::config::CacheWarmingConfig,
    popular_queries: Arc<RwLock<Vec<String>>>,
    recent_queries: Arc<RwLock<Vec<String>>>,
}

impl CacheWarmupManager {
    pub fn new(config: crate::config::CacheWarmingConfig) -> Self {
        Self {
            config,
            popular_queries: Arc::new(RwLock::new(Vec::new())),
            recent_queries: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start warmup process
    pub async fn start_warmup(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        for strategy in &self.config.strategies {
            match strategy {
                WarmupStrategy::PopularQueries => {
                    self.warm_popular_queries().await?;
                }
                WarmupStrategy::RecentQueries => {
                    self.warm_recent_queries().await?;
                }
                WarmupStrategy::PredictedQueries => {
                    self.warm_predicted_queries().await?;
                }
                WarmupStrategy::RandomSampling => {
                    self.warm_random_sampling().await?;
                }
            }
        }

        Ok(())
    }

    /// Warm cache with specific queries
    pub async fn warm_with_queries(&self, queries: Vec<String>) -> Result<()> {
        // In a real implementation, this would execute queries and cache results
        let mut popular = self.popular_queries.write().await;
        popular.extend(queries);
        Ok(())
    }

    async fn warm_popular_queries(&self) -> Result<()> {
        // Load popular queries from analytics or logs
        let popular_queries = vec![
            "machine learning".to_string(),
            "artificial intelligence".to_string(),
            "deep learning".to_string(),
            "neural networks".to_string(),
        ];

        self.warm_with_queries(popular_queries).await
    }

    async fn warm_recent_queries(&self) -> Result<()> {
        let recent = self.recent_queries.read().await;
        let recent_queries = recent.clone();
        drop(recent);

        self.warm_with_queries(recent_queries).await
    }

    async fn warm_predicted_queries(&self) -> Result<()> {
        // Use ML models to predict likely queries
        let predicted_queries = vec![
            "computer vision".to_string(),
            "natural language processing".to_string(),
        ];

        self.warm_with_queries(predicted_queries).await
    }

    async fn warm_random_sampling(&self) -> Result<()> {
        // Sample from historical query patterns
        let sampled_queries = vec![
            "data science".to_string(),
            "python programming".to_string(),
        ];

        self.warm_with_queries(sampled_queries).await
    }

    /// Add query to recent list
    pub async fn track_query(&self, query: String) {
        let mut recent = self.recent_queries.write().await;
        recent.push(query);

        // Keep only last 1000 queries
        if recent.len() > 1000 {
            recent.truncate(1000);
        }
    }
}

/// Query cache specifically for processed queries
pub struct QueryCache {
    cache_manager: Arc<CacheManager>,
}

impl QueryCache {
    pub fn new(cache_manager: Arc<CacheManager>) -> Self {
        Self { cache_manager }
    }

    /// Get processed query from cache
    pub async fn get_processed_query(&self, query: &str) -> Result<Option<ProcessedQuery>> {
        let key = CacheKey::Query(query.to_string());
        self.cache_manager.get(&key).await
    }

    /// Cache processed query
    pub async fn cache_processed_query(&self, query: &str, processed: &ProcessedQuery) -> Result<()> {
        let key = CacheKey::Query(query.to_string());
        self.cache_manager.put(&key, processed).await
    }

    /// Get search results from cache
    pub async fn get_search_results(&self, query_hash: &str) -> Result<Option<Vec<SearchResult>>> {
        let key = CacheKey::SearchResults(query_hash.to_string());
        self.cache_manager.get(&key).await
    }

    /// Cache search results
    pub async fn cache_search_results(&self, query_hash: &str, results: &[SearchResult]) -> Result<()> {
        let key = CacheKey::SearchResults(query_hash.to_string());
        self.cache_manager.put(&key, results).await
    }
}

/// Embedding cache for vector lookups
pub struct EmbeddingCache {
    cache_manager: Arc<CacheManager>,
}

impl EmbeddingCache {
    pub fn new(cache_manager: Arc<CacheManager>) -> Self {
        Self { cache_manager }
    }

    /// Get embedding from cache
    pub async fn get_embedding(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let key = CacheKey::Embedding(text.to_string());
        self.cache_manager.get(&key).await
    }

    /// Cache embedding
    pub async fn cache_embedding(&self, text: &str, embedding: &[f32]) -> Result<()> {
        let key = CacheKey::Embedding(text.to_string());
        self.cache_manager.put(&key, embedding).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_manager_basic_operations() {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config).await.unwrap();

        let key = CacheKey::Query("test query".to_string());
        let value = vec![1, 2, 3, 4, 5];

        // Test put and get
        cache.put(&key, &value).await.unwrap();
        let retrieved: Option<Vec<i32>> = cache.get(&key).await.unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), value);

        // Test contains
        assert!(cache.contains_key(&key).await);

        // Test remove
        let removed = cache.remove(&key).await.unwrap();
        assert!(removed);
        assert!(!cache.contains_key(&key).await);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config).await.unwrap();

        let key1 = CacheKey::Query("query1".to_string());
        let key2 = CacheKey::Query("query2".to_string());

        // Test miss
        let _: Option<String> = cache.get(&key1).await.unwrap();

        // Test hit
        cache.put(&key1, &"value1".to_string()).await.unwrap();
        let _: Option<String> = cache.get(&key1).await.unwrap();

        let stats = cache.stats().await;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[tokio::test]
    async fn test_query_cache() {
        let config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(config).await.unwrap());
        let query_cache = QueryCache::new(cache_manager);

        let query = "test query";
        let processed = ProcessedQuery {
            original: query.to_string(),
            preprocessed: "preprocessed".to_string(),
            expanded: vec![],
            rewritten: vec![],
            features: crate::query::QueryFeatures {
                token_count: 2,
                char_count: 10,
                entities: vec![],
                key_phrases: vec![],
                query_type: crate::query::QueryType::Factual,
                intent: crate::query::QueryIntent::Information,
                complexity: 0.5,
                ambiguity: 0.3,
            },
            metadata: crate::query::QueryMetadata {
                timestamp: 0,
                processing_time_ms: 10,
                language: "en".to_string(),
                steps: vec![],
                warnings: vec![],
            },
        };

        // Cache and retrieve
        query_cache.cache_processed_query(query, &processed).await.unwrap();
        let retrieved = query_cache.get_processed_query(query).await.unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().original, query);
    }

    #[tokio::test]
    async fn test_embedding_cache() {
        let config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(config).await.unwrap());
        let embedding_cache = EmbeddingCache::new(cache_manager);

        let text = "test text";
        let embedding = vec![0.1, 0.2, 0.3, 0.4];

        // Cache and retrieve
        embedding_cache.cache_embedding(text, &embedding).await.unwrap();
        let retrieved = embedding_cache.get_embedding(text).await.unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embedding);
    }

    #[test]
    fn test_cache_key_display() {
        let key = CacheKey::Query("test query".to_string());
        assert_eq!(key.to_string(), "query:test query");

        let key = CacheKey::Embedding("test embedding".to_string());
        assert_eq!(key.to_string(), "embedding:test embedding");
    }
}
"