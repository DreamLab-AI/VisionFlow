//! Cache Layer for Ontology Data
//!
//! Provides in-memory caching with TTL, LRU eviction, and cache invalidation
//! for ontology queries, entities, and graph visualization data.

use chrono::{DateTime, Utc};
use log::{debug, info, warn};
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::QueryResult;

/// Cache entry with TTL
#[derive(Clone, Debug)]
struct CacheEntry<T> {
    data: T,
    inserted_at: Instant,
    ttl: Duration,
    access_count: u64,
}

impl<T: Clone> CacheEntry<T> {
    fn new(data: T, ttl: Duration) -> Self {
        Self {
            data,
            inserted_at: Instant::now(),
            ttl,
            access_count: 0,
        }
    }

    fn is_expired(&self) -> bool {
        self.inserted_at.elapsed() > self.ttl
    }

    fn get(&mut self) -> Option<T> {
        if self.is_expired() {
            None
        } else {
            self.access_count += 1;
            Some(self.data.clone())
        }
    }
}

/// Cache statistics
#[derive(Clone, Debug)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f32,
}

/// Ontology cache with LRU eviction and TTL
pub struct OntologyCache {
    query_cache: Arc<Mutex<LruCache<String, CacheEntry<QueryResult>>>>,
    entity_cache: Arc<Mutex<LruCache<String, CacheEntry<String>>>>, // Stores JSON strings
    stats: Arc<Mutex<CacheStats>>,
    default_ttl: Duration,
}

impl OntologyCache {
    /// Create new cache with default configuration
    pub fn new() -> Self {
        Self::with_capacity(1000, Duration::from_secs(3600))
    }

    /// Create cache with specified capacity and TTL
    pub fn with_capacity(capacity: usize, default_ttl: Duration) -> Self {
        let cache_size = NonZeroUsize::new(capacity).unwrap();

        Self {
            query_cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
            entity_cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
            stats: Arc::new(Mutex::new(CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                size: 0,
                capacity,
                hit_rate: 0.0,
            })),
            default_ttl,
        }
    }

    /// Get query result from cache
    pub fn get_query_result(&self, key: &str) -> Option<QueryResult> {
        let mut cache = self.query_cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(entry) = cache.get_mut(key) {
            if let Some(result) = entry.get() {
                stats.hits += 1;
                stats.hit_rate = stats.hits as f32 / (stats.hits + stats.misses) as f32;
                debug!("Cache hit for query: {}", key);
                return Some(result);
            } else {
                // Entry expired, remove it
                cache.pop(key);
                stats.evictions += 1;
            }
        }

        stats.misses += 1;
        stats.hit_rate = stats.hits as f32 / (stats.hits + stats.misses) as f32;
        debug!("Cache miss for query: {}", key);
        None
    }

    /// Set query result in cache
    pub fn set_query_result(&self, key: &str, result: &QueryResult) {
        let mut cache = self.query_cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let entry = CacheEntry::new(result.clone(), self.default_ttl);

        // Check if we're evicting an entry
        if cache.len() >= cache.cap().get() {
            stats.evictions += 1;
        }

        cache.put(key.to_string(), entry);
        stats.size = cache.len();

        debug!("Cached query result: {}", key);
    }

    /// Get entity from cache
    pub fn get_entity(&self, entity_id: &str) -> Option<String> {
        let mut cache = self.entity_cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(entry) = cache.get_mut(entity_id) {
            if let Some(data) = entry.get() {
                stats.hits += 1;
                stats.hit_rate = stats.hits as f32 / (stats.hits + stats.misses) as f32;
                debug!("Cache hit for entity: {}", entity_id);
                return Some(data);
            } else {
                // Entry expired, remove it
                cache.pop(entity_id);
                stats.evictions += 1;
            }
        }

        stats.misses += 1;
        stats.hit_rate = stats.hits as f32 / (stats.hits + stats.misses) as f32;
        debug!("Cache miss for entity: {}", entity_id);
        None
    }

    /// Set entity in cache
    pub fn set_entity(&self, entity_id: &str, data: &str) {
        let mut cache = self.entity_cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let entry = CacheEntry::new(data.to_string(), self.default_ttl);

        // Check if we're evicting an entry
        if cache.len() >= cache.cap().get() {
            stats.evictions += 1;
        }

        cache.put(entity_id.to_string(), entry);
        stats.size = cache.len();

        debug!("Cached entity: {}", entity_id);
    }

    /// Invalidate specific cache key
    pub fn invalidate(&self, key: &str) {
        let mut query_cache = self.query_cache.lock().unwrap();
        let mut entity_cache = self.entity_cache.lock().unwrap();

        query_cache.pop(key);
        entity_cache.pop(key);

        info!("Invalidated cache key: {}", key);
    }

    /// Invalidate all entries matching a pattern
    pub fn invalidate_pattern(&self, pattern: &str) {
        let mut query_cache = self.query_cache.lock().unwrap();
        let mut entity_cache = self.entity_cache.lock().unwrap();

        let query_keys: Vec<String> = query_cache
            .iter()
            .filter(|(k, _)| k.contains(pattern))
            .map(|(k, _)| k.clone())
            .collect();

        let entity_keys: Vec<String> = entity_cache
            .iter()
            .filter(|(k, _)| k.contains(pattern))
            .map(|(k, _)| k.clone())
            .collect();

        let total_invalidated = query_keys.len() + entity_keys.len();

        for key in query_keys {
            query_cache.pop(&key);
        }

        for key in entity_keys {
            entity_cache.pop(&key);
        }

        info!("Invalidated {} cache entries matching pattern: {}",
            total_invalidated, pattern);
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut query_cache = self.query_cache.lock().unwrap();
        let mut entity_cache = self.entity_cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let total_cleared = query_cache.len() + entity_cache.len();

        query_cache.clear();
        entity_cache.clear();

        stats.size = 0;
        stats.evictions += total_cleared as u64;

        info!("Cleared {} cache entries", total_cleared);
    }

    /// Remove expired entries
    pub fn evict_expired(&self) {
        let mut query_cache = self.query_cache.lock().unwrap();
        let mut entity_cache = self.entity_cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let mut evicted = 0;

        // Collect expired query keys
        let expired_query_keys: Vec<String> = query_cache
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        // Collect expired entity keys
        let expired_entity_keys: Vec<String> = entity_cache
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        // Remove expired entries
        for key in expired_query_keys {
            query_cache.pop(&key);
            evicted += 1;
        }

        for key in expired_entity_keys {
            entity_cache.pop(&key);
            evicted += 1;
        }

        stats.evictions += evicted;
        stats.size = query_cache.len() + entity_cache.len();

        if evicted > 0 {
            debug!("Evicted {} expired cache entries", evicted);
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.hits = 0;
        stats.misses = 0;
        stats.evictions = 0;
        stats.hit_rate = 0.0;
        info!("Reset cache statistics");
    }
}

impl Default for OntologyCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Background task to periodically evict expired entries
pub fn start_cache_eviction_task(cache: Arc<OntologyCache>) {
    use std::thread;

    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(300)); // Every 5 minutes
            cache.evict_expired();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn test_cache_basic_operations() {
        let cache = OntologyCache::new();

        let query_result = QueryResult {
            query_id: Uuid::new_v4().to_string(),
            results: vec![],
            total_count: 0,
            execution_time_ms: 100,
            execution_plan: None,
            from_cache: false,
            timestamp: Utc::now(),
        };

        // Set and get
        cache.set_query_result("test_query", &query_result);
        let retrieved = cache.get_query_result("test_query");
        assert!(retrieved.is_some());

        // Cache miss
        let missing = cache.get_query_result("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_cache_expiration() {
        let cache = OntologyCache::with_capacity(100, Duration::from_millis(100));

        let query_result = QueryResult {
            query_id: Uuid::new_v4().to_string(),
            results: vec![],
            total_count: 0,
            execution_time_ms: 100,
            execution_plan: None,
            from_cache: false,
            timestamp: Utc::now(),
        };

        cache.set_query_result("expiring_query", &query_result);

        // Should exist immediately
        assert!(cache.get_query_result("expiring_query").is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should be expired
        assert!(cache.get_query_result("expiring_query").is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = OntologyCache::new();

        let query_result = QueryResult {
            query_id: Uuid::new_v4().to_string(),
            results: vec![],
            total_count: 0,
            execution_time_ms: 100,
            execution_plan: None,
            from_cache: false,
            timestamp: Utc::now(),
        };

        cache.set_query_result("test", &query_result);

        // Hit
        cache.get_query_result("test");

        // Miss
        cache.get_query_result("nonexistent");

        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = OntologyCache::new();

        let query_result = QueryResult {
            query_id: Uuid::new_v4().to_string(),
            results: vec![],
            total_count: 0,
            execution_time_ms: 100,
            execution_plan: None,
            from_cache: false,
            timestamp: Utc::now(),
        };

        cache.set_query_result("test", &query_result);
        assert!(cache.get_query_result("test").is_some());

        cache.invalidate("test");
        assert!(cache.get_query_result("test").is_none());
    }
}
