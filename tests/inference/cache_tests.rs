// tests/inference/cache_tests.rs
//! Cache System Tests

#[cfg(test)]
mod tests {
    use webxr::inference::cache::{InferenceCache, CacheConfig};
    use webxr::ports::ontology_repository::InferenceResults;
    use chrono::Utc;

    fn create_test_results() -> InferenceResults {
        InferenceResults {
            timestamp: Utc::now(),
            inferred_axioms: Vec::new(),
            inference_time_ms: 100,
            reasoner_version: "test-1.0".to_string(),
        }
    }

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let cache = InferenceCache::default();
        let results = create_test_results();

        // Put
        cache.put("ont1".to_string(), "checksum1".to_string(), results.clone()).await;

        // Get
        let retrieved = cache.get("ont1", "checksum1").await;
        assert!(retrieved.is_some());

        // Invalidate
        cache.invalidate("ont1").await;
        let after_invalidate = cache.get("ont1", "checksum1").await;
        assert!(after_invalidate.is_none());
    }

    #[tokio::test]
    async fn test_cache_checksum_validation() {
        let cache = InferenceCache::default();
        let results = create_test_results();

        cache.put("ont1".to_string(), "checksum1".to_string(), results.clone()).await;

        // Wrong checksum should invalidate
        let retrieved = cache.get("ont1", "checksum2").await;
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let config = CacheConfig {
            max_entries: 3,
            ..Default::default()
        };

        let cache = InferenceCache::new(config);
        let results = create_test_results();

        // Fill cache
        cache.put("ont1".to_string(), "cs1".to_string(), results.clone()).await;
        cache.put("ont2".to_string(), "cs2".to_string(), results.clone()).await;
        cache.put("ont3".to_string(), "cs3".to_string(), results.clone()).await;

        // Add one more (should evict ont1)
        cache.put("ont4".to_string(), "cs4".to_string(), results.clone()).await;

        let stats = cache.get_statistics().await;
        assert_eq!(stats.current_size, 3);
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let cache = InferenceCache::default();
        let results = create_test_results();

        cache.put("ont1".to_string(), "cs1".to_string(), results.clone()).await;

        // Hit
        cache.get("ont1", "cs1").await;

        // Miss
        cache.get("ont2", "cs2").await;

        let stats = cache.get_statistics().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = InferenceCache::default();
        let results = create_test_results();

        cache.put("ont1".to_string(), "cs1".to_string(), results.clone()).await;
        cache.put("ont2".to_string(), "cs2".to_string(), results.clone()).await;

        cache.clear().await;

        let stats = cache.get_statistics().await;
        assert_eq!(stats.current_size, 0);
    }
}
