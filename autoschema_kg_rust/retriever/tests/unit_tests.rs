use retriever::*;
use retriever::prelude::*;
use tokio_test;
use rstest::*;
use test_case::test_case;
use proptest::prelude::*;
use std::collections::HashMap;

mod common;
use common::*;

#[cfg(test)]
mod vector_index_tests {
    use super::*;

    #[test]
    fn test_vector_index_creation() {
        let config = VectorIndexConfig::new(768, "hnsw");
        let index = VectorIndex::new(config);
        assert_eq!(index.dimensions(), 768);
        assert_eq!(index.index_type(), "hnsw");
    }

    #[tokio::test]
    async fn test_vector_index_add_and_search() {
        let config = VectorIndexConfig::new(4, "hnsw");
        let mut index = VectorIndex::new(config);

        // Add test vectors
        let vectors = vec![
            (vec![1.0, 0.0, 0.0, 0.0], "doc1"),
            (vec![0.0, 1.0, 0.0, 0.0], "doc2"),
            (vec![0.0, 0.0, 1.0, 0.0], "doc3"),
        ];

        for (vector, id) in vectors {
            index.add_vector(id, &vector).await.unwrap();
        }

        // Search for similar vectors
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let results = index.search(&query, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1"); // Should be most similar
        assert!(results[0].score > 0.8);
    }

    #[tokio::test]
    async fn test_vector_index_batch_operations() {
        let config = VectorIndexConfig::new(3, "hnsw");
        let mut index = VectorIndex::new(config);

        let batch_vectors: Vec<(String, Vec<f32>)> = (0..100)
            .map(|i| (format!("doc_{}", i), generate_random_vector(3)))
            .collect();

        let result = index.add_batch(batch_vectors).await;
        assert!(result.is_ok());

        let query = generate_random_vector(3);
        let results = index.search(&query, 10).await.unwrap();
        assert_eq!(results.len(), 10);
    }

    fn generate_random_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rand::random::<f32>()).collect()
    }
}

#[cfg(test)]
mod embedding_model_tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_model_creation() {
        let model = EmbeddingModel::new("sentence-transformers/all-MiniLM-L6-v2").await;
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.model_name(), "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[tokio::test]
    async fn test_embedding_generation() {
        let model = EmbeddingModel::new("sentence-transformers/all-MiniLM-L6-v2").await.unwrap();
        let text = "This is a test sentence for embedding generation.";

        let embedding = model.encode(text).await.unwrap();
        assert!(embedding.len() > 0);
        assert_eq!(embedding.len(), model.dimensions());
    }

    #[tokio::test]
    async fn test_batch_embedding_generation() {
        let model = EmbeddingModel::new("sentence-transformers/all-MiniLM-L6-v2").await.unwrap();
        let texts = vec![
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ];

        let embeddings = model.encode_batch(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), model.dimensions());
        }
    }

    #[test]
    fn test_embedding_similarity() {
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.9, 0.1, 0.0];
        let embedding3 = vec![0.0, 1.0, 0.0];

        let sim1 = EmbeddingModel::cosine_similarity(&embedding1, &embedding2);
        let sim2 = EmbeddingModel::cosine_similarity(&embedding1, &embedding3);

        assert!(sim1 > sim2); // embedding1 should be more similar to embedding2
        assert!(sim1 > 0.8);
        assert!(sim2 < 0.2);
    }
}

#[cfg(test)]
mod query_processor_tests {
    use super::*;

    #[test]
    fn test_query_processor_creation() {
        let config = QueryProcessorConfig::default();
        let processor = QueryProcessor::new(config);
        assert!(!processor.is_empty());
    }

    #[tokio::test]
    async fn test_query_expansion() {
        let config = QueryProcessorConfig::default();
        let processor = QueryProcessor::new(config);

        let original_query = "machine learning algorithms";
        let expanded = processor.expand_query(original_query).await.unwrap();

        assert!(expanded.len() >= original_query.len());
        assert!(expanded.contains("machine learning"));
    }

    #[tokio::test]
    async fn test_query_rewriting() {
        let config = QueryProcessorConfig::default();
        let processor = QueryProcessor::new(config);

        let queries = vec![
            "What is ML?",
            "Tell me about artificial intelligence",
            "How does deep learning work?",
        ];

        for query in queries {
            let rewritten = processor.rewrite_query(query).await.unwrap();
            assert!(!rewritten.is_empty());
            assert!(rewritten.len() >= query.len() * 0.5); // Should be reasonably substantial
        }
    }

    #[test]
    fn test_query_preprocessing() {
        let config = QueryProcessorConfig::default();
        let processor = QueryProcessor::new(config);

        let noisy_query = "  What IS  machine-learning??? ";
        let cleaned = processor.preprocess_query(noisy_query);

        assert_eq!(cleaned, "what is machine learning");
        assert!(!cleaned.contains("?"));
        assert!(!cleaned.starts_with(" "));
        assert!(!cleaned.ends_with(" "));
    }
}

#[cfg(test)]
mod graph_traverser_tests {
    use super::*;

    #[test]
    fn test_graph_traverser_creation() {
        let config = TraversalConfig::new(3, TraversalStrategy::BreadthFirst);
        let traverser = GraphTraverser::new(config);
        assert_eq!(traverser.max_hops(), 3);
    }

    #[tokio::test]
    async fn test_single_hop_traversal() {
        let config = TraversalConfig::new(1, TraversalStrategy::BreadthFirst);
        let mut traverser = GraphTraverser::new(config);

        // Setup mock graph
        let graph = create_test_graph();
        traverser.set_graph(graph);

        let results = traverser.traverse_from("node1").await.unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.hop_count <= 1));
    }

    #[tokio::test]
    async fn test_multi_hop_traversal() {
        let config = TraversalConfig::new(3, TraversalStrategy::BreadthFirst);
        let mut traverser = GraphTraverser::new(config);

        let graph = create_test_graph();
        traverser.set_graph(graph);

        let results = traverser.traverse_from("node1").await.unwrap();

        // Should find nodes at different hop distances
        let hop_counts: Vec<_> = results.iter().map(|r| r.hop_count).collect();
        assert!(hop_counts.iter().any(|&h| h == 1));
        assert!(hop_counts.iter().any(|&h| h == 2));
        assert!(hop_counts.iter().all(|&h| h <= 3));
    }

    #[test_case(TraversalStrategy::BreadthFirst; "breadth first strategy")]
    #[test_case(TraversalStrategy::DepthFirst; "depth first strategy")]
    #[test_case(TraversalStrategy::BestFirst; "best first strategy")]
    fn test_traversal_strategies(strategy: TraversalStrategy) {
        let config = TraversalConfig::new(2, strategy);
        let traverser = GraphTraverser::new(config);
        assert_eq!(traverser.strategy(), strategy);
    }

    fn create_test_graph() -> TestGraph {
        let mut graph = TestGraph::new();

        // Add nodes
        for i in 1..=5 {
            graph.add_node(&format!("node{}", i));
        }

        // Add edges: 1->2, 2->3, 3->4, 2->5
        graph.add_edge("node1", "node2", 1.0);
        graph.add_edge("node2", "node3", 1.0);
        graph.add_edge("node3", "node4", 1.0);
        graph.add_edge("node2", "node5", 1.0);

        graph
    }
}

#[cfg(test)]
mod ranking_tests {
    use super::*;

    #[test]
    fn test_ranking_model_creation() {
        let config = RankingConfig::default();
        let model = RankingModel::new(config);
        assert!(model.is_initialized());
    }

    #[tokio::test]
    async fn test_document_scoring() {
        let config = RankingConfig::default();
        let model = RankingModel::new(config);

        let documents = vec![
            SearchResult::new("doc1", "machine learning algorithms", 0.9),
            SearchResult::new("doc2", "deep learning networks", 0.8),
            SearchResult::new("doc3", "artificial intelligence", 0.7),
        ];

        let query = "machine learning";
        let scored = model.rank_documents(&documents, query).await.unwrap();

        assert_eq!(scored.len(), 3);
        // Should be sorted by relevance
        assert!(scored[0].score >= scored[1].score);
        assert!(scored[1].score >= scored[2].score);
    }

    #[test_case(RankingStrategy::TfIdf; "tf-idf ranking")]
    #[test_case(RankingStrategy::BM25; "bm25 ranking")]
    #[test_case(RankingStrategy::Semantic; "semantic ranking")]
    #[test_case(RankingStrategy::Hybrid; "hybrid ranking")]
    fn test_ranking_strategies(strategy: RankingStrategy) {
        let config = RankingConfig::new(strategy);
        let model = RankingModel::new(config);
        assert_eq!(model.strategy(), strategy);
    }

    #[tokio::test]
    async fn test_reranking() {
        let config = RankingConfig::default();
        let model = RankingModel::new(config);

        let mut documents = vec![
            ScoredDocument::new("doc1", "content1", 0.5),
            ScoredDocument::new("doc2", "content2", 0.8),
            ScoredDocument::new("doc3", "content3", 0.6),
        ];

        let query = "test query";
        model.rerank(&mut documents, query).await.unwrap();

        // Should be reordered by relevance
        assert!(documents[0].score >= documents[1].score);
        assert!(documents[1].score >= documents[2].score);
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_cache_creation() {
        let config = CacheConfig::new(100, Duration::from_secs(300));
        let cache = CacheManager::new(config).await;
        assert!(cache.is_ok());
    }

    #[tokio::test]
    async fn test_cache_set_and_get() {
        let config = CacheConfig::new(100, Duration::from_secs(300));
        let mut cache = CacheManager::new(config).await.unwrap();

        let key = "test_key";
        let value = vec![1.0, 2.0, 3.0];

        cache.set(key, value.clone()).await.unwrap();
        let retrieved = cache.get(key).await.unwrap();

        assert_eq!(retrieved, Some(value));
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let config = CacheConfig::new(100, Duration::from_millis(50));
        let mut cache = CacheManager::new(config).await.unwrap();

        let key = "expiring_key";
        let value = vec![1.0, 2.0, 3.0];

        cache.set(key, value).await.unwrap();

        // Should exist immediately
        assert!(cache.get(key).await.unwrap().is_some());

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should be expired
        assert!(cache.get(key).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig::new(100, Duration::from_secs(300));
        let mut cache = CacheManager::new(config).await.unwrap();

        // Add some entries
        for i in 0..5 {
            cache.set(&format!("key_{}", i), vec![i as f32]).await.unwrap();
        }

        // Access some entries (hits)
        for i in 0..3 {
            cache.get(&format!("key_{}", i)).await.unwrap();
        }

        // Access non-existent entries (misses)
        for i in 10..12 {
            cache.get(&format!("key_{}", i)).await.unwrap();
        }

        let stats = cache.stats().await;
        assert_eq!(stats.entries, 5);
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 2);
        assert!(stats.hit_rate > 0.5);
    }
}

#[cfg(test)]
mod context_window_tests {
    use super::*;

    #[test]
    fn test_context_window_creation() {
        let config = ContextConfig::new(4096, 0.8);
        let window = ContextWindow::new(config);
        assert_eq!(window.max_tokens(), 4096);
        assert_eq!(window.overlap_ratio(), 0.8);
    }

    #[test]
    fn test_context_window_add_content() {
        let config = ContextConfig::new(100, 0.5);
        let mut window = ContextWindow::new(config);

        let content = "This is a test content that should fit in the window.";
        let result = window.add_content(content);
        assert!(result.is_ok());
        assert_eq!(window.content_count(), 1);
    }

    #[test]
    fn test_context_window_overflow() {
        let config = ContextConfig::new(50, 0.5); // Small window
        let mut window = ContextWindow::new(config);

        // Add content that exceeds window size
        let large_content = "This is a very long content that should definitely exceed the window size limit and trigger overflow handling.";
        let result = window.add_content(large_content);

        // Should handle overflow gracefully
        assert!(result.is_ok());
        assert!(window.total_tokens() <= window.max_tokens());
    }

    #[test]
    fn test_context_window_merge() {
        let config = ContextConfig::new(200, 0.5);
        let mut window1 = ContextWindow::new(config.clone());
        let mut window2 = ContextWindow::new(config);

        window1.add_content("First window content").unwrap();
        window2.add_content("Second window content").unwrap();

        let merged = window1.merge_with(&window2).unwrap();
        assert!(merged.content_count() >= 2);
    }
}

// Mock and helper types for testing
struct TestGraph {
    nodes: HashMap<String, TestNode>,
    edges: Vec<TestEdge>,
}

struct TestNode {
    id: String,
}

struct TestEdge {
    from: String,
    to: String,
    weight: f32,
}

impl TestGraph {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    fn add_node(&mut self, id: &str) {
        self.nodes.insert(id.to_string(), TestNode { id: id.to_string() });
    }

    fn add_edge(&mut self, from: &str, to: &str, weight: f32) {
        self.edges.push(TestEdge {
            from: from.to_string(),
            to: to.to_string(),
            weight,
        });
    }
}

// Additional property-based tests
proptest! {
    #[test]
    fn test_vector_similarity_properties(
        v1 in prop::collection::vec(-1.0f32..1.0, 10),
        v2 in prop::collection::vec(-1.0f32..1.0, 10)
    ) {
        let sim1 = EmbeddingModel::cosine_similarity(&v1, &v2);
        let sim2 = EmbeddingModel::cosine_similarity(&v2, &v1);

        // Cosine similarity should be symmetric
        prop_assert!((sim1 - sim2).abs() < 1e-6);

        // Should be between -1 and 1
        prop_assert!(sim1 >= -1.0 && sim1 <= 1.0);
    }

    #[test]
    fn test_query_preprocessing_properties(
        query in r"[a-zA-Z0-9\s\?\!\.\,]{1,100}"
    ) {
        let config = QueryProcessorConfig::default();
        let processor = QueryProcessor::new(config);
        let cleaned = processor.preprocess_query(&query);

        // Cleaned query should not be longer than original
        prop_assert!(cleaned.len() <= query.len());

        // Should be lowercase
        prop_assert!(cleaned == cleaned.to_lowercase());

        // Should not contain multiple consecutive spaces
        prop_assert!(!cleaned.contains("  "));
    }
}