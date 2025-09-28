//! Main multi-hop retriever orchestrating all components
//!
//! High-performance RAG retriever with parallel processing and intelligent coordination

use crate::error::{Result, RetrieverError};
use crate::config::RetrieverConfig;
use crate::cache::{CacheManager, QueryCache, EmbeddingCache};
use crate::context::{ContextManager, ContextWindow};
use crate::embeddings::{EmbeddingModel, VectorIndex, EmbeddingProcessor, TransformerEmbeddingModel};
use crate::graph::{KnowledgeGraph, GraphTraverser, HopResult, TraversalStrategy};
use crate::metrics::{MetricsCollector, RetrievalMetrics};
use crate::query::{QueryProcessor, ProcessedQuery};
use crate::ranking::{RankingModel, ReRanker, ScoredDocument};
use crate::search::{SearchEngine, SearchResult, VectorSearchStrategy, KeywordSearchStrategy, GraphSearchStrategy, SemanticSearchStrategy, SearchStrategyTrait};
use crate::utils;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

/// Multi-hop retrieval result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Final context window
    pub context: ContextWindow,

    /// Search results for each hop
    pub hop_results: Vec<HopRetrievalResult>,

    /// Final ranked documents
    pub ranked_documents: Vec<ScoredDocument>,

    /// Retrieval metadata
    pub metadata: RetrievalMetadata,

    /// Performance metrics for this retrieval
    pub performance: RetrievalPerformance,
}

/// Results for a single hop in multi-hop retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopRetrievalResult {
    /// Hop number (0-based)
    pub hop: usize,

    /// Search results from this hop
    pub search_results: Vec<SearchResult>,

    /// Graph traversal results (if applicable)
    pub graph_results: Option<HopResult>,

    /// Cumulative relevance score
    pub cumulative_score: f32,

    /// Processing time for this hop
    pub processing_time_ms: u64,
}

/// Retrieval metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetadata {
    /// Query information
    pub query: ProcessedQuery,

    /// Timestamp
    pub timestamp: i64,

    /// Total processing time
    pub total_time_ms: u64,

    /// Strategy used
    pub strategy: String,

    /// Cache hits during retrieval
    pub cache_hits: u32,

    /// Number of documents processed
    pub documents_processed: usize,

    /// Warnings and issues
    pub warnings: Vec<String>,
}

/// Performance metrics for individual retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalPerformance {
    /// Time breakdown by component
    pub component_times: HashMap<String, u64>,

    /// Memory usage during retrieval
    pub memory_usage_mb: f64,

    /// Parallel efficiency (0.0 to 1.0)
    pub parallel_efficiency: f32,

    /// Quality score
    pub quality_score: f32,

    /// Throughput metrics
    pub documents_per_second: f32,
}

/// Configuration for multi-hop retrieval
#[derive(Debug, Clone)]
pub struct MultiHopConfig {
    /// Maximum number of hops
    pub max_hops: usize,

    /// Maximum documents per hop
    pub max_docs_per_hop: usize,

    /// Minimum relevance threshold
    pub min_relevance: f32,

    /// Enable bidirectional search
    pub bidirectional: bool,

    /// Hop decay factor
    pub hop_decay: f32,

    /// Early stopping threshold
    pub early_stop_threshold: f32,
}

impl Default for MultiHopConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            max_docs_per_hop: 50,
            min_relevance: 0.1,
            bidirectional: true,
            hop_decay: 0.8,
            early_stop_threshold: 0.95,
        }
    }
}

/// Main multi-hop retriever
pub struct MultiHopRetriever {
    /// Configuration
    config: RetrieverConfig,

    /// Query processor
    query_processor: Arc<QueryProcessor>,

    /// Search engine
    search_engine: SearchEngine,

    /// Ranking model
    ranking_model: RankingModel,

    /// Re-ranker
    reranker: ReRanker,

    /// Context manager
    context_manager: ContextManager,

    /// Knowledge graph
    knowledge_graph: Arc<KnowledgeGraph>,

    /// Graph traverser
    graph_traverser: GraphTraverser,

    /// Cache manager
    cache_manager: Arc<CacheManager>,

    /// Query cache
    query_cache: QueryCache,

    /// Embedding cache
    embedding_cache: EmbeddingCache,

    /// Metrics collector
    metrics_collector: Arc<MetricsCollector>,

    /// Concurrency limiter
    concurrency_limiter: Arc<Semaphore>,

    /// Multi-hop configuration
    multihop_config: MultiHopConfig,
}

impl MultiHopRetriever {
    /// Create new multi-hop retriever
    pub async fn new(config: RetrieverConfig) -> Result<Self> {
        // Initialize cache manager
        let cache_manager = Arc::new(CacheManager::new(config.cache.clone()).await?);
        let query_cache = QueryCache::new(cache_manager.clone());
        let embedding_cache = EmbeddingCache::new(cache_manager.clone());

        // Initialize embedding model
        let embedding_model = Arc::new(TransformerEmbeddingModel::new(config.vector.clone()).await?) as Arc<dyn EmbeddingModel>;

        // Initialize query processor
        let query_processor = Arc::new(QueryProcessor::new(config.query.clone(), Some(embedding_model.clone()))?);

        // Initialize vector index
        let vector_index = Arc::new(VectorIndex::new(config.vector.clone())?);

        // Initialize knowledge graph
        let knowledge_graph = Arc::new(KnowledgeGraph::new(config.graph.clone()));
        let graph_traverser = GraphTraverser::new(knowledge_graph.clone(), config.graph.clone());

        // Initialize search strategies
        let vector_strategy = VectorSearchStrategy::new(vector_index.clone(), embedding_model.clone(), 0.4);
        let keyword_strategy = KeywordSearchStrategy::new(0.3);
        let graph_strategy = GraphSearchStrategy::new(knowledge_graph.clone(), graph_traverser.clone(), 0.3);

        // Initialize search engine
        let mut search_engine = SearchEngine::new(config.search.clone(), query_processor.clone());
        search_engine.add_strategy(Box::new(vector_strategy));
        search_engine.add_strategy(Box::new(keyword_strategy));
        search_engine.add_strategy(Box::new(graph_strategy));

        // Initialize ranking and re-ranking
        let ranking_model = RankingModel::new(config.ranking.clone())?;
        let reranker = ReRanker::new(config.ranking.reranking.clone());

        // Initialize context manager
        let context_manager = ContextManager::new(config.context.clone())?;

        // Initialize metrics collector
        let metrics_collector = Arc::new(MetricsCollector::new(config.metrics.clone())?);

        // Create concurrency limiter
        let concurrency_limiter = Arc::new(Semaphore::new(config.performance.num_threads));

        // Multi-hop configuration
        let multihop_config = MultiHopConfig {
            max_hops: config.graph.max_hops,
            max_docs_per_hop: config.graph.max_nodes_per_hop,
            min_relevance: config.graph.min_relevance,
            bidirectional: config.graph.bidirectional,
            hop_decay: config.graph.hop_decay,
            early_stop_threshold: 0.95,
        };

        Ok(Self {
            config,
            query_processor,
            search_engine,
            ranking_model,
            reranker,
            context_manager,
            knowledge_graph,
            graph_traverser,
            cache_manager,
            query_cache,
            embedding_cache,
            metrics_collector,
            concurrency_limiter,
            multihop_config,
        })
    }

    /// Perform multi-hop retrieval
    pub async fn retrieve(&self, query: &str, max_results: usize) -> Result<RetrievalResult> {
        let start_time = Instant::now();
        let mut component_times = HashMap::new();
        let mut warnings = Vec::new();
        let mut cache_hits = 0;

        // Record active query
        self.metrics_collector.update_active_queries(1).await;

        // Check cache first
        let query_hash = utils::generate_hash(query);
        if let Some(cached_results) = self.query_cache.get_search_results(&query_hash).await? {
            cache_hits += 1;
            self.metrics_collector.record_cache_hit().await;

            // Convert cached results to retrieval result
            let context = self.context_manager.create_context(&[]).await?;
            let metadata = RetrievalMetadata {
                query: ProcessedQuery {
                    original: query.to_string(),
                    preprocessed: query.to_string(),
                    expanded: vec![],
                    rewritten: vec![],
                    features: Default::default(),
                    metadata: Default::default(),
                },
                timestamp: chrono::Utc::now().timestamp(),
                total_time_ms: start_time.elapsed().as_millis() as u64,
                strategy: "cached".to_string(),
                cache_hits,
                documents_processed: cached_results.len(),
                warnings,
            };

            return Ok(RetrievalResult {
                context,
                hop_results: vec![],
                ranked_documents: vec![],
                metadata,
                performance: RetrievalPerformance {
                    component_times,
                    memory_usage_mb: 0.0,
                    parallel_efficiency: 1.0,
                    quality_score: 0.8,
                    documents_per_second: cached_results.len() as f32 / 0.001,
                },
            });
        }

        self.metrics_collector.record_cache_miss().await;

        // Process query
        let query_start = Instant::now();
        let processed_query = self.process_query_with_cache(query).await?;
        component_times.insert("query_processing".to_string(), query_start.elapsed().as_millis() as u64);
        self.metrics_collector.record_query_latency(query_start.elapsed()).await;

        // Perform multi-hop retrieval
        let retrieval_start = Instant::now();
        let hop_results = self.multi_hop_search(&processed_query, max_results).await?;
        component_times.insert("multi_hop_search".to_string(), retrieval_start.elapsed().as_millis() as u64);

        // Collect all documents from all hops
        let all_documents: Vec<SearchResult> = hop_results.iter()
            .flat_map(|hop| hop.search_results.clone())
            .collect();

        // Rank documents
        let ranking_start = Instant::now();
        let ranked_documents = self.rank_documents(all_documents, &processed_query).await?;
        component_times.insert("ranking".to_string(), ranking_start.elapsed().as_millis() as u64);
        self.metrics_collector.record_ranking_latency(ranking_start.elapsed()).await;

        // Create context
        let context_start = Instant::now();
        let context = self.context_manager.create_context(&ranked_documents).await?;
        component_times.insert("context_creation".to_string(), context_start.elapsed().as_millis() as u64);

        // Calculate performance metrics
        let total_time = start_time.elapsed();
        let performance = self.calculate_performance(&component_times, total_time, &ranked_documents).await;

        // Cache results
        let search_results: Vec<_> = ranked_documents.iter().map(|sd| sd.result.clone()).collect();
        self.query_cache.cache_search_results(&query_hash, &search_results).await?;

        // Update metrics
        self.metrics_collector.record_query_latency(total_time).await;
        self.metrics_collector.update_active_queries(-1).await;

        let metadata = RetrievalMetadata {
            query: processed_query,
            timestamp: chrono::Utc::now().timestamp(),
            total_time_ms: total_time.as_millis() as u64,
            strategy: "multi_hop".to_string(),
            cache_hits,
            documents_processed: ranked_documents.len(),
            warnings,
        };

        Ok(RetrievalResult {
            context,
            hop_results,
            ranked_documents,
            metadata,
            performance,
        })
    }

    /// Perform parallel retrieval for multiple queries
    pub async fn retrieve_batch(&self, queries: &[String], max_results: usize) -> Result<Vec<RetrievalResult>> {
        // Process queries in parallel with concurrency limiting
        let results: Result<Vec<_>> = stream::iter(queries)
            .map(|query| async move {
                let _permit = self.concurrency_limiter.acquire().await
                    .map_err(|e| RetrieverError::concurrency(format!("Failed to acquire permit: {}", e)))?;
                self.retrieve(query, max_results).await
            })
            .buffer_unordered(self.config.performance.batch.max_size)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        results
    }

    async fn process_query_with_cache(&self, query: &str) -> Result<ProcessedQuery> {
        // Check query cache
        if let Some(cached_query) = self.query_cache.get_processed_query(query).await? {
            self.metrics_collector.record_cache_hit().await;
            return Ok(cached_query);
        }

        self.metrics_collector.record_cache_miss().await;

        // Process query
        let processed_query = self.query_processor.process(query).await?;

        // Cache processed query
        self.query_cache.cache_processed_query(query, &processed_query).await?;

        Ok(processed_query)
    }

    async fn multi_hop_search(&self, query: &ProcessedQuery, max_results: usize) -> Result<Vec<HopRetrievalResult>> {
        let mut hop_results = Vec::new();
        let mut cumulative_documents = Vec::new();
        let mut cumulative_score = 1.0;

        for hop in 0..self.multihop_config.max_hops {
            let hop_start = Instant::now();

            // Determine search query for this hop
            let search_query = if hop == 0 {
                // First hop: use original query
                &query.preprocessed
            } else {
                // Subsequent hops: expand query based on previous results
                &self.expand_query_for_hop(query, &cumulative_documents).await?
            };

            // Perform search for this hop
            let search_results = self.search_engine.search(search_query, self.multihop_config.max_docs_per_hop).await?;

            // Filter results by relevance threshold
            let filtered_results: Vec<_> = search_results.into_iter()
                .filter(|result| result.score >= self.multihop_config.min_relevance)
                .collect();

            if filtered_results.is_empty() {
                break; // No more relevant results found
            }

            // Perform graph traversal if available
            let graph_results = if hop > 0 && !cumulative_documents.is_empty() {
                let start_nodes: Vec<String> = cumulative_documents.iter()
                    .take(10) // Limit starting nodes for performance
                    .map(|r| r.id.clone())
                    .collect();

                let traversal_strategy = TraversalStrategy::from(self.config.graph.strategy.clone());
                Some(self.graph_traverser.traverse(start_nodes, traversal_strategy).await?)
            } else {
                None
            };

            // Update cumulative score with decay
            cumulative_score *= self.multihop_config.hop_decay;

            // Create hop result
            let hop_result = HopRetrievalResult {
                hop,
                search_results: filtered_results.clone(),
                graph_results: graph_results.first().cloned(),
                cumulative_score,
                processing_time_ms: hop_start.elapsed().as_millis() as u64,
            };

            hop_results.push(hop_result);

            // Add to cumulative documents
            cumulative_documents.extend(filtered_results);

            // Early stopping check
            if cumulative_score < self.multihop_config.early_stop_threshold {
                break;
            }

            // Check if we have enough results
            if cumulative_documents.len() >= max_results {
                break;
            }
        }

        Ok(hop_results)
    }

    async fn expand_query_for_hop(&self, original_query: &ProcessedQuery, previous_results: &[SearchResult]) -> Result<String> {
        // Extract key terms from previous results
        let mut key_terms = std::collections::HashSet::new();

        for result in previous_results.iter().take(5) { // Use top 5 results
            let words: Vec<&str> = result.content.split_whitespace().collect();
            for word in words.iter().take(10) { // Take top 10 words per document
                if word.len() > 3 { // Filter short words
                    key_terms.insert(word.to_lowercase());
                }
            }
        }

        // Combine with original query
        let mut expanded_query = original_query.preprocessed.clone();
        for term in key_terms.iter().take(5) { // Limit expansion terms
            if !original_query.preprocessed.contains(term) {
                expanded_query.push_str(&format!(" {}", term));
            }
        }

        Ok(expanded_query)
    }

    async fn rank_documents(&self, documents: Vec<SearchResult>, query: &ProcessedQuery) -> Result<Vec<ScoredDocument>> {
        // Convert to scored documents for ranking
        let scored_documents: Vec<ScoredDocument> = documents.into_iter()
            .enumerate()
            .map(|(i, result)| ScoredDocument {
                result,
                ranking_score: 0.0, // Will be updated by ranking
                feature_scores: HashMap::new(),
                ranking_explanation: String::new(),
                original_position: i,
                new_position: 0,
            })
            .collect();

        // Rank documents
        let ranked_documents = self.ranking_model.rank(scored_documents, query).await?;

        // Apply re-ranking
        let reranked_documents = self.reranker.rerank(ranked_documents, query).await?;

        Ok(reranked_documents)
    }

    async fn calculate_performance(&self, component_times: &HashMap<String, u64>, total_time: Duration, documents: &[ScoredDocument]) -> RetrievalPerformance {
        let memory_usage = self.cache_manager.size_bytes().await as f64 / 1024.0 / 1024.0;

        // Calculate parallel efficiency (ratio of actual speedup to theoretical speedup)
        let total_component_time: u64 = component_times.values().sum();
        let parallel_efficiency = if total_component_time > 0 {
            (total_time.as_millis() as f64 / total_component_time as f64).min(1.0) as f32
        } else {
            1.0
        };

        // Calculate quality score based on relevance scores
        let quality_score = if !documents.is_empty() {
            documents.iter().map(|d| d.ranking_score).sum::<f32>() / documents.len() as f32
        } else {
            0.0
        };

        // Calculate throughput
        let documents_per_second = if total_time.as_secs_f32() > 0.0 {
            documents.len() as f32 / total_time.as_secs_f32()
        } else {
            0.0
        };

        RetrievalPerformance {
            component_times: component_times.clone(),
            memory_usage_mb: memory_usage,
            parallel_efficiency,
            quality_score,
            documents_per_second,
        }
    }

    /// Get retrieval metrics
    pub async fn get_metrics(&self) -> RetrievalMetrics {
        self.metrics_collector.get_metrics().await
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> crate::cache::CacheStats {
        self.cache_manager.stats().await
    }

    /// Warm up caches
    pub async fn warmup(&self, queries: Vec<String>) -> Result<()> {
        self.cache_manager.warm_cache(queries).await
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let mut issues = Vec::new();

        // Check cache health
        let cache_stats = self.cache_manager.stats().await;
        if cache_stats.hit_rate < 0.1 {
            issues.push("Low cache hit rate".to_string());
        }

        // Check memory usage
        let memory_usage = self.cache_manager.size_bytes().await;
        let max_memory = self.config.performance.memory.max_usage_mb as u64 * 1024 * 1024;
        if memory_usage > max_memory {
            issues.push("High memory usage".to_string());
        }

        // Check metrics
        let metrics = self.get_metrics().await;
        if metrics.errors.total_error_rate > 0.05 {
            issues.push("High error rate".to_string());
        }

        Ok(HealthStatus {
            status: if issues.is_empty() { "healthy".to_string() } else { "degraded".to_string() },
            issues,
            cache_hit_rate: cache_stats.hit_rate,
            memory_usage_mb: memory_usage as f64 / 1024.0 / 1024.0,
            error_rate: metrics.errors.total_error_rate,
            uptime_seconds: 0, // Would need to track startup time
        })
    }
}

/// Health status information
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub issues: Vec<String>,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub error_rate: f64,
    pub uptime_seconds: u64,
}

/// Builder for configuring and creating a MultiHopRetriever
pub struct RetrieverBuilder {
    config: RetrieverConfig,
}

impl RetrieverBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: RetrieverConfig::default(),
        }
    }

    /// Create builder from configuration
    pub fn from_config(config: RetrieverConfig) -> Self {
        Self { config }
    }

    /// Set vector configuration
    pub fn with_vector_config(mut self, config: crate::config::VectorConfig) -> Self {
        self.config.vector = config;
        self
    }

    /// Set graph configuration
    pub fn with_graph_config(mut self, config: crate::config::GraphConfig) -> Self {
        self.config.graph = config;
        self
    }

    /// Set search configuration
    pub fn with_search_config(mut self, config: crate::config::SearchConfig) -> Self {
        self.config.search = config;
        self
    }

    /// Set cache configuration
    pub fn with_cache_config(mut self, config: crate::config::CacheConfig) -> Self {
        self.config.cache = config;
        self
    }

    /// Enable metrics collection
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.config.metrics.enabled = enabled;
        self
    }

    /// Set number of parallel threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.config.performance.num_threads = threads;
        self
    }

    /// Build the retriever
    pub async fn build(self) -> Result<MultiHopRetriever> {
        // Validate configuration
        self.config.validate()?;

        // Create retriever
        MultiHopRetriever::new(self.config).await
    }
}

impl Default for RetrieverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retriever_builder() {
        let builder = RetrieverBuilder::new()
            .with_metrics(true)
            .with_threads(4);

        // Note: This test would need actual models and data to work
        // In a real test environment, you'd set up mock components
        assert!(builder.config.metrics.enabled);
        assert_eq!(builder.config.performance.num_threads, 4);
    }

    #[test]
    fn test_multihop_config() {
        let config = MultiHopConfig::default();
        assert_eq!(config.max_hops, 3);
        assert_eq!(config.max_docs_per_hop, 50);
        assert!(config.bidirectional);
    }

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus {
            status: "healthy".to_string(),
            issues: vec![],
            cache_hit_rate: 0.85,
            memory_usage_mb: 256.0,
            error_rate: 0.01,
            uptime_seconds: 3600,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("0.85"));
    }

    #[tokio::test]
    async fn test_query_hash_consistency() {
        let query = "test query for hashing";
        let hash1 = utils::generate_hash(query);
        let hash2 = utils::generate_hash(query);
        assert_eq!(hash1, hash2);
    }
}
"