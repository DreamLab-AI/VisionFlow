//! Metrics collection and monitoring for retrieval system
//!
//! Comprehensive performance monitoring and quality metrics

use crate::error::{Result, RetrieverError};
use crate::config::{MetricsConfig, MetricType, MetricsExportConfig, ExportFormat};
use prometheus::{Counter, Histogram, Gauge, Registry, Encoder, TextEncoder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Comprehensive retrieval metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    /// Latency metrics
    pub latency: LatencyMetrics,

    /// Throughput metrics
    pub throughput: ThroughputMetrics,

    /// Accuracy metrics
    pub accuracy: AccuracyMetrics,

    /// Memory usage metrics
    pub memory: MemoryMetrics,

    /// Cache performance metrics
    pub cache: CacheMetrics,

    /// Error rate metrics
    pub errors: ErrorMetrics,

    /// Quality score metrics
    pub quality: QualityMetrics,

    /// Custom metrics
    pub custom: HashMap<String, f64>,
}

/// Latency measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Query processing latency
    pub query_processing_ms: LatencyStats,

    /// Embedding generation latency
    pub embedding_ms: LatencyStats,

    /// Vector search latency
    pub vector_search_ms: LatencyStats,

    /// Graph traversal latency
    pub graph_traversal_ms: LatencyStats,

    /// Ranking latency
    pub ranking_ms: LatencyStats,

    /// Total retrieval latency
    pub total_retrieval_ms: LatencyStats,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub count: u64,
}

/// Throughput measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Queries per second
    pub queries_per_second: f64,

    /// Documents processed per second
    pub documents_per_second: f64,

    /// Embeddings generated per second
    pub embeddings_per_second: f64,

    /// Peak QPS achieved
    pub peak_qps: f64,

    /// Average QPS over time window
    pub average_qps: f64,
}

/// Accuracy and relevance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Mean Reciprocal Rank
    pub mrr: f64,

    /// Normalized Discounted Cumulative Gain
    pub ndcg_at_k: HashMap<usize, f64>,

    /// Precision at K
    pub precision_at_k: HashMap<usize, f64>,

    /// Recall at K
    pub recall_at_k: HashMap<usize, f64>,

    /// Mean Average Precision
    pub map: f64,

    /// Click-through rate (if available)
    pub ctr: Option<f64>,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total memory usage in bytes
    pub total_bytes: u64,

    /// Vector index memory usage
    pub vector_index_bytes: u64,

    /// Graph memory usage
    pub graph_bytes: u64,

    /// Cache memory usage
    pub cache_bytes: u64,

    /// Peak memory usage
    pub peak_bytes: u64,

    /// Memory usage percentage
    pub usage_percentage: f64,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Cache hit rate
    pub hit_rate: f64,

    /// Cache miss rate
    pub miss_rate: f64,

    /// Average cache lookup time
    pub avg_lookup_time_ms: f64,

    /// Cache size utilization
    pub size_utilization: f64,

    /// Eviction rate
    pub eviction_rate: f64,
}

/// Error tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total error rate
    pub total_error_rate: f64,

    /// Error rates by category
    pub error_rates_by_category: HashMap<String, f64>,

    /// Error counts by type
    pub error_counts: HashMap<String, u64>,

    /// Mean time to recovery
    pub mttr_seconds: f64,
}

/// Quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average relevance score
    pub avg_relevance_score: f64,

    /// Result diversity score
    pub diversity_score: f64,

    /// Context coherence score
    pub coherence_score: f64,

    /// Coverage completeness score
    pub coverage_score: f64,

    /// User satisfaction score (if available)
    pub satisfaction_score: Option<f64>,
}

/// Metrics collector with Prometheus integration
pub struct MetricsCollector {
    /// Prometheus registry
    registry: Registry,

    /// Configuration
    config: MetricsConfig,

    /// Collected metrics
    metrics: Arc<RwLock<RetrievalMetrics>>,

    /// Prometheus counters
    counters: PrometheusCounters,

    /// Prometheus histograms
    histograms: PrometheusHistograms,

    /// Prometheus gauges
    gauges: PrometheusGauges,

    /// Metrics buffer for batching
    metrics_buffer: Arc<RwLock<Vec<MetricEvent>>>,
}

/// Prometheus counter metrics
struct PrometheusCounters {
    total_queries: Counter,
    cache_hits: Counter,
    cache_misses: Counter,
    errors_total: Counter,
    documents_processed: Counter,
}

/// Prometheus histogram metrics
struct PrometheusHistograms {
    query_duration: Histogram,
    embedding_duration: Histogram,
    search_duration: Histogram,
    ranking_duration: Histogram,
}

/// Prometheus gauge metrics
struct PrometheusGauges {
    memory_usage: Gauge,
    cache_size: Gauge,
    active_queries: Gauge,
    quality_score: Gauge,
}

/// Metric event for batching
#[derive(Debug, Clone)]
struct MetricEvent {
    timestamp: Instant,
    event_type: MetricEventType,
    value: f64,
    labels: HashMap<String, String>,
}

#[derive(Debug, Clone)]
enum MetricEventType {
    QueryLatency,
    EmbeddingLatency,
    SearchLatency,
    RankingLatency,
    CacheHit,
    CacheMiss,
    Error(String),
    MemoryUsage,
    QualityScore,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(config: MetricsConfig) -> Result<Self> {
        let registry = Registry::new();

        // Initialize Prometheus metrics
        let counters = PrometheusCounters {
            total_queries: Counter::new("retrieval_queries_total", "Total number of queries processed")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create counter: {}", e)))?,
            cache_hits: Counter::new("retrieval_cache_hits_total", "Total number of cache hits")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create counter: {}", e)))?,
            cache_misses: Counter::new("retrieval_cache_misses_total", "Total number of cache misses")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create counter: {}", e)))?,
            errors_total: Counter::new("retrieval_errors_total", "Total number of errors")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create counter: {}", e)))?,
            documents_processed: Counter::new("retrieval_documents_processed_total", "Total documents processed")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create counter: {}", e)))?,
        };

        let histograms = PrometheusHistograms {
            query_duration: Histogram::with_opts(
                prometheus::HistogramOpts::new("retrieval_query_duration_seconds", "Query processing duration")
                    .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
            ).map_err(|e| RetrieverError::unknown(format!("Failed to create histogram: {}", e)))?,
            embedding_duration: Histogram::with_opts(
                prometheus::HistogramOpts::new("retrieval_embedding_duration_seconds", "Embedding generation duration")
                    .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
            ).map_err(|e| RetrieverError::unknown(format!("Failed to create histogram: {}", e)))?,
            search_duration: Histogram::with_opts(
                prometheus::HistogramOpts::new("retrieval_search_duration_seconds", "Search execution duration")
                    .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
            ).map_err(|e| RetrieverError::unknown(format!("Failed to create histogram: {}", e)))?,
            ranking_duration: Histogram::with_opts(
                prometheus::HistogramOpts::new("retrieval_ranking_duration_seconds", "Ranking duration")
                    .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1])
            ).map_err(|e| RetrieverError::unknown(format!("Failed to create histogram: {}", e)))?,
        };

        let gauges = PrometheusGauges {
            memory_usage: Gauge::new("retrieval_memory_usage_bytes", "Current memory usage in bytes")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create gauge: {}", e)))?,
            cache_size: Gauge::new("retrieval_cache_size_bytes", "Current cache size in bytes")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create gauge: {}", e)))?,
            active_queries: Gauge::new("retrieval_active_queries", "Number of currently active queries")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create gauge: {}", e)))?,
            quality_score: Gauge::new("retrieval_quality_score", "Current quality score")
                .map_err(|e| RetrieverError::unknown(format!("Failed to create gauge: {}", e)))?,
        };

        // Register metrics with Prometheus
        registry.register(Box::new(counters.total_queries.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(counters.cache_hits.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(counters.cache_misses.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(counters.errors_total.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(counters.documents_processed.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;

        registry.register(Box::new(histograms.query_duration.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(histograms.embedding_duration.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(histograms.search_duration.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(histograms.ranking_duration.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;

        registry.register(Box::new(gauges.memory_usage.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(gauges.cache_size.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(gauges.active_queries.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;
        registry.register(Box::new(gauges.quality_score.clone()))
            .map_err(|e| RetrieverError::unknown(format!("Failed to register metric: {}", e)))?;

        Ok(Self {
            registry,
            config,
            metrics: Arc::new(RwLock::new(RetrievalMetrics::default())),
            counters,
            histograms,
            gauges,
            metrics_buffer: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Record query processing time
    pub async fn record_query_latency(&self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        self.histograms.query_duration.observe(seconds);
        self.counters.total_queries.inc();

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::QueryLatency,
            value: duration.as_millis() as f64,
            labels: HashMap::new(),
        }).await;
    }

    /// Record embedding generation time
    pub async fn record_embedding_latency(&self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        self.histograms.embedding_duration.observe(seconds);

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::EmbeddingLatency,
            value: duration.as_millis() as f64,
            labels: HashMap::new(),
        }).await;
    }

    /// Record search execution time
    pub async fn record_search_latency(&self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        self.histograms.search_duration.observe(seconds);

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::SearchLatency,
            value: duration.as_millis() as f64,
            labels: HashMap::new(),
        }).await;
    }

    /// Record ranking time
    pub async fn record_ranking_latency(&self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        self.histograms.ranking_duration.observe(seconds);

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::RankingLatency,
            value: duration.as_millis() as f64,
            labels: HashMap::new(),
        }).await;
    }

    /// Record cache hit
    pub async fn record_cache_hit(&self) {
        self.counters.cache_hits.inc();

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::CacheHit,
            value: 1.0,
            labels: HashMap::new(),
        }).await;
    }

    /// Record cache miss
    pub async fn record_cache_miss(&self) {
        self.counters.cache_misses.inc();

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::CacheMiss,
            value: 1.0,
            labels: HashMap::new(),
        }).await;
    }

    /// Record error occurrence
    pub async fn record_error(&self, error_category: &str) {
        self.counters.errors_total.inc();

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::Error(error_category.to_string()),
            value: 1.0,
            labels: HashMap::from([("category".to_string(), error_category.to_string())]),
        }).await;
    }

    /// Update memory usage
    pub async fn update_memory_usage(&self, bytes: u64) {
        self.gauges.memory_usage.set(bytes as f64);

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::MemoryUsage,
            value: bytes as f64,
            labels: HashMap::new(),
        }).await;
    }

    /// Update cache size
    pub async fn update_cache_size(&self, bytes: u64) {
        self.gauges.cache_size.set(bytes as f64);
    }

    /// Update active query count
    pub async fn update_active_queries(&self, count: i64) {
        if count >= 0 {
            self.gauges.active_queries.set(count as f64);
        }
    }

    /// Update quality score
    pub async fn update_quality_score(&self, score: f64) {
        self.gauges.quality_score.set(score);

        self.buffer_event(MetricEvent {
            timestamp: Instant::now(),
            event_type: MetricEventType::QualityScore,
            value: score,
            labels: HashMap::new(),
        }).await;
    }

    /// Get current metrics snapshot
    pub async fn get_metrics(&self) -> RetrievalMetrics {
        // Process buffered events
        self.process_buffered_events().await;

        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Export metrics in specified format
    pub async fn export_metrics(&self, format: &ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Prometheus => self.export_prometheus().await,
            ExportFormat::Json => self.export_json().await,
            ExportFormat::Csv => self.export_csv().await,
            ExportFormat::Custom { format } => {
                Err(RetrieverError::unknown(format!("Custom format not implemented: {}", format)))
            }
        }
    }

    async fn export_prometheus(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();

        encoder.encode(&metric_families, &mut buffer)
            .map_err(|e| RetrieverError::unknown(format!("Failed to encode metrics: {}", e)))?;

        String::from_utf8(buffer)
            .map_err(|e| RetrieverError::unknown(format!("Failed to convert metrics to string: {}", e)))
    }

    async fn export_json(&self) -> Result<String> {
        let metrics = self.get_metrics().await;
        serde_json::to_string_pretty(&metrics)
            .map_err(|e| RetrieverError::unknown(format!("Failed to serialize metrics: {}", e)))
    }

    async fn export_csv(&self) -> Result<String> {
        let metrics = self.get_metrics().await;
        let mut csv = String::new();

        // Header
        csv.push_str("metric_type,metric_name,value\n");

        // Latency metrics
        csv.push_str(&format!("latency,query_processing_mean,{}\n", metrics.latency.query_processing_ms.mean));
        csv.push_str(&format!("latency,embedding_mean,{}\n", metrics.latency.embedding_ms.mean));
        csv.push_str(&format!("latency,search_mean,{}\n", metrics.latency.vector_search_ms.mean));

        // Throughput metrics
        csv.push_str(&format!("throughput,queries_per_second,{}\n", metrics.throughput.queries_per_second));
        csv.push_str(&format!("throughput,documents_per_second,{}\n", metrics.throughput.documents_per_second));

        // Cache metrics
        csv.push_str(&format!("cache,hit_rate,{}\n", metrics.cache.hit_rate));
        csv.push_str(&format!("cache,miss_rate,{}\n", metrics.cache.miss_rate));

        // Memory metrics
        csv.push_str(&format!("memory,total_bytes,{}\n", metrics.memory.total_bytes));
        csv.push_str(&format!("memory,usage_percentage,{}\n", metrics.memory.usage_percentage));

        Ok(csv)
    }

    async fn buffer_event(&self, event: MetricEvent) {
        let mut buffer = self.metrics_buffer.write().await;
        buffer.push(event);

        // Process if buffer is full
        if buffer.len() >= 1000 {
            drop(buffer);
            self.process_buffered_events().await;
        }
    }

    async fn process_buffered_events(&self) {
        let mut buffer = self.metrics_buffer.write().await;
        let events = buffer.drain(..).collect::<Vec<_>>();
        drop(buffer);

        if events.is_empty() {
            return;
        }

        let mut metrics = self.metrics.write().await;

        // Group events by type and calculate statistics
        let mut latency_events: HashMap<String, Vec<f64>> = HashMap::new();
        let mut counter_events: HashMap<String, u64> = HashMap::new();

        for event in events {
            match event.event_type {
                MetricEventType::QueryLatency => {
                    latency_events.entry("query".to_string()).or_default().push(event.value);
                }
                MetricEventType::EmbeddingLatency => {
                    latency_events.entry("embedding".to_string()).or_default().push(event.value);
                }
                MetricEventType::SearchLatency => {
                    latency_events.entry("search".to_string()).or_default().push(event.value);
                }
                MetricEventType::RankingLatency => {
                    latency_events.entry("ranking".to_string()).or_default().push(event.value);
                }
                MetricEventType::CacheHit => {
                    *counter_events.entry("cache_hits".to_string()).or_insert(0) += 1;
                }
                MetricEventType::CacheMiss => {
                    *counter_events.entry("cache_misses".to_string()).or_insert(0) += 1;
                }
                MetricEventType::Error(category) => {
                    *counter_events.entry(format!("error_{}", category)).or_insert(0) += 1;
                }
                MetricEventType::MemoryUsage => {
                    metrics.memory.total_bytes = event.value as u64;
                }
                MetricEventType::QualityScore => {
                    metrics.quality.avg_relevance_score = event.value;
                }
            }
        }

        // Update latency statistics
        for (metric_type, values) in latency_events {
            let stats = self.calculate_latency_stats(&values);

            match metric_type.as_str() {
                "query" => metrics.latency.query_processing_ms = stats,
                "embedding" => metrics.latency.embedding_ms = stats,
                "search" => metrics.latency.vector_search_ms = stats,
                "ranking" => metrics.latency.ranking_ms = stats,
                _ => {}
            }
        }

        // Update cache metrics
        let cache_hits = counter_events.get("cache_hits").unwrap_or(&0);
        let cache_misses = counter_events.get("cache_misses").unwrap_or(&0);
        let total_cache_requests = cache_hits + cache_misses;

        if total_cache_requests > 0 {
            metrics.cache.hit_rate = *cache_hits as f64 / total_cache_requests as f64;
            metrics.cache.miss_rate = *cache_misses as f64 / total_cache_requests as f64;
        }
    }

    fn calculate_latency_stats(&self, values: &[f64]) -> LatencyStats {
        if values.is_empty() {
            return LatencyStats::default();
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = sorted_values.len();
        let mean = sorted_values.iter().sum::<f64>() / count as f64;
        let median = if count % 2 == 0 {
            (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0
        } else {
            sorted_values[count / 2]
        };

        let p95_index = ((count as f64) * 0.95) as usize;
        let p99_index = ((count as f64) * 0.99) as usize;

        LatencyStats {
            mean,
            median,
            p95: sorted_values[p95_index.min(count - 1)],
            p99: sorted_values[p99_index.min(count - 1)],
            min: sorted_values[0],
            max: sorted_values[count - 1],
            count: count as u64,
        }
    }
}

impl Default for RetrievalMetrics {
    fn default() -> Self {
        Self {
            latency: LatencyMetrics::default(),
            throughput: ThroughputMetrics::default(),
            accuracy: AccuracyMetrics::default(),
            memory: MemoryMetrics::default(),
            cache: CacheMetrics::default(),
            errors: ErrorMetrics::default(),
            quality: QualityMetrics::default(),
            custom: HashMap::new(),
        }
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            query_processing_ms: LatencyStats::default(),
            embedding_ms: LatencyStats::default(),
            vector_search_ms: LatencyStats::default(),
            graph_traversal_ms: LatencyStats::default(),
            ranking_ms: LatencyStats::default(),
            total_retrieval_ms: LatencyStats::default(),
        }
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            p95: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
            count: 0,
        }
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            queries_per_second: 0.0,
            documents_per_second: 0.0,
            embeddings_per_second: 0.0,
            peak_qps: 0.0,
            average_qps: 0.0,
        }
    }
}

impl Default for AccuracyMetrics {
    fn default() -> Self {
        Self {
            mrr: 0.0,
            ndcg_at_k: HashMap::new(),
            precision_at_k: HashMap::new(),
            recall_at_k: HashMap::new(),
            map: 0.0,
            ctr: None,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            vector_index_bytes: 0,
            graph_bytes: 0,
            cache_bytes: 0,
            peak_bytes: 0,
            usage_percentage: 0.0,
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            miss_rate: 0.0,
            avg_lookup_time_ms: 0.0,
            size_utilization: 0.0,
            eviction_rate: 0.0,
        }
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            total_error_rate: 0.0,
            error_rates_by_category: HashMap::new(),
            error_counts: HashMap::new(),
            mttr_seconds: 0.0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            avg_relevance_score: 0.0,
            diversity_score: 0.0,
            coherence_score: 0.0,
            coverage_score: 0.0,
            satisfaction_score: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.latency.query_processing_ms.count, 0);
    }

    #[tokio::test]
    async fn test_latency_recording() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let duration = Duration::from_millis(100);
        collector.record_query_latency(duration).await;

        let metrics = collector.get_metrics().await;
        assert!(metrics.latency.query_processing_ms.count > 0);
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        collector.record_cache_hit().await;
        collector.record_cache_hit().await;
        collector.record_cache_miss().await;

        let metrics = collector.get_metrics().await;
        assert!((metrics.cache.hit_rate - 0.667).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_prometheus_export() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        collector.record_query_latency(Duration::from_millis(50)).await;

        let prometheus_output = collector.export_metrics(&ExportFormat::Prometheus).await.unwrap();
        assert!(prometheus_output.contains("retrieval_queries_total"));
    }

    #[tokio::test]
    async fn test_json_export() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let json_output = collector.export_metrics(&ExportFormat::Json).await.unwrap();
        assert!(json_output.contains("latency"));
        assert!(json_output.contains("throughput"));
    }

    #[test]
    fn test_latency_stats_calculation() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).unwrap();

        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = collector.calculate_latency_stats(&values);

        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.median, 30.0);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 50.0);
        assert_eq!(stats.count, 5);
    }
}
"