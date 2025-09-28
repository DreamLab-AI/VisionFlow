//! Performance metrics and monitoring for vector operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::error::Result;

/// Performance metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    // Operation counters
    embedding_operations: AtomicU64,
    search_operations: AtomicU64,
    index_operations: AtomicU64,
    storage_operations: AtomicU64,

    // Timing metrics
    timing_data: Arc<RwLock<TimingData>>,

    // Error counters
    embedding_errors: AtomicU64,
    search_errors: AtomicU64,
    index_errors: AtomicU64,
    storage_errors: AtomicU64,

    // Resource usage
    memory_usage: Arc<RwLock<MemoryUsage>>,
    gpu_usage: Arc<RwLock<Option<GpuUsage>>>,

    // Custom metrics
    custom_metrics: Arc<RwLock<HashMap<String, f64>>>,
}

/// Timing data for different operations
#[derive(Debug, Default)]
struct TimingData {
    embedding_times: Vec<Duration>,
    search_times: Vec<Duration>,
    index_times: Vec<Duration>,
    storage_times: Vec<Duration>,
}

/// Memory usage statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub total_allocated_bytes: u64,
    pub vectors_memory_bytes: u64,
    pub index_memory_bytes: u64,
    pub cache_memory_bytes: u64,
    pub peak_memory_bytes: u64,
}

/// GPU usage statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GpuUsage {
    pub device_id: usize,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub utilization_percent: f32,
    pub temperature_celsius: f32,
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Operation counts
    pub embedding_operations: u64,
    pub search_operations: u64,
    pub index_operations: u64,
    pub storage_operations: u64,

    // Error rates
    pub embedding_error_rate: f64,
    pub search_error_rate: f64,
    pub index_error_rate: f64,
    pub storage_error_rate: f64,

    // Timing statistics
    pub avg_embedding_time_ms: f64,
    pub avg_search_time_ms: f64,
    pub avg_index_time_ms: f64,
    pub avg_storage_time_ms: f64,

    pub p95_embedding_time_ms: f64,
    pub p95_search_time_ms: f64,
    pub p95_index_time_ms: f64,
    pub p95_storage_time_ms: f64,

    // Resource usage
    pub memory_usage: MemoryUsage,
    pub gpu_usage: Option<GpuUsage>,

    // Throughput metrics
    pub embeddings_per_second: f64,
    pub searches_per_second: f64,

    // Quality metrics
    pub cache_hit_rate: f64,
    pub index_compression_ratio: f64,

    // Custom metrics
    pub custom_metrics: HashMap<String, f64>,

    // Timestamp
    pub collected_at: chrono::DateTime<chrono::Utc>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            embedding_operations: AtomicU64::new(0),
            search_operations: AtomicU64::new(0),
            index_operations: AtomicU64::new(0),
            storage_operations: AtomicU64::new(0),

            timing_data: Arc::new(RwLock::new(TimingData::default())),

            embedding_errors: AtomicU64::new(0),
            search_errors: AtomicU64::new(0),
            index_errors: AtomicU64::new(0),
            storage_errors: AtomicU64::new(0),

            memory_usage: Arc::new(RwLock::new(MemoryUsage::default())),
            gpu_usage: Arc::new(RwLock::new(None)),

            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record an embedding operation
    pub async fn record_embedding_operation(&self, duration: Duration) {
        self.embedding_operations.fetch_add(1, Ordering::Relaxed);
        let mut timing_data = self.timing_data.write().await;
        timing_data.embedding_times.push(duration);

        // Keep only the last 1000 measurements to avoid memory growth
        if timing_data.embedding_times.len() > 1000 {
            timing_data.embedding_times.remove(0);
        }
    }

    /// Record a search operation
    pub async fn record_search_operation(&self, duration: Duration) {
        self.search_operations.fetch_add(1, Ordering::Relaxed);
        let mut timing_data = self.timing_data.write().await;
        timing_data.search_times.push(duration);

        if timing_data.search_times.len() > 1000 {
            timing_data.search_times.remove(0);
        }
    }

    /// Record an index operation
    pub async fn record_index_operation(&self, duration: Duration) {
        self.index_operations.fetch_add(1, Ordering::Relaxed);
        let mut timing_data = self.timing_data.write().await;
        timing_data.index_times.push(duration);

        if timing_data.index_times.len() > 1000 {
            timing_data.index_times.remove(0);
        }
    }

    /// Record a storage operation
    pub async fn record_storage_operation(&self, duration: Duration) {
        self.storage_operations.fetch_add(1, Ordering::Relaxed);
        let mut timing_data = self.timing_data.write().await;
        timing_data.storage_times.push(duration);

        if timing_data.storage_times.len() > 1000 {
            timing_data.storage_times.remove(0);
        }
    }

    /// Record an error for a specific operation type
    pub fn record_error(&self, operation_type: OperationType) {
        match operation_type {
            OperationType::Embedding => {
                self.embedding_errors.fetch_add(1, Ordering::Relaxed);
            }
            OperationType::Search => {
                self.search_errors.fetch_add(1, Ordering::Relaxed);
            }
            OperationType::Index => {
                self.index_errors.fetch_add(1, Ordering::Relaxed);
            }
            OperationType::Storage => {
                self.storage_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Update memory usage statistics
    pub async fn update_memory_usage(&self, memory_usage: MemoryUsage) {
        let mut current_usage = self.memory_usage.write().await;
        *current_usage = memory_usage;
    }

    /// Update GPU usage statistics
    pub async fn update_gpu_usage(&self, gpu_usage: Option<GpuUsage>) {
        let mut current_usage = self.gpu_usage.write().await;
        *current_usage = gpu_usage;
    }

    /// Set a custom metric
    pub async fn set_custom_metric(&self, name: String, value: f64) {
        let mut custom_metrics = self.custom_metrics.write().await;
        custom_metrics.insert(name, value);
    }

    /// Increment a custom counter metric
    pub async fn increment_custom_counter(&self, name: String, increment: f64) {
        let mut custom_metrics = self.custom_metrics.write().await;
        let current = custom_metrics.get(&name).unwrap_or(&0.0);
        custom_metrics.insert(name, current + increment);
    }

    /// Calculate percentile from a sorted list of durations
    fn calculate_percentile(sorted_durations: &[Duration], percentile: f64) -> f64 {
        if sorted_durations.is_empty() {
            return 0.0;
        }

        let index = (percentile / 100.0) * (sorted_durations.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            sorted_durations[lower_index].as_millis() as f64
        } else {
            let lower_value = sorted_durations[lower_index].as_millis() as f64;
            let upper_value = sorted_durations[upper_index].as_millis() as f64;
            let weight = index - lower_index as f64;
            lower_value + weight * (upper_value - lower_value)
        }
    }

    /// Calculate average duration
    fn calculate_average(durations: &[Duration]) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }

        let total_ms: f64 = durations.iter().map(|d| d.as_millis() as f64).sum();
        total_ms / durations.len() as f64
    }

    /// Get comprehensive performance metrics
    pub async fn get_metrics(&self) -> Result<PerformanceMetrics> {
        let timing_data = self.timing_data.read().await;
        let memory_usage = self.memory_usage.read().await;
        let gpu_usage = self.gpu_usage.read().await;
        let custom_metrics = self.custom_metrics.read().await;

        // Calculate timing statistics
        let mut embedding_times_sorted = timing_data.embedding_times.clone();
        embedding_times_sorted.sort();

        let mut search_times_sorted = timing_data.search_times.clone();
        search_times_sorted.sort();

        let mut index_times_sorted = timing_data.index_times.clone();
        index_times_sorted.sort();

        let mut storage_times_sorted = timing_data.storage_times.clone();
        storage_times_sorted.sort();

        // Calculate error rates
        let embedding_ops = self.embedding_operations.load(Ordering::Relaxed);
        let search_ops = self.search_operations.load(Ordering::Relaxed);
        let index_ops = self.index_operations.load(Ordering::Relaxed);
        let storage_ops = self.storage_operations.load(Ordering::Relaxed);

        let embedding_errors = self.embedding_errors.load(Ordering::Relaxed);
        let search_errors = self.search_errors.load(Ordering::Relaxed);
        let index_errors = self.index_errors.load(Ordering::Relaxed);
        let storage_errors = self.storage_errors.load(Ordering::Relaxed);

        let embedding_error_rate = if embedding_ops > 0 {
            embedding_errors as f64 / embedding_ops as f64
        } else {
            0.0
        };
        let search_error_rate = if search_ops > 0 {
            search_errors as f64 / search_ops as f64
        } else {
            0.0
        };
        let index_error_rate = if index_ops > 0 {
            index_errors as f64 / index_ops as f64
        } else {
            0.0
        };
        let storage_error_rate = if storage_ops > 0 {
            storage_errors as f64 / storage_ops as f64
        } else {
            0.0
        };

        // Calculate throughput (simplified - would need time windowing in practice)
        let embeddings_per_second = 0.0; // TODO: Implement proper throughput calculation
        let searches_per_second = 0.0;

        Ok(PerformanceMetrics {
            embedding_operations: embedding_ops,
            search_operations: search_ops,
            index_operations: index_ops,
            storage_operations: storage_ops,

            embedding_error_rate,
            search_error_rate,
            index_error_rate,
            storage_error_rate,

            avg_embedding_time_ms: Self::calculate_average(&timing_data.embedding_times),
            avg_search_time_ms: Self::calculate_average(&timing_data.search_times),
            avg_index_time_ms: Self::calculate_average(&timing_data.index_times),
            avg_storage_time_ms: Self::calculate_average(&timing_data.storage_times),

            p95_embedding_time_ms: Self::calculate_percentile(&embedding_times_sorted, 95.0),
            p95_search_time_ms: Self::calculate_percentile(&search_times_sorted, 95.0),
            p95_index_time_ms: Self::calculate_percentile(&index_times_sorted, 95.0),
            p95_storage_time_ms: Self::calculate_percentile(&storage_times_sorted, 95.0),

            memory_usage: memory_usage.clone(),
            gpu_usage: gpu_usage.clone(),

            embeddings_per_second,
            searches_per_second,

            cache_hit_rate: 0.0,        // TODO: Implement cache hit rate tracking
            index_compression_ratio: 0.0, // TODO: Implement compression ratio tracking

            custom_metrics: custom_metrics.clone(),

            collected_at: chrono::Utc::now(),
        })
    }

    /// Reset all metrics
    pub async fn reset(&self) {
        self.embedding_operations.store(0, Ordering::Relaxed);
        self.search_operations.store(0, Ordering::Relaxed);
        self.index_operations.store(0, Ordering::Relaxed);
        self.storage_operations.store(0, Ordering::Relaxed);

        self.embedding_errors.store(0, Ordering::Relaxed);
        self.search_errors.store(0, Ordering::Relaxed);
        self.index_errors.store(0, Ordering::Relaxed);
        self.storage_errors.store(0, Ordering::Relaxed);

        let mut timing_data = self.timing_data.write().await;
        *timing_data = TimingData::default();

        let mut memory_usage = self.memory_usage.write().await;
        *memory_usage = MemoryUsage::default();

        let mut gpu_usage = self.gpu_usage.write().await;
        *gpu_usage = None;

        let mut custom_metrics = self.custom_metrics.write().await;
        custom_metrics.clear();
    }
}

/// Operation types for error tracking
#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    Embedding,
    Search,
    Index,
    Storage,
}

/// Timer utility for measuring operation duration
pub struct OperationTimer {
    start_time: Instant,
    operation_type: OperationType,
    metrics: Arc<MetricsCollector>,
}

impl OperationTimer {
    pub fn new(operation_type: OperationType, metrics: Arc<MetricsCollector>) -> Self {
        Self {
            start_time: Instant::now(),
            operation_type,
            metrics,
        }
    }

    /// Finish the timer and record the measurement
    pub async fn finish(self) {
        let duration = self.start_time.elapsed();
        match self.operation_type {
            OperationType::Embedding => {
                self.metrics.record_embedding_operation(duration).await;
            }
            OperationType::Search => {
                self.metrics.record_search_operation(duration).await;
            }
            OperationType::Index => {
                self.metrics.record_index_operation(duration).await;
            }
            OperationType::Storage => {
                self.metrics.record_storage_operation(duration).await;
            }
        }
    }

    /// Finish with an error
    pub async fn finish_with_error(self) {
        self.metrics.record_error(self.operation_type);
        self.finish().await;
    }
}

/// Macro for timing operations
#[macro_export]
macro_rules! time_operation {
    ($metrics:expr, $operation_type:expr, $operation:expr) => {{
        let timer = OperationTimer::new($operation_type, $metrics.clone());
        let result = $operation;
        match &result {
            Ok(_) => timer.finish().await,
            Err(_) => timer.finish_with_error().await,
        }
        result
    }};
}

/// Performance monitoring service
pub struct PerformanceMonitor {
    metrics: Arc<MetricsCollector>,
    collection_interval: Duration,
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

impl PerformanceMonitor {
    pub fn new(metrics: Arc<MetricsCollector>, collection_interval: Duration) -> Self {
        Self {
            metrics,
            collection_interval,
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start the performance monitoring service
    pub async fn start(&self) -> Result<()> {
        if self.is_running.load(Ordering::Relaxed) {
            return Ok(());
        }

        self.is_running.store(true, Ordering::Relaxed);

        let metrics = self.metrics.clone();
        let interval = self.collection_interval;
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            while is_running.load(Ordering::Relaxed) {
                interval_timer.tick().await;

                // Collect system metrics
                if let Ok(memory_info) = Self::collect_memory_info().await {
                    metrics.update_memory_usage(memory_info).await;
                }

                // Collect GPU metrics if available
                #[cfg(feature = "gpu")]
                if let Ok(gpu_info) = Self::collect_gpu_info().await {
                    metrics.update_gpu_usage(Some(gpu_info)).await;
                }

                // Log current metrics
                if let Ok(current_metrics) = metrics.get_metrics().await {
                    log::debug!("Performance metrics: {:?}", current_metrics);
                }
            }
        });

        Ok(())
    }

    /// Stop the performance monitoring service
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    async fn collect_memory_info() -> Result<MemoryUsage> {
        // This is a simplified implementation
        // In practice, you'd use system APIs to get accurate memory information
        Ok(MemoryUsage {
            total_allocated_bytes: 0,
            vectors_memory_bytes: 0,
            index_memory_bytes: 0,
            cache_memory_bytes: 0,
            peak_memory_bytes: 0,
        })
    }

    #[cfg(feature = "gpu")]
    async fn collect_gpu_info() -> Result<GpuUsage> {
        // This would interface with CUDA or ROCm APIs
        Ok(GpuUsage {
            device_id: 0,
            memory_used_bytes: 0,
            memory_total_bytes: 0,
            utilization_percent: 0.0,
            temperature_celsius: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        // Record some operations
        collector
            .record_embedding_operation(Duration::from_millis(100))
            .await;
        collector
            .record_search_operation(Duration::from_millis(50))
            .await;

        // Record errors
        collector.record_error(OperationType::Embedding);
        collector.record_error(OperationType::Search);

        // Get metrics
        let metrics = collector.get_metrics().await.unwrap();

        assert_eq!(metrics.embedding_operations, 1);
        assert_eq!(metrics.search_operations, 1);
        assert!(metrics.avg_embedding_time_ms > 0.0);
        assert!(metrics.avg_search_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_operation_timer() {
        let collector = Arc::new(MetricsCollector::new());

        {
            let timer = OperationTimer::new(OperationType::Embedding, collector.clone());
            tokio::time::sleep(Duration::from_millis(10)).await;
            timer.finish().await;
        }

        let metrics = collector.get_metrics().await.unwrap();
        assert_eq!(metrics.embedding_operations, 1);
        assert!(metrics.avg_embedding_time_ms >= 10.0);
    }

    #[tokio::test]
    async fn test_custom_metrics() {
        let collector = MetricsCollector::new();

        collector
            .set_custom_metric("test_metric".to_string(), 42.0)
            .await;
        collector
            .increment_custom_counter("test_counter".to_string(), 1.0)
            .await;
        collector
            .increment_custom_counter("test_counter".to_string(), 2.0)
            .await;

        let metrics = collector.get_metrics().await.unwrap();
        assert_eq!(metrics.custom_metrics.get("test_metric"), Some(&42.0));
        assert_eq!(metrics.custom_metrics.get("test_counter"), Some(&3.0));
    }
}