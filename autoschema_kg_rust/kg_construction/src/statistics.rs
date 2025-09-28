//! Statistics tracking and logging functionality

use crate::{
    error::{KgConstructionError, Result},
    types::{NodeType, Statistics},
};
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Thread-safe statistics collector
#[derive(Debug, Clone)]
pub struct StatisticsCollector {
    stats: Arc<Mutex<Statistics>>,
    start_time: Instant,
}

impl StatisticsCollector {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(Statistics::default())),
            start_time: Instant::now(),
        }
    }

    /// Increment total nodes processed
    pub fn increment_nodes_processed(&self, count: usize) -> Result<()> {
        let mut stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;
        stats.total_nodes_processed += count;
        Ok(())
    }

    /// Increment batches processed
    pub fn increment_batches_processed(&self, count: usize) -> Result<()> {
        let mut stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;
        stats.total_batches_processed += count;
        Ok(())
    }

    /// Increment nodes by type
    pub fn increment_nodes_by_type(&self, node_type: NodeType, count: usize) -> Result<()> {
        let mut stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;

        match node_type {
            NodeType::Entity => stats.entities_processed += count,
            NodeType::Event => stats.events_processed += count,
            NodeType::Relation => stats.relations_processed += count,
        }

        Ok(())
    }

    /// Add concepts for a specific type
    pub fn add_concepts_for_type(&self, concept_type: String, count: usize) -> Result<()> {
        let mut stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;

        *stats.concepts_by_type.entry(concept_type).or_insert(0) += count;
        stats.unique_concepts_generated += count;
        Ok(())
    }

    /// Increment error count
    pub fn increment_errors(&self, count: usize) -> Result<()> {
        let mut stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;
        stats.errors_encountered += count;
        Ok(())
    }

    /// Update processing time
    pub fn update_processing_time(&self) -> Result<()> {
        let elapsed = self.start_time.elapsed();
        let mut stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;
        stats.processing_time_ms = elapsed.as_millis();
        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> Result<Statistics> {
        let stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;
        Ok(stats.clone())
    }

    /// Reset statistics
    pub fn reset(&self) -> Result<()> {
        let mut stats = self.stats.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Stats mutex poisoned: {}", e))
        })?;
        *stats = Statistics::default();
        Ok(())
    }
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Logger for processing events and statistics
pub struct ProcessingLogger {
    log_file: File,
    stats_collector: StatisticsCollector,
}

impl ProcessingLogger {
    pub fn new(log_file_path: impl AsRef<Path>) -> Result<Self> {
        // Ensure directory exists
        if let Some(parent) = log_file_path.as_ref().parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let log_file = File::create(log_file_path)?;
        Ok(Self {
            log_file,
            stats_collector: StatisticsCollector::new(),
        })
    }

    /// Log a message with timestamp
    pub fn log(&mut self, level: LogLevel, message: &str) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let log_entry = format!(
            "{} - {} - {}\n",
            chrono::DateTime::from_timestamp(timestamp as i64, 0)
                .unwrap_or_default()
                .format("%Y-%m-%d %H:%M:%S"),
            level,
            message
        );

        self.log_file.write_all(log_entry.as_bytes())?;
        self.log_file.flush()?;

        // Also print to stdout for important messages
        if matches!(level, LogLevel::Error | LogLevel::Warning) {
            eprint!("{}", log_entry);
        }

        Ok(())
    }

    /// Log batch processing start
    pub fn log_batch_start(&mut self, batch_type: NodeType, batch_size: usize) -> Result<()> {
        self.log(
            LogLevel::Info,
            &format!("Starting {} batch with {} items", batch_type, batch_size),
        )?;
        self.stats_collector.increment_batches_processed(1)?;
        Ok(())
    }

    /// Log batch processing completion
    pub fn log_batch_complete(
        &mut self,
        batch_type: NodeType,
        batch_size: usize,
        concepts_generated: usize,
        processing_time_ms: u128,
    ) -> Result<()> {
        self.log(
            LogLevel::Info,
            &format!(
                "Completed {} batch: {} items, {} concepts, {}ms",
                batch_type, batch_size, concepts_generated, processing_time_ms
            ),
        )?;

        self.stats_collector.increment_nodes_by_type(batch_type, batch_size)?;
        self.stats_collector.add_concepts_for_type(
            batch_type.to_string(),
            concepts_generated,
        )?;

        Ok(())
    }

    /// Log error during processing
    pub fn log_error(&mut self, error: &KgConstructionError, context: &str) -> Result<()> {
        self.log(
            LogLevel::Error,
            &format!("Error in {}: {}", context, error),
        )?;
        self.stats_collector.increment_errors(1)?;
        Ok(())
    }

    /// Log token usage information
    pub fn log_token_usage(
        &mut self,
        node: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
        total_tokens: usize,
    ) -> Result<()> {
        self.log(
            LogLevel::Info,
            &format!(
                "Token usage for node '{}': prompt={}, completion={}, total={}",
                node, prompt_tokens, completion_tokens, total_tokens
            ),
        )?;
        Ok(())
    }

    /// Get statistics collector
    pub fn stats(&self) -> &StatisticsCollector {
        &self.stats_collector
    }

    /// Generate and log final summary
    pub fn log_final_summary(&mut self) -> Result<()> {
        self.stats_collector.update_processing_time()?;
        let stats = self.stats_collector.get_stats()?;

        let summary = format!(
            "Processing Summary:\n\
             - Total nodes processed: {}\n\
             - Total batches processed: {}\n\
             - Entities processed: {}\n\
             - Events processed: {}\n\
             - Relations processed: {}\n\
             - Unique concepts generated: {}\n\
             - Errors encountered: {}\n\
             - Processing time: {}ms\n\
             - Concepts by type: {:?}",
            stats.total_nodes_processed,
            stats.total_batches_processed,
            stats.entities_processed,
            stats.events_processed,
            stats.relations_processed,
            stats.unique_concepts_generated,
            stats.errors_encountered,
            stats.processing_time_ms,
            stats.concepts_by_type
        );

        self.log(LogLevel::Info, &summary)?;
        println!("{}", summary); // Also print to stdout

        Ok(())
    }
}

/// Log levels for different types of messages
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warning => write!(f, "WARNING"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Performance metrics collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_nodes_per_second: f64,
    pub average_batch_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization_percent: f64,
    pub error_rate_percent: f64,
}

impl PerformanceMetrics {
    pub fn calculate(stats: &Statistics) -> Self {
        let processing_time_seconds = stats.processing_time_ms as f64 / 1000.0;
        let throughput = if processing_time_seconds > 0.0 {
            stats.total_nodes_processed as f64 / processing_time_seconds
        } else {
            0.0
        };

        let average_batch_time = if stats.total_batches_processed > 0 {
            stats.processing_time_ms as f64 / stats.total_batches_processed as f64
        } else {
            0.0
        };

        let error_rate = if stats.total_nodes_processed > 0 {
            (stats.errors_encountered as f64 / stats.total_nodes_processed as f64) * 100.0
        } else {
            0.0
        };

        Self {
            throughput_nodes_per_second: throughput,
            average_batch_time_ms: average_batch_time,
            memory_usage_mb: Self::get_memory_usage_mb(),
            cpu_utilization_percent: Self::get_cpu_utilization_percent(),
            error_rate_percent: error_rate,
        }
    }

    fn get_memory_usage_mb() -> f64 {
        // Simplified memory usage estimation
        // In production, you might use system APIs or crates like `sysinfo`
        0.0
    }

    fn get_cpu_utilization_percent() -> f64 {
        // Simplified CPU utilization estimation
        // In production, you might use system APIs or crates like `sysinfo`
        0.0
    }
}

/// Export statistics to JSON file
pub fn export_statistics_to_json(
    stats: &Statistics,
    output_file: impl AsRef<Path>,
) -> Result<()> {
    let json_data = serde_json::to_string_pretty(stats)?;
    std::fs::write(output_file, json_data)?;
    Ok(())
}

/// Export performance metrics to JSON file
pub fn export_performance_metrics_to_json(
    metrics: &PerformanceMetrics,
    output_file: impl AsRef<Path>,
) -> Result<()> {
    let json_data = serde_json::to_string_pretty(metrics)?;
    std::fs::write(output_file, json_data)?;
    Ok(())
}

/// Print statistics in a formatted table
pub fn print_statistics_table(stats: &Statistics) {
    println!("\n╔══════════════════════════════════════════╗");
    println!("║            Processing Statistics         ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║ Total Nodes Processed: {:>16} ║", stats.total_nodes_processed);
    println!("║ Total Batches Processed: {:>14} ║", stats.total_batches_processed);
    println!("║ Entities Processed: {:>19} ║", stats.entities_processed);
    println!("║ Events Processed: {:>21} ║", stats.events_processed);
    println!("║ Relations Processed: {:>18} ║", stats.relations_processed);
    println!("║ Unique Concepts Generated: {:>12} ║", stats.unique_concepts_generated);
    println!("║ Errors Encountered: {:>19} ║", stats.errors_encountered);
    println!("║ Processing Time (ms): {:>17} ║", stats.processing_time_ms);
    println!("╚══════════════════════════════════════════╝");

    if !stats.concepts_by_type.is_empty() {
        println!("\n╔══════════════════════════════════════════╗");
        println!("║           Concepts by Type               ║");
        println!("╠══════════════════════════════════════════╣");
        for (concept_type, count) in &stats.concepts_by_type {
            println!("║ {:<25}: {:>12} ║", concept_type, count);
        }
        println!("╚══════════════════════════════════════════╝");
    }
}

/// Calculate and print performance summary
pub fn print_performance_summary(stats: &Statistics) {
    let metrics = PerformanceMetrics::calculate(stats);

    println!("\n╔══════════════════════════════════════════╗");
    println!("║          Performance Metrics             ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║ Throughput (nodes/sec): {:>15.2} ║", metrics.throughput_nodes_per_second);
    println!("║ Avg Batch Time (ms): {:>18.2} ║", metrics.average_batch_time_ms);
    println!("║ Error Rate (%): {:>24.2} ║", metrics.error_rate_percent);
    println!("║ Memory Usage (MB): {:>21.2} ║", metrics.memory_usage_mb);
    println!("║ CPU Utilization (%): {:>19.2} ║", metrics.cpu_utilization_percent);
    println!("╚══════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_statistics_collector() {
        let collector = StatisticsCollector::new();

        collector.increment_nodes_processed(10).unwrap();
        collector.increment_batches_processed(2).unwrap();
        collector.increment_nodes_by_type(NodeType::Entity, 5).unwrap();
        collector.add_concepts_for_type("entity".to_string(), 15).unwrap();

        let stats = collector.get_stats().unwrap();
        assert_eq!(stats.total_nodes_processed, 10);
        assert_eq!(stats.total_batches_processed, 2);
        assert_eq!(stats.entities_processed, 5);
        assert_eq!(stats.unique_concepts_generated, 15);
    }

    #[test]
    fn test_performance_metrics() {
        let mut stats = Statistics::default();
        stats.total_nodes_processed = 1000;
        stats.total_batches_processed = 10;
        stats.processing_time_ms = 5000;
        stats.errors_encountered = 5;

        let metrics = PerformanceMetrics::calculate(&stats);
        assert_eq!(metrics.throughput_nodes_per_second, 200.0);
        assert_eq!(metrics.average_batch_time_ms, 500.0);
        assert_eq!(metrics.error_rate_percent, 0.5);
    }

    #[test]
    fn test_export_statistics_to_json() {
        let stats = Statistics::default();
        let temp_file = NamedTempFile::new().unwrap();

        export_statistics_to_json(&stats, temp_file.path()).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("total_nodes_processed"));
        assert!(content.contains("total_batches_processed"));
    }

    #[test]
    fn test_processing_logger() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut logger = ProcessingLogger::new(temp_file.path()).unwrap();

        logger.log(LogLevel::Info, "Test message").unwrap();
        logger.log_batch_start(NodeType::Entity, 10).unwrap();
        logger.log_batch_complete(NodeType::Entity, 10, 20, 1000).unwrap();

        let stats = logger.stats().get_stats().unwrap();
        assert_eq!(stats.total_batches_processed, 1);
        assert_eq!(stats.entities_processed, 10);
    }
}