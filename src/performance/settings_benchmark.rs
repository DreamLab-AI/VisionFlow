// Performance Benchmarking System for Settings Optimization
// Measures and compares old bulk fetch vs new path-based performance

use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use log::{info, warn, debug};
use actix::prelude::*;
use futures::future::join_all;

use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, GetSettingByPath, GetSettingsByPaths, SetSettingsByPaths};
use crate::actors::settings_actor::SettingsActor;
use crate::actors::optimized_settings_actor::{OptimizedSettingsActor, PerformanceMetrics};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub duration_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub bandwidth_bytes: u64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub old_system: BenchmarkSuite,
    pub new_system: BenchmarkSuite,
    pub improvements: ImprovementMetrics,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub single_path_fetch: BenchmarkResult,
    pub batch_path_fetch: BenchmarkResult,
    pub full_settings_fetch: BenchmarkResult,
    pub concurrent_reads: BenchmarkResult,
    pub write_operations: BenchmarkResult,
    pub cache_performance: BenchmarkResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    pub response_time_improvement: f64, 
    pub throughput_improvement: f64,    
    pub bandwidth_savings: f64,         
    pub memory_efficiency: f64,         
    pub overall_score: f64,            
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub iterations: usize,
    pub concurrent_requests: usize,
    pub warmup_iterations: usize,
    pub test_paths: Vec<String>,
    pub batch_sizes: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            concurrent_requests: 50,
            warmup_iterations: 100,
            test_paths: vec![
                "visualisation.graphs.logseq.physics.damping".to_string(),
                "visualisation.graphs.logseq.physics.spring_k".to_string(),
                "visualisation.graphs.logseq.physics.repel_k".to_string(),
                "visualisation.graphs.logseq.physics.max_velocity".to_string(),
                "visualisation.graphs.logseq.physics.gravity".to_string(),
                "visualisation.graphs.logseq.physics.temperature".to_string(),
                "visualisation.graphs.logseq.physics.bounds_size".to_string(),
                "visualisation.graphs.logseq.physics.iterations".to_string(),
                "visualisation.graphs.logseq.physics.enabled".to_string(),
            ],
            batch_sizes: vec![1, 5, 10, 25, 50],
        }
    }
}

pub struct SettingsBenchmark {
    config: BenchmarkConfig,
}

impl SettingsBenchmark {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    
    pub async fn run_comparison_benchmark(
        &self,
        old_actor_addr: Addr<SettingsActor>,
        new_actor_addr: Addr<OptimizedSettingsActor>,
    ) -> Result<ComparisonReport, String> {
        info!("Starting comprehensive settings performance benchmark");

        
        self.warmup_system(&old_actor_addr, &new_actor_addr).await?;

        
        let old_suite = self.benchmark_old_system(old_actor_addr).await?;

        
        let new_suite = self.benchmark_new_system(new_actor_addr).await?;

        
        let improvements = self.calculate_improvements(&old_suite, &new_suite);

        
        let recommendations = self.generate_recommendations(&improvements);

        let report = ComparisonReport {
            old_system: old_suite,
            new_system: new_suite,
            improvements,
            recommendations,
        };

        self.print_benchmark_report(&report);

        Ok(report)
    }

    async fn warmup_system(
        &self,
        old_addr: &Addr<SettingsActor>,
        new_addr: &Addr<OptimizedSettingsActor>,
    ) -> Result<(), String> {
        info!("Warming up benchmark systems...");

        for _ in 0..self.config.warmup_iterations {
            
            let _ = old_addr.send(GetSettings).await;

            
            let _ = new_addr.send(GetSettings).await;

            
            for path in &self.config.test_paths[..3] {
                let _ = old_addr.send(GetSettingByPath { path: path.clone() }).await;
                let _ = new_addr.send(GetSettingByPath { path: path.clone() }).await;
            }
        }

        info!("Warmup completed");
        Ok(())
    }

    async fn benchmark_old_system(&self, addr: Addr<SettingsActor>) -> Result<BenchmarkSuite, String> {
        info!("Benchmarking old settings system...");

        Ok(BenchmarkSuite {
            single_path_fetch: self.benchmark_single_path_old(&addr).await?,
            batch_path_fetch: self.benchmark_batch_path_old(&addr).await?,
            full_settings_fetch: self.benchmark_full_settings_old(&addr).await?,
            concurrent_reads: self.benchmark_concurrent_reads_old(&addr).await?,
            write_operations: self.benchmark_write_ops_old(&addr).await?,
            cache_performance: self.benchmark_cache_old(&addr).await?,
        })
    }

    async fn benchmark_new_system(&self, addr: Addr<OptimizedSettingsActor>) -> Result<BenchmarkSuite, String> {
        info!("Benchmarking optimized settings system...");

        Ok(BenchmarkSuite {
            single_path_fetch: self.benchmark_single_path_new(&addr).await?,
            batch_path_fetch: self.benchmark_batch_path_new(&addr).await?,
            full_settings_fetch: self.benchmark_full_settings_new(&addr).await?,
            concurrent_reads: self.benchmark_concurrent_reads_new(&addr).await?,
            write_operations: self.benchmark_write_ops_new(&addr).await?,
            cache_performance: self.benchmark_cache_new(&addr).await?,
        })
    }

    async fn benchmark_single_path_old(&self, addr: &Addr<SettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;
        let mut total_bytes = 0u64;

        for _ in 0..self.config.iterations {
            let path = &self.config.test_paths[successful_ops % self.config.test_paths.len()];

            match addr.send(GetSettingByPath { path: path.clone() }).await {
                Ok(Ok(value)) => {
                    successful_ops += 1;
                    total_bytes += self.estimate_payload_size(&value);
                }
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        Ok(BenchmarkResult {
            test_name: "Single Path Fetch (Old)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: 0.0, 
            bandwidth_bytes: total_bytes,
            cache_hit_rate: 0.0,   
            error_rate: ((self.config.iterations - successful_ops) as f64 / self.config.iterations as f64) * 100.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_single_path_new(&self, addr: &Addr<OptimizedSettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;
        let mut total_bytes = 0u64;

        for _ in 0..self.config.iterations {
            let path = &self.config.test_paths[successful_ops % self.config.test_paths.len()];

            match addr.send(GetSettingByPath { path: path.clone() }).await {
                Ok(Ok(value)) => {
                    successful_ops += 1;
                    total_bytes += self.estimate_payload_size(&value);
                }
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        
        let metrics = addr.send(crate::actors::optimized_settings_actor::GetPerformanceMetrics)
            .await
            .unwrap_or_default();

        Ok(BenchmarkResult {
            test_name: "Single Path Fetch (Optimized)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: (metrics.memory_usage_bytes as f64) / (1024.0 * 1024.0),
            bandwidth_bytes: total_bytes,
            cache_hit_rate: metrics.cache_hit_rate(),
            error_rate: ((self.config.iterations - successful_ops) as f64 / self.config.iterations as f64) * 100.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_batch_path_old(&self, addr: &Addr<SettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;
        let mut total_bytes = 0u64;
        let batch_size = 10;

        for i in 0..(self.config.iterations / batch_size) {
            let paths: Vec<String> = (0..batch_size)
                .map(|j| self.config.test_paths[(i + j) % self.config.test_paths.len()].clone())
                .collect();

            match addr.send(GetSettingsByPaths { paths: paths.clone() }).await {
                Ok(Ok(results)) => {
                    successful_ops += results.len();
                    for value in results.values() {
                        total_bytes += self.estimate_payload_size(value);
                    }
                }
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        Ok(BenchmarkResult {
            test_name: "Batch Path Fetch (Old)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: 0.0,
            bandwidth_bytes: total_bytes,
            cache_hit_rate: 0.0,
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_batch_path_new(&self, addr: &Addr<OptimizedSettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;
        let mut total_bytes = 0u64;
        let batch_size = 10;

        for i in 0..(self.config.iterations / batch_size) {
            let paths: Vec<String> = (0..batch_size)
                .map(|j| self.config.test_paths[(i + j) % self.config.test_paths.len()].clone())
                .collect();

            match addr.send(GetSettingsByPaths { paths: paths.clone() }).await {
                Ok(Ok(results)) => {
                    successful_ops += results.len();
                    for value in results.values() {
                        total_bytes += self.estimate_payload_size(value);
                    }
                }
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        let metrics = addr.send(crate::actors::optimized_settings_actor::GetPerformanceMetrics)
            .await
            .unwrap_or_default();

        Ok(BenchmarkResult {
            test_name: "Batch Path Fetch (Optimized)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: (metrics.memory_usage_bytes as f64) / (1024.0 * 1024.0),
            bandwidth_bytes: total_bytes,
            cache_hit_rate: metrics.cache_hit_rate(),
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_full_settings_old(&self, addr: &Addr<SettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;
        let mut total_bytes = 0u64;

        for _ in 0..100 { 
            match addr.send(GetSettings).await {
                Ok(Ok(settings)) => {
                    successful_ops += 1;
                    
                    total_bytes += 50_000;
                }
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        Ok(BenchmarkResult {
            test_name: "Full Settings Fetch (Old)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: 0.0,
            bandwidth_bytes: total_bytes,
            cache_hit_rate: 0.0,
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_full_settings_new(&self, addr: &Addr<OptimizedSettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;
        let mut total_bytes = 0u64;

        for _ in 0..100 {
            match addr.send(GetSettings).await {
                Ok(Ok(settings)) => {
                    successful_ops += 1;
                    total_bytes += 50_000; 
                }
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        let metrics = addr.send(crate::actors::optimized_settings_actor::GetPerformanceMetrics)
            .await
            .unwrap_or_default();

        Ok(BenchmarkResult {
            test_name: "Full Settings Fetch (Optimized)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: (metrics.memory_usage_bytes as f64) / (1024.0 * 1024.0),
            bandwidth_bytes: total_bytes,
            cache_hit_rate: metrics.cache_hit_rate(),
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_concurrent_reads_old(&self, addr: &Addr<SettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut handles = Vec::new();

        
        for _ in 0..self.config.concurrent_requests {
            let addr_clone = addr.clone();
            let path = self.config.test_paths[0].clone();

            let handle = tokio::spawn(async move {
                addr_clone.send(GetSettingByPath { path }).await
            });

            handles.push(handle);
        }

        
        let results = join_all(handles).await;
        let successful_ops = results.into_iter()
            .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
            .count();

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        Ok(BenchmarkResult {
            test_name: "Concurrent Reads (Old)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: 0.0,
            bandwidth_bytes: successful_ops as u64 * 500, 
            cache_hit_rate: 0.0,
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_concurrent_reads_new(&self, addr: &Addr<OptimizedSettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut handles = Vec::new();

        for _ in 0..self.config.concurrent_requests {
            let addr_clone = addr.clone();
            let path = self.config.test_paths[0].clone();

            let handle = tokio::spawn(async move {
                addr_clone.send(GetSettingByPath { path }).await
            });

            handles.push(handle);
        }

        let results = join_all(handles).await;
        let successful_ops = results.into_iter()
            .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
            .count();

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        let metrics = addr.send(crate::actors::optimized_settings_actor::GetPerformanceMetrics)
            .await
            .unwrap_or_default();

        Ok(BenchmarkResult {
            test_name: "Concurrent Reads (Optimized)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: (metrics.memory_usage_bytes as f64) / (1024.0 * 1024.0),
            bandwidth_bytes: successful_ops as u64 * 500,
            cache_hit_rate: metrics.cache_hit_rate(),
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_write_ops_old(&self, addr: &Addr<SettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;

        let updates: HashMap<String, Value> = self.config.test_paths.iter()
            .map(|path| (path.clone(), serde_json::Value::Number(serde_json::Number::from_f64(0.5).unwrap())))
            .collect();

        for _ in 0..10 { 
            match addr.send(SetSettingsByPaths { updates: updates.clone() }).await {
                Ok(Ok(_)) => successful_ops += updates.len(),
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        Ok(BenchmarkResult {
            test_name: "Write Operations (Old)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: 0.0,
            bandwidth_bytes: 0,
            cache_hit_rate: 0.0,
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_write_ops_new(&self, addr: &Addr<OptimizedSettingsActor>) -> Result<BenchmarkResult, String> {
        let start_time = Instant::now();
        let mut successful_ops = 0;

        let updates: HashMap<String, Value> = self.config.test_paths.iter()
            .map(|path| (path.clone(), serde_json::Value::Number(serde_json::Number::from_f64(0.5).unwrap())))
            .collect();

        for _ in 0..10 {
            match addr.send(SetSettingsByPaths { updates: updates.clone() }).await {
                Ok(Ok(_)) => successful_ops += updates.len(),
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        let metrics = addr.send(crate::actors::optimized_settings_actor::GetPerformanceMetrics)
            .await
            .unwrap_or_default();

        Ok(BenchmarkResult {
            test_name: "Write Operations (Optimized)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: (metrics.memory_usage_bytes as f64) / (1024.0 * 1024.0),
            bandwidth_bytes: 0,
            cache_hit_rate: metrics.cache_hit_rate(),
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_cache_old(&self, _addr: &Addr<SettingsActor>) -> Result<BenchmarkResult, String> {
        
        Ok(BenchmarkResult {
            test_name: "Cache Performance (Old - No Cache)".to_string(),
            duration_ms: 0.0,
            throughput_ops_per_sec: 0.0,
            memory_usage_mb: 0.0,
            bandwidth_bytes: 0,
            cache_hit_rate: 0.0,
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    async fn benchmark_cache_new(&self, addr: &Addr<OptimizedSettingsActor>) -> Result<BenchmarkResult, String> {
        
        for path in &self.config.test_paths {
            let _ = addr.send(GetSettingByPath { path: path.clone() }).await;
        }

        let start_time = Instant::now();
        let mut successful_ops = 0;

        
        for _ in 0..self.config.iterations {
            let path = &self.config.test_paths[successful_ops % self.config.test_paths.len()];

            match addr.send(GetSettingByPath { path: path.clone() }).await {
                Ok(Ok(_)) => successful_ops += 1,
                _ => {}
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        let metrics = addr.send(crate::actors::optimized_settings_actor::GetPerformanceMetrics)
            .await
            .unwrap_or_default();

        Ok(BenchmarkResult {
            test_name: "Cache Performance (Optimized)".to_string(),
            duration_ms,
            throughput_ops_per_sec: (successful_ops as f64 / duration.as_secs_f64()),
            memory_usage_mb: (metrics.memory_usage_bytes as f64) / (1024.0 * 1024.0),
            bandwidth_bytes: metrics.bandwidth_saved_bytes,
            cache_hit_rate: metrics.cache_hit_rate(),
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
        })
    }

    fn calculate_improvements(&self, old: &BenchmarkSuite, new: &BenchmarkSuite) -> ImprovementMetrics {
        let response_time_improvement = self.calculate_percentage_improvement(
            old.single_path_fetch.duration_ms,
            new.single_path_fetch.duration_ms
        );

        let throughput_improvement = self.calculate_percentage_improvement(
            new.single_path_fetch.throughput_ops_per_sec,
            old.single_path_fetch.throughput_ops_per_sec
        ) - 100.0; 

        let bandwidth_savings = self.calculate_bandwidth_savings(old, new);

        let memory_efficiency = if old.single_path_fetch.memory_usage_mb > 0.0 {
            self.calculate_percentage_improvement(
                old.single_path_fetch.memory_usage_mb,
                new.single_path_fetch.memory_usage_mb
            )
        } else {
            0.0
        };

        
        let overall_score = (response_time_improvement * 0.3) +
                           (throughput_improvement * 0.3) +
                           (bandwidth_savings * 0.25) +
                           (memory_efficiency * 0.15);

        ImprovementMetrics {
            response_time_improvement,
            throughput_improvement,
            bandwidth_savings,
            memory_efficiency,
            overall_score,
        }
    }

    fn calculate_percentage_improvement(&self, old_value: f64, new_value: f64) -> f64 {
        if old_value == 0.0 {
            return 0.0;
        }
        ((old_value - new_value) / old_value) * 100.0
    }

    fn calculate_bandwidth_savings(&self, old: &BenchmarkSuite, new: &BenchmarkSuite) -> f64 {
        let old_bandwidth = old.single_path_fetch.bandwidth_bytes + old.batch_path_fetch.bandwidth_bytes;
        let new_bandwidth = new.single_path_fetch.bandwidth_bytes + new.batch_path_fetch.bandwidth_bytes;

        if old_bandwidth == 0 {
            return 0.0;
        }

        ((old_bandwidth as f64 - new_bandwidth as f64) / old_bandwidth as f64) * 100.0
    }

    fn generate_recommendations(&self, improvements: &ImprovementMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        if improvements.response_time_improvement > 50.0 {
            recommendations.push("✅ Excellent response time improvement achieved".to_string());
        } else if improvements.response_time_improvement > 20.0 {
            recommendations.push("✅ Good response time improvement".to_string());
        } else {
            recommendations.push("⚠️ Consider further caching optimizations".to_string());
        }

        if improvements.bandwidth_savings > 90.0 {
            recommendations.push("✅ Outstanding bandwidth savings (>90%)".to_string());
        } else if improvements.bandwidth_savings > 70.0 {
            recommendations.push("✅ Excellent bandwidth savings".to_string());
        } else {
            recommendations.push("💡 Consider implementing compression for larger payloads".to_string());
        }

        if improvements.throughput_improvement > 100.0 {
            recommendations.push("✅ Throughput more than doubled".to_string());
        } else if improvements.throughput_improvement > 50.0 {
            recommendations.push("✅ Significant throughput improvement".to_string());
        } else {
            recommendations.push("💡 Consider connection pooling or async optimizations".to_string());
        }

        recommendations.push(format!("📊 Overall performance score: {:.1}/100", improvements.overall_score));

        recommendations
    }

    fn print_benchmark_report(&self, report: &ComparisonReport) {
        info!("=== SETTINGS PERFORMANCE BENCHMARK REPORT ===");
        info!("");
        info!("📈 Performance Improvements:");
        info!("  Response Time: {:.1}% faster", report.improvements.response_time_improvement);
        info!("  Throughput: {:.1}% improvement", report.improvements.throughput_improvement);
        info!("  Bandwidth Savings: {:.1}%", report.improvements.bandwidth_savings);
        info!("  Memory Efficiency: {:.1}% improvement", report.improvements.memory_efficiency);
        info!("  Overall Score: {:.1}/100", report.improvements.overall_score);
        info!("");
        info!("📋 Detailed Results:");
        info!("  Old System - Single Path: {:.2}ms avg, {:.0} ops/sec",
              report.old_system.single_path_fetch.duration_ms / self.config.iterations as f64,
              report.old_system.single_path_fetch.throughput_ops_per_sec);
        info!("  New System - Single Path: {:.2}ms avg, {:.0} ops/sec",
              report.new_system.single_path_fetch.duration_ms / self.config.iterations as f64,
              report.new_system.single_path_fetch.throughput_ops_per_sec);
        info!("  Cache Hit Rate: {:.1}%", report.new_system.single_path_fetch.cache_hit_rate);
        info!("");
        info!("🔍 Recommendations:");
        for rec in &report.recommendations {
            info!("  {}", rec);
        }
        info!("===============================================");
    }

    fn estimate_payload_size(&self, value: &Value) -> u64 {
        serde_json::to_string(value)
            .map(|s| s.len() as u64)
            .unwrap_or(500) 
    }
}

// Helper function to run benchmark from command line or API
pub async fn run_performance_benchmark(
    old_actor: Addr<SettingsActor>,
    new_actor: Addr<OptimizedSettingsActor>,
) -> Result<ComparisonReport, String> {
    let config = BenchmarkConfig::default();
    let benchmark = SettingsBenchmark::new(config);

    benchmark.run_comparison_benchmark(old_actor, new_actor).await
}