# GPU Analytics Logging Infrastructure Architecture

## Overview

The GPU Analytics system implements a comprehensive, multi-layered logging infrastructure designed for high-performance monitoring, debugging, and analytics. The system captures detailed metrics from GPU kernels, memory operations, system performance, and error conditions.

## Architecture Components

### 1. Advanced Logging System (`advanced_logging.rs`)

**Core Features:**
- **Structured JSON Logging**: All logs are structured as JSON for easy parsing and analysis
- **Component-based Separation**: Logs are segregated by component (GPU, server, client, etc.)
- **Automatic Log Rotation**: Prevents disk overflow with configurable rotation policies
- **Real-time Performance Metrics**: Tracks kernel execution times with statistical analysis
- **Anomaly Detection**: Automatically flags performance anomalies based on statistical deviation

**Key Structures:**
```rust
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub component: String, 
    pub message: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub execution_time_ms: Option<f64>,
    pub memory_usage_mb: Option<f64>,
    pub gpu_metrics: Option<GPULogMetrics>,
}

pub struct GPULogMetrics {
    pub kernel_name: Option<String>,
    pub execution_time_us: Option<f64>,
    pub memory_allocated_mb: Option<f64>,
    pub memory_peak_mb: Option<f64>,
    pub gpu_utilization_percent: Option<f32>,
    pub error_count: Option<u32>,
    pub recovery_attempts: Option<u32>,
    pub performance_anomaly: Option<bool>,
}
```

### 2. Log Components

The system separates logs into the following components:

| Component | Purpose | Log File | Key Metrics |
|-----------|---------|----------|-------------|
| **GPU** | CUDA kernel execution, memory operations | `gpu.log` | Kernel timing, memory usage, error rates |
| **Server** | HTTP server operations, API calls | `server.log` | Request/response times, error rates |
| **Client** | WebGL rendering, user interactions | `client.log` | Frame rates, interaction latency |
| **Analytics** | Data processing, clustering, anomaly detection | `analytics.log` | Algorithm performance, accuracy metrics |
| **Memory** | Memory allocation/deallocation events | `memory.log` | Memory usage patterns, leak detection |
| **Network** | Network communication, WebSocket events | `network.log` | Bandwidth usage, connection stability |
| **Performance** | System-wide performance metrics | `performance.log` | CPU usage, throughput measurements |
| **Error** | All error conditions across components | `error.log` | Error rates, failure patterns |

### 3. GPU-Specific Logging Integration

**Kernel Performance Tracking:**
- CUDA event-based timing for microsecond precision
- Rolling averages for trending analysis
- Per-kernel statistics (min/max/avg execution times)
- Performance anomaly detection using 3-sigma thresholds

**Memory Management Logging:**
- Real-time memory usage calculation
- Peak memory tracking
- Memory allocation/deallocation events
- Memory leak detection through trend analysis

**Error Recovery Logging:**
- GPU failure detection and logging
- Recovery attempt tracking
- Fallback mechanism activation logging

### 4. Log Rotation and Management

**Rotation Configuration:**
```rust
pub struct LogRotationConfig {
    pub max_file_size_mb: u64,      // Default: 50MB
    pub max_files: usize,           // Default: 10 files  
    pub compress_rotated: bool,     // Default: true
    pub rotation_interval_hours: u64, // Default: 24 hours
}
```

**Features:**
- Automatic rotation when file size exceeds threshold
- Timestamp-based archived file naming
- Automatic cleanup of old log files
- Compressed storage for archived logs

## Directory Structure

```
ext/logs/
├── gpu.log                    # Current GPU logs
├── server.log                 # Current server logs  
├── client.log                 # Current client logs
├── analytics.log              # Current analytics logs
├── memory.log                 # Current memory logs
├── network.log                # Current network logs
├── performance.log            # Current performance logs
├── error.log                  # Current error logs
├── archived/                  # Rotated log archives
│   ├── gpu_20250909_143022.log
│   ├── server_20250909_143022.log
│   └── ...
└── aggregated/               # Processed log summaries
    ├── log_analysis_report_20250909_143022.json
    ├── aggregated_logs_20250909_143022.json
    ├── kernel_performance_timeline.png
    └── kernel_performance_distribution.png
```

## Log Aggregation System (`log_aggregator.py`)

### Features

**Data Collection:**
- Collects logs from all components within specified date ranges
- Parses both JSON-structured and legacy text logs
- Handles archived/rotated log files automatically
- Filters by timestamp and component

**GPU Performance Analysis:**
- Kernel execution time statistics
- Memory usage pattern analysis
- Performance anomaly identification
- Error rate calculation by component

**Report Generation:**
- Daily/weekly/monthly summaries
- JSON and CSV export formats
- Statistical analysis of performance trends
- Error pattern identification

**Visualization:**
- Kernel performance timeline charts
- Performance distribution histograms
- Memory usage trend graphs
- Error rate dashboards

### Usage Examples

```bash
# Generate daily summary with charts
python scripts/log_aggregator.py --days 1 --format json

# Generate weekly report without charts
python scripts/log_aggregator.py --days 7 --no-charts --format csv

# Custom output directory
python scripts/log_aggregator.py --log-dir ./logs --output-dir ./reports --days 3
```

## Real-time Monitoring Dashboard (`log_monitor_dashboard.py`)

### Features

**Real-time Monitoring:**
- Live tail of all log files
- Real-time metric updates (1-second intervals)
- GPU kernel performance monitoring
- Memory usage trending
- Error rate tracking

**Interactive Dashboard:**
- Curses-based terminal UI
- ASCII bar charts and sparklines
- Real-time performance trends (last 5 minutes)
- System resource monitoring (CPU, disk usage)

**Key Metrics Displayed:**
- GPU kernel execution times with trends
- Memory allocation/peak usage
- Error counts and recovery attempts
- Performance anomaly detection
- Component-specific error rates

### Dashboard Sections

1. **System Status**
   - Current timestamp and update frequency
   - CPU and disk usage
   - Current memory allocation

2. **GPU Kernel Performance**
   - Per-kernel average execution times
   - Sample count for statistical validity
   - Real-time trend sparklines (last 30 data points)

3. **Error Monitoring**
   - Total error count across all components
   - Recovery attempt statistics
   - Performance anomaly count
   - Error rate breakdown by component

4. **Memory Trends**
   - Current memory usage
   - Peak memory usage
   - Memory usage trend over last 5 minutes

### Usage

```bash
# Start interactive dashboard
python scripts/log_monitor_dashboard.py

# Simple text-only dashboard (no curses)
python scripts/log_monitor_dashboard.py --simple

# Custom update interval
python scripts/log_monitor_dashboard.py --interval 2.0
```

## Integration Points

### 1. GPU Compute Integration

The logging system is tightly integrated with the `UnifiedGPUCompute` module:

```rust
// Kernel timing integration
pub fn record_kernel_time(&mut self, kernel_name: &str, execution_time_ms: f32) {
    // ... internal metrics update ...
    
    // Log to advanced logging system
    let execution_time_us = execution_time_ms * 1000.0;
    let memory_mb = self.performance_metrics.current_memory_usage as f64 / (1024.0 * 1024.0);
    let peak_memory_mb = self.performance_metrics.peak_memory_usage as f64 / (1024.0 * 1024.0);
    
    log_gpu_kernel(kernel_name, execution_time_us as f64, memory_mb, peak_memory_mb);
}

// Memory usage integration
pub fn update_memory_usage(&mut self) {
    // ... memory calculation ...
    
    // Log memory events for significant changes
    if (current_usage as f64 - previous_usage as f64).abs() > (1024.0 * 1024.0) {
        let event_type = if current_usage > previous_usage { "allocation" } else { "deallocation" };
        let allocated_mb = current_usage as f64 / (1024.0 * 1024.0);
        let peak_mb = self.performance_metrics.peak_memory_usage as f64 / (1024.0 * 1024.0);
        log_memory_event(event_type, allocated_mb, peak_mb);
    }
}

// Error logging integration
pub fn log_gpu_error(&self, error_msg: &str, recovery_attempted: bool) {
    log_gpu_error(error_msg, recovery_attempted);
}
```

### 2. Application Initialization

The logging system is initialized during application startup:

```rust
// Initialize standard logging
init_logging()?;

// Initialize advanced structured logging
if let Err(e) = init_advanced_logging() {
    error!("Failed to initialize advanced logging: {}", e);
} else {
    info!("Advanced logging system initialized successfully");
}
```

### 3. Convenience Macros

The system provides convenient macros for structured logging:

```rust
// GPU kernel logging
log_gpu!("kmeans_kernel", 1234.5, 128.0, 256.0);

// Performance logging  
log_perf!("clustering_operation", 45.2);
log_perf!("anomaly_detection", 23.1, 1000.0); // with throughput
```

## Performance Considerations

### 1. Low-Overhead Logging

**Asynchronous Processing:**
- Log entries are processed asynchronously to minimize impact on GPU operations
- Buffered I/O reduces filesystem overhead
- JSON serialization is optimized for performance

**Memory Management:**
- Rolling window approach for metrics (last 100 measurements per kernel)
- Automatic cleanup of old data
- Memory-mapped files for high-frequency logging

### 2. Storage Optimization

**Structured Data:**
- JSON format enables efficient querying and analysis
- Compressed archived logs save storage space
- Automatic purging of old logs prevents disk overflow

**Indexing:**
- Timestamp-based indexing for fast date range queries
- Component-based file separation enables targeted analysis
- Metadata optimization reduces storage overhead

## Monitoring and Alerting

### 1. Performance Anomaly Detection

**Statistical Analysis:**
- 3-sigma deviation threshold for anomaly detection
- Rolling window statistical analysis
- Per-kernel anomaly tracking

**Alerting Mechanisms:**
- Real-time anomaly flagging in logs
- Dashboard visual indicators
- Configurable alert thresholds

### 2. Error Rate Monitoring

**Comprehensive Error Tracking:**
- GPU error count and recovery attempts
- Component-specific error rates
- Error pattern analysis

**Recovery Monitoring:**
- Automatic recovery attempt logging
- Success/failure rate tracking
- Recovery time measurement

## Future Enhancements

### 1. Advanced Analytics
- Machine learning-based anomaly detection
- Predictive performance modeling
- Automatic performance optimization recommendations

### 2. External Integration
- Prometheus/Grafana integration
- ElasticSearch/Kibana support
- Cloud logging service integration

### 3. Enhanced Visualization
- Web-based dashboard
- Real-time 3D performance visualization
- Interactive performance correlation analysis

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_DIR` | `./logs` | Base directory for log files |
| `LOG_LEVEL` | `INFO` | Minimum log level |
| `DEBUG_ENABLED` | `false` | Enable debug mode logging |
| `LOG_ROTATION_SIZE_MB` | `50` | Max file size before rotation |
| `LOG_MAX_FILES` | `10` | Maximum archived files to keep |

### Configuration Files

The logging system respects the existing `settings.yaml` configuration for debug mode and other system settings.

## Troubleshooting

### Common Issues

1. **High Disk Usage**
   - Check log rotation configuration
   - Verify archived log cleanup
   - Monitor log directory size

2. **Missing Log Entries**
   - Verify log level configuration
   - Check file permissions
   - Ensure log directory exists

3. **Performance Impact**
   - Monitor logging overhead
   - Adjust update intervals
   - Consider disabling debug logging in production

### Diagnostic Commands

```bash
# Check log directory structure
ls -la ext/logs/

# Monitor disk usage
du -sh ext/logs/

# Check log file sizes
ls -lh ext/logs/*.log

# Tail specific log file
tail -f ext/logs/gpu.log

# Check for errors in logging system
grep -i error ext/logs/aggregated/aggregator.log
```

This logging architecture provides comprehensive observability into the GPU Analytics system while maintaining high performance and efficient resource usage.