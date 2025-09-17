# Logging System Troubleshooting Guide

**Version**: 1.0.0
**Last Updated**: 2025-09-17
**Component**: Telemetry & Logging System

This guide provides comprehensive troubleshooting procedures for the VisionFlow telemetry and logging system, covering common issues, diagnostic procedures, and resolution strategies.

## 📋 Quick Reference

### Common Issues
- [Log Files Not Created](#log-files-not-created)
- [High Memory Usage](#high-memory-usage-in-logging-system)
- [Position Clustering Not Fixed](#position-clustering-not-resolved)
- [GPU Performance Anomalies](#gpu-performance-anomalies)
- [Log Rotation Failures](#log-rotation-not-working)
- [Docker Volume Issues](#docker-volume-mounting-problems)
- [Performance Degradation](#logging-performance-impact)

### Emergency Procedures
- [Disable Logging](#emergency-disable-logging)
- [Clear Log Storage](#emergency-log-cleanup)
- [Restart Logging System](#restart-logging-subsystem)

---

## Log Files Not Created

### Symptoms
- Missing log files in `/app/logs/` directory
- No logging output visible in expected locations
- Application running but no telemetry data

### Diagnostic Steps

```bash
# 1. Check directory structure and permissions
ls -la /app/logs/
ls -la /app/logs/archived/

# 2. Verify environment variables
env | grep -E "(LOG|RUST_LOG|DEBUG)"

# 3. Check Docker volume mounting
docker inspect container_name | jq '.[0].Mounts'

# 4. Verify logger initialisation
grep -i "logging.* initialized" /app/logs/server.log 2>/dev/null || echo "No init log found"

# 5. Check application startup logs
docker logs container_name | grep -i log

# 6. Verify disk space
df -h /app/logs/
```

### Common Causes and Solutions

#### Cause 1: Incorrect Directory Permissions
```bash
# Diagnosis
ls -la /app/logs/

# Solution
sudo chown -R 1000:1000 /app/logs/
sudo chmod -R 755 /app/logs/
```

#### Cause 2: Missing Environment Variables
```yaml
# Add to docker-compose.yml
environment:
  - LOG_DIR=/app/logs
  - RUST_LOG=info
  - DEBUG_ENABLED=false
```

#### Cause 3: Volume Mount Issues
```yaml
# Correct volume mounting in docker-compose.yml
volumes:
  - ./logs:/app/logs:rw
  # Or use named volume
  - agent-logs:/app/logs

volumes:
  agent-logs:
    driver: local
```

#### Cause 4: Logger Initialisation Failure
```rust
// Check for initialisation in main.rs
use crate::utils::advanced_logging::init_advanced_logging;

match init_advanced_logging() {
    Ok(()) => info!("Advanced logging initialized successfully"),
    Err(e) => eprintln!("Failed to initialize logging: {}", e),
}
```

### Verification Steps
```bash
# 1. Confirm log files are created
ls -la /app/logs/*.log

# 2. Verify initial entries
tail -5 /app/logs/server.log

# 3. Test logging functionality
# (Trigger agent spawn or other logged operation)

# 4. Confirm timestamps are recent
ls -lt /app/logs/*.log | head -5
```

---

## High Memory Usage in Logging System

### Symptoms
- Application memory consumption growing over time
- Performance degradation
- Out of memory errors
- Slow log operations

### Diagnostic Steps

```bash
# 1. Check application memory usage
docker stats container_name

# 2. Monitor performance summary growth
curl -s http://localhost:3000/api/performance/summary | jq 'keys | length'

# 3. Check memory-related logs
jq 'select(.memory_usage_mb? and (.memory_usage_mb | tonumber) > 100)' /app/logs/memory.log

# 4. Monitor log file sizes
ls -lh /app/logs/*.log

# 5. Check for memory leaks in performance tracking
jq 'keys' < <(curl -s http://localhost:3000/api/performance/summary) | wc -l
```

### Memory Leak Detection

```bash
# Monitor performance summary over time
for i in {1..10}; do
    count=$(curl -s http://localhost:3000/api/performance/summary | jq 'keys | length')
    echo "$(date): $count tracked items"
    sleep 60
done

# Check for unbounded metric growth
jq 'to_entries | map(select(.value.sample_count? and (.value.sample_count | tonumber) > 100))' \
   < <(curl -s http://localhost:3000/api/performance/summary)
```

### Solutions

#### Solution 1: Verify Bounded Metrics Collection
```rust
// Ensure metrics are bounded in advanced_logging.rs
if let Some(kernel_metrics) = metrics_guard.get_mut(kernel_name) {
    if kernel_metrics.len() > 100 {
        kernel_metrics.remove(0); // Remove oldest entry
    }
}
```

#### Solution 2: Increase Log Rotation Frequency
```rust
// Adjust rotation configuration
let rotation_config = LogRotationConfig {
    max_file_size_mb: 25,     // Smaller files
    max_files: 5,             // Fewer archived files
    compress_rotated: true,   // Enable compression
    rotation_interval_hours: 6 // More frequent rotation
};
```

#### Solution 3: Reduce Metadata Complexity
```rust
// Avoid large metadata objects
let metadata = HashMap::from([
    ("agent_id".to_string(), json!(agent_id)),
    ("essential_field".to_string(), json!(value))
    // Remove non-essential fields
]);
```

#### Solution 4: Emergency Memory Cleanup
```bash
# Clear performance summary cache
curl -X POST http://localhost:3000/api/performance/clear-cache

# Rotate logs manually
sudo kill -USR1 $(pidof your_application)

# Restart logging system
docker restart container_name
```

---

## Position Clustering Not Resolved

### Symptoms
- Agents remain clustered at origin coordinates
- Position fix logs present but ineffective
- Analytics show clustering detection but no correction

### Diagnostic Steps

```bash
# 1. Check clustering detection logs
jq 'select(.metadata.origin_cluster_detected == true)' /app/logs/analytics.log

# 2. Verify position fixes are applied
jq 'select(.message | contains("Position fix applied"))' /app/logs/analytics.log

# 3. Check corrected positions
jq 'select(.metadata.corrected_position?)' /app/logs/analytics.log | \
   jq '.metadata | {agent_id, original_position, corrected_position}'

# 4. Monitor current agent positions
jq 'select(.metadata.position_x?) |
    {agent: .metadata.agent_id, x: .metadata.position_x, y: .metadata.position_y, z: .metadata.position_z}' \
    /app/logs/analytics.log | tail -20
```

### Analysis Queries

```bash
# Count clustering incidents vs fixes
clustering_detected=$(jq 'select(.metadata.origin_cluster_detected == true)' /app/logs/analytics.log | wc -l)
fixes_applied=$(jq 'select(.message | contains("Position fix applied successfully"))' /app/logs/analytics.log | wc -l)
echo "Clustering detected: $clustering_detected, Fixes applied: $fixes_applied"

# Check fix effectiveness
jq 'select(.metadata.corrected_position?) | .metadata.corrected_position' /app/logs/analytics.log | \
   jq 'fromjson | {x: .[0], y: .[1], z: .[2]}' | \
   jq 'select(.x < 1.0 and .y < 1.0 and .z < 1.0)'
```

### Common Causes and Solutions

#### Cause 1: Insufficient Position Dispersion Range
```rust
// Increase dispersion parameters
let corrected_position = (
    original_x + 10.0 + (agent_index as f64) * 8.0,    // Increased from 5.0
    original_y + 15.0 + (agent_index as f64) * 12.0,   // Increased from 7.0
    original_z + 8.0 + (agent_index as f64) * 6.0      // Increased from 3.0
);
```

#### Cause 2: Position Override After Fix
```rust
// Ensure position fix is not overridden
if !agent.position_fix_applied {
    apply_position_fix(&mut agent);
    agent.position_fix_applied = true;
}
```

#### Cause 3: Detection Threshold Too Strict
```rust
// Adjust clustering detection threshold
let is_origin_cluster = position.x.abs() < 2.0 &&
                       position.y.abs() < 2.0 &&
                       position.z.abs() < 2.0;  // Increased from 1.0
```

#### Solution: Enhanced Position Validation
```rust
fn validate_position_fix(agent_id: &str, original: (f64, f64, f64), corrected: (f64, f64, f64)) {
    let min_distance = 5.0;
    let distance = ((corrected.0 - original.0).powi(2) +
                   (corrected.1 - original.1).powi(2) +
                   (corrected.2 - original.2).powi(2)).sqrt();

    if distance < min_distance {
        warn!("Position fix insufficient for agent {}: distance {}", agent_id, distance);
        // Apply stronger correction
    }
}
```

---

## GPU Performance Anomalies

### Symptoms
- Frequent performance anomaly flags in logs
- GPU operations slower than expected
- High GPU memory usage
- Kernel execution timeouts

### Diagnostic Steps

```bash
# 1. Analyse kernel performance trends
jq 'select(.gpu_metrics.performance_anomaly == true) |
    {time: .timestamp, kernel: .gpu_metrics.kernel_name,
     time_us: .gpu_metrics.execution_time_us, memory_mb: .gpu_metrics.memory_peak_mb}' \
    /app/logs/gpu.log | tail -20

# 2. Check for resource contention
jq 'select(.gpu_metrics.memory_peak_mb? and (.gpu_metrics.memory_peak_mb | tonumber) > 1500)' \
   /app/logs/gpu.log

# 3. Monitor error patterns
jq 'select(.gpu_metrics.error_count? and (.gpu_metrics.error_count | tonumber) > 0)' \
   /app/logs/gpu.log | jq '.gpu_metrics | {kernel: .kernel_name, errors: .error_count}'

# 4. Check GPU utilisation
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

# 5. Review kernel execution patterns
jq 'select(.gpu_metrics.execution_time_us?) | .gpu_metrics.execution_time_us' /app/logs/gpu.log | \
   sort -n | tail -20
```

### Performance Analysis

```bash
# Calculate average execution times per kernel
jq -r 'select(.gpu_metrics.kernel_name? and .gpu_metrics.execution_time_us?) |
       "\(.gpu_metrics.kernel_name) \(.gpu_metrics.execution_time_us)"' /app/logs/gpu.log | \
awk '{kernels[$1] += $2; counts[$1]++}
     END {for (k in kernels) printf "%s: %.2f μs avg\n", k, kernels[k]/counts[k]}'

# Memory usage patterns
jq 'select(.gpu_metrics.memory_peak_mb?) |
    {kernel: .gpu_metrics.kernel_name, peak_mb: .gpu_metrics.memory_peak_mb}' \
    /app/logs/gpu.log | jq -s 'group_by(.kernel) |
    map({kernel: .[0].kernel, avg_memory: (map(.peak_mb | tonumber) | add / length)})'
```

### Solutions

#### Solution 1: Adjust Anomaly Detection Thresholds
```rust
fn detect_performance_anomaly_with_metrics(&self, kernel_name: &str, execution_time_us: f64, metrics: &HashMap<String, Vec<f64>>) -> bool {
    if let Some(kernel_metrics) = metrics.get(kernel_name) {
        if kernel_metrics.len() > 10 {
            let avg: f64 = kernel_metrics.iter().sum::<f64>() / kernel_metrics.len() as f64;
            let variance: f64 = kernel_metrics.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / kernel_metrics.len() as f64;
            let std_dev = variance.sqrt();

            // Increased threshold from 3.0 to 4.0 standard deviations
            execution_time_us > avg + (4.0 * std_dev)
        } else {
            false
        }
    } else {
        false
    }
}
```

#### Solution 2: GPU Memory Management
```rust
// Implement memory cleanup before kernel execution
fn cleanup_gpu_memory() -> Result<(), String> {
    // Force garbage collection
    // Deallocate unused buffers
    // Check available memory
    Ok(())
}

// Use in kernel execution
if available_memory < required_memory {
    cleanup_gpu_memory()?;
}
```

#### Solution 3: Kernel Optimisation
```rust
// Add kernel execution monitoring
let start_time = Instant::now();
let result = execute_gpu_kernel(kernel_name, data);
let execution_time = start_time.elapsed().as_micros() as f64;

// Log with additional context
log_gpu_kernel(
    kernel_name,
    execution_time,
    allocated_memory,
    peak_memory
);

if execution_time > expected_time * 2.0 {
    warn!("Kernel {} took {}μs, expected <{}μs",
          kernel_name, execution_time, expected_time);
}
```

---

## Log Rotation Not Working

### Symptoms
- Log files growing beyond configured size limits
- No archived log files in `archived/` directory
- Disk space issues due to large log files

### Diagnostic Steps

```bash
# 1. Check current file sizes
ls -lh /app/logs/*.log

# 2. Verify rotation configuration
grep -i "rotation" /app/logs/server.log

# 3. Check archived directory
ls -la /app/logs/archived/

# 4. Monitor file growth
watch -n 30 'ls -lh /app/logs/*.log'

# 5. Check available disk space
df -h /app/logs/

# 6. Verify file permissions
ls -la /app/logs/
```

### Manual Rotation Test

```bash
# Test manual rotation
function test_rotation() {
    local component=$1
    local log_file="/app/logs/${component}.log"
    local size_mb=$(stat -c%s "$log_file" 2>/dev/null | awk '{print int($1/1024/1024)}')

    echo "Current ${component}.log size: ${size_mb}MB"

    if [ "$size_mb" -gt 50 ]; then
        echo "File exceeds 50MB limit, should rotate"
    else
        echo "File within limits"
    fi
}

# Test all components
for component in server client gpu analytics memory network performance error; do
    test_rotation $component
done
```

### Solutions

#### Solution 1: Fix Directory Permissions
```bash
# Ensure write permissions for archived directory
sudo chown -R 1000:1000 /app/logs/
sudo chmod -R 755 /app/logs/
sudo chmod 755 /app/logs/archived/
```

#### Solution 2: Manual Rotation
```bash
#!/bin/bash
# manual_rotate.sh - Emergency log rotation script

LOG_DIR="/app/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for component in server client gpu analytics memory network performance error; do
    log_file="${LOG_DIR}/${component}.log"
    if [ -f "$log_file" ] && [ $(stat -c%s "$log_file") -gt 52428800 ]; then  # 50MB
        echo "Rotating $log_file"
        mv "$log_file" "${LOG_DIR}/archived/${component}_${TIMESTAMP}.log"
        touch "$log_file"
        chmod 644 "$log_file"
    fi
done
```

#### Solution 3: Improve Rotation Logic
```rust
fn check_and_rotate_logs(&self, component: LogComponent) {
    let file_path = self.log_dir.join(component.log_file_name());

    if let Ok(metadata) = fs::metadata(&file_path) {
        let size_mb = metadata.len() / (1024 * 1024);

        if size_mb >= self.rotation_config.max_file_size_mb {
            match self.rotate_log_file(component) {
                Ok(()) => info!("Successfully rotated log for {:?}", component),
                Err(e) => error!("Failed to rotate log for {:?}: {}", component, e),
            }
        }
    }
}
```

---

## Docker Volume Mounting Problems

### Symptoms
- Logs not persisting across container restarts
- Permission denied errors when writing logs
- Inconsistent log file locations

### Diagnostic Steps

```bash
# 1. Check container mounts
docker inspect container_name | jq '.[0].Mounts'

# 2. Verify volume exists
docker volume ls | grep logs

# 3. Check host directory permissions (bind mount)
ls -la ./logs/

# 4. Test container access to log directory
docker exec container_name ls -la /app/logs/

# 5. Check for conflicting mount points
docker exec container_name mount | grep logs
```

### Solutions

#### Solution 1: Named Volume (Recommended)
```yaml
# docker-compose.yml
services:
  multi-agent:
    volumes:
      - agent-logs:/app/logs

volumes:
  agent-logs:
    driver: local
```

#### Solution 2: Bind Mount with Correct Permissions
```yaml
# docker-compose.yml
services:
  multi-agent:
    volumes:
      - ./logs:/app/logs:rw
    user: "1000:1000"  # Match host user
```

#### Solution 3: Init Container for Permissions
```yaml
# docker-compose.yml
services:
  log-init:
    image: busybox
    volumes:
      - agent-logs:/app/logs
    command: ["chown", "-R", "1000:1000", "/app/logs"]

  multi-agent:
    depends_on:
      - log-init
    volumes:
      - agent-logs:/app/logs
```

---

## Emergency Procedures

### Emergency Disable Logging

If logging is causing system issues:

```bash
# Method 1: Environment variable
docker exec container_name env LOG_LEVEL=off RUST_LOG=off

# Method 2: Temporary file creation
docker exec container_name touch /tmp/disable_logging

# Method 3: Container restart with logging disabled
docker-compose down
LOGGING_DISABLED=true docker-compose up -d
```

### Emergency Log Cleanup

```bash
#!/bin/bash
# emergency_cleanup.sh - Free up log space immediately

LOG_DIR="/app/logs"

echo "Emergency log cleanup starting..."

# Stop logging temporarily
docker exec container_name pkill -STOP logging_process 2>/dev/null

# Archive current logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${LOG_DIR}/emergency_backup_${TIMESTAMP}"

for log_file in "${LOG_DIR}"/*.log; do
    if [ -f "$log_file" ]; then
        mv "$log_file" "${LOG_DIR}/emergency_backup_${TIMESTAMP}/"
        touch "$log_file"
        chmod 644 "$log_file"
    fi
done

# Clean old archives
find "${LOG_DIR}/archived/" -name "*.log" -mtime +7 -delete

# Resume logging
docker exec container_name pkill -CONT logging_process 2>/dev/null

echo "Emergency cleanup completed. Logs backed up to emergency_backup_${TIMESTAMP}/"
```

### Restart Logging Subsystem

```bash
#!/bin/bash
# restart_logging.sh - Restart logging without full container restart

echo "Restarting logging subsystem..."

# Send signal to reload logging configuration
docker exec container_name kill -USR1 1

# Wait for restart
sleep 5

# Verify logging is working
if docker exec container_name test -w /app/logs/server.log; then
    echo "Logging subsystem restarted successfully"
else
    echo "Logging restart failed, consider full container restart"
fi
```

---

## Prevention and Monitoring

### Automated Monitoring Script

```bash
#!/bin/bash
# monitor_logging.sh - Continuous logging system monitoring

LOG_DIR="/app/logs"
ALERT_EMAIL="admin@example.com"

while true; do
    # Check disk space
    disk_usage=$(df /app/logs | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 85 ]; then
        echo "ALERT: Disk usage at ${disk_usage}%" | mail -s "Log Disk Usage Alert" "$ALERT_EMAIL"
    fi

    # Check log file sizes
    for log_file in "${LOG_DIR}"/*.log; do
        if [ -f "$log_file" ]; then
            size_mb=$(stat -c%s "$log_file" | awk '{print int($1/1024/1024)}')
            if [ "$size_mb" -gt 75 ]; then
                echo "ALERT: $log_file is ${size_mb}MB" | mail -s "Large Log File Alert" "$ALERT_EMAIL"
            fi
        fi
    done

    # Check logging system health
    if ! docker exec container_name test -w /app/logs/server.log; then
        echo "ALERT: Logging system not accessible" | mail -s "Logging System Alert" "$ALERT_EMAIL"
    fi

    sleep 300  # Check every 5 minutes
done
```

### Health Check Integration

```yaml
# docker-compose.yml
services:
  multi-agent:
    healthcheck:
      test: ["CMD", "test", "-w", "/app/logs/server.log"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

---

## Support and Escalation

If issues persist after following this guide:

1. **Collect Diagnostic Information**:
   ```bash
   # Generate diagnostic report
   ./collect_diagnostics.sh > diagnostics_$(date +%Y%m%d_%H%M%S).txt
   ```

2. **Check System Resources**:
   ```bash
   # System resource usage
   docker stats
   df -h
   free -h
   ```

3. **Review Application Logs**:
   ```bash
   # Container logs
   docker logs container_name --tail 100
   ```

4. **Contact Support** with:
   - Diagnostic report
   - System configuration
   - Steps taken to resolve
   - Expected vs actual behaviour

**Last Updated**: 2025-09-17
**Guide Version**: 1.0.0