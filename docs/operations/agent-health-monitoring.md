# Agent Health Monitoring Guide

**Version**: 1.0.0
**Last Updated**: 2025-09-17
**Component**: Multi-Agent System Monitoring

This guide provides comprehensive procedures for monitoring agent health, detecting performance issues, and maintaining optimal system operation in the VisionFlow WebXR multi-agent environment.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Health Metrics](#health-metrics)
3. [Real-time Monitoring](#real-time-monitoring)
4. [Automated Alerts](#automated-alerts)
5. [Performance Analysis](#performance-analysis)
6. [Troubleshooting Procedures](#troubleshooting-procedures)
7. [Best Practices](#best-practices)

---

## Overview

Agent health monitoring encompasses the continuous observation and analysis of multi-agent system performance, including individual agent status, coordination efficiency, and system-wide metrics. The monitoring system provides early warning indicators and automated recovery mechanisms.

### Key Components Monitored

- **Agent Lifecycle**: Spawn, execution, termination events
- **Performance Metrics**: CPU usage, memory consumption, execution times
- **Position Tracking**: Agent positions and clustering detection
- **Coordination Health**: Inter-agent communication and synchronisation
- **Error Patterns**: Failures, recovery attempts, and success rates
- **Resource Utilisation**: GPU, memory, and network usage

---

## Health Metrics

### Core Agent Health Indicators

#### 1. Agent Status Metrics

```bash
# Monitor active agent count
grep -c "Agent spawned successfully" /app/logs/server.log

# Check agent termination rate
grep -c "Agent terminated" /app/logs/server.log

# Calculate agent uptime distribution
jq 'select(.metadata.agent_id? and .message | contains("Agent spawned")) |
    {agent: .metadata.agent_id, spawn_time: .timestamp}' /app/logs/server.log
```

#### 2. Performance Health Scores

```bash
# Agent health scores (0-100%)
jq 'select(.metadata.health? and .metadata.agent_id?) |
    {agent: .metadata.agent_id, health: .metadata.health, time: .timestamp}' \
    /app/logs/analytics.log | tail -20

# Low health alerts (< 50%)
jq 'select(.metadata.health? and (.metadata.health | tonumber) < 50)' \
   /app/logs/analytics.log
```

#### 3. Resource Utilisation Metrics

```bash
# CPU usage monitoring
jq 'select(.metadata.cpu_usage? and .metadata.agent_id?) |
    {agent: .metadata.agent_id, cpu: .metadata.cpu_usage}' \
    /app/logs/analytics.log

# Memory usage tracking
jq 'select(.metadata.memory_usage? and .metadata.agent_id?) |
    {agent: .metadata.agent_id, memory_mb: .metadata.memory_usage}' \
    /app/logs/memory.log
```

### Health Score Calculation

The system calculates composite health scores based on multiple factors:

```python
def calculate_agent_health_score(agent_data):
    """
    Calculate comprehensive health score (0-100)
    """
    factors = {
        'cpu_efficiency': min(100, max(0, 100 - agent_data.cpu_usage)),
        'memory_efficiency': min(100, max(0, 100 - (agent_data.memory_usage / agent_data.memory_limit * 100))),
        'task_success_rate': agent_data.successful_tasks / max(1, agent_data.total_tasks) * 100,
        'response_time': min(100, max(0, 100 - (agent_data.avg_response_time / 1000))),
        'error_rate': max(0, 100 - (agent_data.errors / max(1, agent_data.operations) * 100))
    }

    # Weighted average
    weights = {
        'cpu_efficiency': 0.2,
        'memory_efficiency': 0.2,
        'task_success_rate': 0.3,
        'response_time': 0.2,
        'error_rate': 0.1
    }

    return sum(factors[k] * weights[k] for k in factors)
```

---

## Real-time Monitoring

### Live Dashboard Queries

#### Agent Status Dashboard

```bash
#!/bin/bash
# agent_status_dashboard.sh - Real-time agent status monitoring

while true; do
    clear
    echo "=== Agent Health Dashboard ==="
    echo "$(date)"
    echo

    # Active agents count
    active_agents=$(jq 'select(.message | contains("Agent spawned"))' /app/logs/server.log | \
                   jq -s 'group_by(.metadata.agent_id) | length')
    echo "Active Agents: $active_agents"

    # Health distribution
    echo
    echo "Health Distribution:"
    jq 'select(.metadata.health?)' /app/logs/analytics.log | tail -100 | \
    jq '.metadata.health | tonumber' | \
    awk '{
        if ($1 >= 80) good++
        else if ($1 >= 60) fair++
        else if ($1 >= 40) poor++
        else critical++
    }
    END {
        print "  Excellent (80-100%): " (good+0)
        print "  Fair (60-79%):       " (fair+0)
        print "  Poor (40-59%):       " (poor+0)
        print "  Critical (0-39%):    " (critical+0)
    }'

    # Recent alerts
    echo
    echo "Recent Alerts (Last 10 minutes):"
    recent_time=$(date -d '10 minutes ago' -Iseconds)
    jq --arg since "$recent_time" \
       'select(.timestamp >= $since and (.level == "WARN" or .level == "ERROR"))' \
       /app/logs/*.log 2>/dev/null | \
    jq -r '"\(.timestamp) [\(.level)] \(.component): \(.message)"' | tail -5

    sleep 30
done
```

#### Position Monitoring Dashboard

```bash
#!/bin/bash
# position_monitor.sh - Agent position clustering monitoring

echo "=== Agent Position Monitor ==="

# Check for origin clustering
clustering_incidents=$(jq 'select(.metadata.origin_cluster_detected == true)' \
                      /app/logs/analytics.log | wc -l)
echo "Total Clustering Incidents: $clustering_incidents"

# Recent position fixes
fixes_applied=$(jq 'select(.message | contains("Position fix applied"))' \
               /app/logs/analytics.log | tail -10 | wc -l)
echo "Recent Position Fixes: $fixes_applied"

# Current agent positions
echo
echo "Current Agent Positions:"
jq 'select(.metadata.position_x?) |
    {agent: .metadata.agent_id,
     pos: [.metadata.position_x, .metadata.position_y, .metadata.position_z],
     time: .timestamp}' \
   /app/logs/analytics.log | tail -10 | \
jq -r '"\(.agent): (\(.pos[0] | tonumber | . * 100 | round / 100), \
                     \(.pos[1] | tonumber | . * 100 | round / 100), \
                     \(.pos[2] | tonumber | . * 100 | round / 100))"'
```

### Performance Monitoring Scripts

#### GPU Performance Monitor

```bash
#!/bin/bash
# gpu_performance_monitor.sh - Monitor GPU performance metrics

echo "=== GPU Performance Monitor ==="

# GPU kernel performance
echo "Top 10 Slowest Kernels (Last Hour):"
recent_time=$(date -d '1 hour ago' -Iseconds)
jq --arg since "$recent_time" \
   'select(.timestamp >= $since and .gpu_metrics.execution_time_us?) |
    {kernel: .gpu_metrics.kernel_name, time_us: .gpu_metrics.execution_time_us}' \
   /app/logs/gpu.log | \
jq -s 'sort_by(.time_us | tonumber) | reverse | .[0:10]' | \
jq -r '.[] | "\(.kernel): \(.time_us)μs"'

# Memory usage trends
echo
echo "GPU Memory Usage:"
jq 'select(.gpu_metrics.memory_peak_mb?) |
    {time: .timestamp, memory: .gpu_metrics.memory_peak_mb}' \
   /app/logs/gpu.log | tail -5 | \
jq -r '"\(.time): \(.memory)MB"'

# Error rates
echo
echo "GPU Errors (Last 24h):"
error_count=$(jq 'select(.gpu_metrics.error_count? and (.gpu_metrics.error_count | tonumber) > 0)' \
             /app/logs/gpu.log | wc -l)
echo "Total GPU Errors: $error_count"

# Performance anomalies
anomaly_count=$(jq 'select(.gpu_metrics.performance_anomaly == true)' /app/logs/gpu.log | wc -l)
echo "Performance Anomalies: $anomaly_count"
```

---

## Automated Alerts

### Alert Configuration

#### Health Threshold Alerts

```bash
#!/bin/bash
# health_alert_system.sh - Automated health alerting

ALERT_EMAIL="admin@example.com"
LOG_DIR="/app/logs"

# Function to send alert
send_alert() {
    local subject="$1"
    local message="$2"
    echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
    logger "ALERT SENT: $subject"
}

# Monitor critical health scores
check_critical_health() {
    critical_agents=$(jq 'select(.metadata.health? and (.metadata.health | tonumber) < 30)' \
                     "$LOG_DIR/analytics.log" | tail -10)

    if [ -n "$critical_agents" ]; then
        agent_count=$(echo "$critical_agents" | jq -s 'length')
        send_alert "Critical Agent Health Alert" \
                  "Found $agent_count agents with health < 30%:\n$critical_agents"
    fi
}

# Monitor position clustering
check_position_clustering() {
    recent_time=$(date -d '15 minutes ago' -Iseconds)
    clustering_incidents=$(jq --arg since "$recent_time" \
                          'select(.timestamp >= $since and .metadata.origin_cluster_detected == true)' \
                          "$LOG_DIR/analytics.log" | wc -l)

    if [ "$clustering_incidents" -gt 5 ]; then
        send_alert "Position Clustering Alert" \
                  "High clustering incidents: $clustering_incidents in last 15 minutes"
    fi
}

# Monitor GPU errors
check_gpu_errors() {
    recent_time=$(date -d '10 minutes ago' -Iseconds)
    gpu_errors=$(jq --arg since "$recent_time" \
                'select(.timestamp >= $since and .level == "ERROR" and .component == "gpu")' \
                "$LOG_DIR/gpu.log" | wc -l)

    if [ "$gpu_errors" -gt 3 ]; then
        send_alert "GPU Error Alert" \
                  "High GPU error rate: $gpu_errors errors in last 10 minutes"
    fi
}

# Monitor agent spawn failures
check_spawn_failures() {
    recent_time=$(date -d '5 minutes ago' -Iseconds)
    spawn_failures=$(jq --arg since "$recent_time" \
                    'select(.timestamp >= $since and .message | contains("Agent spawn failed"))' \
                    "$LOG_DIR/error.log" | wc -l)

    if [ "$spawn_failures" -gt 2 ]; then
        send_alert "Agent Spawn Failure Alert" \
                  "Multiple agent spawn failures: $spawn_failures in last 5 minutes"
    fi
}

# Run checks
echo "Running health checks..."
check_critical_health
check_position_clustering
check_gpu_errors
check_spawn_failures
```

#### System Resource Alerts

```bash
#!/bin/bash
# resource_alert_system.sh - Monitor system resource usage

# Memory usage alert
check_memory_usage() {
    memory_usage=$(free | awk '/^Mem:/{printf("%.0f", $3/$2*100)}')
    if [ "$memory_usage" -gt 85 ]; then
        send_alert "High Memory Usage Alert" \
                  "System memory usage at ${memory_usage}%"
    fi
}

# Disk space alert
check_disk_space() {
    disk_usage=$(df /app/logs | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 80 ]; then
        send_alert "Low Disk Space Alert" \
                  "Log directory disk usage at ${disk_usage}%"
    fi
}

# Agent count monitoring
check_agent_count() {
    active_count=$(jq 'select(.message | contains("Agent spawned"))' /app/logs/server.log | \
                  jq -s 'group_by(.metadata.agent_id) | length')
    expected_count=10  # Adjust based on your configuration

    if [ "$active_count" -lt $(( expected_count / 2 )) ]; then
        send_alert "Low Agent Count Alert" \
                  "Only $active_count agents active, expected around $expected_count"
    fi
}

# Run resource checks
check_memory_usage
check_disk_space
check_agent_count
```

### Alert Integration with Monitoring Tools

#### Prometheus Metrics Export

```bash
#!/bin/bash
# prometheus_metrics.sh - Export metrics for Prometheus

METRICS_FILE="/tmp/agent_metrics.prom"

# Agent health metrics
echo "# HELP agent_health_score Agent health score (0-100)" > "$METRICS_FILE"
echo "# TYPE agent_health_score gauge" >> "$METRICS_FILE"
jq 'select(.metadata.health? and .metadata.agent_id?) |
    "agent_health_score{agent_id=\"\(.metadata.agent_id)\"} \(.metadata.health)"' \
   /app/logs/analytics.log | tail -50 >> "$METRICS_FILE"

# GPU performance metrics
echo "# HELP gpu_kernel_execution_time_us GPU kernel execution time in microseconds" >> "$METRICS_FILE"
echo "# TYPE gpu_kernel_execution_time_us gauge" >> "$METRICS_FILE"
jq 'select(.gpu_metrics.execution_time_us? and .gpu_metrics.kernel_name?) |
    "gpu_kernel_execution_time_us{kernel=\"\(.gpu_metrics.kernel_name)\"} \(.gpu_metrics.execution_time_us)"' \
   /app/logs/gpu.log | tail -100 >> "$METRICS_FILE"

# Position clustering metrics
echo "# HELP position_clustering_incidents_total Total position clustering incidents" >> "$METRICS_FILE"
echo "# TYPE position_clustering_incidents_total counter" >> "$METRICS_FILE"
clustering_total=$(jq 'select(.metadata.origin_cluster_detected == true)' /app/logs/analytics.log | wc -l)
echo "position_clustering_incidents_total $clustering_total" >> "$METRICS_FILE"

echo "Metrics exported to $METRICS_FILE"
```

---

## Performance Analysis

### Trend Analysis Scripts

#### Agent Performance Trends

```bash
#!/bin/bash
# performance_trends.sh - Analyse agent performance over time

echo "=== Agent Performance Trend Analysis ==="

# Calculate performance trends over the last 24 hours
analyse_health_trends() {
    echo "Health Score Trends (Last 24 Hours):"

    # Generate hourly averages
    for hour in {0..23}; do
        hour_ago=$(date -d "$hour hours ago" +"%Y-%m-%d %H")
        avg_health=$(jq --arg hour "$hour_ago" \
                    'select(.timestamp | startswith($hour) and .metadata.health?) |
                     .metadata.health | tonumber' /app/logs/analytics.log | \
                    awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        printf "%s: %.1f%%\n" "$hour_ago" "$avg_health"
    done
}

# GPU performance trends
analyse_gpu_trends() {
    echo
    echo "GPU Performance Trends:"

    # Average execution times by kernel
    jq 'select(.gpu_metrics.execution_time_us? and .gpu_metrics.kernel_name?) |
        {kernel: .gpu_metrics.kernel_name, time: .gpu_metrics.execution_time_us}' \
       /app/logs/gpu.log | \
    jq -s 'group_by(.kernel) |
           map({kernel: .[0].kernel,
                avg_time: (map(.time | tonumber) | add / length),
                count: length})' | \
    jq -r '.[] | "\(.kernel): \(.avg_time | . * 100 | round / 100)μs avg (\(.count) samples)"'
}

# Position fix effectiveness
analyse_position_trends() {
    echo
    echo "Position Fix Effectiveness:"

    clustering_detected=$(jq 'select(.metadata.origin_cluster_detected == true)' /app/logs/analytics.log | wc -l)
    fixes_applied=$(jq 'select(.message | contains("Position fix applied successfully"))' /app/logs/analytics.log | wc -l)

    if [ "$clustering_detected" -gt 0 ]; then
        success_rate=$(( fixes_applied * 100 / clustering_detected ))
        echo "Clustering incidents: $clustering_detected"
        echo "Successful fixes: $fixes_applied"
        echo "Fix success rate: ${success_rate}%"
    else
        echo "No clustering incidents detected"
    fi
}

# Run analysis
analyse_health_trends | head -6  # Show last 6 hours
analyse_gpu_trends
analyse_position_trends
```

#### Resource Utilisation Analysis

```bash
#!/bin/bash
# resource_analysis.sh - Analyse system resource utilisation patterns

echo "=== Resource Utilisation Analysis ==="

# Memory usage patterns
analyse_memory_patterns() {
    echo "Memory Usage Patterns:"

    # Peak memory usage by agent
    jq 'select(.metadata.memory_usage? and .metadata.agent_id?) |
        {agent: .metadata.agent_id, memory: (.metadata.memory_usage | tonumber)}' \
       /app/logs/memory.log | \
    jq -s 'group_by(.agent) |
           map({agent: .[0].agent,
                peak_memory: (map(.memory) | max),
                avg_memory: (map(.memory) | add / length)})' | \
    jq -r '.[] | "\(.agent): Peak \(.peak_memory)MB, Avg \(.avg_memory | . * 100 | round / 100)MB"'
}

# CPU usage patterns
analyse_cpu_patterns() {
    echo
    echo "CPU Usage Patterns:"

    jq 'select(.metadata.cpu_usage? and .metadata.agent_id?) |
        {agent: .metadata.agent_id, cpu: (.metadata.cpu_usage | tonumber)}' \
       /app/logs/analytics.log | \
    jq -s 'group_by(.agent) |
           map({agent: .[0].agent,
                max_cpu: (map(.cpu) | max),
                avg_cpu: (map(.cpu) | add / length)})' | \
    jq -r '.[] | "\(.agent): Peak \(.max_cpu)%, Avg \(.avg_cpu | . * 100 | round / 100)%"'
}

# Network activity analysis
analyse_network_patterns() {
    echo
    echo "Network Activity:"

    network_events=$(jq 'select(.component == "network")' /app/logs/network.log | wc -l)
    network_errors=$(jq 'select(.component == "network" and .level == "ERROR")' /app/logs/network.log | wc -l)

    if [ "$network_events" -gt 0 ]; then
        error_rate=$(( network_errors * 100 / network_events ))
        echo "Total network events: $network_events"
        echo "Network errors: $network_errors"
        echo "Error rate: ${error_rate}%"
    else
        echo "No network activity recorded"
    fi
}

# Run analysis
analyse_memory_patterns | head -10
analyse_cpu_patterns | head -10
analyse_network_patterns
```

---

## Troubleshooting Procedures

### Common Health Issues

#### Issue 1: Low Agent Health Scores

**Diagnostic Steps:**
```bash
# Identify agents with consistently low health
jq 'select(.metadata.health? and (.metadata.health | tonumber) < 60) |
    {agent: .metadata.agent_id, health: .metadata.health, time: .timestamp}' \
   /app/logs/analytics.log | tail -20

# Check for resource constraints
jq 'select(.metadata.cpu_usage? and (.metadata.cpu_usage | tonumber) > 80)' \
   /app/logs/analytics.log

# Look for error patterns
grep -E "(timeout|failure|error)" /app/logs/server.log | grep -i agent
```

**Resolution Steps:**
1. Restart problematic agents
2. Check resource allocation
3. Verify network connectivity
4. Review agent configuration

#### Issue 2: Frequent Position Clustering

**Diagnostic Steps:**
```bash
# Check clustering frequency
clustering_rate=$(jq 'select(.metadata.origin_cluster_detected == true)' /app/logs/analytics.log | \
                 wc -l)
echo "Clustering incidents: $clustering_rate"

# Verify fix effectiveness
fix_success_rate=$(jq 'select(.message | contains("Position fix applied successfully"))' \
                  /app/logs/analytics.log | wc -l)
echo "Successful fixes: $fix_success_rate"
```

**Resolution Steps:**
1. Increase position dispersion range
2. Add jitter to position calculations
3. Implement validation after fixes
4. Monitor position update frequency

#### Issue 3: GPU Performance Degradation

**Diagnostic Steps:**
```bash
# Check for GPU errors
jq 'select(.gpu_metrics.error_count? and (.gpu_metrics.error_count | tonumber) > 0)' \
   /app/logs/gpu.log | tail -10

# Monitor kernel execution times
jq 'select(.gpu_metrics.execution_time_us?) | .gpu_metrics.execution_time_us' \
   /app/logs/gpu.log | sort -n | tail -20
```

**Resolution Steps:**
1. Clear GPU memory
2. Restart GPU services
3. Check driver compatibility
4. Reduce batch sizes

### Health Recovery Procedures

#### Automated Recovery Script

```bash
#!/bin/bash
# health_recovery.sh - Automated health recovery procedures

RECOVERY_LOG="/app/logs/recovery.log"

log_recovery() {
    echo "[$(date)] $1" | tee -a "$RECOVERY_LOG"
}

# Restart unhealthy agents
restart_unhealthy_agents() {
    log_recovery "Starting agent health recovery..."

    # Find agents with health < 40%
    unhealthy_agents=$(jq 'select(.metadata.health? and (.metadata.health | tonumber) < 40) |
                          .metadata.agent_id' /app/logs/analytics.log | \
                      sort | uniq | head -5)

    for agent_id in $unhealthy_agents; do
        log_recovery "Restarting unhealthy agent: $agent_id"
        # Send restart signal to agent (implementation depends on your agent system)
        curl -X POST "http://localhost:3000/api/agents/$agent_id/restart" || \
        log_recovery "Failed to restart agent $agent_id"
    done
}

# Clear GPU memory if errors detected
gpu_memory_recovery() {
    gpu_errors=$(jq 'select(.gpu_metrics.error_count? and (.gpu_metrics.error_count | tonumber) > 0)' \
                /app/logs/gpu.log | tail -10 | wc -l)

    if [ "$gpu_errors" -gt 5 ]; then
        log_recovery "High GPU errors detected, clearing GPU memory..."
        # Implementation depends on your GPU management system
        curl -X POST "http://localhost:3000/api/gpu/clear-memory" || \
        log_recovery "Failed to clear GPU memory"
    fi
}

# Position clustering recovery
position_recovery() {
    clustering_incidents=$(jq 'select(.metadata.origin_cluster_detected == true)' \
                          /app/logs/analytics.log | tail -20 | wc -l)

    if [ "$clustering_incidents" -gt 10 ]; then
        log_recovery "High clustering detected, triggering position redistribution..."
        curl -X POST "http://localhost:3000/api/agents/redistribute-positions" || \
        log_recovery "Failed to redistribute positions"
    fi
}

# Run recovery procedures
restart_unhealthy_agents
gpu_memory_recovery
position_recovery

log_recovery "Health recovery procedures completed"
```

---

## Best Practices

### Monitoring Configuration

1. **Baseline Establishment**: Establish performance baselines during normal operation
2. **Threshold Tuning**: Regularly review and adjust alert thresholds
3. **Trend Analysis**: Focus on trends rather than isolated incidents
4. **Correlation Analysis**: Look for patterns across different metrics
5. **Proactive Monitoring**: Address issues before they impact users

### Performance Optimisation

1. **Resource Allocation**: Ensure adequate resources for peak loads
2. **Load Balancing**: Distribute work evenly across agents
3. **Memory Management**: Regular cleanup and garbage collection
4. **Network Optimisation**: Minimise inter-agent communication overhead
5. **GPU Efficiency**: Optimise kernel execution and memory usage

### Alerting Strategy

1. **Severity Levels**: Classify alerts by urgency and impact
2. **Alert Fatigue Prevention**: Avoid excessive notifications
3. **Escalation Procedures**: Define clear escalation paths
4. **Recovery Automation**: Implement automatic recovery where safe
5. **Documentation**: Maintain runbooks for common scenarios

---

**Last Updated**: 2025-09-17
**Guide Version**: 1.0.0
**System Compatibility**: VisionFlow WebXR v2.0.0+