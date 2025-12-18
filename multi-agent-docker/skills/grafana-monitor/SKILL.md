---
name: grafana-monitor
version: 1.0.0
description: Observability stack integration - Grafana dashboards, Prometheus metrics, Loki logs
author: agentic-workstation
tags: [grafana, prometheus, loki, metrics, observability, monitoring]
mcp_server: true
---

# Grafana Observability Stack

Complete observability stack integration for monitoring, metrics, and logs through Grafana, Prometheus, and Loki.

## Overview

This skill provides comprehensive access to your observability infrastructure:
- **Grafana**: Dashboard management, alerting, and data source configuration
- **Prometheus**: Metrics querying (PromQL), targets, and alert rules
- **Loki**: Log aggregation and querying (LogQL)

## Tools

### Grafana Dashboard Management

**grafana_dashboards**
```python
grafana_dashboards(folder: str = None)
```
List all dashboards, optionally filtered by folder.

Example:
```python
# List all dashboards
dashboards = grafana_dashboards()

# List dashboards in specific folder
prod_dashboards = grafana_dashboards(folder="Production")
```

**grafana_dashboard_get**
```python
grafana_dashboard_get(uid: str)
```
Get complete dashboard JSON definition by UID.

Example:
```python
# Get dashboard configuration
dashboard = grafana_dashboard_get(uid="abc123xyz")
```

**grafana_search**
```python
grafana_search(query: str)
```
Search dashboards, folders, and panels.

Example:
```python
# Search for Kubernetes dashboards
results = grafana_search(query="kubernetes")
```

### Grafana Alerting

**grafana_alerts**
```python
grafana_alerts(state: str = None)
```
List alert rules. State options: 'alerting', 'ok', 'pending', 'nodata', 'error'.

Example:
```python
# Get all firing alerts
firing = grafana_alerts(state="alerting")

# Get all alert rules
all_alerts = grafana_alerts()
```

**grafana_alert_status**
```python
grafana_alert_status()
```
Get current alert states across all rules.

Example:
```python
# Check overall alerting status
status = grafana_alert_status()
```

### Grafana Data Sources

**grafana_datasources**
```python
grafana_datasources()
```
List all configured data sources.

Example:
```python
# Get all data sources
datasources = grafana_datasources()
```

### Prometheus Metrics

**prom_query**
```python
prom_query(query: str, time: str = None)
```
Execute instant PromQL query. Time format: RFC3339 or Unix timestamp.

Example:
```python
# Get current CPU usage
cpu = prom_query(query='rate(cpu_usage_seconds[5m])')

# Query at specific time
historical = prom_query(
    query='node_memory_MemAvailable_bytes',
    time='2024-01-15T10:30:00Z'
)
```

**prom_query_range**
```python
prom_query_range(
    query: str,
    start: str,
    end: str,
    step: str = "1m"
)
```
Execute range PromQL query for time series data.

Example:
```python
# Get CPU usage over last hour
cpu_range = prom_query_range(
    query='rate(cpu_usage_seconds[5m])',
    start='2024-01-15T10:00:00Z',
    end='2024-01-15T11:00:00Z',
    step='1m'
)
```

**prom_series**
```python
prom_series(match: str)
```
List time series matching selector.

Example:
```python
# Find all CPU metrics
series = prom_series(match='cpu_usage_seconds')

# Find metrics with labels
series = prom_series(match='{job="kubernetes",namespace="prod"}')
```

**prom_labels**
```python
prom_labels(label: str = None)
```
List label names or values for specific label.

Example:
```python
# Get all label names
labels = prom_labels()

# Get values for specific label
namespaces = prom_labels(label='namespace')
```

**prom_targets**
```python
prom_targets()
```
Get status of all Prometheus scrape targets.

Example:
```python
# Check target health
targets = prom_targets()
unhealthy = [t for t in targets if t['health'] != 'up']
```

**prom_alerts**
```python
prom_alerts()
```
Get active Prometheus alerts.

Example:
```python
# Get firing alerts
alerts = prom_alerts()
```

### Loki Logs

**loki_query**
```python
loki_query(
    query: str,
    limit: int = 100,
    since: str = "1h"
)
```
Execute LogQL query. Since format: duration string (1h, 30m, 1d).

Example:
```python
# Get recent errors
errors = loki_query(
    query='{job="app"} |= "error"',
    limit=50,
    since="2h"
)

# Filter and parse logs
logs = loki_query(
    query='{namespace="prod"} | json | level="error"',
    limit=100,
    since="30m"
)
```

**loki_labels**
```python
loki_labels()
```
List all available log labels.

Example:
```python
# Get available labels
labels = loki_labels()
```

**loki_series**
```python
loki_series(match: str)
```
List log streams matching selector.

Example:
```python
# Find all app logs
series = loki_series(match='{job="app"}')
```

## Query Examples

### Prometheus PromQL Examples

**CPU and Memory**
```promql
# CPU usage rate
rate(node_cpu_seconds_total{mode="user"}[5m])

# Memory usage percentage
100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))

# Top 5 processes by CPU
topk(5, rate(process_cpu_seconds_total[5m]))
```

**HTTP Metrics**
```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Requests per second by status code
sum by (status) (rate(http_requests_total[5m]))
```

**Kubernetes Metrics**
```promql
# Pod CPU usage
sum by (pod) (rate(container_cpu_usage_seconds_total[5m]))

# Pod memory usage
sum by (pod) (container_memory_working_set_bytes)

# Pods not running
count(kube_pod_status_phase{phase!="Running"}) by (namespace)

# Node capacity
sum(kube_node_status_capacity) by (resource)
```

**System Health**
```promql
# Disk usage percentage
100 * (1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes))

# Network traffic
rate(node_network_receive_bytes_total[5m])
rate(node_network_transmit_bytes_total[5m])

# System load
node_load1
node_load5
node_load15
```

### Loki LogQL Examples

**Basic Filtering**
```logql
# All logs from job
{job="app"}

# Logs containing text
{job="app"} |= "error"

# Logs NOT containing text
{job="app"} != "debug"

# Multiple filters
{job="app"} |= "error" != "timeout"
```

**Parsing and Filtering**
```logql
# Parse JSON logs
{job="app"} | json

# Filter parsed field
{job="app"} | json | level="error"

# Parse and extract
{job="app"} | json | user_id="123" | status="failed"

# Regex parsing
{job="app"} | regexp "(?P<status>\\d{3})"
```

**Aggregations**
```logql
# Count logs
count_over_time({job="app"}[5m])

# Rate of errors
rate({job="app"} |= "error" [5m])

# Bytes processed
sum(rate({job="app"} [5m])) by (pod)

# 95th percentile response time
quantile_over_time(0.95, {job="app"} | json | unwrap duration [5m])
```

**Advanced Queries**
```logql
# Errors by endpoint
sum by (endpoint) (
  count_over_time({job="api"} | json | status>=500 [5m])
)

# Top error messages
topk(10,
  sum by (error_message) (
    count_over_time({job="app"} |= "error" [1h])
  )
)

# Slow requests
{job="api"} | json | duration > 1000
```

## Setup Requirements

### Environment Variables

Required environment variables:

```bash
# Grafana Configuration
export GRAFANA_URL="https://grafana.example.com"
export GRAFANA_API_KEY="your-api-key"

# Prometheus Configuration
export PROMETHEUS_URL="http://prometheus:9090"

# Loki Configuration
export LOKI_URL="http://loki:3100"
```

### Grafana API Key

Create API key in Grafana:
1. Go to Configuration > API Keys
2. Create new key with appropriate permissions
3. Copy key to GRAFANA_API_KEY environment variable

Recommended permissions:
- Viewer: Read dashboards, alerts, datasources
- Editor: Create/modify dashboards
- Admin: Full access including user management

### Network Access

Ensure network connectivity to:
- Grafana API endpoint (typically port 3000)
- Prometheus HTTP API (typically port 9090)
- Loki HTTP API (typically port 3100)

### Service Discovery

For Kubernetes environments:
```yaml
# Access via service names
GRAFANA_URL=http://grafana.monitoring.svc.cluster.local:3000
PROMETHEUS_URL=http://prometheus.monitoring.svc.cluster.local:9090
LOKI_URL=http://loki.monitoring.svc.cluster.local:3100
```

## Common Workflows

### Dashboard Analysis
```python
# Find dashboard
results = grafana_search(query="production")
dashboard = grafana_dashboard_get(uid=results[0]['uid'])

# Check datasources
datasources = grafana_datasources()
prom_ds = [ds for ds in datasources if ds['type'] == 'prometheus'][0]
```

### Alert Investigation
```python
# Check firing alerts
alerts = grafana_alerts(state="alerting")

# Get Prometheus alert details
prom_alerts = prom_alerts()

# Query relevant metrics
for alert in alerts:
    metrics = prom_query(query=alert['query'])
```

### Performance Analysis
```python
# Query metrics
cpu = prom_query_range(
    query='rate(cpu_usage[5m])',
    start='1h ago',
    end='now',
    step='1m'
)

# Get related logs
logs = loki_query(
    query='{job="app"} | json | level="error"',
    limit=100,
    since="1h"
)
```

### Capacity Planning
```python
# Check targets
targets = prom_targets()

# Query resource usage
memory = prom_query('node_memory_MemAvailable_bytes')
disk = prom_query('node_filesystem_avail_bytes')

# Analyze trends
usage_trend = prom_query_range(
    query='rate(resource_usage[1h])',
    start='7d ago',
    end='now',
    step='1h'
)
```

## Best Practices

1. **Query Optimization**
   - Use specific label selectors
   - Limit time ranges for large queries
   - Use recording rules for complex queries
   - Cache frequently accessed data

2. **Alert Management**
   - Check alert status regularly
   - Correlate Grafana and Prometheus alerts
   - Use appropriate severity levels
   - Document alert runbooks

3. **Log Analysis**
   - Use structured logging (JSON)
   - Apply appropriate filters
   - Limit query ranges
   - Use aggregations for high-volume logs

4. **Dashboard Organization**
   - Use folders for categorization
   - Follow naming conventions
   - Document dashboard purpose
   - Share dashboards appropriately

5. **Security**
   - Use least-privilege API keys
   - Rotate API keys regularly
   - Restrict data source access
   - Audit dashboard access

## References

- [Grafana API Documentation](https://grafana.com/docs/grafana/latest/http_api/)
- [Prometheus Query Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [LogQL Documentation](https://grafana.com/docs/loki/latest/logql/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
