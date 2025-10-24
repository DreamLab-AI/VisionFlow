# Monitoring & Observability

## Metrics

VisionFlow exposes Prometheus metrics at `/metrics`.

### Application Metrics

- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration
- `db_connections_active`: Active database connections
- `processing_jobs_total`: Total processing jobs
- `processing_jobs_duration_seconds`: Job duration

### System Metrics

- `process_cpu_user_seconds_total`: CPU usage
- `process_resident_memory_bytes`: Memory usage
- `nodejs_heap_size_used_bytes`: Heap usage

## Prometheus Configuration

**prometheus.yml**:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'visionflow'
    static_configs:
      - targets: ['localhost:9090']
```

## Grafana Dashboards

Import VisionFlow dashboard: [Dashboard JSON]

Key panels:
- Request rate
- Error rate
- Response time
- Resource usage

## Logging

### Log Levels

- `error`: Errors requiring attention
- `warn`: Warning messages
- `info`: Informational messages
- `debug`: Debug information

### Log Format

```json
{
  "timestamp": "2025-01-23T10:00:00Z",
  "level": "info",
  "message": "Request completed",
  "context": {
    "method": "GET",
    "path": "/api/projects",
    "duration": 45,
    "userId": "uuid"
  }
}
```

### Centralized Logging

Ship logs to ELK stack:

```yaml
logging:
  elasticsearch:
    host: elasticsearch:9200
    index: visionflow-logs
```

## Alerting

### Alert Rules

**alerts.yml**:

```yaml
groups:
  - name: visionflow
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: High error rate detected

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes > 1e9
        for: 10m
        annotations:
          summary: High memory usage
```

### Notification Channels

- Email
- Slack
- PagerDuty
- Webhooks

## Health Checks

```bash
# Application health
curl http://localhost:9090/health

# Detailed status
curl http://localhost:9090/health/detailed
```

Response:
```json
{
  "status": "healthy",
  "uptime": 3600,
  "components": {
    "database": "connected",
    "redis": "connected",
    "storage": "operational"
  }
}
```
