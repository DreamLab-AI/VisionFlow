# Ontology API Endpoints Reference

**Version**: 1.0.0
**Last Updated**: 2025-10-17
**Base URL**: `http://localhost:8080/api/ontology`

## Overview

The Ontology API provides comprehensive endpoints for loading OWL ontologies, validating knowledge graphs, managing inference rules, and monitoring system health. All endpoints follow RESTful conventions with consistent error handling and response formats.

## Authentication

Currently, ontology endpoints use the same authentication mechanism as the main VisionFlow API:

```http
Authorization: Bearer <jwt_token>
```

Feature flag `ontology_validation` must be enabled in the system configuration.

## Base Response Format

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2025-10-17T12:34:56Z"
}
```

### Error Response
```json
{
  "error": "Description of error",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional context"
  },
  "timestamp": "2025-10-17T12:34:56Z",
  "trace_id": "uuid"
}
```

## Endpoints

### Load Ontology

Load an ontology from content string or file path.

```http
POST /api/ontology/load
```

**Request Body:**
```json
{
  "content": "Prefix(:=<http://example.org/>)\nOntology(...)",
  "format": "functional"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| content | string | Yes | Ontology content or file path |
| format | string | No | "functional" or "owlxml" (default: auto-detect) |

**Response (200 OK):**
```json
{
  "ontology_id": "ontology_abc123def456...",
  "axiom_count": 42
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/ontology/load \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "content": "Prefix(:=<http://example.org/>)\nOntology(<http://example.org/ontology>\nDeclaration(Class(:Person))\nDeclaration(Class(:Company))\nDisjointClasses(:Person :Company))",
    "format": "functional"
  }'
```

**Errors:**
- `400 PARSE_ERROR`: Invalid ontology format or syntax error
- `503 FEATURE_DISABLED`: Ontology validation feature not enabled
- `503 ACTOR_UNAVAILABLE`: Ontology actor not available

---

### Validate Graph Against Ontology

Trigger validation of the current graph against a loaded ontology.

```http
POST /api/ontology/validate
```

**Request Body:**
```json
{
  "ontology_id": "ontology_abc123def456...",
  "mode": "Full",
  "priority": 5,
  "enable_websocket_updates": true,
  "client_id": "client-uuid"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| ontology_id | string | Yes | ID from load ontology response |
| mode | string | No | "Quick", "Full", or "Incremental" (default: "Full") |
| priority | number | No | 1-10, higher is more urgent (default: 5) |
| enable_websocket_updates | boolean | No | Enable real-time progress updates (default: false) |
| client_id | string | No | Client ID for WebSocket subscription |

**Response (202 Accepted):**
```json
{
  "job_id": "job-uuid",
  "status": "queued",
  "estimated_completion": "2025-10-17T12:35:26Z",
  "queue_position": 1,
  "websocket_url": "/api/ontology/ws?client_id=client-uuid"
}
```

**Validation Modes:**

**Quick Mode:**
- Basic constraint checking
- No inference generation
- ~100-500ms for typical graphs
- Best for interactive validation

**Full Mode:**
- Complete constraint validation
- Full inference generation with reasoning
- ~1-3s for typical graphs
- Recommended for production validation

**Incremental Mode:**
- Validates only changed nodes/edges
- Reuses cached validation results
- ~50-200ms per change
- Best for real-time validation

**Example:**
```bash
curl -X POST http://localhost:8080/api/ontology/validate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "ontology_id": "ontology_abc123def456",
    "mode": "Full",
    "priority": 8,
    "enable_websocket_updates": true,
    "client_id": "my-client-id"
  }'
```

**Errors:**
- `400 VALIDATION_FAILED`: Validation process encountered an error
- `404 ONTOLOGY_NOT_FOUND`: Specified ontology_id not found
- `408 TIMEOUT_ERROR`: Validation exceeded timeout limit
- `503 ACTOR_UNAVAILABLE`: Ontology actor not available

---

### Get Validation Report

Retrieve a validation report by ID.

```http
GET /api/ontology/reports/{report_id}
```

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| report_id | string | Yes | Report ID from validation response |

**Response (200 OK):**
```json
{
  "id": "report-uuid",
  "timestamp": "2025-10-17T12:34:56Z",
  "duration_ms": 1523,
  "graph_signature": "hash...",
  "total_triples": 150,
  "violations": [
    {
      "id": "violation-uuid",
      "severity": "Error",
      "rule": "DisjointClasses",
      "message": "Individual person1 cannot be both Person and Company",
      "subject": "http://example.org/person1",
      "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
      "object": "http://example.org/Company",
      "timestamp": "2025-10-17T12:34:56Z"
    }
  ],
  "inferred_triples": [
    {
      "subject": "http://example.org/person1",
      "predicate": "http://example.org/worksFor",
      "object": "http://example.org/company1",
      "is_literal": false,
      "datatype": null,
      "language": null
    }
  ],
  "statistics": {
    "classes_checked": 10,
    "properties_checked": 15,
    "individuals_checked": 50,
    "constraints_evaluated": 3,
    "inference_rules_applied": 4,
    "cache_hits": 12,
    "cache_misses": 3
  }
}
```

**Violation Severity Levels:**
- `Error`: Must be fixed for logical consistency
- `Warning`: Should be reviewed but not critical
- `Info`: Informational notices

**Example:**
```bash
curl http://localhost:8080/api/ontology/reports/report-uuid \
  -H "Authorization: Bearer $TOKEN"
```

**Errors:**
- `404 REPORT_NOT_FOUND`: Report with specified ID not found
- `500 REPORT_RETRIEVAL_FAILED`: Error retrieving report

---

### Get Latest Validation Report

Retrieve the most recent validation report (alternative endpoint).

```http
GET /api/ontology/report?report_id={report_id}
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| report_id | string | No | Specific report ID (returns latest if omitted) |

**Response (200 OK):**
Same as `/api/ontology/reports/{report_id}`

**Example:**
```bash
# Get latest report
curl http://localhost:8080/api/ontology/report \
  -H "Authorization: Bearer $TOKEN"

# Get specific report
curl "http://localhost:8080/api/ontology/report?report_id=report-uuid" \
  -H "Authorization: Bearer $TOKEN"
```

---

### List Loaded Ontologies

Get a list of all loaded ontologies with their metadata.

```http
GET /api/ontology/axioms
```

**Response (200 OK):**
```json
{
  "axioms": [
    {
      "ontology_id": "ontology_abc123",
      "axiom_count": 42,
      "loaded_at": "2025-10-17T12:00:00Z",
      "content_hash": "hash...",
      "ttl_seconds": 3600
    }
  ],
  "count": 1,
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Example:**
```bash
curl http://localhost:8080/api/ontology/axioms \
  -H "Authorization: Bearer $TOKEN"
```

---

### Get Inferred Relationships

Retrieve inferred triples from the latest validation.

```http
GET /api/ontology/inferences
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| ontology_id | string | No | Filter by specific ontology |

**Response (200 OK):**
```json
{
  "report_id": "report-uuid",
  "inferred_count": 5,
  "inferences": [
    {
      "subject": "http://example.org/person1",
      "predicate": "http://example.org/worksFor",
      "object": "http://example.org/company1",
      "is_literal": false,
      "datatype": null,
      "language": null
    }
  ],
  "generated_at": "2025-10-17T12:34:56Z",
  "inference_depth": 3,
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Example:**
```bash
curl "http://localhost:8080/api/ontology/inferences?ontology_id=ontology_abc123" \
  -H "Authorization: Bearer $TOKEN"
```

---

### Apply Inferences to Graph

Generate and optionally apply inferred triples to the graph.

```http
POST /api/ontology/apply
```

**Request Body:**
```json
{
  "rdf_triples": [
    {
      "subject": "http://example.org/person1",
      "predicate": "http://example.org/employedBy",
      "object": "http://example.org/company1",
      "is_literal": false
    }
  ],
  "max_depth": 3,
  "update_graph": true
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| rdf_triples | array | Yes | Array of RDF triple objects |
| max_depth | number | No | Maximum inference depth (default: 3) |
| update_graph | boolean | No | Apply to graph immediately (default: false) |

**Response (200 OK):**
```json
{
  "input_count": 1,
  "inferred_triples": [
    {
      "subject": "http://example.org/company1",
      "predicate": "http://example.org/employs",
      "object": "http://example.org/person1",
      "is_literal": false
    }
  ],
  "processing_time_ms": 45,
  "graph_updated": true
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/ontology/apply \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "rdf_triples": [{
      "subject": "http://example.org/person1",
      "predicate": "http://example.org/employedBy",
      "object": "http://example.org/company1",
      "is_literal": false
    }],
    "max_depth": 3,
    "update_graph": true
  }'
```

**Errors:**
- `400 INFERENCE_FAILED`: Inference generation failed
- `408 TIMEOUT_ERROR`: Inference exceeded timeout

---

### Update Mapping Configuration

Update the ontology-to-graph mapping configuration.

```http
POST /api/ontology/mapping
```

**Request Body:**
```json
{
  "config": {
    "enable_reasoning": true,
    "reasoning_timeout_seconds": 30,
    "enable_inference": true,
    "max_inference_depth": 3,
    "enable_caching": true,
    "cache_ttl_seconds": 3600,
    "validate_cardinality": true,
    "validate_domains_ranges": true,
    "validate_disjoint_classes": true
  },
  "apply_to_all": false
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| config | object | Yes | ValidationConfig object |
| apply_to_all | boolean | No | Apply to all loaded ontologies (default: false) |

**ValidationConfig Fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| enable_reasoning | boolean | true | Enable OWL reasoning |
| reasoning_timeout_seconds | number | 30 | Timeout for reasoning operations |
| enable_inference | boolean | true | Generate inferred triples |
| max_inference_depth | number | 3 | Maximum inference depth |
| enable_caching | boolean | true | Enable result caching |
| cache_ttl_seconds | number | 3600 | Cache time-to-live in seconds |
| validate_cardinality | boolean | true | Check cardinality constraints |
| validate_domains_ranges | boolean | true | Check domain/range constraints |
| validate_disjoint_classes | boolean | true | Check disjoint class violations |

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Mapping configuration updated",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/ontology/mapping \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "config": {
      "enable_reasoning": true,
      "reasoning_timeout_seconds": 60,
      "max_inference_depth": 5,
      "validate_cardinality": true
    }
  }'
```

---

### Get System Health

Get ontology system health metrics and status.

```http
GET /api/ontology/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "health": {
    "loaded_ontologies": 5,
    "cached_reports": 12,
    "validation_queue_size": 2,
    "last_validation": "2025-10-17T12:30:00Z",
    "cache_hit_rate": 0.85,
    "avg_validation_time_ms": 1523.5,
    "active_jobs": 1,
    "memory_usage_mb": 256.7
  },
  "ontology_validation_enabled": true,
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Status Values:**
- `healthy`: System operating normally
- `degraded`: Queue size > 100 or high latency
- `unhealthy`: Critical errors or system unavailable

**Example:**
```bash
curl http://localhost:8080/api/ontology/health \
  -H "Authorization: Bearer $TOKEN" \
  | jq
```

---

### Clear Caches

Clear all ontology and validation caches.

```http
DELETE /api/ontology/cache
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "All caches cleared",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Example:**
```bash
curl -X DELETE http://localhost:8080/api/ontology/cache \
  -H "Authorization: Bearer $TOKEN"
```

**Note:** This operation:
- Clears all loaded ontologies from memory
- Removes cached validation reports
- Resets cache statistics
- Does not affect persistent storage

---

## WebSocket API

### Connection

```
ws://localhost:8080/api/ontology/ws?client_id={client_id}
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| client_id | string | No | Client identifier (auto-generated if omitted) |

**Connection Example:**
```javascript
const clientId = 'my-client-id';
const ws = new WebSocket(`ws://localhost:8080/api/ontology/ws?client_id=${clientId}`);

ws.onopen = () => {
  console.log('Connected to ontology validation stream');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Message:', message);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from validation stream');
};
```

### Message Types

#### Connection Established

Sent immediately after connection.

```json
{
  "type": "connection_established",
  "client_id": "my-client-id",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

#### Validation Progress

Periodic updates during validation.

```json
{
  "type": "validation_progress",
  "job_id": "job-uuid",
  "progress": 0.45,
  "stage": "checking_constraints",
  "message": "Validating domain/range constraints...",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Stages:**
- `parsing_graph`: Converting property graph to RDF
- `loading_ontology`: Loading ontology axioms
- `checking_constraints`: Validating constraints
- `generating_inferences`: Applying inference rules
- `finalizing_report`: Compiling results

#### Validation Complete

Sent when validation finishes successfully.

```json
{
  "type": "validation_complete",
  "job_id": "job-uuid",
  "report_id": "report-uuid",
  "violations_count": 3,
  "inferences_count": 5,
  "duration_ms": 1523,
  "timestamp": "2025-10-17T12:34:56Z"
}
```

#### Validation Error

Sent when validation encounters an error.

```json
{
  "type": "validation_error",
  "job_id": "job-uuid",
  "error": "Reasoning timeout after 30s",
  "code": "TIMEOUT_ERROR",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

#### Echo (Testing)

Echo response for client messages.

```json
{
  "type": "echo",
  "original": "client message",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

### Reconnection Strategy

Implement exponential backoff for reconnection:

```javascript
class OntologyWebSocketClient {
  constructor(baseUrl, clientId) {
    this.baseUrl = baseUrl;
    this.clientId = clientId;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.reconnectAttempts = 0;
  }

  connect() {
    const wsUrl = `${this.baseUrl}/api/ontology/ws?client_id=${this.clientId}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('Connected');
      this.reconnectDelay = 1000;
      this.reconnectAttempts = 0;
    };

    this.ws.onclose = () => {
      this.reconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  reconnect() {
    this.reconnectAttempts++;
    console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);

    // Exponential backoff with max delay
    this.reconnectDelay = Math.min(
      this.reconnectDelay * 2,
      this.maxReconnectDelay
    );
  }
}
```

## Rate Limiting

All endpoints are subject to rate limiting:

| Limit Type | Threshold | Window | Response |
|------------|-----------|--------|----------|
| Per-IP | 100 requests | 1 minute | 429 Too Many Requests |
| Per-User | 1000 requests | 1 hour | 429 Too Many Requests |
| WebSocket | 10 connections | Per client | Connection refused |

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1634567890
```

## Error Codes Reference

| Code | HTTP Status | Description | Recovery |
|------|-------------|-------------|----------|
| FEATURE_DISABLED | 503 | Ontology validation disabled | Enable feature flag |
| ACTOR_UNAVAILABLE | 503 | Ontology actor not running | Check system health |
| LOAD_FAILED | 400 | Ontology load error | Check content format |
| PARSE_ERROR | 400 | Ontology syntax error | Validate ontology syntax |
| VALIDATION_FAILED | 400 | Validation process error | Check graph data |
| REPORT_NOT_FOUND | 404 | Report ID not found | Verify report ID |
| INFERENCE_FAILED | 400 | Inference generation error | Reduce max_depth |
| TIMEOUT_ERROR | 408 | Operation timeout | Increase timeout config |
| CACHE_ERROR | 500 | Cache operation failed | Clear caches |
| ACTOR_ERROR | 500 | Internal actor error | Check logs |
| MAPPING_UPDATE_FAILED | 400 | Invalid mapping config | Validate config schema |
| CACHE_CLEAR_FAILED | 500 | Cache clear error | Restart service |

## Best Practices

### Loading Ontologies

1. **Use caching**: Load ontologies once and reuse the `ontology_id`
2. **Format consistency**: Always specify `format` parameter explicitly
3. **Size limits**: Keep ontologies under 10 MB for optimal performance
4. **Validation**: Validate ontology syntax before loading

### Validation

1. **Choose appropriate mode**:
   - Quick for interactive validation
   - Full for production workflows
   - Incremental for real-time updates

2. **Use WebSockets**: Enable progress updates for long-running validations

3. **Monitor queue**: Check `/api/ontology/health` to avoid overwhelming the system

4. **Handle timeouts**: Set appropriate `reasoning_timeout_seconds` for your graph size

### Performance

1. **Enable caching**: Set `enable_caching: true` in validation config
2. **Adjust TTL**: Balance freshness vs performance with `cache_ttl_seconds`
3. **Incremental validation**: Use for real-time scenarios
4. **Clear caches**: Periodically clear caches to free memory

### Error Handling

1. **Check feature flag**: Verify `ontology_validation_enabled` in health endpoint
2. **Implement retries**: Use exponential backoff for transient errors
3. **Log trace IDs**: Include `trace_id` from error responses in bug reports
4. **Monitor health**: Set up alerts on `/api/ontology/health` metrics

## Example Workflows

### Complete Validation Workflow

```bash
#!/bin/bash

# 1. Check system health
curl http://localhost:8080/api/ontology/health

# 2. Load ontology
ONTOLOGY_ID=$(curl -X POST http://localhost:8080/api/ontology/load \
  -H "Content-Type: application/json" \
  -d @ontology.json \
  | jq -r '.ontology_id')

echo "Loaded ontology: $ONTOLOGY_ID"

# 3. Start validation
JOB_ID=$(curl -X POST http://localhost:8080/api/ontology/validate \
  -H "Content-Type: application/json" \
  -d "{\"ontology_id\":\"$ONTOLOGY_ID\",\"mode\":\"Full\"}" \
  | jq -r '.job_id')

echo "Started validation job: $JOB_ID"

# 4. Wait for completion (polling)
while true; do
  REPORT=$(curl -s "http://localhost:8080/api/ontology/report?report_id=$JOB_ID")
  if [ $? -eq 0 ]; then
    break
  fi
  sleep 1
done

# 5. Display results
echo "Validation complete!"
echo $REPORT | jq '.violations | length' | xargs echo "Violations:"
echo $REPORT | jq '.inferred_triples | length' | xargs echo "Inferences:"

# 6. Get detailed violations
echo $REPORT | jq '.violations[] | select(.severity == "Error")'
```

### Real-Time Validation with WebSocket

```javascript
async function validateWithProgress(ontologyId) {
  const clientId = `client-${Date.now()}`;

  // Connect to WebSocket
  const ws = new WebSocket(`ws://localhost:8080/api/ontology/ws?client_id=${clientId}`);

  return new Promise((resolve, reject) => {
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      switch (message.type) {
        case 'connection_established':
          console.log('Connected, starting validation...');
          startValidation(ontologyId, clientId);
          break;

        case 'validation_progress':
          console.log(`Progress: ${message.progress * 100}% - ${message.message}`);
          break;

        case 'validation_complete':
          console.log('Validation complete!');
          fetchReport(message.report_id).then(resolve);
          ws.close();
          break;

        case 'validation_error':
          console.error('Validation failed:', message.error);
          reject(new Error(message.error));
          ws.close();
          break;
      }
    };

    ws.onerror = reject;
  });
}

async function startValidation(ontologyId, clientId) {
  await fetch('http://localhost:8080/api/ontology/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ontology_id: ontologyId,
      mode: 'Full',
      enable_websocket_updates: true,
      client_id: clientId
    })
  });
}

async function fetchReport(reportId) {
  const response = await fetch(`http://localhost:8080/api/ontology/reports/${reportId}`);
  return response.json();
}

// Usage
const report = await validateWithProgress('ontology_abc123');
console.log('Violations:', report.violations.length);
console.log('Inferences:', report.inferred_triples.length);
```

## Versioning

API version is included in the base URL path. Current version: **v1**

Future versions will maintain backwards compatibility with:
- `/api/v1/ontology/*` - Current stable API
- `/api/v2/ontology/*` - Future enhancements

## Support

For issues, feature requests, or questions:
- GitHub Issues: [VisionFlow Issues](https://github.com/yourusername/VisionFlow/issues)
- Documentation: [Full Documentation](../index.md)
- Examples: [Code Examples](../code-examples/)
