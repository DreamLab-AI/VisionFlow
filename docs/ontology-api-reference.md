# Ontology API Reference

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [REST Endpoints](#rest-endpoints)
- [WebSocket Protocol](#websocket-protocol)
- [Data Models](#data-models)
- [Error Codes](#error-codes)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

The Ontology API provides comprehensive endpoints for managing OWL ontologies, performing semantic validation, and integrating logical reasoning with the knowledge graph visualization system.

**Base URL**: `/api/ontology`

**Content Types**:
- Request: `application/json`
- Response: `application/json`
- WebSocket: `application/json` messages

## Authentication

All endpoints require valid authentication headers. The system uses the existing WebXR authentication mechanism.

```http
Authorization: Bearer <token>
Content-Type: application/json
```

## REST Endpoints

### Load Ontology Axioms

Load ontology definitions from various sources.

```http
POST /api/ontology/load-axioms
```

**Request Body**:
```json
{
  "source": "https://example.org/ontology.owl",
  "format": "turtle",
  "validateImmediately": true
}
```

**Parameters**:
- `source` (string, required): File path, URL, or direct OWL content
- `format` (string, optional): Format hint ("turtle", "rdf-xml", "n-triples")
- `validateImmediately` (boolean, optional): Start validation after loading

**Response** (200 OK):
```json
{
  "ontologyId": "ontology_a1b2c3d4",
  "loadedAt": "2024-01-15T10:30:00Z",
  "axiomCount": 1250,
  "loadingTimeMs": 1200,
  "validationJobId": "job_x9y8z7"
}
```

**Example cURL**:
```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "source": "file:///ontologies/domain.owl",
    "format": "rdf-xml",
    "validateImmediately": false
  }'
```

### Update Mapping Configuration

Update the property graph to RDF mapping rules.

```http
POST /api/ontology/mapping
```

**Request Body**:
```json
{
  "config": {
    "enableReasoning": true,
    "reasoningTimeoutSeconds": 30,
    "enableInference": true,
    "maxInferenceDepth": 3,
    "enableCaching": true,
    "cacheTtlSeconds": 3600,
    "validateCardinality": true,
    "validateDomainsRanges": true,
    "validateDisjointClasses": true
  },
  "applyToAll": false
}
```

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Mapping configuration updated",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Validate Ontology

Run validation against the current graph state.

```http
POST /api/ontology/validate
```

**Request Body**:
```json
{
  "ontologyId": "ontology_a1b2c3d4",
  "mode": "full",
  "priority": 8,
  "enableWebsocketUpdates": true,
  "clientId": "client_abc123"
}
```

**Parameters**:
- `ontologyId` (string, required): Previously loaded ontology ID
- `mode` (string, required): "quick", "full", or "incremental"
- `priority` (integer, optional): 1-10, higher is more urgent
- `enableWebsocketUpdates` (boolean, optional): Enable real-time updates
- `clientId` (string, optional): Client ID for WebSocket routing

**Response** (200 OK):
```json
{
  "jobId": "job_validation_789",
  "status": "queued",
  "estimatedCompletion": "2024-01-15T10:35:00Z",
  "queuePosition": 2,
  "websocketUrl": "/api/ontology/ws?client_id=client_abc123"
}
```

### Get Validation Report

Retrieve the latest validation results.

```http
GET /api/ontology/report?report_id=<id>
```

**Query Parameters**:
- `report_id` (string, optional): Specific report ID, or latest if omitted

**Response** (200 OK):
```json
{
  "id": "report_def456",
  "timestamp": "2024-01-15T10:32:15Z",
  "durationMs": 2100,
  "graphSignature": "blake3_hash_abc123",
  "totalTriples": 15420,
  "violations": [
    {
      "id": "violation_001",
      "severity": "Error",
      "rule": "DisjointClasses",
      "message": "Individual ex:john cannot be both ex:Person and ex:Company",
      "subject": "ex:john",
      "predicate": "rdf:type",
      "object": null,
      "timestamp": "2024-01-15T10:32:15Z"
    }
  ],
  "inferredTriples": [
    {
      "subject": "ex:alice",
      "predicate": "ex:worksFor",
      "object": "ex:acme_corp",
      "isLiteral": false,
      "datatype": null,
      "language": null
    }
  ],
  "statistics": {
    "classesChecked": 45,
    "propertiesChecked": 78,
    "individualsChecked": 1205,
    "constraintsEvaluated": 234,
    "inferenceRulesApplied": 89,
    "cacheHits": 156,
    "cacheMisses": 23
  }
}
```

### Apply Inferences

Apply reasoning results to the graph.

```http
POST /api/ontology/apply
```

**Request Body**:
```json
{
  "rdfTriples": [
    {
      "subject": "ex:alice",
      "predicate": "ex:worksFor",
      "object": "ex:acme_corp",
      "isLiteral": false,
      "datatype": null,
      "language": null
    }
  ],
  "maxDepth": 5,
  "updateGraph": true
}
```

**Response** (200 OK):
```json
{
  "inputCount": 1,
  "inferredTriples": [
    {
      "subject": "ex:acme_corp",
      "predicate": "ex:employs",
      "object": "ex:alice",
      "isLiteral": false,
      "datatype": null,
      "language": null
    }
  ],
  "processingTimeMs": 45,
  "graphUpdated": true
}
```

### System Health

Check ontology system health and performance.

```http
GET /api/ontology/health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "health": {
    "loadedOntologies": 3,
    "cachedReports": 12,
    "validationQueueSize": 1,
    "lastValidation": "2024-01-15T10:30:00Z",
    "cacheHitRate": 0.85,
    "avgValidationTimeMs": 1250.5,
    "activeJobs": 2,
    "memoryUsageMb": 156.8
  },
  "ontologyValidationEnabled": true,
  "timestamp": "2024-01-15T10:35:00Z"
}
```

### Clear Caches

Clear all ontology-related caches.

```http
DELETE /api/ontology/cache
```

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "All caches cleared",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

## WebSocket Protocol

### Connection

Connect to the WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket('/api/ontology/ws?client_id=your_client_id');
```

### Message Format

All WebSocket messages follow this format:

```json
{
  "type": "message_type",
  "payload": {},
  "timestamp": "2024-01-15T10:35:00Z"
}
```

### Message Types

#### Connection Established
```json
{
  "type": "connection_established",
  "payload": {
    "clientId": "client_abc123"
  },
  "timestamp": "2024-01-15T10:35:00Z"
}
```

#### Validation Progress
```json
{
  "type": "validation_progress",
  "payload": {
    "jobId": "job_validation_789",
    "progress": 0.65,
    "stage": "running_inference",
    "estimatedCompletion": "2024-01-15T10:36:00Z"
  },
  "timestamp": "2024-01-15T10:35:30Z"
}
```

#### Validation Complete
```json
{
  "type": "validation_complete",
  "payload": {
    "jobId": "job_validation_789",
    "reportId": "report_def456",
    "violationCount": 3,
    "inferenceCount": 47,
    "durationMs": 2100
  },
  "timestamp": "2024-01-15T10:35:45Z"
}
```

#### Constraint Update
```json
{
  "type": "constraint_update",
  "payload": {
    "groupName": "ontology_separation",
    "constraintCount": 23,
    "appliedConstraints": [
      {
        "id": "constraint_001",
        "type": "separation",
        "nodeIds": [123, 456],
        "strength": 0.8
      }
    ]
  },
  "timestamp": "2024-01-15T10:35:50Z"
}
```

#### Error Message
```json
{
  "type": "error",
  "payload": {
    "code": "VALIDATION_FAILED",
    "message": "Ontology validation failed due to parsing error",
    "details": {
      "line": 45,
      "column": 12,
      "context": "Invalid IRI format"
    }
  },
  "timestamp": "2024-01-15T10:35:20Z"
}
```

## Data Models

### ValidationMode
```typescript
enum ValidationMode {
  "quick",      // Fast consistency checks only
  "full",       // Complete validation with inference
  "incremental" // Delta-based validation
}
```

### Severity
```typescript
enum Severity {
  "Error",   // Logical contradiction
  "Warning", // Potential issue
  "Info"     // Informational notice
}
```

### RdfTriple
```typescript
interface RdfTriple {
  subject: string;
  predicate: string;
  object: string;
  isLiteral?: boolean;
  datatype?: string;
  language?: string;
}
```

### Violation
```typescript
interface Violation {
  id: string;
  severity: Severity;
  rule: string;
  message: string;
  subject?: string;
  predicate?: string;
  object?: string;
  timestamp: string; // ISO 8601
}
```

### ValidationReport
```typescript
interface ValidationReport {
  id: string;
  timestamp: string; // ISO 8601
  durationMs: number;
  graphSignature: string;
  totalTriples: number;
  violations: Violation[];
  inferredTriples: RdfTriple[];
  statistics: ValidationStatistics;
}
```

### ValidationStatistics
```typescript
interface ValidationStatistics {
  classesChecked: number;
  propertiesChecked: number;
  individualsChecked: number;
  constraintsEvaluated: number;
  inferenceRulesApplied: number;
  cacheHits: number;
  cacheMisses: number;
}
```

## Error Codes

### HTTP Status Codes
- `200 OK`: Successful operation
- `202 Accepted`: Validation job queued
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `409 Conflict`: Ontology conflict or inconsistency
- `413 Payload Too Large`: Ontology too large
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: System error
- `503 Service Unavailable`: Feature disabled

### Application Error Codes

#### Loading Errors
- `INVALID_ONTOLOGY_FORMAT`: Unsupported or malformed ontology
- `ONTOLOGY_PARSE_ERROR`: Parsing failed with syntax errors
- `ONTOLOGY_TOO_LARGE`: Ontology exceeds size limits
- `SOURCE_UNAVAILABLE`: Cannot access ontology source

#### Validation Errors
- `ONTOLOGY_NOT_FOUND`: Referenced ontology doesn't exist
- `VALIDATION_TIMEOUT`: Validation exceeded time limit
- `INCONSISTENT_ONTOLOGY`: Ontology contains contradictions
- `MAPPING_ERROR`: Graph-to-RDF mapping failed

#### System Errors
- `FEATURE_DISABLED`: Ontology validation is disabled
- `RESOURCE_EXHAUSTED`: System resources exhausted
- `ACTOR_UNAVAILABLE`: Ontology actor not responding
- `CACHE_ERROR`: Cache operation failed

### Error Response Format
```json
{
  "error": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional context",
    "suggestion": "Recommended action"
  },
  "timestamp": "2024-01-15T10:35:00Z",
  "traceId": "trace_abc123def"
}
```

## Rate Limiting

The API implements rate limiting to ensure system stability:

- **Standard endpoints**: 60 requests/minute per client
- **Validation endpoints**: 10 requests/minute per client
- **WebSocket connections**: 5 concurrent per client
- **Large ontology loading**: 2 requests/hour per client

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705312200
```

## Examples

### Complete Validation Workflow

1. **Load ontology**:
```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -d '{"source": "https://example.org/domain.owl"}'
```

2. **Configure validation**:
```bash
curl -X POST "/api/ontology/mapping" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "enableReasoning": true,
      "validateDisjointClasses": true
    }
  }'
```

3. **Run validation**:
```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "ontologyId": "ontology_a1b2c3d4",
    "mode": "full"
  }'
```

4. **Get results**:
```bash
curl "/api/ontology/report"
```

### WebSocket Integration

```javascript
const ws = new WebSocket('/api/ontology/ws?client_id=my_app');

ws.onopen = () => {
  console.log('Connected to ontology updates');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'validation_progress':
      updateProgressBar(message.payload.progress);
      break;

    case 'validation_complete':
      displayResults(message.payload);
      break;

    case 'constraint_update':
      updatePhysicsConstraints(message.payload);
      break;
  }
};

// Start validation with real-time updates
fetch('/api/ontology/validate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    ontologyId: 'ontology_123',
    mode: 'full',
    enableWebsocketUpdates: true,
    clientId: 'my_app'
  })
});
```

### Error Handling

```javascript
async function validateOntology(ontologyId) {
  try {
    const response = await fetch('/api/ontology/validate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ontologyId,
        mode: 'full'
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new OntologyError(error.code, error.error, error.details);
    }

    return await response.json();
  } catch (error) {
    console.error('Validation failed:', error);

    if (error.code === 'FEATURE_DISABLED') {
      // Handle disabled feature
      showFeatureDisabledMessage();
    } else if (error.code === 'ONTOLOGY_NOT_FOUND') {
      // Handle missing ontology
      promptOntologyLoad();
    } else {
      // Handle general errors
      showErrorMessage(error.message);
    }
  }
}
```

This API reference provides complete documentation for integrating with the ontology validation system, including practical examples and comprehensive error handling patterns.