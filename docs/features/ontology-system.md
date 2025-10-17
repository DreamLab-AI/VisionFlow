# Ontology System

**Version**: 1.0.0
**Last Updated**: 2025-10-17
**Status**: Production-Ready

## Overview

VisionFlow's Ontology System provides OWL/RDF semantic validation, reasoning, and inference capabilities for knowledge graphs. The system validates graph data against formal ontologies, automatically infers new relationships, and ensures logical consistency through advanced reasoning algorithms.

## Key Features

- **OWL 2 Support**: Full support for OWL Functional Syntax and OWL/XML formats
- **Real-Time Validation**: Asynchronous validation with WebSocket progress updates
- **Inference Engine**: Automatic relationship inference with configurable depth
- **Constraint Checking**: Validates cardinality, domain/range, and disjoint class constraints
- **Caching System**: Intelligent caching with configurable TTL for performance
- **Property Graph Mapping**: Bidirectional mapping between property graphs and RDF triples
- **Horned-OWL Integration**: Uses horned-owl 1.2.0 for robust OWL parsing

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │REST API  │  │WebSocket │  │   UI     │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
└───────┼─────────────┼─────────────┼────────────────────────┘
        │             │             │
┌───────┼─────────────┼─────────────┼────────────────────────┐
│       │             │             │   API Layer            │
│  ┌────▼──────────────▼─────────────▼───┐                  │
│  │   Ontology API Handler (mod.rs)     │                  │
│  │  - Load/Validate/Report endpoints   │                  │
│  │  - WebSocket management             │                  │
│  └────────────┬─────────────────────────┘                  │
└───────────────┼────────────────────────────────────────────┘
                │
┌───────────────┼────────────────────────────────────────────┐
│               │           Actor Layer                      │
│  ┌────────────▼──────────────────────┐                    │
│  │  OntologyActor (Actix Actor)      │                    │
│  │  - Message handling               │                    │
│  │  - State management               │                    │
│  │  - Queue management               │                    │
│  └────────────┬──────────────────────┘                    │
└───────────────┼────────────────────────────────────────────┘
                │
┌───────────────┼────────────────────────────────────────────┐
│               │         Service Layer                      │
│  ┌────────────▼──────────────────────┐                    │
│  │  OwlValidatorService               │                    │
│  │  - Ontology parsing (horned-owl)   │                    │
│  │  - Graph-to-RDF mapping            │                    │
│  │  - Constraint validation           │                    │
│  │  - Inference generation            │                    │
│  │  - Cache management (DashMap)      │                    │
│  └────────────┬──────────────────────┘                    │
└───────────────┼────────────────────────────────────────────┘
                │
┌───────────────┼────────────────────────────────────────────┐
│               │      Integration Layer                     │
│  ┌────────────▼──────────────────────┐                    │
│  │  Physics Orchestrator              │                    │
│  │  - Applies ontology constraints    │                    │
│  │  - Updates force parameters        │                    │
│  │  - Visualizes relationships        │                    │
│  └────────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────┐
│ Ontology │  (OWL Functional Syntax or OWL/XML)
│   File   │
└────┬─────┘
     │
     ▼
┌────────────────┐
│ horned-owl     │  Parse OWL axioms
│ Parser         │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ SetOntology    │  In-memory representation
│ Cache          │  (DashMap with TTL)
└────┬───────────┘
     │
     ├──────────────────┐
     │                  │
     ▼                  ▼
┌──────────┐      ┌────────────┐
│ Property │      │ Validation │
│  Graph   │      │   Rules    │
└────┬─────┘      └─────┬──────┘
     │                  │
     ▼                  ▼
┌────────────────────────────┐
│ RDF Triple Generation      │
│ - Nodes → Classes/Types    │
│ - Edges → Object Props     │
│ - Properties → Data Props  │
└─────────┬──────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Constraint Validation    │
│ - Disjoint Classes       │
│ - Domain/Range           │
│ - Cardinality            │
└─────────┬────────────────┘
          │
          ├─────────────┐
          │             │
          ▼             ▼
┌──────────────┐  ┌──────────────┐
│  Violations  │  │  Inferences  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
       ┌────────────────┐
       │ Validation     │
       │ Report         │
       └────────────────┘
```

## API Reference

### REST Endpoints

#### Load Ontology

```http
POST /api/ontology/load
Content-Type: application/json

{
  "content": "Prefix(:=<http://example.org/>)\nOntology(<http://example.org/ontology>...)",
  "format": "functional" // optional: "functional", "owlxml"
}
```

**Response:**
```json
{
  "ontology_id": "ontology_abc123...",
  "axiom_count": 42
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/api/ontology/load \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Prefix(:=<http://example.org/>)\nOntology(<http://example.org/ontology>\nDeclaration(Class(:Person)))",
    "format": "functional"
  }'
```

#### Validate Graph

```http
POST /api/ontology/validate
Content-Type: application/json

{
  "ontology_id": "ontology_abc123...",
  "mode": "Full", // "Quick" | "Full" | "Incremental"
  "priority": 5, // optional: 1-10
  "enable_websocket_updates": true,
  "client_id": "client-uuid"
}
```

**Response:**
```json
{
  "job_id": "job-uuid",
  "status": "queued",
  "estimated_completion": "2025-10-17T12:34:56Z",
  "queue_position": 1,
  "websocket_url": "/api/ontology/ws?client_id=client-uuid"
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/api/ontology/validate \
  -H "Content-Type: application/json" \
  -d '{
    "ontology_id": "ontology_abc123",
    "mode": "Full",
    "enable_websocket_updates": true,
    "client_id": "my-client-id"
  }'
```

#### Get Validation Report

```http
GET /api/ontology/reports/{report_id}
```

**Response:**
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
      "timestamp": "2025-10-17T12:34:56Z"
    }
  ],
  "inferred_triples": [
    {
      "subject": "http://example.org/person1",
      "predicate": "http://example.org/worksFor",
      "object": "http://example.org/company1",
      "is_literal": false
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

**curl Example:**
```bash
curl http://localhost:8080/api/ontology/reports/report-uuid
```

#### Get System Health

```http
GET /api/ontology/health
```

**Response:**
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

**curl Example:**
```bash
curl http://localhost:8080/api/ontology/health
```

#### List Loaded Axioms

```http
GET /api/ontology/axioms
```

**Response:**
```json
{
  "axioms": [
    {
      "ontology_id": "ontology_abc123",
      "axiom_count": 42,
      "loaded_at": "2025-10-17T12:00:00Z"
    }
  ],
  "count": 1,
  "timestamp": "2025-10-17T12:34:56Z"
}
```

#### Get Inferences

```http
GET /api/ontology/inferences?ontology_id=ontology_abc123
```

**Response:**
```json
{
  "report_id": "report-uuid",
  "inferred_count": 5,
  "inferences": [
    {
      "subject": "http://example.org/person1",
      "predicate": "http://example.org/worksFor",
      "object": "http://example.org/company1",
      "is_literal": false
    }
  ],
  "generated_at": "2025-10-17T12:34:56Z",
  "inference_depth": 3,
  "timestamp": "2025-10-17T12:34:56Z"
}
```

#### Apply Inferences

```http
POST /api/ontology/apply
Content-Type: application/json

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

**Response:**
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

#### Update Mapping Configuration

```http
POST /api/ontology/mapping
Content-Type: application/json

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
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Mapping configuration updated",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

#### Clear Caches

```http
DELETE /api/ontology/cache
```

**Response:**
```json
{
  "status": "success",
  "message": "All caches cleared",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

### WebSocket Protocol

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/api/ontology/ws?client_id=my-client-id');

ws.onopen = () => {
  console.log('Connected to ontology validation stream');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  handleMessage(message);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from validation stream');
};
```

#### Message Types

**Connection Established**
```json
{
  "type": "connection_established",
  "client_id": "my-client-id",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Validation Progress**
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

**Validation Complete**
```json
{
  "type": "validation_complete",
  "job_id": "job-uuid",
  "report_id": "report-uuid",
  "violations_count": 3,
  "inferences_count": 5,
  "timestamp": "2025-10-17T12:34:56Z"
}
```

**Validation Error**
```json
{
  "type": "validation_error",
  "job_id": "job-uuid",
  "error": "Reasoning timeout after 30s",
  "timestamp": "2025-10-17T12:34:56Z"
}
```

## Configuration

### Validation Modes

**Quick Mode**
- Basic constraint checking only
- No inference generation
- Fastest validation (100-500ms)
- Suitable for interactive validation

**Full Mode**
- Complete constraint validation
- Full inference generation
- Reasoning with timeout protection
- Recommended for production (1-3s)

**Incremental Mode**
- Validates only changed nodes/edges
- Reuses cached results where possible
- Best for real-time validation
- Performance: ~50-200ms per change

### Validation Configuration

```rust
ValidationConfig {
    enable_reasoning: true,           // Enable OWL reasoning
    reasoning_timeout_seconds: 30,    // Timeout for reasoning operations
    enable_inference: true,            // Generate inferred triples
    max_inference_depth: 3,            // Maximum inference depth
    enable_caching: true,              // Enable result caching
    cache_ttl_seconds: 3600,           // Cache time-to-live (1 hour)
    validate_cardinality: true,        // Check cardinality constraints
    validate_domains_ranges: true,     // Check domain/range constraints
    validate_disjoint_classes: true,   // Check disjoint class violations
}
```

### Mapping Configuration (mapping.toml)

```toml
[metadata]
title = "VisionFlow Knowledge Graph Ontology Mapping"
version = "1.0.0"
description = "Mapping configuration for property graph to RDF/OWL conversion"

[global]
base_iri = "http://example.org/"
default_vocabulary = "http://example.org/vocab/"
default_language = "en"
strict_mode = true
auto_generate_inverses = true

[namespaces]
rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
rdfs = "http://www.w3.org/2000/01/rdf-schema#"
owl = "http://www.w3.org/2002/07/owl#"
xsd = "http://www.w3.org/2001/XMLSchema#"
foaf = "http://xmlns.com/foaf/0.1/"

[defaults]
default_node_class = "owl:Thing"
default_edge_property = "rdfs:seeAlso"
default_datatype = "xsd:string"

[class_mappings.Person]
owl_class = "foaf:Person"
rdfs_label = "Person"
rdfs_comment = "Represents a human individual"
rdfs_subclass_of = ["foaf:Agent"]

[object_property_mappings.employedBy]
owl_property = ":employedBy"
rdfs_label = "employed by"
rdfs_domain = "Single(foaf:Person)"
rdfs_range = "Single(:Organization)"
owl_inverse_of = ":employs"
property_type = "ObjectProperty"
```

## Property Graph to RDF Mapping

### Node Mapping

Property Graph:
```json
{
  "id": "person1",
  "labels": ["Person"],
  "properties": {
    "name": "Alice Smith",
    "email": "alice@example.com",
    "age": 30
  }
}
```

RDF Triples:
```turtle
<http://example.org/person1>
    rdf:type foaf:Person ;
    foaf:name "Alice Smith"^^xsd:string ;
    foaf:mbox "alice@example.com"^^xsd:string ;
    :age 30^^xsd:integer .
```

### Edge Mapping

Property Graph:
```json
{
  "id": "edge1",
  "source": "person1",
  "target": "company1",
  "relationship_type": "employedBy",
  "properties": {
    "since": "2020-01-15"
  }
}
```

RDF Triples:
```turtle
<http://example.org/person1> :employedBy <http://example.org/company1> .
<http://example.org/company1> :employs <http://example.org/person1> . # Auto-generated inverse
<http://example.org/edge1> :since "2020-01-15"^^xsd:date .
```

## Constraint Types

### Disjoint Classes

Ensures individuals cannot be instances of disjoint classes.

**Ontology Definition:**
```
DisjointClasses(:Person :Company)
```

**Violation Example:**
```turtle
# Violation: person1 cannot be both Person and Company
:person1 rdf:type :Person .
:person1 rdf:type :Company .
```

### Domain and Range

Validates subject and object types for properties.

**Ontology Definition:**
```
ObjectProperty(:employs)
  Domain(:Organization)
  Range(:Person)
```

**Violation Example:**
```turtle
# Violation: :person1 must be an Organization to use :employs
:person1 :employs :person2 .
```

### Cardinality Constraints

Enforces minimum and maximum property occurrences.

**Ontology Definition:**
```
Class(:Person)
  SubClassOf(
    ObjectExactCardinality(1 :hasSSN)
  )
```

**Violation Example:**
```turtle
# Violation: Person must have exactly 1 SSN
:person1 rdf:type :Person .
:person1 :hasSSN "123-45-6789" .
:person1 :hasSSN "987-65-4321" .
```

## Inference Rules

### Inverse Properties

**Ontology Definition:**
```
ObjectProperty(:employs)
  InverseOf(:employedBy)
```

**Inference:**
```turtle
# Given:
:company1 :employs :person1 .

# Inferred:
:person1 :employedBy :company1 .
```

### Transitive Properties

**Ontology Definition:**
```
ObjectProperty(:partOf)
  Characteristics(Transitive)
```

**Inference:**
```turtle
# Given:
:wheel :partOf :car .
:car :partOf :vehicle .

# Inferred:
:wheel :partOf :vehicle .
```

### Symmetric Properties

**Ontology Definition:**
```
ObjectProperty(:knows)
  Characteristics(Symmetric)
```

**Inference:**
```turtle
# Given:
:person1 :knows :person2 .

# Inferred:
:person2 :knows :person1 .
```

### Subclass Inference

**Ontology Definition:**
```
Class(:Employee)
  SubClassOf(:Person)
```

**Inference:**
```turtle
# Given:
:person1 rdf:type :Employee .

# Inferred:
:person1 rdf:type :Person .
```

## Performance Benchmarks

### Load Performance

| Ontology Size | Axiom Count | Load Time | Memory Usage |
|---------------|-------------|-----------|--------------|
| Small         | < 100       | 10-50ms   | 5 MB         |
| Medium        | 100-1000    | 50-200ms  | 20 MB        |
| Large         | 1000-10000  | 200-1000ms| 100 MB       |
| Very Large    | > 10000     | 1-5s      | 500 MB       |

### Validation Performance

| Graph Size | Nodes | Edges | Mode        | Time      |
|------------|-------|-------|-------------|-----------|
| Small      | 100   | 200   | Quick       | 50ms      |
| Small      | 100   | 200   | Full        | 200ms     |
| Medium     | 1000  | 2000  | Quick       | 200ms     |
| Medium     | 1000  | 2000  | Full        | 1.5s      |
| Large      | 10000 | 20000 | Quick       | 1s        |
| Large      | 10000 | 20000 | Full        | 8s        |
| Very Large | 100000| 200000| Incremental | 2s/change |

### Cache Performance

| Operation       | Cache Hit | Cache Miss | Speedup |
|-----------------|-----------|------------|---------|
| Load Ontology   | 5ms       | 150ms      | 30x     |
| Validation      | 50ms      | 1500ms     | 30x     |
| Inference       | 10ms      | 200ms      | 20x     |

## Client Integration Guide

### TypeScript Client

```typescript
interface OntologyClient {
  loadOntology(content: string, format?: string): Promise<LoadOntologyResponse>;
  validate(ontologyId: string, mode: ValidationMode): Promise<ValidationResponse>;
  getReport(reportId: string): Promise<ValidationReport>;
  getHealth(): Promise<HealthStatusResponse>;
  subscribeToUpdates(clientId: string, callback: (message: any) => void): WebSocket;
}

class VisionFlowOntologyClient implements OntologyClient {
  constructor(private baseUrl: string) {}

  async loadOntology(content: string, format = 'functional'): Promise<LoadOntologyResponse> {
    const response = await fetch(`${this.baseUrl}/api/ontology/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, format })
    });

    if (!response.ok) {
      throw new Error(`Failed to load ontology: ${response.statusText}`);
    }

    return response.json();
  }

  async validate(ontologyId: string, mode: ValidationMode): Promise<ValidationResponse> {
    const response = await fetch(`${this.baseUrl}/api/ontology/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ontology_id: ontologyId,
        mode,
        enable_websocket_updates: true
      })
    });

    return response.json();
  }

  async getReport(reportId: string): Promise<ValidationReport> {
    const response = await fetch(`${this.baseUrl}/api/ontology/reports/${reportId}`);
    return response.json();
  }

  async getHealth(): Promise<HealthStatusResponse> {
    const response = await fetch(`${this.baseUrl}/api/ontology/health`);
    return response.json();
  }

  subscribeToUpdates(clientId: string, callback: (message: any) => void): WebSocket {
    const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/api/ontology/ws?client_id=${clientId}`);

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      callback(message);
    };

    return ws;
  }
}

// Usage
const client = new VisionFlowOntologyClient('http://localhost:8080');

// Load ontology
const { ontology_id } = await client.loadOntology(ontologyContent);

// Subscribe to updates
const ws = client.subscribeToUpdates('my-client-id', (message) => {
  console.log('Validation update:', message);
});

// Validate graph
const { job_id } = await client.validate(ontology_id, 'Full');

// Get report when complete
const report = await client.getReport(job_id);
console.log(`Found ${report.violations.length} violations`);
console.log(`Inferred ${report.inferred_triples.length} new relationships`);
```

### React Hook

```typescript
function useOntologyValidation(ontologyId: string) {
  const [validating, setValidating] = useState(false);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const validate = async (mode: ValidationMode = 'Full') => {
    setValidating(true);
    setError(null);

    try {
      const client = new VisionFlowOntologyClient('http://localhost:8080');
      const clientId = uuidv4();

      // Subscribe to progress updates
      const ws = client.subscribeToUpdates(clientId, (message) => {
        if (message.type === 'validation_progress') {
          setProgress(message.progress);
        } else if (message.type === 'validation_complete') {
          client.getReport(message.report_id).then(setReport);
          setValidating(false);
          ws.close();
        } else if (message.type === 'validation_error') {
          setError(message.error);
          setValidating(false);
          ws.close();
        }
      });

      // Start validation
      await client.validate(ontologyId, mode);
    } catch (err) {
      setError(err.message);
      setValidating(false);
    }
  };

  return { validate, validating, report, error, progress };
}

// Usage in component
function OntologyPanel({ ontologyId }: { ontologyId: string }) {
  const { validate, validating, report, error, progress } = useOntologyValidation(ontologyId);

  return (
    <div>
      <button onClick={() => validate('Full')} disabled={validating}>
        Validate Graph
      </button>

      {validating && <ProgressBar value={progress} />}

      {report && (
        <div>
          <h3>Validation Results</h3>
          <p>Violations: {report.violations.length}</p>
          <p>Inferences: {report.inferred_triples.length}</p>
          <p>Duration: {report.duration_ms}ms</p>
        </div>
      )}

      {error && <ErrorMessage message={error} />}
    </div>
  );
}
```

## Error Handling

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| FEATURE_DISABLED | Ontology validation feature is disabled | 503 |
| ACTOR_UNAVAILABLE | Ontology actor not available | 503 |
| LOAD_FAILED | Failed to load ontology | 400 |
| PARSE_ERROR | Failed to parse ontology | 400 |
| VALIDATION_FAILED | Validation process failed | 400 |
| REPORT_NOT_FOUND | Validation report not found | 404 |
| INFERENCE_FAILED | Inference generation failed | 400 |
| TIMEOUT_ERROR | Reasoning timeout exceeded | 408 |
| CACHE_ERROR | Cache operation failed | 500 |
| ACTOR_ERROR | Internal actor communication error | 500 |

### Error Response Format

```json
{
  "error": "Failed to parse ontology",
  "code": "PARSE_ERROR",
  "details": {
    "line": 42,
    "message": "Unexpected token 'Declaration'"
  },
  "timestamp": "2025-10-17T12:34:56Z",
  "trace_id": "uuid"
}
```

## Security Considerations

### Feature Flag Protection

All ontology endpoints are protected by a feature flag that must be enabled:

```rust
// Check in API handler
let flags = FEATURE_FLAGS.lock().await;
if !flags.ontology_validation {
    return HttpResponse::ServiceUnavailable().json(error);
}
```

### Input Validation

- Ontology content is parsed in isolated context
- Maximum content size: 10 MB
- Timeout protection for reasoning operations
- Sanitized error messages (no internal paths)

### Rate Limiting

- Per-IP rate limiting: 100 requests/minute
- Per-user rate limiting: 1000 requests/hour
- WebSocket connection limits: 10 per client

## Troubleshooting

### Common Issues

**Issue: Ontology parsing fails**
- Ensure ontology is in OWL Functional Syntax or OWL/XML format
- Turtle and RDF/XML formats are not supported
- Check for syntax errors in the ontology file

**Issue: Validation timeout**
- Increase `reasoning_timeout_seconds` in configuration
- Use Quick mode for large graphs
- Enable incremental validation for real-time updates

**Issue: High memory usage**
- Reduce `cache_ttl_seconds` to expire caches sooner
- Use incremental validation mode
- Clear caches periodically via DELETE /api/ontology/cache

**Issue: WebSocket disconnections**
- Check network connectivity
- Verify client_id is unique per connection
- Implement reconnection logic with exponential backoff

### Debug Mode

Enable debug logging:

```bash
RUST_LOG=debug cargo run
```

View ontology statistics:

```bash
curl http://localhost:8080/api/ontology/health | jq
```

Monitor WebSocket messages:

```javascript
ws.onmessage = (event) => {
  console.log('WS Message:', JSON.parse(event.data));
};
```

## References

- [Ontology-Physics Integration](../../docs/ontology-physics-integration.md)
- [Migration Guide](../../docs/specialized/ontology/MIGRATION_GUIDE.md)
- [Horned-OWL Upgrade Notes](../../HORNED_OWL_UPGRADE.md)
- [API Endpoints Documentation](../api/ontology-endpoints.md)
- [Architecture Overview](../architecture/system-overview.md)
