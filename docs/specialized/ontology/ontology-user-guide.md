# Ontology System User Guide

## Table of Contents
- [Getting Started](#getting-started)
- [Configuration Guide](#configuration-guide)
- [Basic Operations](#basic-operations)
- [Advanced Features](#advanced-features)
- [Use Cases & Examples](#use-cases--examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)

## Getting Started

The Ontology System provides semantic validation and reasoning capabilities for your knowledge graph. This guide will help you set up and use the system effectively.

### Prerequisites

1. **WebXR Server**: Running instance with ontology feature enabled
2. **Authentication**: Valid API credentials
3. **Ontology File**: OWL/RDF ontology defining your domain model

### Quick Start

1. **Enable the feature** in your configuration:
```toml
[features]
ontology_validation = true
```

2. **Load your first ontology**:
```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "source": "/path/to/your/ontology.owl",
    "format": "rdf-xml"
  }'
```

3. **Run validation**:
```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "ontologyId": "<returned_id>",
    "mode": "quick"
  }'
```

## Configuration Guide

### Mapping Configuration

The mapping file (`mapping.toml`) defines how your property graph maps to RDF semantics:

```toml
# Global settings
[global]
base_iri = "https://yourcompany.com/graph#"
default_class = "ex:Thing"

# Node type mappings
[classes.node_type]
"person" = "foaf:Person"
"organization" = "org:Organization"
"document" = "foaf:Document"
"project" = "ex:Project"

# Edge type mappings
[properties.edge_type]
"employs" = "org:employs"
"knows" = "foaf:knows"
"partOf" = "ex:partOf"
"dependsOn" = "ex:dependsOn"

# Metadata mappings
[properties.metadata]
"email" = "foaf:mbox"
"name" = "foaf:name"
"created" = "dcterms:created"
"fileSize" = "ex:fileSize"

# Inverse relationships
[inverses]
"org:employs" = "ex:worksFor"
"ex:partOf" = "ex:hasPart"
"ex:dependsOn" = "ex:supports"

# IRI templates
[templates]
node_iri = "ex:node/{id}"
edge_iri = "ex:edge/{source}-{target}"

# Domain and range constraints
[constraints.domain]
"org:employs" = "org:Organization"
"foaf:knows" = "foaf:Person"

[constraints.range]
"org:employs" = "foaf:Person"
"foaf:knows" = "foaf:Person"
```

### Validation Configuration

Configure validation behaviour through the API:

```json
{
  "config": {
    "enableReasoning": true,
    "reasoningTimeoutSeconds": 60,
    "enableInference": true,
    "maxInferenceDepth": 5,
    "enableCaching": true,
    "cacheTtlSeconds": 3600,
    "validateCardinality": true,
    "validateDomainsRanges": true,
    "validateDisjointClasses": true
  }
}
```

**Configuration Options**:
- `enableReasoning`: Enable logical inference
- `reasoningTimeoutSeconds`: Maximum time for reasoning
- `enableInference`: Generate implicit relationships
- `maxInferenceDepth`: Limit inference chain length
- `enableCaching`: Cache validation results
- `cacheTtlSeconds`: Cache expiration time
- `validateCardinality`: Check property usage limits
- `validateDomainsRanges`: Verify property domains/ranges
- `validateDisjointClasses`: Check class disjointness

## Basic Operations

### Loading Ontologies

#### From File
```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/ontologies/domain.owl",
    "format": "rdf-xml"
  }'
```

#### From URL
```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://www.w3.org/2006/vcard/ns-2006.rdf",
    "format": "rdf-xml"
  }'
```

#### Direct Content
```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "@prefix ex: <http://example.org/> . ex:Person a owl:Class .",
    "format": "turtle"
  }'
```

### Running Validation

#### Quick Validation (< 100ms)
```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "ontologyId": "ont_123",
    "mode": "quick"
  }'
```

#### Full Validation (< 5s)
```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "ontologyId": "ont_123",
    "mode": "full",
    "priority": 8
  }'
```

#### Incremental Validation (< 50ms)
```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "ontologyId": "ont_123",
    "mode": "incremental"
  }'
```

### Retrieving Results

#### Get Latest Report
```bash
curl "/api/ontology/report"
```

#### Get Specific Report
```bash
curl "/api/ontology/report?report_id=report_abc123"
```

#### System Health Check
```bash
curl "/api/ontology/health"
```

## Advanced Features

### Real-Time Updates with WebSockets

```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('/api/ontology/ws?client_id=my_client');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'validation_progress':
      console.log(`Progress: ${message.payload.progress * 100}%`);
      break;

    case 'validation_complete':
      console.log('Validation completed:', message.payload);
      break;

    case 'constraint_update':
      updateVisualization(message.payload.appliedConstraints);
      break;
  }
};

// Start validation with real-time updates
fetch('/api/ontology/validate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    ontologyId: 'ont_123',
    mode: 'full',
    enableWebsocketUpdates: true,
    clientId: 'my_client'
  })
});
```

### Applying Inferences

```javascript
// Apply inferred relationships to the graph
const inferences = [
  {
    subject: "ex:alice",
    predicate: "ex:worksFor",
    object: "ex:acme_corp",
    isLiteral: false
  }
];

const response = await fetch('/api/ontology/apply', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    rdfTriples: inferences,
    updateGraph: true
  })
});

const result = await response.json();
console.log(`Applied ${result.inferredTriples.length} inferences`);
```

### Physics Integration

The system automatically translates ontological constraints into physics forces:

```javascript
// Monitor constraint updates
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'constraint_update') {
    const { groupName, constraintCount, appliedConstraints } = message.payload;

    switch (groupName) {
      case 'ontology_separation':
        // Disjoint classes create separation forces
        console.log(`Applied ${constraintCount} separation constraints`);
        break;

      case 'ontology_alignment':
        // Hierarchical relationships create alignment forces
        console.log(`Applied ${constraintCount} alignment constraints`);
        break;

      case 'ontology_boundaries':
        // Cardinality constraints create boundaries
        console.log(`Applied ${constraintCount} boundary constraints`);
        break;

      case 'ontology_identity':
        // Same-as relationships create co-location forces
        console.log(`Applied ${constraintCount} identity constraints`);
        break;
    }
  }
};
```

## Use Cases & Examples

### Use Case 1: Corporate Knowledge Graph

**Scenario**: Model employees, departments, and projects with validation.

**Ontology** (`corporate.owl`):
```turtle
@prefix corp: <http://company.com/ontology#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix org: <http://www.w3.org/ns/org#> .

# Classes
corp:Employee a owl:Class ;
    rdfs:subClassOf foaf:Person .

corp:Manager a owl:Class ;
    rdfs:subClassOf corp:Employee .

corp:Department a owl:Class ;
    rdfs:subClassOf org:Organization .

corp:Project a owl:Class .

# Properties
corp:manages a owl:ObjectProperty ;
    rdfs:domain corp:Manager ;
    rdfs:range corp:Employee .

corp:worksOn a owl:ObjectProperty ;
    rdfs:domain corp:Employee ;
    rdfs:range corp:Project .

# Constraints
corp:Employee owl:disjointWith corp:Department .

# A person can only have one manager
corp:managedBy a owl:FunctionalProperty ;
    owl:inverseOf corp:manages .
```

**Mapping Configuration**:
```toml
[classes.node_type]
"employee" = "corp:Employee"
"manager" = "corp:Manager"
"department" = "corp:Department"
"project" = "corp:Project"

[properties.edge_type]
"manages" = "corp:manages"
"worksOn" = "corp:worksOn"
"memberOf" = "org:member"

[inverses]
"corp:manages" = "corp:managedBy"
```

**Validation Results**:
- **Violations**: Employee node with Department type
- **Inferences**: If Alice manages Bob, then Bob is managed by Alice
- **Physics**: Employees and Departments repel each other

### Use Case 2: Document Management System

**Scenario**: Track documents, authors, and topics with semantic validation.

**Ontology** (`documents.owl`):
```turtle
@prefix doc: <http://documents.com/ontology#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

# Classes
doc:Document a owl:Class .
doc:Person a owl:Class ;
    rdfs:subClassOf foaf:Person .
doc:Topic a owl:Class .

# Properties
doc:author a owl:ObjectProperty ;
    rdfs:domain doc:Document ;
    rdfs:range doc:Person .

doc:covers a owl:ObjectProperty ;
    rdfs:domain doc:Document ;
    rdfs:range doc:Topic .

doc:cites a owl:ObjectProperty ;
    rdfs:domain doc:Document ;
    rdfs:range doc:Document ;
    a owl:TransitiveProperty .

# Constraints
doc:Document owl:disjointWith doc:Person .
```

**Expected Behaviors**:
- **Separation**: Documents and People appear in distinct clusters
- **Transitivity**: Citation chains automatically inferred
- **Validation**: Documents cannot be classified as People

### Use Case 3: Scientific Data Model

**Scenario**: Model experiments, researchers, and findings.

**Ontology** (`science.owl`):
```turtle
@prefix sci: <http://science.org/ontology#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

# Classes
sci:Experiment a owl:Class .
sci:Researcher a owl:Class ;
    rdfs:subClassOf foaf:Person .
sci:Finding a owl:Class .
sci:Equipment a owl:Class .

# Properties
sci:conducts a owl:ObjectProperty ;
    rdfs:domain sci:Researcher ;
    rdfs:range sci:Experiment .

sci:produces a owl:ObjectProperty ;
    rdfs:domain sci:Experiment ;
    rdfs:range sci:Finding .

sci:uses a owl:ObjectProperty ;
    rdfs:domain sci:Experiment ;
    rdfs:range sci:Equipment .

# Same researcher, different institutions
sci:sameResearcher a owl:SymmetricProperty .
```

**Physics Integration**:
- Experiments cluster around their researchers
- Related findings group together
- Equipment spreads around experiments that use it

## Best Practices

### Ontology Design

1. **Start Simple**: Begin with basic class hierarchies and properties
2. **Use Standards**: Leverage existing vocabularies (FOAF, Dublin Core, etc.)
3. **Clear Naming**: Use descriptive, consistent naming conventions
4. **Document**: Include rdfs:comment for all classes and properties

### Mapping Strategy

1. **Consistent Mapping**: Map similar concepts consistently
2. **Namespace Organization**: Use clear namespace prefixes
3. **IRI Patterns**: Establish consistent IRI templates
4. **Validate Early**: Test mappings with small datasets first

### Performance Optimization

1. **Incremental Updates**: Use incremental validation for frequent changes
2. **Cache Configuration**: Set appropriate TTL values
3. **Selective Validation**: Disable expensive checks when not needed
4. **Batch Operations**: Group multiple changes together

### Validation Workflow

1. **Development**: Use quick validation during development
2. **Testing**: Run full validation in test environments
3. **Production**: Use incremental validation with periodic full checks
4. **Monitoring**: Set up alerts for validation failures

## Troubleshooting

### Common Issues

#### Ontology Loading Failures

**Problem**: Ontology fails to load with parse errors
```json
{
  "error": "Failed to parse ontology",
  "code": "ONTOLOGY_PARSE_ERROR",
  "details": {
    "line": 42,
    "column": 15,
    "context": "Invalid IRI format"
  }
}
```

**Solutions**:
1. Validate ontology syntax using online validators
2. Check for missing namespace declarations
3. Ensure proper XML/Turtle escaping
4. Verify IRI formats are valid

#### Mapping Errors

**Problem**: Graph-to-RDF mapping fails
```json
{
  "error": "Mapping failed for node type 'unknown'",
  "code": "MAPPING_ERROR"
}
```

**Solutions**:
1. Add missing node types to mapping configuration
2. Provide fallback mappings for unknown types
3. Check IRI template syntax
4. Validate domain/range constraints

#### Performance Issues

**Problem**: Validation takes too long
```json
{
  "error": "Validation timeout after 60 seconds",
  "code": "VALIDATION_TIMEOUT"
}
```

**Solutions**:
1. Increase timeout configuration
2. Use incremental validation mode
3. Reduce inference depth
4. Split large ontologies
5. Enable result caching

#### Memory Exhaustion

**Problem**: System runs out of memory during validation
```json
{
  "error": "System resources exhausted",
  "code": "RESOURCE_EXHAUSTED"
}
```

**Solutions**:
1. Increase system memory allocation
2. Enable incremental processing
3. Clear caches regularly
4. Reduce concurrent validation jobs

### Debugging Tools

#### Validation Reports

Examine validation reports for detailed information:
```bash
# Get detailed report
curl "/api/ontology/report" | jq '.violations[] | select(.severity == "Error")'

# Check statistics
curl "/api/ontology/report" | jq '.statistics'
```

#### Health Monitoring

Monitor system health:
```bash
# Check system status
curl "/api/ontology/health" | jq '.health'

# Monitor queue size
watch 'curl -s "/api/ontology/health" | jq ".health.validationQueueSize"'
```

#### WebSocket Debugging

Monitor real-time events:
```javascript
const ws = new WebSocket('/api/ontology/ws?client_id=debug');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(`[${message.timestamp}] ${message.type}:`, message.payload);
};
```

## Performance Tuning

### Optimal Configuration

For different scenarios:

#### Development Environment
```json
{
  "config": {
    "enableReasoning": true,
    "reasoningTimeoutSeconds": 10,
    "enableInference": false,
    "enableCaching": false,
    "validateCardinality": false
  }
}
```

#### Testing Environment
```json
{
  "config": {
    "enableReasoning": true,
    "reasoningTimeoutSeconds": 30,
    "enableInference": true,
    "maxInferenceDepth": 3,
    "enableCaching": true,
    "validateCardinality": true
  }
}
```

#### Production Environment
```json
{
  "config": {
    "enableReasoning": true,
    "reasoningTimeoutSeconds": 60,
    "enableInference": true,
    "maxInferenceDepth": 5,
    "enableCaching": true,
    "cacheTtlSeconds": 7200,
    "validateCardinality": true,
    "validateDomainsRanges": true,
    "validateDisjointClasses": true
  }
}
```

### Monitoring Metrics

Track these performance indicators:

1. **Validation Time**: Average time per validation
2. **Cache Hit Rate**: Percentage of cache hits
3. **Queue Length**: Number of pending validation jobs
4. **Memory Usage**: System memory consumption
5. **Error Rate**: Percentage of failed validations

### Scaling Strategies

For large-scale deployments:

1. **Horizontal Scaling**: Deploy multiple ontology actors
2. **Caching Layers**: Use external caching systems
3. **Batch Processing**: Group validation requests
4. **Load Balancing**: Distribute validation load
5. **Resource Limits**: Set appropriate memory/CPU limits

This user guide provides comprehensive information for effectively using the ontology system, from basic operations to advanced configuration and troubleshooting.