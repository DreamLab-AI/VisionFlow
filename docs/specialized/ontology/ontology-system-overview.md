# Ontology System Overview

## Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Integration Points](#integration-points)
- [Feature Capabilities](#feature-capabilities)
- [Performance Characteristics](#performance-characteristics)
- [Technology Stack](#technology-stack)

## Introduction

The Ontology System provides a formal validation and logical inference layer for the WebXR knowledge graph, acting as a "truth engine." It combines the flexibility of property graphs with the formal rigor of OWL/RDF semantics to ensure logical consistency and enable automated knowledge discovery.

### Key Benefits

- **Logical Consistency**: Prevents inconsistencies through formal validation
- **Knowledge Discovery**: Infers new relationships from existing data and rules
- **Semantic Physics**: Translates logical constraints into physical forces for intuitive visualization
- **Data Quality**: Provides diagnostics and fixes for data integrity issues
- **Incremental Processing**: Supports efficient updates without full recomputation

## System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web Interface]
        API_CLIENT[API Clients]
        WS_CLIENT[WebSocket Clients]
    end

    subgraph "API Layer"
        REST[REST Endpoints<br/>/api/ontology/*]
        WS[WebSocket Handler<br/>Real-time Updates]
    end

    subgraph "Actor System"
        OA[OntologyActor<br/>Async Processing]
        GA[GraphServiceActor<br/>Graph State]
        PA[PhysicsOrchestratorActor<br/>Constraint Application]
        SA[SemanticProcessorActor<br/>Inference Processing]
    end

    subgraph "Core Services"
        OVS[OwlValidatorService<br/>Validation Engine]
        CT[ConstraintTranslator<br/>Physics Integration]
        MS[Mapping Service<br/>Graph↔RDF Translation]
    end

    subgraph "Storage & Caching"
        CACHE[In-Memory Cache<br/>Ontologies & Reports]
        CONFIG[Configuration<br/>mapping.toml]
    end

    subgraph "External Libraries"
        HORNED[horned-owl<br/>OWL Parsing]
        WHELK[whelk-rs<br/>Reasoning Engine]
        RIO[rio-*<br/>RDF Processing]
    end

    UI --> REST
    API_CLIENT --> REST
    WS_CLIENT --> WS

    REST --> OA
    WS --> OA

    OA --> OVS
    OA --> CT
    OA --> GA
    OA --> PA
    OA --> SA

    OVS --> MS
    OVS --> CACHE
    OVS --> CONFIG

    OVS --> HORNED
    OVS --> WHELK
    OVS --> RIO

    CT --> PA

    style OA fill:#38A169,stroke:#276749,stroke-width:2px,colour:#fff
    style OVS fill:#2B6CB0,stroke:#1A4E8D,stroke-width:2px,colour:#fff
    style CT fill:#D69E2E,stroke:#B7791F,stroke-width:2px,colour:#fff
```

## Core Components

### 1. OntologyActor
**Location**: `src/actors/ontology_actor.rs`

The central coordinator for all ontology operations, handling:
- Asynchronous validation jobs with priority queuing
- Caching and incremental processing
- Actor communication and coordination
- Health monitoring and performance metrics

**Key Features**:
- Job queue with priority levels (Critical, High, Normal, Low)
- LRU cache for validation reports
- Backpressure handling for resource management
- Integration with physics and semantic processors

### 2. OwlValidatorService
**Location**: `src/services/owl_validator.rs`

The core validation engine that:
- Loads and parses OWL ontologies from various sources
- Maps property graphs to RDF triples
- Performs consistency checking and logical reasoning
- Generates comprehensive validation reports

**Validation Types**:
- **Domain/Range Validation**: Ensures properties are used correctly
- **Disjoint Class Checking**: Prevents logical contradictions
- **Cardinality Constraints**: Validates property usage limits
- **Inference Generation**: Discovers implicit relationships

### 3. Constraint Translator
**Location**: `src/physics/ontology_constraints.rs`

Bridges semantic logic and physics simulation by:
- Converting OWL axioms to physics constraints
- Translating inference results to dynamic forces
- Organizing constraints into logical groups
- Providing configurable constraint strengths

**Translation Mappings**:
| OWL Axiom | Physics Constraint | Visual Effect |
|-----------|-------------------|---------------|
| DisjointClasses(A,B) | Separation force | Push instances apart |
| SubClassOf(A,B) | Hierarchical alignment | Group A near B |
| InverseOf(P,Q) | Bidirectional edges | Symmetric relationships |
| SameAs(a,b) | Co-location force | Pull entities together |
| FunctionalProperty(P) | Cardinality boundaries | Limit connections |

### 4. API Handler
**Location**: `src/handlers/api_handler/ontology/mod.rs`

Provides comprehensive REST and WebSocket APIs for:
- Loading ontology axioms from files/URLs
- Configuring validation parameters
- Running validation jobs (quick/full/incremental)
- Retrieving reports and applying inferences
- Real-time progress updates

## Data Flow

### 1. Ontology Loading
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant OntologyActor
    participant OwlValidatorService
    participant Cache

    Client->>API: POST /api/ontology/load-axioms
    API->>OntologyActor: LoadOntologyAxioms
    OntologyActor->>OwlValidatorService: load_ontology()
    OwlValidatorService->>OwlValidatorService: parse_ontology()
    OwlValidatorService->>Cache: store parsed ontology
    OwlValidatorService-->>OntologyActor: ontology_id
    OntologyActor-->>API: LoadAxiomsResponse
    API-->>Client: 200 OK with ontology_id
```

### 2. Validation Process
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant OntologyActor
    participant OwlValidatorService
    participant ConstraintTranslator
    participant PhysicsOrchestrator

    Client->>API: POST /api/ontology/validate
    API->>OntologyActor: ValidateOntology
    OntologyActor->>OntologyActor: enqueue_validation_job()
    OntologyActor->>OwlValidatorService: validate()
    OwlValidatorService->>OwlValidatorService: map_graph_to_rdf()
    OwlValidatorService->>OwlValidatorService: run_consistency_checks()
    OwlValidatorService->>OwlValidatorService: perform_inference()
    OwlValidatorService-->>OntologyActor: ValidationReport
    OntologyActor->>ConstraintTranslator: translate_to_constraints()
    ConstraintTranslator-->>PhysicsOrchestrator: constraint_updates
    OntologyActor-->>API: ValidationResponse
    API-->>Client: 200 OK with job_id
```

### 3. Graph to RDF Mapping
```mermaid
flowchart TD
    PG[Property Graph]
    MAP[Mapping Configuration<br/>mapping.toml]

    subgraph "Node Processing"
        N1[Node Properties] --> T1[Type Assertions<br/>rdf:type triples]
        N2[Node Metadata] --> D1[Data Properties<br/>Literal values]
    end

    subgraph "Edge Processing"
        E1[Edge Types] --> O1[Object Properties<br/>IRI relationships]
        E2[Edge Properties] --> D2[Reified Statements<br/>Edge metadata]
    end

    subgraph "RDF Output"
        RDF[RDF Triple Store]
        T1 --> RDF
        D1 --> RDF
        O1 --> RDF
        D2 --> RDF
    end

    PG --> N1
    PG --> N2
    PG --> E1
    PG --> E2
    MAP --> N1
    MAP --> N2
    MAP --> E1
    MAP --> E2
```

## Integration Points

### Actor System Integration
The ontology system integrates seamlessly with the existing actor ecosystem:

- **GraphServiceActor**: Provides graph data for validation
- **PhysicsOrchestratorActor**: Receives ontology-derived constraints
- **SemanticProcessorActor**: Processes inference results
- **Message Bus**: Coordinates communication between actors

### Physics Engine Integration
Ontology constraints are translated into physics forces:

```rust
// Example constraint group generation
pub enum OntologyConstraintGroup {
    OntologySeparation,    // Disjoint classes
    OntologyAlignment,     // Hierarchical relationships
    OntologyBoundaries,    // Cardinality limits
    OntologyIdentity,      // Same-as relationships
}
```

### Feature Flag Integration
The system respects the analytics feature flag system:

```rust
pub struct FeatureFlags {
    pub ontology_validation: bool,
    // ... other flags
}
```

## Feature Capabilities

### Validation Modes

1. **Quick Validation** (< 100ms)
   - Basic consistency checks
   - Domain/range validation
   - Minimal inference

2. **Full Validation** (< 5s)
   - Complete consistency analysis
   - Full inference generation
   - Comprehensive reporting

3. **Incremental Validation** (< 50ms)
   - Delta-based updates
   - Cached intermediate results
   - Optimized for frequent changes

### Caching Strategy

The system employs multi-level caching:

- **Ontology Cache**: Parsed OWL structures with TTL
- **Validation Cache**: Reports keyed by graph signature
- **Node Type Cache**: Accelerated type lookups
- **Constraint Cache**: Pre-computed physics constraints

### Error Handling

Comprehensive error reporting with:
- Violation descriptions and locations
- Suggested fixes with examples
- Confidence scores for recommendations
- Actionable resolution steps

## Performance Characteristics

### Scalability Metrics
- **Small Graphs** (< 1k nodes): < 50ms validation
- **Medium Graphs** (1k-10k nodes): < 500ms validation
- **Large Graphs** (10k+ nodes): Background processing with progress updates

### Memory Usage
- **Base overhead**: ~50MB for reasoner components
- **Per-ontology**: ~5-20MB depending on axiom complexity
- **Cache overhead**: ~10KB per validation report
- **Constraint overhead**: ~1KB per generated constraint

### Optimization Features
- **Graph signatures**: Blake3 hashing for change detection
- **Incremental reasoning**: Delta processing for updates
- **Lazy loading**: On-demand ontology parsing
- **Background processing**: Non-blocking validation jobs

## Technology Stack

### Core Dependencies
- **horned-owl**: OWL parsing and manipulation
- **whelk-rs**: High-performance reasoning engine
- **rio-turtle**: RDF/Turtle format support
- **rio-api**: RDF processing abstractions
- **sophia**: Additional RDF handling utilities

### Optional Dependencies
- **oxigraph**: SPARQL query support (feature-gated)
- **blake3**: Fast hashing for signatures
- **dashmap**: Concurrent hash maps for caching

### Rust Ecosystem Integration
- **actix**: Actor system and web framework
- **tokio**: Async runtime for non-blocking operations
- **serde**: Serialization for API communication
- **chrono**: Timestamp handling for reports

## Configuration

The system uses a declarative mapping configuration in `mapping.toml`:

```toml
[global]
base_iri = "https://example.org/graph#"
default_class = "ex:Thing"

[classes.node_type]
"person" = "ex:Person"
"company" = "ex:Company"
"file" = "ex:File"

[properties.edge_type]
"employs" = "ex:employs"
"knows" = "foaf:knows"

[inverses]
"ex:employs" = "ex:worksFor"
"foaf:knows" = "foaf:knows"  # symmetric

[templates]
node_iri = "ex:node/{id}"
edge_iri = "ex:edge/{source}-{target}"
```

This mapping provides flexible translation between the property graph model and formal RDF semantics, enabling powerful reasoning while maintaining usability.