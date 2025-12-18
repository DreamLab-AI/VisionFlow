---
title: OntologyRepository Port
description: The **OntologyRepository** port manages the ontology graph structure parsed from GitHub markdown files. It handles OWL (Web Ontology Language) classes, properties, axioms, inference results, and pa...
category: explanation
tags:
  - architecture
  - rust
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: advanced
---


# OntologyRepository Port

## Purpose

The **OntologyRepository** port manages the ontology graph structure parsed from GitHub markdown files. It handles OWL (Web Ontology Language) classes, properties, axioms, inference results, and pathfinding caches.

## Location

- **Trait Definition**: `src/ports/ontology-repository.rs`
- **Adapter Implementation**: `src/adapters/sqlite-ontology-repository.rs`

## Interface

```rust
#[async-trait]
pub trait OntologyRepository: Send + Sync {
    // Graph operations
    async fn load-ontology-graph(&self) -> Result<Arc<GraphData>>;
    async fn save-ontology-graph(&self, graph: &GraphData) -> Result<()>;

    // Batch ontology import
    async fn save-ontology(
        &self,
        classes: &[OwlClass],
        properties: &[OwlProperty],
        axioms: &[OwlAxiom],
    ) -> Result<()>;

    // OWL Class operations
    async fn add-owl-class(&self, class: &OwlClass) -> Result<String>;
    async fn get-owl-class(&self, iri: &str) -> Result<Option<OwlClass>>;
    async fn list-owl-classes(&self) -> Result<Vec<OwlClass>>;

    // OWL Property operations
    async fn add-owl-property(&self, property: &OwlProperty) -> Result<String>;
    async fn get-owl-property(&self, iri: &str) -> Result<Option<OwlProperty>>;
    async fn list-owl-properties(&self) -> Result<Vec<OwlProperty>>;

    // Axiom operations
    async fn add-axiom(&self, axiom: &OwlAxiom) -> Result<u64>;
    async fn get-class-axioms(&self, class-iri: &str) -> Result<Vec<OwlAxiom>>;

    // Inference operations
    async fn store-inference-results(&self, results: &InferenceResults) -> Result<()>;
    async fn get-inference-results(&self) -> Result<Option<InferenceResults>>;

    // Validation and queries
    async fn validate-ontology(&self) -> Result<ValidationReport>;
    async fn query-ontology(&self, query: &str) -> Result<Vec<HashMap<String, String>>>;
    async fn get-metrics(&self) -> Result<OntologyMetrics>;

    // Pathfinding caches
    async fn cache-sssp-result(&self, entry: &PathfindingCacheEntry) -> Result<()>;
    async fn get-cached-sssp(&self, source-node-id: u32) -> Result<Option<PathfindingCacheEntry>>;
    async fn cache-apsp-result(&self, distance-matrix: &Vec<Vec<f32>>) -> Result<()>;
    async fn get-cached-apsp(&self) -> Result<Option<Vec<Vec<f32>>>>;
    async fn invalidate-pathfinding-caches(&self) -> Result<()>;
}
```

## Types

### OwlClass

OWL class definition with hierarchy and raw markdown storage:

```rust
pub struct OwlClass {
    pub iri: String,                    // Class IRI (e.g., "http://example.org/MyClass")
    pub label: Option<String>,          // Human-readable label
    pub description: Option<String>,    // Class description
    pub parent-classes: Vec<String>,    // IRIs of parent classes
    pub properties: HashMap<String, String>, // Additional properties
    pub source-file: Option<String>,    // Source markdown file

    // NEW: Raw markdown storage for zero semantic loss
    pub markdown-content: Option<String>,     // Full markdown with OWL blocks
    pub file-sha1: Option<String>,           // SHA1 hash for change detection
    pub last-synced: Option<DateTime<Utc>>,  // Last GitHub sync timestamp
}
```

**Architecture Note**: The `markdown-content` field stores complete markdown files including embedded OWL Functional Syntax blocks. This enables downstream parsing with horned-owl without semantic loss. See [Ontology Storage Architecture](../ontology-storage-architecture.md) for details.

### OwlProperty

OWL property with domain and range:

```rust
pub struct OwlProperty {
    pub iri: String,
    pub label: Option<String>,
    pub property-type: PropertyType,
    pub domain: Vec<String>,  // Class IRIs
    pub range: Vec<String>,   // Class/Datatype IRIs
}

pub enum PropertyType {
    ObjectProperty,       // Links between individuals
    DataProperty,         // Links to literal values
    AnnotationProperty,   // Metadata annotations
}
```

### OwlAxiom

OWL axiom (logical statement):

```rust
pub struct OwlAxiom {
    pub id: Option<u64>,
    pub axiom-type: AxiomType,
    pub subject: String,      // Subject IRI
    pub object: String,       // Object IRI or value
    pub annotations: HashMap<String, String>,
}

pub enum AxiomType {
    SubClassOf,           // A ⊆ B
    EquivalentClass,      // A ≡ B
    DisjointWith,         // A ⊓ B = ⊥
    ObjectPropertyAssertion,  // individual-property-individual
    DataPropertyAssertion,    // individual-property-value
}
```

### InferenceResults

Results from reasoning engine:

```rust
pub struct InferenceResults {
    pub timestamp: DateTime<Utc>,
    pub inferred-axioms: Vec<OwlAxiom>,
    pub inference-time-ms: u64,
    pub reasoner-version: String,
}
```

### ValidationReport

Ontology consistency check:

```rust
pub struct ValidationReport {
    pub is-valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub timestamp: DateTime<Utc>,
}
```

### PathfindingCacheEntry

SSSP cache for GPU pathfinding:

```rust
pub struct PathfindingCacheEntry {
    pub source-node-id: u32,
    pub target-node-id: Option<u32>,
    pub distances: Vec<f32>,
    pub paths: HashMap<u32, Vec<u32>>,
    pub computed-at: DateTime<Utc>,
    pub computation-time-ms: f32,
}
```

## Usage Examples

### Batch Ontology Import

```rust
let repo: Arc<dyn OntologyRepository> = Arc::new(SqliteOntologyRepository::new(pool));

// Import complete ontology from GitHub sync
let classes = vec![
    OwlClass {
        iri: "http://example.org/Person".to-string(),
        label: Some("Person".to-string()),
        description: Some("A human being".to-string()),
        parent-classes: vec!["http://example.org/Agent".to-string()],
        properties: HashMap::new(),
        source-file: Some("ontology/person.md".to-string()),
    },
    // ... more classes
];

let properties = vec![
    OwlProperty {
        iri: "http://example.org/hasName".to-string(),
        label: Some("has name".to-string()),
        property-type: PropertyType::DataProperty,
        domain: vec!["http://example.org/Person".to-string()],
        range: vec!["http://www.w3.org/2001/XMLSchema#string".to-string()],
    },
];

let axioms = vec![
    OwlAxiom {
        id: None,
        axiom-type: AxiomType::SubClassOf,
        subject: "http://example.org/Student".to-string(),
        object: "http://example.org/Person".to-string(),
        annotations: HashMap::new(),
    },
];

// Atomic import with transaction
repo.save-ontology(&classes, &properties, &axioms).await?;
```

### OWL Class Operations

```rust
// Add a class
let class = OwlClass {
    iri: "http://example.org/Book".to-string(),
    label: Some("Book".to-string()),
    description: Some("A published work".to-string()),
    parent-classes: vec!["http://example.org/Publication".to-string()],
    properties: HashMap::new(),
    source-file: Some("ontology/book.md".to-string()),
};

let class-iri = repo.add-owl-class(&class).await?;

// Get a class
if let Some(class) = repo.get-owl-class("http://example.org/Book").await? {
    println!("Class: {}", class.label.unwrap-or-default());
    println!("Parents: {:?}", class.parent-classes);
}

// List all classes
let all-classes = repo.list-owl-classes().await?;
println!("Total classes: {}", all-classes.len());
```

### Axiom Operations

```rust
// Add axiom
let axiom = OwlAxiom {
    id: None,
    axiom-type: AxiomType::SubClassOf,
    subject: "http://example.org/Novel".to-string(),
    object: "http://example.org/Book".to-string(),
    annotations: HashMap::new(),
};

let axiom-id = repo.add-axiom(&axiom).await?;

// Get axioms for a class
let axioms = repo.get-class-axioms("http://example.org/Novel").await?;
for axiom in axioms {
    println!("Axiom: {:?} {} {}", axiom.axiom-type, axiom.subject, axiom.object);
}
```

### Inference Results

```rust
// Store inference results from reasoner
let results = InferenceResults {
    timestamp: Utc::now(),
    inferred-axioms: vec![/* ... */],
    inference-time-ms: 1500,
    reasoner-version: "whelk-0.1.0".to-string(),
};

repo.store-inference-results(&results).await?;

// Retrieve latest inference
if let Some(results) = repo.get-inference-results().await? {
    println!("Inferred {} axioms in {}ms",
        results.inferred-axioms.len(),
        results.inference-time-ms
    );
}
```

### Validation

```rust
// Validate ontology consistency
let report = repo.validate-ontology().await?;

if report.is-valid {
    println!("✓ Ontology is consistent");
} else {
    println!("✗ Ontology has errors:");
    for error in &report.errors {
        println!("  - {}", error);
    }
}

if !report.warnings.is-empty() {
    println!("Warnings:");
    for warning in &report.warnings {
        println!("  ! {}", warning);
    }
}
```

### Pathfinding Cache

```rust
// Cache SSSP result from GPU computation
let cache-entry = PathfindingCacheEntry {
    source-node-id: 42,
    target-node-id: None, // All targets
    distances: vec![0.0, 1.5, 2.3, 3.1],
    paths: HashMap::from([
        (1, vec![42, 1]),
        (2, vec![42, 1, 2]),
    ]),
    computed-at: Utc::now(),
    computation-time-ms: 12.5,
};

repo.cache-sssp-result(&cache-entry).await?;

// Retrieve cached SSSP
if let Some(cached) = repo.get-cached-sssp(42).await? {
    println!("Found cached SSSP for node 42");
    println!("Distance to node 2: {}", cached.distances[2]);
}

// Cache APSP distance matrix
let distance-matrix = vec![
    vec![0.0, 1.0, 2.0],
    vec![1.0, 0.0, 1.5],
    vec![2.0, 1.5, 0.0],
];
repo.cache-apsp-result(&distance-matrix).await?;

// Invalidate caches after graph changes
repo.invalidate-pathfinding-caches().await?;
```

### Ontology Queries

```rust
// SPARQL-like query
let results = repo.query-ontology(
    "SELECT ?class WHERE { ?class SubClassOf http://example.org/Person }"
).await?;

for result in results {
    println!("Subclass: {}", result.get("class").unwrap());
}

// Get ontology metrics
let metrics = repo.get-metrics().await?;
println!("Classes: {}", metrics.class-count);
println!("Properties: {}", metrics.property-count);
println!("Axioms: {}", metrics.axiom-count);
println!("Max depth: {}", metrics.max-depth);
println!("Avg branching: {}", metrics.average-branching-factor);
```

## Implementation Notes

### Atomic Ontology Import

The `save-ontology` method should use a single transaction:

```rust
async fn save-ontology(
    &self,
    classes: &[OwlClass],
    properties: &[OwlProperty],
    axioms: &[OwlAxiom],
) -> Result<()> {
    let conn = self.pool.get()?;
    let tx = conn.transaction()?;

    // Clear existing data
    tx.execute("DELETE FROM owl-classes", [])?;
    tx.execute("DELETE FROM owl-properties", [])?;
    tx.execute("DELETE FROM owl-axioms", [])?;

    // Insert new data
    for class in classes {
        // Insert class
    }
    for property in properties {
        // Insert property
    }
    for axiom in axioms {
        // Insert axiom
    }

    tx.commit()?;
    Ok(())
}
```

### IRI Validation

Validate IRIs before storage:

```rust
fn validate-iri(iri: &str) -> Result<()> {
    if !iri.starts-with("http://") && !iri.starts-with("https://") {
        return Err(OntologyRepositoryError::InvalidData(
            format!("Invalid IRI: {}", iri)
        ));
    }
    Ok(())
}
```

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS owl-classes (
    iri TEXT PRIMARY KEY,
    label TEXT,
    description TEXT,
    parent-classes TEXT, -- JSON array
    properties TEXT,     -- JSON object
    source-file TEXT,
    created-at DATETIME DEFAULT CURRENT-TIMESTAMP
);

CREATE TABLE IF NOT EXISTS owl-properties (
    iri TEXT PRIMARY KEY,
    label TEXT,
    property-type TEXT NOT NULL, -- 'object', 'data', 'annotation'
    domain TEXT,   -- JSON array of IRIs
    range TEXT,    -- JSON array of IRIs
    created-at DATETIME DEFAULT CURRENT-TIMESTAMP
);

CREATE TABLE IF NOT EXISTS owl-axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    axiom-type TEXT NOT NULL,
    subject TEXT NOT NULL,
    object TEXT NOT NULL,
    annotations TEXT, -- JSON object
    created-at DATETIME DEFAULT CURRENT-TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inference-results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    inferred-axioms TEXT NOT NULL, -- JSON
    inference-time-ms INTEGER NOT NULL,
    reasoner-version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pathfinding-cache (
    source-node-id INTEGER PRIMARY KEY,
    target-node-id INTEGER,
    distances BLOB NOT NULL,
    paths TEXT NOT NULL, -- JSON
    computed-at DATETIME NOT NULL,
    computation-time-ms REAL NOT NULL
);

CREATE INDEX idx-owl-classes-label ON owl-classes(label);
CREATE INDEX idx-owl-axioms-subject ON owl-axioms(subject);
CREATE INDEX idx-owl-axioms-type ON owl-axioms(axiom-type);
```

## Performance Considerations

### Benchmarks

Target performance:
- Batch import (1000 classes): < 500ms
- Get class by IRI: < 5ms
- List all classes: < 50ms
- Store inference results: < 20ms
- Cache SSSP result: < 10ms

---

---

## Related Documentation

- [InferenceEngine Port](05-inference-engine.md)
- [GpuSemanticAnalyzer Port](07-gpu-semantic-analyzer.md)
- [GpuPhysicsAdapter Port](06-gpu-physics-adapter.md)
- [Semantic Physics Architecture](../semantic-physics.md)
- [Stress Majorization for GPU-Accelerated Graph Layout](../stress-majorization.md)

## References

- **OWL 2 Specification**: https://www.w3.org/TR/owl2-overview/
- **SPARQL Query Language**: https://www.w3.org/TR/sparql11-query/
- **Ontology Design Patterns**: http://ontologydesignpatterns.org/

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
