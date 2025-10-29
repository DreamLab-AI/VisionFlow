# OntologyRepository Port

## Purpose

The **OntologyRepository** port manages the ontology graph structure parsed from GitHub markdown files. It handles OWL (Web Ontology Language) classes, properties, axioms, inference results, and pathfinding caches.

## Location

- **Trait Definition**: `src/ports/ontology_repository.rs`
- **Adapter Implementation**: `src/adapters/sqlite_ontology_repository.rs`

## Interface

```rust
#[async_trait]
pub trait OntologyRepository: Send + Sync {
    // Graph operations
    async fn load_ontology_graph(&self) -> Result<Arc<GraphData>>;
    async fn save_ontology_graph(&self, graph: &GraphData) -> Result<()>;

    // Batch ontology import
    async fn save_ontology(
        &self,
        classes: &[OwlClass],
        properties: &[OwlProperty],
        axioms: &[OwlAxiom],
    ) -> Result<()>;

    // OWL Class operations
    async fn add_owl_class(&self, class: &OwlClass) -> Result<String>;
    async fn get_owl_class(&self, iri: &str) -> Result<Option<OwlClass>>;
    async fn list_owl_classes(&self) -> Result<Vec<OwlClass>>;

    // OWL Property operations
    async fn add_owl_property(&self, property: &OwlProperty) -> Result<String>;
    async fn get_owl_property(&self, iri: &str) -> Result<Option<OwlProperty>>;
    async fn list_owl_properties(&self) -> Result<Vec<OwlProperty>>;

    // Axiom operations
    async fn add_axiom(&self, axiom: &OwlAxiom) -> Result<u64>;
    async fn get_class_axioms(&self, class_iri: &str) -> Result<Vec<OwlAxiom>>;

    // Inference operations
    async fn store_inference_results(&self, results: &InferenceResults) -> Result<()>;
    async fn get_inference_results(&self) -> Result<Option<InferenceResults>>;

    // Validation and queries
    async fn validate_ontology(&self) -> Result<ValidationReport>;
    async fn query_ontology(&self, query: &str) -> Result<Vec<HashMap<String, String>>>;
    async fn get_metrics(&self) -> Result<OntologyMetrics>;

    // Pathfinding caches
    async fn cache_sssp_result(&self, entry: &PathfindingCacheEntry) -> Result<()>;
    async fn get_cached_sssp(&self, source_node_id: u32) -> Result<Option<PathfindingCacheEntry>>;
    async fn cache_apsp_result(&self, distance_matrix: &Vec<Vec<f32>>) -> Result<()>;
    async fn get_cached_apsp(&self) -> Result<Option<Vec<Vec<f32>>>>;
    async fn invalidate_pathfinding_caches(&self) -> Result<()>;
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
    pub parent_classes: Vec<String>,    // IRIs of parent classes
    pub properties: HashMap<String, String>, // Additional properties
    pub source_file: Option<String>,    // Source markdown file

    // NEW: Raw markdown storage for zero semantic loss
    pub markdown_content: Option<String>,     // Full markdown with OWL blocks
    pub file_sha1: Option<String>,           // SHA1 hash for change detection
    pub last_synced: Option<DateTime<Utc>>,  // Last GitHub sync timestamp
}
```

**Architecture Note**: The `markdown_content` field stores complete markdown files including embedded OWL Functional Syntax blocks. This enables downstream parsing with horned-owl without semantic loss. See [Ontology Storage Architecture](../ontology-storage-architecture.md) for details.

### OwlProperty

OWL property with domain and range:

```rust
pub struct OwlProperty {
    pub iri: String,
    pub label: Option<String>,
    pub property_type: PropertyType,
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
    pub axiom_type: AxiomType,
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
    pub inferred_axioms: Vec<OwlAxiom>,
    pub inference_time_ms: u64,
    pub reasoner_version: String,
}
```

### ValidationReport

Ontology consistency check:

```rust
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub timestamp: DateTime<Utc>,
}
```

### PathfindingCacheEntry

SSSP cache for GPU pathfinding:

```rust
pub struct PathfindingCacheEntry {
    pub source_node_id: u32,
    pub target_node_id: Option<u32>,
    pub distances: Vec<f32>,
    pub paths: HashMap<u32, Vec<u32>>,
    pub computed_at: DateTime<Utc>,
    pub computation_time_ms: f32,
}
```

## Usage Examples

### Batch Ontology Import

```rust
let repo: Arc<dyn OntologyRepository> = Arc::new(SqliteOntologyRepository::new(pool));

// Import complete ontology from GitHub sync
let classes = vec![
    OwlClass {
        iri: "http://example.org/Person".to_string(),
        label: Some("Person".to_string()),
        description: Some("A human being".to_string()),
        parent_classes: vec!["http://example.org/Agent".to_string()],
        properties: HashMap::new(),
        source_file: Some("ontology/person.md".to_string()),
    },
    // ... more classes
];

let properties = vec![
    OwlProperty {
        iri: "http://example.org/hasName".to_string(),
        label: Some("has name".to_string()),
        property_type: PropertyType::DataProperty,
        domain: vec!["http://example.org/Person".to_string()],
        range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
    },
];

let axioms = vec![
    OwlAxiom {
        id: None,
        axiom_type: AxiomType::SubClassOf,
        subject: "http://example.org/Student".to_string(),
        object: "http://example.org/Person".to_string(),
        annotations: HashMap::new(),
    },
];

// Atomic import with transaction
repo.save_ontology(&classes, &properties, &axioms).await?;
```

### OWL Class Operations

```rust
// Add a class
let class = OwlClass {
    iri: "http://example.org/Book".to_string(),
    label: Some("Book".to_string()),
    description: Some("A published work".to_string()),
    parent_classes: vec!["http://example.org/Publication".to_string()],
    properties: HashMap::new(),
    source_file: Some("ontology/book.md".to_string()),
};

let class_iri = repo.add_owl_class(&class).await?;

// Get a class
if let Some(class) = repo.get_owl_class("http://example.org/Book").await? {
    println!("Class: {}", class.label.unwrap_or_default());
    println!("Parents: {:?}", class.parent_classes);
}

// List all classes
let all_classes = repo.list_owl_classes().await?;
println!("Total classes: {}", all_classes.len());
```

### Axiom Operations

```rust
// Add axiom
let axiom = OwlAxiom {
    id: None,
    axiom_type: AxiomType::SubClassOf,
    subject: "http://example.org/Novel".to_string(),
    object: "http://example.org/Book".to_string(),
    annotations: HashMap::new(),
};

let axiom_id = repo.add_axiom(&axiom).await?;

// Get axioms for a class
let axioms = repo.get_class_axioms("http://example.org/Novel").await?;
for axiom in axioms {
    println!("Axiom: {:?} {} {}", axiom.axiom_type, axiom.subject, axiom.object);
}
```

### Inference Results

```rust
// Store inference results from reasoner
let results = InferenceResults {
    timestamp: Utc::now(),
    inferred_axioms: vec![/* ... */],
    inference_time_ms: 1500,
    reasoner_version: "whelk-0.1.0".to_string(),
};

repo.store_inference_results(&results).await?;

// Retrieve latest inference
if let Some(results) = repo.get_inference_results().await? {
    println!("Inferred {} axioms in {}ms",
        results.inferred_axioms.len(),
        results.inference_time_ms
    );
}
```

### Validation

```rust
// Validate ontology consistency
let report = repo.validate_ontology().await?;

if report.is_valid {
    println!("✓ Ontology is consistent");
} else {
    println!("✗ Ontology has errors:");
    for error in &report.errors {
        println!("  - {}", error);
    }
}

if !report.warnings.is_empty() {
    println!("Warnings:");
    for warning in &report.warnings {
        println!("  ! {}", warning);
    }
}
```

### Pathfinding Cache

```rust
// Cache SSSP result from GPU computation
let cache_entry = PathfindingCacheEntry {
    source_node_id: 42,
    target_node_id: None, // All targets
    distances: vec![0.0, 1.5, 2.3, 3.1],
    paths: HashMap::from([
        (1, vec![42, 1]),
        (2, vec![42, 1, 2]),
    ]),
    computed_at: Utc::now(),
    computation_time_ms: 12.5,
};

repo.cache_sssp_result(&cache_entry).await?;

// Retrieve cached SSSP
if let Some(cached) = repo.get_cached_sssp(42).await? {
    println!("Found cached SSSP for node 42");
    println!("Distance to node 2: {}", cached.distances[2]);
}

// Cache APSP distance matrix
let distance_matrix = vec![
    vec![0.0, 1.0, 2.0],
    vec![1.0, 0.0, 1.5],
    vec![2.0, 1.5, 0.0],
];
repo.cache_apsp_result(&distance_matrix).await?;

// Invalidate caches after graph changes
repo.invalidate_pathfinding_caches().await?;
```

### Ontology Queries

```rust
// SPARQL-like query
let results = repo.query_ontology(
    "SELECT ?class WHERE { ?class SubClassOf http://example.org/Person }"
).await?;

for result in results {
    println!("Subclass: {}", result.get("class").unwrap());
}

// Get ontology metrics
let metrics = repo.get_metrics().await?;
println!("Classes: {}", metrics.class_count);
println!("Properties: {}", metrics.property_count);
println!("Axioms: {}", metrics.axiom_count);
println!("Max depth: {}", metrics.max_depth);
println!("Avg branching: {}", metrics.average_branching_factor);
```

## Implementation Notes

### Atomic Ontology Import

The `save_ontology` method should use a single transaction:

```rust
async fn save_ontology(
    &self,
    classes: &[OwlClass],
    properties: &[OwlProperty],
    axioms: &[OwlAxiom],
) -> Result<()> {
    let conn = self.pool.get()?;
    let tx = conn.transaction()?;

    // Clear existing data
    tx.execute("DELETE FROM owl_classes", [])?;
    tx.execute("DELETE FROM owl_properties", [])?;
    tx.execute("DELETE FROM owl_axioms", [])?;

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
fn validate_iri(iri: &str) -> Result<()> {
    if !iri.starts_with("http://") && !iri.starts_with("https://") {
        return Err(OntologyRepositoryError::InvalidData(
            format!("Invalid IRI: {}", iri)
        ));
    }
    Ok(())
}
```

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS owl_classes (
    iri TEXT PRIMARY KEY,
    label TEXT,
    description TEXT,
    parent_classes TEXT, -- JSON array
    properties TEXT,     -- JSON object
    source_file TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS owl_properties (
    iri TEXT PRIMARY KEY,
    label TEXT,
    property_type TEXT NOT NULL, -- 'object', 'data', 'annotation'
    domain TEXT,   -- JSON array of IRIs
    range TEXT,    -- JSON array of IRIs
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS owl_axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    axiom_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    object TEXT NOT NULL,
    annotations TEXT, -- JSON object
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    inferred_axioms TEXT NOT NULL, -- JSON
    inference_time_ms INTEGER NOT NULL,
    reasoner_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pathfinding_cache (
    source_node_id INTEGER PRIMARY KEY,
    target_node_id INTEGER,
    distances BLOB NOT NULL,
    paths TEXT NOT NULL, -- JSON
    computed_at DATETIME NOT NULL,
    computation_time_ms REAL NOT NULL
);

CREATE INDEX idx_owl_classes_label ON owl_classes(label);
CREATE INDEX idx_owl_axioms_subject ON owl_axioms(subject);
CREATE INDEX idx_owl_axioms_type ON owl_axioms(axiom_type);
```

## Performance Considerations

### Benchmarks

Target performance:
- Batch import (1000 classes): < 500ms
- Get class by IRI: < 5ms
- List all classes: < 50ms
- Store inference results: < 20ms
- Cache SSSP result: < 10ms

## References

- **OWL 2 Specification**: https://www.w3.org/TR/owl2-overview/
- **SPARQL Query Language**: https://www.w3.org/TR/sparql11-query/
- **Ontology Design Patterns**: http://ontologydesignpatterns.org/

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
