# Inference API Reference

## InferenceEngine Trait

Core trait for ontology reasoning operations.

### Methods

#### `load_ontology`
```rust
async fn load_ontology(
    &mut self,
    classes: Vec<OwlClass>,
    axioms: Vec<OwlAxiom>
) -> Result<()>
```

Load ontology classes and axioms for reasoning.

**Parameters:**
- `classes`: OWL classes to load
- `axioms`: OWL axioms (SubClassOf, EquivalentClass, etc.)

**Returns:** `Result<()>`

---

#### `infer`
```rust
async fn infer(&mut self) -> Result<InferenceResults>
```

Perform inference to derive new axioms.

**Returns:** `InferenceResults` containing:
- `inferred_axioms`: Newly derived axioms
- `inference_time_ms`: Time taken
- `reasoner_version`: Version string
- `timestamp`: When inference was run

---

#### `is_entailed`
```rust
async fn is_entailed(&self, axiom: &OwlAxiom) -> Result<bool>
```

Check if an axiom is entailed by the ontology.

**Returns:** `true` if entailed, `false` otherwise

---

#### `get_subclass_hierarchy`
```rust
async fn get_subclass_hierarchy(&self) -> Result<Vec<(String, String)>>
```

Get all inferred subclass relationships.

**Returns:** Vector of `(child_iri, parent_iri)` pairs

---

#### `classify_instance`
```rust
async fn classify_instance(&self, instance_iri: &str) -> Result<Vec<String>>
```

Classify an instance into classes.

**Returns:** Vector of class IRIs the instance belongs to

---

#### `check_consistency`
```rust
async fn check_consistency(&self) -> Result<bool>
```

Check if the ontology is consistent.

**Returns:** `true` if consistent, `false` if inconsistent

---

#### `explain_entailment`
```rust
async fn explain_entailment(&self, axiom: &OwlAxiom) -> Result<Vec<OwlAxiom>>
```

Explain why an axiom is entailed.

**Returns:** Supporting axioms that lead to the entailment

---

#### `clear`
```rust
async fn clear(&mut self) -> Result<()>
```

Clear loaded ontology from memory.

---

#### `get_statistics`
```rust
async fn get_statistics(&self) -> Result<InferenceStatistics>
```

Get inference engine statistics.

**Returns:** `InferenceStatistics` with metrics

---

## InferenceService

Application service for high-level inference operations.

### Methods

#### `run_inference`
```rust
async fn run_inference(&self, ontology_id: &str) -> EngineResult<InferenceResults>
```

Run complete inference pipeline:
1. Load ontology from repository
2. Check cache
3. Run inference if needed
4. Store results
5. Publish events

---

#### `validate_ontology`
```rust
async fn validate_ontology(&self, ontology_id: &str) -> EngineResult<ValidationResult>
```

Validate ontology consistency and find unsatisfiable classes.

---

#### `classify_ontology`
```rust
async fn classify_ontology(&self, ontology_id: &str) -> EngineResult<ClassificationResult>
```

Classify ontology and extract hierarchy with equivalent classes.

---

#### `batch_inference`
```rust
async fn batch_inference(
    &self,
    ontology_ids: Vec<String>
) -> EngineResult<HashMap<String, InferenceResults>>
```

Run inference on multiple ontologies in parallel.

---

#### `invalidate_cache`
```rust
async fn invalidate_cache(&self, ontology_id: &str)
```

Invalidate cached results for an ontology.

---

## OWLParser

Parse OWL ontologies in various formats.

### Methods

#### `parse`
```rust
pub fn parse(content: &str) -> Result<ParseResult, ParseError>
```

Parse OWL with automatic format detection.

---

#### `parse_with_format`
```rust
pub fn parse_with_format(
    content: &str,
    format: OWLFormat
) -> Result<ParseResult, ParseError>
```

Parse OWL with explicit format specification.

**Supported Formats:**
- `OWLFormat::OwlXml` - OWL/XML
- `OWLFormat::RdfXml` - RDF/XML
- `OWLFormat::Turtle` - Turtle
- `OWLFormat::Manchester` - Manchester (partial)

---

#### `detect_format`
```rust
pub fn detect_format(content: &str) -> OWLFormat
```

Auto-detect OWL format from content.

---

## InferenceCache

LRU cache for inference results.

### Methods

#### `get`
```rust
async fn get(&self, ontology_id: &str, checksum: &str) -> Option<InferenceResults>
```

Get cached results if valid.

---

#### `put`
```rust
async fn put(
    &self,
    ontology_id: String,
    checksum: String,
    results: InferenceResults
)
```

Store results in cache.

---

#### `invalidate`
```rust
async fn invalidate(&self, ontology_id: &str)
```

Invalidate cache entry.

---

#### `get_statistics`
```rust
async fn get_statistics(&self) -> CacheStatistics
```

Get cache performance metrics.

---

## Types

### InferenceResults
```rust
pub struct InferenceResults {
    pub timestamp: DateTime<Utc>,
    pub inferred_axioms: Vec<OwlAxiom>,
    pub inference_time_ms: u64,
    pub reasoner_version: String,
}
```

### ValidationResult
```rust
pub struct ValidationResult {
    pub consistent: bool,
    pub unsatisfiable: Vec<UnsatisfiableClass>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub validation_time_ms: u64,
}
```

### ClassificationResult
```rust
pub struct ClassificationResult {
    pub hierarchy: Vec<(String, String)>,
    pub equivalent_classes: Vec<Vec<String>>,
    pub classification_time_ms: u64,
    pub inferred_count: usize,
}
```

### InferenceStatistics
```rust
pub struct InferenceStatistics {
    pub loaded_classes: usize,
    pub loaded_axioms: usize,
    pub inferred_axioms: usize,
    pub last_inference_time_ms: u64,
    pub total_inferences: u64,
}
```

---

## Error Types

### InferenceEngineError
```rust
pub enum InferenceEngineError {
    InferenceError(String),
    OntologyNotLoaded,
    InconsistentOntology(String),
    UnsupportedOperation(String),
    ReasonerError(String),
}
```

### ParseError
```rust
pub enum ParseError {
    UnsupportedFormat(String),
    ParseError(String),
    IoError(std::io::Error),
    InvalidSyntax(String),
    FeatureNotEnabled,
}
```

---

## Events

### InferenceEvent
```rust
pub enum InferenceEvent {
    InferenceStarted { ontology_id: String },
    InferenceCompleted {
        ontology_id: String,
        inference_count: usize,
        duration_ms: u64,
    },
    InferenceFailed {
        ontology_id: String,
        error: String,
    },
    ValidationCompleted {
        ontology_id: String,
        consistent: bool,
    },
    ClassificationCompleted {
        ontology_id: String,
        hierarchy_count: usize,
    },
}
```

### OntologyEvent
```rust
pub enum OntologyEvent {
    OntologyImported {
        ontology_id: String,
        class_count: usize,
        axiom_count: usize,
    },
    ClassAdded {
        ontology_id: String,
        class_iri: String,
    },
    AxiomAdded {
        ontology_id: String,
        axiom_id: String,
    },
    OntologyModified {
        ontology_id: String,
        change_type: String,
    },
}
```
