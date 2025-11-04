# InferenceEngine Port

## Purpose

The **InferenceEngine** port provides ontology reasoning and inference capabilities using whelk-rs or similar OWL reasoners. It performs classification, consistency checking, entailment, and explanation.

## Location

- **Trait Definition**: `src/ports/inference_engine.rs`
- **Adapter Implementation**: `src/adapters/whelk_inference_engine.rs`

## Interface

```rust
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    // Ontology loading
    async fn load_ontology(&mut self, classes: Vec<OwlClass>, axioms: Vec<OwlAxiom>) -> Result<()>;
    async fn clear(&mut self) -> Result<()>;

    // Reasoning operations
    async fn infer(&mut self) -> Result<InferenceResults>;
    async fn is_entailed(&self, axiom: &OwlAxiom) -> Result<bool>;
    async fn check_consistency(&self) -> Result<bool>;

    // Classification
    async fn get_subclass_hierarchy(&self) -> Result<Vec<(String, String)>>;
    async fn classify_instance(&self, instance_iri: &str) -> Result<Vec<String>>;

    // Explanation
    async fn explain_entailment(&self, axiom: &OwlAxiom) -> Result<Vec<OwlAxiom>>;

    // Statistics
    async fn get_statistics(&self) -> Result<InferenceStatistics>;
}
```

## Types

### InferenceResults

Inference computation results:

```rust
pub struct InferenceResults {
    pub timestamp: DateTime<Utc>,
    pub inferred_axioms: Vec<OwlAxiom>,
    pub inference_time_ms: u64,
    pub reasoner_version: String,
}
```

### InferenceStatistics

Reasoner statistics:

```rust
pub struct InferenceStatistics {
    pub loaded_classes: usize,
    pub loaded_axioms: usize,
    pub inferred_axioms: usize,
    pub last_inference_time_ms: u64,
    pub total_inferences: u64,
}
```

## Usage Examples

### Basic Reasoning

```rust
let mut engine: Box<dyn InferenceEngine> = Box::new(WhelkInferenceEngine::new());

// Load ontology
let classes = vec![
    OwlClass {
        iri: "http://example.org/Person".to_string(),
        label: Some("Person".to_string()),
        parent_classes: vec!["http://example.org/Agent".to_string()],
        ..Default::default()
    },
    OwlClass {
        iri: "http://example.org/Student".to_string(),
        parent_classes: vec!["http://example.org/Person".to_string()],
        ..Default::default()
    },
];

let axioms = vec![
    OwlAxiom {
        axiom_type: AxiomType::SubClassOf,
        subject: "http://example.org/GraduateStudent".to_string(),
        object: "http://example.org/Student".to_string(),
        ..Default::default()
    },
];

engine.load_ontology(classes, axioms).await?;

// Perform inference
let results = engine.infer().await?;
println!("Inferred {} new axioms in {}ms",
    results.inferred_axioms.len(),
    results.inference_time_ms
);

// Check entailment
let test_axiom = OwlAxiom {
    axiom_type: AxiomType::SubClassOf,
    subject: "http://example.org/GraduateStudent".to_string(),
    object: "http://example.org/Person".to_string(), // Transitively inferred
    ..Default::default()
};

if engine.is_entailed(&test_axiom).await? {
    println!("✓ Axiom is entailed (follows from ontology)");
}
```

### Consistency Checking

```rust
// Check if ontology is consistent
if engine.check_consistency().await? {
    println!("✓ Ontology is consistent");
} else {
    println!("✗ Ontology contains contradictions");
}

// Example of inconsistency
let inconsistent_axioms = vec![
    OwlAxiom {
        axiom_type: AxiomType::DisjointWith,
        subject: "http://example.org/Cat".to_string(),
        object: "http://example.org/Dog".to_string(),
        ..Default::default()
    },
    OwlAxiom {
        axiom_type: AxiomType::SubClassOf,
        subject: "http://example.org/CatDog".to_string(),
        object: "http://example.org/Cat".to_string(),
        ..Default::default()
    },
    OwlAxiom {
        axiom_type: AxiomType::SubClassOf,
        subject: "http://example.org/CatDog".to_string(),
        object: "http://example.org/Dog".to_string(),
        ..Default::default()
    },
];

engine.load_ontology(vec![], inconsistent_axioms).await?;
assert!(!engine.check_consistency().await?); // Should be false
```

### Classification

```rust
// Get complete subclass hierarchy
let hierarchy = engine.get_subclass_hierarchy().await?;
for (child, parent) in hierarchy {
    println!("{} ⊆ {}", child, parent);
}

// Classify an instance
let classes = engine.classify_instance("http://example.org/john").await?;
println!("John is a member of:");
for class_iri in classes {
    println!("  - {}", class_iri);
}
```

### Explanation

```rust
// Explain why an axiom is entailed
let axiom = OwlAxiom {
    axiom_type: AxiomType::SubClassOf,
    subject: "http://example.org/GraduateStudent".to_string(),
    object: "http://example.org/Agent".to_string(),
    ..Default::default()
};

let explanation = engine.explain_entailment(&axiom).await?;
println!("Axiom is entailed because of:");
for supporting_axiom in explanation {
    println!("  {:?}: {} ⊆ {}", supporting_axiom.axiom_type, supporting_axiom.subject, supporting_axiom.object);
}
```

## Implementation Notes

### Whelk Integration

Whelk is a high-performance OWL reasoner written in Rust:

```rust
use whelk::{Reasoner, Ontology};

pub struct WhelkInferenceEngine {
    reasoner: Option<Reasoner>,
    ontology: Option<Ontology>,
    stats: InferenceStatistics,
}

impl WhelkInferenceEngine {
    pub fn new() -> Self {
        Self {
            reasoner: Some(Reasoner::new()),
            ontology: None,
            stats: InferenceStatistics::default(),
        }
    }
}

#[async_trait]
impl InferenceEngine for WhelkInferenceEngine {
    async fn load_ontology(&mut self, classes: Vec<OwlClass>, axioms: Vec<OwlAxiom>) -> Result<()> {
        // Convert to whelk format
        let mut ontology = Ontology::new();

        for class in classes {
            ontology.add_class(&class.iri);
            for parent in class.parent_classes {
                ontology.add_subclass_of(&class.iri, &parent);
            }
        }

        for axiom in axioms {
            match axiom.axiom_type {
                AxiomType::SubClassOf => {
                    ontology.add_subclass_of(&axiom.subject, &axiom.object);
                }
                // ... handle other axiom types
                _ => {}
            }
        }

        self.ontology = Some(ontology);
        self.stats.loaded_classes = classes.len();
        self.stats.loaded_axioms = axioms.len();

        Ok(())
    }

    async fn infer(&mut self) -> Result<InferenceResults> {
        let start = Instant::now();

        let reasoner = self.reasoner.as_mut()
            .ok_or(InferenceEngineError::OntologyNotLoaded)?;
        let ontology = self.ontology.as_ref()
            .ok_or(InferenceEngineError::OntologyNotLoaded)?;

        // Perform reasoning
        let inferences = reasoner.classify(ontology);

        let inference_time = start.elapsed().as_millis() as u64;

        // Convert whelk inferences to OwlAxiom
        let inferred_axioms = inferences.iter().map(|inf| {
            OwlAxiom {
                id: None,
                axiom_type: AxiomType::SubClassOf,
                subject: inf.subclass.clone(),
                object: inf.superclass.clone(),
                annotations: HashMap::new(),
            }
        }).collect();

        let results = InferenceResults {
            timestamp: Utc::now(),
            inferred_axioms,
            inference_time_ms: inference_time,
            reasoner_version: "whelk-0.1.0".to_string(),
        };

        self.stats.inferred_axioms = results.inferred_axioms.len();
        self.stats.last_inference_time_ms = inference_time;
        self.stats.total_inferences += 1;

        Ok(results)
    }
}
```

### Performance Optimization

1. **Incremental Reasoning**: Only re-reason changed parts
2. **Caching**: Cache classification results
3. **Parallel Processing**: Use rayon for parallel classification
4. **Memory Management**: Clear reasoner state when not in use

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum InferenceEngineError {
    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Ontology not loaded")]
    OntologyNotLoaded,

    #[error("Inconsistent ontology: {0}")]
    InconsistentOntology(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Reasoner error: {0}")]
    ReasonerError(String),
}
```

## Testing

### Mock Implementation

```rust
pub struct MockInferenceEngine {
    classes: Vec<OwlClass>,
    axioms: Vec<OwlAxiom>,
    inferred: Vec<OwlAxiom>,
}

#[async_trait]
impl InferenceEngine for MockInferenceEngine {
    async fn load_ontology(&mut self, classes: Vec<OwlClass>, axioms: Vec<OwlAxiom>) -> Result<()> {
        self.classes = classes;
        self.axioms = axioms;
        Ok(())
    }

    async fn infer(&mut self) -> Result<InferenceResults> {
        // Simple transitive closure for testing
        let mut inferred = Vec::new();

        // For each SubClassOf axiom A ⊆ B
        // And each SubClassOf axiom B ⊆ C
        // Infer A ⊆ C
        for axiom1 in &self.axioms {
            if axiom1.axiom_type == AxiomType::SubClassOf {
                for axiom2 in &self.axioms {
                    if axiom2.axiom_type == AxiomType::SubClassOf
                        && axiom1.object == axiom2.subject
                    {
                        inferred.push(OwlAxiom {
                            axiom_type: AxiomType::SubClassOf,
                            subject: axiom1.subject.clone(),
                            object: axiom2.object.clone(),
                            ..Default::default()
                        });
                    }
                }
            }
        }

        self.inferred = inferred.clone();

        Ok(InferenceResults {
            timestamp: Utc::now(),
            inferred_axioms: inferred,
            inference_time_ms: 10,
            reasoner_version: "mock-1.0.0".to_string(),
        })
    }

    async fn is_entailed(&self, axiom: &OwlAxiom) -> Result<bool> {
        Ok(self.axioms.iter().any(|a| a.subject == axiom.subject && a.object == axiom.object)
            || self.inferred.iter().any(|a| a.subject == axiom.subject && a.object == axiom.object))
    }

    async fn check_consistency(&self) -> Result<bool> {
        // Mock always consistent
        Ok(true)
    }

    // ... implement remaining methods
}
```

## Performance Benchmarks

Target performance (Whelk adapter):
- Load ontology (100 classes): < 50ms
- Inference (100 classes, 200 axioms): < 200ms
- Entailment check: < 1ms
- Consistency check: < 100ms

## References

- **OWL 2 Primer**: https://www.w3.org/TR/owl2-primer/
- **Whelk Reasoner**: https://github.com/balhoff/whelk
- **Description Logics**: https://arxiv.org/abs/cs/0007031

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
