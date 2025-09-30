# Ontology Constraints Translator

The `OntologyConstraintTranslator` is a powerful component that converts OWL/RDF axioms and logical inferences into physics-based constraints for knowledge graph layout optimisation. It bridges semantic reasoning with physical simulation to create meaningful spatial arrangements that reflect ontological relationships.

## Features

- **Axiom Translation**: Converts logical OWL axioms into specific physics constraints
- **Inference Integration**: Applies reasoning results as dynamic constraints
- **Performance Optimization**: Includes caching and batch processing capabilities
- **GPU Compatibility**: Generates constraints compatible with GPU compute pipeline
- **Constraint Grouping**: Organizes constraints into logical categories for efficient processing

## Supported OWL Axioms

| OWL Axiom Type | Physics Constraint | Effect |
|----------------|-------------------|---------|
| `DisjointClasses(A,B)` | Separation forces | Push instances of A and B apart |
| `SubClassOf(A,B)` | Hierarchical alignment | Group A instances near B centroid |
| `SameAs(a,b)` | Co-location/merge | Pull individuals a and b together |
| `DifferentFrom(a,b)` | Separation forces | Push individuals a and b apart |
| `FunctionalProperty(P)` | Cardinality boundaries | Limit connections per node |
| `InverseOf(P,Q)` | Bidirectional edges | Create symmetric relationships |
| `EquivalentClasses(A,B)` | Co-location forces | Group A and B instances together |

## Architecture

### Core Components

#### `OntologyConstraintTranslator`
The main translator class with methods for:
- Converting axioms to constraints
- Processing inferences
- Managing constraint caches
- Grouping constraints by category

#### `OWLAxiom`
Represents an OWL axiom with:
```rust
pub struct OWLAxiom {
    pub axiom_type: OWLAxiomType,
    pub subject: String,
    pub object: Option<String>,
    pub property: Option<String>,
    pub confidence: f32, // 0.0 to 1.0 for weighted constraints
}
```

#### `OntologyInference`
Represents an inference result:
```rust
pub struct OntologyInference {
    pub inferred_axiom: OWLAxiom,
    pub premise_axioms: Vec<String>,
    pub reasoning_confidence: f32,
    pub is_derived: bool,
}
```

### Constraint Groups

The translator organizes constraints into four main categories:

1. **`ontology_separation`**: Separation forces from disjoint classes
2. **`ontology_alignment`**: Hierarchical and clustering constraints
3. **`ontology_boundaries`**: Cardinality and boundary constraints
4. **`ontology_identity`**: Co-location forces from identity relationships

## Usage

### Basic Usage

```rust
use crate::physics::ontology_constraints::{
    OntologyConstraintTranslator,
    OWLAxiom,
    OWLAxiomType,
    OntologyReasoningReport
};

// Create translator
let mut translator = OntologyConstraintTranslator::new();

// Define axioms
let axioms = vec![
    OWLAxiom {
        axiom_type: OWLAxiomType::DisjointClasses,
        subject: "Animal".to_string(),
        object: Some("Plant".to_string()),
        property: None,
        confidence: 1.0,
    },
];

// Create reasoning report
let reasoning_report = OntologyReasoningReport {
    axioms,
    inferences: vec![],
    consistency_checks: vec![],
    reasoning_time_ms: 100,
};

// Apply constraints to graph
let constraint_set = translator.apply_ontology_constraints(&graph_data, &reasoning_report)?;
```

### Advanced Configuration

```rust
use crate::physics::ontology_constraints::OntologyConstraintConfig;

let config = OntologyConstraintConfig {
    disjoint_separation_strength: 0.9,
    hierarchy_alignment_strength: 0.7,
    sameas_colocation_strength: 0.95,
    cardinality_boundary_strength: 0.6,
    max_separation_distance: 100.0,
    min_colocation_distance: 5.0,
    enable_constraint_caching: true,
    cache_invalidation_enabled: true,
};

let translator = OntologyConstraintTranslator::with_config(config);
```

### Working with Constraint Groups

```rust
// Apply constraints
let mut constraint_set = translator.apply_ontology_constraints(&graph_data, &reasoning_report)?;

// Enable/disable specific constraint groups
constraint_set.set_group_active("ontology_separation", true);
constraint_set.set_group_active("ontology_alignment", false);

// Get active constraints for GPU processing
let gpu_constraints = constraint_set.to_gpu_data();
```

## Translation Process

### 1. Disjoint Classes Translation

```rust
// DisjointClasses(Animal, Plant) creates separation constraints
// between all Animal and Plant instances
fn create_disjoint_class_constraints() -> Vec<Constraint> {
    // Find all nodes of each type
    // Create pairwise separation constraints
    // Apply configured separation distance and strength
}
```

### 2. Subclass Hierarchy Translation

```rust
// SubClassOf(Mammal, Animal) creates clustering constraints
// pulling Mammal instances toward Animal centroid
fn create_subclass_constraints() -> Vec<Constraint> {
    // Calculate superclass centroid
    // Create clustering constraints for subclass instances
    // Apply hierarchical alignment strength
}
```

### 3. Identity Constraints

```rust
// SameAs(john_smith, j_smith) creates co-location constraints
fn create_sameas_constraints() -> Vec<Constraint> {
    // Find referenced individuals
    // Create strong clustering constraint
    // Apply minimum co-location distance
}
```

## Performance Optimizations

### Caching System

The translator includes a multi-level caching system:

- **Constraint Cache**: Caches translated constraints by axiom hash
- **Node Type Cache**: Caches node type mappings for fast lookups
- **TTL-based Invalidation**: Automatic cache expiration

### Batch Processing

```rust
// Process multiple axioms efficiently
let constraints = translator.axioms_to_constraints(&all_axioms, &nodes)?;

// Process inferences with confidence weighting
let inference_constraints = translator.inferences_to_constraints(&inferences, &graph)?;
```

### Memory Management

- Efficient node type lookups using HashSets
- Minimal memory allocation during constraint generation
- Configurable cache sizes and TTL values

## Integration with Physics Engine

The generated constraints integrate seamlessly with the existing physics system:

```rust
// Convert to GPU-compatible format
let gpu_data = constraint_set.to_gpu_data();

// Apply in physics simulation
physics_engine.apply_constraints(&gpu_data);
```

### Constraint Parameters

Each constraint type uses specific parameters:

- **Separation**: `[min_distance]`
- **Clustering**: `[cluster_id, strength, target_x, target_y, target_z]`
- **Boundary**: `[min_x, max_x, min_y, max_y, min_z, max_z]`
- **Fixed Position**: `[x, y, z]`

## Error Handling

The translator provides comprehensive error handling:

```rust
pub enum TranslationError {
    MissingAxiomComponent(String),
    InvalidNodeReference(String),
    InferenceProcessingError(String),
    CacheError(String),
}
```

## Testing and Validation

### Unit Tests

```rust
#[test]
fn test_disjoint_classes_translation() {
    let mut translator = OntologyConstraintTranslator::new();
    let constraints = translator.axioms_to_constraints(&[disjoint_axiom], &nodes)?;

    assert!(constraints.iter().all(|c| c.kind == ConstraintKind::Separation));
    assert_eq!(constraints.len(), expected_pair_count);
}
```

### Integration Tests

The translator includes comprehensive integration tests that validate:
- End-to-end axiom processing
- Constraint strength calculations
- Cache functionality
- Group management
- GPU compatibility

## Best Practices

### 1. Axiom Quality
- Ensure axiom confidence values reflect actual certainty
- Validate axiom completeness before processing
- Handle contradictory axioms gracefully

### 2. Performance Tuning
```rust
// For large ontologies, tune cache settings
let config = OntologyConstraintConfig {
    enable_constraint_caching: true,
    cache_ttl_seconds: 3600,
    // ... other settings
};
```

### 3. Constraint Management
```rust
// Use constraint groups for selective application
constraint_set.set_group_active("ontology_separation", physics_params.enforce_separation);
constraint_set.set_group_active("ontology_alignment", physics_params.enforce_hierarchy);
```

### 4. Memory Efficiency
- Clear caches periodically for long-running applications
- Monitor cache hit rates for optimisation opportunities
- Use batch processing for large axiom sets

## Example Applications

### 1. Biological Taxonomy Visualization
```rust
// Separate kingdoms (Animal, Plant, Fungi)
// Group species within genera
// Align genera within families
```

### 2. Software Architecture Diagrams
```rust
// Separate different architectural layers
// Group components within modules
// Align interfaces and implementations
```

### 3. Knowledge Graph Organization
```rust
// Separate concept domains
// Group related concepts
// Align instances with their types
```

## Future Enhancements

### Planned Features
- Support for more OWL constructs (hasValue, someValuesFrom, allValuesFrom)
- Temporal constraint evolution
- Probabilistic constraint weighting
- Distributed constraint processing

### Performance Improvements
- Parallel constraint generation
- Incremental constraint updates
- Adaptive cache sizing
- Memory pool optimisation

## API Reference

For detailed API documentation, see the inline documentation in the source code. Key methods include:

- `new()` - Create translator with default config
- `with_config(config)` - Create translator with custom config
- `axioms_to_constraints(axioms, nodes)` - Convert axioms to constraints
- `apply_ontology_constraints(graph, report)` - Main entry point
- `get_constraint_strength(axiom_type)` - Get strength for axiom type
- `clear_cache()` - Clear all caches
- `get_cache_stats()` - Get cache performance statistics

## Contributing

When contributing to the ontology constraints translator:

1. Add comprehensive tests for new axiom types
2. Update documentation for new features
3. Ensure GPU compatibility for new constraint types
4. Follow existing code style and patterns
5. Add performance benchmarks for significant changes

## Conclusion

The Ontology Constraints Translator provides a robust bridge between semantic reasoning and physics-based graph layout, enabling the creation of visually meaningful knowledge graph visualizations that respect ontological relationships while maintaining performance and flexibility.