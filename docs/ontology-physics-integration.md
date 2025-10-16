# Ontology-Physics Integration Documentation

## Overview

The ontology constraint system is now fully integrated with the `PhysicsOrchestratorActor`, allowing semantic knowledge from OWL ontologies to influence the physics simulation of the graph layout.

## Architecture

```
┌─────────────────────────┐
│   OntologyActor         │
│                         │
│  - Load OWL axioms      │
│  - Parse ontologies     │
│  - Perform reasoning    │
└───────────┬─────────────┘
            │
            │ OntologyReasoningReport
            ▼
┌─────────────────────────┐
│ OntologyConstraint      │
│ Translator              │
│                         │
│ - axioms_to_constraints │
│ - apply_ontology_       │
│   constraints           │
└───────────┬─────────────┘
            │
            │ ConstraintSet
            ▼
┌─────────────────────────┐
│ PhysicsOrchestrator     │
│ Actor                   │
│                         │
│ - Manage constraints    │
│ - Upload to GPU         │
│ - Apply physics forces  │
└───────────┬─────────────┘
            │
            │ GPU Constraint Data
            ▼
┌─────────────────────────┐
│   GPU Force Compute     │
│                         │
│ - Apply constraints     │
│ - Calculate forces      │
│ - Update positions      │
└─────────────────────────┘
```

## Components

### 1. OntologyConstraintTranslator

Located in `src/physics/ontology_constraints.rs`, this module converts OWL axioms into physics constraints:

**Key Methods:**
- `axioms_to_constraints(&[OWLAxiom], &[Node])` - Convert axioms to constraints
- `apply_ontology_constraints(&GraphData, &OntologyReasoningReport)` - Complete constraint generation
- `inferences_to_constraints(&[OntologyInference], &GraphData)` - Generate constraints from reasoning

**Supported Axiom Types:**
- `DisjointClasses(A,B)` → Separation constraints (push apart)
- `SubClassOf(A,B)` → Hierarchical alignment (cluster together)
- `SameAs(a,b)` → Co-location forces (pull together)
- `DifferentFrom(a,b)` → Separation forces
- `FunctionalProperty(P)` → Cardinality boundaries

### 2. PhysicsOrchestratorActor

Located in `src/actors/physics_orchestrator_actor.rs`, manages the integration:

**New Fields:**
```rust
ontology_constraints: Option<ConstraintSet>  // Ontology-derived constraints
user_constraints: Option<ConstraintSet>      // User-defined constraints
```

**New Methods:**
- `apply_ontology_constraints_internal()` - Apply constraints with merge modes
- `upload_constraints_to_gpu()` - Send constraints to GPU
- `get_constraint_statistics()` - Get constraint stats
- `set_constraint_group_active()` - Toggle constraint groups

### 3. Message Types

Located in `src/actors/messages.rs`:

```rust
// Apply ontology constraints
pub struct ApplyOntologyConstraints {
    pub constraint_set: ConstraintSet,
    pub merge_mode: ConstraintMergeMode,
}

// Merge modes
pub enum ConstraintMergeMode {
    Replace,           // Replace all ontology constraints
    Merge,             // Merge with existing
    AddIfNoConflict,   // Only add non-conflicting
}

// Toggle constraint groups
pub struct SetConstraintGroupActive {
    pub group_name: String,
    pub active: bool,
}

// Get statistics
pub struct GetConstraintStats;

pub struct ConstraintStats {
    pub total_constraints: usize,
    pub active_constraints: usize,
    pub constraint_groups: HashMap<String, usize>,
    pub ontology_constraints: usize,
    pub user_constraints: usize,
}
```

## Usage Examples

### Example 1: Apply Ontology Constraints

```rust
use crate::physics::ontology_constraints::{
    OntologyConstraintTranslator, OntologyReasoningReport
};
use crate::actors::messages::{ApplyOntologyConstraints, ConstraintMergeMode};

// 1. Create reasoning report from ontology validation
let reasoning_report = OntologyReasoningReport {
    axioms: vec![
        OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "Person".to_string(),
            object: Some("Company".to_string()),
            property: None,
            confidence: 1.0,
        },
        OWLAxiom {
            axiom_type: OWLAxiomType::SubClassOf,
            subject: "Employee".to_string(),
            object: Some("Person".to_string()),
            property: None,
            confidence: 0.9,
        },
    ],
    inferences: vec![],
    consistency_checks: vec![],
    reasoning_time_ms: 150,
};

// 2. Generate constraints from axioms
let mut translator = OntologyConstraintTranslator::new();
let constraint_set = translator.apply_ontology_constraints(
    &graph_data,
    &reasoning_report
)?;

// 3. Send to physics orchestrator
physics_orchestrator.do_send(ApplyOntologyConstraints {
    constraint_set,
    merge_mode: ConstraintMergeMode::Merge,
});
```

### Example 2: Toggle Constraint Groups

```rust
use crate::actors::messages::SetConstraintGroupActive;

// Disable disjoint class constraints
physics_orchestrator.do_send(SetConstraintGroupActive {
    group_name: "ontology_separation".to_string(),
    active: false,
});

// Enable hierarchy constraints
physics_orchestrator.do_send(SetConstraintGroupActive {
    group_name: "ontology_alignment".to_string(),
    active: true,
});
```

### Example 3: Get Constraint Statistics

```rust
use crate::actors::messages::GetConstraintStats;

let stats = physics_orchestrator.send(GetConstraintStats).await?;

println!("Total constraints: {}", stats.total_constraints);
println!("Active constraints: {}", stats.active_constraints);
println!("Ontology constraints: {}", stats.ontology_constraints);
println!("User constraints: {}", stats.user_constraints);

for (group, count) in stats.constraint_groups {
    println!("  {}: {} constraints", group, count);
}
```

## Constraint Groups

Ontology constraints are automatically organized into groups:

- `ontology_separation` - Separation forces from DisjointClasses, DifferentFrom
- `ontology_alignment` - Hierarchical alignment from SubClassOf
- `ontology_boundaries` - Cardinality boundaries from FunctionalProperty
- `ontology_identity` - Co-location from SameAs, EquivalentClasses

## Merge Modes

### Replace Mode
Completely replaces all ontology constraints with new ones. User constraints are preserved.

```rust
ApplyOntologyConstraints {
    constraint_set,
    merge_mode: ConstraintMergeMode::Replace,
}
```

### Merge Mode
Adds new constraints to existing ontology constraints, adjusting group indices.

```rust
ApplyOntologyConstraints {
    constraint_set,
    merge_mode: ConstraintMergeMode::Merge,
}
```

### AddIfNoConflict Mode
Only adds constraints that don't conflict with existing ones (same nodes + constraint type).

```rust
ApplyOntologyConstraints {
    constraint_set,
    merge_mode: ConstraintMergeMode::AddIfNoConflict,
}
```

## Integration Flow

1. **Ontology Loading**: OntologyActor loads OWL files and parses axioms
2. **Validation**: Graph data is validated against ontology rules
3. **Reasoning**: Inference engine generates derived axioms
4. **Translation**: OntologyConstraintTranslator converts axioms to constraints
5. **Application**: PhysicsOrchestratorActor receives ConstraintSet
6. **Merging**: Constraints are merged based on merge mode
7. **GPU Upload**: Active constraints are uploaded to GPU
8. **Force Calculation**: GPU applies constraint forces during simulation

## Configuration

### Translator Configuration

```rust
use crate::physics::ontology_constraints::OntologyConstraintConfig;

let config = OntologyConstraintConfig {
    disjoint_separation_strength: 0.8,
    hierarchy_alignment_strength: 0.6,
    sameas_colocation_strength: 0.9,
    cardinality_boundary_strength: 0.7,
    max_separation_distance: 50.0,
    min_colocation_distance: 2.0,
    enable_constraint_caching: true,
    cache_invalidation_enabled: true,
};

let translator = OntologyConstraintTranslator::with_config(config);
```

## Testing

The integration is tested in `tests/ontology_smoke_test.rs`:

```bash
cargo test --test ontology_smoke_test integration_tests::test_apply_constraints_to_physics
```

Key test coverage:
- Axiom to constraint translation
- Constraint grouping
- Merge mode handling
- GPU upload verification
- Constraint toggling

## Performance Considerations

1. **Constraint Count**: Large ontologies generate many constraints. Use constraint groups to selectively enable/disable.

2. **GPU Memory**: Each constraint uses GPU memory. Monitor with `GetConstraintStats`.

3. **Update Frequency**: Don't apply constraints on every frame. Update when ontology changes.

4. **Caching**: Enable constraint caching in translator configuration for better performance.

## Future Enhancements

- [ ] Constraint priority system
- [ ] Dynamic constraint strength adjustment
- [ ] Temporal constraint evolution
- [ ] Constraint conflict resolution
- [ ] Performance profiling metrics
- [ ] Incremental constraint updates

## References

- Physics Constraints: `src/models/constraints.rs`
- Ontology Parser: `src/ontology/parser/parser.rs`
- OWL Validator: `src/services/owl_validator.rs`
- GPU Force Compute: `src/actors/gpu/force_compute_actor.rs`
