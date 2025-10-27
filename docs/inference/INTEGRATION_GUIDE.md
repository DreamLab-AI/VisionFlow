# Whelk-rs Inference Engine Integration Guide

## Overview

VisionFlow v1.0.0 integrates **whelk-rs** for OWL 2 DL reasoning and ontology inference. This guide explains how to use the inference capabilities.

## Architecture

```
┌─────────────────────────────────────────┐
│         API Layer (HTTP)                │
│  /api/inference/*                       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Inference Service                    │
│  - run_inference()                      │
│  - validate_ontology()                  │
│  - classify_ontology()                  │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼───────────┐
    │          │           │
┌───▼────┐ ┌──▼──────┐ ┌─▼─────────┐
│ Cache  │ │ whelk-rs│ │ Events    │
│ LRU    │ │ Engine  │ │ Bus       │
└────────┘ └─────────┘ └───────────┘
```

## Features

### 1. OWL 2 DL Reasoning
- **Classification**: Compute class hierarchy
- **Consistency Checking**: Validate ontology consistency
- **Entailment Checking**: Verify axiom entailment
- **Explanation Generation**: Explain inferred axioms

### 2. OWL Parsers
Supports multiple formats:
- OWL/XML (Functional Syntax)
- RDF/XML
- Turtle (TTL)
- Manchester Syntax (partial)

### 3. Performance Optimization
- **Caching**: LRU cache with TTL and checksum validation
- **Batch Processing**: Parallel inference for multiple ontologies
- **Incremental Reasoning**: Optimize for small changes
- **Async/Await**: Non-blocking operations

### 4. Event-Driven Inference
Automatic reasoning triggers:
- On ontology import
- On class addition
- On axiom addition

## Quick Start

### 1. Parse OWL Ontology

```rust
use webxr::inference::owl_parser::OWLParser;

let owl_content = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.com/animal"/>
    <owl:Class rdf:about="http://example.com/Animal"/>
    <owl:Class rdf:about="http://example.com/Dog">
        <rdfs:subClassOf rdf:resource="http://example.com/Animal"/>
    </owl:Class>
</rdf:RDF>"#;

let parse_result = OWLParser::parse(owl_content)?;

println!("Parsed {} classes and {} axioms",
    parse_result.classes.len(),
    parse_result.axioms.len()
);
```

### 2. Run Inference

```rust
use webxr::adapters::whelk_inference_engine::WhelkInferenceEngine;
use webxr::ports::inference_engine::InferenceEngine;

let mut engine = WhelkInferenceEngine::new();

// Load ontology
engine.load_ontology(classes, axioms).await?;

// Run inference
let results = engine.infer().await?;

println!("Inferred {} new axioms in {}ms",
    results.inferred_axioms.len(),
    results.inference_time_ms
);
```

### 3. Check Consistency

```rust
// Check if ontology is consistent
let is_consistent = engine.check_consistency().await?;

if !is_consistent {
    println!("Ontology has inconsistencies!");
}
```

### 4. Get Class Hierarchy

```rust
// Get inferred subclass hierarchy
let hierarchy = engine.get_subclass_hierarchy().await?;

for (child, parent) in hierarchy {
    println!("{} ⊑ {}", child, parent);
}
```

## API Endpoints

### POST /api/inference/run
Run inference on an ontology.

**Request:**
```json
{
  "ontology_id": "animal-ontology",
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "ontology_id": "animal-ontology",
  "inferred_axioms_count": 42,
  "inference_time_ms": 150,
  "reasoner_version": "whelk-rs-1.0"
}
```

### POST /api/inference/batch
Run inference on multiple ontologies in parallel.

**Request:**
```json
{
  "ontology_ids": ["ont1", "ont2", "ont3"],
  "max_parallel": 4
}
```

### POST /api/inference/validate
Validate ontology consistency.

**Request:**
```json
{
  "ontology_id": "animal-ontology"
}
```

**Response:**
```json
{
  "consistent": true,
  "unsatisfiable": [],
  "warnings": [],
  "errors": [],
  "validation_time_ms": 50
}
```

### GET /api/inference/results/{ontology_id}
Get cached inference results.

### GET /api/inference/classify/{ontology_id}
Get classified hierarchy.

**Response:**
```json
{
  "hierarchy": [
    ["Dog", "Animal"],
    ["Cat", "Animal"]
  ],
  "equivalent_classes": [
    ["Canine", "Dog"]
  ],
  "classification_time_ms": 100,
  "inferred_count": 42
}
```

### GET /api/inference/consistency/{ontology_id}
Get detailed consistency report.

### DELETE /api/inference/cache/{ontology_id}
Invalidate cache for ontology.

## Configuration

### Inference Service Config

```rust
use webxr::application::inference_service::InferenceServiceConfig;

let config = InferenceServiceConfig {
    enable_cache: true,
    auto_inference: true,
    max_parallel: 4,
    publish_events: true,
};
```

### Cache Config

```rust
use webxr::inference::cache::CacheConfig;

let cache_config = CacheConfig {
    max_entries: 1000,
    ttl_seconds: 3600, // 1 hour
    persist_to_db: true,
    enable_stats: true,
};
```

### Auto-Inference Config

```rust
use webxr::events::inference_triggers::AutoInferenceConfig;

let auto_config = AutoInferenceConfig {
    auto_on_import: true,
    auto_on_class_add: false,
    auto_on_axiom_add: false,
    min_delay_ms: 1000,
    batch_changes: true,
};
```

## Performance Tuning

### 1. Enable Caching
```rust
// Cache speeds up repeated inferences by 10-100x
let config = InferenceServiceConfig {
    enable_cache: true,
    ..Default::default()
};
```

### 2. Batch Processing
```rust
// Process multiple ontologies in parallel
let ontologies = vec!["ont1", "ont2", "ont3"];
let results = service.batch_inference(ontologies).await?;
```

### 3. Incremental Reasoning
```rust
// For small changes, use incremental mode
optimizer.add_change(IncrementalChange {
    added_classes: vec![new_class],
    removed_classes: vec![],
    added_axioms: vec![new_axiom],
    removed_axioms: vec![],
}).await;
```

## Error Handling

```rust
use webxr::ports::inference_engine::InferenceEngineError;

match engine.infer().await {
    Ok(results) => println!("Success: {} inferences", results.inferred_axioms.len()),
    Err(InferenceEngineError::OntologyNotLoaded) => {
        eprintln!("Load ontology first");
    }
    Err(InferenceEngineError::InconsistentOntology(msg)) => {
        eprintln!("Inconsistent: {}", msg);
    }
    Err(e) => eprintln!("Error: {:?}", e),
}
```

## Best Practices

1. **Always validate first**: Check consistency before classification
2. **Use caching**: Enable for production environments
3. **Batch operations**: Process multiple ontologies together
4. **Monitor performance**: Check statistics regularly
5. **Handle errors gracefully**: Ontologies may be inconsistent

## Limitations

- **EL Profile**: whelk-rs supports EL (Existential Logic) subset of OWL 2 DL
- **Not Full OWL**: Complex constructs like cardinality restrictions may not be fully supported
- **Performance**: Large ontologies (>10k classes) may require optimization

## Troubleshooting

### Inference is slow
- Enable caching
- Use batch processing
- Check ontology size
- Consider incremental mode

### Cache not working
- Verify `enable_cache: true`
- Check TTL settings
- Ensure checksum matches

### Inconsistent results
- Clear cache with `invalidate_cache()`
- Run fresh inference with `force: true`
- Check ontology for errors

## See Also

- [OWL 2 DL Specification](https://www.w3.org/TR/owl2-syntax/)
- [whelk-rs GitHub](https://github.com/ontodev/whelk.rs)
- [API Reference](./API_REFERENCE.md)
- [Performance Guide](./PERFORMANCE_GUIDE.md)
