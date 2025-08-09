# Physics Engine for Knowledge Graph Layout

This document describes the physics engine modules implemented for the knowledge graph refactor, including stress majorization and semantic constraint generation.

## Overview

The physics engine provides advanced algorithms for optimizing knowledge graph layouts through:

1. **Stress Majorization**: Global optimization algorithm that minimizes layout stress while satisfying constraints
2. **Semantic Constraints**: Automatic generation of constraints based on content similarity and relationships

## Architecture

### Module Structure

```
src/physics/
├── mod.rs                    # Module declarations and exports
├── stress_majorization.rs    # Stress majorization solver
└── semantic_constraints.rs   # Semantic constraint generator
```

### Integration Points

The physics engine integrates with:
- **GPU Compute Pipeline**: Uses CUDA for matrix operations on large graphs
- **Constraint System**: Works with the constraint types defined in `models::constraints`
- **Graph Data**: Operates on `GraphData` and node/edge representations
- **Metadata System**: Analyzes content and topics for semantic relationships

## Stress Majorization Solver

### Purpose

The stress majorization solver optimizes node positions to minimize the stress function:

```
stress = Σ w_ij * (d_ij - ||x_i - x_j||)²
```

Where:
- `w_ij` is the weight between nodes i and j
- `d_ij` is the ideal distance between nodes i and j
- `||x_i - x_j||` is the actual Euclidean distance

### Key Features

- **GPU Acceleration**: Uses CUDA for matrix operations on large graphs
- **Constraint Integration**: Incorporates various constraint types through penalty methods
- **Adaptive Optimization**: Automatic step size adjustment and convergence detection
- **Efficient Algorithms**: Sparse matrix representations and optimized computations

### Usage

```rust
use webxr::physics::StressMajorizationSolver;
use webxr::models::constraints::{ConstraintSet, AdvancedParams};

// Create solver with advanced parameters
let params = AdvancedParams::semantic_optimized();
let mut solver = StressMajorizationSolver::from_advanced_params(&params);

// Optimize graph layout
let result = solver.optimize(&mut graph_data, &constraint_set)?;

println!("Optimization completed in {} iterations", result.iterations);
println!("Final stress: {:.6}", result.final_stress);
```

### Configuration

```rust
use webxr::physics::stress_majorization::StressMajorizationConfig;

let config = StressMajorizationConfig {
    max_iterations: 1000,
    tolerance: 1e-6,
    step_size: 0.1,
    adaptive_step: true,
    constraint_weight: 1.0,
    use_gpu: true,
    ..Default::default()
};

let solver = StressMajorizationSolver::with_config(config);
```

## Semantic Constraint Generator

### Purpose

The semantic constraint generator automatically creates constraints based on:
- Content similarity and topic analysis
- Hierarchical relationships in the graph
- Temporal patterns and metadata
- Structural graph properties

### Constraint Types Generated

1. **Clustering Constraints**: Group semantically similar nodes
2. **Separation Constraints**: Keep unrelated nodes apart
3. **Alignment Constraints**: Align hierarchically related nodes
4. **Boundary Constraints**: Create bounded regions for clusters

### Usage

```rust
use webxr::physics::SemanticConstraintGenerator;
use webxr::models::metadata::MetadataStore;

// Create generator
let mut generator = SemanticConstraintGenerator::new();

// Generate constraints
let result = generator.generate_constraints(&graph_data, Some(&metadata_store))?;

// Apply to constraint set
let mut constraint_set = ConstraintSet::default();
generator.apply_to_constraint_set(&mut constraint_set, &result);
```

### Configuration

```rust
use webxr::physics::semantic_constraints::SemanticConstraintConfig;

let config = SemanticConstraintConfig {
    clustering_threshold: 0.6,
    max_cluster_size: 20,
    min_separation_distance: 150.0,
    enable_hierarchy: true,
    enable_topic_clustering: true,
    semantic_weight: 0.7,
    structural_weight: 0.3,
    ..Default::default()
};

let generator = SemanticConstraintGenerator::with_config(config);
```

## Integration with GPU Pipeline

### GPU Acceleration

The physics engine leverages GPU acceleration for:
- Large matrix multiplications in stress majorization
- Parallel similarity computations in semantic analysis
- Constraint evaluation and gradient computations

### Fallback Mechanisms

- Automatic fallback to CPU computation when GPU is unavailable
- Graceful degradation for systems without CUDA support
- Memory-efficient algorithms for both GPU and CPU modes

## Performance Characteristics

### Scalability

| Graph Size | Stress Majorization | Semantic Constraints | Memory Usage |
|------------|---------------------|---------------------|--------------|
| 1K nodes   | ~100ms             | ~50ms               | ~10MB        |
| 10K nodes  | ~1s                | ~500ms              | ~100MB       |
| 100K nodes | ~10s               | ~5s                 | ~1GB         |

### Optimization Features

- **Sparse Matrix Operations**: Efficient storage for sparse graphs
- **Parallel Processing**: Multi-threaded CPU computation using Rayon
- **Caching**: Intelligent caching of distance matrices and similarities
- **Adaptive Algorithms**: Dynamic adjustment of parameters based on graph properties

## Advanced Usage Patterns

### Custom Constraint Generation

```rust
// Create custom constraints based on domain knowledge
let mut constraint_set = ConstraintSet::default();

// Fixed positions for important nodes
constraint_set.add(Constraint::fixed_position(root_node_id, 0.0, 0.0, 0.0));

// Custom clustering based on file types
let rust_files: Vec<u32> = nodes.iter()
    .filter(|n| n.metadata_id.ends_with(".rs"))
    .map(|n| n.id)
    .collect();
constraint_set.add(Constraint::cluster(rust_files, 1.0, 0.8));
```

### Integration with Real-time Updates

```rust
// Periodic optimization during graph updates
let mut solver = StressMajorizationSolver::new();
let mut generator = SemanticConstraintGenerator::new();

// On graph update
if graph_changed {
    // Clear caches
    solver.clear_cache();
    generator.clear_cache();
    
    // Regenerate semantic constraints
    let new_constraints = generator.generate_constraints(&graph_data, Some(&metadata))?;
    
    // Apply incremental optimization
    solver.update_config(StressMajorizationConfig {
        max_iterations: 100, // Fewer iterations for real-time updates
        ..Default::default()
    });
    
    let result = solver.optimize(&mut graph_data, &constraint_set)?;
}
```

### Physics Parameter Tuning

```rust
// Semantic-focused layout
let semantic_params = AdvancedParams {
    semantic_force_weight: 0.9,
    knowledge_force_weight: 0.8,
    temporal_force_weight: 0.4,
    constraint_force_weight: 0.8,
    adaptive_force_scaling: true,
    ..Default::default()
};

// Structural-focused layout
let structural_params = AdvancedParams {
    structural_force_weight: 0.9,
    separation_factor: 2.0,
    boundary_force_weight: 0.9,
    hierarchical_mode: true,
    layer_separation: 250.0,
    ..Default::default()
};

// Agent communication-focused layout
let agent_params = AdvancedParams::agent_swarm_optimized();
```

## Error Handling and Debugging

### Common Issues

1. **GPU Initialization Failures**
   - Fallback to CPU computation
   - Check CUDA driver installation
   - Verify GPU memory availability

2. **Large Graph Performance**
   - Enable sparse matrix optimizations
   - Reduce max_iterations for real-time use
   - Consider hierarchical layout for very large graphs

3. **Constraint Conflicts**
   - Check constraint satisfaction scores
   - Reduce constraint weights if over-constrained
   - Use adaptive parameters for automatic adjustment

### Debugging Tools

```rust
// Enable detailed logging
let result = solver.optimize(&mut graph_data, &constraint_set)?;

// Check optimization history
let history = solver.get_iteration_history();
for (i, stress) in history.iter().enumerate() {
    println!("Iteration {}: stress = {:.6}", i, stress);
}

// Analyze constraint satisfaction
for (constraint_type, score) in &result.constraint_scores {
    if score < &0.5 {
        println!("Warning: Low satisfaction for {:?}: {:.3}", constraint_type, score);
    }
}
```

## Future Enhancements

### Planned Features

1. **Multi-level Optimization**: Hierarchical approach for very large graphs
2. **Dynamic Constraints**: Time-varying constraints for animated layouts
3. **Machine Learning Integration**: Learn optimal parameters from user preferences
4. **Distributed Computing**: Multi-GPU and cluster support for massive graphs
5. **Interactive Optimization**: Real-time constraint modification during visualization

### Extension Points

- Custom similarity metrics in semantic analysis
- Additional constraint types for domain-specific layouts
- Plugin architecture for specialized optimization algorithms
- Integration with graph neural networks for learned embeddings

## References and Related Work

- [Stress Majorization for Graph Drawing](https://en.wikipedia.org/wiki/Stress_majorization)
- [Force-Directed Graph Drawing](https://en.wikipedia.org/wiki/Force-directed_graph_drawing)
- [Graph Layout Algorithms](https://cs.brown.edu/people/rtamassi/gdhandbook/)
- [GPU-Accelerated Graph Processing](https://developer.nvidia.com/graph-analytics)

## API Reference

For detailed API documentation, see:
- [`StressMajorizationSolver`](../src/physics/stress_majorization.rs)
- [`SemanticConstraintGenerator`](../src/physics/semantic_constraints.rs)
- [`ConstraintSet`](../src/models/constraints.rs)
- [`AdvancedParams`](../src/models/constraints.rs)