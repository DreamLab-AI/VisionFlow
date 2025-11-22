# Semantic Forces Quick Reference

**Version:** 2.0.0

---

## Force Types & Parameters

| Force | Default State | Key Parameter | Default Value | Effect |
|-------|---------------|---------------|---------------|--------|
| **Requires** | ✅ Enabled | `requires_strength` | 0.7 | Dependency → Prerequisite |
| **Enables** | ✅ Enabled | `enables_strength` | 0.4 | Capability attraction |
| **Has-Part** | ✅ Enabled | `has_part_strength` | 0.9 | Part orbits whole |
| **Bridges-To** | ✅ Enabled | `bridges_to_strength` | 0.3 | Cross-domain bridge |
| **Physicality** | ✅ Enabled | `cluster_attraction` | 0.5 | Nature grouping |
| **Role** | ✅ Enabled | `cluster_attraction` | 0.45 | Function grouping |
| **Maturity** | ✅ Enabled | `level_attraction` | 0.4 | Lifecycle staging |
| **Cross-Domain** | ✅ Enabled | `base_strength` | 0.3 | Domain strength |
| **DAG Layout** | ❌ Disabled | `level_attraction` | 0.5 | Hierarchy |
| **Type Cluster** | ❌ Disabled | `cluster_attraction` | 0.4 | Type groups |
| **Collision** | ✅ Enabled | `collision_strength` | 1.0 | Prevent overlap |
| **Attr Spring** | ❌ Disabled | `base_spring_k` | 0.01 | Edge weights |

---

## Metadata Format

### Node Properties

```markdown
- owl:physicality:: VirtualEntity       # or PhysicalEntity, ConceptualEntity
- owl:role:: Process                    # or Agent, Resource, Concept
- maturity:: mature                     # or emerging, declining
- cross-domain-links:: [[ai-rb:...]], [[tc-mv:...]]
```

### Edge Types

```markdown
- requires:: [[Training Data]], [[GPU Resources]]
- enables:: [[Few-Shot Learning]], [[Text Generation]]
- has-part:: [[Visual Mesh]], [[Animation Rig]]
- bridges-to:: [[tc-mv:virtual-meetings]]
```

---

## Configuration Snippets

### Stronger Dependencies

```rust
config.ontology_relationship.requires_strength = 0.9;
config.ontology_relationship.requires_rest_length = 60.0;
```

### Tighter Component Clusters

```rust
config.ontology_relationship.has_part_strength = 1.2;
config.ontology_relationship.has_part_orbit_radius = 40.0;
```

### More Cross-Domain Bridging

```rust
config.cross_domain.link_count_multiplier = 0.2;
config.cross_domain.max_strength_boost = 3.0;
```

### Larger Maturity Separation

```rust
config.maturity_layout.stage_separation = 200.0;
config.maturity_layout.level_attraction = 0.6;
```

### Disable a Force

```rust
config.ontology_relationship.enabled = false;
config.physicality_cluster.enabled = false;
```

---

## Common Adjustments

### Layout Too Tight

```rust
// Increase rest lengths
requires_rest_length: 120.0,      // from 80.0
enables_rest_length: 180.0,       // from 120.0
has_part_orbit_radius: 90.0,      // from 60.0
bridges_to_rest_length: 350.0,    // from 250.0

// Increase cluster radii
cluster_radius: 250.0,             // from 180.0
```

### Layout Too Loose

```rust
// Increase force strengths
requires_strength: 1.0,            // from 0.7
cluster_attraction: 0.8,           // from 0.5
level_attraction: 0.6,             // from 0.4
```

### Dependencies Not Clear

```rust
// Strengthen requires, weaken enables
requires_strength: 0.9,
enables_strength: 0.3,
```

### Domains Not Bridging

```rust
// Increase cross-domain forces
base_strength: 0.5,                // from 0.3
link_count_multiplier: 0.15,       // from 0.1
```

---

## Force Strength Hierarchy

**Strongest to Weakest (at equal distance):**

1. Collision (1.0) - Prevents overlap
2. Has-Part (0.9) - Tight component clusters
3. Requires (0.7) - Dependency flow
4. Physicality (0.5) - Nature grouping
5. Role (0.45) - Function grouping
6. Enables (0.4) - Capability attraction
7. Maturity (0.4) - Lifecycle staging
8. Bridges-To (0.3-0.6) - Cross-domain bridge

**Principle:** Local forces > Medium forces > Global forces

---

## Centroid Calculation

### Physicality Centroids

```rust
for each node with physicality type:
    accumulate position
centroid = sum_positions / count_per_type
```

**Types:** VirtualEntity(1), PhysicalEntity(2), ConceptualEntity(3)

### Role Centroids

```rust
for each node with role type:
    accumulate position
centroid = sum_positions / count_per_type
```

**Types:** Process(1), Agent(2), Resource(3), Concept(4)

---

## Edge Type IDs

| ID | Edge Type | Force Type |
|----|-----------|------------|
| 0 | generic | Standard spring |
| 1 | dependency | Standard spring |
| 2 | hierarchy | DAG layout |
| 3 | association | Standard spring |
| 4 | sequence | Standard spring |
| 5 | subClassOf | Standard spring |
| 6 | instanceOf | Standard spring |
| **7** | **requires** | **Directional dependency** |
| **8** | **enables** | **Capability attraction** |
| **9** | **has-part** | **Component clustering** |
| **10** | **bridges-to** | **Cross-domain bridge** |

---

## Physics Formulas

### Spring Force (Hooke's Law)

```
F = k * (distance - rest_length) * direction
```

### Clustering Force

```
if distance > radius:
    F_attract = k * (distance - radius) * direction

if different_type && distance < 2*radius:
    F_repel = k / (distance^2) * direction
```

### Maturity Force

```
target_z = {-separation, 0, +separation}
F_z = k * (target_z - current_z)
```

### Cross-Domain Strength

```
boost = min(1 + count * multiplier, max_boost)
strength = base * boost
```

---

## GPU Kernel Dispatch

```rust
// Calculate grid dimensions
let threads_per_block = 256;
let num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

// Launch kernel
kernel<<<num_blocks, threads_per_block>>>(args...);
```

**Typical Workloads:**
- 10K nodes → 40 blocks × 256 threads
- 50K edges → 196 blocks × 256 threads

---

## Debugging Tips

### Force Vectors Visualization

Enable debug mode to see force magnitudes:

```rust
// In debug builds
#[cfg(debug_assertions)]
fn visualize_forces(&self, graph: &GraphData) {
    for (i, node) in graph.nodes.iter().enumerate() {
        println!("Node {}: vx={}, vy={}, vz={}",
                 i, node.data.vx, node.data.vy, node.data.vz);
    }
}
```

### Check Centroids

```rust
println!("Physicality centroids: {:?}", engine.physicality_centroids);
println!("Role centroids: {:?}", engine.role_centroids);
```

### Verify Metadata Extraction

```rust
println!("Node physicality: {:?}", engine.node_physicality);
println!("Node roles: {:?}", engine.node_role);
println!("Node maturity: {:?}", engine.node_maturity);
println!("Cross-domain counts: {:?}", engine.node_cross_domain_count);
```

---

## Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `/src/gpu/semantic_forces.rs` | Rust engine | ~1100 |
| `/src/utils/semantic_forces.cu` | CUDA kernels | ~760 |
| `/docs/guides/ontology-semantic-forces.md` | Full guide | Comprehensive |
| `/docs/SEMANTIC-FORCES-IMPLEMENTATION-SUMMARY.md` | Implementation details | Detailed |

---

## API Example

```rust
// Create engine
let config = SemanticConfig::default();
let mut engine = SemanticForcesEngine::new(config);

// Initialize with graph
engine.initialize(&graph)?;

// Apply forces
engine.apply_semantic_forces(&mut graph)?;

// Update config
let mut new_config = engine.config().clone();
new_config.ontology_relationship.requires_strength = 0.9;
engine.update_config(new_config);

// Re-initialize if needed
engine.initialize(&graph)?;
```

---

## Performance Tuning

### For Large Graphs (>10K nodes)

```rust
// Reduce clustering iterations
physicality_cluster.enabled = false;
role_cluster.enabled = false;

// Or increase cluster radius to reduce repulsion calculations
cluster_radius: 300.0,
```

### For Real-Time Interaction

```rust
// Use weaker forces for smoother animation
requires_strength: 0.4,
enables_strength: 0.2,
has_part_strength: 0.6,
```

### For Final Layout

```rust
// Use stronger forces for stable positioning
requires_strength: 1.0,
cluster_attraction: 0.8,
level_attraction: 0.6,
```

---

**Quick Reference v2.0.0**
