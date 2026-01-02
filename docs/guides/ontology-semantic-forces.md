---
layout: default
title: Ontology-Driven Semantic Forces
parent: Guides
nav_order: 43
description: Ontology-driven physics forces for semantically meaningful 3D layouts
---


# Ontology-Driven Semantic Forces

**Status:** Implemented
**Version:** 2.0.0
**Last Updated:** 2025-11-22

---

## Overview

The Semantic Forces Engine has been enhanced with rich ontology-driven physics forces that leverage relationship types, physicality classifications, role categorizations, maturity stages, and cross-domain connections. This creates semantically meaningful 3D layouts that reflect the knowledge structure.

## Architecture

### Force Categories

| Category | Purpose | Default State |
|----------|---------|---------------|
| **DAG Layout** | Hierarchical positioning | Disabled |
| **Type Clustering** | Group by node type | Disabled |
| **Collision Detection** | Prevent node overlap | Enabled |
| **Attribute Springs** | Edge weight-based attraction | Disabled |
| **Ontology Relationships** | Semantic relationship forces | **Enabled** |
| **Physicality Clustering** | Group by physical nature | **Enabled** |
| **Role Clustering** | Group by functional role | **Enabled** |
| **Maturity Layout** | Stage-based positioning | **Enabled** |
| **Cross-Domain Links** | Inter-domain bridges | **Enabled** |

---

## Ontology Relationship Forces

### 1. Requires Relationship

**Type:** Directional dependency spring
**Effect:** Dependencies are pulled toward their prerequisites

**Physics:**
```rust
// Directional spring: A requires B → A is pulled toward B
F_requires = k_requires * (distance - rest_length) * direction
// Only source node receives force
```

**Configuration:**
```rust
OntologyRelationshipConfig {
    requires_strength: 0.7,         // Strong pull
    requires_rest_length: 80.0,     // Moderate separation
    ...
}
```

**Example:**
- `LLM` requires `Training Data` → LLM node pulled toward Training Data
- `Deep Learning` requires `GPU Resources` → Forms dependency cluster

---

### 2. Enables Relationship

**Type:** Bidirectional capability attraction (weaker)
**Effect:** Technologies that enable capabilities attract toward each other

**Physics:**
```rust
// Bidirectional spring with weaker force
F_enables = k_enables * (distance - rest_length) * direction
// Both nodes receive equal and opposite forces
```

**Configuration:**
```rust
OntologyRelationshipConfig {
    enables_strength: 0.4,          // Weaker than requires
    enables_rest_length: 120.0,     // Longer rest length
    ...
}
```

**Example:**
- `LLM` enables `Few-Shot Learning` → Mutual attraction
- `Cloud Computing` enables `Scalability` → Form capability clusters

---

### 3. Has-Part Relationship

**Type:** Strong clustering (parts orbit whole)
**Effect:** Components orbit around their parent entity

**Physics:**
```rust
// Strong bidirectional spring with tight orbit
F_has_part = k_has_part * (distance - orbit_radius) * direction
// Creates tight orbital clusters
```

**Configuration:**
```rust
OntologyRelationshipConfig {
    has_part_strength: 0.9,         // Very strong
    has_part_orbit_radius: 60.0,    // Tight orbit
    ...
}
```

**Example:**
- `3D Model` has-part `Visual Mesh` → Mesh orbits Model
- `3D Model` has-part `Animation Rig` → Rig orbits Model
- Result: Tight component cluster around parent

---

### 4. Bridges-To Relationship

**Type:** Long-range cross-domain spring
**Effect:** Connects concepts across domain boundaries with adaptive strength

**Physics:**
```rust
// Adaptive strength based on cross-domain link count
link_boost = 1.0 + avg_link_count * multiplier
strength = base_strength * min(link_boost, max_boost)
F_bridges = strength * (distance - rest_length) * direction
```

**Configuration:**
```rust
OntologyRelationshipConfig {
    bridges_to_strength: 0.3,       // Moderate base
    bridges_to_rest_length: 250.0,  // Long-range
}
CrossDomainConfig {
    link_count_multiplier: 0.1,     // Boost per link
    max_strength_boost: 2.0,        // Cap at 2x
    ...
}
```

**Example:**
- `Remote Collaboration` (TC domain) bridges-to `Virtual Meetings` (MV domain)
- `AI Ethics` (AI domain) bridges-to `Privacy` (Tech domain)
- Nodes with more cross-domain links create stronger bridges

---

## Physicality Clustering

**Purpose:** Group nodes by their physical nature

### Physicality Types

| Type | ID | Description | Examples |
|------|-----|-------------|----------|
| **VirtualEntity** | 1 | Digital/software entities | LLM, Virtual Meeting, Software |
| **PhysicalEntity** | 2 | Physical objects | Robot, Server, Hardware |
| **ConceptualEntity** | 3 | Abstract concepts | Ethics, Collaboration, Theory |

### Physics

```rust
// Attraction to physicality centroid
if distance > cluster_radius:
    F_attract = k_attract * (distance - cluster_radius) * direction

// Repulsion from different physicality types
if different_physicality && distance < 2 * cluster_radius:
    F_repel = k_repel / (distance^2) * direction
```

### Configuration

```rust
PhysicalityClusterConfig {
    cluster_attraction: 0.5,              // Moderate attraction
    cluster_radius: 180.0,                // Large clusters
    inter_physicality_repulsion: 0.25,    // Moderate separation
    enabled: true,
}
```

### Node Metadata

```markdown
- owl:physicality:: VirtualEntity
- owl:physicality:: PhysicalEntity
- owl:physicality:: ConceptualEntity
```

---

## Role Clustering

**Purpose:** Group nodes by functional role

### Role Types

| Type | ID | Description | Examples |
|------|-----|-------------|----------|
| **Process** | 1 | Actions/workflows | Collaboration, Training, Inference |
| **Agent** | 2 | Active entities | Person, AI Agent, Robot |
| **Resource** | 3 | Consumed/used items | Data, GPU, Bandwidth |
| **Concept** | 4 | Ideas/theories | Ethics, Ontology, Algorithm |

### Physics

```rust
// Same structure as physicality clustering
F_role = similar to physicality forces, but for role types
```

### Configuration

```rust
RoleClusterConfig {
    cluster_attraction: 0.45,         // Moderate attraction
    cluster_radius: 160.0,            // Medium-large clusters
    inter_role_repulsion: 0.2,        // Gentle separation
    enabled: true,
}
```

### Node Metadata

```markdown
- owl:role:: Process
- owl:role:: Agent
- owl:role:: Resource
- owl:role:: Concept
```

---

## Maturity Layout

**Purpose:** Arrange nodes by lifecycle stage along Z-axis

### Maturity Stages

| Stage | ID | Z Position | Description |
|-------|-----|-----------|-------------|
| **Emerging** | 1 | -stage_separation | New/experimental technologies |
| **Mature** | 2 | 0 | Established/stable technologies |
| **Declining** | 3 | +stage_separation | Legacy/phasing out technologies |

### Physics

```rust
// Z-axis positioning by maturity stage
target_z = match maturity {
    1 => -stage_separation,  // Emerging at bottom
    2 => 0.0,                // Mature at center
    3 => +stage_separation,  // Declining at top
}
F_maturity = k_maturity * (target_z - current_z) * z_direction
```

### Configuration

```rust
MaturityLayoutConfig {
    vertical_spacing: 150.0,      // Spacing between stages
    level_attraction: 0.4,        // Pull toward target level
    stage_separation: 100.0,      // Z-axis separation
    enabled: true,
}
```

### Node Metadata

```markdown
- maturity:: emerging
- maturity:: mature
- maturity:: declining
```

---

## Cross-Domain Force Strength

**Purpose:** Strengthen connections between domains based on link count

### Adaptive Strength

```rust
// Calculate strength boost from cross-domain links
avg_link_count = (source_links + target_links) / 2
strength_boost = min(
    1.0 + avg_link_count * link_count_multiplier,
    max_strength_boost
)
final_strength = base_strength * strength_boost
```

### Configuration

```rust
CrossDomainConfig {
    base_strength: 0.3,              // Base cross-domain force
    link_count_multiplier: 0.1,      // Boost per link
    max_strength_boost: 2.0,         // Cap at 2x strength
    rest_length: 200.0,              // Long-range connections
    enabled: true,
}
```

### Node Metadata

```markdown
- cross-domain-links:: [[ai-rb:robot-perception]], [[tc-mv:virtual-meetings]]
```

The engine counts comma-separated links and `bridges-to` edges.

---

## Force Composition

All forces combine through superposition:

```rust
F_total = F_dag + F_type_cluster + F_collision + F_attribute_spring +
          F_requires + F_enables + F_has_part + F_bridges_to +
          F_physicality + F_role + F_maturity
```

---

## Configuration Example

```rust
use visionflow::gpu::semantic_forces::{SemanticConfig, SemanticForcesEngine};

let config = SemanticConfig {
    // Ontology relationships (ENABLED by default)
    ontology_relationship: OntologyRelationshipConfig {
        requires_strength: 0.7,
        requires_rest_length: 80.0,
        enables_strength: 0.4,
        enables_rest_length: 120.0,
        has_part_strength: 0.9,
        has_part_orbit_radius: 60.0,
        bridges_to_strength: 0.3,
        bridges_to_rest_length: 250.0,
        enabled: true,
    },

    // Physicality clustering (ENABLED by default)
    physicality_cluster: PhysicalityClusterConfig {
        cluster_attraction: 0.5,
        cluster_radius: 180.0,
        inter_physicality_repulsion: 0.25,
        enabled: true,
    },

    // Role clustering (ENABLED by default)
    role_cluster: RoleClusterConfig {
        cluster_attraction: 0.45,
        cluster_radius: 160.0,
        inter_role_repulsion: 0.2,
        enabled: true,
    },

    // Maturity layout (ENABLED by default)
    maturity_layout: MaturityLayoutConfig {
        vertical_spacing: 150.0,
        level_attraction: 0.4,
        stage_separation: 100.0,
        enabled: true,
    },

    // Cross-domain forces (ENABLED by default)
    cross_domain: CrossDomainConfig {
        base_strength: 0.3,
        link_count_multiplier: 0.1,
        max_strength_boost: 2.0,
        rest_length: 200.0,
        enabled: true,
    },

    // Standard forces (configure as needed)
    dag: DAGConfig::default(),
    type_cluster: TypeClusterConfig::default(),
    collision: CollisionConfig::default(),
    attribute_spring: AttributeSpringConfig::default(),
};

let mut engine = SemanticForcesEngine::new(config);
engine.initialize(&graph)?;
engine.apply_semantic_forces(&mut graph)?;
```

---

## GPU Implementation

All forces have parallel CUDA kernels for high performance:

### Kernels

| Kernel | Purpose | Threads |
|--------|---------|---------|
| `apply_ontology_relationship_force` | Process requires/enables/has-part/bridges-to edges | 1 per edge |
| `apply_physicality_cluster_force` | Physicality clustering | 1 per node |
| `apply_role_cluster_force` | Role clustering | 1 per node |
| `apply_maturity_layout_force` | Maturity staging | 1 per node |
| `calculate_physicality_centroids` | Compute physicality centroids | 1 per node |
| `calculate_role_centroids` | Compute role centroids | 1 per node |

### Performance

- **CPU Fallback:** O(N²) for clustering, O(E) for relationships
- **GPU Parallel:** Concurrent force calculation across all nodes/edges
- **Memory:** Constant memory for config, global memory for node/edge data

---

## Visual Effects

### Expected Layout Characteristics

1. **Relationship Clustering**
   - Dependencies form directional flows toward prerequisites
   - Enabled capabilities cluster around enabling technologies
   - Component parts orbit their parent entities tightly
   - Cross-domain bridges create long-range connections

2. **Physicality Separation**
   - Virtual entities (software/digital) cluster separately
   - Physical entities (hardware/objects) form distinct group
   - Conceptual entities (ideas/theories) gather in separate region
   - Clear visual separation between physical natures

3. **Role Organization**
   - Processes (workflows/actions) group together
   - Agents (actors/entities) form active cluster
   - Resources (data/materials) collect separately
   - Concepts (theories/ideas) cluster distinctly

4. **Maturity Staging**
   - Emerging technologies at lower Z (experimental layer)
   - Mature technologies at Z=0 (established layer)
   - Declining technologies at upper Z (legacy layer)
   - Clear lifecycle progression visualization

5. **Cross-Domain Bridging**
   - Heavily-connected domains pull closer together
   - Hub nodes with many cross-domain links attract domains
   - Domain boundaries remain visible but connected
   - Strength adapts to relationship density

---

## Troubleshooting

### Forces Too Strong

Reduce force strengths:
```rust
requires_strength: 0.4,  // from 0.7
enables_strength: 0.2,   // from 0.4
```

### Clusters Too Tight

Increase rest lengths and radii:
```rust
cluster_radius: 250.0,   // from 180.0
has_part_orbit_radius: 90.0,  // from 60.0
```

### Cross-Domain Too Weak

Increase cross-domain multipliers:
```rust
link_count_multiplier: 0.2,  // from 0.1
max_strength_boost: 3.0,     // from 2.0
```

### Maturity Stages Overlap

Increase stage separation:
```rust
stage_separation: 150.0,  // from 100.0
```

---

## Related Documentation

- [Ontology Parser Guide](./ontology-parser.md)
- 
- 
- 

---

## Implementation Files

- **Rust Engine:** `/src/gpu/semantic_forces.rs`
- **CUDA Kernels:** `/src/utils/semantic_forces.cu`
- **Configuration:** Defined in `SemanticConfig` struct
- **Tests:** `/tests/semantic_forces_test.rs`
