# Architecture Decision Records (ADRs)
## VisionFlow Future Architecture

**Document**: Key architectural decisions with rationale
**Date**: 2025-10-31
**Status**: Decision Log

---

## ADR Index

1. [ADR-001: Single Database Strategy](#adr-001-single-database-strategy)
2. [ADR-002: Ontology-First Architecture](#adr-002-ontology-first-architecture)
3. [ADR-003: GPU Physics Over CPU](#adr-003-gpu-physics-over-cpu)
4. [ADR-004: Custom Reasoner vs. External](#adr-004-custom-reasoner-vs-external)
5. [ADR-005: Hierarchical Expansion Strategy](#adr-005-hierarchical-expansion-strategy)
6. [ADR-006: Constraint Priority System](#adr-006-constraint-priority-system)
7. [ADR-007: WebSocket Over REST for Updates](#adr-007-websocket-over-rest-for-updates)
8. [ADR-008: Semantic LOD Over Distance LOD](#adr-008-semantic-lod-over-distance-lod)

---

## ADR-001: Single Database Strategy

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team

### Context

Currently have multiple databases:
- `ontology.db` - OWL classes, properties, axioms
- `knowledge_graph.db` - Graph nodes, edges, positions
- `settings.db` - Application settings

**Problem**: Data duplication, synchronization issues, unclear single source of truth.

### Decision

**Use `ontology.db` as the primary database. Repurpose `knowledge_graph.db` for runtime state only.**

**Structure**:
```
ontology.db (Persistent)
├── owl_classes
├── owl_properties
├── owl_axioms
├── owl_individuals
├── owl_class_hierarchy
└── inference_results

knowledge_graph.db (Runtime/Ephemeral) - OPTIONAL
├── node_positions (x, y, z, vx, vy, vz)
├── node_states (expanded, pinned, etc.)
└── user_overrides (temporary constraints)

settings.db (Unchanged)
└── Application configuration
```

### Rationale

**Pros**:
- ✅ Single source of truth for ontology
- ✅ Clear separation: persistent (ontology) vs. ephemeral (positions)
- ✅ Easier backup and version control
- ✅ Faster reasoning (fewer joins)

**Cons**:
- ❌ Runtime state in separate DB (could use in-memory instead)
- ❌ Migration effort from current system

**Alternatives Considered**:

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **A: Single DB** | Simple, no sync issues | Mixed concerns | ❌ Rejected |
| **B: Three DBs** | Clear separation | Duplication, sync | ❌ Current (problematic) |
| **C: ontology.db + runtime state** | Best of both | Slight complexity | ✅ **CHOSEN** |
| **D: Full in-memory** | Fastest | No persistence | ❌ Rejected |

### Consequences

**Migration Steps**:
1. Create new `ontology.db` schema
2. Migrate data from `knowledge_graph.db`
3. Update application to use `ontology.db`
4. Deprecate `knowledge_graph.db` or repurpose

**Impact**: Low-risk, high-benefit refactoring

---

## ADR-002: Ontology-First Architecture

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team, Product Team

### Context

Traditional graph visualization: Nodes + Edges + Layout Algorithm

**Problem**: Layout is purely aesthetic, doesn't reflect semantic meaning.

### Decision

**Design system architecture with ontology as the core, treating OWL axioms as physics constraints.**

**Architecture Layers**:
```
Markdown (Source) → Ontology (Semantics) → Constraints (Physics) → GPU (Computation) → Visualization (Display)
```

### Rationale

**Pros**:
- ✅ Layout has **meaning** (reflects logical structure)
- ✅ Semantic queries are natural (class hierarchy, disjointness)
- ✅ Extensible (new axioms = new physics constraints)
- ✅ Reasoning improves layout quality

**Cons**:
- ❌ More complex than basic force-directed
- ❌ Requires OWL knowledge for users
- ❌ Performance depends on reasoning speed

**Alternatives Considered**:

| Approach | Layout Quality | Semantic Fidelity | Complexity | Decision |
|----------|---------------|------------------|-----------|----------|
| **Pure Force-Directed** | Medium | None | Low | ❌ Too simple |
| **Hierarchical Only** | High | Medium | Medium | ❌ Not flexible |
| **Ontology-Driven** | High | High | High | ✅ **CHOSEN** |
| **ML-Based** | Variable | Low | Very High | ❌ Unpredictable |

### Consequences

**Benefits**:
- Users can understand graph structure from ontology
- Layout automatically adapts to ontology changes
- Queries are more powerful (semantic, not just spatial)

**Risks**:
- Learning curve for ontology concepts
- Mitigated by: Good defaults, tutorials, tooltips

**Impact**: High-benefit, manageable complexity

---

## ADR-003: GPU Physics Over CPU

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team, Performance Team

### Context

Need to simulate 10,000+ nodes at 60 FPS.

**Options**:
- CPU-based physics (sequential or multi-threaded)
- GPU-based physics (CUDA/OpenCL)
- Hybrid (CPU for small graphs, GPU for large)

### Decision

**Use GPU (CUDA) for ALL physics computation, regardless of graph size.**

### Rationale

**Performance Comparison** (10,000 nodes):

| Approach | Frame Time | FPS | Scalability |
|----------|-----------|-----|-------------|
| **CPU Single-Thread** | ~500ms | 2 FPS | ❌ Poor |
| **CPU Multi-Thread (8 cores)** | ~80ms | 12 FPS | ❌ Limited |
| **GPU (RTX 3060)** | ~8ms | 125 FPS | ✅ Excellent |

**Pros**:
- ✅ 10-100x faster than CPU
- ✅ Scales to 100K+ nodes
- ✅ Frees CPU for other tasks (rendering, UI)
- ✅ Already have CUDA kernels implemented

**Cons**:
- ❌ Requires NVIDIA GPU
- ❌ More complex debugging
- ❌ GPU memory constraints (8-12GB typical)

**Alternatives Considered**:

| Option | Performance | Portability | Complexity | Decision |
|--------|------------|------------|-----------|----------|
| **CPU Only** | Poor | Excellent | Low | ❌ Too slow |
| **WebGPU** | Good | Excellent | Medium | ⏳ Future (browser-based) |
| **CUDA** | Excellent | Poor | Medium | ✅ **CHOSEN** |
| **OpenCL** | Good | Good | High | ❌ Less mature |
| **Hybrid** | Variable | Good | High | ❌ Complexity |

### Consequences

**System Requirements**:
- NVIDIA GPU (GTX 1060 or newer)
- CUDA 11.0+ toolkit
- 4GB+ GPU memory

**Fallback Strategy**:
- If no GPU: Limit to 1000 nodes, use CPU physics
- Show warning: "For best experience, use NVIDIA GPU"

**Impact**: High performance, acceptable hardware requirement

---

## ADR-004: Custom Reasoner vs. External

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team

### Context

Need OWL reasoning for:
- Subsumption (A ⊑ B, B ⊑ C → A ⊑ C)
- Disjointness inference
- Property chain composition
- Cardinality constraints

**Options**:
1. **Custom Reasoner**: Implement core algorithms in Rust
2. **Horned-OWL**: Rust-native OWL 2 EL reasoner
3. **HermiT/Pellet**: JVM-based full OWL 2 DL reasoners

### Decision

**Hybrid Approach**:
- **Custom reasoner** for hot-path (physics computation)
- **Horned-OWL** for validation and advanced queries
- **Cache inference results** in `inference_results` table

### Rationale

**Comparison**:

| Reasoner | Speed | OWL Support | Integration | GPU-Friendly | Decision |
|----------|-------|------------|------------|--------------|----------|
| **Custom** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ Hot-path |
| **Horned-OWL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ✅ Validation |
| **HermiT** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | ⏳ Offline only |

**Custom Reasoner Scope** (Minimum Viable):
```rust
// Core algorithms to implement
1. Subsumption (transitivity)
   If A ⊑ B and B ⊑ C, then A ⊑ C

2. Disjointness
   If A ⊓ B = ⊥, store constraint

3. Property Characteristics
   - Functional (≤1 value)
   - Transitive (P ∘ P ⊑ P)
   - Symmetric (P(x,y) → P(y,x))
```

**Horned-OWL Integration**:
```rust
use horned_owl::ontology::Ontology;
use horned_owl::model::*;

// Use for comprehensive validation
pub fn validate_ontology(ont: &Ontology) -> Vec<ValidationError> {
    // Let Horned-OWL check consistency
    // Run offline or on-demand
}
```

### Consequences

**Performance**:
- Custom reasoner: <1ms for 1000 axioms (in-memory)
- Horned-OWL: <100ms for validation (acceptable for offline)

**Trade-offs**:
- ✅ Fast physics computation
- ✅ Comprehensive validation when needed
- ❌ Two reasoning systems to maintain

**Migration Path**:
1. Start with custom reasoner only
2. Add Horned-OWL for validation (Phase 2)
3. Optionally integrate HermiT for full OWL 2 DL (Phase 3)

**Impact**: Good balance of performance and capability

---

## ADR-005: Hierarchical Expansion Strategy

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team, UX Team

### Context

Large ontologies (100K+ entities) cannot all be displayed at once.

**Need**: Progressive disclosure mechanism.

### Decision

**Implement hierarchical expansion with level-based navigation.**

**Expansion Model**:
```
Level 0: Meta-node (entire ontology collapsed)
Level 1: Top-level classes
Level 2: Immediate subclasses
Level 3-5: Nested subclasses
Level 6+: Individuals
```

**User Interaction**:
- Click node → Expand children
- Ctrl+Click → Collapse to parent
- Shift+Click → Expand all descendants (with limit)

### Rationale

**Alternatives Considered**:

| Strategy | Scalability | Usability | Implementation | Decision |
|----------|-----------|-----------|----------------|----------|
| **Show All** | ❌ Poor | ❌ Overwhelming | Easy | ❌ Rejected |
| **Filtering Only** | ❌ Poor | ⭐⭐⭐ | Easy | ❌ Insufficient |
| **Hierarchical Expansion** | ✅ Excellent | ⭐⭐⭐⭐⭐ | Medium | ✅ **CHOSEN** |
| **Semantic Zoom** | ✅ Good | ⭐⭐⭐ | Hard | ⏳ Future enhancement |

**Expansion Algorithm**:
```rust
pub enum ExpansionStrategy {
    // Show top N by centrality
    TopNCentrality { n: usize },

    // Show representatives (distributed across hierarchy)
    Representatives { n: usize },

    // Show all if count <= threshold
    ShowAll { threshold: usize },
}
```

### Consequences

**Benefits**:
- Users start with overview, drill down to details
- Graph remains navigable at all scales
- Smooth animations improve UX

**Performance**:
- Expanded nodes cached in memory
- Lazy loading from database
- Animations: 0.5s ease-out

**Impact**: Critical for large-scale ontologies

---

## ADR-006: Constraint Priority System

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team

### Context

Multiple constraints can conflict (e.g., user pins node, but axiom says it should be elsewhere).

**Need**: Deterministic priority resolution.

### Decision

**Implement 10-level priority system with user overrides.**

**Priority Levels**:
```
1. User Manual Override (highest)
2. Identity Constraints (owl:sameAs)
3. Disjoint Constraints (owl:disjointWith)
4. Hierarchy Constraints (rdfs:subClassOf)
5. Property Constraints (owl:ObjectProperty)
6. Inferred Constraints (reasoner output)
7. Soft Alignment (aesthetic)
8. Base Physics (force-directed)
9. Optimization (LOD, performance)
10. Background (hints)
```

**Resolution Algorithm**:
```rust
fn resolve_constraints(constraints: Vec<Constraint>) -> Vec<Constraint> {
    // Group by node pair
    let grouped = group_by_node_pair(constraints);

    grouped.into_iter().map(|(pair, group)| {
        // If user-defined exists, use it exclusively
        if let Some(user) = group.iter().find(|c| c.user_defined) {
            return user.clone();
        }

        // Otherwise, weighted blend by priority
        blend_by_priority(group)
    }).collect()
}

fn blend_by_priority(constraints: Vec<Constraint>) -> Constraint {
    // Weight = 10^(-(priority - 1) / 9)
    // Priority 1 = 10x weight of priority 10
    let weights: Vec<f32> = constraints.iter()
        .map(|c| 10.0_f32.powf(-(c.priority as f32 - 1.0) / 9.0))
        .collect();

    // Weighted average of parameters
    // ...
}
```

### Rationale

**Pros**:
- ✅ Deterministic (same inputs = same output)
- ✅ User always has final say
- ✅ Respects ontology semantics
- ✅ Extensible (can add new priorities)

**Cons**:
- ❌ Blending can be complex
- ❌ User might not understand priority system

**Mitigation**:
- Visualize constraint conflicts in UI
- Explain priority levels in documentation
- Provide "why is this node here?" tooltip

### Consequences

**UI Implications**:
- Show constraint priority in node inspector
- Highlight conflicting constraints
- Allow users to promote constraint priority

**Performance**:
- Priority resolution: <0.1ms per node pair
- Cached after first frame

**Impact**: Essential for user control and predictability

---

## ADR-007: WebSocket Over REST for Updates

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team

### Context

Need to stream graph position updates to browser at 60 FPS.

**Options**:
- REST API with polling (GET /api/graph/positions every 16ms)
- Server-Sent Events (SSE)
- WebSocket (bidirectional)

### Decision

**Use WebSocket for real-time graph updates.**

**Protocol**:
```typescript
// Server → Client: Delta updates (incremental)
interface GraphUpdate {
    type: 'delta_update';
    timestamp: number;
    updates: Array<{
        node_id: number;
        position: [number, number, number];
        velocity?: [number, number, number];
    }>;
}

// Client → Server: User actions
interface UserAction {
    type: 'user_action';
    action: 'pin' | 'expand' | 'adjust_constraint';
    node_id?: number;
    parameters?: Record<string, any>;
}
```

### Rationale

**Comparison**:

| Method | Latency | Bandwidth | Scalability | Decision |
|--------|---------|-----------|------------|----------|
| **REST Polling** | High | ❌ Wasteful | Poor | ❌ Rejected |
| **Server-Sent Events** | Medium | Medium | Good | ⏳ Consider for notifications |
| **WebSocket** | Low | ✅ Efficient | Excellent | ✅ **CHOSEN** |

**Bandwidth Analysis** (10,000 nodes @ 60 FPS):

```
Full Update (naive):
  10,000 nodes × 12 bytes (3 floats) × 60 FPS = 7.2 MB/s

Delta Update (optimized):
  ~100 moving nodes × 12 bytes × 60 FPS = 72 KB/s
  Compression (gzip): ~20 KB/s

Result: 360x bandwidth reduction
```

**Protocol Optimizations**:
1. **Delta Encoding**: Only send changed nodes
2. **Throttling**: Client requests update rate (realtime/throttled/on_demand)
3. **Binary Protocol**: Use MessagePack instead of JSON (optional)
4. **Compression**: Gzip for large updates

### Consequences

**Benefits**:
- ✅ Low latency (<5ms typical)
- ✅ Efficient bandwidth usage
- ✅ Bidirectional (client can send actions)

**Challenges**:
- WebSocket connection management (reconnection logic)
- Load balancing with sticky sessions
- Scaling to 1000+ concurrent users

**Mitigation**:
- Implement heartbeat/ping-pong
- Use Redis for session state (multi-server)
- Connection pooling and rate limiting

**Impact**: Critical for real-time experience

---

## ADR-008: Semantic LOD Over Distance LOD

**Date**: 2025-10-31
**Status**: Accepted
**Deciders**: Architecture Team, UX Team

### Context

Level of Detail (LOD) needed for performance with large graphs.

**Traditional**: LOD based on distance from camera
**Problem**: Important nodes disappear when camera moves

### Decision

**Implement semantic LOD that considers both distance AND importance.**

**Importance Calculation**:
```rust
importance =
    0.3 × structural_centrality +      // Betweenness
    0.2 × ontology_role +               // Root class > property > individual
    0.3 × query_relevance +             // Match active query
    0.2 × user_interest                 // Interaction history
```

**LOD Selection**:
```rust
adjusted_distance = distance / (1 + importance × 2)

if adjusted_distance < 10:
    LOD::Full       // Full geometry + labels + metadata
else if adjusted_distance < 50:
    LOD::Medium     // Simplified geometry + label
else if adjusted_distance < 200:
    LOD::Low        // Billboard + no label
else:
    LOD::Culled     // Not rendered
```

### Rationale

**Comparison**:

| LOD Strategy | Important Nodes | Performance | User Experience | Decision |
|--------------|----------------|-------------|-----------------|----------|
| **Distance Only** | ❌ Can disappear | ✅ Simple | ⭐⭐ | ❌ Rejected |
| **Semantic Only** | ✅ Always visible | ❌ Poor (too many) | ⭐⭐⭐ | ❌ Rejected |
| **Hybrid (Semantic + Distance)** | ✅ Context-aware | ✅ Good | ⭐⭐⭐⭐⭐ | ✅ **CHOSEN** |

**Example Scenario**:

```
User searches for "Student" class

Without semantic LOD:
  - Student node disappears if camera moves away
  - User confused: "Where did my search result go?"

With semantic LOD:
  - Student node stays visible (high query_relevance)
  - Other unrelated nodes fade out
  - User happy: "I can always see what I searched for"
```

### Consequences

**Benefits**:
- ✅ Important content always accessible
- ✅ Better user experience (less frustration)
- ✅ Adapts to user context (search, interaction)

**Challenges**:
- Compute importance score efficiently
- Update importance when context changes
- Balance importance vs. performance

**Implementation**:
```typescript
class SemanticLodSystem {
    // Cache importance scores (update every 1s, not every frame)
    importanceCache: Map<NodeId, number>;

    update(deltaTime: number) {
        this.cacheTimer += deltaTime;

        if (this.cacheTimer > 1.0) {
            this.recomputeImportance();
            this.cacheTimer = 0;
        }

        // Use cached importance for LOD selection
        for (const node of this.nodes) {
            const importance = this.importanceCache.get(node.id) || 0;
            node.lod = this.selectLod(node, importance);
        }
    }
}
```

**Impact**: Significant UX improvement, moderate implementation complexity

---

## Summary Table: All Decisions

| ADR | Decision | Impact | Risk | Status |
|-----|----------|--------|------|--------|
| ADR-001 | Single database (ontology.db) | High | Low | ✅ Accepted |
| ADR-002 | Ontology-first architecture | High | Medium | ✅ Accepted |
| ADR-003 | GPU physics over CPU | High | Low | ✅ Accepted |
| ADR-004 | Hybrid reasoner (custom + Horned-OWL) | Medium | Low | ✅ Accepted |
| ADR-005 | Hierarchical expansion | High | Low | ✅ Accepted |
| ADR-006 | 10-level constraint priority | Medium | Low | ✅ Accepted |
| ADR-007 | WebSocket for updates | High | Low | ✅ Accepted |
| ADR-008 | Semantic LOD | High | Medium | ✅ Accepted |

---

## Decision Framework

When making future architectural decisions, use this framework:

### 1. Context
- What problem are we solving?
- What are the constraints?
- What are the alternatives?

### 2. Analysis
- Create comparison table (options vs. criteria)
- Quantify when possible (performance numbers)
- Consider non-functional requirements (scalability, maintainability)

### 3. Decision
- Choose one option
- Document rationale clearly
- Identify trade-offs explicitly

### 4. Consequences
- What changes in the system?
- What are the benefits?
- What are the risks?
- What is the mitigation plan?

### 5. Review
- Revisit decisions quarterly
- Update when new information emerges
- Mark as superseded if changed

---

**Document End** | Version 1.0 | 2025-10-31

**Next**: Implement decisions in order of impact (ADR-002, ADR-003, ADR-001, ...)
