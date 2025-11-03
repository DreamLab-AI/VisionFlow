# VisionFlow Ontology Integration Roadmap

**Last Updated**: 2025-01-03
**Current Version**: v0.1.0 (Ontology Foundation Complete)
**Target Version**: v1.0.0 (Full Semantic Graph Visualization)

---

## Executive Summary

This roadmap tracks the implementation of a complete ontology-driven graph visualization system. The vision: **Every graph node is semantically typed via OWL ontologies, enabling intelligent physics simulation, hierarchical visualization, and semantic reasoning.**

### Current Status: **Phase 1 Complete (40% of Vision)**

✅ **What's Working:**
- Basic ontology pipeline (GitHub → Parse → Classify → DB → GPU)
- Class-based physics modifiers (charge/mass scaling)
- Semantic edge classification (relationship types)
- Automated class inference (path, content, metadata)
- GPU-accelerated clustering & community detection
- Real-time binary WebSocket streaming

❌ **Critical Gaps:**
- Neo4j dual persistence (graph queries)
- Full ontology reasoning pipeline (whelk inference)
- Ontology-driven physics (disjointWith, subClassOf forces)
- Stress majorization global optimization
- Client-side hierarchical nesting
- Advanced semantic constraints

---

## Gap Analysis: Current vs. Vision

### ✅ Phase 1: Data Ingestion & Dual Persistence (70% Complete)

| Component | Status | Implementation |
|-----------|--------|----------------|
| GitHubSyncService | ✅ **DONE** | Webhook-triggered sync, batch processing |
| OntologyExtractor | ✅ **DONE** | Parses OWL classes, enriches nodes with `owl_class_iri` |
| SQLite Persistence | ✅ **DONE** | Stores classes, properties, axioms, nodes, edges |
| **Neo4j Adapter** | ❌ **MISSING** | No graph database for Cypher queries |
| Classification | ✅ **DONE** | Path-based, content-based, metadata-based inference |

**Gap**: Neo4j dual persistence would enable powerful graph queries like:
```cypher
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
WHERE c.industry = 'Technology'
RETURN p.name, c.name
```

---

### ⚠️ Phase 2: Ontology Reasoning (30% Complete)

| Component | Status | Implementation |
|-----------|--------|----------------|
| whelk-rs Engine | ✅ **PRESENT** | EL++ reasoner integrated |
| hornedowl Parser | ✅ **PRESENT** | OWL ontology parsing capability |
| **Active Reasoning** | ❌ **MISSING** | No automated inference pipeline |
| **Inferred Axioms** | ❌ **MISSING** | Not saved back to database |
| Transitive Closure | ❌ **MISSING** | No `subClassOf*` inference |
| Inverse Properties | ❌ **MISSING** | No `worksAt ↔ employs` inference |

**Gap**: The reasoner exists but isn't actively used. Need to:
1. Load ontology into whelk on startup
2. Run reasoning after each ontology change
3. Materialize inferred axioms to database
4. Use inferred knowledge in physics & visualization

**Example Missing Inference**:
```turtle
# Input
Person subClassOf Agent
Employee subClassOf Person

# Should Infer (but doesn't)
Employee subClassOf Agent  # Transitive closure
```

---

### ⚠️ Phase 3: Graph Loading (100% Complete)

| Component | Status | Implementation |
|-----------|--------|----------------|
| GraphServiceActor | ✅ **DONE** | Loads enriched graph data |
| owl_class_iri Links | ✅ **DONE** | Every node has semantic type |
| In-Memory Graph | ✅ **DONE** | Efficient node_map lookup |

**Status**: **FULLY IMPLEMENTED** ✅

---

### ⚠️ Phase 4: Unified Physics Simulation (50% Complete)

| Component | Status | Implementation |
|-----------|--------|----------------|
| Basic Forces | ✅ **DONE** | Repulsion (all-pairs), Attraction (edges) |
| Class Modifiers | ✅ **DONE** | `class_charge`, `class_mass` scaling |
| **Ontology Forces** | ❌ **MISSING** | No semantic axiom-based forces |
| **disjointWith** | ❌ **MISSING** | Should create strong repulsion |
| **subClassOf** | ❌ **MISSING** | Should create hierarchical attraction |
| User Constraints | ✅ **DONE** | Fixed position, distance constraints |
| **Stress Majorization** | ❌ **MISSING** | No global layout optimization |

**Gap**: Current physics is "ontology-aware" (uses class metadata) but not "ontology-driven" (doesn't enforce semantic rules).

**Example Missing Forces**:
```rust
// Should implement:
if class_i.disjointWith(class_j) {
    repulsion_force *= 5.0;  // Strong separation
}

if class_i.subClassOf(class_j) {
    apply_hierarchical_attraction(i, j, parent_force);
}

if property.domain != node.owl_class {
    apply_constraint_violation_penalty();
}
```

**Stress Majorization**: Global optimization algorithm that minimizes stress:
```
stress = Σ w_ij (||p_i - p_j|| - d_ij)²
```
Should run every 200 iterations to correct accumulated layout drift.

---

### ✅ Phase 5: Advanced Analysis (90% Complete)

| Component | Status | Implementation |
|-----------|--------|----------------|
| Leiden Clustering | ✅ **DONE** | GPU-accelerated community detection |
| k-means Clustering | ✅ **DONE** | GPU-accelerated spatial clustering |
| Anomaly Detection | ✅ **DONE** | LOF (Local Outlier Factor) |
| **Semantic Clustering** | ❌ **MISSING** | Clustering by owl_class_iri |

**Gap**: Current clustering is purely topological. Should add semantic dimensions:
```rust
// Cluster by both topology AND semantics
let semantic_similarity = if node_i.owl_class == node_j.owl_class { 1.0 } else { 0.0 };
let distance = topology_distance * (1.0 - semantic_weight)
             + semantic_distance * semantic_weight;
```

---

### ⚠️ Phase 6: Real-time Visualization (70% Complete)

| Component | Status | Implementation |
|-----------|--------|----------------|
| Binary WebSocket | ✅ **DONE** | High-performance streaming |
| Position Updates | ✅ **DONE** | 60 FPS node position sync |
| **Visual Nesting** | ❌ **MISSING** | No hierarchical class grouping |
| **Class Filtering** | ❌ **MISSING** | Can't hide/show by class |
| **Semantic Zoom** | ❌ **MISSING** | No level-of-detail by class hierarchy |

**Gap**: Client receives `owl_class_iri` but doesn't use it for visual organization.

**Example Missing Features**:
```typescript
// Should implement:
interface ClassGroupOptions {
    groupByClass: boolean;      // Visually nest nodes in parent meshes
    collapseClasses: string[];  // Hide specific classes
    highlightClass: string;     // Emphasize one class
    semanticZoom: number;       // Show detail based on class hierarchy level
}
```

---

## Detailed Roadmap

### 🎯 **Phase 1: Full Ontology Reasoning Pipeline** (2-3 weeks)
**Priority**: 🔴 CRITICAL
**Goal**: Activate whelk reasoner to materialize inferred knowledge

#### Tasks:
1. **Create OntologyReasoningService** (`src/services/ontology_reasoning_service.rs`)
   ```rust
   pub struct OntologyReasoningService {
       reasoner: Arc<WhelkInferenceEngine>,
       ontology_repo: Arc<dyn OntologyRepository>,
   }

   impl OntologyReasoningService {
       pub async fn run_inference(&self) -> Result<Vec<OwlAxiom>> {
           // 1. Load all axioms from database
           // 2. Build whelk ontology
           // 3. Run reasoning
           // 4. Extract inferred axioms
           // 5. Save to database
       }
   }
   ```

2. **Implement Axiom Materialization**
   - Transitive closure for `subClassOf`
   - Inverse property inference
   - Domain/range validation
   - Disjointness propagation

3. **Integrate with GitHub Sync**
   ```rust
   // After saving ontology data
   self.reasoning_service.run_inference().await?;
   ```

4. **Add Inference Triggers**
   - Run on startup (warm cache)
   - Run after ontology updates
   - Run on-demand via API

**Success Metrics**:
- Reasoner produces >100 inferred axioms from test ontology
- Inference completes in <5 seconds for 1000 classes
- Materialized axioms appear in database

---

### 🎯 **Phase 2: Ontology-Driven Physics Forces** (3-4 weeks)
**Priority**: 🔴 CRITICAL
**Goal**: Make physics enforce semantic constraints from ontology

#### Tasks:

1. **Disjoint Class Repulsion** (`visionflow_unified.cu`)
   ```cuda
   __global__ void force_pass_kernel(...,
       const int* __restrict__ disjoint_pairs,  // [num_pairs * 2]
       const int num_disjoint_pairs) {

       // Check if nodes belong to disjoint classes
       int my_class = class_id[idx];
       int neighbor_class = class_id[neighbor_idx];

       for (int i = 0; i < num_disjoint_pairs; i++) {
           if ((disjoint_pairs[i*2] == my_class &&
                disjoint_pairs[i*2+1] == neighbor_class) ||
               (disjoint_pairs[i*2] == neighbor_class &&
                disjoint_pairs[i*2+1] == my_class)) {
               repulsion *= 5.0f;  // Strong separation
               break;
           }
       }
   }
   ```

2. **Hierarchical Class Attraction**
   ```cuda
   __global__ void apply_hierarchical_forces(...,
       const int* __restrict__ subclass_pairs,  // [num_pairs * 2]
       const int num_subclass_pairs) {

       // Pull subclasses toward superclasses
       for (int i = 0; i < num_subclass_pairs; i++) {
           int subclass = subclass_pairs[i*2];
           int superclass = subclass_pairs[i*2+1];

           if (class_id[idx] == subclass) {
               // Find nearest superclass node
               // Apply gentle attraction
           }
       }
   }
   ```

3. **Domain/Range Constraint Forces**
   ```cuda
   // For edges: if property.domain != source.class
   // Apply constraint violation penalty
   if (edge_property_domain[edge_idx] != class_id[source]) {
       float penalty_force = c_params.constraint_violation_penalty;
       // Push nodes apart or highlight violation
   }
   ```

4. **Upload Ontology Buffers to GPU**
   ```rust
   // In UnifiedGPUCompute
   pub disjoint_pairs: DeviceBuffer<i32>,  // Pairs of disjoint class IDs
   pub subclass_pairs: DeviceBuffer<i32>,  // Child→Parent class pairs
   pub property_domains: DeviceBuffer<i32>, // domain[property_id] = class_id
   pub property_ranges: DeviceBuffer<i32>,  // range[property_id] = class_id

   pub fn upload_ontology_constraints(&mut self,
       disjoint: &[(i32, i32)],
       subclass: &[(i32, i32)]) -> Result<()>
   ```

**Success Metrics**:
- Disjoint classes (Person/Organization) visually separated by 2x normal distance
- Subclass nodes (Employee) cluster near superclass nodes (Person)
- Domain/range violations create visible tension in layout

---

### 🎯 **Phase 3: Neo4j Dual Persistence** (2-3 weeks)
**Priority**: 🟡 MEDIUM
**Goal**: Enable graph database queries alongside SQLite

#### Tasks:

1. **Create Neo4j Adapter** (`src/adapters/neo4j_adapter.rs`)
   ```rust
   pub struct Neo4jAdapter {
       client: neo4rs::Graph,
   }

   impl Neo4jAdapter {
       pub async fn create_graph_constructs(
           &self,
           nodes: &[Node],
           edges: &[Edge]
       ) -> Result<()> {
           // Batch Cypher queries
           let query = "
               UNWIND $nodes AS node
               MERGE (n {id: node.id})
               SET n.label = node.label,
                   n.owl_class_iri = node.owl_class_iri
           ";
           self.client.run(query).await?;
       }
   }
   ```

2. **Sync Pipeline Integration**
   ```rust
   // In GitHubSyncService
   self.neo4j_adapter.create_graph_constructs(&nodes, &edges).await?;
   ```

3. **Query Interface**
   ```rust
   pub async fn query_semantic_path(
       &self,
       start_class: &str,
       relationship: &str,
       end_class: &str
   ) -> Result<Vec<Path>> {
       let query = "
           MATCH path = (a)-[r]->(b)
           WHERE a.owl_class_iri = $start_class
           AND type(r) = $relationship
           AND b.owl_class_iri = $end_class
           RETURN path
       ";
       // Execute and return results
   }
   ```

**Success Metrics**:
- All nodes/edges synced to Neo4j within 5 seconds of GitHub sync
- Complex graph queries (3+ hop paths) complete in <100ms
- Cypher queries can filter by `owl_class_iri`

---

### 🎯 **Phase 4: Stress Majorization** (2 weeks)
**Priority**: 🟡 MEDIUM
**Goal**: Periodic global layout optimization

#### Tasks:

1. **CUDA Stress Majorization Kernel** (`stress_majorization.cu`)
   ```cuda
   __global__ void compute_stress_kernel(
       const float* pos_x, const float* pos_y, const float* pos_z,
       const float* graph_distances,  // Shortest path distances
       float* stress_gradients_x,
       float* stress_gradients_y,
       float* stress_gradients_z,
       int num_nodes
   ) {
       // Compute gradient: ∂stress/∂p_i
       // stress = Σ w_ij (||p_i - p_j|| - d_ij)²
   }
   ```

2. **Integration with Physics Loop**
   ```rust
   if iteration % 200 == 0 {
       self.run_stress_majorization(max_iterations: 50)?;
   }
   ```

3. **Distance Matrix Computation**
   - Use existing SSSP (Single-Source Shortest Path) for graph distances
   - Cache distance matrix on GPU

**Success Metrics**:
- Layout quality (measured by graph drawing metrics) improves by 30%
- Stress majorization converges in <100ms for 10k nodes
- Visual appearance: fewer edge crossings, more uniform edge lengths

---

### 🎯 **Phase 5: Client-Side Hierarchical Visualization** (2-3 weeks)
**Priority**: 🟢 LOW
**Goal**: Visual nesting and semantic zoom

#### Tasks:

1. **Class Grouping UI Controls**
   ```typescript
   interface OntologyViewOptions {
       groupByClass: boolean;
       collapsedClasses: Set<string>;
       highlightedClass: string | null;
       semanticZoomLevel: number;  // 0-5
   }
   ```

2. **Visual Nesting Implementation**
   ```typescript
   class HierarchicalRenderer {
       createClassParentMeshes() {
           for (const classIri of uniqueClasses) {
               const parentMesh = new THREE.Mesh(
                   new THREE.BoxGeometry(),
                   new THREE.MeshBasicMaterial({
                       color: getClassColor(classIri),
                       transparent: true,
                       opacity: 0.2
                   })
               );
               this.classParents.set(classIri, parentMesh);
           }
       }

       updateNodePositions(nodes: NodeData[]) {
           for (const node of nodes) {
               if (this.options.groupByClass && node.owl_class_iri) {
                   const parent = this.classParents.get(node.owl_class_iri);
                   // Position node relative to parent
                   parent.add(nodeMesh);
               }
           }
       }
   }
   ```

3. **Semantic Zoom Levels**
   - Level 0: Show all nodes
   - Level 1: Collapse leaf classes
   - Level 2: Show only mid-level classes
   - Level 3: Show only top-level classes

**Success Metrics**:
- User can toggle "Group by Class" and see visual nesting
- Collapsing a class hides all its nodes instantly
- Semantic zoom reduces visible node count by 80% at level 3

---

### 🎯 **Phase 6: Advanced Semantic Features** (3-4 weeks)
**Priority**: 🟢 LOW
**Goal**: Leverage full ontology reasoning

#### Tasks:

1. **Semantic Search**
   ```rust
   pub async fn search_by_class_and_property(
       &self,
       class_iri: &str,
       property_iri: &str,
       value: &str
   ) -> Result<Vec<Node>> {
       // Search nodes by semantic type + property
   }
   ```

2. **Ontology Validation**
   ```rust
   pub fn validate_graph_against_ontology(&self) -> Vec<ValidationError> {
       // Check domain/range constraints
       // Check disjointness violations
       // Check cardinality constraints
   }
   ```

3. **Semantic Recommendations**
   ```rust
   pub fn suggest_relationships(&self, node_id: u32) -> Vec<SuggestedEdge> {
       // Based on node's owl_class and available properties
   }
   ```

---

## Implementation Priorities

### High Priority (Next 2 Months)
1. ✅ **DONE**: Basic ontology classification pipeline
2. 🔴 **Phase 1**: Full reasoning with whelk (inferred axioms)
3. 🔴 **Phase 2**: Ontology-driven physics forces

### Medium Priority (Months 3-4)
4. 🟡 **Phase 3**: Neo4j dual persistence
5. 🟡 **Phase 4**: Stress majorization

### Lower Priority (Months 5-6)
6. 🟢 **Phase 5**: Client hierarchical visualization
7. 🟢 **Phase 6**: Advanced semantic features

---

## Success Criteria: Vision Achievement

### When can we declare "Vision Complete"?

✅ **Data Ingestion**: GitHub → Ontology → Dual Persistence (SQLite + Neo4j)
✅ **Reasoning**: Automated inference with materialized axioms
✅ **Physics**: Unified force model with semantic constraints
✅ **Visualization**: Hierarchical nesting by class
✅ **Analysis**: Semantic-aware clustering
✅ **Queries**: Graph database support for complex patterns

### Key Performance Indicators

| Metric | Current | Target |
|--------|---------|--------|
| Classification Accuracy | 85% | 95% |
| Reasoning Time (1k classes) | N/A | <5s |
| Physics Semantic Awareness | 20% | 100% |
| Client Features | 70% | 100% |
| Query Performance (Neo4j) | N/A | <100ms |

---

## Investment Analysis

### Time Investment Required

| Phase | Duration | FTE Required | Total Person-Hours |
|-------|----------|--------------|-------------------|
| Phase 1: Reasoning | 2-3 weeks | 1 FTE | 80-120 hours |
| Phase 2: Physics | 3-4 weeks | 1 FTE | 120-160 hours |
| Phase 3: Neo4j | 2-3 weeks | 1 FTE | 80-120 hours |
| Phase 4: Stress | 2 weeks | 1 FTE | 80 hours |
| Phase 5: Client | 2-3 weeks | 1 FTE | 80-120 hours |
| Phase 6: Advanced | 3-4 weeks | 1 FTE | 120-160 hours |

**Total**: 16-24 weeks @ 1 FTE = **560-840 person-hours**

### Return on Investment

**Current State Value**: 40% of vision
- Basic ontology classification works
- GPU physics functional
- Real-time visualization working

**After Critical Phases (1-2)**:
- **Value**: 75% of vision
- **Time**: 5-7 weeks
- **ROI**: Very High (semantic intelligence activated)

**After All Phases**:
- **Value**: 100% of vision
- **Time**: 16-24 weeks
- **ROI**: Complete semantic graph platform

## Technical Debt & Risks

### Known Issues
1. **whelk reasoning not active**: Engine present but pipeline incomplete
2. **No Neo4j integration**: Missing graph query capability
3. **Limited ontology forces**: Only basic charge/mass modifiers
4. **Client visual nesting**: Not implemented

### Mitigation Strategies
1. **Reasoning**: Implement OntologyReasoningService with robust error handling
2. **Neo4j**: Use neo4rs crate, add connection pooling
3. **Physics**: Incremental addition of semantic forces (disjoint → hierarchical → constraints)
4. **Client**: Iterative UI enhancements with user feedback

---

## Architecture Decisions

### Why Dual Persistence (SQLite + Neo4j)?
- **SQLite**: Fast lookups, ACID transactions, embedded database
- **Neo4j**: Graph queries, relationship traversal, Cypher patterns
- **Trade-off**: 2x write cost, but enables powerful semantic queries

### Why whelk-rs over other reasoners?
- **Performance**: Tractable EL++ fragment, linear complexity
- **Rust-native**: No FFI overhead, memory-safe
- **Trade-off**: Limited expressivity vs. full OWL (but sufficient for our use case)

### Why CUDA for physics?
- **Performance**: 100x faster than CPU for 10k+ nodes
- **Parallelism**: Natural fit for all-pairs repulsion
- **Trade-off**: GPU dependency, but acceptable for desktop application

---

## Timeline Summary

| Phase | Duration | Dependencies | Outcome |
|-------|----------|--------------|---------|
| **Phase 1: Reasoning** | 2-3 weeks | None | Inferred axioms materialized |
| **Phase 2: Physics** | 3-4 weeks | Phase 1 | Semantic forces active |
| **Phase 3: Neo4j** | 2-3 weeks | None | Graph queries enabled |
| **Phase 4: Stress** | 2 weeks | Phase 2 | Layout optimization |
| **Phase 5: Client** | 2-3 weeks | Phase 2 | Hierarchical UI |
| **Phase 6: Advanced** | 3-4 weeks | All above | Full semantic features |

**Total Estimated Time**: 4-6 months to full vision implementation

---

## Next Steps (Immediate Actions)

1. **Week 1-2**: Implement `OntologyReasoningService`
   - Create service structure
   - Integrate whelk inference
   - Add database save for inferred axioms

2. **Week 3-4**: Test reasoning pipeline
   - Create test ontology with 100+ classes
   - Validate transitive closure
   - Measure inference performance

3. **Week 5-7**: Begin ontology-driven physics
   - Implement disjoint class repulsion
   - Upload constraint buffers to GPU
   - Visual validation

4. **Week 8**: Evaluate progress and adjust roadmap

---

**Navigation:** [📖 Documentation Index](INDEX.md) | [🏗️ Architecture](architecture/) | [📊 Progress Chart](PROGRESS_CHART.md) | [📚 Main README](../README.md)

---

**Document Maintainer**: Development Team
**Review Frequency**: Bi-weekly
**Last Major Update**: Phase 1 (Classification) completed 2025-01-03
