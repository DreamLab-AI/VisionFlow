# VisionFlow - Ontology-Based Graph Visualization Pipeline

**Status:** Major Architecture Transition in Progress
**Goal:** Complete migration from hybrid KG/Ontology nodes to unified ontology-based nodes with ported physics/networking
**Architecture:** Single unified.db with ontology classes as primary nodes (OWL classes → graph nodes)
**Last Updated:** November 2, 2025

---

## 🎯 High-Level Strategy

### Legacy System (Being Replaced)
```
GitHub (Logseq) → KnowledgeGraphParser → graph_nodes (KG concepts)
                                      → owl_classes (separate ontology)
                                      → owl_class_iri link (one-way reference)
```
**Problem:** Two parallel node systems, ontology was metadata, KG was primary

### New System (Current Target)
```
GitHub (Logseq) → OntologyExtractor → owl_classes (primary nodes)
                → OntologyConverter  → graph_nodes (ontology instances)
                                   → Properties, hierarchies, axioms
```
**Benefit:** Single authoritative source (OWL), physics/constraints/networking unchanged

### Porting Strategy
- ✅ GPU physics engine: **Reusable as-is** (works on any node set)
- ✅ CUDA kernels: **No changes needed** (node IDs stay same, just different source)
- ✅ WebSocket networking: **Needs protocol update** (initial node handshake)
- ⚠️ Client visualization: **Minor updates** (same rendering, different node metadata)
- 🔄 Database schema: **Minimal changes** (already supports both, just flip primary)

---

## 📋 Current State (November 2, 2025)

### ✅ Legacy System (Working)
- [x] GitHub sync → database (983 nodes, 1268 edges in unified.db)
- [x] KnowledgeGraphParser extracting nodes/edges from markdown
- [x] OntologyParser extracting OWL blocks from files
- [x] GPU physics initialized and running (50+ FPS target)
- [x] REST API returning graph data (with fix: now returns KG nodes not ontology root)
- [x] Client can render nodes/edges (at localhost:4000)

### 🔧 Critical Issues Found & Fixed (Today)
- [x] **API Bug:** `UpdateGraphData` in main.rs was overwriting 983 KG nodes with 1 OWL root
  - **Root Cause:** Race condition - UpdateGraphData sent AFTER ReloadGraphFromDatabase
  - **Fix Applied:** Removed UpdateGraphData send, let ReloadGraphFromDatabase load KG nodes
  - **Commit:** 20db1e98 "fix: Prevent ontology graph from overwriting KG nodes"

### 📊 Current Database State
```
unified.db:
  - graph_nodes:  983 nodes (from GitHub sync)
  - graph_edges:  1268 edges
  - owl_classes:  1 root node (mv:rb0088iso13482compliance)
  - file_metadata: tracking SHA1 for differential sync
  - owl_axioms:   empty (no OWL relationships yet)
```

### ⚠️ Remaining Issues (Blocking Client)
1. **GPU Physics:** Nodes at origin (0,0,0) - physics not updating positions
2. **WebSocket Protocol:** No initial full-node handshake, client not receiving node metadata
3. **Edge Rendering:** 1268 edges in DB but not displayed in client
4. **Node Positioning:** No force-directed layout, all nodes need physics simulation

---

## 🎯 Phase 0: Complete Legacy System (This Week)

### Task 0.1: Verify End-to-End Pipeline with Full Node Set
**Goal:** Get all 983 nodes + 1268 edges rendering at localhost:4000
**Status:** In Progress
- [ ] Wait for GitHub sync to complete (currently: 983 nodes)
- [ ] Restart container with fixed code (main.rs UpdateGraphData removal)
- [ ] Verify API returns 983 nodes + 1268 edges (not 1 OWL node)
- [ ] Check client renders all nodes with edges

**Files to Monitor:**
- `src/main.rs:315-329` - Ontology graph handling (FIXED)
- `src/app_state.rs:220-227` - ReloadGraphFromDatabase after sync
- `/app/logs/rust-error.log` - Application startup and sync completion logs

**Success Criteria:**
```
curl http://localhost:4000/api/graph/data
→ { "nodes": [...983 nodes...], "edges": [...1268 edges...] }
→ Client at localhost:4000 shows full graph with nodes positioned by GPU physics
```

### Task 0.2: Fix GPU Physics Engine
**Goal:** Make nodes spread out with force-directed layout (not stuck at origin)
**Status:** Pending
- [ ] Check if GPU kernels are receiving and processing data
- [ ] Verify position updates are being sent to API
- [ ] Confirm client receives position updates in WebSocket
- [ ] Debug why all nodes have vx=0, vy=0, vz=0 (no velocity)

**Files Involved:**
- `src/actors/gpu/gpu_manager_actor.rs` - GPU computation
- `src/utils/unified_gpu_compute.rs` - CUDA kernel interface
- `src/handlers/api_handler/graph/mod.rs` - Position response formatting

### Task 0.3: Implement Client WebSocket Protocol
**Goal:** Proper initial node load + real-time position updates
**Status:** Pending
**Architecture:**
```
Client connects → Server sends initial full graph (all 983 nodes + metadata)
                → Client stores node index: Map<NodeID, Node>
                → GPU physics runs
                → Server sends position updates (NodeID, x, y, z, vx, vy, vz)
                → Client updates via ID index (NOT full node objects)
```

**Files to Create/Update:**
- `src/handlers/websocket/*.rs` - WebSocket message handlers
- `client/src/hooks/useGraphWebSocket.ts` - Client WebSocket connection
- `client/src/stores/graphStore.ts` - Client-side node/edge store

**Success Criteria:**
- [ ] Client receives all 983 nodes at connection
- [ ] Client updates node positions in real-time
- [ ] No lag or full-object re-transfers for position updates

---

## 🔄 Phase 1: Migrate to Ontology-Based Nodes (Next Sprint)

### Task 1.1: Understand Ontology Extraction from GitHub
**Goal:** Map current KG parsing to OWL extraction
**Status:** Analysis needed
- [ ] Review OntologyParser (`src/services/parsers/ontology_parser.rs`)
- [ ] Check what OWL blocks are in GitHub markdown files
- [ ] Design mapping: KG page → OWL class IRI
- [ ] Plan: Should each KG node become an OWL class?

### Task 1.2: Create OntologyConverter
**Goal:** Transform OWL classes to graph_nodes
**Status:** Not started
**Implementation:**
- Create new service: `src/services/ontology_converter.rs`
- For each `owl_class`:
  - Create `graph_node` with `metadata_id = class.iri`
  - Extract position from class hierarchy (compute layout)
  - Store properties in node metadata
- Link axioms to edges: `SubClassOf` → edge, `DisjointClasses` → repulsion constraint

### Task 1.3: Update GitHub Sync Pipeline
**Goal:** Extract OWL → database, skip old KG parser
**Status:** Design pending
- [ ] Decide: Keep both parsers or replace?
- [ ] If replace: Remove KnowledgeGraphParser, use OntologyExtractor for all
- [ ] If hybrid: Keep KG parser for backwards compat, toggle via config
- [ ] Update batch processing to handle OWL conversion

---

## 🔌 Phase 2: Port Physics/Networking (Parallel with Phase 1)

### Task 2.1: GPU Physics on Ontology Nodes
**Goal:** Ensure physics engine works with ontology-based nodes (no code changes)
**Status:** Low priority (should work unchanged)
- [ ] Verify CUDA kernels don't depend on KG-specific metadata
- [ ] Test with 983 ontology nodes instead of KG nodes
- [ ] Benchmark: FPS, memory usage, constraint handling

**Files (No changes expected):**
- `src/physics/` - All constraint logic
- `src/actors/gpu/` - All GPU management

### Task 2.2: WebSocket Protocol for Ontology Nodes
**Goal:** Send ontology metadata in initial handshake
**Status:** Ready to implement
**Additional Metadata to Include:**
```json
{
  "id": 123,
  "metadataId": "mv:concept-name",
  "label": "Concept Name",
  "iri": "http://example.com/ontology#ConceptName",
  "parentClass": "http://example.com/ontology#ParentClass",
  "properties": { "definition": "...", "source": "..." },
  "position": { "x": 10, "y": 20, "z": 30 },
  "metadata": { ... }
}
```

---

## 📚 Supporting Tasks

### Documentation
- [ ] Update README.md: Explain ontology-based architecture
- [ ] Create MIGRATION.md: Legacy KG → Ontology transition guide
- [ ] Document OWL extraction from GitHub markdown
- [ ] API docs: List all ontology endpoints

### Testing
- [ ] Unit tests: OntologyConverter logic
- [ ] Integration tests: GitHub → OWL → Database → API
- [ ] Performance tests: 1000+ ontology nodes at 30+ FPS
- [ ] Client tests: WebSocket handshake, position updates

### Code Cleanup
- [ ] Archive old KnowledgeGraphParser if replaced
- [ ] Remove any KG-specific logic from GPU/networking
- [ ] Clean up temporary debug logging

---

## 🗂️ Repository Structure Reference

### Backend (Rust)
```
src/
├── services/
│   ├── github_sync_service.rs      [✅ Working] Batch sync from GitHub
│   ├── parsers/
│   │   ├── knowledge_graph_parser.rs  [✅ Working] Extract KG nodes/edges
│   │   ├── ontology_parser.rs        [✅ Working] Extract OWL from markdown
│   │   └── converter.rs              [TBD] OWL → Graph node conversion
│   └── edge_generation.rs           [✅ Available] Multi-modal edges
├── repositories/
│   └── unified_graph_repository.rs  [✅ Working] SQLite persistence
├── actors/
│   ├── graph_actor.rs               [✅ Working] Graph state management
│   ├── graph_service_supervisor.rs [✅ Working] Actor orchestration
│   └── gpu/
│       ├── gpu_manager_actor.rs     [✅ Working] CUDA kernel calls
│       └── ontology_constraint_actor.rs [✅ Partial] OWL axiom → forces
├── handlers/
│   ├── websocket/                   [🔄 Needs update] WebSocket protocol
│   └── api_handler/graph/           [✅ Working] REST API endpoints
└── physics/
    ├── ontology_constraints.rs      [✅ Partial] 5/6 axiom types
    └── stress_majorization.rs       [✅ Available] Graph optimization
```

### Frontend (React + TypeScript)
```
client/src/
├── components/
│   └── GraphVisualization.tsx       [✅ Working] Three.js renderer
├── hooks/
│   └── useGraphWebSocket.ts         [🔄 Needs protocol update] WS connection
├── stores/
│   └── graphStore.ts                [🔄 Needs ID-index update] State management
└── types/
    └── graph.ts                     [✅ Working] Node/Edge interfaces
```

### Database (SQLite)
```
unified.db
├── graph_nodes (983 rows)           [✅] KG nodes (to be replaced by ontology)
├── graph_edges (1268 rows)          [✅] Relationships
├── owl_classes (1 row)              [🔄] Ontology definitions
├── owl_properties                   [🔄] OWL properties
├── owl_axioms                       [🔄] OWL relationships (empty)
├── owl_class_hierarchy              [🔄] Class inheritance (empty)
└── file_metadata                    [✅] GitHub sync tracking
```

---

## 🔑 Key Decisions (Architecture)

### 1. Keep or Replace KnowledgeGraphParser?
**Option A: Keep Both** (Safer)
- Pro: Backwards compatible, can run both in parallel for validation
- Con: Maintains two parsers, potential confusion
- Recommendation: Keep during transition, archive after validation

**Option B: Replace** (Cleaner)
- Pro: Single source of truth, simpler codebase
- Con: Risk if OWL extraction misses anything
- Recommendation: After full validation on 1000+ nodes

### 2. Physics Engine Architecture
**Current:** Works on any node ID set
**Decision:** NO CHANGES NEEDED - WASM SIMD + CUDA kernels are generic

### 3. Client Communication Protocol
**Current:** Individual node updates (inefficient)
**New:** Batch initial load + ID-indexed updates
**Implementation:** See WebSocket tasks above

---

## ✅ Definition of Done

### For Phase 0 (This Week)
- [ ] All 983 nodes + 1268 edges visible at localhost:4000
- [ ] GPU physics spreading nodes naturally (force-directed)
- [ ] WebSocket protocol sending full graph on initial connection
- [ ] Client updates positions in real-time with minimal latency

### For Phase 1 (Next Sprint)
- [ ] OWL extraction producing usable ontology nodes
- [ ] OntologyConverter transforms OWL → graph_nodes correctly
- [ ] GitHub sync using new pipeline (OWL-based)
- [ ] All nodes have proper metadata, hierarchy, properties

### For Phase 2 (Following Sprint)
- [ ] Benchmarks: 1000+ nodes at 30+ FPS
- [ ] Documentation complete and accurate
- [ ] All tests passing (unit + integration + performance)
- [ ] Legacy KG system archived or removed

---

## 🚀 Immediate Next Steps

1. **Monitor container sync** - Wait for 983 nodes to finish processing
2. **Restart with fix** - Container with removed UpdateGraphData call
3. **Verify API** - Test `/api/graph/data` returns all 983 nodes
4. **Debug GPU physics** - Check why nodes are at origin
5. **Test WebSocket** - Verify client receives full initial node set

**Blockers:** None identified - all infrastructure exists, just needs integration
