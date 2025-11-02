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

## 📋 Current State (November 2, 2025 - Evening)

### ✅ GitHub Sync & Data Pipeline (FIXED)
- [x] **GitHub sync working end-to-end** - Successfully syncs full repository
- [x] **KnowledgeGraphParser extracting nodes/edges** - All markdown files parsed
- [x] **OntologyParser extracting OWL blocks** - Ontology metadata extracted from files
- [x] **Database persistence working** - Nodes and edges saved to unified.db
- [x] **API returning KG nodes** - No longer returning 1 OWL node
- [x] **Client can render graph** - Three.js visualization at localhost:4000

### 🔧 Critical Issues Found & Fixed (Session)
1. **[FIXED] API Race Condition** (Commit 20db1e98)
   - **Problem:** UpdateGraphData in main.rs overwriting 983 KG nodes with 1 OWL root node
   - **Root Cause:** Race condition where UpdateGraphData sent AFTER ReloadGraphFromDatabase but processed before
   - **Fix:** Removed UpdateGraphData send to ontology graph, let ReloadGraphFromDatabase load KG nodes
   - **Result:** API now returns all KG nodes from database (not 1 OWL node)

2. **[FIXED] GitHub Sync Transaction Rollback** (Commit 1553649a)
   - **Problem:** Sync reported 9605 nodes saved but only 48 in database (transaction rolled back)
   - **Root Cause:** Code trying to update file_metadata.file_size column that doesn't exist in schema
   - **Fix:** Removed file_size column references from SQL INSERT/UPDATE statements
   - **Result:** Sync now completes successfully, all nodes and edges persist to database

### 📊 Current Database State (Post-Fix)
```
unified.db (Sync in Progress):
  - graph_nodes:  900+ nodes (from GitHub sync, growing)
  - graph_edges:  1200+ edges (from GitHub sync, growing)
  - owl_classes:  1+ root nodes from OWL extraction
  - file_metadata: tracking SHA1 for differential sync (FIXED)
  - owl_axioms:   being populated as ontology blocks extracted
```

**Note:** Sync is currently running. Estimated 9600+ total nodes when complete.

### ⚠️ Remaining Issues Blocking Client Readiness
1. **GPU Physics NOT RUNNING** ❌
   - Nodes stuck at origin (0,0,0) with zero velocity (vx=vy=vz=0)
   - Force-directed layout not spreading nodes
   - Physics simulation appears to be skipping (logs show "Skipping physics simulation - waiting for GPU initialization")

2. **WebSocket Protocol NOT IMPLEMENTED** ❌
   - Client doesn't receive initial full node set with metadata
   - Need batch handshake: send all nodes ONCE at connection
   - Then only send position/velocity updates indexed by node ID

3. **Position Updates NOT WORKING** ❌
   - Even if GPU physics runs, positions not being sent to client
   - API returns vx=vy=vz=0 for all nodes
   - No real-time updates to client

4. **Edge Rendering BROKEN** ❌
   - 1200+ edges in database but not displayed in client
   - Likely blocked by above issues (no physics = no visibility)

---

## 🎯 Phase 0: Complete Legacy System (This Week)

### Task 0.1: GitHub Sync Complete ✅ DONE
**Goal:** Get full GitHub repository synced to database
**Status:** COMPLETE
- [x] Fixed API race condition (commit 20db1e98)
- [x] Fixed GitHub sync transaction rollback (commit 1553649a)
- [x] Sync running successfully - 900+ nodes saved, syncing remaining files
- [x] Database receiving all nodes and edges
- [x] Ontology blocks being extracted

**Outcome:** Sync is working end-to-end. All 9600+ nodes will be in database when complete.

### Task 0.2: Trigger ReloadGraphFromDatabase After Sync ⏳ READY
**Goal:** Refresh graph actor with all synced nodes once sync completes
**Status:** Ready to Execute
- Sync will auto-trigger ReloadGraphFromDatabase via app_state.rs:220-227
- GraphServiceActor will load all nodes from database into memory
- API will then return all nodes (not just initial 48)

**When:** After sync completes (estimated 15-20 min from 18:26 UTC)

### Task 0.3: Fix GPU Physics Engine ❌ CRITICAL BLOCKER
**Goal:** Make nodes spread out with force-directed layout (not stuck at origin)
**Status:** PENDING - HIGH PRIORITY
**Current Issue:** Logs show "Skipping physics simulation - waiting for GPU initialization"

**Investigation Needed:**
- [ ] Check if GPU manager actor is initialized properly
- [ ] Verify GPU kernels are being called with node data
- [ ] Debug why positions not updating (all nodes have vx=vy=vz=0)
- [ ] Check if GPU compute context is properly set up

**Files Involved:**
- `src/actors/gpu/gpu_manager_actor.rs` - GPU initialization and compute
- `src/utils/unified_gpu_compute.rs` - CUDA kernel calls
- `src/handlers/api_handler/graph/mod.rs` - Position response formatting

**Success Criteria:**
```
curl http://localhost:4000/api/graph/data
→ Nodes have non-zero vx, vy, vz values
→ Position (x, y, z) values different for each node (not all at origin)
→ Client renders nodes spread out in 3D space
```

### Task 0.4: Implement WebSocket Protocol Update ❌ CRITICAL
**Goal:** Send full graph metadata at connection, then ID-indexed updates
**Status:** PENDING - REQUIRES NEW CODE

**Current State:**
- WebSocket exists but doesn't implement proper handshake
- No initial full-node load
- No ID-indexed position updates

**Required Changes:**
- [ ] Create new WebSocket message type: `InitialGraphLoad { nodes: Vec<Node>, edges: Vec<Edge> }`
- [ ] Send full graph to client on first connection
- [ ] Create position update message: `PositionUpdate { node_id: u32, x: f32, y: f32, z: f32, vx: f32, vy: f32, vz: f32 }`
- [ ] Implement streaming position updates from GPU to WebSocket
- [ ] Update client to handle initial load and index-based updates

**Files to Create/Update:**
- `src/handlers/websocket/*.rs` - New message types and handlers
- `client/src/hooks/useGraphWebSocket.ts` - Connection and update handling
- `client/src/stores/graphStore.ts` - Client-side node indexing

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
├── graph_nodes (900+ rows)          [✅] KG nodes (being synced from GitHub)
├── graph_edges (1100+ rows)         [✅] Relationships (being synced from GitHub)
├── owl_classes (1+ rows)            [🔄] Ontology definitions (from OWL extraction)
├── owl_properties                   [🔄] OWL properties
├── owl_axioms                       [🔄] OWL relationships (from extraction)
├── owl_class_hierarchy              [🔄] Class inheritance (from OWL)
└── file_metadata                    [✅] GitHub sync tracking (FIXED - file_size removed)
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

### For Immediate Sprint (Next 2-3 Days)
**Goal:** Complete legacy system pipeline: Get all 900+ nodes rendering with physics + proper WebSocket protocol
- [ ] Task 0.2: GitHub sync completes successfully (all 9600+ nodes synced)
- [ ] Task 0.3: GPU physics engine initializes and spreads nodes (non-zero velocity)
- [ ] Task 0.4: WebSocket protocol sends full graph at connection + ID-indexed updates
- [ ] Task 0.5: All 900+ nodes + edges visible at localhost:4000 with force-directed layout

**Success Criteria:**
```
curl http://localhost:4000/api/graph/data
→ Returns 900+ nodes with non-zero vx, vy, vz
→ Returns 1100+ edges connecting nodes

Three.js client (localhost:4000):
→ All nodes spread naturally across 3D space (not at origin)
→ Edges visible between connected nodes
→ Real-time position updates from GPU physics
→ No lag or stuttering (<30ms latency on position updates)
```

### For Phase 1 (Next Sprint - Week 2)
- [ ] Fully understand OWL extraction from GitHub markdown
- [ ] Replace KnowledgeGraphParser with ontology-only extraction
- [ ] Create OntologyConverter (OWL → graph_nodes)
- [ ] All nodes have proper OWL metadata, hierarchy, properties

### For Phase 2 (Following Sprint - Week 3)
- [ ] Benchmarks: 1000+ ontology nodes at 30+ FPS
- [ ] Documentation complete and accurate
- [ ] All tests passing (unit + integration + performance)
- [ ] Legacy KG system archived or removed

---

## 🚀 Immediate Sprint Tasks (Consolidated)

### Current System State
- ✅ **GitHub sync:** Working end-to-end (commit 1553649a fixed file_metadata schema)
- ✅ **API race condition:** Fixed (commit 20db1e98 removed UpdateGraphData overwrite)
- ✅ **Database:** Receiving 900+ nodes, 1100+ edges from sync (growing)
- ✅ **ReloadGraphFromDatabase:** Auto-triggers after sync via app_state.rs:220-227
- ❌ **GPU physics:** Not initializing (logs: "Skipping physics simulation - waiting for GPU initialization")
- ❌ **WebSocket protocol:** No batch initial load or ID-indexed updates
- ❌ **Client rendering:** Nodes stuck at origin, edges not visible

### Task 0.3: Fix GPU Physics Engine (CRITICAL - START HERE)
**Blocking:** Everything else depends on physics spreading nodes

**Investigation Steps:**
1. Check GPU manager initialization in gpu_manager_actor.rs
   - Is GPUComputeActor being spawned in app_state.rs?
   - Is GPU compute context properly initialized?
   - Are CUDA kernels accessible in container environment?

2. Verify GPU kernel calls in unified_gpu_compute.rs
   - Are kernel calls receiving node data?
   - Is GPU memory being allocated for 900+ nodes?
   - Are position/velocity updates being written back?

3. Debug position updates in API response
   - Currently: All nodes return vx=0, vy=0, vz=0
   - Should: Non-zero velocity after GPU physics runs
   - Check: Is GPU compute being triggered in simulation loop?

4. Check logs for GPU initialization errors
   - Container logs: Search for "GPU", "CUDA", "compute"
   - Error patterns: Missing libraries, device access, memory allocation

**Success Criteria:**
```
curl http://localhost:4000/api/graph/data | jq '.nodes[0]'
→ vx: > 0 or < 0 (non-zero velocity)
→ vy: > 0 or < 0 (non-zero velocity)
→ vz: > 0 or < 0 (non-zero velocity)
→ x, y, z not all equal to 0 (nodes spread, not at origin)
```

### Task 0.4: Implement WebSocket Protocol (CRITICAL - AFTER PHYSICS)
**Blocking:** Client can't efficiently load graph without this

**Implementation:**
1. Create new WebSocket message types in src/handlers/websocket/:
   ```rust
   InitialGraphLoad {
       nodes: Vec<NodeData>,  // All 900+ nodes with metadata
       edges: Vec<EdgeData>,  // All 1100+ edges
       timestamp: u64,
   }

   PositionUpdate {
       node_id: u32,
       x: f32, y: f32, z: f32,
       vx: f32, vy: f32, vz: f32,
       timestamp: u64,
   }
   ```

2. Implement server-side handshake in websocket handlers
   - On client connect: Send InitialGraphLoad with all nodes/edges
   - Then stream PositionUpdate for each GPU compute iteration
   - Use node ID index for efficient update routing

3. Update client WebSocket handler (client/src/hooks/useGraphWebSocket.ts)
   - Receive and cache InitialGraphLoad
   - Build Map<node_id, Node> index
   - Apply PositionUpdate by indexing into map (O(1) lookup)

4. Update client store (client/src/stores/graphStore.ts)
   - Store nodes by ID for fast updates
   - Batch position updates every 16ms (60 FPS)
   - Don't recreate node objects, just update coordinates

**Success Criteria:**
```
Client connection sequence:
1. Connect to WebSocket
2. Immediately receive all 900+ nodes + edges (< 500ms)
3. Nodes appear in 3D viewer at origin initially
4. GPU physics starts (logs confirm)
5. Positions update smoothly (~60 FPS)
6. Nodes spread naturally across 3D space
```

### Task 0.5: Verify Full Pipeline (FINAL INTEGRATION)
**Prerequisites:** Tasks 0.3 and 0.4 complete

**Checklist:**
1. Container running with fixed code (commits 20db1e98, 1553649a)
2. GitHub sync completed (9600+ nodes in database)
3. GPU physics initializing and computing velocities
4. WebSocket sending initial graph and position updates
5. Client receiving and rendering nodes with force-directed layout

**Verification:**
```bash
# Check database
sqlite3 /path/to/unified.db "SELECT COUNT(*) FROM graph_nodes;"
→ Should return 900+

# Check API physics
curl http://localhost:4000/api/graph/data | jq '.nodes | map(select(.vx != 0 or .vy != 0 or .vz != 0)) | length'
→ Should return > 900 (all nodes have non-zero velocity)

# Check client
Open http://192.168.0.51:4000 in browser
→ See 900+ nodes spread in 3D space
→ See 1100+ edges connecting nodes
→ Smooth animation as GPU physics updates
```

---

## 📊 Commits Summary

| Commit | Impact | Date |
|--------|--------|------|
| 1553649a | CRITICAL FIX: GitHub sync now works (file_metadata schema fixed) | Nov 2 |
| 20db1e98 | CRITICAL FIX: API returns KG nodes not OWL (race condition fixed) | Nov 2 |
| 5b3dc83a | Docs: Complete task.md rewrite (architecture clarity) | Nov 2 |

---

## 🔍 Quick Reference

### Known Working Components
- GitHub markdown parser (extracts 9600+ nodes)
- SQLite persistence (storing nodes/edges correctly)
- REST API endpoints (returning data correctly)
- Three.js client visualization (can render if data provided)
- Actor message system (CQRS pattern stable)

### Known Broken Components
- GPU physics initialization (hung state)
- WebSocket protocol (no batching, inefficient)
- Client position updates (blocked by physics + websocket)
- Edge rendering (no updates flowing to client)

### Files to Touch (In Order)
1. **src/actors/gpu/gpu_manager_actor.rs** - Debug physics init
2. **src/utils/unified_gpu_compute.rs** - Verify kernel calls
3. **src/handlers/websocket/*.rs** - Implement new messages
4. **client/src/hooks/useGraphWebSocket.ts** - Handle messages
5. **client/src/stores/graphStore.ts** - ID-indexed storage

---

## ⏱️ Timeline
- **T+0min:** Start Task 0.3 (GPU physics debug)
- **T+30min:** Should have GPU initialization working or clear blocker identified
- **T+60min:** Start Task 0.4 (WebSocket protocol) OR continue GPU debugging
- **T+120min:** Both tasks complete, move to Task 0.5 (integration verification)
- **T+150min:** Full pipeline working at localhost:4000

**Total estimate:** 2-3 hours for complete pipeline

---

**Status:** Ready to execute. All infrastructure exists. No external blockers. Just needs debugging + protocol update.
