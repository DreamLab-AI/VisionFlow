# VisionFlow System Refactor & Validation Tasks

**Objective:** Fix CQRS async issues, validate all functionality, ensure knowledge and ontology graphs render correctly.

**Status:** 🔴 In Progress
**Started:** 2025-10-23 18:45 UTC

---

## Phase 1: Core Backend Fixes ⚙️

### 1.1 Fix Async Runtime Issues
- [x] Fix CQRS handlers using Runtime::new() pattern in queries.rs (5 handlers)
- [x] Fix CQRS handlers using Runtime::new() pattern in directives.rs (6 handlers)
- [x] Replace `tokio::spawn` with `actix::spawn` in 3 actor files
- [x] Run `cargo build` - compilation successful
- [x] Add missing API routes (/api/health, /api/config)

### 1.2 Backend Restart & Validation
- [x] Build succeeded (1m 48s, 132 warnings, 0 errors)
- [x] Backend started and stable (no crashes)
- [x] Test `/api/health` endpoint - ✅ WORKS (200 OK)
- [⚠️] Test `/api/config` endpoint - HANGS (timeout)
- [⚠️] Test `/api/settings` endpoint - HANGS (timeout)
- [⚠️] Test `/api/settings/batch` endpoint - HANGS (timeout)
- [⚠️] Tokio runtime panics still occurring in actor initialization

### 1.3 Critical Issues to Fix
- [x] Fix CQRS async runtime issue (removed block_in_place wrapper)
- [x] Add request timeouts (30s HTTP middleware, 5s actor timeout utils)
- [⚠️] **ACTIVE ISSUE:** Actor initialization timing - `ctx.run_later`/`ctx.run_interval` called before Tokio reactor ready
  - **Root Cause:** Actors call these in `started()` method when no reactor context exists yet
  - **Affected:** GraphServiceActor (line 2497), AgentMonitorActor (line 202), OntologyActor (line 602), TaskOrchestratorActor
  - **Error:** "there is no reactor running, must be called from the context of a Tokio 1.x runtime"
  - **Solution Needed:** Move timer initialization to message handler or delay until runtime ready
- [x] Add actix_timeout utilities for preventing actor hangs

---

## Phase 2: Frontend Integration Testing 🖥️

### 2.1 Settings API Endpoints
- [ ] Test GET /api/settings - load all settings
- [ ] Test POST /api/settings - update all settings
- [ ] Test GET /api/settings/path/{path} - get by path
- [ ] Test PUT /api/settings/path/{path} - update by path
- [ ] Test POST /api/settings/batch - batch load
- [ ] Test GET /api/settings/physics/{profile} - physics settings
- [ ] Test POST /api/settings/cache/clear - clear cache

### 2.2 Knowledge Graph Endpoints
- [ ] Test GET /api/graph - load graph data
- [ ] Test POST /api/graph/nodes - add nodes
- [ ] Test PUT /api/graph/nodes/{id} - update node
- [ ] Test DELETE /api/graph/nodes/{id} - delete node
- [ ] Verify graph renders in browser at http://192.168.0.51:3001

### 2.3 Ontology Endpoints
- [ ] Test GET /api/ontology - load ontology
- [ ] Test POST /api/ontology/classes - add class
- [ ] Test GET /api/ontology/hierarchy - get hierarchy
- [ ] Verify ontology graph renders in browser

---

## Phase 3: Visual Validation 👁️

### 3.1 Knowledge Graph Rendering
- [ ] Open VisionFlow UI in Chrome DevTools
- [ ] Navigate to Knowledge Graph view
- [ ] Verify nodes render correctly
- [ ] Verify edges render correctly
- [ ] Test physics simulation (nodes should move)
- [ ] Test zoom/pan controls
- [ ] Test node selection
- [ ] Test node filtering

### 3.2 Ontology Graph Rendering
- [ ] Navigate to Ontology view
- [ ] Verify class hierarchy renders
- [ ] Verify relationships display correctly
- [ ] Test ontology navigation
- [ ] Test class expansion/collapse

---

## Phase 4: Control Center Settings 🎛️

### 4.1 System Settings
- [ ] Test system.debug.enabled toggle
- [ ] Test system.logging.level dropdown
- [ ] Test system.network.port input
- [ ] Test system.gpu.enabled toggle
- [ ] Verify settings persist after reload
- [ ] Verify settings reflect in backend logs

### 4.2 Physics Settings
- [ ] Test physics.enabled toggle
- [ ] Test physics.spring_constant slider
- [ ] Test physics.repulsion_constant slider
- [ ] Test physics.damping slider
- [ ] Test physics.gravity toggle
- [ ] Verify physics changes affect graph simulation

### 4.3 Rendering Settings
- [ ] Test rendering.node_size slider
- [ ] Test rendering.edge_width slider
- [ ] Test rendering.color_scheme dropdown
- [ ] Test rendering.labels_visible toggle
- [ ] Verify visual changes apply immediately

### 4.4 Performance Settings
- [ ] Test performance.max_nodes input
- [ ] Test performance.fps_limit slider
- [ ] Test performance.gpu_acceleration toggle
- [ ] Test performance.cache_enabled toggle
- [ ] Verify performance metrics update

---

## Phase 5: Database Validation 💾

### 5.1 Settings Database
- [ ] Verify settings.db exists at /app/data/settings.db
- [ ] Check settings table schema
- [ ] Verify default settings are populated
- [ ] Test settings CRUD operations
- [ ] Verify cache invalidation works

### 5.2 Knowledge Graph Database
- [ ] Verify knowledge_graph.db exists
- [ ] Check nodes table has data (should have 185 nodes)
- [ ] Check edges table has data (should have 4014 edges)
- [ ] Verify graph integrity (no orphaned edges)
- [ ] Test graph queries performance

### 5.3 Ontology Database
- [ ] Verify ontology.db exists
- [ ] Check ontology classes table
- [ ] Check ontology properties table
- [ ] Verify OWL/RDF data is valid
- [ ] Test ontology reasoning

---

## Phase 6: Integration Points 🔗

### 6.1 RAGFlow Integration
- [ ] Verify RAGFLOW_API_KEY is set
- [ ] Test /api/ragflow/query endpoint
- [ ] Test document upload to RAGFlow
- [ ] Verify RAG responses in UI
- [ ] **NOTE:** If missing, mock with placeholder responses

### 6.2 Nostr Integration
- [ ] Verify NOSTR credentials exist
- [ ] Test /api/nostr/publish endpoint
- [ ] Test event subscription
- [ ] **NOTE:** If missing, mock with simulated events

### 6.3 GitHub Integration
- [ ] Verify GitHub token exists
- [ ] Test background sync (logs should show sync attempts)
- [ ] **NOTE:** If missing, disable or mock sync

---

## Phase 7: Error Handling & Edge Cases ⚠️

### 7.1 API Error Responses
- [ ] Test invalid settings paths (should 404)
- [ ] Test malformed JSON (should 400)
- [ ] Test unauthorized access (should 401 if auth enabled)
- [ ] Test database connection failure handling
- [ ] Verify error messages are user-friendly

### 7.2 Frontend Error Handling
- [ ] Test connection lost scenarios
- [ ] Test invalid graph data handling
- [ ] Test settings load failure
- [ ] Verify error toasts/notifications appear
- [ ] Test graceful degradation

---

## Phase 8: Performance & Optimization 🚀

### 8.1 Backend Performance
- [ ] Profile settings endpoint response times (<50ms target)
- [ ] Check database query performance
- [ ] Verify cache hit rates (should be >80%)
- [ ] Test concurrent request handling
- [ ] Monitor memory usage under load

### 8.2 Frontend Performance
- [ ] Check graph rendering FPS (>30fps target)
- [ ] Test large graph performance (1000+ nodes)
- [ ] Verify GPU acceleration is active
- [ ] Test physics simulation performance
- [ ] Monitor WebSocket latency

---

## Phase 9: Documentation Updates 📚

### 9.1 Code Documentation
- [ ] Update CLAUDE.md with findings
- [ ] Document async pattern changes
- [ ] Document CQRS architecture
- [ ] Add inline code comments for complex logic
- [ ] Update API documentation

### 9.2 User Documentation
- [ ] Update README with current architecture
- [ ] Document settings structure
- [ ] Add troubleshooting guide
- [ ] Document database schema
- [ ] Add development setup guide

---

## Phase 10: Final Validation ✅

### 10.1 End-to-End Testing
- [ ] Fresh container restart
- [ ] Load UI in clean browser session
- [ ] Walk through complete user flow:
  - [ ] View knowledge graph
  - [ ] Adjust physics settings
  - [ ] Add/edit nodes
  - [ ] View ontology
  - [ ] Change system settings
  - [ ] Verify all changes persist
- [ ] Check for console errors
- [ ] Check for backend panics

### 10.2 Sign-off Checklist
- [ ] All critical endpoints returning 200
- [ ] Knowledge graph renders with 185 nodes
- [ ] Ontology graph renders correctly
- [ ] All settings functional
- [ ] No panics in rust-error.log
- [ ] No console errors in browser
- [ ] Documentation updated
- [ ] Task list complete

---

## Notes & Issues 📝

### Mocked Credentials/Data
*(Will be populated as discovered)*

### Known Issues
1. **FIXED:** CQRS handlers - removed `tokio::task::block_in_place` wrapper, now uses `Runtime::new().block_on()` directly
2. **CRITICAL - IN PROGRESS:** Actor initialization timing issue
   - **Error:** "there is no reactor running" when actors call `ctx.run_later`/`ctx.run_interval` in `started()` method
   - **First Attempt:** Fixed `actix::spawn` → `ctx.spawn` (this was correct but not the root cause)
   - **Real Issue:** Tokio reactor not available during actor `started()` method execution
   - **Workaround Needed:** Delay timer initialization until after first message or use different initialization pattern

### Performance Metrics
*(To be filled during testing)*

---

**Last Updated:** 2025-10-23 18:45 UTC
**Completion:** 0% (0/100+ tasks)
