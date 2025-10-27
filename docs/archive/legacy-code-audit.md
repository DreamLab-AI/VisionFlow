# Legacy Code Audit Report

**Generated**: 2025-10-26
**Project**: WebXR Graph Visualization
**Total Files Scanned**: 244 Rust source files
**Purpose**: Comprehensive audit of deprecated/removed code references

---

## Executive Summary

### Overall Status: ⚠️ MODERATE CLEANUP NEEDED

- **Critical Issues**: 3 (active deprecated code still in use)
- **Warning Issues**: 52 (documentation markers only)
- **Info Issues**: 11 (commented-out code)
- **Clean Areas**: Most deleted modules have been fully removed

### Key Findings

1. **GpuPhysicsAdapter**: Port trait still exists in codebase (should be removed)
2. **DockerHiveMind**: Referenced in deprecated files that are marked but not deleted
3. **NetworkRecoveryManager**: Still implemented despite being marked deprecated
4. **ErrorRecoveryMiddleware**: Still implemented despite being marked deprecated

---

## Critical Issues (Require Action)

### 1. GpuPhysicsAdapter - Port Still Exists

**Status**: ❌ **ACTIVE CODE THAT SHOULD BE REMOVED**

**Location**: `/home/devuser/workspace/project/src/ports/gpu_physics_adapter.rs`

**Issue**: The entire port trait and error types are still defined, even though adapters were removed.

**Found References**:
```rust
src/ports/mod.rs:36:    GpuDeviceInfo, GpuPhysicsAdapter, GpuPhysicsAdapterError, PhysicsStatistics, PhysicsStepResult,
src/ports/gpu_physics_adapter.rs:15:pub type Result<T> = std::result::Result<T, GpuPhysicsAdapterError>;
src/ports/gpu_physics_adapter.rs:18:pub enum GpuPhysicsAdapterError {
src/ports/gpu_physics_adapter.rs:69:pub trait GpuPhysicsAdapter: Send + Sync {
```

**Commented References**:
```rust
src/adapters/mod.rs:22:// pub use gpu_physics_adapter::GpuPhysicsAdapter as GpuPhysicsAdapterImpl;  // REMOVED: Incomplete stub
```

**Recommendation**:
- Delete `/home/devuser/workspace/project/src/ports/gpu_physics_adapter.rs`
- Remove exports from `/home/devuser/workspace/project/src/ports/mod.rs`
- Remove commented line from `/home/devuser/workspace/project/src/adapters/mod.rs`

---

### 2. NetworkRecoveryManager - Still Implemented

**Status**: ❌ **DEPRECATED BUT ACTIVE**

**Location**: `/home/devuser/workspace/project/src/utils/hybrid_fault_tolerance.rs`

**Issue**: Marked as deprecated in main.rs but still fully implemented in hybrid_fault_tolerance.rs

**Found References**:
```rust
src/main.rs:58:// DEPRECATED: ErrorRecoveryMiddleware removed - NetworkRecoveryManager deleted
src/main.rs:60:/// Simple error recovery middleware that integrates with NetworkRecoveryManager
src/main.rs:62:    recovery_manager: Option<Arc<webxr::utils::hybrid_fault_tolerance::NetworkRecoveryManager>>,
src/main.rs:72:    pub fn with_recovery_manager(recovery_manager: Arc<webxr::utils::hybrid_fault_tolerance::NetworkRecoveryManager>) -> Self {
src/main.rs:101:    recovery_manager: Option<Arc<webxr::utils::hybrid_fault_tolerance::NetworkRecoveryManager>>,

src/utils/hybrid_fault_tolerance.rs:458:pub struct NetworkRecoveryManager {
src/utils/hybrid_fault_tolerance.rs:498:impl NetworkRecoveryManager {
src/utils/hybrid_fault_tolerance.rs:863:) -> NetworkRecoveryManager {
src/utils/hybrid_fault_tolerance.rs:864:    NetworkRecoveryManager::new(
```

**Recommendation**:
- Remove `NetworkRecoveryManager` struct and implementation from `hybrid_fault_tolerance.rs`
- Remove related code from `main.rs` (struct definition, usage)
- Note: The entire `hybrid_fault_tolerance.rs` file has a deprecation notice at the top

---

### 3. ErrorRecoveryMiddleware - Still Implemented

**Status**: ❌ **DEPRECATED BUT ACTIVE**

**Location**: `/home/devuser/workspace/project/src/main.rs`

**Issue**: Marked as deprecated but still fully implemented with Transform trait

**Found References**:
```rust
src/main.rs:43:// DEPRECATED: std::future imports removed (were for ErrorRecoveryMiddleware)
src/main.rs:44:// DEPRECATED: Actix dev imports removed (were for ErrorRecoveryMiddleware)
src/main.rs:45:// DEPRECATED: LocalBoxFuture import removed (was for ErrorRecoveryMiddleware)
src/main.rs:58:// DEPRECATED: ErrorRecoveryMiddleware removed - NetworkRecoveryManager deleted

src/main.rs:61:pub struct ErrorRecoveryMiddleware {
src/main.rs:65:impl ErrorRecoveryMiddleware {
src/main.rs:79:impl<S, B> Transform<S, ServiceRequest> for ErrorRecoveryMiddleware
src/main.rs:88:    type Transform = ErrorRecoveryMiddlewareService<S>;
src/main.rs:92:        ready(Ok(ErrorRecoveryMiddlewareService {
src/main.rs:99:pub struct ErrorRecoveryMiddlewareService<S> {
src/main.rs:104:impl<S, B> Service<ServiceRequest> for ErrorRecoveryMiddlewareService<S>
src/main.rs:438:            // DEPRECATED: ErrorRecoveryMiddleware removed
```

**Recommendation**:
- Remove `ErrorRecoveryMiddleware` struct, implementation, and service
- Remove related commented-out import statements
- Clean up any middleware registration code

---

## Warning Issues (Documentation Only)

### Successfully Removed Modules (Documented)

The following modules have been successfully removed with proper documentation markers:

#### hybrid_health_handler
```rust
src/main.rs:18:        // DEPRECATED: hybrid_health_handler removed
src/main.rs:397:    // DEPRECATED: hybrid_health_manager_data, mcp_session_bridge, session_correlation_bridge removed
src/main.rs:453:            // DEPRECATED: hybrid_health_manager_data, mcp_session_bridge, session_correlation_bridge removed
src/main.rs:457:            // DEPRECATED: hybrid health routes removed
src/main.rs:468:                    // DEPRECATED: hybrid health routes removed
src/handlers/mod.rs:11:// pub mod hybrid_health_handler; // REMOVED: Deprecated hybrid health system
```

**Status**: ✅ **CLEAN** - Module properly removed, only documentation remains

---

#### sessions API
```rust
src/handlers/api_handler/mod.rs:8:// pub mod sessions; // REMOVED: Deprecated sessions API (returns 410 Gone)
src/handlers/api_handler/mod.rs:137:            // .configure(sessions::config)  // REMOVED: Deprecated sessions API (returns 410 Gone)
```

**Status**: ✅ **CLEAN** - Module properly removed, only documentation remains

---

#### SessionCorrelationBridge
```rust
src/handlers/client_log_handler.rs:50:    // DEPRECATED: SessionCorrelationBridge removed - using client session ID directly
src/telemetry/agent_telemetry.rs:44:    /// Create from UUID (used when SessionCorrelationBridge provides mapping)
src/telemetry/agent_telemetry.rs:350:        _bridge: Option<()>, // DEPRECATED: SessionCorrelationBridge removed
src/telemetry/agent_telemetry.rs:357:        // DEPRECATED: Bridge parameter removed, using fallback correlation ID
```

**Status**: ✅ **CLEAN** - Replaced with placeholder `Option<()>`, functionality removed

---

#### ActorGraphRepository
```rust
src/adapters/mod.rs:8:// pub mod actor_graph_repository;  // REMOVED: Incomplete stub adapter
src/adapters/mod.rs:19:// pub use actor_graph_repository::ActorGraphRepository;  // REMOVED: Incomplete stub
```

**Status**: ✅ **CLEAN** - Module properly removed, only documentation remains

---

### DEPRECATED Markers (52 total)

**Category**: Documentation/Warning markers

**Files with DEPRECATED markers**:

1. **src/main.rs** (11 occurrences)
   - HybridHealthManager removal notes
   - Import cleanup notes
   - Middleware removal notes
   - Route removal notes

2. **src/handlers/speech_socket_handler.rs** (5 occurrences)
   - HybridHealthManager parameter replaced with `Option<()>`
   - Swarm voice commands deprecated

3. **src/services/speech_voice_integration.rs** (2 occurrences)
   - SupervisorActor voice command handler removed

4. **src/handlers/multi_mcp_websocket_handler.rs** (3 occurrences)
   - HybridHealthManager parameter replaced with `Option<()>`

5. **src/handlers/bots_handler.rs** (3 occurrences)
   - HybridHealthManager parameter replaced with `Option<()>`

6. **src/utils/binary_protocol.rs** (10 occurrences)
   - Wire format v1 deprecation (backwards compatibility maintained)
   - `#[deprecated]` attributes on functions
   - `#[allow(deprecated)]` on legacy code paths

7. **src/handlers/socket_flow_handler.rs** (1 occurrence)
   - WebSocket physics update path deprecated

8. **src/actors/supervisor.rs** (1 occurrence)
   - Voice command handler using DockerHiveMind

9. **src/services/speech_service.rs** (1 occurrence)
   - Docker orchestration function removed

10. **src/utils/mcp_connection.rs** (1 occurrence)
    - Legacy Docker orchestration functions removed

11. **src/handlers/settings_handler.rs** (1 occurrence)
    - Legacy endpoints kept for compatibility

12. **src/handlers/api_handler/analytics/anomaly.rs** (1 occurrence)
    - Old anomaly detection function deprecated

13. **src/telemetry/agent_telemetry.rs** (2 occurrences)
    - SessionCorrelationBridge parameter removed

**Status**: ℹ️ **INFO** - These are documentation markers, no action needed

---

## Info Issues (Commented Code)

### Commented Module Declarations

**Location**: Various mod.rs files

```rust
src/lib.rs:21:// pub mod test_settings_fix;
src/lib.rs:30:// pub use models::ui_settings::UISettings; // Removed - consolidated into AppFullSettings"

src/adapters/mod.rs:8:// pub mod actor_graph_repository;  // REMOVED: Incomplete stub adapter
src/adapters/mod.rs:9:// pub mod gpu_physics_adapter;  // REMOVED: Incomplete stub adapter

src/config/mod.rs:184:// pub mod tests;

src/ontology/physics/mod.rs:3:// pub mod ontology_constraints;  // REMOVED: Duplicate stub, use src/physics/ontology_constraints.rs

src/actors/mod.rs:13:// pub mod supervisor_voice; // Removed - duplicate handlers in supervisor.rs

src/handlers/api_handler/mod.rs:8:// pub mod sessions; // REMOVED: Deprecated sessions API (returns 410 Gone)

src/handlers/mod.rs:11:// pub mod hybrid_health_handler; // REMOVED: Deprecated hybrid health system
```

**Recommendation**: Consider removing these commented lines entirely for cleaner code.

---

### Commented Use Statements

```rust
src/lib.rs:30:// pub use models::ui_settings::UISettings; // Removed - consolidated into AppFullSettings"

src/adapters/mod.rs:19:// pub use actor_graph_repository::ActorGraphRepository;  // REMOVED: Incomplete stub
src/adapters/mod.rs:22:// pub use gpu_physics_adapter::GpuPhysicsAdapter as GpuPhysicsAdapterImpl;  // REMOVED: Incomplete stub

src/ontology/physics/mod.rs:4:// pub use ontology_constraints::*;  // REMOVED: Module no longer exists
```

**Recommendation**: Remove these commented use statements.

---

## DockerHiveMind References

**Status**: ⚠️ **LEGACY CODE IN DEPRECATED FILES**

**Issue**: DockerHiveMind is referenced in files that are themselves deprecated:

```rust
src/utils/hybrid_performance_optimizer.rs:1:// DEPRECATED: Legacy hybrid performance optimizer removed
src/utils/hybrid_performance_optimizer.rs:16:use crate::utils::docker_hive_mind::{DockerHiveMind, SessionInfo, SwarmMetrics};
src/utils/hybrid_performance_optimizer.rs:46:    hive_mind: Arc<DockerHiveMind>,

src/utils/hybrid_fault_tolerance.rs:1:// DEPRECATED: Legacy hybrid fault tolerance removed
src/utils/hybrid_fault_tolerance.rs:16:use crate::utils::docker_hive_mind::{DockerHiveMind, SessionInfo, ContainerHealth};
src/utils/hybrid_fault_tolerance.rs:461:    docker_hive_mind: Arc<DockerHiveMind>,
src/utils/hybrid_fault_tolerance.rs:500:        docker_hive_mind: DockerHiveMind,
src/utils/hybrid_fault_tolerance.rs:724:    docker_hive_mind: Arc<DockerHiveMind>,
src/utils/hybrid_fault_tolerance.rs:755:    pub fn new(docker_hive_mind: DockerHiveMind, mcp_pool: MCPConnectionPool) -> Self {
src/utils/hybrid_fault_tolerance.rs:861:    docker_hive_mind: DockerHiveMind,
```

**Active References** (still in use):
```rust
src/services/management_api_client.rs:5://! operations, replacing the legacy DockerHiveMind system.

src/actors/supervisor.rs:388:// DEPRECATED: Voice command handler removed - uses legacy DockerHiveMind
src/actors/supervisor.rs:403:                // Use DockerHiveMind to spawn real agents
src/actors/supervisor.rs:460:                        // Use DockerHiveMind to terminate the agent
src/actors/supervisor.rs:486:                // Use DockerHiveMind to execute tasks
```

**Recommendation**:
- If `hybrid_performance_optimizer.rs` and `hybrid_fault_tolerance.rs` are deprecated, delete them entirely
- Update `supervisor.rs` to remove DockerHiveMind voice command handler
- Verify `docker_hive_mind` module is still needed or can be removed

---

## TODO/FIXME Analysis

### Active TODOs Found

```rust
src/actors/metadata_actor.rs:45:    // TODO: Re-implement or remove get_files_by_tag and get_files_by_type
```

**Status**: ℹ️ **INFO** - Single TODO for future implementation

**Other Technical Debt Markers**: Found in 24 files (see file list below)

---

## Stub Code Analysis

### Legitimate Stub Implementations

```rust
src/adapters/sqlite_settings_repository.rs:166:        // Return a stub implementation
src/adapters/sqlite_settings_repository.rs:188:        // Return success as stub

src/adapters/whelk_inference_engine.rs:330:                reasoner_version: "stub-0.1.0".to_string(),
```

**Status**: ✅ **ACCEPTABLE** - These are intentional placeholder implementations

---

### Removed Stub References (Documented)

```rust
src/adapters/mod.rs:8:// pub mod actor_graph_repository;  // REMOVED: Incomplete stub adapter
src/adapters/mod.rs:9:// pub mod gpu_physics_adapter;  // REMOVED: Incomplete stub adapter
src/adapters/mod.rs:19:// pub use actor_graph_repository::ActorGraphRepository;  // REMOVED: Incomplete stub
src/adapters/mod.rs:22:// pub use gpu_physics_adapter::GpuPhysicsAdapter as GpuPhysicsAdapterImpl;  // REMOVED: Incomplete stub

src/ontology/physics/mod.rs:3:// pub mod ontology_constraints;  // REMOVED: Duplicate stub, use src/physics/ontology_constraints.rs
```

**Status**: ✅ **CLEAN** - Properly documented removals

---

## Verification of Specific Deletions

### ArcCudaStream
```
grep -r "ArcCudaStream" src/
```
**Result**: ✅ **NO REFERENCES FOUND** - Successfully removed

---

### ontology_constraints Module
```rust
src/ontology/physics/mod.rs:3:// pub mod ontology_constraints;  // REMOVED: Duplicate stub, use src/physics/ontology_constraints.rs
src/ontology/physics/mod.rs:4:// pub use ontology_constraints::*;  // REMOVED: Module no longer exists
```

**Active Usage** (legitimate):
```rust
src/actors/gpu/ontology_constraint_actor.rs:23:use crate::physics::ontology_constraints::{
```

**Status**: ✅ **CLEAN** - Old module removed, using correct module at `src/physics/ontology_constraints.rs`

---

### supervisor_voice Module
```rust
src/actors/mod.rs:13:// pub mod supervisor_voice; // Removed - duplicate handlers in supervisor.rs
```

**Result**: ✅ **NO ACTIVE REFERENCES** - Successfully removed

---

### test_settings_fix Module
```rust
src/lib.rs:21:// pub mod test_settings_fix;
```

**Result**: ✅ **NO ACTIVE REFERENCES** - Successfully removed

---

### UISettings Type
```rust
src/lib.rs:30:// pub use models::ui_settings::UISettings; // Removed - consolidated into AppFullSettings"
```

**Result**: ✅ **NO ACTIVE REFERENCES** - Successfully removed

---

## Files with Technical Debt Markers

**24 files contain TODO/FIXME/HACK/XXX markers**:

1. `/home/devuser/workspace/project/src/actors/client_coordinator_actor.rs`
2. `/home/devuser/workspace/project/src/actors/metadata_actor.rs`
3. `/home/devuser/workspace/project/src/actors/multi_mcp_visualization_actor.rs`
4. `/home/devuser/workspace/project/src/actors/graph_actor.rs`
5. `/home/devuser/workspace/project/src/actors/graph_service_supervisor.rs`
6. `/home/devuser/workspace/project/src/actors/ontology_actor.rs`
7. `/home/devuser/workspace/project/src/actors/gpu/clustering_actor.rs`
8. `/home/devuser/workspace/project/src/actors/gpu/stress_majorization_actor.rs`
9. `/home/devuser/workspace/project/src/utils/unified_gpu_compute.rs`
10. `/home/devuser/workspace/project/src/utils/gpu_diagnostics.rs`
11. `/home/devuser/workspace/project/src/physics/stress_majorization.rs`
12. `/home/devuser/workspace/project/src/models/constraints.rs`
13. `/home/devuser/workspace/project/src/app_state.rs`
14. `/home/devuser/workspace/project/src/adapters/sqlite_knowledge_graph_repository.rs`
15. `/home/devuser/workspace/project/src/handlers/bots_visualization_handler.rs`
16. `/home/devuser/workspace/project/src/handlers/graph_state_handler.rs`
17. `/home/devuser/workspace/project/src/handlers/settings_handler.rs`
18. `/home/devuser/workspace/project/src/handlers/consolidated_health_handler.rs`
19. `/home/devuser/workspace/project/src/handlers/bots_handler.rs`
20. `/home/devuser/workspace/project/src/handlers/api_handler/ontology/mod.rs`
21. `/home/devuser/workspace/project/src/handlers/socket_flow_handler.rs`
22. `/home/devuser/workspace/project/src/services/owl_validator.rs`
23. `/home/devuser/workspace/project/src/services/bots_client.rs`
24. `/home/devuser/workspace/project/src/ontology/services/owl_validator.rs`

---

## Deprecated Files That Should Be Deleted

### 1. `/home/devuser/workspace/project/src/utils/hybrid_performance_optimizer.rs`

**Header**: `// DEPRECATED: Legacy hybrid performance optimizer removed`

**Status**: ❌ **DELETE ENTIRE FILE**

**Reason**: Marked as deprecated, imports deleted DockerHiveMind

---

### 2. `/home/devuser/workspace/project/src/utils/hybrid_fault_tolerance.rs`

**Header**: `// DEPRECATED: Legacy hybrid fault tolerance removed`

**Status**: ❌ **DELETE ENTIRE FILE**

**Reason**: Marked as deprecated, imports deleted DockerHiveMind, contains NetworkRecoveryManager

---

## Action Items

### Priority 1 - Critical (Remove Active Deprecated Code)

1. ✅ **Delete deprecated files**:
   - `/home/devuser/workspace/project/src/utils/hybrid_performance_optimizer.rs`
   - `/home/devuser/workspace/project/src/utils/hybrid_fault_tolerance.rs`
   - `/home/devuser/workspace/project/src/ports/gpu_physics_adapter.rs`

2. ✅ **Remove ErrorRecoveryMiddleware from main.rs**:
   - Delete struct definition (lines 61-78)
   - Delete Transform implementation (lines 79-97)
   - Delete Service implementation (lines 99-130)
   - Remove commented import statements (lines 43-45)
   - Remove registration code (line 438)

3. ✅ **Clean up supervisor.rs**:
   - Remove deprecated voice command handler (around line 388)
   - Remove DockerHiveMind references

4. ✅ **Update ports/mod.rs**:
   - Remove GpuPhysicsAdapter exports

### Priority 2 - Cleanup (Remove Documentation Markers)

1. ✅ **Remove commented module declarations**:
   - `src/lib.rs` (lines 21, 30)
   - `src/adapters/mod.rs` (lines 8, 9, 19, 22)
   - `src/config/mod.rs` (line 184)
   - `src/ontology/physics/mod.rs` (lines 3, 4)
   - `src/actors/mod.rs` (line 13)
   - `src/handlers/api_handler/mod.rs` (lines 8, 137)
   - `src/handlers/mod.rs` (line 11)

2. ✅ **Clean up DEPRECATED markers**:
   - Consider removing excessive documentation comments
   - Keep only essential migration notes

### Priority 3 - Future Work

1. ✅ **Address TODO in metadata_actor.rs**:
   - Decide whether to implement or remove `get_files_by_tag` and `get_files_by_type`

2. ✅ **Review technical debt markers**:
   - Audit 24 files with TODO/FIXME markers
   - Create issues for legitimate work items

3. ✅ **Backwards compatibility review**:
   - `binary_protocol.rs` has 10 deprecation markers for wire format v1
   - Plan migration timeline for protocol v2

---

## Summary Statistics

| Category | Count | Status |
|----------|-------|--------|
| Total Files Scanned | 244 | - |
| Critical Issues | 3 | ❌ Action Required |
| Deprecated Files to Delete | 3 | ❌ Action Required |
| Warning Markers (DEPRECATED) | 52 | ℹ️ Info Only |
| Commented Code Lines | 11 | ⚠️ Cleanup Recommended |
| TODO/FIXME Files | 24 | ℹ️ Info Only |
| Successfully Removed Modules | 8 | ✅ Clean |

---

## Conclusion

The codebase has undergone significant cleanup with most deprecated modules properly removed. However, there are **3 critical issues** that require immediate attention:

1. Delete `hybrid_performance_optimizer.rs` and `hybrid_fault_tolerance.rs`
2. Remove `ErrorRecoveryMiddleware` implementation from `main.rs`
3. Delete `gpu_physics_adapter.rs` port definition

Additionally, there are **11 lines of commented code** that should be removed for cleaner maintenance.

The majority of deprecation markers are documentation-only and serve as helpful migration notes. The codebase is in good shape overall with clear documentation of removed functionality.

---

**Next Steps**:

1. Execute Priority 1 actions (file deletions and code removal)
2. Run `cargo check` to verify no broken references
3. Run `cargo test` to ensure functionality
4. Execute Priority 2 cleanup (remove commented code)
5. Plan Priority 3 future work items
