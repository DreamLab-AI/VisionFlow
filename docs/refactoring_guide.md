---
layout: default
title: Refactoring Guide
description: Hexagonal architecture compliance refactoring examples
nav_exclude: true
---

# Refactoring Guide: Hexagonal Architecture Compliance

This guide provides concrete examples for fixing architectural violations identified in the analysis.

---

## Issue 1: Dependency Inversion Violation in Ports

### Current State (INCORRECT)

**File**: `src/ports/graph_repository.rs:11`
```rust
// ❌ VIOLATION: Port depends on concrete infrastructure (actors)
use crate::actors::graph_actor::{AutoBalanceNotification, PhysicsState};

#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_physics_state(&self) -> Result<PhysicsState>;
    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>>;
}
```

### Solution: Move Domain Types to Models Layer

**Step 1**: Create domain models

**New File**: `src/models/physics_state.rs`
```rust
use serde::{Deserialize, Serialize};

/// Physics simulation state - pure domain model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsState {
    pub kinetic_energy: f32,
    pub potential_energy: f32,
    pub total_forces: f32,
    pub settling: bool,
    pub iteration: u32,
}

impl Default for PhysicsState {
    fn default() -> Self {
        Self {
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            total_forces: 0.0,
            settling: false,
            iteration: 0,
        }
    }
}

/// Auto-balance notification - pure domain event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoBalanceNotification {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub node_id: u32,
    pub force_magnitude: f32,
    pub recommended_action: BalanceAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BalanceAction {
    IncreaseRepulsion,
    DecreaseRepulsion,
    AddConstraint,
    RemoveConstraint,
}
```

**Step 2**: Update models module

**File**: `src/models/mod.rs`
```rust
pub mod physics_state;

pub use physics_state::{PhysicsState, AutoBalanceNotification, BalanceAction};
```

**Step 3**: Fix port to use domain models

**File**: `src/ports/graph_repository.rs`
```rust
// ✅ CORRECT: Port depends on domain models only
use crate::models::physics_state::{AutoBalanceNotification, PhysicsState};

#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_physics_state(&self) -> Result<PhysicsState>;
    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>>;
}
```

**Step 4**: Update actor to use shared domain types

**File**: `src/actors/graph_actor.rs`
```rust
// Remove duplicate type definitions
// REMOVE:
// pub struct PhysicsState { ... }
// pub struct AutoBalanceNotification { ... }

// ADD:
use crate::models::physics_state::{PhysicsState, AutoBalanceNotification};

// Actor now uses domain types for message payloads
impl Handler<GetPhysicsState> for GraphActor {
    type Result = MessageResult<GetPhysicsState>;

    fn handle(&mut self, _msg: GetPhysicsState, _ctx: &mut Self::Context) -> Self::Result {
        // Convert actor internal state to domain model
        let state = PhysicsState {
            kinetic_energy: self.kinetic_energy,
            settling: self.is_settling(),
            // ...
        };
        MessageResult(Ok(state))
    }
}
```

### Verification

```bash
# Should show NO dependencies from ports to actors
grep -r "use crate::actors" src/ports/

# Should show actors importing from models
grep -r "use crate::models::physics_state" src/actors/
```

---

## Issue 2: Leaky Actor Abstractions in AppState

### Current State (INCORRECT)

**File**: `src/app_state.rs:65-114`
```rust
pub struct AppState {
    // ❌ Exposes actor implementation details
    pub graph_service_addr: Addr<GraphServiceSupervisor>,
    pub gpu_manager_addr: Option<Addr<GPUManagerActor>>,
    pub settings_addr: Addr<OptimizedSettingsActor>,
    pub metadata_addr: Addr<MetadataActor>,
    pub client_manager_addr: Addr<ClientCoordinatorActor>,
    // ... 10 more actor addresses
}
```

### Solution: Service Trait Abstraction

**Step 1**: Define service traits

**New File**: `src/services/graph_service_trait.rs`
```rust
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait GraphService: Send + Sync {
    async fn get_graph_data(&self) -> Result<Arc<GraphData>, GraphServiceError>;
    async fn add_node(&self, node: Node) -> Result<u32, GraphServiceError>;
    async fn add_edge(&self, edge: Edge) -> Result<String, GraphServiceError>;
    async fn update_node_position(&self, node_id: u32, x: f32, y: f32, z: f32) -> Result<(), GraphServiceError>;
}

#[derive(Debug, thiserror::Error)]
pub enum GraphServiceError {
    #[error("Graph not found")]
    NotFound,
    #[error("Actor communication failed: {0}")]
    ActorError(String),
    #[error("Internal error: {0}")]
    Internal(String),
}
```

**Step 2**: Implement trait with actor adapter

**New File**: `src/adapters/actor_graph_service.rs`
```rust
use crate::actors::graph_service_supervisor::GraphServiceSupervisor;
use crate::actors::messages::{GetGraphData, AddNode, UpdateNodePosition};
use crate::services::graph_service_trait::{GraphService, GraphServiceError};
use actix::Addr;
use async_trait::async_trait;

pub struct ActorGraphService {
    supervisor_addr: Addr<GraphServiceSupervisor>,
}

impl ActorGraphService {
    pub fn new(supervisor_addr: Addr<GraphServiceSupervisor>) -> Self {
        Self { supervisor_addr }
    }
}

#[async_trait]
impl GraphService for ActorGraphService {
    async fn get_graph_data(&self) -> Result<Arc<GraphData>, GraphServiceError> {
        self.supervisor_addr
            .send(GetGraphData)
            .await
            .map_err(|e| GraphServiceError::ActorError(e.to_string()))?
            .map_err(|e| GraphServiceError::Internal(e.to_string()))
    }

    async fn add_node(&self, node: Node) -> Result<u32, GraphServiceError> {
        self.supervisor_addr
            .send(AddNode { node })
            .await
            .map_err(|e| GraphServiceError::ActorError(e.to_string()))?
            .map_err(|e| GraphServiceError::Internal(e.to_string()))
    }

    // ... implement remaining methods
}
```

**Step 3**: Update AppState to use trait objects

**File**: `src/app_state.rs`
```rust
use crate::services::graph_service_trait::GraphService;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    // ✅ CORRECT: Uses trait abstraction, not concrete actors
    pub graph_service: Arc<dyn GraphService>,
    pub settings_service: Arc<dyn SettingsService>,
    pub metadata_service: Arc<dyn MetadataService>,

    // Internal: Keep actors private for backward compatibility during migration
    graph_service_addr: Addr<GraphServiceSupervisor>,  // private!
}

impl AppState {
    pub async fn new(/* ... */) -> Result<Self, AppError> {
        // Initialize actors (internal implementation detail)
        let graph_supervisor_addr = GraphServiceSupervisor::new(neo4j_adapter.clone()).start();

        // Wrap actors in service trait implementations
        let graph_service: Arc<dyn GraphService> = Arc::new(
            ActorGraphService::new(graph_supervisor_addr.clone())
        );

        Ok(Self {
            graph_service,
            graph_service_addr,  // kept for internal use only
        })
    }
}
```

**Step 4**: Update handlers to use trait

**File**: `src/handlers/api_handler/graph/mod.rs`
```rust
// BEFORE (coupled to actors)
async fn get_graph(
    state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let graph = state.graph_service_addr
        .send(GetGraphData)
        .await??;
    Ok(HttpResponse::Ok().json(graph))
}

// AFTER (uses trait abstraction)
async fn get_graph(
    state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let graph = state.graph_service
        .get_graph_data()
        .await?;
    Ok(HttpResponse::Ok().json(graph))
}
```

### Migration Strategy

**Phase 1**: Create traits and adapters (non-breaking)
- Add trait definitions
- Implement actor adapters
- Keep both old and new fields in AppState

**Phase 2**: Migrate handlers incrementally
```rust
// During migration, AppState has both:
pub struct AppState {
    pub graph_service: Arc<dyn GraphService>,        // NEW
    pub graph_service_addr: Addr<GraphServiceSupervisor>,  // OLD (deprecated)
}
```

**Phase 3**: Remove old fields after full migration
```rust
pub struct AppState {
    pub graph_service: Arc<dyn GraphService>,  // Only trait remains
    // graph_service_addr removed
}
```

---

## Issue 3: Error Handling - Replace .unwrap() with Proper Propagation

### Current State (INCORRECT)

**File**: `src/handlers/settings_handler.rs:45`
```rust
pub async fn get_settings(
    state: web::Data<AppState>,
) -> HttpResponse {
    // ❌ PANICS on actor communication failure
    let settings = state.settings_addr
        .send(GetSettings)
        .await
        .unwrap()     // Panic if actor is dead
        .unwrap();    // Panic if settings load fails

    HttpResponse::Ok().json(settings)
}
```

### Solution: Proper Error Propagation

**Step 1**: Define handler error type

**File**: `src/handlers/error.rs`
```rust
use actix_web::{error::ResponseError, HttpResponse};
use std::fmt;

#[derive(Debug)]
pub enum HandlerError {
    ActorMailbox(String),
    Service(String),
    NotFound(String),
    Validation(String),
}

impl fmt::Display for HandlerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ActorMailbox(msg) => write!(f, "Actor communication failed: {}", msg),
            Self::Service(msg) => write!(f, "Service error: {}", msg),
            Self::NotFound(msg) => write!(f, "Not found: {}", msg),
            Self::Validation(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl ResponseError for HandlerError {
    fn error_response(&self) -> HttpResponse {
        match self {
            Self::ActorMailbox(msg) => HttpResponse::ServiceUnavailable().json(json!({
                "error": "Service temporarily unavailable",
                "details": msg
            })),
            Self::Service(msg) => HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error",
                "details": msg
            })),
            Self::NotFound(msg) => HttpResponse::NotFound().json(json!({
                "error": "Resource not found",
                "details": msg
            })),
            Self::Validation(msg) => HttpResponse::BadRequest().json(json!({
                "error": "Validation failed",
                "details": msg
            })),
        }
    }
}
```

**Step 2**: Rewrite handler with error propagation

**File**: `src/handlers/settings_handler.rs`
```rust
use crate::handlers::error::HandlerError;

pub async fn get_settings(
    state: web::Data<AppState>,
) -> Result<HttpResponse, HandlerError> {
    // ✅ CORRECT: Propagates errors with context
    let settings = state.settings_addr
        .send(GetSettings)
        .await
        .map_err(|e| HandlerError::ActorMailbox(format!(
            "Failed to communicate with settings actor: {}", e
        )))?
        .map_err(|e| HandlerError::Service(format!(
            "Settings loading failed: {}", e
        )))?;

    Ok(HttpResponse::Ok().json(settings))
}
```

**Step 3**: Alternative using anyhow for simpler error handling

```rust
use anyhow::{Context, Result};

pub async fn get_settings(
    state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let settings = state.settings_addr
        .send(GetSettings)
        .await
        .context("Settings actor mailbox error")?
        .context("Settings load failed")?;

    Ok(HttpResponse::Ok().json(settings))
}
```

### Automated Refactoring Pattern

**Search Pattern**:
```bash
# Find all unwrap calls in handlers
grep -rn "\.unwrap()" src/handlers/ --include="*.rs"
```

**Replacement Template**:
```rust
// Pattern to find:
.await.unwrap()

// Replace with:
.await.map_err(|e| HandlerError::ActorMailbox(e.to_string()))?
```

**Example Script** (`scripts/fix_unwraps.sh`):
```bash
#!/bin/bash
# Find and report unwrap locations

echo "=== High-Priority Unwrap Calls ==="
echo ""

# Handlers (user-facing, must not panic)
echo "Handlers (CRITICAL - user-facing):"
grep -rn "\.unwrap()" src/handlers/ --include="*.rs" | wc -l

# Services (business logic, should not panic)
echo "Services (HIGH - business logic):"
grep -rn "\.unwrap()" src/services/ --include="*.rs" | wc -l

# Application layer (MEDIUM)
echo "Application (MEDIUM):"
grep -rn "\.unwrap()" src/application/ --include="*.rs" | wc -l

echo ""
echo "Generate detailed report? (y/n)"
read answer

if [ "$answer" = "y" ]; then
    grep -rn "\.unwrap()" src/handlers/ --include="*.rs" > unwrap_report.txt
    echo "Report saved to unwrap_report.txt"
fi
```

---

## Issue 4: God Object - Split unified_gpu_compute.rs

### Current State (INCORRECT)

**File**: `src/utils/unified_gpu_compute.rs` (3723 lines)

Contains:
1. GPU device initialization (500 lines)
2. CUDA kernel management (800 lines)
3. Async memory transfer (600 lines)
4. Buffer pool management (400 lines)
5. Physics simulation logic (400 lines)
6. Documentation (1023 lines)

### Solution: Module Decomposition

**Step 1**: Create GPU module structure

```bash
mkdir -p src/gpu/device
mkdir -p src/gpu/kernels
mkdir -p src/gpu/memory
```

**New File**: `src/gpu/device/initialization.rs` (500 lines)
```rust
//! GPU device initialization and management

use cust::prelude::*;
use std::sync::Arc;

pub struct GpuDevice {
    context: Context,
    stream: Stream,
    device_props: DeviceProperties,
}

impl GpuDevice {
    pub fn new() -> Result<Self, GpuError> {
        init_cuda()?;
        let device = Device::get_device(0)?;
        let context = Context::create_and_push(ContextFlags::MAP_HOST, device)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let device_props = device.properties()?;

        Ok(Self {
            context,
            stream,
            device_props,
        })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn properties(&self) -> &DeviceProperties {
        &self.device_props
    }
}
```

**New File**: `src/gpu/kernels/physics_kernel.rs` (400 lines)
```rust
//! CUDA physics kernels

use cust::module::Module;
use cust::stream::Stream;

pub struct PhysicsKernel {
    module: Module,
    compute_forces_kernel: Function,
    integrate_positions_kernel: Function,
}

impl PhysicsKernel {
    pub fn load(ptx_content: &str) -> Result<Self, KernelError> {
        let module = Module::load_from_string(ptx_content)?;
        let compute_forces_kernel = module.get_function("compute_forces")?;
        let integrate_positions_kernel = module.get_function("integrate_positions")?;

        Ok(Self {
            module,
            compute_forces_kernel,
            integrate_positions_kernel,
        })
    }

    pub async fn execute_physics_step(
        &self,
        stream: &Stream,
        params: &PhysicsParams,
    ) -> Result<(), KernelError> {
        // Kernel execution logic
        todo!()
    }
}
```

**New File**: `src/gpu/memory/async_transfer.rs` (600 lines)
```rust
//! Asynchronous GPU-CPU memory transfers with double buffering

pub struct AsyncTransfer<T> {
    buffer_a: DeviceBuffer<T>,
    buffer_b: DeviceBuffer<T>,
    active_buffer: BufferId,
    stream: Stream,
}

impl<T: DeviceCopy> AsyncTransfer<T> {
    pub fn new(size: usize, stream: Stream) -> Result<Self, TransferError> {
        let buffer_a = DeviceBuffer::zeroed(size)?;
        let buffer_b = DeviceBuffer::zeroed(size)?;

        Ok(Self {
            buffer_a,
            buffer_b,
            active_buffer: BufferId::A,
            stream,
        })
    }

    pub async fn download_async(&mut self) -> Result<Vec<T>, TransferError> {
        // Double-buffering logic
        todo!()
    }
}
```

**New File**: `src/gpu/unified_compute.rs` (300 lines - coordinator only)
```rust
//! High-level GPU compute coordinator

use crate::gpu::device::GpuDevice;
use crate::gpu::kernels::PhysicsKernel;
use crate::gpu::memory::AsyncTransfer;

pub struct UnifiedGPUCompute {
    device: Arc<GpuDevice>,
    physics_kernel: PhysicsKernel,
    position_transfer: AsyncTransfer<f32>,
    velocity_transfer: AsyncTransfer<f32>,
}

impl UnifiedGPUCompute {
    pub fn new(num_nodes: usize, ptx_content: &str) -> Result<Self, ComputeError> {
        let device = Arc::new(GpuDevice::new()?);
        let physics_kernel = PhysicsKernel::load(ptx_content)?;
        let position_transfer = AsyncTransfer::new(num_nodes * 3, device.stream().clone())?;
        let velocity_transfer = AsyncTransfer::new(num_nodes * 3, device.stream().clone())?;

        Ok(Self {
            device,
            physics_kernel,
            position_transfer,
            velocity_transfer,
        })
    }

    pub async fn execute_physics_step(&mut self, params: &SimulationParams) -> Result<(), ComputeError> {
        // Delegate to specialized components
        self.physics_kernel
            .execute_physics_step(self.device.stream(), params)
            .await?;
        Ok(())
    }

    pub async fn get_node_positions_async(&mut self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), ComputeError> {
        let positions = self.position_transfer.download_async().await?;
        // Split into x, y, z components
        todo!()
    }
}
```

**New File**: `src/gpu/mod.rs`
```rust
pub mod device;
pub mod kernels;
pub mod memory;
pub mod unified_compute;

pub use unified_compute::UnifiedGPUCompute;
```

### Migration Steps

1. **Keep old file**: Rename `unified_gpu_compute.rs` → `unified_gpu_compute_legacy.rs`
2. **Create new structure**: Implement modular files above
3. **Port incrementally**: Move 500 lines at a time, testing after each move
4. **Update imports**: Change `use crate::utils::unified_gpu_compute::*` → `use crate::gpu::*`
5. **Delete legacy**: Remove old file when all tests pass

---

## Verification Checklist

After each refactoring, verify:

### Compilation
```bash
cargo check --all-targets
cargo clippy -- -D warnings
```

### Tests
```bash
cargo test
cargo test --doc
```

### Architecture Compliance
```bash
# No ports depending on actors
grep -r "use crate::actors" src/ports/ && echo "VIOLATION" || echo "PASS"

# No handlers with .unwrap()
grep -r "\.unwrap()" src/handlers/ && echo "VIOLATIONS REMAIN" || echo "PASS"

# No god objects (>1000 lines)
find src -name "*.rs" -exec wc -l {} + | awk '$1 > 1000 {print $0}'
```

### Runtime Validation
```bash
cargo run --bin webxr
# Check logs for initialization errors
# Test critical endpoints: /api/graph, /api/settings
```

---

## Next Steps

1. **Start with Critical Fixes** (DIP violation, error handling)
2. **Create Feature Branch**: `git checkout -b refactor/hexagonal-compliance`
3. **Incremental Commits**: One issue per commit
4. **Run Full Test Suite**: After each change
5. **Document Changes**: Update ADRs and architecture docs

**Estimated Timeline**: 2-3 weeks for Phase 1 critical fixes
