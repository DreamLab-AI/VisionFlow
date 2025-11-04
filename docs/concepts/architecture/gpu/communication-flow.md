# GPU Actor Communication Flow - Corrected Data Flow

> ⚠️ **DEPRECATION NOTICE** ⚠️
> **GraphServiceActor** is being replaced by the **hexagonal CQRS architecture**.
> This document describes legacy patterns and is being updated. See `/docs/guides/graphserviceactor-migration.md` for current recommendations.

## Overview

This document provides a comprehensive overview of the corrected GPU actor communication flow, showing how **GraphServiceActor** properly communicates with **GPUManagerActor** as supervisor, which then delegates to specialized GPU actors including **GPUResourceActor** and **ForceComputeActor**.

**Status**: 🔄 IN TRANSITION (Nov 2025) - Use TransitionalGraphSupervisor for now

## Key Corrections

### Previous (Incorrect) Flow
- **GraphServiceActor** → **ForceComputeActor** (direct communication)
- GPU initialization handled by ForceComputeActor
- No supervision pattern

### Current (Corrected) Flow

**Legacy Pattern** ❌ **(DEPRECATED)**

1. **AppState** → **GPUManagerActor** creation
2. **AppState** → **GraphServiceActor** (InitializeGPUConnection with GPUManagerActor address)
3. **GraphServiceActor** → **GPUManagerActor** (InitializeGPU, UpdateGPUGraphData)
4. **GPUManagerActor** → **GPUResourceActor** (delegation for GPU operations)
5. **GPUResourceActor** → GPU hardware (CUDA initialization, data upload)
6. **GPUManagerActor** → **ForceComputeActor** (physics simulation steps only)

**Current Pattern** ✅ **(PRODUCTION)**

1. **AppState** → **GPUManagerActor** creation
2. **AppState** → **TransitionalGraphSupervisor** (replaces GraphServiceActor)
3. **TransitionalGraphSupervisor** → **GPUManagerActor** (InitializeGPU, UpdateGPUGraphData)
4. **GPUManagerActor** → **GPUResourceActor** (delegation for GPU operations)
5. **GPUResourceActor** → GPU hardware (CUDA initialization, data upload)
6. **GPUManagerActor** → **ForceComputeActor** (physics simulation steps only)

See `/docs/guides/graphserviceactor-migration.md` for migration details.

## Actor Responsibilities

### GraphServiceActor

**Legacy Pattern** ❌ **(DEPRECATED)**
- **Role**: Graph state management and client coordination
- **GPU Interaction**: Stores GPUManagerActor address, sends high-level requests
- **Key Messages**:
  - `InitializeGPUConnection` (receives GPUManagerActor address)
  - `InitializeGPU` (requests GPU initialization)
  - `UpdateGPUGraphData` (sends graph data for GPU processing)

**Current Pattern** ✅ **(PRODUCTION)**
- **Actor**: TransitionalGraphSupervisor (replaces GraphServiceActor)
- **Role**: CQRS command/query separation, hexagonal architecture
- **GPU Interaction**: Uses command handlers to communicate with GPUManagerActor
- **Key Messages**: Same interface, different implementation pattern

See `/docs/guides/graphserviceactor-migration.md` for migration details.

### GPUManagerActor
- **Role**: Supervisor and message router for GPU operations
- **Responsibilities**:
  - Supervise specialized GPU actors
  - Route messages to appropriate GPU components
  - Coordinate GPU resource allocation
  - Handle GPU initialization requests from GraphServiceActor *(legacy)* or TransitionalGraphSupervisor *(current)*
- **Key Messages**:
  - `InitializeGPU` (from GraphServiceActor *(deprecated)* or TransitionalGraphSupervisor *(current)*)
  - `UpdateGPUGraphData` (from GraphServiceActor *(deprecated)* or TransitionalGraphSupervisor *(current)*)
  - Delegates to GPUResourceActor and ForceComputeActor

### GPUResourceActor
- **Role**: CUDA device and memory management
- **Responsibilities**:
  - GPU device initialization
  - CUDA memory allocation and deallocation
  - Data upload/download to/from GPU
  - GPU buffer management
- **Key Operations**:
  - CUDA context creation
  - Memory buffer allocation
  - Data transfer operations
  - Resource cleanup

### ForceComputeActor
- **Role**: Physics simulation execution (after GPU initialization)
- **Responsibilities**:
  - Execute physics simulation steps
  - Force calculations
  - Position and velocity updates
  - Physics parameter application
- **Dependencies**: Requires GPU to be initialized by GPUResourceActor first

## Complete Communication Sequence

**Legacy Pattern** ❌ **(DEPRECATED)** - The following sequence diagram shows GraphServiceActor which is deprecated.

**Current Pattern** ✅ **(PRODUCTION)** - Replace GraphServiceActor with TransitionalGraphSupervisor in production code.

See `/docs/guides/graphserviceactor-migration.md` for migration details.

```mermaid
sequenceDiagram
    participant AppState as AppState
    participant GraphService as GraphServiceActor (DEPRECATED)
    participant GPUManager as GPUManagerActor
    participant GPUResource as GPUResourceActor
    participant ForceCompute as ForceComputeActor
    participant GPU as GPU Hardware

    Note over AppState: System Startup
    AppState->>GPUManager: Create GPUManagerActor
    AppState->>GraphService: Create GraphServiceActor (DEPRECATED)
    AppState->>GraphService: InitializeGPUConnection(GPUManagerActor address)

    Note over GraphService, GPU: GPU Initialization Sequence
    GraphService->>GPUManager: InitializeGPU message
    GPUManager->>GPUResource: Initialize CUDA device
    GPUResource->>GPU: CUDA context creation
    GPU-->>GPUResource: Context created
    GPUResource->>GPU: Memory allocation
    GPU-->>GPUResource: Memory allocated
    GPUResource-->>GPUManager: GPU initialization complete
    GPUManager-->>GraphService: GPU ready for operations

    Note over GraphService, GPU: Graph Data Processing
    GraphService->>GPUManager: UpdateGPUGraphData(nodes, edges)
    GPUManager->>GPUResource: Upload graph data to GPU
    GPUResource->>GPU: Data transfer (nodes, edges, positions)
    GPU-->>GPUResource: Data uploaded successfully
    GPUResource-->>GPUManager: Graph data ready on GPU

    Note over GraphService, GPU: Physics Simulation Loop
    loop Simulation Step
        GraphService->>GPUManager: RequestPhysicsStep
        GPUManager->>ForceCompute: Execute physics simulation
        ForceCompute->>GPU: Force calculations kernel
        GPU-->>ForceCompute: Calculated forces
        ForceCompute->>GPU: Position integration kernel
        GPU-->>ForceCompute: Updated positions
        ForceCompute-->>GPUManager: Physics step complete
        GPUManager->>GPUResource: Download updated positions
        GPUResource->>GPU: Memory transfer (positions)
        GPU-->>GPUResource: Position data
        GPUResource-->>GPUManager: Position data available
        GPUManager-->>GraphService: Updated node positions
        GraphService->>GraphService: Update internal state
        GraphService->>GraphService: Broadcast to clients
    end

    Note over GraphService, GPU: Cleanup
    GraphService->>GPUManager: Shutdown GPU operations
    GPUManager->>GPUResource: Cleanup GPU resources
    GPUResource->>GPU: Free memory, destroy context
    GPU-->>GPUResource: Cleanup complete
    GPUResource-->>GPUManager: Resources freed
    GPUManager-->>GraphService: GPU shutdown complete
```

## Message Types and Data Flow

**Legacy Pattern** ❌ **(DEPRECATED)** - These message flows reference GraphServiceActor.

### Initialization Messages
1. **InitializeGPUConnection**
   - **Source**: AppState → GraphServiceActor *(deprecated)* or TransitionalGraphSupervisor *(current)*
   - **Payload**: GPUManagerActor address
   - **Purpose**: Establishes communication channel

2. **InitializeGPU**
   - **Source**: GraphServiceActor *(deprecated)* or TransitionalGraphSupervisor *(current)* → GPUManagerActor
   - **Payload**: Initialization parameters
   - **Delegation**: GPUManagerActor → GPUResourceActor

### Data Processing Messages
3. **UpdateGPUGraphData**
   - **Source**: GraphServiceActor *(deprecated)* or TransitionalGraphSupervisor *(current)* → GPUManagerActor
   - **Payload**: Nodes, edges, metadata
   - **Delegation**: GPUManagerActor → GPUResourceActor (data upload)

### Physics Simulation Messages
4. **RequestPhysicsStep**
   - **Source**: GraphServiceActor *(deprecated)* or TransitionalGraphSupervisor *(current)* → GPUManagerActor
   - **Delegation**: GPUManagerActor → ForceComputeActor
   - **Return Path**: ForceComputeActor → GPUManagerActor → GraphServiceActor *(deprecated)* or TransitionalGraphSupervisor *(current)*

**Current Pattern** ✅ **(PRODUCTION)**
- All messages now flow through CQRS command handlers in TransitionalGraphSupervisor
- Message interfaces remain the same, implementation uses hexagonal architecture

See `/docs/guides/graphserviceactor-migration.md` for migration details.

## Error Handling and Recovery

### GPU Initialization Failures

**Legacy Pattern** ❌ **(DEPRECATED)** - These error handling flows reference GraphServiceActor.

```mermaid
sequenceDiagram
    participant GraphService as GraphServiceActor (DEPRECATED)
    participant GPUManager as GPUManagerActor
    participant GPUResource as GPUResourceActor

    GraphService->>GPUManager: InitializeGPU
    GPUManager->>GPUResource: Initialize CUDA device
    GPUResource-->>GPUManager: Error: CUDA not available
    GPUManager->>GPUManager: Set fallback mode (CPU)
    GPUManager-->>GraphService: GPU unavailable, using CPU fallback
    GraphService->>GraphService: Update simulation mode
```

### Runtime GPU Errors

**Legacy Pattern** ❌ **(DEPRECATED)** - Replace GraphServiceActor with TransitionalGraphSupervisor.

```mermaid
sequenceDiagram
    participant GraphService as GraphServiceActor (DEPRECATED)
    participant GPUManager as GPUManagerActor
    participant ForceCompute as ForceComputeActor

    GraphService->>GPUManager: RequestPhysicsStep
    GPUManager->>ForceCompute: Execute physics simulation
    ForceCompute-->>GPUManager: Error: GPU kernel failure
    GPUManager->>GPUManager: Reset GPU state
    GPUManager->>ForceCompute: Retry physics step
    ForceCompute-->>GPUManager: Physics step complete
    GPUManager-->>GraphService: Updated positions (after recovery)
```

**Note**: Error handling patterns remain the same, but TransitionalGraphSupervisor now handles these flows using CQRS command handlers.

See `/docs/guides/graphserviceactor-migration.md` for migration details.

## Performance Optimizations

### Batched Operations
- **GPU Data Uploads**: GPUResourceActor batches multiple data transfers
- **Physics Steps**: ForceComputeActor processes multiple simulation steps per GPU call
- **Memory Management**: GPUResourceActor reuses GPU buffers when possible

### Resource Sharing
- **GPU Memory Pools**: GPUResourceActor maintains memory pools for different data types
- **Kernel Reuse**: ForceComputeActor caches compiled GPU kernels
- **Buffer Recycling**: Automatic buffer recycling to minimize allocation overhead

## Migration Benefits

### Before (Incorrect Architecture)
- Direct coupling between GraphServiceActor and ForceComputeActor
- No proper supervision of GPU resources
- Mixed responsibilities (physics + GPU management)
- Difficult to handle GPU initialization failures
- Limited scalability for additional GPU operations

### After (Corrected Architecture) - Phase 1 ✅ **COMPLETED**
- Clear separation of concerns
- Proper supervision pattern with GPUManagerActor
- Dedicated GPU resource management (GPUResourceActor)
- Specialized physics simulation (ForceComputeActor)
- Better error handling and recovery
- Scalable architecture for additional GPU actors

### Current (CQRS Architecture) - Phase 2 🔄 **IN PROGRESS**
- **TransitionalGraphSupervisor** replaces GraphServiceActor
- Hexagonal architecture with ports and adapters
- CQRS command/query separation
- Domain-driven design patterns
- Enhanced testability and maintainability
- Prepared for event sourcing and distributed systems

See `/docs/guides/graphserviceactor-migration.md` for migration details.

## Implementation Status

### Phase 1: GPU Supervision Pattern
✅ **Completed**: Message routing through GPUManagerActor
✅ **Completed**: GPUResourceActor for CUDA device management
✅ **Completed**: ForceComputeActor specialization for physics
✅ **Completed**: Error handling and fallback mechanisms
✅ **Completed**: Documentation updates reflecting corrected flow

### Phase 2: CQRS Architecture Migration 🔄 **IN PROGRESS (Nov 2025)**
🔄 **In Progress**: TransitionalGraphSupervisor replaces GraphServiceActor
🔄 **In Progress**: Hexagonal architecture implementation
🔄 **In Progress**: Command/Query separation
⏳ **Pending**: Full GraphServiceActor deprecation
⏳ **Pending**: Event sourcing implementation

**Status**: 🔄 IN TRANSITION (Nov 2025) - Use TransitionalGraphSupervisor for now

See `/docs/guides/graphserviceactor-migration.md` for migration details.

## Future Enhancements

### Additional GPU Actors
- **ClusteringActor**: GPU-accelerated graph clustering
- **LayoutActor**: Advanced layout algorithms
- **AnalyticsActor**: Real-time graph analytics
- **RenderingActor**: GPU-based visualization preprocessing

### Advanced Features
- **Multi-GPU Support**: Distribute operations across multiple GPUs
- **Dynamic Load Balancing**: Automatically balance GPU workload
- **Predictive Resource Management**: Anticipate GPU memory needs
- **Real-time Performance Metrics**: Monitor GPU utilization and optimise

This corrected architecture provides a solid foundation for scalable GPU operations while maintaining clear separation of concerns and robust error handling.