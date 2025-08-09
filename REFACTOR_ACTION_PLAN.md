# Refactor Action Plan

## Overview
This document outlines the remaining work to complete the actor-based architecture refactor and replace mock data with live integrations.

## Priority 1: Complete Core Refactors

### 1.1 Settings System Consolidation
**Status:** Partially complete - two competing models exist

**Current State:**
- Legacy: `AppFullSettings` in `src/config/mod.rs` (complex, multi-layer)
- New: `Settings` in `src/config/settings.rs` (clean, single source of truth)
- Conversion logic exists in multiple handlers

**Actions Required:**
```rust
// 1. Migrate all consumers to new Settings model
// 2. Remove AppFullSettings completely
// 3. Update SettingsActor to only use new model
// 4. Remove all conversion functions
```

**Files to Update:**
- [ ] `src/config/mod.rs` - Remove AppFullSettings
- [ ] `src/handlers/settings_handler.rs` - Remove conversion logic
- [ ] `src/handlers/api_handler/visualisation/mod.rs` - Use new Settings directly
- [ ] `src/actors/settings_actor.rs` - Remove AppFullSettings references

### 1.2 Remove Legacy Service Layer
**Status:** Old handlers exist alongside new actor-based handlers

**Files to Remove:**
- [ ] `src/handlers/graph_handler.rs` (replaced by GraphServiceActor)
- [ ] `src/handlers/file_handler.rs` (replaced by api_handler/files)
- [ ] `src/handlers/visualization_handler.rs` (replaced by api_handler/visualisation)
- [ ] `src/services/graph_service.rs` (replaced by graph_actor.rs)

**Migration Checklist:**
- [x] Routes migrated to api_handler module
- [x] Actor-based message passing implemented
- [ ] Remove Arc<RwLock<>> patterns
- [ ] Clean up main.rs imports

## Priority 2: Live Backend Integration

### 2.1 Connect ClaudeFlowActor to MCP Backend
**Status:** Currently using mock data generation

**Implementation Path:**
```rust
// In claude_flow_actor_enhanced.rs
async fn execute_mcp_tool(&mut self, tool: McpTool, params: Value) -> Result<Value> {
    // TODO: Replace mock with actual MCP client calls
    match self.mcp_client {
        Some(ref client) => {
            // Real implementation:
            client.execute(tool, params).await
        }
        None => {
            // Current mock fallback
            self.generate_mock_response(tool, params)
        }
    }
}
```

**Tasks:**
- [ ] Implement MCP client connection
- [ ] Replace `generate_enhanced_mock_agents()` with live data
- [ ] Replace `generate_mock_message_flow()` with real message stream
- [ ] Implement real swarm initialization

### 2.2 System Metrics Collection
**Status:** Placeholder values (0.0) for CPU, memory, GPU

**Implementation:**
```rust
// Add system monitoring crate
// Cargo.toml: sysinfo = "0.30"

use sysinfo::{System, SystemExt, CpuExt};

fn update_system_metrics(&mut self) {
    let mut sys = System::new_all();
    sys.refresh_all();
    
    self.system_metrics = SystemMetrics {
        cpu_usage: sys.global_cpu_info().cpu_usage(),
        memory_usage: sys.used_memory() as f32 / sys.total_memory() as f32,
        gpu_usage: self.get_gpu_metrics(), // Via nvidia-ml
        // ...
    };
}
```

### 2.3 Consolidate Agent Control
**Status:** Three competing implementations

**Current Implementations:**
1. `ClaudeFlowActor` (most advanced, mock data)
2. `AgentControlClient` (direct, lower-level)
3. `BotsClient` (disabled, wrong protocol)

**Action:** Keep only ClaudeFlowActor
- [ ] Remove `AgentControlClient` and related handler
- [ ] Remove `BotsClient` completely
- [ ] Update all consumers to use ClaudeFlowActor

## Priority 3: GPU Features

### 3.1 Stress Majorization Implementation
**Status:** Placeholder function

**Implementation Outline:**
```rust
fn perform_stress_majorization(&mut self) -> Result<(), Error> {
    // 1. Calculate current stress tensor
    let stress = self.calculate_stress_tensor()?;
    
    // 2. Compute gradient descent direction
    let gradient = self.compute_stress_gradient(stress)?;
    
    // 3. Apply majorization step
    self.apply_majorization_update(gradient)?;
    
    // 4. Update node positions on GPU
    self.upload_positions_to_gpu()?;
    
    Ok(())
}
```

### 3.2 Visual Analytics Integration
**Status:** Falls back to legacy kernel

**Required Implementation:**
```rust
fn compute_forces_with_visual_analytics(&mut self) -> Result<(), Error> {
    let va_gpu = self.visual_analytics_gpu.as_mut()
        .ok_or_else(|| Error::new(ErrorKind::Other, "VA GPU not initialized"))?;
    
    // 1. Convert node data to TSNode format
    let ts_nodes = self.convert_to_ts_nodes()?;
    let ts_edges = self.convert_to_ts_edges()?;
    
    // 2. Stream to GPU
    va_gpu.stream_nodes(&ts_nodes)?;
    va_gpu.stream_edges(&ts_edges)?;
    
    // 3. Execute visual analytics pipeline
    va_gpu.execute(&self.visual_analytics_params.unwrap(), 
                   ts_nodes.len(), 
                   ts_edges.len(), 
                   self.isolation_layers.len())?;
    
    // 4. Copy results back
    self.copy_visual_analytics_results(va_gpu)?;
    
    Ok(())
}
```

## Priority 4: Complete Actor Features

### 4.1 Metadata Actor Queries
**Status:** Methods commented out due to struct changes

**Fix Required:**
```rust
// Update to work with new Metadata structure
pub fn get_files_by_tag(&self, tag: &str) -> Vec<String> {
    self.metadata_store.iter()
        .filter(|(_, meta)| meta.tags.contains(&tag.to_string()))
        .map(|(id, _)| id.clone())
        .collect()
}

pub fn get_files_by_type(&self, file_type: &str) -> Vec<String> {
    self.metadata_store.iter()
        .filter(|(_, meta)| meta.file_type == Some(file_type.to_string()))
        .map(|(id, _)| id.clone())
        .collect()
}
```

### 4.2 Metadata Refresh
**Status:** TODO placeholder

**Implementation:**
```rust
pub fn refresh_metadata(&mut self) -> Result<(), String> {
    // 1. Scan file system for changes
    let current_files = self.scan_directories()?;
    
    // 2. Compare with cached metadata
    let changes = self.detect_changes(current_files)?;
    
    // 3. Update metadata for changed files
    for change in changes {
        match change {
            FileChange::Added(path) => self.add_file_metadata(path)?,
            FileChange::Modified(path) => self.update_file_metadata(path)?,
            FileChange::Deleted(path) => self.remove_file_metadata(path)?,
        }
    }
    
    // 4. Notify subscribers of changes
    self.broadcast_metadata_changes()?;
    
    Ok(())
}
```

## Implementation Timeline

### Week 1: Core Refactors
- Day 1-2: Settings consolidation
- Day 3-4: Remove legacy handlers/services
- Day 5: Testing and validation

### Week 2: Backend Integration
- Day 1-2: MCP client connection
- Day 3: System metrics implementation
- Day 4: Agent control consolidation
- Day 5: Integration testing

### Week 3: GPU Features
- Day 1-2: Stress majorization
- Day 3-4: Visual analytics pipeline
- Day 5: Performance optimization

### Week 4: Polish
- Day 1-2: Metadata actor completion
- Day 3: Health diagnostics
- Day 4-5: Documentation and testing

## Testing Strategy

### Unit Tests
- [ ] Settings migration tests
- [ ] Actor message handling tests
- [ ] GPU kernel tests

### Integration Tests
- [ ] MCP backend connection
- [ ] End-to-end data flow
- [ ] WebSocket binary protocol

### Performance Tests
- [ ] GPU compute benchmarks
- [ ] Actor system throughput
- [ ] Memory usage monitoring

## Risk Mitigation

### Backwards Compatibility
- Keep conversion functions temporarily
- Implement feature flags for new code paths
- Maintain parallel implementations during transition

### Data Migration
- Create migration scripts for settings
- Backup existing configurations
- Provide rollback mechanism

### Performance Degradation
- Profile before/after each major change
- Maintain benchmark suite
- Monitor production metrics

## Success Criteria

1. **Single Settings Model**: All code uses new `Settings` structure
2. **No Mock Data**: All agent data comes from live MCP backend
3. **Actor-Only Architecture**: No `Arc<RwLock<>>` patterns remain
4. **GPU Features Complete**: Stress majorization and visual analytics working
5. **All TODOs Resolved**: No placeholder implementations remain

## Notes

- The actor-based architecture is well-designed and mostly complete
- Mock data strategy was good for parallel development
- GPU integration is sophisticated but needs completion
- Overall system architecture is sound and scalable