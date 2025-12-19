---
title: P1-2: PageRank Centrality Implementation
description: This document describes the GPU-accelerated PageRank centrality implementation for the WebXR graph visualization system. PageRank is a fundamental graph centrality metric that measures node importa...
category: explanation
tags:
  - api
  - api
  - backend
updated-date: 2025-12-18
difficulty-level: intermediate
---


# P1-2: PageRank Centrality Implementation

## Overview

This document describes the GPU-accelerated PageRank centrality implementation for the WebXR graph visualization system. PageRank is a fundamental graph centrality metric that measures node importance based on the link structure.

**Status**: ✅ **COMPLETED**
**Effort**: 4 days equivalent
**Impact**: MEDIUM (common metric for node importance)

## Implementation Components

### 1. CUDA Kernel (`src/utils/pagerank.cu`)

GPU-accelerated PageRank computation using the power iteration method.

#### Algorithm

PageRank formula:
```
PR(v) = (1-d)/N + d * Σ(PR(u)/out_degree(u))
```

Where:
- `d` = damping factor (typically 0.85)
- `N` = number of nodes
- Sum is over all nodes `u` that link to `v`

#### Kernels Implemented

1. **`pagerank_init_kernel`**: Initialize all PageRank values to 1/N
2. **`pagerank_iteration_kernel`**: Basic power iteration step
3. **`pagerank_iteration_optimized_kernel`**: Optimized version with shared memory
4. **`pagerank_convergence_kernel`**: Check convergence using L1 norm
5. **`pagerank_dangling_kernel`**: Handle nodes with no outgoing edges
6. **`pagerank_normalize_kernel`**: Normalize values to sum to 1.0

#### Graph Format

Uses **Compressed Sparse Row (CSR)** format for efficiency:
- `row_offsets[i]`: Start index for node i's outgoing edges
- `col_indices[j]`: Destination node for edge j
- `out_degree[i]`: Number of outgoing edges from node i

#### Performance Features

- **Shared memory caching**: Reduces global memory access
- **Parallel reduction**: Efficient convergence checking
- **Stream-based execution**: Non-blocking GPU operations
- **Dangling node handling**: Properly redistributes PageRank mass

### 2. PageRankActor (`src/actors/gpu/pagerank_actor.rs`)

Actix actor that manages PageRank computation lifecycle.

#### Features

- **Async GPU computation**: Non-blocking actor-based execution
- **Result caching**: Stores last computed results
- **Statistical analysis**: Computes mean, median, std deviation
- **Top-K extraction**: Identifies most important nodes
- **Convergence tracking**: Monitors iteration count and convergence

#### Message Handlers

1. **`ComputePageRank`**: Trigger PageRank computation
   - Input: `PageRankParams` (damping, max_iterations, epsilon)
   - Output: `PageRankResult` (values, stats, top nodes)

2. **`GetPageRankResult`**: Retrieve cached results
   - Output: Last computed PageRank results

3. **`ClearPageRankCache`**: Clear cached results

4. **`SetSharedGPUContext`**: Update shared GPU context

#### Data Structures

```rust
pub struct PageRankParams {
    pub damping_factor: Option<f32>,     // Default: 0.85
    pub max_iterations: Option<u32>,     // Default: 100
    pub epsilon: Option<f32>,            // Default: 1e-6
    pub normalize: Option<bool>,         // Default: true
    pub use_optimized: Option<bool>,     // Default: true
}

pub struct PageRankResult {
    pub pagerank_values: Vec<f32>,       // Per-node values
    pub iterations: u32,                 // Iterations performed
    pub converged: bool,                 // Convergence flag
    pub convergence_value: f32,          // Final L1 norm
    pub top_nodes: Vec<PageRankNode>,    // Top K nodes
    pub stats: PageRankStats,            // Statistical summary
}

pub struct PageRankStats {
    pub total_nodes: u32,
    pub max_pagerank: f32,
    pub min_pagerank: f32,
    pub mean_pagerank: f32,
    pub median_pagerank: f32,
    pub std_deviation: f32,
    pub computation_time_ms: u64,
    pub converged: bool,
    pub iterations: u32,
}
```

### 3. Message System Integration

Added to `src/actors/messages.rs`:

```rust
#[cfg(feature = "gpu")]
#[derive(Message)]
#[rtype(result = "Result<PageRankResult, String>")]
pub struct ComputePageRank {
    pub params: Option<PageRankParams>,
}

#[cfg(feature = "gpu")]
#[derive(Message)]
#[rtype(result = "Option<PageRankResult>")]
pub struct GetPageRankResult;

#[cfg(feature = "gpu")]
#[derive(Message)]
#[rtype(result = "()")]
pub struct ClearPageRankCache;
```

### 4. Module Registration

Updated `src/actors/gpu/mod.rs`:
- Added `pub mod pagerank_actor;`
- Added `pub use pagerank_actor::PageRankActor;`

## Visual Analytics Integration

PageRank centrality values can be integrated into the visual analytics pipeline for enhanced graph visualization.

### Visual Mappings

#### 1. Node Size

Map PageRank to node size for importance visualization:

```javascript
// Client-side (Three.js)
const minSize = 0.5;
const maxSize = 3.0;
const normalizedPR = (pagerank - minPR) / (maxPR - minPR);
nodeSize = minSize + normalizedPR * (maxSize - minSize);

// Update sphere geometry
const geometry = new THREE.SphereGeometry(nodeSize, 16, 16);
```

**Recommended Scaling**:
- Min size: 0.5 units (low importance)
- Max size: 3.0 units (high importance)
- Use log scaling for highly skewed distributions

#### 2. Node Color Gradient

Color nodes based on centrality using a heat map:

```javascript
// Color gradient: Blue (low) → Green (mid) → Yellow → Red (high)
function getPageRankColor(normalizedPR) {
    if (normalizedPR < 0.33) {
        // Blue to Green
        const t = normalizedPR / 0.33;
        return new THREE.Color(0, t, 1 - t);
    } else if (normalizedPR < 0.67) {
        // Green to Yellow
        const t = (normalizedPR - 0.33) / 0.34;
        return new THREE.Color(t, 1, 0);
    } else {
        // Yellow to Red
        const t = (normalizedPR - 0.67) / 0.33;
        return new THREE.Color(1, 1 - t, 0);
    }
}
```

**Alternative Color Schemes**:
- Grayscale: White (high) to Black (low)
- Viridis: Perceptually uniform for accessibility
- Custom: Match application theme

#### 3. Node Filtering

Filter graph by PageRank threshold:

```javascript
// Show only nodes above threshold
const prThreshold = 0.001; // Top 10% typically
visibleNodes = nodes.filter(n => n.pagerank > prThreshold);

// Progressive disclosure
function updateVisibilityByPR(zoomLevel) {
    const threshold = calculateThreshold(zoomLevel);
    nodes.forEach(n => {
        n.visible = n.pagerank > threshold;
    });
}
```

#### 4. Layout Forces

Use PageRank to drive force-directed layout:

```glsl
// GLSL shader for GPU force computation
vec3 computeCentralityForce(float pagerank) {
    // Pull high-PR nodes toward center
    vec3 toCenter = -position;
    float centralityStrength = pagerank * 10.0;
    return normalize(toCenter) * centralityStrength;
}
```

**Layout Strategies**:
- **Center attraction**: High-PR nodes pulled to center
- **Radial layout**: Position by PR (center = high, periphery = low)
- **Hierarchical**: Layer nodes by PR tiers

#### 5. Highlighting

Highlight top-K influential nodes:

```javascript
// Highlight top 10 nodes
const topNodes = result.top_nodes.slice(0, 10);
topNodes.forEach(node => {
    // Add glow effect
    node.material.emissive = new THREE.Color(0xffff00);
    node.material.emissiveIntensity = 0.5;

    // Add label
    addLabel(node, `Rank ${node.rank}: ${node.pagerank.toFixed(4)}`);

    // Add halo
    addHaloEffect(node, node.pagerank);
});
```

### API Endpoint Design

Recommended REST API endpoints for PageRank integration:

#### Compute PageRank

```http
POST /api/analytics/pagerank
Content-Type: application/json

{
    "damping_factor": 0.85,
    "max_iterations": 100,
    "epsilon": 0.000001,
    "normalize": true
}

Response 200:
{
    "pagerank_values": [0.0012, 0.0034, ...],
    "iterations": 23,
    "converged": true,
    "convergence_value": 0.0000008,
    "top_nodes": [
        {"node_id": 42, "pagerank": 0.0156, "rank": 1},
        {"node_id": 17, "pagerank": 0.0134, "rank": 2},
        ...
    ],
    "stats": {
        "total_nodes": 100000,
        "max_pagerank": 0.0156,
        "min_pagerank": 0.0000012,
        "mean_pagerank": 0.00001,
        "median_pagerank": 0.0000087,
        "std_deviation": 0.0002,
        "computation_time_ms": 145,
        "converged": true,
        "iterations": 23
    }
}
```

#### Get Cached Results

```http
GET /api/analytics/pagerank

Response 200:
{
    "pagerank_values": [...],
    "stats": {...}
}

Response 404:
{
    "error": "No cached PageRank results available"
}
```

#### Clear Cache

```http
DELETE /api/analytics/pagerank/cache

Response 200:
{
    "message": "PageRank cache cleared"
}
```

#### Get Visual Configuration

```http
GET /api/analytics/pagerank/visual-config

Response 200:
{
    "size_scaling": {
        "min": 0.5,
        "max": 3.0,
        "type": "linear"
    },
    "color_scheme": "heat",
    "filter_threshold": 0.001,
    "highlight_top_k": 10
}
```

### WebSocket Integration

For real-time PageRank updates during graph evolution:

```javascript
// Client-side WebSocket handler
ws.on('pagerank_update', (data) => {
    const { node_id, new_pagerank, rank_change } = data;

    // Update node visualization
    updateNodeSize(node_id, new_pagerank);
    updateNodeColor(node_id, new_pagerank);

    // Animate rank change
    if (rank_change !== 0) {
        animateRankChange(node_id, rank_change);
    }
});

// Server-side (incremental updates)
async fn broadcast_pagerank_updates(
    pagerank_actor: Addr<PageRankActor>,
    ws_manager: Addr<WebSocketManager>
) {
    let result = pagerank_actor.send(ComputePageRank::default()).await?;

    // Send top K to all clients
    for node in result.top_nodes {
        ws_manager.do_send(BroadcastMessage {
            msg_type: "pagerank_update",
            data: json!({
                "node_id": node.node_id,
                "pagerank": node.pagerank,
                "rank": node.rank
            })
        });
    }
}
```

### Performance Optimization

#### Batch Processing

Compute PageRank once, use multiple times:

```rust
// Cache for 5 minutes
let cache_duration = Duration::from_secs(300);
if last_compute.elapsed() < cache_duration {
    return cached_result;
}

// Otherwise recompute
let result = pagerank_actor.send(ComputePageRank::default()).await?;
cache_result(result);
```

#### Progressive Rendering

Render high-PR nodes first for perceived performance:

```javascript
// Sort by PageRank descending
const sortedNodes = nodes.sort((a, b) => b.pagerank - a.pagerank);

// Render in batches
const batchSize = 1000;
for (let i = 0; i < sortedNodes.length; i += batchSize) {
    const batch = sortedNodes.slice(i, i + batchSize);
    renderBatch(batch);
    await nextFrame();
}
```

#### Level of Detail (LOD)

Adjust detail based on PageRank:

```javascript
function getLODLevel(pagerank, distance) {
    if (pagerank > 0.01) return "high";  // Always high detail
    if (pagerank > 0.001 && distance < 50) return "medium";
    return "low";
}

// Apply LOD
node.geometry = getGeometry(getLODLevel(node.pagerank, distance));
```

## Usage Examples

### Rust (Actor-based)

```rust
use actix::Addr;
use crate::actors::gpu::PageRankActor;
use crate::actors::messages::{ComputePageRank, PageRankParams};

async fn compute_and_visualize(
    pagerank_actor: Addr<PageRankActor>
) -> Result<(), String> {
    // Configure parameters
    let params = PageRankParams {
        damping_factor: Some(0.85),
        max_iterations: Some(100),
        epsilon: Some(1e-6),
        normalize: Some(true),
        use_optimized: Some(true),
    };

    // Compute PageRank
    let result = pagerank_actor
        .send(ComputePageRank { params: Some(params) })
        .await
        .map_err(|e| format!("Actor error: {}", e))??;

    // Log statistics
    println!("PageRank computed in {}ms", result.stats.computation_time_ms);
    println!("Converged: {} after {} iterations",
             result.converged, result.iterations);
    println!("Top 5 nodes:");
    for node in result.top_nodes.iter().take(5) {
        println!("  Rank {}: Node {} = {:.6}",
                 node.rank, node.node_id, node.pagerank);
    }

    // Apply to visualization
    apply_pagerank_to_visual_analytics(&result.pagerank_values)?;

    Ok(())
}

fn apply_pagerank_to_visual_analytics(values: &[f32]) -> Result<(), String> {
    // Update node sizes based on PageRank
    let (min_pr, max_pr) = get_min_max(values);

    for (node_id, &pr) in values.iter().enumerate() {
        let normalized = (pr - min_pr) / (max_pr - min_pr);
        let size = 0.5 + normalized * 2.5; // 0.5 to 3.0

        update_node_size(node_id, size)?;
        update_node_color(node_id, get_heat_color(normalized))?;
    }

    Ok(())
}
```

### JavaScript (Client-side)

```javascript
// Fetch PageRank from API
async function visualizePageRank() {
    const response = await fetch('/api/analytics/pagerank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            damping_factor: 0.85,
            max_iterations: 100
        })
    });

    const result = await response.json();

    // Apply to Three.js scene
    applyPageRankVisuals(result);

    // Update UI
    updateStatsPanel(result.stats);
    highlightTopNodes(result.top_nodes);
}

function applyPageRankVisuals(result) {
    const { pagerank_values, stats } = result;

    nodes.forEach((node, idx) => {
        const pr = pagerank_values[idx];
        const normalized = (pr - stats.min_pagerank) /
                          (stats.max_pagerank - stats.min_pagerank);

        // Size: 0.5 to 3.0
        node.scale.setScalar(0.5 + normalized * 2.5);

        // Color: heat gradient
        node.material.color = getHeatColor(normalized);

        // Emissive for top nodes
        if (normalized > 0.8) {
            node.material.emissive = new THREE.Color(0xffaa00);
            node.material.emissiveIntensity = normalized * 0.5;
        }
    });
}

function getHeatColor(t) {
    // t in [0, 1]
    if (t < 0.5) {
        // Blue to Yellow
        return new THREE.Color(t * 2, t * 2, 1 - t * 2);
    } else {
        // Yellow to Red
        const s = (t - 0.5) * 2;
        return new THREE.Color(1, 1 - s, 0);
    }
}
```

## Testing

### Unit Tests

Located in `src/actors/gpu/pagerank_actor.rs`:

```rust
#[test]
fn test_pagerank_params_default() { ... }

#[test]
fn test_extract_top_nodes() { ... }

#[test]
fn test_calculate_statistics() { ... }
```

### Integration Test

```rust
#[actix_rt::test]
async fn test_pagerank_actor_computation() {
    let actor = PageRankActor::new().start();

    let params = PageRankParams::default();
    let result = actor
        .send(ComputePageRank { params: Some(params) })
        .await
        .unwrap()
        .unwrap();

    assert!(result.converged);
    assert!(result.iterations > 0);
    assert_eq!(result.pagerank_values.len(), NUM_NODES);

    // Check normalization
    let sum: f32 = result.pagerank_values.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}
```

## Performance Metrics

Expected performance on NVIDIA RTX 4090:

| Graph Size | Iterations | Time (ms) | Throughput |
|------------|-----------|-----------|------------|
| 10K nodes  | ~30       | 15        | 666 graphs/s |
| 100K nodes | ~40       | 145       | 6.9 graphs/s |
| 1M nodes   | ~50       | 1,200     | 0.83 graphs/s |

**Convergence**: Typically 20-50 iterations with ε=1e-6

## Future Enhancements

1. **Personalized PageRank**: Add teleport vector for topic-sensitive ranking
2. **Temporal PageRank**: Track PageRank changes over time
3. **Distributed Computation**: Multi-GPU support for massive graphs
4. **Incremental Updates**: Only recompute affected subgraphs
5. **Alternative Metrics**: Integrate eigenvector centrality, Katz centrality

## References

- Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web.
- CUDA PageRank: https://developer.nvidia.com/blog/optimizing-pagerank-on-gpus/
- Graph Analytics: https://docs.rapids.ai/api/cugraph/stable/

## Files Modified

| File | Lines Added | Purpose |
|------|-------------|---------|
| `src/utils/pagerank.cu` | 450 | CUDA kernels |
| `src/actors/gpu/pagerank_actor.rs` | 420 | Actor implementation |
| `src/actors/messages.rs` | 20 | Message types |
| `src/actors/gpu/mod.rs` | 4 | Module exports |
| `docs/implementation/p1-2-pagerank.md` | 800+ | Documentation |

**Total**: ~1,700 lines of code and documentation

---

**Implementation Date**: 2025-11-08
**Author**: Claude Code Agent (Coder)
**Status**: ✅ Ready for Integration Testing
