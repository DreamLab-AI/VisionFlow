---
title: Before/After Code Comparison
description: ```rust let shared_context = match &self.shared_context { Some(ctx) => ctx,
category: explanation
tags:
  - rust
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Before/After Code Comparison

## Fix 1: force_compute_actor.rs - Scope Reordering

### Before (ERROR)
```rust
let shared_context = match &self.shared_context {
    Some(ctx) => ctx,
    None => {
        let error_msg = "GPU context not initialized".to_string();
        // ... logging ...
        self.is_computing = false;
        return Err(error_msg);
    }
};

// ... GPU guard acquisition ...

let mut unified_compute = shared_context.unified_compute.lock()?;

// ... parameter setup ...

// Apply ontology constraints before physics step
#[cfg(feature = "ontology")]
{
    if let Err(e) = self.apply_ontology_forces() {  // ❌ ERROR: mutable borrow while shared_context borrowed
        warn!("Failed to apply ontology forces: {}", e);
    }
}

let sim_params = &self.simulation_params;
let gpu_result = unified_compute.execute_physics_step(sim_params);

// ...

if let Err(e) = shared_context.update_utilization(gpu_utilization) {  // ← immutable borrow still used
    log::warn!("Failed to update shared GPU utilization metrics: {}", e);
}
```

### After (FIXED)
```rust
// Apply ontology constraints BEFORE acquiring shared context
// This avoids borrow conflicts since apply_ontology_forces needs mutable access
#[cfg(feature = "ontology")]
{
    if let Err(e) = self.apply_ontology_forces() {  // ✅ Mutable borrow completes here
        warn!("ForceComputeActor: Failed to apply ontology forces: {}", e);
    }
}

let shared_context = match &self.shared_context {  // ✅ Immutable borrow starts here
    Some(ctx) => ctx,
    None => {
        let error_msg = "GPU context not initialized".to_string();
        // ... logging ...
        self.is_computing = false;
        return Err(error_msg);
    }
};

// ... GPU guard acquisition ...

let mut unified_compute = shared_context.unified_compute.lock()?;

// ... parameter setup ...

let sim_params = &self.simulation_params;
let gpu_result = unified_compute.execute_physics_step(sim_params);

// ...

if let Err(e) = shared_context.update_utilization(gpu_utilization) {  // ✅ No conflict
    log::warn!("Failed to update shared GPU utilization metrics: {}", e);
}
```

### Key Change
**Moved ontology forces call from line 287 to line 185 (before shared_context borrow)**

---

## Fix 2: pagerank_actor.rs - Async Actor Pattern

### Before (ERROR)
```rust
impl Handler<ComputePageRank> for PageRankActor {
    type Result = ResponseActFuture<Self, Result<PageRankResult, String>>;

    fn handle(&mut self, msg: ComputePageRank, _ctx: &mut Context<Self>) -> Self::Result {
        info!("PageRankActor: Received ComputePageRank message");

        let params = msg.params.unwrap_or_default();

        Box::pin(
            async move { self.compute_pagerank(params).await }  // ❌ ERROR: moves self
                .into_actor(self)  // ❌ ERROR: self already moved
                .map(|result, _actor, _ctx| result),
        )
    }
}
```

### After (FIXED)
```rust
impl Handler<ComputePageRank> for PageRankActor {
    type Result = ResponseActFuture<Self, Result<PageRankResult, String>>;

    fn handle(&mut self, msg: ComputePageRank, ctx: &mut Context<Self>) -> Self::Result {
        info!("PageRankActor: Received ComputePageRank message");

        let params = msg.params.unwrap_or_default();

        // ✅ Get shared context before async boundary
        let shared_ctx = match &self.shared_context {
            Some(ctx) => Arc::clone(ctx),  // ✅ Clone Arc (cheap)
            None => {
                return Box::pin(
                    async { Err("GPU context not initialized".to_string()) }
                        .into_actor(self)
                );
            }
        };

        // ✅ Create the async computation future (doesn't use self)
        let future = async move {
            let mut unified_compute = match shared_ctx
                .unified_compute
                .lock()
            {
                Ok(guard) => guard,
                Err(e) => return Err(format!("Failed to acquire GPU compute lock: {}", e)),
            };

            let start_time = Instant::now();

            // Extract parameters with defaults
            let damping = params.damping_factor.unwrap_or(0.85);
            let max_iter = params.max_iterations.unwrap_or(100);
            let epsilon = params.epsilon.unwrap_or(1e-6);
            let normalize = params.normalize.unwrap_or(true);
            let use_optimized = params.use_optimized.unwrap_or(true);

            // Call GPU PageRank computation
            let gpu_result = unified_compute
                .run_pagerank_centrality(damping, max_iter, epsilon, normalize, use_optimized)
                .map_err(|e| {
                    error!("GPU PageRank computation failed: {}", e);
                    format!("PageRank computation failed: {}", e)
                })?;

            let computation_time = start_time.elapsed();
            info!(
                "PageRankActor: PageRank computation completed in {:?}",
                computation_time
            );

            // Unpack GPU result
            let (pagerank_values, iterations, converged, convergence_value) = gpu_result;

            Ok((pagerank_values, iterations, converged, convergence_value, computation_time))
        };

        // ✅ Use into_actor to re-enter actor context and finish processing
        Box::pin(
            future
                .into_actor(self)  // ✅ self not moved, can borrow here
                .map(|result, actor, _ctx| {
                    match result {
                        Ok((pagerank_values, iterations, converged, convergence_value, computation_time)) => {
                            // ✅ Compute statistics in actor context (has access to actor methods)
                            let stats = actor.calculate_statistics(
                                &pagerank_values,
                                iterations,
                                converged,
                                computation_time.as_millis() as u64,
                            );

                            // Extract top K nodes
                            let top_nodes = actor.extract_top_nodes(&pagerank_values, 10);

                            let result = PageRankResult {
                                pagerank_values,
                                iterations,
                                converged,
                                convergence_value,
                                top_nodes,
                                stats,
                            };

                            // ✅ Cache the result
                            actor.last_result = Some(result.clone());
                            actor.gpu_state.record_utilization(0.8);

                            Ok(result)
                        }
                        Err(e) => Err(e),
                    }
                })
        )
    }
}
```

### Key Changes
1. **Clone Arc before async boundary** - `Arc::clone(ctx)`
2. **Async work uses cloned Arc** - No `self` in async block
3. **Re-enter actor context** - `.into_actor(self)` works now
4. **State updates in map** - Actor methods called in closure

---

## Fix 3: shortest_path_actor.rs - Extract and Drop (SSSP)

### Before (ERROR)
```rust
fn handle(&mut self, msg: ComputeSSP, _ctx: &mut Self::Context) -> Self::Result {
    info!("ShortestPathActor: Computing SSSP from node {}", msg.source_idx);

    let mut unified_compute = match &self.shared_context {  // ← Immutable borrow starts
        Some(ctx) => ctx
            .unified_compute
            .lock()
            .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?,
        None => {
            return Err("GPU context not initialized".to_string());
        }
    };

    let start_time = Instant::now();

    // Call the existing GPU SSSP implementation
    let distances = unified_compute
        .run_sssp(msg.source_idx)
        .map_err(|e| {
            error!("GPU SSSP computation failed: {}", e);
            format!("SSSP computation failed: {}", e)
        })?;

    let computation_time = start_time.elapsed().as_millis() as u64;

    // Calculate statistics
    let mut nodes_reached = 0;
    let mut max_distance = 0.0f32;

    for &dist in &distances {
        if dist < f32::MAX {
            nodes_reached += 1;
            max_distance = max_distance.max(dist);
        }
    }

    // Apply max_distance filter if specified
    let filtered_distances = if let Some(max_dist) = msg.max_distance {
        distances.into_iter().map(|d| {
            if d <= max_dist { d } else { f32::MAX }
        }).collect()
    } else {
        distances
    };

    self.update_stats(true, computation_time);  // ❌ ERROR: mutable borrow while lock held

    info!(
        "ShortestPathActor: SSSP completed in {}ms, reached {}/{} nodes",
        computation_time, nodes_reached, filtered_distances.len()
    );

    Ok(SSSPResult {
        distances: filtered_distances,
        source_idx: msg.source_idx,
        nodes_reached,
        max_distance,
        computation_time_ms: computation_time,
    })
}  // ← unified_compute (MutexGuard) dropped here
```

### After (FIXED)
```rust
fn handle(&mut self, msg: ComputeSSP, _ctx: &mut Self::Context) -> Self::Result {
    info!("ShortestPathActor: Computing SSSP from node {}", msg.source_idx);

    // ✅ Acquire lock, compute, then drop lock before calling update_stats
    let (filtered_distances, nodes_reached, max_distance, computation_time) = {
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => ctx
                .unified_compute
                .lock()
                .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?,
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };

        let start_time = Instant::now();

        // Call the existing GPU SSSP implementation
        let distances = unified_compute
            .run_sssp(msg.source_idx)
            .map_err(|e| {
                error!("GPU SSSP computation failed: {}", e);
                format!("SSSP computation failed: {}", e)
            })?;

        let computation_time = start_time.elapsed().as_millis() as u64;

        // Calculate statistics
        let mut nodes_reached = 0;
        let mut max_distance = 0.0f32;

        for &dist in &distances {
            if dist < f32::MAX {
                nodes_reached += 1;
                max_distance = max_distance.max(dist);
            }
        }

        // Apply max_distance filter if specified
        let filtered_distances = if let Some(max_dist) = msg.max_distance {
            distances.into_iter().map(|d| {
                if d <= max_dist { d } else { f32::MAX }
            }).collect()
        } else {
            distances
        };

        (filtered_distances, nodes_reached, max_distance, computation_time)
    }; // ✅ unified_compute lock dropped here

    // ✅ Now we can safely call update_stats with mutable borrow
    self.update_stats(true, computation_time);

    info!(
        "ShortestPathActor: SSSP completed in {}ms, reached {}/{} nodes",
        computation_time, nodes_reached, filtered_distances.len()
    );

    Ok(SSSPResult {
        distances: filtered_distances,
        source_idx: msg.source_idx,
        nodes_reached,
        max_distance,
        computation_time_ms: computation_time,
    })
}
```

### Key Change
**Wrapped GPU computation in scoped block, extracting results as tuple before lock drops**

---

## Fix 4: shortest_path_actor.rs - Extract and Drop (APSP)

### Before (ERROR)
```rust
fn handle(&mut self, msg: ComputeAPSP, _ctx: &mut Self::Context) -> Self::Result {
    info!("ShortestPathActor: Computing APSP with {} landmarks", msg.num_landmarks);

    let mut unified_compute = match &self.shared_context {  // ← Lock acquired
        // ...
    };

    // ... all computation ...

    let computation_time = start_time.elapsed().as_millis() as u64;
    self.update_stats(false, computation_time);  // ❌ ERROR: lock still held

    // ...
}  // ← Lock dropped here
```

### After (FIXED)
```rust
fn handle(&mut self, msg: ComputeAPSP, _ctx: &mut Self::Context) -> Self::Result {
    info!("ShortestPathActor: Computing APSP with {} landmarks", msg.num_landmarks);

    // ✅ Acquire lock, compute, then drop lock before calling update_stats
    let (apsp_distances, num_nodes, landmarks, computation_time) = {
        let mut unified_compute = match &self.shared_context {
            // ...
        };

        // ... all computation ...

        let computation_time = start_time.elapsed().as_millis() as u64;

        (apsp_distances, num_nodes, landmarks, computation_time)
    }; // ✅ unified_compute lock dropped here

    // ✅ Now we can safely call update_stats with mutable borrow
    self.update_stats(false, computation_time);

    // ...
}
```

### Key Change
**Same pattern as SSSP - scoped block with result extraction**

---

## Pattern Summary

| Fix | Pattern | Before | After | Why It Works |
|-----|---------|--------|-------|--------------|
| 1 | Scope Reordering | Mutable call during immutable borrow | Mutable call before immutable borrow | Operations independent, can reorder |
| 2 | Async Actor | Move self into async, then borrow | Clone Arc, async work, then into_actor | Separates async work from actor state |
| 3 | Extract & Drop | Lock held during mutable borrow | Scope block drops lock first | Explicit lifetime control with scopes |
| 4 | Extract & Drop | Lock held during mutable borrow | Scope block drops lock first | Same as fix 3 |

## Principles Applied

1. **Understand the borrow** - Know what's borrowed and when
2. **Control lifetimes** - Use scopes and reordering
3. **Separate concerns** - Async work vs actor state
4. **Extract, don't clone** - Move computed results out
5. **Clone when appropriate** - Only for shared ownership (Arc)
