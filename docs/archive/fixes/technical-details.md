---
title: Borrow Checker Fixes - Technical Details
description: ``` error[E0502]: cannot borrow `*self` as mutable because it is also borrowed as immutable --> src/actors/gpu/force_compute_actor.rs:287:29
category: explanation
tags:
  - rust
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Borrow Checker Fixes - Technical Details

## Error Analysis and Solutions

### Error 1: force_compute_actor.rs:287 - E0502

#### Original Error
```
error[E0502]: cannot borrow `*self` as mutable because it is also borrowed as immutable
   --> src/actors/gpu/force_compute_actor.rs:287:29
    |
186 |         let shared_context = match &self.shared_context {
    |                                    -------------------- immutable borrow occurs here
...
287 |             if let Err(e) = self.apply_ontology_forces() {
    |                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here
...
314 |                 if let Err(e) = shared_context.update_utilization(gpu_utilization) {
    |                                 -------------- immutable borrow later used here
```

#### Root Cause
The code had this execution order:
1. Line 186: Create immutable borrow `&self.shared_context`
2. Line 287: Try to call `self.apply_ontology_forces()` (needs `&mut self`)
3. Line 314: Use the immutable borrow from step 1

Rust's borrow checker prevents having both immutable and mutable borrows alive simultaneously.

#### Solution
**Reorder operations** - Move the mutable borrow before the immutable borrow:

```rust
// NEW ORDER:
// 1. Apply ontology forces (mutable borrow, completes immediately)
#[cfg(feature = "ontology")]
{
    if let Err(e) = self.apply_ontology_forces() {
        warn!("ForceComputeActor: Failed to apply ontology forces: {}", e);
    }
}

// 2. THEN acquire shared context (immutable borrow)
let shared_context = match &self.shared_context {
    Some(ctx) => ctx,
    None => { return Err("GPU context not initialized".to_string()); }
};

// 3. Use shared context
// ...
shared_context.update_utilization(gpu_utilization);
```

#### Why This Works
- `apply_ontology_forces()` internally creates its own borrow of `shared_context`
- That internal borrow completes before returning
- No overlapping borrows between the two uses of `shared_context`

#### No Clone Needed
The operations are independent - we just needed to execute them in the right order.

---

### Error 2: pagerank_actor.rs:335 - E0382

#### Original Error
```
error[E0382]: borrow of moved value: `self`
   --> src/actors/gpu/pagerank_actor.rs:335:29
    |
328 |     fn handle(&mut self, msg: ComputePageRank, _ctx: &mut Context<Self>) -> Self::Result {
    |               --------- move occurs because `self` has type `&mut PageRankActor`
...
334 |             async move { self.compute_pagerank(params).await }
    |             ----------   ---- variable moved due to use in coroutine
    |             |
    |             value moved here
335 |                 .into_actor(self)
    |                             ^^^^ value borrowed here after move
```

#### Root Cause
Classic async/actor pattern issue:
- `async move { }` captures `self` by move (transfers ownership into closure)
- `.into_actor(self)` tries to use `self` after it was moved

#### Original Code
```rust
fn handle(&mut self, msg: ComputePageRank, _ctx: &mut Context<Self>) -> Self::Result {
    let params = msg.params.unwrap_or_default();

    Box::pin(
        async move { self.compute_pagerank(params).await }  // ERROR: moves self
            .into_actor(self)  // ERROR: self already moved
            .map(|result, _actor, _ctx| result),
    )
}
```

#### Solution
Split the computation into two phases:

**Phase 1: Async Computation** (outside actor context)
```rust
// Clone Arc before async boundary
let shared_ctx = match &self.shared_context {
    Some(ctx) => Arc::clone(ctx),
    None => {
        return Box::pin(
            async { Err("GPU context not initialized".to_string()) }
                .into_actor(self)
        );
    }
};

// Async computation with cloned Arc
let future = async move {
    let mut unified_compute = shared_ctx.unified_compute.lock()?;

    // Perform GPU computation
    let result = unified_compute.run_pagerank_centrality(...)?;

    Ok((pagerank_values, iterations, converged, ...))
};
```

**Phase 2: State Updates** (back in actor context)
```rust
Box::pin(
    future
        .into_actor(self)
        .map(|result, actor, _ctx| {
            match result {
                Ok((values, iter, conv, ...)) => {
                    // Compute stats using actor's methods
                    let stats = actor.calculate_statistics(...);
                    let top_nodes = actor.extract_top_nodes(...);

                    // Update actor state
                    actor.last_result = Some(result);
                    actor.gpu_state.record_utilization(0.8);

                    Ok(result)
                }
                Err(e) => Err(e),
            }
        })
)
```

#### Why Arc Clone IS Appropriate
- **Arc is designed for this** - Shared ownership across async boundaries
- **Cheap operation** - Only increments reference count
- **Semantically correct** - GPU context is shared across multiple actors
- **No data duplication** - Arc wraps the same underlying data

#### Pattern Benefits
- Clean separation of async computation and actor state
- No borrowing conflicts
- Proper use of Actix actor model
- Maintains single-threaded actor guarantee

---

### Error 3 & 4: shortest_path_actor.rs:240 & 334 - E0502

#### Original Error
```
error[E0502]: cannot borrow `*self` as mutable because it is also borrowed as immutable
   --> src/actors/gpu/shortest_path_actor.rs:240:9
    |
198 |         let mut unified_compute = match &self.shared_context {
    |                                         -------------------- immutable borrow occurs here
...
240 |         self.update_stats(true, computation_time);
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here
...
254 |     }
    |     - immutable borrow might be used here, when `unified_compute` is dropped
```

#### Root Cause
- `unified_compute` is a `MutexGuard` from `self.shared_context.unified_compute.lock()`
- The `MutexGuard` holds an immutable reference to `self.shared_context`
- When we call `self.update_stats()`, we need a mutable borrow of `self`
- Can't have both the immutable borrow (MutexGuard) and mutable borrow simultaneously
- Guard is not dropped until end of function

#### Original Code (ComputeSSP handler)
```rust
fn handle(&mut self, msg: ComputeSSP, _ctx: &mut Self::Context) -> Self::Result {
    let mut unified_compute = match &self.shared_context {
        Some(ctx) => ctx.unified_compute.lock()?,  // Immutable borrow starts
        None => return Err("..."),
    };

    // ... use unified_compute ...

    self.update_stats(true, computation_time);  // ERROR: need mutable borrow

    // ... unified_compute dropped here
}
```

#### Solution
Use a **scoped block** to control the lifetime of the MutexGuard:

```rust
fn handle(&mut self, msg: ComputeSSP, _ctx: &mut Self::Context) -> Self::Result {
    // Scope the lock to extract what we need
    let (filtered_distances, nodes_reached, max_distance, computation_time) = {
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => ctx.unified_compute.lock()?,
            None => return Err("..."),
        };

        // Perform all GPU computation
        let distances = unified_compute.run_sssp(msg.source_idx)?;

        // Calculate statistics
        let mut nodes_reached = 0;
        let mut max_distance = 0.0f32;
        for &dist in &distances {
            if dist < f32::MAX {
                nodes_reached += 1;
                max_distance = max_distance.max(dist);
            }
        }

        // Filter distances
        let filtered_distances = if let Some(max_dist) = msg.max_distance {
            distances.into_iter().map(|d| {
                if d <= max_dist { d } else { f32::MAX }
            }).collect()
        } else {
            distances
        };

        (filtered_distances, nodes_reached, max_distance, computation_time)
    };  // ← MutexGuard dropped here!

    // Now safe to call methods requiring mutable borrow
    self.update_stats(true, computation_time);

    Ok(SSSPResult {
        distances: filtered_distances,
        source_idx: msg.source_idx,
        nodes_reached,
        max_distance,
        computation_time_ms: computation_time,
    })
}
```

#### Same Fix for ComputeAPSP Handler
Applied identical pattern:
- Scoped block for GPU computation
- Extract results as tuple
- Drop lock at end of scope
- Call `update_stats()` after lock is released

#### Why No Clone Needed
- We extracted the **computed results** (Vec<f32>, usize, f32, u64)
- These are already **owned values** returned by the GPU computation
- No need to clone - we moved them out of the scope
- Only extracted what we needed for the return value

---

## Verification Commands

### Check for borrow checker errors
```bash
# Count E0502 errors
cargo build 2>&1 | grep -c "error\[E0502\]"

# Count E0382 errors
cargo build 2>&1 | grep -c "error\[E0382\]"

# Show all borrow checker errors
cargo build 2>&1 | grep -E "error\[E0502\]|error\[E0382\]"
```

### Build with all features
```bash
cargo build --features gpu,ontology
```

## Performance Impact

### No Performance Penalty
- **force_compute_actor**: Reordering has zero overhead
- **shortest_path_actor**: Extracting results has zero overhead (values were already computed)

### Minimal Overhead
- **pagerank_actor**: Arc::clone only increments refcount (atomic operation)
  - Cost: ~1-2 CPU cycles
  - Benefit: Proper async handling, prevents data races

## Summary Table

| File | Error | Line | Pattern Used | Clone Used? | Reason |
|------|-------|------|--------------|-------------|---------|
| force_compute_actor.rs | E0502 | 287 | Scope Reordering | No | Operations independent |
| pagerank_actor.rs | E0382 | 335 | Async Actor | Yes (Arc) | Shared ownership semantics |
| shortest_path_actor.rs | E0502 | 240 | Extract & Drop | No | Results already owned |
| shortest_path_actor.rs | E0502 | 334 | Extract & Drop | No | Results already owned |

## Code Review Checklist

✅ All borrow checker errors resolved
✅ No unnecessary clones added
✅ Proper Rust idioms used
✅ Performance maintained
✅ Safety guarantees preserved
✅ Code more maintainable
✅ Comments added explaining fixes

## References

- [Rust Book - References and Borrowing](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)
- [Actix Actor Model](https://actix.rs/docs/actix/actor/)
- [Arc Documentation](https://doc.rust-lang.org/std/sync/struct.Arc.html)
- [MutexGuard and RAII](https://doc.rust-lang.org/std/sync/struct.MutexGuard.html)
