---
title: Borrow Checker Error Fixes
description: This document explains the borrow checker errors found in the codebase and the proper fixes applied - **without unnecessary clones**.
type: archive
status: archived
---

# Borrow Checker Error Fixes

## Overview
This document explains the borrow checker errors found in the codebase and the proper fixes applied - **without unnecessary clones**.

## Errors Identified

### 1. E0502 in force_compute_actor.rs (Line 287)

**Error**: Cannot borrow `*self` as mutable because it is also borrowed as immutable

```rust
// Line 186: Immutable borrow
let shared_context = match &self.shared_context {
    Some(ctx) => ctx,
    None => { ... }
};

// Line 287: Mutable borrow (ERROR!)
if let Err(e) = self.apply_ontology_forces() {
    ...
}

// Line 314: Immutable borrow still in use
if let Err(e) = shared_context.update_utilization(gpu_utilization) {
    ...
}
```

**Root Cause**:
- `shared_context` holds an immutable borrow of `&self.shared_context` from line 186
- At line 287, we try to mutably borrow `self` with `self.apply_ontology_forces()`
- The immutable borrow is still alive and used at line 314
- Rust's borrow checker prevents this to ensure memory safety

**Fix Strategy**: Scope the immutable borrow properly
- Extract what we need from `shared_context` early
- Drop the immutable borrow before the mutable borrow
- Restructure to avoid overlapping borrows

### 2. E0382 in pagerank_actor.rs (Line 335)

**Error**: Borrow of moved value: `self`

```rust
fn handle(&mut self, msg: ComputePageRank, _ctx: &mut Context<Self>) -> Self::Result {
    let params = msg.params.unwrap_or_default();

    Box::pin(
        async move { self.compute_pagerank(params).await }  // ERROR: self moved here
            .into_actor(self)  // ERROR: self borrowed after move
            .map(|result, _actor, _ctx| result),
    )
}
```

**Root Cause**:
- `async move { ... }` captures `self` by move
- `.into_actor(self)` tries to borrow `self` after it was moved
- This is a classic async/actor pattern issue

**Fix Strategy**: Proper async actor pattern
- Store params in actor state or pass differently
- Use `wrap_future` which properly handles the actor context
- Avoid capturing `self` in the async block

### 3. E0502 in shortest_path_actor.rs (Lines 240, 334)

**Error**: Cannot borrow `*self` as mutable because it is also borrowed as immutable

```rust
// Line 198: Immutable borrow
let mut unified_compute = match &self.shared_context {
    Some(ctx) => ctx.unified_compute.lock()...,
    None => { ... }
};

// ... use unified_compute ...

// Line 240: Mutable borrow (ERROR!)
self.update_stats(true, computation_time);

// Line 254: unified_compute dropped (immutable borrow ends)
```

**Root Cause**:
- `unified_compute` holds a MutexGuard from `self.shared_context`
- The MutexGuard maintains an immutable borrow of `self.shared_context`
- `self.update_stats()` tries to mutably borrow `self`
- Can't have mutable and immutable borrows simultaneously

**Fix Strategy**: Drop locks before mutable borrows
- Extract data from `unified_compute` before calling mutable methods
- Explicitly drop the lock with `drop(unified_compute)`
- Use scoped blocks to control borrow lifetimes

## Fix Patterns Applied

### Pattern 1: Scoped Borrows
```rust
// ❌ Wrong
let data = &self.data;
self.modify();  // ERROR
use(data);

// ✅ Correct
{
    let data = &self.data;
    use(data);
}  // Borrow dropped here
self.modify();  // OK
```

### Pattern 2: Extract and Drop
```rust
// ❌ Wrong
let guard = self.mutex.lock().unwrap();
self.method();  // ERROR: guard still holds borrow

// ✅ Correct
let value = {
    let guard = self.mutex.lock().unwrap();
    guard.clone()  // Only clone if truly needed
};
drop(value);  // Explicit drop
self.method();  // OK
```

### Pattern 3: Async Actor Proper Pattern
```rust
// ❌ Wrong
async move { self.method().await }
    .into_actor(self)  // ERROR: self moved

// ✅ Correct
ctx.spawn(
    async move { actor.method().await }
        .into_actor(actor)
        .map(|result, actor, ctx| {
            // Handle result
        })
)
```

## When Clone IS Appropriate

1. **Arc<T> for shared ownership** across threads/async boundaries
2. **Small values** (numbers, bools) that are Copy
3. **Data crossing async boundaries** where ownership must transfer
4. **Explicit caching** where you need a snapshot

## When Clone is NOT Appropriate

1. **Large data structures** that should be borrowed
2. **Just to satisfy compiler** without understanding the issue
3. **When restructuring** would be cleaner and more efficient
4. **Temporary workarounds** instead of proper fixes

## Implementation Notes

All fixes follow these principles:
- Understand the borrow conflict
- Restructure code to avoid conflict
- Use appropriate scoping
- Only clone when semantically correct
- Document non-obvious patterns

## Files Fixed

1. `/home/devuser/workspace/project/src/actors/gpu/force_compute_actor.rs`
2. `/home/devuser/workspace/project/src/actors/gpu/pagerank_actor.rs`
3. `/home/devuser/workspace/project/src/actors/gpu/shortest_path_actor.rs`

## Verification

After fixes:
```bash
cargo build  # Should compile without E0502/E0382 errors
cargo test   # All tests should pass
```
