---
title: Rust Type Correction Guide
description: The Rust compiler's type errors are **design feedback**, not obstacles to work around. When you see a type error, it's telling you that your mental model doesn't match the actual structure of the c...
category: explanation
tags:
  - guide
  - rust
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Rust Type Correction Guide

## Philosophy: Fix Root Causes, Not Symptoms

The Rust compiler's type errors are **design feedback**, not obstacles to work around. When you see a type error, it's telling you that your mental model doesn't match the actual structure of the code.

## The Wrong Way ❌

```rust
// ❌ Adding conversions everywhere
let value = wrong_type.into();
let value = wrong_type as RightType;
let value = wrong_type.try_into().unwrap();

// ❌ Creating stub implementations
impl MessageResponse for SomeType {
    fn handle(self, ctx: &mut Context) {
        unimplemented!()  // "I'll fix this later" (you won't)
    }
}

// ❌ Ignoring errors
check_something().ok();  // Silently dropping Result
```

## The Right Way ✅

### 1. Read the Error Completely

```
error[E0308]: mismatched types
  --> src/actors/gpu/pagerank_actor.rs:168:47
   |
168|     .run_pagerank_centrality(damping, max_iter, epsilon, ...)
   |      -----------------------          ^^^^^^^^ expected `usize`, found `u32`
   |      |
   |      arguments to this function are incorrect
```

**Extract**:
- **What**: `max_iter` is wrong type
- **Expected**: `usize`
- **Found**: `u32`
- **Where**: Function call at line 168

### 2. Understand the WHY

Ask yourself:
- Why does the function expect `usize`?
- Why is my code providing `u32`?
- Which type makes more sense for the domain?

**Example**: PageRank iteration count
- **API uses**: `u32` (common for external interfaces)
- **Internal uses**: `usize` (for array indexing)
- **Solution**: Convert at the boundary

### 3. Fix at the Source

#### Pattern: Domain vs Implementation Types

**Problem**: GPU kernels expect numeric IDs but domain uses string types

```rust
// ❌ WRONG: Try to cast strings
node.node_type as i32  // Won't compile!

// ✅ RIGHT: Create explicit mapping
let mut type_to_id: HashMap<String, i32> = HashMap::new();
let mut next_id = 0;

for node in &nodes {
    if let Some(ref type_str) = node.node_type {
        type_to_id.entry(type_str.clone()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
    }
}

// Now convert using the mapping
let type_id = node.node_type.as_ref()
    .and_then(|t| type_to_id.get(t))
    .copied()
    .unwrap_or(0);
```

**Key Principle**: Make the conversion explicit and centralized

#### Pattern: Type Conversion at Boundaries

```rust
// ✅ Convert at entry point
fn handle(&mut self, msg: MyMessage) {
    let internal_value = msg.external_value as usize;
    self.process(internal_value);  // Uses usize internally
}

// ✅ Convert at exit point
fn get_result(&self) -> ExternalType {
    let internal_result = self.compute();
    internal_result as ExternalType
}

// ❌ WRONG: Convert in the middle
fn process(&mut self, value: u32) {
    // ... some code ...
    let usize_value = value as usize;  // BAD: conversion buried in logic
    // ... more code ...
}
```

#### Pattern: Return Complete Information

**Problem**: Function returns partial data, requiring multiple round-trips

```rust
// ❌ WRONG: Return only values
pub fn run_pagerank(...) -> Result<Vec<f32>> {
    // ... computation ...
    Ok(scores)
}

// Caller needs iterations and convergence info but can't get it!
```

**Fix**: Return everything callers might need

```rust
// ✅ RIGHT: Return complete result
pub fn run_pagerank(...) -> Result<(Vec<f32>, usize, bool, f32)> {
    let mut iterations = max_iterations;
    let mut converged = false;
    let mut delta = 0.0;

    for i in 0..max_iterations {
        // ... computation ...
        if delta < epsilon {
            iterations = i + 1;
            converged = true;
            break;
        }
    }

    Ok((scores, iterations, converged, delta))
}
```

#### Pattern: Message Types Must Match Handler Types

```rust
// ✅ RIGHT: Types match exactly
#[derive(Message)]
#[rtype(result = "Result<SemanticConfig, String>")]
pub struct GetSemanticConfig;

impl Handler<GetSemanticConfig> for MyActor {
    type Result = Result<SemanticConfig, String>;  // Matches rtype

    fn handle(&mut self, _msg: GetSemanticConfig, _ctx: &mut Context) -> Self::Result {
        Ok(self.config.clone())
    }
}

// ❌ WRONG: Types don't match
#[rtype(result = "SemanticConfig")]           // Says this
type Result = MessageResult<SemanticConfig>;  // Does this
```

### 4. Verify the Fix

After making changes:

```bash
# 1. Does it compile?
cargo build

# 2. Does it still work?
cargo test

# 3. Does it make sense?
# Read your code - is the type conversion explicit and justified?
```

## Common Error Patterns and Solutions

### E0308: Mismatched Types

**Symptom**: Type A provided where type B expected

**Fix Process**:
1. Identify which type is correct for the domain
2. Change the source to produce correct type
3. OR convert at the boundary if both types are valid

### E0277: Trait Bound Not Satisfied

**Symptom**: Type doesn't implement required trait

**Fix Process**:
1. Check if type SHOULD implement the trait
   - YES → Add `#[derive(...)]` or `impl` block
   - NO → Change the function signature or use a different type

**Example**:
```rust
// Error: SemanticConfig doesn't implement Message
#[rtype(result = "SemanticConfig")]

// Fix: Use Result which does implement Message
#[rtype(result = "Result<SemanticConfig, String>")]
```

### E0605: Invalid Cast

**Symptom**: Trying to cast between incompatible types

```rust
// ❌ Can't cast Option<String> to i32
node.node_type as i32

// ✅ Create mapping layer
type_to_id.get(&node.node_type).copied().unwrap_or(0)
```

### E0596: Cannot Borrow as Mutable

**Symptom**: Variable not declared mutable

```rust
// ❌ WRONG
let unified_compute = ctx.get_compute();
unified_compute.run_pagerank(...);  // Error: needs mut

// ✅ RIGHT
let mut unified_compute = ctx.get_compute();
unified_compute.run_pagerank(...);
```

### E0428: Duplicate Definitions

**Symptom**: Same name defined twice

**Fix Process**:
1. Check if they're truly duplicates or serve different purposes
2. If duplicate: remove one
3. If different: rename to clarify intent

```rust
// Found: Two get_num_nodes() functions
// Line 879: returns self.num_nodes (cached count)
// Line 3534: returns self.pos_in_x.len() (actual buffer size)

// Solution: Keep line 3534 (more accurate), remove line 879
```

## Case Study: PageRank Type Fixes

### Problem

```rust
// GPU function signature
fn run_pagerank(..., max_iterations: usize) -> Result<Vec<f32>>

// Actor code
let max_iter = params.max_iterations.unwrap_or(100);  // u32
let result = gpu.run_pagerank(..., max_iter, ...);    // Error: u32 vs usize
let (values, iterations, ...) = result;               // Error: Vec<f32> not a tuple
```

### Solution

**Step 1**: Fix the return type
```rust
// Change GPU function to return complete information
fn run_pagerank(...) -> Result<(Vec<f32>, usize, bool, f32)> {
    let mut iterations = max_iterations;
    let mut converged = false;
    let mut delta = 0.0;

    // ... computation tracking convergence ...

    Ok((scores, iterations, converged, delta))
}
```

**Step 2**: Convert types at boundary
```rust
// Actor code - convert u32 to usize at entry
let max_iter = params.max_iterations.unwrap_or(100) as usize;

// Call GPU function
let result = gpu.run_pagerank(..., max_iter, ...)?;

// Extract tuple
let (values, iterations, converged, delta) = result;

// Convert usize back to u32 for external API
let iterations = iterations as u32;
```

**Why This is Right**:
- ✅ GPU function returns all computed information
- ✅ Type conversion happens once at each boundary
- ✅ Internal code uses appropriate types (usize for indexing)
- ✅ External API maintains its contract (u32)
- ✅ No hidden conversions in the middle of logic

## Checklist for Type Corrections

- [ ] Read the complete error message
- [ ] Understand WHAT type is expected
- [ ] Understand WHY that type is expected
- [ ] Identify the ROOT CAUSE of the mismatch
- [ ] Fix at the SOURCE, not with conversions
- [ ] If conversion needed, do it at boundaries
- [ ] Verify the fix compiles
- [ ] Verify tests still pass
- [ ] Check that the fix makes logical sense
- [ ] Document any non-obvious type conversions

---

---

## Related Documentation

- [Borrow Checker Error Fixes](borrow-checker.md)
- [Borrow Checker Error Fixes - Summary](borrow-checker-summary.md)
- [Borrow Checker Error Fixes - Documentation](README.md)
- [Reasoning Module - Week 2 Deliverable](../../explanations/ontology/reasoning-engine.md)
- [OntologyRepository Port](../../explanations/architecture/ports/04-ontology-repository.md)

## Remember

**The Rust compiler is your friend**, not your enemy. Type errors are design feedback. Listen to them, understand them, and fix the root cause. Your code will be better for it.

> "If you're fighting the borrow checker, you're probably doing something wrong architecturally."
> - Rust Community Wisdom

The same applies to type errors. If you're adding lots of `.into()` and `as` casts, step back and reconsider your design.
