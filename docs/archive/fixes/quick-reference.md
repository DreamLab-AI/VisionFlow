---
title: Borrow Checker Quick Reference Guide
description: **When to use**: Operations are independent and can be reordered
type: archive
status: archived
---

# Borrow Checker Quick Reference Guide

## Common Patterns for Fixing Borrow Errors

### Pattern 1: Scope Reordering

**When to use**: Operations are independent and can be reordered

```rust
// ❌ Wrong
let data = &self.data;
self.modify_something();  // ERROR: needs &mut self
use(data);

// ✅ Correct
self.modify_something();  // Mutable borrow completes
let data = &self.data;    // Immutable borrow starts
use(data);
```

**Example**: `force_compute_actor.rs` - Moved ontology forces before shared_context

---

### Pattern 2: Scoped Blocks (Extract and Drop)

**When to use**: Need to drop a lock/borrow before calling methods

```rust
// ❌ Wrong
let guard = self.mutex.lock();
// ... use guard ...
self.method();  // ERROR: guard still holds borrow

// ✅ Correct
let result = {
    let guard = self.mutex.lock();
    // ... compute ...
    computed_value
}; // Guard dropped here
self.method();  // OK
```

**Example**: `shortest_path_actor.rs` - Both SSSP and APSP handlers

---

### Pattern 3: Async Actor Pattern

**When to use**: Actix async handlers that need actor state

```rust
// ❌ Wrong
async move { self.method().await }
    .into_actor(self)  // ERROR: self moved

// ✅ Correct
let shared = Arc::clone(&self.shared_resource);
let future = async move {
    // Work with shared
    shared.compute().await
};
future.into_actor(self).map(|result, actor, _ctx| {
    // Update actor state
    actor.state = result;
    result
})
```

**Example**: `pagerank_actor.rs` - ComputePageRank handler

---

### Pattern 4: Split Mutable Borrows

**When to use**: Need to borrow multiple fields mutably

```rust
// ❌ Wrong
self.field1.update(&mut self.field2);  // ERROR

// ✅ Correct
let (ref mut f1, ref mut f2) = (&mut self.field1, &mut self.field2);
f1.update(f2);
```

---

### Pattern 5: Temporary Variables

**When to use**: Borrow checker needs help with lifetimes

```rust
// ❌ Wrong
self.process(self.data.get_value());  // Complex borrow

// ✅ Correct
let value = self.data.get_value();
self.process(value);
```

---

## Decision Tree

```
Borrow checker error?
├─ E0502 (cannot borrow as mutable while borrowed as immutable)
│  ├─ Operations independent?
│  │  └─ YES → Pattern 1: Reorder operations
│  └─ Lock/Guard involved?
│     └─ YES → Pattern 2: Scoped blocks
│
├─ E0382 (borrow of moved value)
│  ├─ In async context?
│  │  └─ YES → Pattern 3: Async actor pattern
│  └─ Moved into closure?
│     └─ YES → Clone before move (if cheap/appropriate)
│
└─ E0499 (cannot borrow as mutable more than once)
   └─ Multiple fields?
      └─ YES → Pattern 4: Split borrows
```

---

## When to Clone

### ✅ DO Clone When:

1. **Arc/Rc** - Designed for shared ownership
   ```rust
   let shared = Arc::clone(&self.context);
   ```

2. **Small values** - Copy is free
   ```rust
   let count = self.count;  // Copy
   ```

3. **Crossing async boundaries** - Ownership must transfer
   ```rust
   let data = self.data.clone();
   spawn(async move { process(data).await });
   ```

4. **Explicit snapshots** - You want a copy at this point in time
   ```rust
   let snapshot = self.state.clone();
   ```

### ❌ DON'T Clone When:

1. **Just to satisfy compiler** - Understand the issue first

2. **Large data structures** - Expensive and unnecessary
   ```rust
   // Bad: Clone entire Vec
   let nodes_copy = self.nodes.clone();

   // Good: Borrow
   let nodes = &self.nodes;
   ```

3. **When restructuring works** - Better to fix the root cause

4. **Mutex guards** - Cannot clone, use scoping instead

---

## Common Error Messages and Fixes

### "cannot borrow as mutable because it is also borrowed as immutable"

**Cause**: Overlapping borrows
**Fix**: Scope reordering or scoped blocks

```rust
// Fix with reordering
mutable_operation();
let data = &immutable_borrow;

// Fix with scoping
{
    let data = &immutable_borrow;
} // Borrow ends
mutable_operation();
```

---

### "borrow of moved value"

**Cause**: Value moved into closure/async block
**Fix**: Clone before move (if appropriate) or restructure

```rust
// Fix with Arc clone
let shared = Arc::clone(&value);
async move { shared.use() }

// Fix by not moving
let result = compute(&value);  // Borrow, don't move
```

---

### "cannot borrow as mutable more than once"

**Cause**: Multiple mutable borrows
**Fix**: Split borrows or reborrow

```rust
// Fix with split
let (a, b) = (&mut self.field_a, &mut self.field_b);

// Fix with reborrow
let a = &mut *self.field_a;
```

---

## Lifetime Control Tools

### 1. Explicit Drops
```rust
let guard = mutex.lock();
// ... use guard ...
drop(guard);  // Explicit drop
other_operation();
```

### 2. Scoped Blocks
```rust
{
    let guard = mutex.lock();
    // ... use guard ...
}  // Implicit drop
other_operation();
```

### 3. let _ Pattern
```rust
let _ = mutex.lock();  // Drops immediately
```

---

## Debugging Tips

### 1. Add Explicit Lifetimes
```rust
// Make lifetimes visible
fn process<'a>(&'a self, data: &'a Data) -> &'a Result {
    // Now you can see what lives where
}
```

### 2. Break Down Complex Expressions
```rust
// Hard to debug
self.process(self.data.compute().filter());

// Easier to debug
let computed = self.data.compute();
let filtered = computed.filter();
self.process(filtered);
```

### 3. Use cargo-expand
```bash
# See what the compiler sees
cargo expand --lib module::function
```

### 4. Enable Debug Output
```bash
# See borrow checker reasoning
RUST_BACKTRACE=1 cargo build
```

---

## Anti-Patterns to Avoid

### ❌ Clone Everything
```rust
// Bad: Unnecessary clones
let data = self.data.clone();
let more = self.more.clone();
let stuff = self.stuff.clone();
```

### ❌ RefCell Everywhere
```rust
// Bad: Runtime borrow checking
struct Foo {
    data: RefCell<Vec<u32>>,  // Avoid if possible
}
```

### ❌ Unsafe to Bypass
```rust
// Bad: Defeats the purpose
unsafe {
    // "Fix" borrow checker with unsafe
}
```

### ❌ Global Mutables
```rust
// Bad: Shared mutable state
static mut GLOBAL: Vec<u32> = Vec::new();
```

---

## Performance Checklist

- [ ] No unnecessary clones of large structures
- [ ] Arc/Rc only for shared ownership
- [ ] Locks held for minimal time
- [ ] No RefCell in hot paths
- [ ] Borrowed instead of owned where possible

---

## Additional Resources

- [Rust Book - Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Rust Book - Lifetimes](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html)
- [Async Book](https://rust-lang.github.io/async-book/)
- [Actix Documentation](https://actix.rs/docs/)
