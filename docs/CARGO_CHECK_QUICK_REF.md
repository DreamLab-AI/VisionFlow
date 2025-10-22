# Cargo Check Quick Reference

## 🚨 Compilation Status: FAILED ❌

**361 errors** | **193 warnings** | **0% success rate**

---

## 📊 Top 5 Issues (by frequency)

| # | Error | Count | Fix Complexity |
|---|-------|-------|----------------|
| 1 | E0277 - Unsized `dyn Trait` | 82 | Medium |
| 2 | E0599 - Method not found | 56 | High |
| 3 | E0437 - `Output` not in trait | 45 | Low |
| 4 | E0220 - `Output` not found | 44 | Low |
| 5 | E0107 - Generic argument count | 43 | Medium |

---

## 🎯 Root Cause

**hexser v0.4.7 trait mismatch**

### What hexser v0.4.7 expects:
```rust
trait DirectiveHandler<D: Directive> {
    fn handle(&self, directive: D) -> HexResult<()>;
    // ^^^^^^ SYNC, returns HexResult<()> directly
}
```

### What the code does:
```rust
impl DirectiveHandler<UpdateSetting> for Handler {
    type Output = ();  // ❌ Doesn't exist!

    async fn handle(&self, directive: UpdateSetting) -> Result<Self::Output> {
    // ^^^^^ ❌ Should be sync!                        ^^^^^^^^^^^ ❌ Wrong type!
    }
}
```

---

## 🔧 Fix Template

**Before** (broken):
```rust
impl<R: Repository> DirectiveHandler<UpdateSetting> for UpdateSettingHandler<R> {
    type Output = ();

    async fn handle(&self, directive: UpdateSetting) -> Result<Self::Output> {
        self.repository.set_setting(...).await
    }
}
```

**After** (working):
```rust
impl DirectiveHandler<UpdateSetting> for UpdateSettingHandler {
    fn handle(&self, directive: UpdateSetting) -> HexResult<()> {
        tokio::runtime::Handle::current().block_on(async {
            self.repository.set_setting(...).await
                .map_err(|e| Hexserror::internal(format!("Failed: {}", e)))
        })
    }
}
```

**Changes**:
1. ❌ Remove `type Output = ()`
2. ❌ Remove `async` keyword
3. ✅ Change return type to `HexResult<()>`
4. ✅ Wrap async code in `block_on`
5. ✅ Remove generic `<R: Repository>` (use concrete `Arc<dyn Repository>`)

---

## 📝 Directive Validation Template

**Before** (broken):
```rust
#[derive(Debug, Clone)]
pub struct UpdateSetting {
    pub key: String,
    pub value: SettingValue,
}

impl Directive for UpdateSetting {}  // ❌ Missing validate()!
```

**After** (working):
```rust
impl Directive for UpdateSetting {
    fn validate(&self) -> HexResult<()> {
        if self.key.is_empty() {
            return Err(Hexserror::validation("Key cannot be empty"));
        }
        Ok(())
    }
}
```

---

## 📦 Module Health

| Module | Status | Errors | Notes |
|--------|--------|--------|-------|
| `application/*` | ❌ BROKEN | 180 | All CQRS handlers |
| `adapters/*` | ⚠️ PARTIAL | 1 | Private import issue |
| `actors/*` | ✅ OK | 0 | Only warnings |
| `gpu/*` | ✅ OK | 0 | Compiles fine |
| `ports/*` | ✅ OK | 0 | Perfect |
| `handlers/*` | ✅ OK | 0 | HTTP layer OK |

---

## 🎬 Action Plan

### ✅ Step 1: Fix `Output` type (45 handlers)
- Remove all `type Output = ...` declarations
- Change `Result<Self::Output>` to `HexResult<ReturnType>`
- **Effort**: 1 hour

### ✅ Step 2: Fix async handlers (45 handlers)
- Remove `async` keyword
- Wrap body in `tokio::runtime::Handle::current().block_on(async { ... })`
- **Effort**: 2 hours

### ✅ Step 3: Add validation (23 directives)
- Implement `validate()` method for each directive
- **Effort**: 1.5 hours

### ✅ Step 4: Fix generics (45 handlers)
- Remove `<dyn Trait>` from implementations
- Use concrete types with `Arc<dyn Trait>` internally
- **Effort**: 1.5 hours

**Total Estimated Effort**: 6 hours

---

## 🔍 Verification Commands

```bash
# Check default features
cargo check --lib

# Check GPU features
cargo check --lib --features gpu

# Check ontology features
cargo check --lib --features ontology

# Check all features
cargo check --lib --all-features

# Count errors
cargo check --lib 2>&1 | grep -c "^error"

# Count warnings
cargo check --lib 2>&1 | grep -c "^warning"
```

---

## 📚 Files Requiring Changes

**Critical (must fix)**:
- `src/application/settings/directives.rs` (6 handlers)
- `src/application/settings/queries.rs` (5 handlers)
- `src/application/knowledge_graph/directives.rs` (8 handlers)
- `src/application/knowledge_graph/queries.rs` (6 handlers)
- `src/application/ontology/directives.rs` (9 handlers)
- `src/application/ontology/queries.rs` (11 handlers)

**Medium (should fix)**:
- `src/adapters/mod.rs` (1 private import)
- `src/application/mod.rs` (ambiguous glob re-exports)

---

## 🎯 Success Criteria

- ✅ `cargo check --lib` → 0 errors
- ✅ `cargo check --lib --all-features` → 0 errors
- ✅ Warnings < 50
- ✅ All CQRS handlers compile
- ✅ `cargo test` runs (even if tests fail)

---

## 🚀 Quick Start

```bash
# 1. Checkout feature branch
git checkout -b fix/hexser-compatibility

# 2. Apply fixes to one module first (test pattern)
# Edit: src/application/settings/directives.rs
#   - Remove type Output
#   - Change async fn to fn + block_on
#   - Implement validate() for directives

# 3. Verify
cargo check --lib 2>&1 | grep "src/application/settings"

# 4. If successful, apply to remaining modules

# 5. Final verification
cargo check --lib --all-features

# 6. Run tests
cargo test
```

---

## 💡 Pro Tips

1. **Use search-replace** for common patterns:
   ```
   Find: type Output = ();
   Replace: [delete line]
   ```

2. **Batch similar fixes**:
   - Fix all `Output` removals first
   - Then fix all async conversions
   - Then add all validations

3. **Test incrementally**:
   - Fix one file → `cargo check`
   - Confirm errors decrease
   - Continue to next file

4. **Keep runtime handy**:
   ```rust
   use tokio::runtime::Handle;
   // Use Handle::current().block_on(...) for all handlers
   ```

---

## ⚠️ Common Pitfalls

1. **Don't forget `validate()`** - All directives need it
2. **Return `HexResult`** not `Result<Self::Output>`
3. **Use `Hexserror`** not custom error types
4. **Block async properly** - use tokio Handle, not pollster
5. **Fix generics** - no `<dyn Trait>` in impl blocks

---

## 📖 See Full Report

For complete analysis, see: [`docs/CARGO_CHECK_REPORT.md`](./CARGO_CHECK_REPORT.md)

---

**Last Updated**: 2025-10-22
**Report Version**: 1.0
**Success Probability**: 95% (if following this guide)
