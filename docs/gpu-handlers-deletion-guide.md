# GPU Handler Deletion Guide

Quick reference for removing incorrect Handler implementations.

## File: clustering_actor.rs

### Handler 1: RunKMeans
**Lines to DELETE**: 628-651 (24 lines)

```bash
# Preview what will be deleted
sed -n '628,651p' src/actors/gpu/clustering_actor.rs
```

**Reason**: Returns error "KMeans clustering not yet implemented" but implementation exists at lines 85-166

---

### Handler 2: RunCommunityDetection  
**Lines to DELETE**: 653-678 (26 lines)

```bash
# Preview what will be deleted
sed -n '653,678p' src/actors/gpu/clustering_actor.rs
```

**Reason**: Returns error "Community detection not yet implemented" but implementation exists at lines 169-261

---

### Handler 3: PerformGPUClustering
**Lines to DELETE**: 680-702 (23 lines)

```bash
# Preview what will be deleted
sed -n '680,702p' src/actors/gpu/clustering_actor.rs
```

**Reason**: Returns error "GPU clustering not yet implemented" - GPUManagerActor already handles this

---

## File: force_compute_actor.rs (Optional Cleanup)

### Delegation-Only Handlers
**Lines to DELETE**: 885-1016 (132 lines total)

Individual handlers:
- `RunCommunityDetection`: 885-892 (8 lines)
- `GetConstraints`: 907-914 (8 lines)  
- `UpdateConstraints`: 916-923 (8 lines)
- `UploadConstraintsToGPU`: 925-932 (8 lines)
- `TriggerStressMajorization`: 934-945 (12 lines)
- `GetStressMajorizationStats`: 947-962 (16 lines)
- `ResetStressMajorizationSafety`: 964-978 (15 lines)
- `UpdateStressMajorizationParams`: 980-991 (12 lines)
- `PerformGPUClustering`: 993-1002 (10 lines)
- `GetClusteringResults`: 1004-1016 (13 lines)

```bash
# Preview what will be deleted
sed -n '885,1016p' src/actors/gpu/force_compute_actor.rs
```

**Reason**: These only return errors directing to other actors. GPUManagerActor handles routing.

---

## Deletion Commands

### Critical Fixes (clustering_actor.rs)

```bash
# Create backup first
cp src/actors/gpu/clustering_actor.rs src/actors/gpu/clustering_actor.rs.backup

# Delete all three incorrect handlers at once
# This removes lines 628-702 (75 lines total)
sed -i '628,702d' src/actors/gpu/clustering_actor.rs

# Verify it still compiles
cargo check
```

### Optional Cleanup (force_compute_actor.rs)

```bash
# Create backup first
cp src/actors/gpu/force_compute_actor.rs src/actors/gpu/force_compute_actor.rs.backup

# Delete delegation-only handlers
# This removes lines 885-1016 (132 lines total)
sed -i '885,1016d' src/actors/gpu/force_compute_actor.rs

# Verify it still compiles
cargo check
```

---

## Verification Steps

After deletion, verify:

1. **Compilation**: `cargo check` should still pass
2. **Tests**: `cargo test` (if applicable)
3. **Functionality**: Test GPU clustering and force computation

---

## Rollback (If Needed)

```bash
# Restore from backup
cp src/actors/gpu/clustering_actor.rs.backup src/actors/gpu/clustering_actor.rs
cp src/actors/gpu/force_compute_actor.rs.backup src/actors/gpu/force_compute_actor.rs
```
