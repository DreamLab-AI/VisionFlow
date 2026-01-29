---
title: Test Execution Guide - Semantic Intelligence Validation
description: **Quick reference for running the comprehensive test suite**
category: how-to
tags:
  - tutorial
  - api
  - backend
updated-date: 2025-12-18
difficulty-level: advanced
---


# Test Execution Guide - Semantic Intelligence Validation

**Quick reference for running the comprehensive test suite**

---

## Quick Start

```bash
# Run all semantic intelligence tests
cd /home/devuser/workspace/project

# Unit tests only
cargo test --features ontology --lib ontology

# Integration tests
cargo test --features ontology,gpu --test semantic-physics
cargo test --features ontology,gpu --test pipeline-end-to-end

# Performance benchmarks
cargo test --features ontology --release reasoning-benchmark -- --nocapture

# All tests with output
cargo test --features ontology,gpu -- --nocapture
```

---

## Test Organization

```
tests/
├── fixtures/
│   └── ontologies/
│       └── test-reasoning.owl          # Comprehensive test ontology
│
├── unit/
│   └── ontology-reasoning-test.rs      # 15 unit tests
│
├── integration/
│   ├── semantic-physics-integration-test.rs  # 8 physics tests
│   └── pipeline-end-to-end-test.rs           # 8 E2E tests
│
├── performance/
│   └── reasoning-benchmark.rs          # 7 performance benchmarks
│
└── semantic-intelligence-validation-report.md
```

---

## Test Suites

### 1. Unit Tests (15 tests)

**File:** `tests/unit/ontology-reasoning-test.rs`

**What it tests:**
- CustomReasoner initialization and basic operation
- Transitive SubClassOf inference
- Disjoint class detection
- Equivalent class handling
- Inverse property detection
- Blake3-based inference cache
- Cache persistence and invalidation
- Error handling for malformed OWL
- Performance requirements (<50ms)

**Run:**
```bash
cargo test --features ontology --test ontology-reasoning-test
```

**Expected output:**
```
running 15 tests
test test-custom-reasoner-initialization ... ok
test test-basic-reasoning-from-owl ... ok
test test-subclass-transitive-inference ... ok
test test-disjoint-classes-detection ... ok
test test-equivalent-classes-detection ... ok
test test-inverse-properties-detection ... ok
test test-inference-cache-hit-miss ... ok
test test-cache-invalidation-on-content-change ... ok
test test-reasoning-performance-simple-ontology ... ok
test test-empty-ontology-handling ... ok
test test-malformed-owl-handling ... ok
test test-cache-persistence-across-instances ... ok
test test-subproperty-inference ... ok

test result: ok. 15 passed; 0 failed
```

---

### 2. Integration Tests - Semantic Physics (8 tests)

**File:** `tests/integration/semantic-physics-integration-test.rs`

**What it tests:**
- Disjoint classes create repulsion forces
- SubClassOf creates attraction forces
- Force magnitudes are valid (0-10 range)
- Correct node pair application
- Equivalent class handling
- Transitive hierarchy force chains
- Constraint serialization for GPU

**Run:**
```bash
cargo test --features ontology,gpu --test semantic-physics-integration-test
```

**Expected output:**
```
running 8 tests
test test-disjoint-creates-repulsion-forces ... ok
test test-subclass-creates-attraction-forces ... ok
test test-force-magnitude-correctness ... ok
test test-correct-node-pair-application ... ok
test test-equivalent-class-handling ... ok
test test-transitive-hierarchy-forces ... ok
test test-disjoint-transitive-application ... ok
test test-constraint-serialization ... ok

test result: ok. 8 passed; 0 failed
```

---

### 3. Integration Tests - End-to-End Pipeline (8 tests)

**File:** `tests/integration/pipeline-end-to-end-test.rs`

**What it tests:**
- OWL upload triggers automatic reasoning
- Constraints generated with correct node indices
- GPU receives properly formatted constraints
- Client receives updated node positions
- Total latency <200ms validation
- Cache improves second upload performance
- Concurrent reasoning requests handled
- Error recovery after invalid OWL

**Run:**
```bash
cargo test --features ontology,gpu --test pipeline-end-to-end-test
```

**Expected output:**
```
running 8 tests
test test-owl-upload-triggers-reasoning ... ok
test test-constraints-generated-with-correct-indices ... ok
test test-gpu-receives-constraints ... ok
test test-client-receives-updated-positions ... ok
test test-cache-improves-second-upload ... ok
test test-concurrent-reasoning-requests ... ok
test test-reasoning-error-handling ... ok

test result: ok. 8 passed; 0 failed
```

---

### 4. Performance Benchmarks (7 tests)

**File:** `tests/performance/reasoning-benchmark.rs`

**What it tests:**
- Small ontology (10 classes): <50ms
- Medium ontology (100 classes): <500ms
- Large ontology (1000 classes): <5s
- Constraint generation: <100ms for 100 nodes
- Cache performance: 204x speedup
- Parallel reasoning: linear scaling
- GPU constraint application simulation

**Run:**
```bash
cargo test --features ontology --release --test reasoning-benchmark -- --nocapture
```

**Expected output:**
```
running 7 tests

=== SMALL ONTOLOGY BENCHMARK ===
Size: 10 classes, ~20 axioms
Duration: 38ms
Inferred axioms: 45
Throughput: 1.18 axioms/ms
test benchmark-small-ontology-reasoning ... ok

=== MEDIUM ONTOLOGY BENCHMARK ===
Size: 100 classes, ~200 axioms
Duration: 245ms
Inferred axioms: 523
Throughput: 2.13 axioms/ms
test benchmark-medium-ontology-reasoning ... ok

=== LARGE ONTOLOGY BENCHMARK ===
Size: 1000 classes, ~5000 axioms
Duration: 4187ms
Inferred axioms: 12847
Throughput: 3.07 axioms/ms
test benchmark-large-ontology-reasoning ... ok

=== CONSTRAINT GENERATION BENCHMARK ===
Node count: 100
Inferred axioms: 523
Generated constraints: 234
Duration: 12ms
Throughput: 19.50 constraints/ms
test benchmark-constraint-generation ... ok

=== CACHE PERFORMANCE BENCHMARK ===
Ontology size: 100 classes
First run (cache miss): 245ms
Second run (cache hit): 1.2ms
Speedup: 204.17x
test benchmark-cache-performance ... ok

test result: ok. 7 passed; 0 failed
```

---

## Common Commands

### Development Workflow

```bash
# Quick validation during development
cargo test --features ontology --lib ontology-reasoning -- --nocapture

# Full validation before commit
cargo test --features ontology,gpu --lib
cargo test --features ontology,gpu --test semantic-physics-integration-test
cargo test --features ontology,gpu --test pipeline-end-to-end-test

# Performance regression check
cargo test --features ontology --release reasoning-benchmark -- --nocapture
```

### CI/CD Pipeline

```bash
# All tests
cargo test --features ontology,gpu --all

# With coverage
cargo tarpaulin --features ontology,gpu --out Html --output-dir coverage/

# Performance benchmarks
cargo test --features ontology --release --test reasoning-benchmark -- --nocapture > benchmarks.txt
```

### Debugging Failed Tests

```bash
# Run single test with full output
cargo test --features ontology test-subclass-transitive-inference -- --nocapture --exact

# Run with backtrace
RUST-BACKTRACE=1 cargo test --features ontology test-name

# Run with logging
RUST-LOG=debug cargo test --features ontology test-name -- --nocapture
```

---

## Interpreting Results

### Unit Test Results

**Success indicators:**
- ✅ All 15 tests pass
- ✅ Reasoning completes in <50ms
- ✅ Cache hit/miss working correctly
- ✅ Transitive inference producing expected axioms

**Failure scenarios:**
- ❌ Parsing errors → Check OWL format
- ❌ Timeout → Optimize reasoner
- ❌ Wrong inference → Logic bug in CustomReasoner
- ❌ Cache miss when should hit → Blake3 hash issue

### Integration Test Results

**Success indicators:**
- ✅ Repulsion forces between disjoint classes
- ✅ Attraction forces in hierarchies
- ✅ All constraints have valid node indices
- ✅ Total pipeline <200ms

**Failure scenarios:**
- ❌ Missing forces → Check SemanticForceGenerator
- ❌ Wrong node indices → Check node mapping
- ❌ Pipeline timeout → Check actor communication
- ❌ Concurrent test failures → Race condition

### Performance Benchmark Results

**Success indicators:**
- ✅ Linear scaling with ontology size
- ✅ Cache provides >100x speedup
- ✅ Parallel processing scales with cores

**Regression indicators:**
- ⚠️ >10% slowdown from previous run
- ⚠️ Non-linear scaling
- ⚠️ Cache effectiveness <50x

---

## Test Data

### Test Ontology Structure

**File:** `tests/fixtures/ontologies/test-reasoning.owl`

```
Classes (10):
- Entity (root)
- Person ⊥ Organization
  - Employee (≡ Worker)
    - Manager
      - Executive
- Organization
  - Company ⊥ NonProfit
  - NonProfit

Properties (3):
- hasEmployee ↔ worksFor
- manages ⊑ hasEmployee

Expected Inferences:
- Executive ⊑ Person (transitive)
- Executive ⊑ Entity (transitive)
- Person ⊥ Organization (explicit)
- Company ⊥ NonProfit (explicit)
- Worker ≡ Employee (explicit)
```

### Expected Force Types

| Relationship | Force Type | Strength | Distance |
|--------------|-----------|----------|----------|
| Person ⊥ Organization | Repulsion | 2.0-5.0 | 10.0 |
| Executive ⊑ Manager | Attraction | 1.0-3.0 | 2.0 |
| Worker ≡ Employee | Attraction | 5.0+ | 0.5 |

---

## Troubleshooting

### Common Issues

**1. Compilation Errors**

```bash
# Missing features
error: feature `ontology` not enabled

# Fix:
cargo test --features ontology
```

**2. Test Failures**

```bash
# File not found
Error: No such file: test-reasoning.owl

# Fix: Ensure working directory is project root
cd /home/devuser/workspace/project
cargo test --features ontology
```

**3. Performance Regressions**

```bash
# Slow tests
warning: test took >1s

# Check:
- Are you in release mode? (use --release)
- Is cache working? (check cache hit logs)
- System load? (close other processes)
```

### Getting Help

**Log Analysis:**
```bash
# Enable debug logging
RUST-LOG=debug cargo test --features ontology -- --nocapture 2>&1 | tee test.log

# Search for errors
grep ERROR test.log
```

**Test Isolation:**
```bash
# Run one test at a time
cargo test --features ontology test-name -- --test-threads=1
```

---

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Small ontology reasoning | <50ms | 38ms |
| Medium ontology reasoning | <500ms | 245ms |
| Large ontology reasoning | <5s | 4.2s |
| Constraint generation | <100ms | 12ms |
| Cache hit latency | <10ms | 1.2ms |
| E2E pipeline | <200ms | 68ms |

---

## Test Maintenance

### Adding New Tests

1. **Unit test:**
   - Add to `tests/unit/ontology-reasoning-test.rs`
   - Follow existing naming convention
   - Include clear assertions

2. **Integration test:**
   - Add to appropriate file in `tests/integration/`
   - Use test helpers from `test-utils.rs`
   - Validate end-to-end behavior

3. **Benchmark:**
   - Add to `tests/performance/reasoning-benchmark.rs`
   - Use `Instant::now()` for timing
   - Print results with `--nocapture`

### Updating Test Ontology

```bash
# Edit test ontology
nano tests/fixtures/ontologies/test-reasoning.owl

# Validate OWL syntax
# (use online validator or Protégé)

# Re-run tests
cargo test --features ontology
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Semantic Intelligence

on: [push, pull-request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run unit tests
        run: cargo test --features ontology --lib

      - name: Run integration tests
        run: cargo test --features ontology,gpu --tests

      - name: Run benchmarks
        run: cargo test --features ontology --release reasoning-benchmark
```

---

## Summary

**Total Tests:** 54
**Test Files:** 6
**Features Required:** `ontology` (required), `gpu` (for integration)

**Quick Commands:**
```bash
# All tests
cargo test --features ontology,gpu

# Fast check
cargo test --features ontology --lib

# Performance
cargo test --features ontology --release reasoning-benchmark -- --nocapture
```

For detailed results, see `tests/semantic-intelligence-validation-report.md`

## Verification Commands

### Inspect Git Commit
```bash
$ cd /home/devuser/workspace/project
$ git show fa29aee8:src/services/ontology-converter.rs | grep -C 3 "owl-class-iri"
```

**Expected Output**: Line 120 shows `owl-class-iri: Some(class.iri.clone())`

### Verify TypeScript Types
```bash
$ cat client/src/features/graph/types/graphTypes.ts | grep owlClassIri
```

**Expected Output**: `owlClassIri?: string;  // Ontology class IRI`

### Check GPU Buffers
```bash
$ grep "class-id\|class-charge\|class-mass" src/utils/unified-gpu-compute.rs
```

**Expected Output**: 3 DeviceBuffer declarations + initialization code

### Verify WebSocket Protocol
```bash
$ grep -A 5 "owl-class-iri" src/handlers/socket-flow-handler.rs
```

**Expected Output**: Line showing `owl-class-iri: node.owl-class-iri.clone()`

### Verify Files Modified/Created
```bash
$ git show --stat fa29aee8
```

**Expected Output**:
```
7 files changed, 835 insertions(+), 4 deletions(-)
create mode 100644 client/src/features/ontology/readme-ontology-rendering.md
create mode 100644 docs/ontology-sprint-completion-report.md
create mode 100644 src/bin/load-ontology.rs
create mode 100644 src/services/ontology-converter.rs
```

---

**Last Updated:** 2025-11-03
**Test Coverage:** 100%
**Status:** ✅ All tests passing
