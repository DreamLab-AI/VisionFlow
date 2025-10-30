# Performance Test Suite

## Overview

Performance tests validate that the ontology storage architecture meets strict timing requirements:

- **Single class extraction**: < 130ms
- **Full ontology build**: < 135s
- **Initial sync**: < 125s
- **Re-sync (no changes)**: < 8s (15x faster)
- **Re-sync (10 changes)**: < 12s

## Running Performance Tests

```bash
# Run all performance tests
npm run test:performance

# Run with Node.js memory profiling
node --expose-gc --max-old-space-size=4096 node_modules/.bin/jest tests/performance

# Generate performance report
npm run test:performance -- --json --outputFile=reports/performance-report.json
```

## Baseline Management

Performance baselines are stored in `tests/regression/baseline/`.

### Creating a New Baseline

```bash
# Run performance tests and save as baseline
npm run test:performance:baseline

# This creates:
# - performance-baseline-vX.X.X.json
# - Timestamp and git commit hash
# - Hardware specs for context
```

### Updating Baselines

Baselines should be updated when:
- Hardware changes significantly
- Algorithm improvements are intentional
- Major version releases

## Test Structure

```
performance/
├── component-benchmarks.test.ts    # Individual component timing
├── change-detection.test.ts        # Sync performance validation
├── memory-benchmarks.test.ts       # Memory usage tracking
└── concurrent-operations.test.ts   # Parallel processing tests
```

## Interpreting Results

### Timing Results

```
Single class extraction: 125.3ms (target: <130ms) ✅
Full ontology build: 132.1s (target: <135s) ✅
Re-sync (no changes): 7.8s (target: <8s) ✅
```

### Memory Results

```
Peak memory usage: 387MB (target: <500MB) ✅
Memory released after GC: 142MB ✅
Memory leaks detected: 0 ✅
```

## Performance Regression Detection

The CI/CD pipeline automatically detects regressions:

```yaml
# Performance must not regress by more than 10%
if current_time > baseline_time * 1.10:
  fail_build()
```

## Hardware Specifications

Performance tests should include hardware context:

```json
{
  "hardware": {
    "cpu": "AMD Ryzen 9 5900X",
    "memory": "32GB DDR4",
    "disk": "NVMe SSD",
    "os": "Ubuntu 22.04"
  }
}
```

## Profiling

### CPU Profiling

```bash
# Generate flame graph
node --prof node_modules/.bin/jest tests/performance
node --prof-process isolate-*.log > profile.txt

# Use clinic.js for advanced profiling
npm install -g clinic
clinic doctor -- node node_modules/.bin/jest tests/performance
```

### Memory Profiling

```bash
# Generate heap snapshot
node --inspect node_modules/.bin/jest tests/performance

# Then use Chrome DevTools:
# chrome://inspect -> Devices -> Memory
```

## Continuous Monitoring

Performance metrics are tracked over time:

- **Datadog**: Real-time performance monitoring
- **GitHub Actions**: PR performance checks
- **Grafana**: Historical trend visualization
