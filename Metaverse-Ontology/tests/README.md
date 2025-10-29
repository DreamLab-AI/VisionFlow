# Ontology Storage Architecture - Test Suite

## 📋 Overview

Comprehensive testing strategy for validating the complete data flow: **GitHub Markdown → Database → OWL Extraction → Reasoning**

## 🎯 Success Criteria

✅ All 1,297 ObjectSomeValuesFrom restrictions preserved
✅ Zero semantic loss through the pipeline
✅ 15x speed improvement on re-sync operations
✅ < 135s full ontology build time
✅ 100% Actor system coordination reliability

## 🧪 Test Categories

### 1. Unit Tests (`tests/unit/`)
Fast, isolated tests for individual components.

```bash
npm run test:unit
```

**Coverage**:
- OWL parser validation
- Hash computation (SHA1, SHA-256)
- Edge case handling
- Data transformation logic

**Execution time**: ~5 minutes

### 2. Integration Tests (`tests/integration/`)
Service coordination and Actor system messaging.

```bash
npm run test:integration
```

**Coverage**:
- GitHub sync → Database → OWL extraction flow
- OwlValidatorService integration
- whelk-rs reasoning integration
- Error recovery and retry logic

**Execution time**: ~10 minutes

### 3. Performance Tests (`tests/performance/`)
Benchmark validation and regression detection.

```bash
npm run test:performance
```

**Coverage**:
- Single class extraction timing
- Full ontology build timing
- Change detection performance
- Memory usage validation

**Execution time**: ~15 minutes

### 4. End-to-End Tests (`tests/e2e/`)
Complete pipeline validation with real data.

```bash
npm run test:e2e
```

**Coverage**:
- Complete GitHub → Reasoning flow
- Semantic preservation validation
- 988-class ontology processing
- Real-world scenarios

**Execution time**: ~20 minutes

### 5. Regression Tests (`tests/regression/`)
Baseline comparison and backward compatibility.

```bash
npm run test:regression
```

**Coverage**:
- Semantic preservation vs baseline
- Performance regression detection
- API compatibility validation

**Execution time**: ~30 minutes

## 🚀 Quick Start

### Setup

```bash
# Install dependencies
npm install

# Set up test environment
cp tests/.env.test.example tests/.env.test
# Edit .env.test with your configuration

# Create test database
createdb ontology_test

# Run migrations
npm run db:migrate

# Seed test data
npm run db:seed:test
```

### Running Tests

```bash
# Run all tests
npm test

# Run specific test suite
npm run test:unit
npm run test:integration
npm run test:performance
npm run test:e2e

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm run test:watch

# Run specific test file
npm test -- tests/unit/owl-parser.test.ts

# Run with debugging
npm run test:debug
```

## 📊 Test Execution Matrix

| Test Type | Duration | When to Run | Required for PR |
|-----------|----------|-------------|-----------------|
| Unit | 5 min | Every commit | ✅ Yes |
| Integration | 10 min | Every commit | ✅ Yes |
| Performance | 15 min | Nightly | ❌ No |
| E2E | 20 min | Pre-release | ❌ No |
| Regression | 30 min | Weekly | ❌ No |

## 🎯 Performance Acceptance Criteria

### Timing Requirements

| Operation | Target | Maximum | Baseline |
|-----------|--------|---------|----------|
| Single class extraction | < 130ms | 200ms | 125ms |
| Full ontology build | < 135s | 150s | 125s |
| Initial GitHub sync | < 125s | 135s | 120s |
| Re-sync (no changes) | < 8s | 10s | 8.3s |
| Re-sync (10 changes) | < 12s | 15s | 11.7s |

### Semantic Preservation Requirements

| Metric | Requirement |
|--------|-------------|
| Restriction preservation | 100% (all 1,297) |
| Class hierarchy preservation | 100% |
| Property preservation | 100% |
| Annotation preservation | 95%+ |

### Resource Requirements

| Resource | Target | Maximum |
|----------|--------|---------|
| Memory usage (peak) | < 400MB | 500MB |
| Database size | ~50MB | 100MB |
| CPU usage | < 60% | 80% |

## 🏗️ Test Architecture

```
tests/
├── unit/                       # Fast, isolated tests
│   ├── owl-parser.test.ts
│   ├── hash-computation.test.ts
│   ├── edge-cases.test.ts
│   └── data-transformation.test.ts
│
├── integration/                # Service coordination
│   ├── service-coordination.test.ts
│   ├── actor-messaging.test.ts
│   ├── error-recovery.test.ts
│   └── database-integration.test.ts
│
├── performance/                # Benchmarks
│   ├── component-benchmarks.test.ts
│   ├── change-detection.test.ts
│   ├── memory-benchmarks.test.ts
│   └── concurrent-operations.test.ts
│
├── e2e/                        # End-to-end flows
│   ├── happy-path.test.ts
│   ├── complete-pipeline.test.ts
│   └── real-world-scenarios.test.ts
│
├── regression/                 # Baseline comparison
│   ├── baseline/
│   │   ├── ontology-snapshot-v1.0.0.json
│   │   ├── restrictions-snapshot-v1.0.0.json
│   │   └── performance-baseline-v1.0.0.json
│   ├── semantic-preservation.test.ts
│   ├── performance-regression.test.ts
│   └── api-compatibility.test.ts
│
├── setup.ts                    # Global test setup
├── global-setup.ts             # Database setup
├── global-teardown.ts          # Cleanup
├── jest.config.js              # Jest configuration
└── TEST_PLAN.md                # Comprehensive test plan
```

## 🔍 Test Coverage

Target coverage thresholds:

```json
{
  "coverageThreshold": {
    "global": {
      "branches": 75,
      "functions": 80,
      "lines": 80,
      "statements": 80
    }
  }
}
```

View coverage report:

```bash
npm test -- --coverage
open coverage/lcov-report/index.html
```

## 🐛 Debugging Tests

### Using VS Code

1. Add to `.vscode/launch.json`:

```json
{
  "type": "node",
  "request": "launch",
  "name": "Jest Debug",
  "program": "${workspaceFolder}/node_modules/.bin/jest",
  "args": ["--runInBand", "${file}"],
  "console": "integratedTerminal"
}
```

2. Set breakpoints and press F5

### Using Chrome DevTools

```bash
node --inspect-brk node_modules/.bin/jest --runInBand tests/unit/owl-parser.test.ts
```

Then open `chrome://inspect` in Chrome.

## 🔄 CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Every push to `main` or `develop`
- Every pull request
- Nightly (performance + regression tests)

See `.github/workflows/ontology-tests.yml`

### Pre-commit Hooks

```bash
# Install pre-commit hooks
npm run prepare

# Hooks will run:
# - Linting
# - Type checking
# - Unit tests
# - Format checking
```

## 📈 Monitoring

### Test Metrics

- Test coverage: > 80%
- Test execution time: < 10 minutes (full suite)
- Test flakiness: < 1%
- Mean time to detect (MTTD): < 1 hour

### Performance Monitoring

Performance metrics are tracked in:
- **Datadog**: Real-time monitoring
- **GitHub Actions**: PR checks
- **Grafana**: Historical trends

## 🆘 Troubleshooting

### Tests are slow

```bash
# Run tests in parallel
npm test -- --maxWorkers=4

# Skip slow tests
ENABLE_SLOW_TESTS=false npm test
```

### Database connection errors

```bash
# Check PostgreSQL is running
pg_isready

# Check connection string
echo $TEST_DATABASE_URL

# Reset test database
dropdb ontology_test && createdb ontology_test
npm run db:migrate
```

### Memory issues

```bash
# Increase heap size
node --max-old-space-size=4096 node_modules/.bin/jest

# Enable garbage collection
node --expose-gc node_modules/.bin/jest
```

### Flaky tests

```bash
# Run test multiple times to identify flakiness
npm test -- --testNamePattern="flaky test" --maxWorkers=1 --runInBand

# Check for race conditions
# Add delays or better synchronization
```

## 📚 Additional Resources

- [Jest Documentation](https://jestjs.io/)
- [Testing Best Practices](https://testingjavascript.com/)
- [TEST_PLAN.md](./TEST_PLAN.md) - Detailed test specifications
- [Performance Test Guide](./performance/README.md)

## 🤝 Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain coverage > 80%
4. Update test documentation
5. Add performance benchmarks if relevant

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact QA team
- Check test plan documentation
