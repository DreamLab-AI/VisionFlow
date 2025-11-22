# Integration Testing Documentation

## Overview

This document describes the comprehensive integration test suite for the Logseq Knowledge Graph Ontology project. The test suite validates that all tools (Python converters, JavaScript pipeline, Rust audit/WASM, and cross-tool integrations) work correctly across all 6 ontology domains.

## Test Suite Structure

```
tests/
├── integration/
│   ├── test-data/              # Test data files
│   │   ├── ai/                 # AI domain (3 files)
│   │   ├── mv/                 # Metaverse domain (3 files)
│   │   ├── tc/                 # Telecollaboration domain (3 files)
│   │   ├── rb/                 # Robotics domain (3 files)
│   │   ├── dt/                 # Disruptive Tech domain (3 files)
│   │   └── bc/                 # Blockchain domain (3 files)
│   ├── fixtures/               # Test fixtures
│   ├── outputs/                # Generated test outputs
│   ├── reports/                # Test reports
│   ├── test_python_tools.py    # Python converters tests
│   ├── test_rust_tools.rs      # Rust audit/WASM tests
│   ├── test_js_pipeline.js     # JS pipeline tests
│   └── test_interoperability.py # Cross-tool tests
└── run_all_tests.sh            # Master test runner

docs/
└── testing/
    └── INTEGRATION-TESTS.md    # This file
```

## Test Data

### Domain Coverage

The test suite includes **18 test files** (3 per domain):

1. **AI (Artificial Intelligence)** - `ai/`
2. **MV (Metaverse)** - `mv/`
3. **TC (Telecollaboration)** - `tc/`
4. **RB (Robotics)** - `rb/`
5. **DT (Disruptive Technologies)** - `dt/`
6. **BC (Blockchain)** - `bc/`

### Test File Types

Each domain has 3 test files:

1. **Valid File** (`valid-*.md`)
   - Complete, correct ontology block
   - All required properties present
   - Proper OWL2 compliance
   - Valid domain-specific properties

2. **Invalid File** (`invalid-*.md`)
   - Intentionally missing required properties
   - Invalid property values
   - Wrong physicality/role for domain
   - Tests error handling

3. **Edge Case File** (`edge-*.md`)
   - Minimal valid ontology
   - Maximal property sets
   - Unusual but valid structures
   - Boundary conditions
   - Complex property values

### Example Test Files

#### Valid File Example (`ai/valid-neural-network.md`)
```markdown
# AI-0001-neural-network

- ontology:: AI
  term-id:: AI-0001
  preferred-term:: Neural Network
  definition:: A computational model...
  source-domain:: ai:machine-learning:deep-learning
  status:: complete
  maturity:: established
  public-access:: true
  version:: 2.1.0
  last-updated:: 2025-11-20
  owl:class:: NeuralNetwork
  owl:physicality:: ConceptualEntity
  owl:role:: Process
  algorithm-type:: supervised, unsupervised
  computational-complexity:: O(n*m*k)
  ...
```

#### Invalid File Example (`rb/invalid-namespace-mismatch.md`)
```markdown
# RB-0002-namespace-error

- ontology:: Robotics
  term-id:: RB-0002
  preferred-term:: Namespace Mismatch Robot
  source-domain:: mv:autonomous-systems   # ERROR: Should be rb:
  ...
```

## Test Suites

### 1. Python Converters Test Suite

**File**: `tests/integration/test_python_tools.py`

**Tests 10 Python converters:**
1. `convert-to-csv.py`
2. `convert-to-cypher.py`
3. `convert-to-jsonld.py`
4. `convert-to-skos.py`
5. `convert-to-sql.py`
6. `convert-to-turtle.py`
7. `generate_page_api.py`
8. `generate_search_index.py`
9. `ttl_to_webvowl_json.py`
10. `webvowl_header_only_converter.py`

**What it tests:**
- ✓ All converters process valid files correctly
- ✓ Output format is correct (CSV, Cypher, JSON-LD, SKOS, SQL, TTL)
- ✓ All 6 domains are handled
- ✓ IRI handling is correct (no Logseq [[links]])
- ✓ Error handling on invalid files
- ✓ Edge cases are handled properly

**Run command:**
```bash
python3 tests/integration/test_python_tools.py
```

**Report**: `tests/integration/reports/python-converters-report.json`

### 2. Rust Tools Test Suite

**File**: `tests/integration/test_rust_tools.rs`

**Tests Rust tools:**
1. Audit tool (`Ontology-Tools/tools/audit/`)
2. WASM parser (`publishing-tools/WasmVOWL/rust-wasm/`)

**What it tests:**
- ✓ Audit tool validates valid files
- ✓ Audit tool detects invalid files
- ✓ OWL2 compliance checking
- ✓ Namespace validation (detects mv: vs rb: errors)
- ✓ All 6 domains are checked
- ✓ Edge cases are handled
- ✓ WASM module accessibility

**Run command:**
```bash
# First, copy test to Rust project
cp tests/integration/test_rust_tools.rs Ontology-Tools/tools/audit/tests/

# Then run
cd Ontology-Tools/tools/audit
cargo test --test test_rust_tools -- --nocapture
```

**Note**: Requires Rust/Cargo to be installed and audit tool to be built.

### 3. JavaScript Pipeline Test Suite

**File**: `tests/integration/test_js_pipeline.js`

**Tests JS migration pipeline:**
1. Scanner (`scanner.js`)
2. Parser (`parser.js`)
3. Generator (`generator.js`)
4. Validator (`validator.js`)
5. IRI Registry (`iri-registry.js`)
6. Domain Detector (`domain-detector.js`)

**What it tests:**
- ✓ Scanner finds and classifies files
- ✓ Parser extracts ontology blocks
- ✓ Parser handles all property types
- ✓ Generator creates canonical format
- ✓ Generator fixes namespace errors
- ✓ Generator normalizes status/maturity
- ✓ Validator scores ontologies correctly
- ✓ Validator detects invalid files
- ✓ IRI registry resolves domain IRIs
- ✓ IRI registry converts Logseq links
- ✓ Domain detector classifies correctly
- ✓ Single ontology block enforcement
- ✓ End-to-end pipeline works

**Run command:**
```bash
node tests/integration/test_js_pipeline.js
```

**Report**: `tests/integration/reports/javascript-pipeline-report.json`

### 4. Cross-Tool Interoperability Test Suite

**File**: `tests/integration/test_interoperability.py`

**Tests tool integration:**

**What it tests:**
- ✓ JS pipeline generates files → Python converters process them
- ✓ Python converters generate TTL → WASM can visualize it
- ✓ Audit tool validates → Converters respect rules
- ✓ IRI consistency across all tools
- ✓ Complete end-to-end flow through all tools

**Run command:**
```bash
python3 tests/integration/test_interoperability.py
```

**Report**: `tests/integration/reports/interoperability-report.json`

## Running Tests

### Quick Start

Run all tests with the master script:

```bash
cd /home/user/logseq
./tests/run_all_tests.sh
```

This will:
1. Check dependencies
2. Run Python converter tests
3. Run JavaScript pipeline tests
4. Run Rust tool tests (if available)
5. Run interoperability tests
6. Generate consolidated report

### Run Individual Test Suites

**Python Converters:**
```bash
cd /home/user/logseq/tests/integration
python3 test_python_tools.py -v
```

**JavaScript Pipeline:**
```bash
cd /home/user/logseq/tests/integration
node test_js_pipeline.js
```

**Rust Tools:**
```bash
cp tests/integration/test_rust_tools.rs Ontology-Tools/tools/audit/tests/
cd Ontology-Tools/tools/audit
cargo test --test test_rust_tools -- --nocapture
```

**Interoperability:**
```bash
cd /home/user/logseq/tests/integration
python3 test_interoperability.py -v
```

### Run Specific Tests

**Python - specific test:**
```bash
python3 test_python_tools.py TestPythonConverters.test_01_turtle_converter
```

**JavaScript - run with debug output:**
```bash
DEBUG=* node test_js_pipeline.js
```

## Test Reports

All test suites generate JSON reports in `tests/integration/reports/`:

### Python Converters Report
**File**: `python-converters-report.json`

```json
{
  "test_suite": "Python Converters Integration Tests",
  "timestamp": "2025-11-21T...",
  "summary": {
    "total_tests": 52,
    "passed": 48,
    "failed": 2,
    "warnings": 2
  },
  "errors": [...],
  "warnings": [...],
  "converters_tested": [...]
}
```

### JavaScript Pipeline Report
**File**: `javascript-pipeline-report.json`

```json
{
  "testSuite": "JavaScript Pipeline Integration Tests",
  "timestamp": "2025-11-21T...",
  "summary": {
    "totalTests": 35,
    "passed": 33,
    "failed": 1,
    "warnings": 1,
    "successRate": 94
  },
  "testDetails": [...]
}
```

### Interoperability Report
**File**: `interoperability-report.json`

```json
{
  "test_suite": "Cross-Tool Interoperability Tests",
  "timestamp": "2025-11-21T...",
  "summary": {
    "total_tests": 15,
    "passed": 13,
    "failed": 1,
    "warnings": 1
  },
  "tools_tested": ["JS Pipeline", "Python Converters", "Rust Audit", "WASM Parser"]
}
```

### Consolidated Report
**File**: `consolidated-report.json`

```json
{
  "testRun": {
    "timestamp": "2025-11-21T...",
    "projectRoot": "/home/user/logseq"
  },
  "summary": {
    "totalSuites": 4,
    "passedSuites": 3,
    "failedSuites": 1,
    "successRate": 75.0
  },
  "suites": {
    "pythonConverters": "completed",
    "javascriptPipeline": "completed",
    "rustTools": "skipped",
    "interoperability": "completed"
  }
}
```

## Prerequisites

### Required Dependencies

1. **Python 3** (3.7+)
   ```bash
   python3 --version
   ```

2. **Node.js** (14+)
   ```bash
   node --version
   ```

3. **Rust/Cargo** (optional, for Rust tests)
   ```bash
   cargo --version
   ```

### Build Required Tools

**Build Audit Tool:**
```bash
cd /home/user/logseq/Ontology-Tools/tools/audit
cargo build --release
```

**Build WASM Module:**
```bash
cd /home/user/logseq/publishing-tools/WasmVOWL/rust-wasm
wasm-pack build
```

## Interpreting Results

### Success Indicators

✓ **All tests passed**: All tools working correctly
✓ **90%+ success rate**: Excellent, minor issues only
✓ **80-90% success rate**: Good, some edge cases need work

### Warning Indicators

⚠ **70-80% success rate**: Acceptable, but improvements needed
⚠ **Warnings present**: Non-critical issues detected
⚠ **Tests skipped**: Missing dependencies or tools not built

### Failure Indicators

✗ **Below 70% success rate**: Critical issues need attention
✗ **Multiple failures**: Core functionality broken
✗ **Errors in interoperability**: Tools not working together

## Troubleshooting

### Common Issues

**Issue**: Tests not finding modules
```bash
# Solution: Run from correct directory
cd /home/user/logseq
./tests/run_all_tests.sh
```

**Issue**: Rust tests fail - binary not found
```bash
# Solution: Build audit tool first
cd Ontology-Tools/tools/audit
cargo build --release
```

**Issue**: WASM tests skipped
```bash
# Solution: Build WASM module
cd publishing-tools/WasmVOWL/rust-wasm
wasm-pack build
```

**Issue**: Python import errors
```bash
# Solution: Ensure you're in the right directory
cd /home/user/logseq/tests/integration
python3 test_python_tools.py
```

**Issue**: Node module not found
```bash
# Solution: Install dependencies
cd scripts/ontology-migration
npm install
```

## Continuous Integration

### Adding to CI/CD

Example GitHub Actions workflow:

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - uses: actions/setup-node@v2
        with:
          node-version: '16'
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Build tools
        run: |
          cd Ontology-Tools/tools/audit
          cargo build --release
      - name: Run tests
        run: ./tests/run_all_tests.sh
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: tests/integration/reports/
```

## Extending the Test Suite

### Adding New Test Data

1. Create new test file in appropriate domain directory
2. Follow naming convention: `{type}-{description}.md`
3. Ensure it tests a specific scenario

### Adding New Tests

**Python:**
```python
def test_09_my_new_test(self):
    """Test description."""
    print("\n=== Testing My Feature ===")
    # Test implementation
    self.results["passed"] += 1
```

**JavaScript:**
```javascript
function testMyNewFeature() {
    console.log('\n=== Testing My Feature ===');
    runTest('My Feature: Description', () => {
        // Test implementation
        return true;
    });
}
```

### Adding New Converters/Tools

When adding new tools:

1. Add test data if needed
2. Create test cases in appropriate suite
3. Update this documentation
4. Run full test suite to verify

## Best Practices

1. **Run tests before committing changes**
2. **Add tests for new features**
3. **Keep test data realistic**
4. **Document expected behavior**
5. **Fix failing tests immediately**
6. **Review test reports regularly**
7. **Update tests when ontology schema changes**

## Test Statistics

### Current Test Coverage

- **Domains Tested**: 6 (AI, MV, TC, RB, DT, BC)
- **Test Files**: 18 (3 per domain)
- **Python Converters**: 10 tools tested
- **JavaScript Modules**: 6 modules tested
- **Rust Tools**: 2 tools tested
- **Total Test Cases**: ~100+
- **Interoperability Scenarios**: 5

### Expected Test Counts

| Suite | Test Cases | Expected Duration |
|-------|-----------|-------------------|
| Python Converters | 50-60 | 2-3 minutes |
| JavaScript Pipeline | 30-40 | 1-2 minutes |
| Rust Tools | 10-15 | 1 minute |
| Interoperability | 10-15 | 2-3 minutes |
| **Total** | **100-130** | **6-9 minutes** |

## Maintenance

### Regular Tasks

- **Weekly**: Run full test suite
- **Before releases**: Run tests + review reports
- **After schema changes**: Update test data
- **After tool changes**: Add/update tests

### Version Compatibility

Tests are designed to be compatible with:
- Python 3.7+
- Node.js 14+
- Rust 1.60+

## References

- [Project README](/home/user/logseq/README.md)
- [Ontology Migration Pipeline](/home/user/logseq/scripts/ontology-migration/README.md)
- [Python Converters](/home/user/logseq/Ontology-Tools/tools/README.md)
- [Domain Configuration](/home/user/logseq/scripts/ontology-migration/domain-config.json)

---

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Maintainer**: Claude Code Agent
**Status**: Complete and Ready for Use
