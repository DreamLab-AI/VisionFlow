# Comprehensive Test Framework for CachyOS and Z.AI Docker System

Complete testing suite with unit, integration, E2E, performance, load, and security tests.

## Quick Start

```bash
# Install dependencies
cd docker/cachyos/tests
npm install

# Start Docker containers
cd ..
./start-agentic-flow.sh

# Run all tests
cd tests
./run-tests.sh
```

## Test Categories

### 1. Unit Tests (`unit/`)
Tests for individual components and utilities.

```bash
npm run test:unit
```

**Coverage:**
- Prometheus metrics collection
- Utility functions
- Helper modules

**Files:**
- `unit/metrics.test.js` - Metrics recording and aggregation

### 2. Integration Tests (`integration/`)
Tests for service integration and API endpoints.

```bash
npm run test:integration
```

**Coverage:**
- Management API endpoints
- Z.AI service communication
- MCP tool integration
- Authentication & authorization
- Rate limiting
- CORS

**Files:**
- `integration/api.test.js` - Management API tests
- `integration/zai-service.test.js` - Z.AI service tests
- `integration/mcp-tools.test.js` - MCP tools tests

### 3. End-to-End Tests (`e2e/`)
Full workflow tests simulating real usage scenarios.

```bash
npm run test:e2e
```

**Coverage:**
- Complete web summarization workflow
- MCP tool lifecycle management
- Monitoring and observability
- Error handling and resilience
- Network integration
- Data persistence

**Files:**
- `e2e/full-workflow.test.js` - Complete workflows

### 4. Performance Tests (`performance/`)
Benchmarks for latency and throughput.

```bash
npm run test:performance
```

**Metrics:**
- Request latency (mean, p50, p95, p99)
- Throughput (ops/sec)
- Concurrent request handling
- Memory footprint

**Files:**
- `performance/benchmark.js` - Performance benchmarks

### 5. Load Tests (`load-tests/`)
Artillery-based load testing.

```bash
npm run test:load
```

**Scenarios:**
- Health check load
- Metrics scraping
- Authenticated API calls
- Documentation access

**Files:**
- `load-tests/api-load-test.yml` - Artillery configuration

### 6. Security Tests
Dependency and vulnerability scanning.

```bash
npm run test:security
```

**Tools:**
- npm audit
- Snyk (if installed)

## Test Runner

The `run-tests.sh` script provides a comprehensive test orchestration:

```bash
# Run all tests
./run-tests.sh

# Run specific suites
./run-tests.sh --unit
./run-tests.sh --integration --e2e
./run-tests.sh --performance

# Generate coverage
./run-tests.sh --coverage

# Skip Docker checks (for CI)
./run-tests.sh --skip-docker
```

### Options:
- `--all` - Run all tests (default)
- `--unit` - Unit tests only
- `--integration` - Integration tests only
- `--e2e` - End-to-end tests only
- `--performance` - Performance benchmarks only
- `--load` - Load tests only
- `--security` - Security tests only
- `--coverage` - Generate coverage report
- `--skip-setup` - Skip dependency installation
- `--skip-docker` - Skip Docker container checks

## CI/CD Integration

GitHub Actions workflow at `.github/workflows/test.yml`:

```yaml
jobs:
  - unit-tests
  - integration-tests
  - e2e-tests
  - performance-tests
  - security-scan
  - docker-build
```

### Required Secrets:
- `GOOGLE_API_KEY` - For web summary tests
- `ZAI_API_KEY` - For Z.AI tests
- `SNYK_TOKEN` - For security scanning (optional)

## Coverage Reports

Coverage reports are generated in `coverage/`:

```bash
# Generate and view coverage
npm run coverage

# View report
open coverage/lcov-report/index.html
```

### Coverage Thresholds:
- Branches: 80%
- Functions: 80%
- Lines: 80%
- Statements: 80%

## Writing Tests

### Unit Test Example:

```javascript
const { describe, it, expect } = require('@jest/globals');

describe('My Module', () => {
  it('should do something', () => {
    expect(true).toBe(true);
  });
});
```

### Integration Test Example:

```javascript
const axios = require('axios');

describe('API Integration', () => {
  it('should return data', async () => {
    const response = await axios.get('http://localhost:9090/api');
    expect(response.status).toBe(200);
  });
});
```

### E2E Test Example:

```javascript
describe('Complete Workflow', () => {
  it('should complete end-to-end', async () => {
    // Step 1: Setup
    // Step 2: Execute
    // Step 3: Verify
    // Step 4: Cleanup
  });
});
```

## Environment Variables

```bash
# API endpoints
API_BASE_URL=http://localhost:9090
ZAI_CONTAINER_URL=http://localhost:9600

# Authentication
MANAGEMENT_API_KEY=change-this-secret-key

# Container names
CONTAINER_NAME=agentic-flow-cachyos
ZAI_CONTAINER=claude-zai-service

# Test configuration
SKIP_DOCKER_CHECK=false
```

## Test Data

Test fixtures and data:
- Sample URLs for web summary tests
- Mock API responses
- Test MCP tool configurations

## Troubleshooting

### Tests failing to connect:

```bash
# Check containers are running
docker ps

# Check service health
curl http://localhost:9090/health
curl http://localhost:9600/health

# View logs
docker logs agentic-flow-cachyos
docker logs claude-zai-service
```

### Timeout errors:

Increase test timeout in `package.json`:

```json
{
  "jest": {
    "testTimeout": 60000
  }
}
```

### Coverage not generating:

```bash
# Clean and regenerate
rm -rf coverage
npm run test:unit -- --coverage
```

## Performance Benchmarks

Example output:

```
📊 Benchmark: Health Check Latency
Running 100 iterations...

Results for Health Check Latency:
  Mean:    12.34ms
  Min:     8.12ms
  Max:     45.67ms
  P50:     11.23ms
  P95:     18.90ms
  P99:     32.11ms
  Ops/sec: 81.04
```

## Load Test Results

Example Artillery output:

```
Summary report @ 15:23:45
  Scenarios launched:  1000
  Scenarios completed: 1000
  Requests completed:  1000
  Mean response time:  23.4 ms
  Response time (95th percentile): 45.2 ms
  Response codes:
    200: 1000
```

## Continuous Integration

Tests run automatically on:
- Push to main/develop
- Pull requests
- Daily schedule (2 AM UTC)

View results: GitHub Actions tab

## Dependencies

### Runtime:
- axios - HTTP client
- dockerode - Docker API

### Development:
- jest - Test framework
- supertest - HTTP testing
- artillery - Load testing
- eslint - Linting
- testcontainers - Container testing

## Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Always cleanup resources
3. **Mocking**: Mock external services when appropriate
4. **Assertions**: Use specific, meaningful assertions
5. **Coverage**: Aim for >80% coverage
6. **Performance**: Keep tests fast (<30s)
7. **Documentation**: Document complex test scenarios

## Contributing

When adding tests:

1. Place in appropriate directory (unit/integration/e2e)
2. Follow naming convention: `*.test.js`
3. Add to test runner if needed
4. Update this README
5. Ensure CI passes

## Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Artillery Documentation](https://www.artillery.io/docs)
- [Docker Testing Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions](https://docs.github.com/en/actions)

## Support

For issues or questions:
1. Check test logs: `./run-tests.sh --help`
2. View service logs: `docker logs <container>`
3. Check CI results: GitHub Actions tab
4. Open issue: GitHub Issues
