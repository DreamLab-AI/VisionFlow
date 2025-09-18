# Integration Test Suite

This directory contains comprehensive integration tests for the MCP service fixes and enhancements.

## Test Suites

### 1. TCP Persistence Tests (`tcp_persistence_test.py`)
Tests TCP connection stability and persistence:
- Basic connection establishment
- Connection persistence over time
- Idle connection handling
- Reconnection after disconnect
- Multiple concurrent clients
- Large payload handling
- Timeout behavior
- Async connection patterns

**Key Fixes Tested:**
- TCP connection pooling and reuse
- Connection timeout management
- Graceful disconnect handling
- Multi-client support

### 2. GPU Stability Tests (`gpu_stability_test.py`)
Tests GPU container stability and CUDA operations:
- GPU container status verification
- CUDA availability and configuration
- Basic CUDA operations
- GPU memory stability
- Concurrent GPU operations
- Error recovery mechanisms
- GUI tools connectivity
- Performance persistence

**Key Fixes Tested:**
- GPU container integration
- CUDA error handling
- Memory leak prevention
- Container health monitoring

### 3. Client Polling Tests (`client_polling_test.py`)
Tests client polling behavior and reconnection logic:
- WebSocket polling patterns
- HTTP long-polling
- Reconnection with backoff
- Concurrent client polling
- Connection drop handling
- Rate limiting compliance
- Graceful shutdown
- Async polling patterns

**Key Fixes Tested:**
- Client reconnection logic
- Polling interval optimization
- Connection stability
- Resource cleanup

### 4. Security Validation Tests (`security_validation_test.py`)
Tests security measures and attack prevention:
- SQL injection prevention
- XSS attack prevention
- Command injection prevention
- Path traversal prevention
- Rate limiting enforcement
- Authentication requirements
- Input validation
- DoS protection
- Secret exposure prevention
- CORS configuration

**Key Fixes Tested:**
- Input sanitization
- Authentication mechanisms
- Rate limiting implementation
- Security header configuration

## Running Tests

### Quick Start
```bash
# Make script executable (if not already)
chmod +x run_tests.sh

# Run all tests
./run_tests.sh

# Run specific test suite
./run_tests.sh tcp
./run_tests.sh gpu
./run_tests.sh polling
./run_tests.sh security
```

### Setup and Dependencies
```bash
# Install dependencies and check services
./run_tests.sh setup
```

### Generate Reports
```bash
# Generate comprehensive test report
./run_tests.sh report
```

### Manual Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run specific test
python -m pytest tcp_persistence_test.py -v

# Run all tests with coverage
python -m pytest . -v --tb=short

# Generate detailed report
python test_runner.py
```

## Test Results and Reports

Test results are automatically saved in multiple formats:

- **JSON Results**: `test_results_YYYYMMDD_HHMMSS.json` - Detailed machine-readable results
- **Markdown Report**: `test_report_YYYYMMDD_HHMMSS.md` - Human-readable report
- **Latest Report**: `latest_test_report.md` - Always contains the most recent results

### Report Contents
- Executive summary with pass/fail counts
- Detailed results for each test suite
- Performance analysis
- Issue identification and recommendations
- Test execution metrics

## Prerequisites

### Services Required
- TCP server running on port 9500
- WebSocket bridge on port 3002
- Health check endpoint on port 9501
- GPU container (mcp-gui-tools) for GPU tests

### Dependencies
- Python 3.8+
- pytest framework
- requests library
- websocket-client
- Docker (for GPU container tests)

### Environment Setup
```bash
# Check if services are running
nc -z localhost 9500  # TCP server
nc -z localhost 3002  # WebSocket bridge
curl http://localhost:9501/health  # Health check

# Check GPU container
docker ps --filter "name=mcp-gui-tools"
```

## Test Configuration

### Timeouts
- Individual test timeout: 30 seconds
- Test suite timeout: 5 minutes
- Connection timeout: 10 seconds
- Long polling timeout: 25 seconds

### Test Data
- Large payload size: 10MB (configurable)
- Concurrent client count: 3-5 (configurable)
- Rate limiting threshold: 20 requests/minute
- Memory monitoring duration: 10 seconds

### Security Test Payloads
The security tests include comprehensive attack vectors:
- SQL injection patterns
- XSS payloads
- Command injection attempts
- Path traversal sequences
- DoS simulation patterns

## Troubleshooting

### Common Issues

1. **Service Not Available**
   - Ensure all required services are running
   - Check port availability with `netstat -tulpn`
   - Verify Docker containers are running

2. **GPU Tests Failing**
   - Confirm GPU container is running: `docker ps | grep mcp-gui-tools`
   - Check CUDA availability: `docker exec mcp-gui-tools nvidia-smi`
   - Verify GPU drivers are properly installed

3. **Connection Tests Failing**
   - Check if ports are blocked by firewall
   - Verify network connectivity
   - Review service logs for errors

4. **Permission Errors**
   - Ensure test script is executable: `chmod +x run_tests.sh`
   - Check Docker permissions for GPU container access
   - Verify file system permissions

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
./run_tests.sh
```

### Manual Service Checks
```bash
# TCP server health
telnet localhost 9500

# WebSocket connectivity
wscat -c ws://localhost:3002

# HTTP health check
curl -v http://localhost:9501/health

# GPU container status
docker exec mcp-gui-tools nvidia-smi
```

## Test Coverage

The integration tests provide comprehensive coverage of:

### Functional Areas
- ✅ Network connectivity and persistence
- ✅ GPU container integration
- ✅ Client communication patterns
- ✅ Security and authentication
- ✅ Error handling and recovery
- ✅ Performance and stability
- ✅ Concurrent operations
- ✅ Resource management

### Fix Validation
- ✅ TCP connection pooling improvements
- ✅ GPU stability enhancements
- ✅ Client polling optimizations
- ✅ Security hardening measures
- ✅ Error recovery mechanisms
- ✅ Resource leak prevention
- ✅ Performance optimizations
- ✅ Monitoring and health checks

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Include comprehensive error handling
3. Add appropriate timeouts
4. Document test purpose and coverage
5. Update this README with new test descriptions
6. Ensure tests are deterministic and repeatable

### Test Naming Convention
- Test files: `*_test.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Fixtures: Use descriptive names

### Test Documentation
Each test should include:
- Docstring describing the test purpose
- Clear assertions with meaningful messages
- Proper setup and teardown
- Error message logging for debugging

## Integration with CI/CD

These tests are designed to be easily integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions integration
- name: Run Integration Tests
  run: |
    cd tests/integration
    ./run_tests.sh all
    
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: integration-test-results
    path: tests/integration/latest_test_report.md
```