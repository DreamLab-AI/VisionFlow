# Settings Sync Integration Tests

## Overview

This document describes the comprehensive integration test suite for the settings synchronization functionality, with a particular focus on the robustness of the REST API and proper handling of bloom/glow field validation.

## Test Coverage

The test suite covers the complete settings sync flow:

### 1. REST API Endpoints
- GET `/api/settings` - Fetch current settings with bloom fields
- POST `/api/settings` - Update settings with validation
- POST `/api/settings/reset` - Reset to defaults
- GET `/api/settings/validation/stats` - Validation service statistics
- POST `/api/physics/update` - Physics-specific updates
- POST `/api/clustering/algorithm` - Clustering configuration
- POST `/api/stress/optimisation` - Stress optimisation parameters

### 2. Bloom/Glow Field Validation
- **Valid Cases**: Proper intensity, radius, colour formats, opacity ranges
- **Invalid Cases**: Negative values, invalid colours, out-of-range parameters
- **Edge Cases**: Boundary conditions, extreme values, malformed data

### 3. Physics Settings Propagation
- Validation of physics parameters (damping, iterations, spring constants)
- Auto-balance synchronization across both graphs (logseq and visionflow)
- GPU parameter validation and propagation

### 4. Bidirectional Sync
- Settings update → fetch verification
- Concurrent request handling
- Data consistency checks

### 5. Nostr Authentication Integration
- Settings persistence with authenticated users
- User-specific settings isolation
- Power user vs regular user feature access
- Session management and token validation

### 6. Security and Rate Limiting
- Rate limiting enforcement
- Input validation and sanitization
- Request size limits
- Malformed JSON handling

## Test Files

### Rust Backend Tests

#### `/tests/integration_settings_sync.rs`
Comprehensive integration tests for the Rust backend:

```rust
// Core functionality tests
- test_get_settings_endpoint()
- test_update_bloom_settings_valid()
- test_update_bloom_settings_invalid()
- test_physics_settings_propagation()
- test_rate_limiting()
- test_settings_reset()

// Authentication tests
- test_nostr_auth_flow()
- test_nostr_verify_token()

// Error handling tests
- test_malformed_json_handling()
- test_concurrent_settings_updates()

// Performance tests
- test_settings_response_time()
- test_large_settings_update()
```

#### `/tests/e2e-settings-validation.rs`
End-to-end validation focusing on the brittle bloom field handling:

```rust
// Comprehensive validation tests
- test_bloom_field_validation_comprehensive()
- test_physics_field_validation_comprehensive()
- test_settings_reset_robustness()

// Stress tests
- test_large_settings_payload()
- test_response_time_performance()
- test_concurrent_requests_handling()

// Edge cases
- test_malformed_json_handling()
- test_auto_balance_sync_robustness()
```

### TypeScript Frontend Tests

#### `/client/src/tests/settings-sync-integration.test.ts`
Client-side integration tests:

```typescript
// REST API endpoint tests
- should fetch settings with bloom fields
- should update settings with bloom fields
- should handle server validation errors
- should handle rate limiting

// Physics settings propagation
- should update physics settings via dedicated endpoint
- should validate physics parameters

// Bidirectional sync
- should maintain consistency between client and server
- should handle concurrent updates gracefully

// Error handling
- should handle network errors gracefully
- should handle malformed server responses
- should validate import/export functionality
```

#### `/client/src/tests/nostr-settings-integration.test.ts`
Nostr authentication + settings integration:

```typescript
// Authentication flow integration
- should sync settings after successful authentication
- should handle settings persistence across sessions
- should clear settings on logout

// User-specific settings isolation
- should handle different settings for different users
- should handle power user exclusive features

// Bloom/glow settings with authentication
- should sync bloom settings for authenticated power users
- should restrict advanced bloom features for regular users
- should validate bloom settings against user permissions
```

## Test Execution

### Running Rust Backend Tests

```bash
# Run all integration tests
cargo test --test integration_settings_sync

# Run end-to-end validation tests
cargo test --test e2e-settings-validation

# Run with output
cargo test --test integration_settings_sync -- --nocapture

# Run specific test
cargo test test_bloom_field_validation_comprehensive
```

### Running TypeScript Frontend Tests

```bash
# Install dependencies
cd client
npm install

# Run settings sync tests
npm run test -- settings-sync-integration.test.ts

# Run nostr integration tests
npm run test -- nostr-settings-integration.test.ts

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch
```

### Running End-to-End Shell Script

```bash
# Make executable (if not already)
chmod +x scripts/test-settings-sync.sh

# Run with default server (localhost:8080)
./scripts/test-settings-sync.sh

# Run with custom server URL
./scripts/test-settings-sync.sh --server-url http://localhost:3000

# View help
./scripts/test-settings-sync.sh --help
```

## Test Data and Scenarios

### Valid Bloom Settings Test Data

```json
{
  "visualisation": {
    "glow": {
      "enabled": true,
      "intensity": 2.5,
      "radius": 0.9,
      "threshold": 0.2,
      "diffuseStrength": 1.8,
      "atmosphericDensity": 0.9,
      "volumetricIntensity": 1.5,
      "baseColor": "#ff6b6b",
      "emissionColor": "#4ecdc4",
      "opacity": 0.95,
      "nodeGlowStrength": 3.5,
      "edgeGlowStrength": 4.0,
      "environmentGlowStrength": 3.2
    }
  }
}
```

### Invalid Test Cases

| Test Case | Invalid Data | Expected Error |
|-----------|-------------|----------------|
| Negative intensity | `"intensity": -1.0` | "intensity must be positive" |
| Invalid colour | `"baseColor": "not-a-colour"` | "invalid colour format" |
| Damping out of range | `"damping": 1.5` | "damping must be between 0.0 and 1.0" |
| Zero iterations | `"iterations": 0` | "iterations must be positive" |
| Extreme values | `"intensity": 999999.0` | "intensity out of valid range" |

### Physics Parameter Tests

```json
{
  "springK": 0.15,
  "repelK": 2.5,
  "attractionK": 0.02,
  "gravity": 0.0002,
  "damping": 0.9,
  "maxVelocity": 8.0,
  "dt": 0.02,
  "temperature": 0.02,
  "iterations": 75,
  "boundsSize": 1200.0,
  "separationRadius": 2.5
}
```

## Expected Test Results

### Success Criteria

1. **API Response Structure**: All endpoints return proper JSON structure with required fields
2. **Validation Enforcement**: Invalid inputs are rejected with appropriate error messages
3. **Data Consistency**: Updated settings are reflected in subsequent fetches
4. **Rate Limiting**: Excessive requests are throttled appropriately
5. **Authentication Flow**: Nostr auth integration works correctly
6. **Performance**: Response times under acceptable thresholds (<500ms for settings fetch)

### Performance Benchmarks

| Operation | Expected Time | Acceptable Range |
|-----------|---------------|------------------|
| GET settings | <100ms | <500ms |
| POST settings update | <200ms | <1000ms |
| Settings validation | <50ms | <200ms |
| Rate limit check | <10ms | <50ms |

## Debugging Failed Tests

### Common Issues

1. **Server Not Running**
   ```bash
   # Start the server first
   cargo run --bin server
   ```

2. **Port Conflicts**
   ```bash
   # Check if port 8080 is in use
   lsof -i :8080
   # Use different port if needed
   ./scripts/test-settings-sync.sh --server-url http://localhost:3001
   ```

3. **Missing Dependencies**
   ```bash
   # Install required tools
   sudo apt-get install curl jq  # Ubuntu/Debian
   brew install curl jq          # macOS
   ```

4. **Database Issues**
   ```bash
   # Reset test database if needed
   cargo run --bin reset-test-db
   ```

### Test Output Analysis

#### Successful Test Output
```
[INFO] Starting comprehensive settings sync integration tests
[INFO] Server URL: http://localhost:8080
[PASS] GET /api/settings
[PASS] POST /api/settings (valid bloom)
[PASS] POST /api/settings (invalid bloom)
[PASS] Bidirectional sync
[INFO] Total tests: 15
[PASS] Passed: 15
[PASS] All tests passed! Settings sync is working correctly.
```

#### Failed Test Output
```
[INFO] Running test: POST /api/settings (valid bloom)
[FAIL] POST /api/settings (valid bloom)
Expected status 200, got 400
Response: {"error": "Invalid settings: intensity must be between 0.0 and 10.0"}

[INFO] Test Summary:
[INFO] Total tests: 15
[PASS] Passed: 14
[FAIL] Failed: 1
[FAIL] Some tests failed. Please review the output above.
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Settings Sync Integration Tests

on: [push, pull_request]

jobs:
  test-settings-sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          
      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y curl jq
          
      - name: Build server
        run: cargo build --release
        
      - name: Install client dependencies
        run: cd client && npm ci
        
      - name: Start server in background
        run: |
          cargo run --bin server &
          sleep 10  # Wait for server to start
          
      - name: Run Rust integration tests
        run: |
          cargo test --test integration_settings_sync
          cargo test --test e2e-settings-validation
          
      - name: Run TypeScript tests
        run: cd client && npm run test
        
      - name: Run end-to-end shell tests
        run: ./scripts/test-settings-sync.sh
```

## Maintenance and Updates

### Adding New Tests

1. **New Bloom Fields**: When adding new bloom/glow parameters, update validation tests in both Rust and TypeScript
2. **New API Endpoints**: Add corresponding integration tests following the existing patterns
3. **New User Roles**: Update Nostr authentication tests to cover new permission levels

### Updating Test Data

1. Modify test data structures in the `validation_test_data` modules
2. Update expected responses when default settings change
3. Adjust performance benchmarks as the system evolves

### Test Environment Setup

```bash
# Create test environment script
#!/bin/bash
# setup-test-env.sh

# Install dependencies
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y curl jq python3
elif command -v brew &> /dev/null; then
    brew install curl jq python3
fi

# Setup Rust environment
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Setup Node.js environment
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install project dependencies
cd client && npm install && cd ..
cargo build

echo "Test environment setup complete!"
```

This comprehensive test suite ensures that the settings sync functionality is robust, secure, and reliable, with particular attention to the bloom field validation that was mentioned as brittle in the requirements.
