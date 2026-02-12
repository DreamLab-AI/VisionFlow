---
title: VisionFlow Testing Guide
description: Comprehensive guide to running and writing tests for the Rust backend, React frontend, integration tests, ontology agent tests, and WebSocket testing.
category: how-to
tags:
  - testing
  - development
  - rust
  - react
  - integration
updated-date: 2026-02-12
difficulty-level: intermediate
---

# VisionFlow Testing Guide

This guide covers the full testing strategy for VisionFlow: Rust unit and integration tests, React component and Vitest tests, ontology reasoning tests, and WebSocket testing approaches.

## Test Infrastructure Overview

| Layer | Tool | Command | Location |
|-------|------|---------|----------|
| Rust unit tests | `cargo test` | `cargo test --lib` | `src/` (inline `#[cfg(test)]` modules) |
| Rust integration tests | `cargo test` | `cargo test --test '*'` | `tests/` |
| React unit tests | Vitest + Testing Library | `npm test` (in `client/`) | `client/src/**/*.test.ts(x)` |
| React E2E tests | Playwright | `npx playwright test` | `client/tests/` |
| API endpoint tests | cargo test / curl | `cargo test api` | `tests/api/`, `tests/api_validation_tests.rs` |
| GPU tests | cargo test (feature-gated) | `cargo test --features gpu` | `tests/gpu_*.rs` |

## Rust Backend Tests

### Running All Rust Tests

```bash
# From the project root (or inside the container)
cargo test

# Run with output visible (useful for debugging)
cargo test -- --nocapture

# Run a specific test file
cargo test --test ontology_smoke_test

# Run tests matching a pattern
cargo test settings_validation
```

### Unit Tests

Unit tests live alongside the code in `src/` using `#[cfg(test)]` modules. They test individual functions and types in isolation.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = GraphNode::new("test-id", "Test Label");
        assert_eq!(node.id, "test-id");
    }

    #[tokio::test]
    async fn test_async_handler() {
        // Async test using tokio runtime
    }
}
```

Dev dependencies used for testing:
- `tokio-test` -- Async test utilities
- `mockall` -- Mock trait implementations
- `pretty_assertions` -- Readable diff output for assertion failures
- `tempfile` -- Temporary directories for file-based tests
- `actix-rt` -- Actix runtime for handler tests

### Integration Tests

Integration tests live in the top-level `tests/` directory. Key test files:

| File | What It Tests |
|------|--------------|
| `ontology_smoke_test.rs` | Basic ontology loading and parsing |
| `ontology_agent_integration_test.rs` | Ontology agent actor lifecycle |
| `ontology_reasoning_integration_test.rs` | Whelk reasoning engine integration |
| `neo4j_settings_integration_tests.rs` | Settings persistence to Neo4j |
| `settings_validation_tests.rs` | Settings schema validation |
| `voice_agent_integration_test.rs` | Voice pipeline agent integration |
| `high_perf_networking_tests.rs` | QUIC/WebTransport protocol tests |
| `gpu_safety_tests.rs` | GPU memory management and fallback |
| `mcp_parsing_tests.rs` | MCP message parsing and relay |

### Running Tests Inside Docker

```bash
# Shell into the dev container
docker exec -it visionflow_container bash

# Run Rust tests
cargo test

# Run with specific log level
RUST_LOG=debug cargo test -- --nocapture
```

## React Frontend Tests

### Running Frontend Tests

```bash
cd client

# Run all tests once
npm test

# Watch mode (re-runs on file changes)
npm run test:watch

# With coverage report
npm run test:coverage

# Interactive UI
npm run test:ui
```

### Vitest Configuration

The frontend uses Vitest (configured in `client/vitest.config.ts`) with jsdom for DOM simulation and `@testing-library/react` for component testing.

```typescript
// Example component test
import { render, screen } from '@testing-library/react';
import { SettingsPanel } from './SettingsPanel';

describe('SettingsPanel', () => {
  it('renders physics controls', () => {
    render(<SettingsPanel />);
    expect(screen.getByText('Physics')).toBeInTheDocument();
  });
});
```

### Playwright E2E Tests

End-to-end tests use Playwright (configured in `client/playwright.config.ts`):

```bash
cd client
npx playwright test

# Run with browser visible
npx playwright test --headed

# Generate test report
npx playwright show-report
```

## Ontology Agent Tests

The ontology subsystem has dedicated tests for the Whelk reasoning engine and OWL parsing:

```bash
# Smoke test for ontology loading
cargo test --test ontology_smoke_test

# Full reasoning integration
cargo test --test ontology_reasoning_integration_test

# Schema compliance
cargo test --test test_ontology_schema_fixes

# Ontology constraint validation
cargo test --test ontology_constraints_gpu_test
```

These tests verify:
- OWL file parsing via `horned-owl`
- Whelk subsumption reasoning over the VisionFlow ontology
- Ontology-driven constraint application to graph physics
- Actor lifecycle for the ontology agent

## WebSocket Testing

VisionFlow uses WebSockets extensively for real-time graph updates, voice, and MCP relay. Testing approaches:

### Manual WebSocket Testing with wscat

```bash
# Install wscat (available in the client dev dependencies)
npx wscat -c ws://localhost:3001/wss

# Send a graph subscription message
> {"type":"subscribe","channel":"graph"}
```

### Programmatic WebSocket Tests

```bash
# Run the WebSocket rate limit test
cargo test --test test_websocket_rate_limit

# Run the wire format test
cargo test --test test_wire_format
```

### WebSocket Test Patterns

For testing WebSocket handlers in Rust:

```rust
#[actix_rt::test]
async fn test_ws_connection() {
    let srv = actix_test::start(|| {
        App::new().route("/wss", web::get().to(ws_handler))
    });
    let mut ws = srv.ws_at("/wss").await.unwrap();
    ws.send(Message::Text("ping".into())).await.unwrap();
    let response = ws.next().await.unwrap().unwrap();
    assert!(matches!(response, Frame::Text(_)));
}
```

## GPU Tests

GPU tests require an NVIDIA GPU and CUDA runtime. They are feature-gated:

```bash
# Run GPU-specific tests
cargo test --features gpu -- gpu

# GPU memory manager tests
cargo test --test gpu_memory_manager_test

# GPU safety and fallback tests
cargo test --test gpu_safety_tests
```

See `tests/README_GPU_TESTS.md` for hardware requirements and skip conditions.

## Test Organization Conventions

1. **File naming:** Test files use `snake_case` with a `_test.rs` suffix (e.g., `ontology_smoke_test.rs`).
2. **Test naming:** Test functions use `test_` prefix with descriptive names.
3. **Fixtures:** Shared test data lives in `tests/fixtures/`.
4. **Test utilities:** Common helpers are in `tests/test_utils.rs` and `src/test_helpers.rs`.
5. **Feature gates:** GPU and ontology tests use `#[cfg(feature = "gpu")]` / `#[cfg(feature = "ontology")]`.

## Continuous Integration

Tests should pass before any PR merge. Run the full suite:

```bash
# Rust (from project root)
cargo test --all-features

# Frontend (from client/)
cd client && npm test

# Linting
cd client && npm run lint
cargo clippy --all-features -- -D warnings
```

## See Also

- [Contributing Guide](./contributing.md) -- Code contribution workflow and conventions
- [Development Setup](./01-development-setup.md) -- Setting up a development environment
- [Docker Environment Setup](../deployment/docker-environment-setup.md) -- Running tests inside Docker
- `tests/README.md` -- Test directory index
- `tests/README_GPU_TESTS.md` -- GPU test prerequisites
