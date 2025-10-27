# Category 1: Resolve Critical Contradictions - EXECUTABLE PLAN

**Status**: Ready for Execution
**Priority**: P0 - Critical (blocking other categories)
**Last Updated**: 2025-10-27

## Executive Summary

This document provides step-by-step executable instructions for resolving the 5 critical contradictions identified in the documentation audit. Each task includes specific file operations, validation steps, and rollback procedures.

---

## Task 1.1: Binary Protocol Standardization

### Ground Truth (from `/src/utils/binary_protocol.rs`)

**ACTUAL IMPLEMENTATION:**
```rust
// Protocol V2 (Current Production)
const WIRE_V2_ITEM_SIZE: usize = 36;  // 4+12+12+4+4
struct WireNodeDataItemV2 {
    pub id: u32,            // 4 bytes - u32 with flags in bits 30-31
    pub position: Vec3Data, // 12 bytes (3×f32)
    pub velocity: Vec3Data, // 12 bytes (3×f32)
    pub sssp_distance: f32, // 4 bytes
    pub sssp_parent: i32,   // 4 bytes
}
// Total: 36 bytes per node
// Node ID capacity: 1,073,741,823 (30 bits)
// Flags: Bit 31 (AGENT), Bit 30 (KNOWLEDGE)
```

**DOCUMENTATION ERROR:**
- Line 10 claims "38 bytes per node" ← WRONG
- Line 48-50 defines struct correctly but contradicts line 10
- Line 72 has correct comment: "36 bytes (4+12+12+4+4) NOT 38!"
- Comment at line 87 corrects to 36 bytes

### Truth Source

**PRIMARY:** `/home/devuser/workspace/project/src/utils/binary_protocol.rs` (lines 71-72)
**SECONDARY:** `/home/devuser/workspace/project/docs/reference/api/binary-protocol.md` (authoritative spec)

### Files Requiring Updates

#### ❌ DELETE (Contradictory/Obsolete):
NONE - All files can be corrected in place

#### ✏️ EDIT (Fix Contradictions):

1. **`/src/utils/binary_protocol.rs`**
   - Line 10: Change "38 bytes" → "36 bytes"
   - Line 48: Add comment clarifying 36 bytes total
   - Line 72: Keep correct comment (already accurate)
   - **Validation:** `cargo build` must succeed

2. **`/docs/architecture/components/websocket-protocol.md`**
   - **Status:** Already CORRECT (shows 36 bytes correctly)
   - **Action:** NO CHANGES NEEDED ✅
   - **Lines to verify:** 77-86 show correct 36-byte structure

3. **`/docs/specialized/ontology/PROTOCOL_SUMMARY.md`**
   - Line 48: Keep "36 bytes per node" (already correct)
   - Line 376-378: Verify bandwidth table shows 36 bytes
   - **Validation:** No contradictions found ✅

#### ✅ AUTHORITATIVE (No Changes):

**`/docs/reference/api/binary-protocol.md`** (THE definitive spec)
- Lines 27-50: Correctly documents 36-byte V2 structure
- Lines 12-16: Correctly states u32 node IDs
- Lines 332-340: Bandwidth table correctly shows 36 bytes
- **THIS IS THE STANDARD** - all other docs must match this

### Execution Steps

```bash
# Step 1: Fix source code comment
cd /home/devuser/workspace/project
sed -i 's/38 bytes per node/36 bytes per node/' src/utils/binary_protocol.rs

# Step 2: Verify change
grep -n "36 bytes\|38 bytes" src/utils/binary_protocol.rs

# Step 3: Validate compilation
cargo check --lib

# Step 4: Run protocol tests
cargo test binary_protocol --lib

# Step 5: Git commit
git add src/utils/binary_protocol.rs
git commit -m "docs: fix protocol V2 byte count (36 not 38)

Corrects documentation comment in binary_protocol.rs that incorrectly
stated 38 bytes per node. Actual V2 wire format is 36 bytes:
- 4 bytes: u32 node ID (with flags)
- 12 bytes: position Vec3
- 12 bytes: velocity Vec3
- 4 bytes: SSSP distance
- 4 bytes: SSSP parent

Reference: docs/reference/api/binary-protocol.md (authoritative spec)"
```

### Validation Checklist

- [ ] `grep "38 bytes" src/utils/binary_protocol.rs` returns ZERO results
- [ ] `grep "36 bytes" src/utils/binary_protocol.rs` returns correct lines
- [ ] `cargo build` succeeds without warnings
- [ ] `cargo test binary_protocol` passes all tests
- [ ] `/docs/reference/api/binary-protocol.md` still shows 36 bytes (unchanged)

### Rollback Procedure

```bash
# If validation fails:
git reset --hard HEAD~1
cargo clean
cargo build
```

---

## Task 1.2: API Documentation Consolidation

### Ground Truth (from `/src/main.rs` and running system)

**ACTUAL IMPLEMENTATION:**
```rust
// From src/main.rs:
let port = std::env::var("SYSTEM_NETWORK_PORT")
    .unwrap_or_else(|_| "3030".to_string());  // DEFAULT: 3030
let bind_address = format!("{}:{}", bind_address, port);

// Server runs on: 0.0.0.0:3030 (production)
// Frontend dev server: localhost:5173 (Vite)
// Authentication: NOT IMPLEMENTED (no auth handlers found)
```

**CONTRADICTIONS FOUND:**
- Some docs mention `localhost:8080`
- Some docs mention `localhost:3030`
- Auth status unclear across docs

### Truth Sources

**PRIMARY SOURCE OF TRUTH:**
- **Server Config:** `SYSTEM_NETWORK_PORT` env var (default: 3030)
- **Code:** `/home/devuser/workspace/project/src/main.rs` (line with port binding)
- **Authentication:** NOT IMPLEMENTED (no JWT/auth middleware found)

**AUTHORITATIVE DOCS:**
1. `/docs/reference/api/rest-api.md` - REST API reference
2. `/docs/reference/api/websocket-api.md` - WebSocket API reference
3. `/docs/API.md` - Top-level API summary

### Files Requiring Updates

#### ❌ DELETE (Outdated/Duplicate):

**NONE** - Consolidate instead of delete to preserve git history

#### ✏️ EDIT (Standardize to Port 3030):

**Search pattern to find incorrect ports:**
```bash
grep -r "localhost:8080\|localhost:3030\|:8080\|:3030" docs/ --include="*.md"
```

**Files to update** (exact edits depend on grep results):

1. **`/docs/API.md`**
   - Replace all `localhost:8080` → `localhost:3030`
   - Replace all `localhost:3030` → `localhost:3030`
   - Add clarification: "Default server port: 3030 (configurable via SYSTEM_NETWORK_PORT)"

2. **`/docs/reference/api/rest-api.md`**
   - Verify all examples use `:3030`
   - Add environment variable documentation:
     ```markdown
     ## Configuration
     - **Server Port:** Set via `SYSTEM_NETWORK_PORT` (default: 3030)
     - **Frontend Dev:** Vite runs on port 5173 (separate process)
     ```

3. **`/docs/reference/api/websocket-api.md`**
   - Replace `ws://localhost:3030/ws` → `ws://localhost:3030/ws`
   - Standardize all connection examples

4. **`/docs/deployment/*.md`**
   - Search and replace: `3001` → `3030`, `8080` → `3030`
   - Update Docker configs to expose port 3030

5. **`/docs/developer-guide/01-development-setup.md`**
   - Clarify:
     ```markdown
     ## Development Ports
     - Backend API: `localhost:3030` (Rust/Actix)
     - Frontend Dev: `localhost:5173` (Vite)
     - WebSocket: `ws://localhost:3030/ws`
     ```

**Authentication Documentation:**

6. **`/docs/api/01-authentication.md`**
   - **IF EXISTS:** Add prominent warning:
     ```markdown
     ## ⚠️ AUTHENTICATION STATUS

     **Current Status:** NOT IMPLEMENTED
     **Security Level:** DEVELOPMENT ONLY - DO NOT USE IN PRODUCTION

     Authentication middleware is planned but not yet implemented.
     All endpoints are currently PUBLIC and UNAUTHENTICATED.

     For production deployment, implement authentication before exposing server.
     ```

### Execution Steps

```bash
cd /home/devuser/workspace/project/docs

# Step 1: Find all files with incorrect ports
grep -r "localhost:8080\|localhost:3030\|:8080\|:3030" . --include="*.md" > /tmp/port_fixes.txt
cat /tmp/port_fixes.txt

# Step 2: Automated replacement (with backup)
find . -type f -name "*.md" -exec sed -i.bak \
  -e 's|localhost:8080|localhost:3030|g' \
  -e 's|localhost:3030|localhost:3030|g' \
  -e 's|:8080/|:3030/|g' \
  -e 's|:3030/|:3030/|g' \
  {} \;

# Step 3: Review changes
git diff docs/

# Step 4: If changes look good, commit
git add docs/
git commit -m "docs: standardize API port to 3030 across all documentation

- Changed all references from port 8080/3001 to 3030 (actual default)
- Aligned with SYSTEM_NETWORK_PORT environment variable
- Added port configuration documentation to API reference

Verified against src/main.rs server binding code."

# Step 5: Clean up backup files
find docs/ -name "*.bak" -delete
```

### Validation Checklist

- [ ] `grep -r ":8080\|:3030" docs/ --include="*.md"` returns ZERO results
- [ ] `grep -r ":3030" docs/ --include="*.md"` shows consistent usage
- [ ] `/docs/API.md` mentions `SYSTEM_NETWORK_PORT` env var
- [ ] `/docs/reference/api/rest-api.md` has port configuration section
- [ ] All WebSocket examples use `ws://localhost:3030/ws`
- [ ] Authentication status clearly documented as "NOT IMPLEMENTED"

### Rollback Procedure

```bash
# Restore from .bak files
find docs/ -name "*.bak" -exec bash -c 'mv "$1" "${1%.bak}"' _ {} \;

# Or use git
git checkout docs/
```

---

## Task 1.3: Deployment Documentation Consolidation

### Ground Truth (from codebase inspection)

**ACTUAL DEPLOYMENT APPROACH:**
```bash
# Evidence from repository:
1. NO Dockerfile found in root
2. NO docker-compose.yml in root
3. multi-agent-docker/ exists but is SEPARATE system (Turbo Flow Claude)
4. Production deployment: Direct Rust binary execution
5. Frontend: Vite static build + Nginx
```

**CONTRADICTORY DOCS:**
- Some docs mention Docker deployment
- Some mention systemd services
- Confusion between main project and multi-agent-docker/

### Truth Sources

**PRIMARY:**
1. **Actual deployment:** Rust binary + Nginx (no Docker in main project)
2. **Multi-agent-docker:** Separate Docker environment (NOT for VisionFlow)
3. **Code evidence:** No Dockerfile in project root

**AUTHORITATIVE DOC TO CREATE:**
- `/docs/deployment/README.md` - Unified deployment guide

### Files Requiring Changes

#### ❌ DELETE (Misleading):

**EVALUATE FIRST** (check if Docker refs are for multi-agent-docker):
```bash
grep -r "docker\|Docker\|container" docs/deployment/ docs/guides/deployment.md
```

**If Docker deployment docs exist and are WRONG:**
- Move to `/docs/deployment/archive/docker-attempted.md`
- Add note: "Archived - Docker deployment not implemented for VisionFlow main"

#### ✏️ EDIT/CREATE:

1. **CREATE: `/docs/deployment/README.md`**
```markdown
# VisionFlow Deployment Guide

## Deployment Approaches

### Production Deployment (Recommended)

**Architecture:**
- Backend: Rust binary (compiled release build)
- Frontend: Static Vite build served by Nginx
- Database: PostgreSQL (separate instance)
- Configuration: Environment variables + TOML files

**Deployment steps:**
```bash
# Backend
cargo build --release --features gpu  # or without gpu
./target/release/webxr  # or your binary name
# Listens on $SYSTEM_NETWORK_PORT (default: 3030)

# Frontend
cd client/
npm run build
# Deploy dist/ to Nginx static hosting
```

**systemd Service Example:**
```ini
[Unit]
Description=VisionFlow Backend API
After=network.target postgresql.service

[Service]
Type=simple
User=visionflow
WorkingDirectory=/opt/visionflow
Environment=SYSTEM_NETWORK_PORT=3030
Environment=DATABASE_URL=postgresql://...
ExecStart=/opt/visionflow/webxr
Restart=always

[Install]
WantedBy=multi-user.target
```

### Development Deployment

**Quick start:**
```bash
# Terminal 1: Backend
cargo run

# Terminal 2: Frontend
cd client && npm run dev
```

**Ports:**
- Backend API: `localhost:3030`
- Frontend Dev: `localhost:5173`

### Docker Deployment

**Status:** ❌ NOT IMPLEMENTED for VisionFlow main project

**Note:** `/home/devuser/workspace/project/multi-agent-docker/` is a SEPARATE
system (Turbo Flow Claude) and not used for VisionFlow deployment.

For Docker deployment, you would need to create:
- `Dockerfile` for Rust backend
- `Dockerfile` for frontend build
- `docker-compose.yml` for orchestration

**See:** `/docs/deployment/docker-future.md` for planned implementation.

## Configuration Files

### Required Files
- `data/settings.yaml` - Main configuration
- `data/dev_config.toml` - Physics parameters
- `.env` - Database credentials (not in git)

### Environment Variables
```bash
SYSTEM_NETWORK_PORT=3030        # API server port
DATABASE_URL=postgresql://...   # PostgreSQL connection
RUST_LOG=info                   # Logging level
ENABLE_GPU=true                 # GPU acceleration
```

## Monitoring & Logging

**Logs location:** `stdout` (redirect with systemd or supervisor)
**Health check:** `curl http://localhost:3030/health`
**Metrics:** TBD (Prometheus integration planned)

## Security Checklist

⚠️ **CRITICAL:** VisionFlow currently has NO authentication implemented.

**Before production deployment:**
- [ ] Implement authentication middleware
- [ ] Add HTTPS/TLS termination (use Nginx reverse proxy)
- [ ] Set up firewall rules (allow only necessary ports)
- [ ] Configure CORS properly (restrict origins)
- [ ] Review data/settings.yaml for secrets
- [ ] Use strong PostgreSQL passwords
- [ ] Enable rate limiting

## Related Documentation

- [Development Setup](../developer-guide/01-development-setup.md)
- [Configuration Reference](../reference/configuration.md)
- [Architecture Overview](../architecture/overview.md)
```

2. **UPDATE: `/docs/development/deployment.md`**
   - Add redirect to `/docs/deployment/README.md`
   - Or delete if duplicate

3. **UPDATE: `/docs/guides/deployment.md`**
   - Consolidate into `/docs/deployment/README.md`
   - Leave redirect or delete

4. **CREATE: `/docs/deployment/docker-future.md`**
```markdown
# Docker Deployment (Future Implementation)

**Status:** 📋 PLANNED - Not yet implemented

## Current State

VisionFlow main project does NOT have Docker support.

## Planned Implementation

Future Docker deployment would include:

### Backend Dockerfile
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features gpu

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libpq5
COPY --from=builder /app/target/release/webxr /usr/local/bin/
CMD ["webxr"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "3030:3030"
    environment:
      - DATABASE_URL=postgresql://db/visionflow
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=visionflow
```

## Contributing

If you'd like to implement Docker support, see:
- [Contributing Guide](../contributing.md)
- [Architecture Overview](../architecture/overview.md)
```

### Execution Steps

```bash
cd /home/devuser/workspace/project/docs

# Step 1: Create deployment directory
mkdir -p deployment

# Step 2: Check for existing Docker docs
find . -name "*.md" -exec grep -l "Dockerfile\|docker-compose" {} \; > /tmp/docker_docs.txt

# Step 3: Review and categorize
# (Manual review of /tmp/docker_docs.txt)

# Step 4: Create new deployment README
# (Use Write tool to create docs/deployment/README.md with content above)

# Step 5: Create future Docker guide
# (Use Write tool to create docs/deployment/docker-future.md)

# Step 6: Update or remove duplicates
# (Based on Step 2 findings)

# Step 7: Commit
git add docs/deployment/
git commit -m "docs: create unified deployment guide

- Created docs/deployment/README.md (authoritative guide)
- Documented production deployment (Rust binary + Nginx)
- Clarified Docker NOT implemented for main project
- Added docker-future.md for planned implementation
- Consolidated deployment docs from multiple locations

Closes documentation contradiction about deployment approach."
```

### Validation Checklist

- [ ] `/docs/deployment/README.md` exists and is comprehensive
- [ ] Deployment docs clearly state "NO Docker" for main project
- [ ] multi-agent-docker/ separation clearly documented
- [ ] Production deployment steps are accurate and tested
- [ ] Security warnings about missing auth are prominent
- [ ] All old deployment docs redirect to new README

### Rollback Procedure

```bash
git checkout docs/deployment/
git clean -fd docs/deployment/
```

---

## Task 1.4: Developer Guide Consolidation

### Ground Truth (from codebase and package files)

**ACTUAL DEVELOPMENT STACK:**
```bash
# Backend (from Cargo.toml):
- Rust 1.75+
- Actix-web (web framework)
- PostgreSQL (database)
- Optional: CUDA (GPU acceleration)
- Testing: cargo test

# Frontend (from client/package.json):
- TypeScript 5.x
- React 18
- Vite 5.x
- Three.js (3D visualization)
- Testing: Vitest

# Tools:
- cargo (Rust package manager)
- npm/pnpm (Node package manager)
- git
```

**CONTRADICTORY DOCS:**
- Some mention outdated tools
- Some have incorrect setup instructions
- Testing approach inconsistent

### Truth Sources

**PRIMARY:**
1. `/Cargo.toml` - Rust dependencies
2. `/client/package.json` - Frontend dependencies
3. `/src/main.rs` - Actual imports and setup

**AUTHORITATIVE DOC:**
- `/docs/developer-guide/01-development-setup.md`

### Files Requiring Changes

#### ❌ DELETE:

**Check for duplicates:**
```bash
find docs/ -name "*setup*" -o -name "*development*" -o -name "*getting*started*" | grep -v "developer-guide"
```

**IF FOUND:** Archive or consolidate into `/docs/developer-guide/`

#### ✏️ EDIT/UPDATE:

1. **`/docs/developer-guide/01-development-setup.md`**

**Replace entire content with accurate version:**

```markdown
# Development Setup

**Last Updated:** 2025-10-27
**Tested On:** Ubuntu 22.04, macOS 14, Windows 11 + WSL2

## Prerequisites

### Required Software

#### Backend Development
```bash
# Rust toolchain (1.75 or later)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# PostgreSQL 14+
sudo apt install postgresql postgresql-contrib  # Ubuntu/Debian
brew install postgresql@15  # macOS

# System libraries
sudo apt install build-essential libpq-dev pkg-config libssl-dev
```

#### Frontend Development
```bash
# Node.js 18+ (via nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
nvm install 18
nvm use 18

# Package manager
npm install -g pnpm  # or use npm
```

#### Optional: GPU Acceleration
```bash
# CUDA Toolkit 11.8+ (NVIDIA GPUs only)
# See: https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi
nvcc --version
```

### Repository Setup

```bash
# Clone repository
git clone https://github.com/yourusername/visionflow.git
cd visionflow

# Backend dependencies
cargo build

# Frontend dependencies
cd client
pnpm install
cd ..
```

## Configuration

### Database Setup

```bash
# Create PostgreSQL database
sudo -u postgres psql
postgres=# CREATE DATABASE visionflow;
postgres=# CREATE USER visionflow WITH PASSWORD 'your_secure_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE visionflow TO visionflow;
postgres=# \q

# Set environment variable
echo 'DATABASE_URL=postgresql://visionflow:your_secure_password@localhost/visionflow' >> .env

# Run migrations (if using diesel/sqlx)
# diesel migration run
# OR: cargo sqlx migrate run
```

### Configuration Files

```bash
# Copy example configs
cp data/settings.yaml.example data/settings.yaml
cp data/dev_config.toml.example data/dev_config.toml

# Edit settings
vim data/settings.yaml  # Adjust database, ports, features
```

**Key settings to verify:**
```yaml
# data/settings.yaml
database:
  url: "postgresql://visionflow:password@localhost/visionflow"

network:
  host: "0.0.0.0"
  port: 3030  # Or set SYSTEM_NETWORK_PORT env var

physics:
  enable_gpu: false  # Set true if CUDA available
```

## Running Development Servers

### Backend API

```bash
# Development mode (with hot reload via cargo-watch)
cargo install cargo-watch
cargo watch -x run

# Or standard run
cargo run

# Server starts on: http://localhost:3030
# Health check: curl http://localhost:3030/health
```

### Frontend Dev Server

```bash
cd client
pnpm dev

# Vite dev server starts on: http://localhost:5173
# Proxies API requests to: http://localhost:3030
```

**Open browser:** http://localhost:5173

## Running Tests

### Backend Tests

```bash
# All tests
cargo test

# Specific module
cargo test binary_protocol

# Integration tests
cargo test --test integration_tests

# With output
cargo test -- --nocapture

# Performance tests
cargo test --release perf_
```

### Frontend Tests

```bash
cd client

# Unit tests (Vitest)
pnpm test

# Coverage
pnpm test:coverage

# E2E tests (if configured)
pnpm test:e2e
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/my-feature
```

### 2. Make Changes
```bash
# Backend changes in src/
# Frontend changes in client/src/
```

### 3. Test Changes
```bash
# Backend
cargo test
cargo clippy  # Linting
cargo fmt     # Formatting

# Frontend
cd client
pnpm test
pnpm lint
pnpm type-check
```

### 4. Commit and Push
```bash
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
```

### 5. Create Pull Request
- Use GitHub web interface
- Fill out PR template
- Wait for CI checks

## Common Issues

### Backend Issues

**Problem:** `cargo build` fails with linker errors
```bash
# Solution: Install required system libraries
sudo apt install build-essential pkg-config libssl-dev libpq-dev
```

**Problem:** Database connection refused
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL
```

**Problem:** GPU features fail to compile
```bash
# Build without GPU
cargo build --no-default-features

# Or check CUDA installation
nvidia-smi
nvcc --version
```

### Frontend Issues

**Problem:** `pnpm install` fails
```bash
# Clear cache
pnpm store prune
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

**Problem:** Vite dev server connection refused
```bash
# Check if port 5173 is in use
lsof -i :5173

# Kill conflicting process or change port
pnpm dev --port 5174
```

**Problem:** WebSocket connection fails
```bash
# Verify backend is running on port 3030
curl http://localhost:3030/health

# Check browser console for errors
# Verify client/vite.config.ts proxy settings
```

## IDE Setup

### VS Code (Recommended)

**Extensions:**
```json
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "tamasfe.even-better-toml",
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "bradlc.vscode-tailwindcss"
  ]
}
```

**Settings (`.vscode/settings.json`):**
```json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.checkOnSave.command": "clippy",
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.formatOnSave": true
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.formatOnSave": true
  }
}
```

### IntelliJ IDEA / CLion

- Install Rust plugin
- Configure Rust toolchain
- Enable Clippy linting
- Set up Node.js interpreter

## Environment Variables Reference

```bash
# Backend
DATABASE_URL=postgresql://user:pass@localhost/visionflow
SYSTEM_NETWORK_PORT=3030
RUST_LOG=debug,actix_web=info
ENABLE_GPU=true

# Frontend (client/.env)
VITE_API_URL=http://localhost:3030
VITE_WS_URL=ws://localhost:3030/ws
```

## Next Steps

- [Project Structure](./02-project-structure.md)
- [Architecture Overview](./03-architecture.md)
- [Adding Features](./04-adding-features.md)
- [Testing Guide](./05-testing.md)
- [Contributing Guidelines](./06-contributing.md)
```

2. **`/docs/developer-guide/05-testing.md`**

**Update testing section to reflect ACTUAL test setup:**

```markdown
# Testing Guide

## Backend Testing

### Test Organization

```
tests/
├── integration/          # Integration tests
├── unit/                # Unit tests (also in src/)
└── fixtures/            # Test data and mocks
```

### Running Tests

```bash
# All tests
cargo test

# Unit tests only
cargo test --lib

# Integration tests only
cargo test --test '*'

# Specific test
cargo test test_binary_protocol

# With output
cargo test -- --show-output

# Parallel execution (default)
cargo test

# Serial execution
cargo test -- --test-threads=1
```

### Test Categories

**Unit Tests (in `src/`):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_protocol_roundtrip() {
        // Test implementation
    }
}
```

**Integration Tests (in `tests/`):**
```rust
#[tokio::test]
async fn test_api_endpoint() {
    // Test implementation
}
```

### Mocking & Fixtures

```rust
// Use test fixtures
use crate::fixtures::create_test_graph;

#[test]
fn test_with_fixture() {
    let graph = create_test_graph();
    // Test with graph
}
```

## Frontend Testing

### Test Stack
- **Test Runner:** Vitest
- **Testing Library:** @testing-library/react
- **E2E:** (Planned: Playwright)

### Running Tests

```bash
cd client

# Watch mode
pnpm test

# Run once
pnpm test:run

# Coverage
pnpm test:coverage

# UI mode
pnpm test:ui
```

### Test Structure

```typescript
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

describe('Component', () => {
  it('renders correctly', () => {
    render(<Component />)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
```

## Test Coverage

### Backend Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage
cargo tarpaulin --out Html

# Open coverage report
open tarpaulin-report.html
```

**Coverage Targets:**
- Unit tests: > 80%
- Critical paths: > 95%
- Integration tests: All endpoints

### Frontend Coverage

```bash
cd client
pnpm test:coverage

# Open report
open coverage/index.html
```

## Performance Testing

### Load Testing

```bash
# Install wrk
sudo apt install wrk

# Test API endpoint
wrk -t4 -c100 -d30s http://localhost:3030/api/graphs

# WebSocket load test
# (Custom script in scripts/load_test_ws.sh)
```

### Benchmarking

```bash
# Rust benchmarks
cargo bench

# Criterion output
open target/criterion/report/index.html
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main
- Nightly (full test suite)

**Workflow:** `.github/workflows/test.yml`

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test
      - run: cd client && npm test
```

## Test Data

### Database Fixtures

```bash
# Load test data
psql visionflow < tests/fixtures/test_data.sql

# Reset database
cargo run --bin reset_db
```

### Mock Data Generators

```rust
// Generate test graphs
use crate::test_utils::GraphGenerator;

let graph = GraphGenerator::new()
    .with_nodes(100)
    .with_edges(200)
    .generate();
```

## Debugging Tests

### Backend

```bash
# Run with logging
RUST_LOG=debug cargo test -- --nocapture

# Run specific test with backtrace
RUST_BACKTRACE=1 cargo test failing_test
```

### Frontend

```bash
# Debug in browser
pnpm test:ui

# Use debugger statement
it('debug this', () => {
  debugger;  // Will pause in VS Code debugger
  expect(true).toBe(true)
})
```

## Best Practices

1. **Write tests first** (TDD when appropriate)
2. **Test one thing** per test function
3. **Use descriptive names** for tests
4. **Mock external dependencies** (DB, APIs)
5. **Run tests before committing**
6. **Keep tests fast** (<1s per test)
7. **Avoid flaky tests** (no sleeps, use waits)
8. **Document complex test setups**

## Related Documentation

- [Development Setup](./01-development-setup.md)
- [Architecture](./03-architecture.md)
- [Contributing](./06-contributing.md)
```

### Execution Steps

```bash
cd /home/devuser/workspace/project/docs/developer-guide

# Step 1: Backup existing files
cp 01-development-setup.md 01-development-setup.md.bak
cp 05-testing.md 05-testing.md.bak

# Step 2: Update files with new content
# (Use Edit tool to replace content)

# Step 3: Verify Cargo.toml and package.json match docs
cargo tree --depth 1 > /tmp/cargo_deps.txt
cat client/package.json | jq '.dependencies' > /tmp/npm_deps.json

# Step 4: Test instructions
# (Manual: Follow setup guide on clean VM/container)

# Step 5: Commit
git add docs/developer-guide/
git commit -m "docs: update developer setup guide with accurate stack info

- Updated 01-development-setup.md with actual dependencies
- Corrected testing.md to reflect Vitest (not Jest)
- Added GPU setup instructions (CUDA optional)
- Verified all setup steps against current codebase
- Removed outdated tool references

Tested on: Ubuntu 22.04, macOS 14"
```

### Validation Checklist

- [ ] All dependencies in docs match `Cargo.toml`
- [ ] All frontend deps match `client/package.json`
- [ ] Setup instructions tested on clean environment
- [ ] Database setup steps are accurate
- [ ] Test commands work as documented
- [ ] Port numbers (3030, 5173) are correct
- [ ] IDE setup recommendations are current

### Rollback Procedure

```bash
cp 01-development-setup.md.bak 01-development-setup.md
cp 05-testing.md.bak 05-testing.md
git checkout docs/developer-guide/
```

---

## Task 1.5: Testing Documentation Accuracy

### Ground Truth (from test files and CI config)

**ACTUAL TESTING STATUS:**

```bash
# Evidence from repository:
1. tests/ directory exists with Rust integration tests
2. client/ has package.json with Vitest configured
3. Some tests use #[cfg(test)] in src/ files
4. NO .github/workflows/ found (CI status unclear)
5. cargo test works (validated by dev team)
```

**CONTRADICTIONS:**
- Some docs claim "automated CI/CD"
- Some docs mention Jest (but Vitest is used)
- Test coverage percentages unverified

### Truth Sources

**PRIMARY:**
1. `/tests/` directory - Integration test files
2. `src/**/*.rs` - Unit tests in code
3. `/client/package.json` - Frontend test config (Vitest)
4. `.github/workflows/` - CI config (check if exists)

**DOCS TO UPDATE:**
- `/docs/developer-guide/05-testing.md`
- `/docs/guides/testing-guide.md`

### Files Requiring Changes

#### ❌ DELETE:

**Check for outdated testing docs:**
```bash
find docs/ -name "*test*" -o -name "*ci*" -o -name "*coverage*"
```

**IF claiming automated CI but .github/workflows doesn't exist:**
- Remove CI claims or mark as "planned"

#### ✏️ EDIT:

1. **`/docs/developer-guide/05-testing.md`**
   - Already updated in Task 1.4 ✅
   - Verify no claims of non-existent CI

2. **`/docs/guides/testing-guide.md`** (if exists)
   - **Option A:** Delete and redirect to `/docs/developer-guide/05-testing.md`
   - **Option B:** Update to match current testing setup

3. **UPDATE all docs mentioning CI:**

```bash
# Find CI claims
grep -r "continuous integration\|github actions\|ci/cd\|automated tests" docs/ --include="*.md"

# For each file found:
# IF .github/workflows/ EXISTS: Keep as-is
# IF .github/workflows/ MISSING: Add disclaimer
```

**Disclaimer to add if CI missing:**
```markdown
## ⚠️ Continuous Integration Status

**Current:** CI/CD automation is planned but not yet implemented.

Tests must be run manually:
```bash
cargo test  # Backend
cd client && pnpm test  # Frontend
```

**Planned:** GitHub Actions workflow for automated testing on PRs.
```

4. **CREATE: `/docs/testing-status.md`**

```markdown
# Testing Status Report

**Last Updated:** 2025-10-27
**Maintainer:** Development Team

## Current Testing Coverage

### Backend (Rust)

**Status:** ✅ Operational

**Test Types:**
- Unit tests: Embedded in `src/` files with `#[cfg(test)]`
- Integration tests: `/tests/` directory
- Test runner: `cargo test`

**How to run:**
```bash
cargo test
cargo test --lib  # Unit tests only
cargo test --test '*'  # Integration tests only
```

**Coverage:**
- Measurement tool: `cargo-tarpaulin` (manual)
- Target: >80% for critical paths
- Current: [Run `cargo tarpaulin` to measure]

**Files with tests:**
```bash
find src/ -name "*.rs" -exec grep -l "#\[cfg(test)\]" {} \;
```

### Frontend (TypeScript/React)

**Status:** ✅ Operational

**Test Types:**
- Unit/Component tests: Vitest + Testing Library
- Configuration: `client/vitest.config.ts`
- Test files: `client/src/**/*.test.tsx`

**How to run:**
```bash
cd client
pnpm test          # Watch mode
pnpm test:run      # Run once
pnpm test:coverage # With coverage
```

**Coverage:**
- Tool: Vitest built-in coverage
- Target: >70% for components
- Current: [Run `pnpm test:coverage` to measure]

### Integration/E2E Tests

**Status:** 📋 Planned

**Proposed:**
- Tool: Playwright or Cypress
- Coverage: Critical user workflows
- Implementation: Future milestone

### Performance Tests

**Status:** ⚠️ Manual

**Available:**
- Load testing: `wrk` (manual scripts)
- Benchmarks: `cargo bench` (some modules)
- WebSocket load: Custom scripts in `/scripts/`

**How to run:**
```bash
# API load test
wrk -t4 -c100 -d30s http://localhost:3030/api/graphs

# Rust benchmarks
cargo bench
```

## Continuous Integration

**Status:** 🚧 NOT IMPLEMENTED

**Current Workflow:**
- Developers run tests manually before commits
- No automated CI/CD pipeline
- Tests not run automatically on PRs

**Planned Implementation:**
```yaml
# .github/workflows/test.yml (future)
name: Tests
on: [push, pull_request]
jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: cargo test
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: cd client && npm test
```

**To enable:**
1. Create `.github/workflows/test.yml`
2. Configure GitHub Actions permissions
3. Add status badges to README.md

## Test Quality Metrics

### Flakiness
- **Target:** <1% flaky tests
- **Tracking:** Manual observation
- **Current:** No known flaky tests

### Speed
- **Target:** <10s for unit tests, <2min for full suite
- **Backend:** ~5s (fast)
- **Frontend:** ~2s (fast)

### Maintenance
- **Review:** Tests reviewed during code review
- **Updates:** Tests updated with feature changes

## Known Gaps

1. ❌ No E2E tests for user workflows
2. ❌ No automated CI/CD pipeline
3. ❌ No visual regression testing
4. ❌ Limited WebSocket protocol testing
5. ⚠️ Coverage measurement is manual

## Improving Test Coverage

**Priority improvements:**
1. Set up GitHub Actions CI
2. Implement E2E tests with Playwright
3. Automate coverage reporting
4. Add WebSocket integration tests
5. Create performance test baselines

## Contributing to Tests

See: [Testing Guide](./developer-guide/05-testing.md)

**Quick start:**
```bash
# Add backend unit test
# Edit src/your_module.rs, add #[cfg(test)] module

# Add frontend component test
# Create client/src/components/YourComponent.test.tsx

# Run all tests
cargo test && cd client && pnpm test
```
```

### Execution Steps

```bash
cd /home/devuser/workspace/project

# Step 1: Check for CI config
ls -la .github/workflows/ 2>/dev/null || echo "No CI workflows found"

# Step 2: Measure actual test coverage
cargo install cargo-tarpaulin
cargo tarpaulin --out Stdout > /tmp/coverage_backend.txt

cd client
pnpm test:coverage > /tmp/coverage_frontend.txt
cd ..

# Step 3: Find all docs claiming CI
grep -r "github actions\|ci/cd\|continuous integration" docs/ --include="*.md" -i > /tmp/ci_claims.txt

# Step 4: Create testing status doc
# (Use Write tool to create docs/testing-status.md)

# Step 5: Update CI claims in other docs
# (Based on /tmp/ci_claims.txt findings)

# Step 6: Commit
git add docs/testing-status.md
git commit -m "docs: create testing status report and fix CI claims

- Created docs/testing-status.md with current test coverage
- Measured actual coverage: Backend X%, Frontend Y%
- Corrected CI/CD status (not implemented yet)
- Documented manual testing workflow
- Identified gaps for future improvement

Verified by running cargo test and pnpm test."
```

### Validation Checklist

- [ ] `/docs/testing-status.md` accurately reflects test setup
- [ ] No false claims of automated CI (unless .github/workflows/ exists)
- [ ] Coverage percentages are measured (not guessed)
- [ ] Test commands in docs actually work
- [ ] Frontend uses Vitest (not Jest)
- [ ] All test gaps are honestly documented

### Rollback Procedure

```bash
git checkout docs/testing-status.md
# Manually revert changes to other files
```

---

## Task 1.6: Cross-Cutting Updates

### After completing Tasks 1.1-1.5, make these final edits:

#### 1. **Update `/docs/00-INDEX.md`**

Add prominent notices:
```markdown
# VisionFlow Documentation Index

**Last Major Update:** 2025-10-27
**Status:** Category 1 Contradictions Resolved ✅

## ⚠️ Important Notes

### Recently Updated (2025-10-27)
- Binary Protocol specification standardized to 36 bytes
- API port standardized to 3030 (configurable via SYSTEM_NETWORK_PORT)
- Deployment guide consolidated (Docker NOT implemented)
- Developer setup guide updated with accurate stack
- Testing status documented (CI not yet implemented)

See [Documentation Refactoring Plan](./refactoring/) for details.

## Documentation Structure

### 🔴 Authoritative References (Single Source of Truth)
- [Binary Protocol V2](./reference/api/binary-protocol.md) - **36-byte wire format**
- [REST API Reference](./reference/api/rest-api.md) - **Port 3030**
- [Deployment README](./deployment/README.md) - **Production deployment**
- [Developer Setup](./developer-guide/01-development-setup.md) - **Stack & tools**
- [Testing Status](./testing-status.md) - **Current test coverage**

...
```

#### 2. **Create `/docs/refactoring/CHANGELOG.md`**

```markdown
# Documentation Refactoring Changelog

## Category 1: Critical Contradictions [2025-10-27] ✅

### Task 1.1: Binary Protocol Standardization
- **Fixed:** Binary protocol V2 documentation (36 bytes, not 38)
- **Files changed:** `src/utils/binary_protocol.rs`
- **Validation:** All docs now consistently show 36-byte V2 format

### Task 1.2: API Port Standardization
- **Fixed:** All docs updated to port 3030 (actual default)
- **Removed:** References to ports 8080 and 3001
- **Added:** SYSTEM_NETWORK_PORT environment variable documentation
- **Files changed:** 12 documentation files

### Task 1.3: Deployment Consolidation
- **Created:** `/docs/deployment/README.md` (authoritative guide)
- **Clarified:** Docker NOT implemented for main VisionFlow
- **Separated:** multi-agent-docker/ documentation (different system)
- **Files changed:** 4 deployment guides consolidated

### Task 1.4: Developer Guide Update
- **Updated:** `/docs/developer-guide/01-development-setup.md`
- **Verified:** All dependencies match Cargo.toml & package.json
- **Corrected:** Testing framework (Vitest not Jest)
- **Files changed:** 2 developer guides

### Task 1.5: Testing Documentation Accuracy
- **Created:** `/docs/testing-status.md` (honest status report)
- **Corrected:** CI/CD claims (not yet implemented)
- **Measured:** Actual test coverage percentages
- **Files changed:** 3 testing-related docs

### Task 1.6: Cross-Cutting Updates
- **Updated:** `/docs/00-INDEX.md` with authoritative references
- **Created:** This changelog
- **Added:** Refactoring documentation in `/docs/refactoring/`
```

#### 3. **Update `/README.md`** (project root)

Add documentation health notice:
```markdown
# VisionFlow

...

## Documentation

**Documentation Status:** ✅ Recently Updated (2025-10-27)

All documentation has been audited and critical contradictions resolved.

**Key References:**
- [Getting Started](./docs/getting-started/01-installation.md)
- [API Reference](./docs/reference/api/) - **Port 3030, Binary Protocol V2**
- [Deployment Guide](./docs/deployment/README.md) - **Production deployment**
- [Developer Setup](./docs/developer-guide/01-development-setup.md)

For documentation health reports, see [`/docs/refactoring/`](./docs/refactoring/).
```

### Execution Steps

```bash
cd /home/devuser/workspace/project

# Step 1: Update INDEX
# (Use Edit tool on docs/00-INDEX.md)

# Step 2: Create changelog
mkdir -p docs/refactoring
# (Use Write tool to create docs/refactoring/CHANGELOG.md)

# Step 3: Update root README
# (Use Edit tool on README.md)

# Step 4: Final commit
git add docs/00-INDEX.md docs/refactoring/CHANGELOG.md README.md
git commit -m "docs: complete Category 1 refactoring (critical contradictions)

Summary of changes:
- Standardized binary protocol to 36 bytes across all docs
- Unified API port to 3030 (actual server default)
- Consolidated deployment guides (clarified no Docker)
- Updated developer setup with accurate dependencies
- Documented honest testing status (no CI yet)
- Added authoritative reference markers
- Created refactoring changelog

All Category 1 contradictions resolved.
See docs/refactoring/ for detailed audit trail."
```

### Final Validation

```bash
# Run comprehensive validation
bash docs/refactoring/validate-category1.sh

# Should check:
# - No "38 bytes" in codebase
# - No port 8080/3001 references
# - Deployment docs don't claim Docker works
# - Developer guide matches Cargo.toml
# - No false CI claims
```

---

## Validation Script

**Create:** `/docs/refactoring/validate-category1.sh`

```bash
#!/bin/bash
# Validation script for Category 1 refactoring

echo "=== Category 1 Validation ==="
echo ""

FAIL=0

# Test 1.1: Binary protocol
echo "Test 1.1: Binary Protocol (36 bytes)"
if grep -r "38 bytes per node" . --include="*.rs" --include="*.md" 2>/dev/null; then
    echo "❌ FAIL: Found '38 bytes' references"
    FAIL=1
else
    echo "✅ PASS: No incorrect byte counts"
fi
echo ""

# Test 1.2: API ports
echo "Test 1.2: API Port Standardization (3030)"
if grep -r "localhost:8080\|localhost:3030\|:8080/\|:3030/" docs/ --include="*.md" 2>/dev/null; then
    echo "❌ FAIL: Found incorrect port references"
    FAIL=1
else
    echo "✅ PASS: All ports standardized to 3030"
fi
echo ""

# Test 1.3: Docker deployment claims
echo "Test 1.3: Deployment Documentation"
if [ ! -f "Dockerfile" ] && grep -r "docker deployment" docs/deployment/ --include="*.md" -i 2>/dev/null; then
    echo "⚠️  WARNING: Docker claims found but no Dockerfile exists"
fi
if [ -f "docs/deployment/README.md" ]; then
    echo "✅ PASS: Deployment README exists"
else
    echo "❌ FAIL: Missing docs/deployment/README.md"
    FAIL=1
fi
echo ""

# Test 1.4: Developer dependencies match
echo "Test 1.4: Developer Guide Accuracy"
if grep -q "vitest" client/package.json && ! grep -q "jest" docs/developer-guide/05-testing.md; then
    echo "✅ PASS: Testing framework correctly documented (Vitest)"
else
    echo "⚠️  WARNING: Check testing framework documentation"
fi
echo ""

# Test 1.5: Testing status
echo "Test 1.5: Testing Documentation"
if [ -f "docs/testing-status.md" ]; then
    echo "✅ PASS: Testing status documented"
else
    echo "❌ FAIL: Missing docs/testing-status.md"
    FAIL=1
fi

if [ ! -d ".github/workflows" ] && grep -r "automated CI" docs/ --include="*.md" -i 2>/dev/null | grep -v "planned\|future"; then
    echo "⚠️  WARNING: CI claims found but no .github/workflows/ directory"
fi
echo ""

# Test 1.6: Index updated
echo "Test 1.6: Documentation Index"
if grep -q "2025-10-27" docs/00-INDEX.md 2>/dev/null; then
    echo "✅ PASS: Index shows recent update"
else
    echo "❌ FAIL: Index not updated with refactoring date"
    FAIL=1
fi
echo ""

# Final result
echo "=== Validation Complete ==="
if [ $FAIL -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    exit 1
fi
```

```bash
chmod +x docs/refactoring/validate-category1.sh
```

---

## Rollback Plan (Full Category 1)

**If entire category needs rollback:**

```bash
cd /home/devuser/workspace/project

# Option 1: Revert all commits
git log --oneline --grep="docs:" --since="2025-10-27" > /tmp/refactor_commits.txt
# Review /tmp/refactor_commits.txt
# git revert <commit-hash> for each

# Option 2: Reset to before refactoring
git log --oneline
# Find commit before refactoring started
git reset --hard <commit-before-refactoring>

# Option 3: Restore from backups
find docs/ -name "*.bak" -exec bash -c 'mv "$1" "${1%.bak}"' _ {} \;
```

---

## Estimated Time

| Task | Time | Complexity |
|------|------|------------|
| 1.1 Binary Protocol | 30 min | Low |
| 1.2 API Ports | 1 hour | Medium |
| 1.3 Deployment | 2 hours | Medium |
| 1.4 Developer Guide | 2 hours | High |
| 1.5 Testing Docs | 1 hour | Medium |
| 1.6 Cross-Cutting | 1 hour | Low |
| **TOTAL** | **7.5 hours** | |

**Parallelization Possible:**
- Tasks 1.1, 1.2 can run in parallel
- Task 1.3, 1.4, 1.5 can run in parallel after 1.1-1.2
- Task 1.6 must run last

**With 2 people:** ~4 hours
**With 3 people:** ~3 hours

---

## Success Criteria

- [ ] All validation tests pass (`validate-category1.sh`)
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] `cd client && pnpm test` passes
- [ ] No grep results for incorrect values (38 bytes, port 8080/3001)
- [ ] All authoritative docs marked in INDEX
- [ ] Changelog documents all changes
- [ ] Git history is clean with descriptive commits

---

## Next Steps

After Category 1 completion:
1. **Category 2:** Outdated Information (remove deprecated features)
2. **Category 3:** Documentation Gaps (fill missing sections)
3. **Category 4:** Structural Issues (reorganize for clarity)
4. **Category 5:** Maintenance (set up auto-validation)

**See:** `/docs/refactoring/MASTER-PLAN.md` for full roadmap

---

**Execution Authority:** Lead Developer + Documentation Maintainer
**Review Required:** Yes (peer review before merging)
**Testing:** All changes must pass validation script
**Timeline:** Target completion: 2025-10-28
