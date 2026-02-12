---
title: Contributing to VisionFlow
description: Code contribution guide covering branch naming, PR process, code review, Rust and React conventions, and hexagonal architecture rules.
category: how-to
tags:
  - contributing
  - development
  - rust
  - react
  - architecture
updated-date: 2026-02-12
difficulty-level: intermediate
---

# Contributing to VisionFlow

This guide covers the code contribution workflow for VisionFlow, including branch conventions, pull request process, code review expectations, and the architectural rules that keep the codebase maintainable.

## Branch Naming Conventions

Use the following prefixes for branch names:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New feature | `feat/voice-spatial-audio` |
| `fix/` | Bug fix | `fix/neo4j-connection-timeout` |
| `refactor/` | Code restructuring (no behavior change) | `refactor/extract-graph-port` |
| `docs/` | Documentation only | `docs/deployment-guide` |
| `test/` | Adding or fixing tests | `test/ontology-reasoning-coverage` |
| `chore/` | Build, CI, tooling changes | `chore/update-livekit-image` |

Branch names should be lowercase, kebab-case, and descriptive. Include a ticket number if one exists (e.g., `fix/VF-123-websocket-reconnect`).

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feat/your-feature-name
```

### 2. Make Changes and Commit

Write clear, atomic commits. Use conventional commit messages:

```
feat: add LiveKit room token generation endpoint

Implements the /api/voice/token endpoint that generates
short-lived LiveKit access tokens for authenticated users.
```

### 3. Run Tests Before Pushing

```bash
# Rust tests
cargo test --all-features
cargo clippy --all-features -- -D warnings

# Frontend tests
cd client && npm test && npm run lint
```

### 4. Push and Open a PR

```bash
git push -u origin feat/your-feature-name
```

Open a pull request against `main`. Include in the PR description:
- **What** -- Summary of changes
- **Why** -- Motivation and context
- **How** -- Key implementation decisions
- **Testing** -- How you verified the changes work

### 5. Code Review

Every PR requires at least one approving review. Reviewers check for:
- Correctness and completeness
- Adherence to hexagonal architecture rules (see below)
- Test coverage for new behavior
- No regressions in existing tests
- Clean compilation with no Clippy warnings

## Rust Conventions

### Code Style

- Follow standard `rustfmt` formatting (run `cargo fmt` before committing)
- Use `cargo clippy --all-features -- -D warnings` to catch lint issues
- Prefer `thiserror` for library errors and `anyhow` for application errors
- Use `tracing` macros (`tracing::info!`, `tracing::error!`) instead of `log` macros
- All public types should derive `Debug` and `Serialize` where practical

### Async Patterns

- Use `tokio` as the async runtime (already configured for Actix-web)
- Prefer `async fn` over manual `Future` implementations
- Use `tokio::spawn` for background tasks, not `std::thread::spawn`
- All database operations (Neo4j via `neo4rs`) must be async

### Error Handling

```rust
// Define domain errors with thiserror
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Neo4j error: {0}")]
    Database(#[from] neo4rs::Error),
}
```

### Type Generation

VisionFlow uses `specta` to generate TypeScript types from Rust structs. When adding or modifying API types:

```bash
cargo run --bin generate_types
```

This outputs types to `client/src/types/generated/`. Never edit generated files by hand.

## React Conventions

### Code Style

- TypeScript strict mode is enabled
- Use functional components with hooks exclusively
- State management uses Zustand stores with Immer middleware
- UI components use Radix UI primitives with Tailwind CSS
- Three.js rendering uses `@react-three/fiber` and `@react-three/drei`

### File Organization

```
client/src/
  components/    # React components (PascalCase files)
  hooks/         # Custom hooks (useXxx naming)
  stores/        # Zustand stores
  types/         # TypeScript types (generated/ subdirectory is auto-generated)
  services/      # API client functions
  utils/         # Pure utility functions
```

### Testing Frontend Code

- Use Vitest with `@testing-library/react` for component tests
- Place test files next to the component: `Button.test.tsx` beside `Button.tsx`
- Use `vi.mock()` for module mocking
- Playwright tests go in `client/tests/` for E2E scenarios

## Hexagonal Architecture Rules

VisionFlow follows hexagonal (ports and adapters) architecture. These rules are enforced during code review.

### Layer Structure

```
src/
  ports/         # Trait definitions (interfaces)
  adapters/      # Implementations of port traits
  application/   # Use cases and application services
  models/        # Domain models
  actors/        # Actix actors (runtime boundaries)
  handlers/      # HTTP/WS request handlers
```

### Critical Rule: No Adapter-to-Adapter Imports

**Adapters must never import from other adapters.** All cross-cutting communication goes through ports (trait interfaces) or the application layer.

```rust
// WRONG: adapter importing another adapter
use crate::adapters::neo4j_repository::Neo4jGraphRepository;
use crate::adapters::websocket_notifier::WsNotifier;

// CORRECT: adapter depends on port traits
use crate::ports::graph_repository::GraphRepository;
use crate::ports::notifier::Notifier;
```

### Dependency Direction

```
handlers -> application -> ports <- adapters
                            ^
                            |
                          models
```

- **Handlers** call application services
- **Application services** depend on port traits (not concrete adapters)
- **Adapters** implement port traits
- **Models** are shared across all layers but depend on nothing else

### Adding a New Feature

1. Define the port trait in `src/ports/`
2. Implement the adapter in `src/adapters/`
3. Write the application service in `src/application/`
4. Wire it up in `src/main.rs` or the relevant actor
5. Add the HTTP handler in `src/handlers/`

## Commit Checklist

Before pushing, verify:

- [ ] `cargo fmt` -- Code is formatted
- [ ] `cargo clippy --all-features -- -D warnings` -- No lint warnings
- [ ] `cargo test --all-features` -- All Rust tests pass
- [ ] `cd client && npm test` -- All frontend tests pass
- [ ] `cd client && npm run lint` -- No ESLint errors
- [ ] No adapter-to-adapter imports introduced
- [ ] Generated types are up to date (`cargo run --bin generate_types`)
- [ ] PR description includes What/Why/How/Testing sections

## See Also

- [Testing Guide](./testing-guide.md) -- Detailed test execution reference
- [Development Setup](./01-development-setup.md) -- Environment setup for new contributors
- [Project Structure](./02-project-structure.md) -- Codebase layout overview
- [Docker Environment Setup](../deployment/docker-environment-setup.md) -- Running VisionFlow locally in Docker
- [Architecture](../infrastructure/architecture.md) -- System architecture reference
