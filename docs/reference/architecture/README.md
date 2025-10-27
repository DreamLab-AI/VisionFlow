# Architecture Documentation

Welcome to the VisionFlow Architecture documentation. This section covers system design, technical details, and implementation patterns.

## Quick Navigation

### Core Architecture

- **[Architecture Overview](./architecture.md)** - High-level design principles, system layers, and architectural philosophy
- **[Hexagonal & CQRS Implementation](./hexagonal-cqrs.md)** - Deep dive into ports/adapters pattern and CQRS architecture
- **[Actor System Integration](./actor-system.md)** - Legacy system compatibility and GPU acceleration
- **[Database Schema](./database-schema.md)** - SQLite schema details, optimization, and best practices

### By Task

**I want to...**

**Understand the overall system design**
→ Start with [Architecture Overview](./architecture.md)

**Implement new features**
→ Read [Hexagonal & CQRS Implementation](./hexagonal-cqrs.md) and [Database Schema](./database-schema.md)

**Optimize performance**
→ Review [Actor System Integration](./actor-system.md) (GPU acceleration) and [Database Schema](./database-schema.md) (query optimization)

**Migrate from legacy actors**
→ See [Actor System Integration](./actor-system.md) for adapter patterns

**Design database queries**
→ Reference [Database Schema](./database-schema.md) for table structures and indexes

## Key Concepts

### Hexagonal Architecture (Ports & Adapters)

The application core is isolated from external systems through abstract ports:

```
Domain Logic (Core)
    ↓
Ports (Interfaces)
    ↓
Adapters (Implementations)
    ↓
External Systems (Databases, APIs, etc.)
```

### CQRS Pattern

Write operations (Directives) are separated from read operations (Queries) for:
- Clearer intent
- Optimized read/write performance
- Complex business rule implementation

### Three-Database Architecture

- **Settings Database** - Configuration and preferences
- **Knowledge Graph Database** - Local markdown files
- **Ontology Database** - Semantic web knowledge

### GPU-Accelerated Physics

- 40+ CUDA kernels for force calculations
- 60 FPS at 100k nodes
- 82% bandwidth reduction via binary protocol

## Architecture Layers

```
┌─────────────────────────────────────┐
│  REST & WebSocket APIs              │
│  (Actix-web, tungstenite)           │
├─────────────────────────────────────┤
│  Application Layer (CQRS)           │
│  Directives & Queries               │
├─────────────────────────────────────┤
│  Domain Logic (Core)                │
│  Business rules & validation        │
├─────────────────────────────────────┤
│  Ports (Abstract Interfaces)        │
├─────────────────────────────────────┤
│  Adapters                           │
│  • SQLite Adapter                   │
│  • CUDA Adapter (GPU)               │
│  • HTTP Adapter                     │
├─────────────────────────────────────┤
│  External Systems                   │
│  • SQLite Databases                 │
│  • NVIDIA CUDA                      │
│  • File Systems                     │
└─────────────────────────────────────┘
```

## Technology Stack

- **Language:** Rust 1.70+
- **Web Framework:** Actix-web
- **Database:** SQLite 3.35+ (WAL mode)
- **GPU Acceleration:** CUDA 11.0+
- **Protocol:** Binary (V2) over WebSocket
- **Architecture Pattern:** Hexagonal (Ports & Adapters)
- **Application Pattern:** CQRS

## Key Metrics

| Metric | Value |
|--------|-------|
| Nodes per graph | Up to 1.07 billion |
| Edges per graph | Up to 4.3 billion |
| Physics FPS | 60 at 100k nodes |
| Latency | <10ms |
| Bandwidth | 3.6 MB/s @ 100k nodes |

## Patterns & Best Practices

### When Adding Features

1. **Define the Port** - Create abstract interface in domain layer
2. **Implement Directives/Queries** - Add CQRS handlers
3. **Create Adapters** - Implement concrete port interfaces
4. **Add REST/WebSocket** - Expose via API layer

### When Optimizing Performance

1. **Profile with CUDA** - Use GPU for compute-intensive operations
2. **Optimize Queries** - Add indexes, analyze query plans
3. **Use Binary Protocol** - For real-time updates
4. **Implement Caching** - Cache read-heavy operations

### When Fixing Bugs

1. **Write Domain Tests** - Test core logic in isolation
2. **Check Adapter Implementations** - Verify concrete behavior
3. **Review Constraints** - Check FOREIGN KEY constraints
4. **Validate Integration** - Test end-to-end via API

## Related Documentation

- **[API Reference](../api/)** - REST and WebSocket endpoints
- **[Developer Guides](../../guides/developer/)** - Implementation tutorials
- **[Concepts](../../concepts/)** - Understanding-oriented background

---

## Performance Targets

### Throughput

- Add Node: <5ms
- Query Graph: <100ms
- Update Position: <2ms
- Add OWL Axiom: <10ms

### Scalability

- 100k nodes at 60 FPS
- 500k edges supported
- 1M+ nodes at reduced FPS
- Concurrent WebSocket connections: Unlimited (per server resources)

### Efficiency

- 82% bandwidth reduction vs JSON
- 4-32x memory reduction via quantization (planned)
- Zero-copy shared ownership with Arc

---

**Last Updated:** 2025-10-25
**Status:** Current and verified against codebase
**Maintainer:** VisionFlow Architecture Team
