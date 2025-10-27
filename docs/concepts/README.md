# Concepts & Background Knowledge

Welcome to the VisionFlow Concepts documentation. This section provides understanding-oriented background knowledge about the system's design, patterns, and domain concepts.

## Quick Navigation

### Core Concepts

- **[Architecture Overview](./architecture.md)** - High-level design principles, system layers, and architectural philosophy

### By Topic

**I want to understand...**

**How the system is designed**
→ Start with [Architecture Overview](./architecture.md)

**How data flows through the system**
→ Read [Architecture Overview](./architecture.md) - Architecture Layers section

**Why we use hexagonal architecture**
→ See [Architecture Overview](./architecture.md) - Architectural Philosophy

**How physics simulation works**
→ Review [Architecture Overview](./architecture.md) - Physics Simulation Architecture

**Why we have three databases**
→ See [Architecture Overview](./architecture.md) - Database Architecture

---

## Key Concepts Explained

### Hexagonal Architecture (Ports & Adapters)

A design pattern that isolates your application core (business logic) from external systems through abstract "ports" (interfaces).

**Why it matters:**
- Core logic is testable without databases or web frameworks
- Easy to swap implementations (e.g., SQLite → PostgreSQL)
- Framework-agnostic design

### CQRS Pattern

Command Query Responsibility Segregation - separates write operations (Directives) from read operations (Queries).

**Why it matters:**
- Write operations can be thoroughly validated
- Read operations can be optimized independently
- Clear separation of concerns

### Three-Database Model

Separate SQLite databases for different domains:

| Database | Domain | Characteristics |
|----------|--------|-----------------|
| settings.db | Configuration | Small, high read/write |
| knowledge_graph.db | User data | Large, moderate activity |
| ontology.db | Semantic knowledge | Medium, low write |

**Why it matters:**
- Clear semantic boundaries
- Independent scaling
- Prevents ID conflicts

### GPU Acceleration

Using NVIDIA CUDA for physics simulation computations.

**Why it matters:**
- 60 FPS at 100k nodes (CPU: 8 FPS)
- Enables interactive visualization of large graphs
- Leverages hardware efficiently

---

## Mental Models

### The Layered Architecture

```
┌─────────────────────────────────────┐
│  External Users (REST/WebSocket)    │
├─────────────────────────────────────┤
│  Application Layer (CQRS)           │
│  Directives & Queries               │
├─────────────────────────────────────┤
│  Domain Logic (Core)                │
│  Business rules, validation         │
├─────────────────────────────────────┤
│  Ports (Abstract Interfaces)        │
├─────────────────────────────────────┤
│  Adapters (Concrete Implementations)│
├─────────────────────────────────────┤
│  External Systems                   │
│  (Databases, GPU, APIs)             │
└─────────────────────────────────────┘
```

### Data Flow Example

```
1. Client sends REST request
         ↓
2. HTTP Adapter receives request
         ↓
3. Application layer (CQRS)
   - Parse DirectiveCommand
   - Validate input
         ↓
4. Domain logic
   - Apply business rules
   - Check constraints
         ↓
5. Port interface
   - Call abstract method
         ↓
6. Adapter implementation
   - Execute SQL query
   - Update database
         ↓
7. Response back to client
```

---

## Design Decisions

### Why Rust?

- **Memory safety** - Prevents entire classes of bugs
- **Performance** - Comparable to C/C++
- **Async/await** - Native support for concurrent operations
- **Type system** - Catches many errors at compile time

### Why Three Databases Instead of One?

- **Domain separation** - Clear semantic boundaries
- **Independent scaling** - Each database sized to its workload
- **Conflict prevention** - Knowledge graph IDs won't clash with ontology IRIs
- **Simpler queries** - No complex JOINs across domains

### Why Binary Protocol?

- **Bandwidth** - 82% savings vs JSON at 100k nodes
- **Latency** - <10ms updates vs >50ms for JSON parsing
- **Scalability** - Supports more concurrent clients

### Why GPU Acceleration?

- **Performance** - 375× speedup for repulsion forces
- **Interactivity** - 60 FPS visualization of large graphs
- **Hardware efficiency** - Leverage available NVIDIA GPUs

---

## Key Design Patterns

### Pattern: Dependency Injection

Ports are injected into handlers, allowing easy substitution of implementations.

```rust
// Handler receives abstract port
pub async fn add_node(
    directive: AddNodeDirective,
    repos: &Repositories,  // Injected dependencies
) -> Result<NodeId> {
    repos.knowledge_graph.add_node(node).await
}
```

### Pattern: Adapter

Different implementations of the same port for different contexts.

```rust
// Abstract port
pub trait KnowledgeGraphRepository { ... }

// SQLite adapter
pub struct SqliteKnowledgeGraphRepository { ... }
impl KnowledgeGraphRepository for SqliteKnowledgeGraphRepository { ... }

// In-memory adapter (for testing)
pub struct InMemoryKnowledgeGraphRepository { ... }
impl KnowledgeGraphRepository for InMemoryKnowledgeGraphRepository { ... }
```

### Pattern: Arc<Mutex<>>

Safe concurrent access to shared resources without channels.

```rust
// Multiple threads can safely access the connection
let connection = Arc::new(Mutex::new(sqlite_conn));

// Clone for each thread
let conn_clone = connection.clone();
tokio::spawn(async move {
    let conn = conn_clone.lock().unwrap();
    // Use connection
});
```

---

## Terminology

| Term | Definition |
|------|-----------|
| **Port** | Abstract interface defining a capability |
| **Adapter** | Concrete implementation of a port |
| **Directive** | Write command (e.g., AddNodeDirective) |
| **Query** | Read operation (e.g., GetGraphQuery) |
| **Repository** | Port for data persistence |
| **Domain Logic** | Core business rules independent of frameworks |
| **WAL Mode** | Write-Ahead Logging for SQLite concurrency |
| **CQRS** | Command Query Responsibility Segregation |
| **Hexagonal** | Ports & Adapters architectural pattern |

---

## Performance Characteristics

### System Capacity

| Metric | Value |
|--------|-------|
| Nodes per graph | Up to 1.07 billion |
| Edges per graph | Up to 4.3 billion |
| Physics FPS | 60 at 100k nodes |
| REST latency | <50ms |
| WebSocket latency | <10ms |
| Bandwidth @ 100k nodes | 3.6 MB/s |

### Throughput

| Operation | Latency |
|-----------|---------|
| Add node | <5ms |
| Update position | <2ms |
| Query graph | <100ms |

---

## Failure Modes & Recovery

### Database Failures

- **Locked database** → WAL mode enables concurrent access
- **Corrupt database** → Restore from automated backups
- **Disk full** → Monitor disk space, implement cleanup

### Physics Simulation

- **GPU error** → Fallback to CPU computation
- **Memory overflow** → Reduce node count or increase VRAM
- **Numerical instability** → Adjust damping and timestep

### Network Issues

- **Connection lost** → Client reconnects with exponential backoff
- **Message corruption** → Binary protocol includes validation
- **Rate limited** → Client backs off per X-RateLimit-Reset header

---

## Related Documentation

- **[Architecture Details](../reference/architecture/)** - Technical deep dives
- **[API Reference](../reference/api/)** - REST and WebSocket specifications
- **[Developer Guides](../guides/developer/)** - Implementation tutorials

---

**Last Updated:** 2025-10-25
**Audience:** Architects, senior developers, stakeholders
**Maintenance:** Updated quarterly or with major changes
