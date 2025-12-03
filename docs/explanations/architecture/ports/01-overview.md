---
title: Hexagonal Architecture Ports - Overview
description: This document provides an overview of VisionFlow's **Hexagonal Architecture** implementation, specifically the **Ports Layer** that defines technology-agnostic interfaces between the domain logic a...
type: explanation
status: stable
---

# Hexagonal Architecture Ports - Overview

## Purpose

This document provides an overview of VisionFlow's **Hexagonal Architecture** implementation, specifically the **Ports Layer** that defines technology-agnostic interfaces between the domain logic and infrastructure.

## What is Hexagonal Architecture?

Hexagonal Architecture (also known as Ports and Adapters pattern) is an architectural pattern that:

1. **Isolates domain logic** from external concerns (databases, UI, external services)
2. **Defines ports** - trait interfaces that represent application boundaries
3. **Implements adapters** - concrete implementations that fulfill port contracts
4. **Enables testability** through dependency inversion and mock implementations
5. **Supports flexibility** by allowing multiple adapter implementations per port

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Core                        │
│                    (Domain Logic)                            │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Business Rules                      │  │
│  │        Graph Visualization, Physics, Ontology         │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ▲                                  │
│                           │                                  │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │                     Ports (Traits)                     │  │
│  │   SettingsRepository | KnowledgeGraphRepository       │  │
│  │   OntologyRepository | InferenceEngine                │  │
│  │   GpuPhysicsAdapter  | GpuSemanticAnalyzer            │  │
│  └────────────────────────┬──────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            │ (implements)
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼────────┐            ┌────────▼────────┐
    │   Adapters     │            │   Adapters      │
    │  (Neo4j/DB)    │            │  (CUDA GPU)     │
    │                │            │                 │
    │ - Neo4jSettings│            │ - CudaPhysics  │
    │ - UnifiedGraph │            │ - CudaSemantic │
    │ - UnifiedOntol.│            │                 │
    └────────────────┘            └─────────────────┘
```

## Port Categories

### 1. **Data Persistence Ports**
- `SettingsRepository` - Application/user settings management
- `KnowledgeGraphRepository` - Main knowledge graph storage
- `OntologyRepository` - OWL ontology and inference results

### 2. **Computational Ports**
- `InferenceEngine` - Ontology reasoning and classification
- `GpuPhysicsAdapter` - GPU-accelerated physics simulation
- `GpuSemanticAnalyzer` - GPU-accelerated graph algorithms

### 3. **Legacy Ports** (to be refactored)
- `GraphRepository` - Actor-based graph access (CQRS)
- `PhysicsSimulator` - Abstract physics simulation
- `SemanticAnalyzer` - CPU-based semantic analysis

## Design Principles

### 1. **Dependency Inversion**
The application core depends on **abstractions (ports)**, not concrete implementations.

```rust
// ❌ BAD: Direct dependency on concrete type
pub struct GraphService {
    db: SqliteDatabase,  // Tight coupling!
}

// ✅ GOOD: Dependency on trait (port)
pub struct GraphService {
    repo: Arc<dyn KnowledgeGraphRepository>,  // Loose coupling!
}
```

### 2. **Single Responsibility**
Each port has a single, well-defined responsibility.

### 3. **Interface Segregation**
Ports are focused and minimal - clients only depend on methods they use.

### 4. **Async First**
All port methods are async to support non-blocking I/O and GPU operations.

### 5. **Error Handling**
Each port defines its own error type with comprehensive variants.

## Port Trait Structure

Every port follows this pattern:

```rust
// 1. Error type definition
#[derive(Debug, thiserror::Error)]
pub enum MyPortError {
    #[error("Specific error: {0}")]
    SpecificError(String),
    // ... more variants
}

pub type Result<T> = std::result::Result<T, MyPortError>;

// 2. Supporting types (DTOs, enums)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyData {
    // ... fields
}

// 3. Port trait definition
#[async-trait]
pub trait MyPort: Send + Sync {
    /// Clear documentation
    async fn my-method(&self, param: &MyData) -> Result<MyData>;

    /// More methods...
}
```

## Adapter Implementation Pattern

Adapters implement port traits:

```rust
pub struct Neo4jSettingsRepository {
    graph: Arc<Graph>,
    cache: Arc<RwLock<SettingsCache>>,
    config: Neo4jSettingsConfig,
}

#[async-trait]
impl SettingsRepository for Neo4jSettingsRepository {
    async fn get-setting(&self, key: &str) -> Result<Option<SettingValue>> {
        // Check cache first
        // Query Neo4j graph database if not cached
        // Update cache and return result
    }

    // ... implement all trait methods
}
```

## Testing Strategy

### 1. **Mock Implementations**
Each port has a mock implementation for testing:

```rust
pub struct MockSettingsRepository {
    data: Arc<RwLock<HashMap<String, SettingValue>>>,
}

#[async-trait]
impl SettingsRepository for MockSettingsRepository {
    async fn get-setting(&self, key: &str) -> Result<Option<SettingValue>> {
        Ok(self.data.read().await.get(key).cloned())
    }
}
```

### 2. **Port Contract Tests**
Verify that mock implementations fulfill port contracts:

```rust
#[tokio::test]
async fn test-settings-repository-contract() {
    let repo: Box<dyn SettingsRepository> = Box::new(MockSettingsRepository::new());

    // Test all required methods
    repo.set-setting("key", SettingValue::String("value".into()), None).await.unwrap();
    let value = repo.get-setting("key").await.unwrap();
    assert-eq!(value, Some(SettingValue::String("value".into())));
}
```

## Benefits

### 1. **Testability**
- Easy to test domain logic with mock implementations
- No database required for unit tests
- Fast test execution

### 2. **Flexibility**
- Swap implementations without changing domain logic
- Support multiple storage backends
- Enable feature flags (e.g., GPU vs CPU)

### 3. **Maintainability**
- Clear separation of concerns
- Easy to understand boundaries
- Simple refactoring within adapters

### 4. **Scalability**
- Add new features by creating new ports
- Parallel development of core and infrastructure
- Independent deployment of components

## TypeScript Integration

All port types are exported to TypeScript using `specta`:

```rust
#[derive(Serialize, Deserialize, Type)]
pub struct SettingValue {
    // ... fields
}
```

Generated TypeScript:

```typescript
export type SettingValue =
  | { String: string }
  | { Integer: number }
  | { Float: number }
  | { Boolean: boolean }
  | { Json: any };
```

## Port Documentation

Each port has detailed documentation:

- **02-settings-repository.md** - Settings management
- **03-knowledge-graph-repository.md** - Graph data operations
- **04-ontology-repository.md** - OWL ontology operations
- **05-inference-engine.md** - Reasoning and classification
- **06-gpu-physics-adapter.md** - GPU physics simulation
- **07-gpu-semantic-analyzer.md** - GPU graph algorithms

## Migration Path

### Legacy Code → Hexagonal Architecture

1. **Identify dependencies** - What external systems does the code interact with?
2. **Define port traits** - Create async trait interfaces
3. **Implement adapters** - Migrate existing code into adapter implementations
4. **Update domain logic** - Inject ports instead of concrete types
5. **Add tests** - Create mock implementations and contract tests

### Example:

**Before:**
```rust
pub struct GraphService {
    db: rusqlite::Connection,
}

impl GraphService {
    pub fn load-graph(&self) -> Result<GraphData> {
        // Direct SQL queries
    }
}
```

**After:**
```rust
pub struct GraphService {
    repo: Arc<dyn KnowledgeGraphRepository>,
}

impl GraphService {
    pub async fn load-graph(&self) -> Result<Arc<GraphData>> {
        self.repo.load-graph().await
    }
}
```

## Best Practices

### 1. **Keep Ports Small**
Each port should have 5-15 methods maximum. Split large ports into smaller ones.

### 2. **Use DTOs**
Don't expose infrastructure types (e.g., `rusqlite::Row`) through ports. Use domain types.

### 3. **Batch Operations**
Provide batch methods for performance (e.g., `batch-add-nodes`).

### 4. **Health Checks**
Include `health-check()` methods for monitoring.

### 5. **Transactions**
Support transactions for data consistency.

### 6. **Clear Errors**
Define specific error variants, not generic `Error(String)`.

## Next Steps

1. Review individual port documentation
2. Understand adapter implementations
3. Write tests using mock implementations
4. Integrate ports into domain services
5. Monitor and optimize adapter performance

## References

- **Hexagonal Architecture**: https://alistair.cockburn.us/hexagonal-architecture/
- **Ports and Adapters**: https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/
- **Clean Architecture**: Robert C. Martin
- **Domain-Driven Design**: Eric Evans

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
