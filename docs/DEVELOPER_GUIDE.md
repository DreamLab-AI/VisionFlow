# VisionFlow Developer Guide

**Version:** 3.0.0
**Last Updated:** 2025-10-22
**Audience:** Backend developers contributing to VisionFlow

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Hexagonal Architecture](#hexagonal-architecture)
3. [Adding New Features](#adding-new-features)
4. [Creating Ports](#creating-ports)
5. [Implementing Adapters](#implementing-adapters)
6. [Writing CQRS Handlers](#writing-cqrs-handlers)
7. [Database Operations](#database-operations)
8. [Testing Strategies](#testing-strategies)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

**Required Tools:**
- Rust 1.70+ with Cargo
- SQLite 3.35+
- CUDA Toolkit 11.0+ (for GPU features)
- Node.js 18+ and npm (for client development)

**Recommended:**
- `cargo-watch` for automatic recompilation
- `rust-analyzer` for IDE support
- `sqlitebrowser` for database inspection

### Development Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/visionflow.git
cd visionflow

# Install Rust dependencies
cargo build --features gpu,ontology

# Initialize databases
cargo run --bin init-databases

# Migrate existing data (if upgrading)
cargo run --bin migrate-data

# Run development server
cargo run --features gpu,ontology

# In separate terminal, run client
cd client
npm install
npm run dev
```

### Project Structure

```
visionflow/
├── src/
│   ├── ports/                 # Port trait definitions
│   │   ├── settings_repository.rs
│   │   ├── knowledge_graph_repository.rs
│   │   ├── ontology_repository.rs
│   │   ├── gpu_physics_adapter.rs
│   │   ├── gpu_semantic_analyzer.rs
│   │   └── inference_engine.rs
│   ├── adapters/              # Adapter implementations
│   │   ├── sqlite_settings_repository.rs
│   │   ├── sqlite_knowledge_graph_repository.rs
│   │   ├── sqlite_ontology_repository.rs
│   │   ├── physics_orchestrator_adapter.rs
│   │   ├── semantic_processor_adapter.rs
│   │   └── whelk_inference_engine.rs
│   ├── application/           # CQRS business logic
│   │   ├── settings/
│   │   │   ├── directives.rs  # Write operations
│   │   │   └── queries.rs     # Read operations
│   │   ├── knowledge_graph/
│   │   ├── ontology/
│   │   └── mod.rs
│   ├── handlers/              # HTTP/WebSocket handlers
│   ├── actors/                # Legacy actor system (being phased out)
│   ├── models/                # Domain models
│   ├── services/              # Infrastructure services
│   └── main.rs
├── schema/                    # Database schemas
│   ├── settings_db.sql
│   ├── knowledge_graph_db.sql
│   └── ontology_db.sql
├── data/                      # Runtime databases
│   ├── settings.db
│   ├── knowledge_graph.db
│   └── ontology.db
├── client/                    # TypeScript/React frontend
└── docs/                      # Documentation

```

---

## Hexagonal Architecture

### Core Principles

VisionFlow follows **hexagonal architecture (ports and adapters)** with these principles:

1. **Business logic depends on interfaces (ports), not implementations**
2. **Infrastructure implements interfaces (adapters)**
3. **Business logic has no knowledge of databases, HTTP, or external systems**
4. **Adapters can be swapped without changing business logic**

### Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│ External (HTTP, WebSocket, Database, GPU)               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Adapters (Infrastructure - How?)                        │
│ • SqliteSettingsRepository                              │
│ • PhysicsOrchestratorAdapter                            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Ports (Interfaces - What?)                              │
│ • SettingsRepository trait                              │
│ • GpuPhysicsAdapter trait                               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Application (Business Logic - CQRS)                     │
│ • Directives (write operations)                         │
│ • Queries (read operations)                             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Domain (Pure business models)                           │
│ • Node, Edge, GraphData                                 │
│ • PhysicsSettings, SimulationParams                     │
└─────────────────────────────────────────────────────────┘
```

### Why Hexagonal Architecture?

**Benefits:**
- ✅ **Testability:** Business logic can be tested with mock adapters
- ✅ **Flexibility:** Can swap SQLite for PostgreSQL without changing business logic
- ✅ **Clarity:** Clear separation between "what" (ports) and "how" (adapters)
- ✅ **Maintainability:** Changes to infrastructure don't affect business logic

**Trade-offs:**
- ❌ More boilerplate code (traits, implementations)
- ❌ Additional abstraction layer
- ✅ **Worth it for long-term maintainability**

---

## Adding New Features

### Feature Development Workflow

**Step 1: Define Domain Model**
```rust
// src/models/my_feature.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyFeature {
    pub id: u32,
    pub name: String,
    pub properties: HashMap<String, String>,
}
```

**Step 2: Create Port (Interface)**
```rust
// src/ports/my_feature_repository.rs
use async_trait::async_trait;

#[async_trait]
pub trait MyFeatureRepository: Send + Sync {
    async fn get_feature(&self, id: u32) -> Result<Option<MyFeature>, String>;
    async fn save_feature(&self, feature: &MyFeature) -> Result<(), String>;
    async fn list_features(&self) -> Result<Vec<MyFeature>, String>;
}
```

**Step 3: Implement Adapter**
```rust
// src/adapters/sqlite_my_feature_repository.rs
use async_trait::async_trait;
use crate::ports::my_feature_repository::MyFeatureRepository;

pub struct SqliteMyFeatureRepository {
    db: Arc<DatabaseService>,
}

impl SqliteMyFeatureRepository {
    pub fn new(db: Arc<DatabaseService>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl MyFeatureRepository for SqliteMyFeatureRepository {
    async fn get_feature(&self, id: u32) -> Result<Option<MyFeature>, String> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            // SQLite query implementation
            let conn = db.connection.lock().unwrap();
            // ... query logic
        })
        .await
        .map_err(|e| format!("Task error: {}", e))?
    }

    async fn save_feature(&self, feature: &MyFeature) -> Result<(), String> {
        let db = self.db.clone();
        let feature = feature.clone();
        tokio::task::spawn_blocking(move || {
            // SQLite insert/update implementation
            let conn = db.connection.lock().unwrap();
            // ... insert logic
        })
        .await
        .map_err(|e| format!("Task error: {}", e))?
    }

    async fn list_features(&self) -> Result<Vec<MyFeature>, String> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            // SQLite select all implementation
            let conn = db.connection.lock().unwrap();
            // ... select logic
        })
        .await
        .map_err(|e| format!("Task error: {}", e))?
    }
}
```

**Step 4: Define CQRS Operations**

**Directives (Write Operations):**
```rust
// src/application/my_feature/directives.rs
use hexser::{Directive, DirectiveHandler};

#[derive(Debug, Clone, Directive)]
pub struct SaveFeature {
    pub feature: MyFeature,
}

pub struct SaveFeatureHandler<R: MyFeatureRepository> {
    repository: R,
}

impl<R: MyFeatureRepository> SaveFeatureHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl<R: MyFeatureRepository> DirectiveHandler<SaveFeature>
    for SaveFeatureHandler<R>
{
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: SaveFeature)
        -> Result<Self::Output, Self::Error>
    {
        log::info!("Saving feature: {}", directive.feature.name);
        self.repository.save_feature(&directive.feature).await?;
        log::info!("Feature saved successfully");
        Ok(())
    }
}
```

**Queries (Read Operations):**
```rust
// src/application/my_feature/queries.rs
use hexser::{Query, QueryHandler};

#[derive(Debug, Clone, Query)]
pub struct GetFeature {
    pub id: u32,
}

pub struct GetFeatureHandler<R: MyFeatureRepository> {
    repository: R,
}

impl<R: MyFeatureRepository> GetFeatureHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl<R: MyFeatureRepository> QueryHandler<GetFeature>
    for GetFeatureHandler<R>
{
    type Output = Option<MyFeature>;
    type Error = String;

    async fn handle(&self, query: GetFeature)
        -> Result<Self::Output, Self::Error>
    {
        self.repository.get_feature(query.id).await
    }
}
```

**Step 5: Create HTTP Endpoints**
```rust
// src/handlers/my_feature_handler.rs
use actix_web::{web, HttpResponse, Error};

pub async fn get_feature(
    path: web::Path<u32>,
    services: web::Data<ApplicationServices>,
) -> Result<HttpResponse, Error> {
    let query = GetFeature { id: path.into_inner() };

    let result = services.my_feature
        .get_feature_handler
        .handle(query)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    Ok(HttpResponse::Ok().json(result))
}

pub async fn save_feature(
    body: web::Json<MyFeature>,
    services: web::Data<ApplicationServices>,
) -> Result<HttpResponse, Error> {
    let directive = SaveFeature {
        feature: body.into_inner(),
    };

    services.my_feature
        .save_feature_handler
        .handle(directive)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    Ok(HttpResponse::Ok().json(json!({"status": "ok"})))
}

// Register routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/features")
            .route("/{id}", web::get().to(get_feature))
            .route("", web::post().to(save_feature))
    );
}
```

**Step 6: Update AppState and ApplicationServices**
```rust
// src/app_state.rs
pub struct ApplicationServices {
    pub settings: SettingsServices,
    pub knowledge_graph: KnowledgeGraphServices,
    pub my_feature: MyFeatureServices, // NEW
}

pub struct MyFeatureServices {
    pub get_feature_handler: GetFeatureHandler<SqliteMyFeatureRepository>,
    pub save_feature_handler: SaveFeatureHandler<SqliteMyFeatureRepository>,
}

impl AppState {
    pub fn new() -> Self {
        // Initialize databases
        let settings_db = Arc::new(DatabaseService::new("data/settings.db").unwrap());
        // ... other databases

        // Initialize adapters
        let my_feature_repo = SqliteMyFeatureRepository::new(settings_db.clone());

        // Initialize handlers
        let my_feature_services = MyFeatureServices {
            get_feature_handler: GetFeatureHandler::new(my_feature_repo.clone()),
            save_feature_handler: SaveFeatureHandler::new(my_feature_repo),
        };

        let services = ApplicationServices {
            // ... other services
            my_feature: my_feature_services,
        };

        Self {
            services: Arc::new(services),
            // ... other fields
        }
    }
}
```

**Step 7: Add Database Schema**
```sql
-- schema/settings_db.sql (or appropriate database)

CREATE TABLE IF NOT EXISTS my_features (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    properties TEXT, -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_my_features_name ON my_features(name);
```

**Step 8: Write Tests**
```rust
// tests/my_feature_tests.rs

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::mock;

    mock! {
        MyFeatureRepo {}
        #[async_trait]
        impl MyFeatureRepository for MyFeatureRepo {
            async fn get_feature(&self, id: u32) -> Result<Option<MyFeature>, String>;
            async fn save_feature(&self, feature: &MyFeature) -> Result<(), String>;
        }
    }

    #[tokio::test]
    async fn test_save_feature() {
        let mut mock_repo = MockMyFeatureRepo::new();
        mock_repo
            .expect_save_feature()
            .times(1)
            .returning(|_| Ok(()));

        let handler = SaveFeatureHandler::new(mock_repo);
        let directive = SaveFeature {
            feature: MyFeature {
                id: 1,
                name: "test".to_string(),
                properties: HashMap::new(),
            },
        };

        let result = handler.handle(directive).await;
        assert!(result.is_ok());
    }
}
```

---

## Creating Ports

### Port Design Guidelines

**Ports should:**
- ✅ Be async-first (`#[async_trait]`)
- ✅ Be thread-safe (`Send + Sync`)
- ✅ Use domain types, not infrastructure types
- ✅ Return `Result<T, String>` for error handling
- ✅ Be focused on a single responsibility

**Example Port:**
```rust
use async_trait::async_trait;

/// Repository for managing user preferences
#[async_trait]
pub trait UserPreferencesRepository: Send + Sync {
    /// Get all preferences for a user
    async fn get_preferences(&self, user_id: &str)
        -> Result<UserPreferences, String>;

    /// Update specific preference
    async fn set_preference(&self, user_id: &str, key: &str, value: String)
        -> Result<(), String>;

    /// Delete all preferences for a user
    async fn delete_preferences(&self, user_id: &str)
        -> Result<(), String>;
}
```

### Common Port Patterns

#### Repository Pattern (Data Access)
```rust
#[async_trait]
pub trait EntityRepository: Send + Sync {
    async fn get(&self, id: u32) -> Result<Option<Entity>, String>;
    async fn save(&self, entity: &Entity) -> Result<(), String>;
    async fn delete(&self, id: u32) -> Result<(), String>;
    async fn list(&self, filter: Option<Filter>) -> Result<Vec<Entity>, String>;
}
```

#### Service Pattern (Business Operations)
```rust
#[async_trait]
pub trait EmailService: Send + Sync {
    async fn send_email(&self, to: &str, subject: &str, body: &str)
        -> Result<(), String>;
    async fn send_batch(&self, emails: Vec<Email>) -> Result<(), String>;
}
```

#### Adapter Pattern (External Systems)
```rust
#[async_trait]
pub trait CacheAdapter: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String>;
    async fn set(&self, key: &str, value: Vec<u8>, ttl_seconds: u64)
        -> Result<(), String>;
    async fn invalidate(&self, key: &str) -> Result<(), String>;
}
```

---

## Implementing Adapters

### Adapter Implementation Guidelines

**Adapters should:**
- ✅ Implement one or more port traits
- ✅ Handle all infrastructure concerns (DB connections, error mapping)
- ✅ Use `tokio::task::spawn_blocking` for blocking operations
- ✅ Implement proper error handling and logging
- ✅ Be testable in isolation (integration tests)

### SQLite Adapter Pattern

```rust
use async_trait::async_trait;
use std::sync::Arc;
use rusqlite::Connection;

pub struct SqliteEntityRepository {
    db: Arc<Mutex<Connection>>,
}

impl SqliteEntityRepository {
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        // Set pragmas for performance
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = 10000;
            PRAGMA foreign_keys = ON;
        ")
        .map_err(|e| format!("Failed to set pragmas: {}", e))?;

        Ok(Self {
            db: Arc::new(Mutex::new(conn)),
        })
    }

    /// Initialize schema (idempotent)
    pub fn initialize_schema(&self) -> Result<(), String> {
        let conn = self.db.lock().unwrap();
        conn.execute_batch(include_str!("../schema/entities.sql"))
            .map_err(|e| format!("Schema initialization failed: {}", e))
    }
}

#[async_trait]
impl EntityRepository for SqliteEntityRepository {
    async fn get(&self, id: u32) -> Result<Option<Entity>, String> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT id, name, data FROM entities WHERE id = ?")
                .map_err(|e| format!("Prepare failed: {}", e))?;

            let result = stmt
                .query_row([id], |row| {
                    Ok(Entity {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        data: serde_json::from_str(row.get::<_, String>(2)?.as_str())
                            .unwrap(),
                    })
                })
                .optional()
                .map_err(|e| format!("Query failed: {}", e))?;

            Ok(result)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    async fn save(&self, entity: &Entity) -> Result<(), String> {
        let db = self.db.clone();
        let entity = entity.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.lock().unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO entities (id, name, data, updated_at)
                 VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                rusqlite::params![
                    entity.id,
                    &entity.name,
                    serde_json::to_string(&entity.data).unwrap(),
                ],
            )
            .map_err(|e| format!("Insert failed: {}", e))?;

            Ok(())
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }
}
```

### Actor Adapter Pattern

When wrapping existing actors as adapters:

```rust
use actix::prelude::*;

pub struct ActorPhysicsAdapter {
    actor_addr: Addr<PhysicsOrchestratorActor>,
}

impl ActorPhysicsAdapter {
    pub fn new(actor_addr: Addr<PhysicsOrchestratorActor>) -> Self {
        Self { actor_addr }
    }
}

#[async_trait]
impl GpuPhysicsAdapter for ActorPhysicsAdapter {
    async fn simulate_step(&mut self, params: &SimulationParams)
        -> Result<PhysicsStepResult, String>
    {
        // Send message to actor, await response
        self.actor_addr
            .send(SimulateStepMessage {
                params: params.clone(),
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Actor error: {}", e))
    }

    async fn get_positions(&self) -> Result<Vec<(u32, f32, f32, f32)>, String> {
        self.actor_addr
            .send(GetPositionsMessage)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Actor error: {}", e))
    }
}
```

---

## Writing CQRS Handlers

### Directive Handlers (Write Operations)

**Guidelines:**
- Use for operations that modify state
- Log important operations
- Validate input before executing
- Return `()` on success, `String` error on failure
- Consider emitting events for audit trail

**Template:**
```rust
use hexser::{Directive, DirectiveHandler};
use async_trait::async_trait;

#[derive(Debug, Clone, Directive)]
pub struct MyDirective {
    pub field1: String,
    pub field2: i32,
}

pub struct MyDirectiveHandler<R: MyRepository> {
    repository: R,
}

impl<R: MyRepository> MyDirectiveHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl<R: MyRepository> DirectiveHandler<MyDirective>
    for MyDirectiveHandler<R>
{
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: MyDirective)
        -> Result<Self::Output, Self::Error>
    {
        log::info!("Executing MyDirective: {:?}", directive);

        // Validate input
        if directive.field2 < 0 {
            return Err("field2 must be non-negative".to_string());
        }

        // Execute business logic
        self.repository
            .some_operation(&directive.field1, directive.field2)
            .await?;

        log::info!("MyDirective completed successfully");

        // Optional: Emit event for audit trail
        // self.event_bus.emit(MyDirectiveExecuted { ... }).await;

        Ok(())
    }
}
```

### Query Handlers (Read Operations)

**Guidelines:**
- Use for operations that read state
- Should not modify state (idempotent)
- Can be cached aggressively
- Return data directly

**Template:**
```rust
use hexser::{Query, QueryHandler};
use async_trait::async_trait;

#[derive(Debug, Clone, Query)]
pub struct MyQuery {
    pub filter: Option<String>,
}

pub struct MyQueryHandler<R: MyRepository> {
    repository: R,
}

impl<R: MyRepository> MyQueryHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl<R: MyRepository> QueryHandler<MyQuery>
    for MyQueryHandler<R>
{
    type Output = Vec<MyData>;
    type Error = String;

    async fn handle(&self, query: MyQuery)
        -> Result<Self::Output, Self::Error>
    {
        log::debug!("Executing MyQuery: {:?}", query);

        let results = self.repository
            .query_data(query.filter.as_deref())
            .await?;

        log::debug!("MyQuery returned {} results", results.len());

        Ok(results)
    }
}
```

---

## Database Operations

### Choosing the Right Database

VisionFlow uses **three separate databases**:

| Database | Purpose | Use When |
|----------|---------|----------|
| `settings.db` | Configuration, preferences, physics settings | Storing user/developer/system config |
| `knowledge_graph.db` | Main graph structure from local markdown | Storing nodes, edges, graph state |
| `ontology.db` | Ontology graph from GitHub markdown | Storing OWL classes, properties, axioms |

### Database Schema Updates

**Process:**
1. Update SQL schema file in `/schema/`
2. Increment schema version
3. Create migration script
4. Test migration on dev data
5. Document breaking changes

**Example Migration:**
```sql
-- schema/settings_db.sql

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Migration: Add new column (idempotent)
ALTER TABLE settings ADD COLUMN IF NOT EXISTS namespace TEXT;

-- Insert migration record
INSERT OR IGNORE INTO schema_version (version) VALUES (2);
```

### Performance Optimization

**SQLite Pragmas (Already Set):**
```sql
PRAGMA journal_mode = WAL;         -- Write-Ahead Logging for concurrency
PRAGMA synchronous = NORMAL;       -- Balance between safety and speed
PRAGMA cache_size = 10000;         -- 10000 pages (~40MB cache)
PRAGMA foreign_keys = ON;          -- Enforce referential integrity
PRAGMA temp_store = MEMORY;        -- Store temp tables in RAM
```

**Indexing Strategy:**
```sql
-- Index frequently queried columns
CREATE INDEX IF NOT EXISTS idx_nodes_metadata_id ON nodes(metadata_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

-- Composite indexes for common joins
CREATE INDEX IF NOT EXISTS idx_nodes_type_created
    ON nodes(type, created_at DESC);
```

**Query Optimization:**
```rust
// Use prepared statements (automatically cached)
let mut stmt = conn.prepare_cached(
    "SELECT id, name FROM nodes WHERE type = ? LIMIT ?"
)?;

// Use transactions for batch operations
let tx = conn.transaction()?;
for node in nodes {
    tx.execute("INSERT INTO nodes ...", params![...])?;
}
tx.commit()?;

// Use EXPLAIN QUERY PLAN to analyze slow queries
let plan = conn.prepare("EXPLAIN QUERY PLAN SELECT ...")?
    .query_map([], |row| row.get::<_, String>(3))?;
```

---

## Testing Strategies

### Unit Testing with Mock Adapters

Use `mockall` crate for mocking ports:

```rust
use mockall::{predicate::*, mock};

mock! {
    MyRepo {}
    #[async_trait]
    impl MyRepository for MyRepo {
        async fn get_data(&self, id: u32) -> Result<Option<MyData>, String>;
        async fn save_data(&self, data: &MyData) -> Result<(), String>;
    }
}

#[tokio::test]
async fn test_my_directive_handler() {
    // Arrange
    let mut mock_repo = MockMyRepo::new();
    mock_repo
        .expect_save_data()
        .with(eq(MyData { id: 1, name: "test".to_string() }))
        .times(1)
        .returning(|_| Ok(()));

    let handler = MyDirectiveHandler::new(mock_repo);

    // Act
    let directive = MyDirective { id: 1, name: "test".to_string() };
    let result = handler.handle(directive).await;

    // Assert
    assert!(result.is_ok());
}
```

### Integration Testing with Real Databases

```rust
#[tokio::test]
async fn test_sqlite_repository_integration() {
    // Use temporary database
    let temp_db = tempfile::NamedTempFile::new().unwrap();
    let repo = SqliteMyRepository::new(temp_db.path().to_str().unwrap()).unwrap();
    repo.initialize_schema().unwrap();

    // Test operations
    let data = MyData { id: 1, name: "test".to_string() };
    repo.save_data(&data).await.unwrap();

    let retrieved = repo.get_data(1).await.unwrap();
    assert_eq!(retrieved, Some(data));
}
```

### End-to-End Testing

```rust
use actix_web::test;

#[actix_web::test]
async fn test_endpoint_e2e() {
    // Initialize app with real dependencies
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(AppState::new()))
            .configure(configure_routes)
    ).await;

    // Test POST request
    let req = test::TestRequest::post()
        .uri("/api/features")
        .set_json(MyFeature { id: 1, name: "test".to_string() })
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    // Test GET request
    let req = test::TestRequest::get()
        .uri("/api/features/1")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    let body: MyFeature = test::read_body_json(resp).await;
    assert_eq!(body.name, "test");
}
```

---

## Common Patterns

### Async Blocking Operations

**Problem:** SQLite operations are blocking (not async-friendly)

**Solution:** Use `tokio::task::spawn_blocking`

```rust
async fn get_data(&self, id: u32) -> Result<Option<MyData>, String> {
    let db = self.db.clone();
    tokio::task::spawn_blocking(move || {
        let conn = db.lock().unwrap();
        // Blocking SQLite operations here
        conn.query_row(...)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}
```

### Error Handling

**Pattern:** Convert infrastructure errors to domain errors

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DomainError {
    #[error("Entity not found: {0}")]
    NotFound(u32),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Validation error: {0}")]
    Validation(String),
}

impl From<rusqlite::Error> for DomainError {
    fn from(err: rusqlite::Error) -> Self {
        DomainError::Database(err.to_string())
    }
}
```

### Caching Pattern

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct CachingRepository<R: Repository> {
    inner: R,
    cache: Arc<RwLock<HashMap<u32, CachedValue>>>,
    ttl_seconds: u64,
}

impl<R: Repository> CachingRepository<R> {
    pub fn new(inner: R, ttl_seconds: u64) -> Self {
        Self {
            inner,
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl_seconds,
        }
    }

    async fn get_from_cache(&self, id: u32) -> Option<MyData> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(&id) {
            if cached.timestamp.elapsed().as_secs() < self.ttl_seconds {
                return Some(cached.value.clone());
            }
        }
        None
    }
}

#[async_trait]
impl<R: Repository> Repository for CachingRepository<R> {
    async fn get_data(&self, id: u32) -> Result<Option<MyData>, String> {
        // Check cache first
        if let Some(cached) = self.get_from_cache(id).await {
            return Ok(Some(cached));
        }

        // Fetch from inner repository
        let result = self.inner.get_data(id).await?;

        // Update cache
        if let Some(ref data) = result {
            let mut cache = self.cache.write().await;
            cache.insert(id, CachedValue {
                value: data.clone(),
                timestamp: Instant::now(),
            });
        }

        Ok(result)
    }
}
```

---

## Troubleshooting

### Common Issues

#### Issue: "Database is locked"

**Cause:** Multiple threads trying to write simultaneously

**Solution:** Ensure WAL mode is enabled
```rust
conn.execute_batch("PRAGMA journal_mode = WAL;")?;
```

#### Issue: "Task join error" in async operations

**Cause:** Panic in `spawn_blocking` task

**Solution:** Add better error handling
```rust
tokio::task::spawn_blocking(move || {
    // Wrap in Result to catch panics
    std::panic::catch_unwind(|| {
        // Your blocking code here
    })
    .map_err(|_| "Panic in blocking task".to_string())
})
.await
.map_err(|e| format!("Task join error: {}", e))??
```

#### Issue: "Actor mailbox error"

**Cause:** Actor has stopped or mailbox is full

**Solution:** Check actor lifecycle and increase mailbox size
```rust
let addr = SyncArbiter::start(4, || MyActor::default())
    .recipient();
```

### Debugging Tips

**Enable SQL query logging:**
```rust
conn.trace(Some(|stmt| {
    eprintln!("SQL: {}", stmt);
}));
```

**Check database integrity:**
```bash
sqlite3 data/settings.db "PRAGMA integrity_check;"
```

**Analyze query performance:**
```bash
sqlite3 data/settings.db "EXPLAIN QUERY PLAN SELECT ..."
```

**Monitor database size:**
```bash
du -h data/*.db
sqlite3 data/settings.db "PRAGMA page_count; PRAGMA page_size;"
```

---

## Additional Resources

- [VisionFlow Architecture Overview](/docs/ARCHITECTURE.md)
- [API Documentation](/docs/API.md)
- [Database Documentation](/docs/DATABASE.md)
- [Client Integration Guide](/docs/CLIENT_INTEGRATION.md)
- [hexser Documentation](https://docs.rs/hexser)
- [actix-web Documentation](https://actix.rs/docs/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)

---

**Document Maintained By:** VisionFlow Development Team
**Last Review:** 2025-10-22
**Next Review:** 2025-11-22

