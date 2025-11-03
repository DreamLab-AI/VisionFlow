# Todo List: Full Migration to Neo4j

This document outlines the tasks required to migrate the application's persistence layer entirely to Neo4j, deprecating all SQLite-based components. The tasks are designed to be executed in parallel where possible by a swarm of coding agents.

## Phase 1: Deprecate SQL-based Graph Repository

The goal of this phase is to make `Neo4jAdapter` the sole implementation for the `KnowledgeGraphRepository`.

### Task 1.1: Update Application State to use Neo4j Exclusively for Graph Data

**File to modify:** [`src/app_state.rs`](src/app_state.rs)

**Objective:** Remove the `DualGraphRepository` and `UnifiedGraphRepository`, and configure the application to use `Neo4jAdapter` directly as the implementation for `KnowledgeGraphRepository`.

**Instructions:**

1.  Locate the initialization of `knowledge_graph_repository` and `graph_repository_with_neo4j`.
2.  Remove the `DualGraphRepository` and `UnifiedGraphRepository` initializations.
3.  Instantiate `Neo4jAdapter` and assign it to a field, ensuring it's used as the definitive `KnowledgeGraphRepository`.
4.  Update any dependent initializations, such as `ActorGraphRepository`, to use the `Neo4jAdapter`.

### Task 1.2: Remove `DualGraphRepository`

**File to delete:** [`src/adapters/dual_graph_repository.rs`](src/adapters/dual_graph_repository.rs)

**Objective:** Delete the `DualGraphRepository` implementation file.

**Instructions:**

1.  Delete the file `src/adapters/dual_graph_repository.rs`.
2.  Update `src/adapters/mod.rs` to remove the module declaration for `dual_graph_repository`.

### Task 1.3: Remove `UnifiedGraphRepository`

**File to delete:** [`src/repositories/unified_graph_repository.rs`](src/repositories/unified_graph_repository.rs)

**Objective:** Delete the `UnifiedGraphRepository` implementation file.

**Instructions:**

1.  Delete the file `src/repositories/unified_graph_repository.rs`.
2.  Update `src/repositories/mod.rs` to remove the module declaration for `unified_graph_repository`.

## Phase 2: Migrate Settings to Neo4j

This phase involves creating a new repository for settings that uses Neo4j and migrating the data from SQLite.

### Task 2.1: Create `Neo4jSettingsRepository`

**File to create:** [`src/adapters/neo4j_settings_repository.rs`](src/adapters/neo4j_settings_repository.rs)

**Objective:** Implement the `SettingsRepository` trait using Neo4j as the backend.

**Instructions:**

1.  Create a new file `src/adapters/neo4j_settings_repository.rs`.
2.  Implement the `SettingsRepository` trait for a new `Neo4jSettingsRepository` struct.
3.  Model settings in Neo4j. A good approach is to use a single `:SettingsRoot` node as an entry point, connected to nodes representing settings categories (e.g., `:PhysicsSettings`, `:RenderingSettings`). Store individual settings as properties on these nodes.
4.  Implement all methods of the `SettingsRepository` trait using Cypher queries.

**Example Cypher for creating a setting:**

```cypher
MERGE (s:SettingsRoot {id: "default"})
MERGE (p:PhysicsSettings)
ON CREATE SET p.damping = 0.8, p.springK = 0.01
ON MATCH SET p.damping = 0.8, p.springK = 0.01
MERGE (s)-[:HAS_PHYSICS_SETTINGS]->(p)
```

### Task 2.2: Create a Data Migration Script

**File to create:** [`src/bin/migrate_settings_to_neo4j.rs`](src/bin/migrate_settings_to_neo4j.rs)

**Objective:** Write a one-time script to migrate settings from the SQLite database to Neo4j.

**Instructions:**

1.  Create a new binary in the `src/bin` directory.
2.  The script should:
    a.  Connect to the SQLite database using `SqliteSettingsRepository`.
    b.  Load all existing settings.
    c.  Connect to Neo4j using the new `Neo4jSettingsRepository`.
    d.  Write the loaded settings into Neo4j using the data model defined in `Neo4jSettingsRepository`.
3.  Add the new binary to `Cargo.toml`.

### Task 2.3: Update Application State to use `Neo4jSettingsRepository`

**File to modify:** [`src/app_state.rs`](src/app_state.rs)

**Objective:** Replace the `SqliteSettingsRepository` with the new `Neo4jSettingsRepository`.

**Instructions:**

1.  In `src/app_state.rs`, find the instantiation of `SqliteSettingsRepository`.
2.  Replace it with the instantiation of `Neo4jSettingsRepository`.

## Phase 3: Code Cleanup and Finalization

This phase involves removing all legacy SQLite code and dependencies.

### Task 3.1: Delete `SqliteSettingsRepository`

**File to delete:** [`src/adapters/sqlite_settings_repository.rs`](src/adapters/sqlite_settings_repository.rs)

**Objective:** Remove the SQLite implementation for settings.

**Instructions:**

1.  Delete the file `src/adapters/sqlite_settings_repository.rs`.
2.  Update `src/adapters/mod.rs` to remove the module declaration for `sqlite_settings_repository`.

### Task 3.2: Delete SQL Migration Files

**Files to delete:**

*   [`src/migrations/006_settings_tables.sql`](src/migrations/006_settings_tables.sql)
*   The entire `src/migrations` directory.

**Objective:** Remove all SQL migration files and the migration runner binary.

**Instructions:**

1.  Delete the `src/migrations` directory.
2.  Delete the migration runner binary: `src/bin/migrate.rs`.
3.  Remove the `migrate` binary from `Cargo.toml`.

### Task 3.3: Update `Cargo.toml`

**File to modify:** [`Cargo.toml`](Cargo.toml)

**Objective:** Remove SQLite dependencies and make Neo4j a non-optional dependency.

**Instructions:**

1.  Remove `rusqlite`, `r2d2`, and `r2d2_sqlite` from the `[dependencies]` section.
2.  In the `[dependencies]` section, make `neo4rs` a non-optional dependency by removing `optional = true`.
3.  In the `[features]` section, remove the `neo4j` feature definition and add `dep:neo4rs` to the `default` features if it's not already there.

### Task 3.4: Remove `generic_repository.rs`

**File to delete:** [`src/repositories/generic_repository.rs`](src/repositories/generic_repository.rs)

**Objective:** Remove the generic SQLite repository base, as it will no longer be needed.

**Instructions:**

1.  Delete the file `src/repositories/generic_repository.rs`.
2.  Update `src/repositories/mod.rs` to remove the module declaration for `generic_repository`.

## Phase 4: Verification

**Objective:** Ensure the application runs correctly with the new Neo4j-only persistence layer.

**Instructions:**

1.  Run all existing tests to ensure they pass.
2.  Create new integration tests for the `Neo4jSettingsRepository`.
3.  Manually run the application and test all functionality related to graph data and settings to confirm everything works as expected.