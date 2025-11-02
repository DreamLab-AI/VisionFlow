# Unified Database Schema (unified.db)

## Overview

The unified.db SQLite database combines both **graph data** (knowledge graph nodes/edges) and **ontology data** (OWL classes, properties, axioms) in a single database. This unified architecture enables efficient cross-referencing between graph structures and ontology definitions.

**Location**: `data/unified.db`
**Foreign Keys**: Enabled (`PRAGMA foreign_keys = ON`)

---

## Table Schemas

### 1. `graph_nodes` - Knowledge Graph Nodes

Stores nodes representing entities in the knowledge graph with 3D spatial coordinates and metadata.

```sql
CREATE TABLE IF NOT EXISTS graph_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metadata_id TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    x REAL NOT NULL DEFAULT 0.0,
    y REAL NOT NULL DEFAULT 0.0,
    z REAL NOT NULL DEFAULT 0.0,
    vx REAL NOT NULL DEFAULT 0.0,
    vy REAL NOT NULL DEFAULT 0.0,
    vz REAL NOT NULL DEFAULT 0.0,
    mass REAL NOT NULL DEFAULT 1.0,
    charge REAL NOT NULL DEFAULT 0.0,
    owl_class_iri TEXT,
    color TEXT,
    size REAL DEFAULT 10.0,
    node_type TEXT,
    weight REAL DEFAULT 1.0,
    group_name TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
- `idx_graph_nodes_metadata_id` on `metadata_id`

**Purpose**: Stores knowledge graph nodes with spatial coordinates for 3D visualization and CUDA physics simulation.

**Populated By**: `GitHubSyncService` → `KnowledgeGraphParser` → `UnifiedGraphRepository.save_graph()`

**Key Fields**:
- `id`: Auto-incrementing primary key used for node references
- `metadata_id`: Unique string identifier (e.g., page name from markdown)
- `x, y, z`: 3D spatial coordinates for graph layout
- `vx, vy, vz`: Velocity vectors for physics simulation
- `owl_class_iri`: Optional reference to OWL ontology class (links graph ↔ ontology)
- `metadata`: JSON-encoded additional properties

---

### 2. `graph_edges` - Knowledge Graph Edges

Stores relationships between graph nodes.

```sql
CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    relation_type TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
);
```

**Indexes**:
- `idx_graph_edges_source` on `source_id`
- `idx_graph_edges_target` on `target_id`

**Purpose**: Stores directed relationships between nodes with cascading deletes.

**Populated By**: `KnowledgeGraphParser` → `UnifiedGraphRepository.save_graph()`

**Key Fields**:
- `id`: String identifier (format: `"source-target"`)
- `source_id`, `target_id`: Foreign keys to `graph_nodes.id`
- `weight`: Edge strength/importance (default 1.0)
- `relation_type`: Semantic type of relationship (e.g., "linked_page", "tag")

---

### 3. `graph_statistics` - Cached Graph Metrics

Singleton table storing precomputed graph statistics for performance.

```sql
CREATE TABLE IF NOT EXISTS graph_statistics (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    node_count INTEGER NOT NULL DEFAULT 0,
    edge_count INTEGER NOT NULL DEFAULT 0,
    average_degree REAL NOT NULL DEFAULT 0.0,
    connected_components INTEGER NOT NULL DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Purpose**: Caches expensive graph metrics computations.

**Updated By**: `UnifiedGraphRepository.update_statistics_cache()`

---

### 4. `file_metadata` - GitHub Sync Tracking

Tracks processed markdown files for incremental sync (SHA1 change detection).

```sql
CREATE TABLE IF NOT EXISTS file_metadata (
    file_name TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    file_blob_sha TEXT,
    github_node_id TEXT,
    sha1 TEXT,
    content_hash TEXT,
    last_modified DATETIME,
    last_content_change DATETIME,
    last_commit DATETIME,
    change_count INTEGER DEFAULT 0,
    processing_status TEXT DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Purpose**: Enables incremental sync by tracking file SHA1 hashes (only reprocess changed files).

**Populated By**: `GitHubSyncService.update_file_metadata()`

**Key Fields**:
- `file_name`: Primary key (e.g., `"example.md"`)
- `file_blob_sha`: GitHub blob SHA for change detection
- `change_count`: Increments each time file content changes
- `processing_status`: `'pending'` | `'complete'` | `'error'`

---

### 5. `owl_classes` - OWL Ontology Classes

Stores OWL class definitions from ontology markdown files.

```sql
CREATE TABLE IF NOT EXISTS owl_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT DEFAULT 'default',
    iri TEXT UNIQUE NOT NULL,
    label TEXT,
    description TEXT,
    file_sha1 TEXT,
    last_synced INTEGER,
    markdown_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
- `idx_owl_classes_iri` on `iri`
- `idx_owl_classes_ontology_id` on `ontology_id`

**Purpose**: Stores OWL ontology class definitions with markdown source tracking.

**Populated By**: `GitHubSyncService` → `OntologyParser` → `UnifiedOntologyRepository.save_ontology()`

**Key Fields**:
- `iri`: Internationalized Resource Identifier (unique class URI)
- `ontology_id`: Namespace/ontology grouping (default: `'default'`)
- `markdown_content`: Original markdown source for the class
- `file_sha1`: SHA1 hash for change detection

---

### 6. `owl_class_hierarchy` - OWL Class Inheritance

Stores parent-child relationships between OWL classes.

```sql
CREATE TABLE IF NOT EXISTS owl_class_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_iri TEXT NOT NULL,
    parent_iri TEXT NOT NULL,
    UNIQUE(class_iri, parent_iri),
    FOREIGN KEY (class_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE,
    FOREIGN KEY (parent_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE
);
```

**Indexes**:
- `idx_owl_hierarchy_class` on `class_iri`
- `idx_owl_hierarchy_parent` on `parent_iri`

**Purpose**: Represents subclass-of relationships (OWL `rdfs:subClassOf`).

**Populated By**: `OntologyParser` → `UnifiedOntologyRepository.save_ontology()`

---

### 7. `owl_properties` - OWL Properties

Stores OWL property definitions (object, data, annotation properties).

```sql
CREATE TABLE IF NOT EXISTS owl_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT DEFAULT 'default',
    iri TEXT UNIQUE NOT NULL,
    label TEXT,
    property_type TEXT NOT NULL,
    domain TEXT,
    range TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
- `idx_owl_properties_iri` on `iri`

**Purpose**: Stores OWL property metadata with domain/range constraints.

**Populated By**: `OntologyParser` → `UnifiedOntologyRepository.save_ontology()`

**Key Fields**:
- `property_type`: `'ObjectProperty'` | `'DataProperty'` | `'AnnotationProperty'`
- `domain`: JSON-encoded list of valid subject classes
- `range`: JSON-encoded list of valid object types/classes

---

### 8. `owl_axioms` - OWL Axioms

Stores OWL logical statements and assertions.

```sql
CREATE TABLE IF NOT EXISTS owl_axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT DEFAULT 'default',
    axiom_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    object TEXT NOT NULL,
    annotations TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
- `idx_owl_axioms_type` on `axiom_type`
- `idx_owl_axioms_subject` on `subject`

**Purpose**: Stores logical assertions and constraints in the ontology.

**Populated By**: `OntologyParser` → `UnifiedOntologyRepository.save_ontology()`

**Key Fields**:
- `axiom_type`: `'SubClassOf'` | `'EquivalentClass'` | `'DisjointWith'` | `'ObjectPropertyAssertion'` | `'DataPropertyAssertion'`
- `subject`: Source entity IRI
- `object`: Target entity IRI
- `annotations`: JSON-encoded metadata about the axiom

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ GitHub API (Markdown Repository)                                 │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ EnhancedContentAPI                                               │
│  - Fetches markdown files recursively                            │
│  - Returns GitHubFileBasicMetadata with SHA1 hashes              │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ GitHubSyncService (Orchestrator)                                 │
│  1. Fetches all markdown files                                   │
│  2. SHA1 filtering (skip unchanged files)                        │
│  3. Processes in batches (BATCH_SIZE = 50)                       │
│  4. Detects file type: KnowledgeGraph | Ontology | Skip          │
└────────┬───────────────────────────────────┬─────────────────────┘
         │                                   │
         │ Knowledge Graph Files             │ Ontology Files
         │ (public:: true)                   │ (### OntologyBlock)
         ▼                                   ▼
┌─────────────────────────┐      ┌─────────────────────────────────┐
│ KnowledgeGraphParser    │      │ OntologyParser                  │
│  - Parses markdown      │      │  - Extracts OWL classes         │
│  - Extracts nodes       │      │  - Extracts properties          │
│  - Extracts edges       │      │  - Extracts axioms              │
│  - Detects [[links]]    │      │  - Builds class hierarchy       │
└────────┬────────────────┘      └────────┬────────────────────────┘
         │                                │
         │ Returns: GraphData             │ Returns: OntologyData
         │  - Vec<Node>                   │  - Vec<OwlClass>
         │  - Vec<Edge>                   │  - Vec<OwlProperty>
         │                                │  - Vec<OwlAxiom>
         ▼                                ▼
┌─────────────────────────┐      ┌─────────────────────────────────┐
│ UnifiedGraphRepository  │      │ UnifiedOntologyRepository       │
│  .save_graph()          │      │  .save_ontology()               │
└────────┬────────────────┘      └────────┬────────────────────────┘
         │                                │
         ▼                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ unified.db (SQLite)                                              │
│                                                                  │
│ ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│ │ graph_nodes    │  │ owl_classes    │  │ file_metadata      │ │
│ │ graph_edges    │  │ owl_class_...  │  │ (sync tracking)    │ │
│ │ graph_stats    │  │ owl_properties │  │                    │ │
│ └────────────────┘  │ owl_axioms     │  └────────────────────┘ │
│                     └────────────────┘                          │
│                                                                  │
│ Cross-referencing via: graph_nodes.owl_class_iri → owl_classes.iri│
└──────────────────────────────────────────────────────────────────┘
```

---

## Processing Flow Details

### 1. Initial Sync (First Run)
```
GitHub → EnhancedContentAPI → GitHubSyncService
    ├─> filter_changed_files() → [EMPTY] → Process ALL files
    ├─> process_batch() (50 files per batch)
    │   ├─> KnowledgeGraphParser → graph_nodes, graph_edges
    │   └─> OntologyParser → owl_classes, owl_class_hierarchy
    └─> update_file_metadata() → file_metadata (SHA1 tracking)
```

### 2. Incremental Sync (Subsequent Runs)
```
GitHub → EnhancedContentAPI → GitHubSyncService
    ├─> filter_changed_files() → Compare SHA1 hashes
    │   ├─> file_metadata.file_blob_sha vs GitHub SHA
    │   └─> ONLY process changed files (incremental)
    ├─> process_batch() (only changed files)
    └─> update_file_metadata() → Update SHA1 + change_count
```

### 3. Force Full Sync (Environment Override)
```bash
FORCE_FULL_SYNC=1 cargo run sync
# Bypasses SHA1 filtering, processes ALL files
```

---

## Key Relationships

### Graph ↔ Ontology Link
```
graph_nodes.owl_class_iri → owl_classes.iri
```
This enables:
- Semantic type annotation of graph nodes
- Ontology-driven graph queries
- Inference over graph structures

### Cascading Deletes
```
owl_classes.iri
    ← owl_class_hierarchy.class_iri (ON DELETE CASCADE)
    ← owl_class_hierarchy.parent_iri (ON DELETE CASCADE)

graph_nodes.id
    ← graph_edges.source_id (ON DELETE CASCADE)
    ← graph_edges.target_id (ON DELETE CASCADE)
```

---

## Performance Optimizations

### Batch Processing
- **BATCH_SIZE = 50 files**: Prevents memory overflow on large repos
- Transaction-based saves: All 50 files committed as single transaction

### Incremental Sync (SHA1 Filtering)
- Only reprocesses files with changed `file_blob_sha`
- Reduces API calls and database writes by ~90% on subsequent syncs

### Position Updates (CUDA Integration)
```rust
batch_update_positions(positions: Vec<(u32, f32, f32, f32)>)
```
- Chunked updates (10,000 positions per chunk)
- Optimized for CUDA kernel output

### Statistics Caching
- `graph_statistics` table caches expensive computations
- Updated after graph modifications
- Avoids full table scans on metrics queries

---

## Transaction Patterns

### Save Ontology (All-or-Nothing)
```rust
BEGIN TRANSACTION
DELETE FROM owl_class_hierarchy
DELETE FROM owl_axioms
DELETE FROM owl_properties
DELETE FROM owl_classes
INSERT owl_classes [...]
INSERT owl_class_hierarchy [...]
INSERT owl_properties [...]
INSERT owl_axioms [...]
COMMIT
```

### Save Graph (Incremental Upserts)
```rust
BEGIN TRANSACTION
-- Check file_metadata count to detect initial sync
IF metadata_count == 0:
    DELETE FROM graph_edges
    DELETE FROM graph_nodes
-- Upsert nodes (INSERT OR REPLACE)
-- Upsert edges (INSERT OR REPLACE)
COMMIT
```

---

## Metadata JSON Encoding

### graph_nodes.metadata
```json
{
  "type": "page" | "linked_page" | "tag",
  "owl_class_iri": "http://example.org/Person",
  "custom_field": "value"
}
```

### owl_properties.domain/range
```json
["http://example.org/Person", "http://example.org/Organization"]
```

### owl_axioms.annotations
```json
{
  "rdfs:label": "Custom label",
  "rdfs:comment": "Description",
  "source": "inference"
}
```

---

## Database Location & Access

**File Path**: `data/unified.db`

**Rust Access**:
```rust
use crate::repositories::{UnifiedGraphRepository, UnifiedOntologyRepository};

let graph_repo = UnifiedGraphRepository::new("data/unified.db")?;
let ontology_repo = UnifiedOntologyRepository::new("data/unified.db")?;
```

**CLI Access**:
```bash
sqlite3 data/unified.db

# Example queries
SELECT COUNT(*) FROM graph_nodes;
SELECT COUNT(*) FROM owl_classes;
SELECT * FROM file_metadata WHERE change_count > 0;
```

---

## Schema Evolution Notes

**Created**: Unified architecture (v1.0)
- Replaces separate `knowledge_graph.db` and `ontology.db`
- Maintains 100% API compatibility with legacy adapters
- CUDA kernel compatibility preserved via identical data structures

**Future Additions** (Not Yet Implemented):
- Inference cache tables
- Pathfinding cache tables
- Validation report tables
- SPARQL query cache

---

## Summary Statistics (Example)

```
Database: unified.db (45.2 MB)
├─ graph_nodes: 12,453 rows
├─ graph_edges: 34,821 rows
├─ owl_classes: 1,284 rows
├─ owl_class_hierarchy: 2,104 rows
├─ owl_properties: 342 rows
├─ owl_axioms: 789 rows
└─ file_metadata: 1,521 rows (sync tracking)

Last Sync: 2025-11-02 14:32:01 UTC
Changed Files: 34 / 1,521 (2.2%)
Sync Duration: 12.3 seconds
```
