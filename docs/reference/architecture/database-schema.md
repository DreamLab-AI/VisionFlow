# Database Schema Documentation

**Version:** 3.0.0
**Last Updated:** 2025-10-22
**Database System:** SQLite 3.35+ with WAL mode

---

## Three-Database Architecture

VisionFlow uses **three separate SQLite databases** for clear domain separation, independent scaling, and conflict prevention:

### 1. Settings Database (`data/settings.db`)

**Purpose:** Application configuration, user preferences, and physics parameters

**Size:** Small (~1-5 MB)
**Access Pattern:** High read/write frequency, low data volume
**Backup Priority:** High (user preferences)

### 2. Knowledge Graph Database (`data/knowledge_graph.db`)

**Purpose:** Main graph structure from local markdown files (Logseq integration)

**Size:** Large (~50-500 MB depending on graph size)
**Access Pattern:** Moderate read/write, large data volume
**Backup Priority:** Critical (user data)

### 3. Ontology Database (`data/ontology.db`)

**Purpose:** Semantic ontology graph from GitHub markdown

**Size:** Medium (~10-100 MB)
**Access Pattern:** Low write frequency, moderate read
**Backup Priority:** Medium (can be regenerated from GitHub)

---

## Settings Database Schema

### Purpose

Stores application configuration, user preferences, and physics simulation parameters.

### Tables

#### settings

Flexible key-value store for application configuration:

```sql
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_string TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER,
    value_json TEXT,
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_settings_updated ON settings(updated_at DESC);
```

**Columns:**
- `key` - Unique configuration key (e.g., "application.theme")
- `value_*` - Typed values (only one populated per row)
- `description` - Human-readable setting documentation
- `created_at` / `updated_at` - Timestamps

**Example Rows:**
```
application.theme         | value_string | "dark"
application.language      | value_string | "en"
developer.debug_mode      | value_boolean | 0
developer.log_level       | value_string | "info"
visualisation.graphs      | value_json   | {"logseq": {...}, "visionflow": {...}}
```

#### physics_settings

Per-graph physics simulation configuration:

```sql
CREATE TABLE IF NOT EXISTS physics_settings (
    profile_name TEXT PRIMARY KEY
        CHECK(profile_name IN ('logseq', 'visionflow', 'ontology', 'default')),
    time_step REAL DEFAULT 0.016,
    damping REAL DEFAULT 0.85,
    repulsion_strength REAL DEFAULT 500.0,
    attraction_strength REAL DEFAULT 0.01,
    max_velocity REAL DEFAULT 100.0,
    convergence_threshold REAL DEFAULT 0.001,
    settings_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_physics_updated ON physics_settings(updated_at DESC);
```

**Columns:**
- `profile_name` - Graph type (logseq, visionflow, ontology, or default)
- `time_step` - Physics simulation timestep (0.016 = ~60 FPS)
- `damping` - Velocity damping coefficient (0-1)
- `repulsion_strength` - Coulomb-like repulsion between nodes
- `attraction_strength` - Spring-like attraction along edges
- `max_velocity` - Maximum node velocity cap
- `convergence_threshold` - Delta below which simulation is "settled"

#### namespaces

OWL namespace prefix mappings:

```sql
CREATE TABLE IF NOT EXISTS namespaces (
    prefix TEXT PRIMARY KEY,
    namespace_uri TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**Example Rows:**
```
owl      | http://www.w3.org/2002/07/owl#
rdfs     | http://www.w3.org/2000/01/rdf-schema#
xsd      | http://www.w3.org/2001/XMLSchema#
custom   | http://example.org/ontology#
```

#### schema_version

Tracks database schema migrations:

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## Knowledge Graph Database Schema

### Purpose

Stores the main graph structure: nodes, edges, file metadata, and statistics.

### Key Tables

#### nodes

Graph node definitions with physics state:

```sql
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY,
    metadata_id TEXT,
    label TEXT NOT NULL,
    type TEXT,
    position_x REAL DEFAULT 0.0,
    position_y REAL DEFAULT 0.0,
    position_z REAL DEFAULT 0.0,
    velocity_x REAL DEFAULT 0.0,
    velocity_y REAL DEFAULT 0.0,
    velocity_z REAL DEFAULT 0.0,
    color_rgba INTEGER DEFAULT 4294967295,
    is_pinned INTEGER DEFAULT 0,
    is_visible INTEGER DEFAULT 1,
    properties TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nodes_metadata ON nodes(metadata_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
```

**Columns:**
- `id` - Primary key (auto-incrementing)
- `metadata_id` - Reference to source file
- `label` - Node display name
- `type` - Node classification
- `position_*` - 3D coordinates for visualization
- `velocity_*` - Current movement vector
- `color_rgba` - RGBA color (32-bit packed)
- `is_pinned` - User-locked position
- `is_visible` - Culling flag
- `properties` - JSON object for extensions

#### edges

Graph relationships between nodes:

```sql
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT,
    weight REAL DEFAULT 1.0,
    properties TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
```

**Columns:**
- `id` - Unique edge identifier
- `source_id` - Source node (referential integrity enforced)
- `target_id` - Target node (referential integrity enforced)
- `edge_type` - Relationship type
- `weight` - Edge weight for algorithms
- `properties` - JSON for extensions

**FOREIGN KEY Constraints:**
- Prevent orphaned edges
- CASCADE delete when nodes removed

#### file_metadata

Source file information for provenance tracking:

```sql
CREATE TABLE IF NOT EXISTS file_metadata (
    metadata_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    last_modified TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**Columns:**
- `metadata_id` - Unique file identifier
- `file_path` - Path to markdown file
- `file_hash` - SHA256 hash for change detection
- `last_modified` - Last modification timestamp

#### file_topics

Topic associations for files:

```sql
CREATE TABLE IF NOT EXISTS file_topics (
    metadata_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    PRIMARY KEY (metadata_id, topic),
    FOREIGN KEY (metadata_id) REFERENCES file_metadata(metadata_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_file_topics_topic ON file_topics(topic);
```

#### graph_statistics

Cached graph statistics (updated periodically):

```sql
CREATE TABLE IF NOT EXISTS graph_statistics (
    key TEXT PRIMARY KEY,
    value_integer INTEGER,
    value_float REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**Example Keys:**
```
node_count            | value_integer | 1523
edge_count            | value_integer | 4821
average_degree        | value_float   | 6.3
connected_components  | value_integer | 3
```

---

## Ontology Database Schema

### Purpose

Stores OWL ontology definitions, axioms, and reasoning results.

### Key Tables

#### ontologies

Ontology metadata:

```sql
CREATE TABLE IF NOT EXISTS ontologies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_iri TEXT UNIQUE NOT NULL,
    version TEXT,
    source_url TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

#### owl_classes

OWL class definitions:

```sql
CREATE TABLE IF NOT EXISTS owl_classes (
    iri TEXT PRIMARY KEY,
    label TEXT,
    description TEXT,
    parent_classes TEXT,
    properties TEXT,
    source_file TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);
```

**Columns:**
- `iri` - Unique IRI identifier
- `label` - Human-readable name
- `description` - Class documentation
- `parent_classes` - JSON array of parent IRIs
- `properties` - JSON object of class properties

#### owl_properties

OWL property definitions:

```sql
CREATE TABLE IF NOT EXISTS owl_properties (
    iri TEXT PRIMARY KEY,
    label TEXT,
    property_type TEXT CHECK(property_type IN ('ObjectProperty', 'DataProperty', 'AnnotationProperty')),
    domain TEXT,
    range TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_owl_properties_type ON owl_properties(property_type);
```

**Columns:**
- `iri` - Unique property IRI
- `property_type` - ObjectProperty, DataProperty, or AnnotationProperty
- `domain` - JSON array of applicable classes
- `range` - JSON array of value types

#### owl_axioms

Ontological axioms (logical statements):

```sql
CREATE TABLE IF NOT EXISTS owl_axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    axiom_type TEXT CHECK(axiom_type IN ('SubClassOf', 'EquivalentClass', 'DisjointWith', 'ObjectPropertyAssertion', 'DataPropertyAssertion')),
    subject TEXT NOT NULL,
    object TEXT NOT NULL,
    annotations TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_owl_axioms_subject ON owl_axioms(subject);
CREATE INDEX IF NOT EXISTS idx_owl_axioms_type ON owl_axioms(axiom_type);
```

**Examples:**
```
SubClassOf        | Student | Person
EquivalentClass   | Author  | Agent
DisjointWith      | Man     | Woman
```

#### inference_results

Results from ontology reasoning:

```sql
CREATE TABLE IF NOT EXISTS inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    inferred_axioms TEXT NOT NULL,
    inference_time_ms INTEGER,
    reasoner_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_inference_timestamp ON inference_results(timestamp DESC);
```

#### validation_reports

Ontology validation results:

```sql
CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    is_valid INTEGER,
    errors TEXT,
    warnings TEXT
);
```

---

## Database Operations

### Connection Management

All databases use **connection pooling** with Arc<Mutex<Connection>>:

```rust
pub struct DatabaseService {
    connection: Arc<Mutex<Connection>>,
    db_path: String,
}

impl DatabaseService {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        // Configure for optimal performance
        conn.execute_batch("
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = 10000;
            PRAGMA foreign_keys = ON;
            PRAGMA temp_store = MEMORY;
        ")?;

        Ok(Self {
            connection: Arc::new(Mutex::new(conn)),
            db_path: db_path.to_string(),
        })
    }
}
```

### Performance Optimizations

#### WAL Mode (Write-Ahead Logging)
- Allows concurrent reads while writing
- Better crash recovery
- Automatic checkpointing

#### Prepared Statements
```rust
let mut stmt = conn.prepare_cached(
    "SELECT * FROM nodes WHERE id = ?"
)?;
```

#### Batch Transactions
```rust
let tx = conn.transaction()?;
for node in nodes {
    tx.execute("INSERT INTO nodes ...", params![...])?;
}
tx.commit()?;
```

---

## Backup & Restore

### Automated Backups

```bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="data/backups/$DATE"

mkdir -p "$BACKUP_DIR"

sqlite3 data/settings.db ".backup $BACKUP_DIR/settings.db"
sqlite3 data/knowledge_graph.db ".backup $BACKUP_DIR/knowledge_graph.db"
sqlite3 data/ontology.db ".backup $BACKUP_DIR/ontology.db"

echo "Backup complete: $BACKUP_DIR"
```

### Restore from Backup

```bash
BACKUP_DIR=$1

# Stop application first!
systemctl stop visionflow

# Restore databases
cp "$BACKUP_DIR/settings.db" data/settings.db
cp "$BACKUP_DIR/knowledge_graph.db" data/knowledge_graph.db
cp "$BACKUP_DIR/ontology.db" data/ontology.db

# Restart application
systemctl start visionflow

echo "Restore complete from: $BACKUP_DIR"
```

---

## Performance Tuning

### Query Analysis

```sql
EXPLAIN QUERY PLAN
SELECT n.* FROM nodes n
INNER JOIN edges e ON n.id = e.source_id
WHERE n.type = 'concept';
```

### Database Optimization

```sql
-- Rebuild index statistics for query planner
ANALYZE;

-- Reclaim unused space
VACUUM;

-- Check database integrity
PRAGMA integrity_check;
```

### Monitor Database Size

```bash
du -h data/*.db

sqlite3 data/knowledge_graph.db "PRAGMA page_count; PRAGMA page_size;"
```

---

## Troubleshooting

### Common Issues

**Issue:** "Database is locked"
- **Cause:** Multiple writers without WAL mode
- **Solution:** Ensure WAL mode is enabled (automatic)

**Issue:** "Disk I/O error"
- **Cause:** Insufficient disk space or permissions
- **Solution:** Check disk space and file permissions

**Issue:** "Corrupt database"
- **Cause:** Unexpected shutdown or disk failure
- **Solution:** Restore from backup or run integrity check

```bash
sqlite3 data/settings.db "PRAGMA integrity_check;"
```

---

## Related Documentation

- **[Architecture Overview](./architecture.md)** - High-level design
- **[Hexagonal & CQRS](./hexagonal-cqrs.md)** - Core patterns
- **[API Reference](../api/)** - REST and WebSocket endpoints

---

**Document Maintained By:** VisionFlow Database Team
**Last Review:** 2025-10-22
**Next Review:** 2025-11-22
