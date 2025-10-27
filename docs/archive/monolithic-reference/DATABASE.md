# VisionFlow Database Documentation

**Version:** 3.0.0
**Last Updated:** 2025-10-22
**Database System:** SQLite 3.35+ with WAL mode

---

## Three-Database Architecture

VisionFlow uses **three separate SQLite databases** for clear domain separation:

### 1. Settings Database (`data/settings.db`)

**Purpose:** Application configuration, user preferences, and physics parameters

**Size:** Small (~1-5 MB)
**Access Pattern:** High read/write frequency, low data volume
**Backup Priority:** High (user preferences)

**Key Tables:**
- `settings` - Key-value configuration store
- `physics_settings` - Per-graph physics profiles
- `namespaces` - Namespace mappings
- `class_mappings` - OWL class mappings
- `property_mappings` - OWL property mappings

### 2. Knowledge Graph Database (`data/knowledge_graph.db`)

**Purpose:** Main graph structure from local markdown files (Logseq)

**Size:** Large (~50-500 MB depending on graph size)
**Access Pattern:** Moderate read/write, large data volume
**Backup Priority:** Critical (user data)

**Key Tables:**
- `nodes` - Graph nodes with positions
- `edges` - Graph edges and relationships
- `file_metadata` - Source file information
- `file_topics` - Topic associations
- `graph_statistics` - Cached statistics

### 3. Ontology Database (`data/ontology.db`)

**Purpose:** Semantic ontology graph from GitHub markdown

**Size:** Medium (~10-100 MB)
**Access Pattern:** Low write frequency, moderate read
**Backup Priority:** Medium (can be regenerated from GitHub)

**Key Tables:**
- `ontologies` - Ontology metadata
- `owl_classes` - OWL class definitions
- `owl_properties` - OWL property definitions
- `owl_axioms` - Ontological axioms
- `inference_results` - Reasoner output

---

## Database Schemas

### Settings Database Schema

```sql
-- Settings table: flexible key-value store
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

-- Physics settings per graph profile
CREATE TABLE IF NOT EXISTS physics_settings (
    profile_name TEXT PRIMARY KEY CHECK(profile_name IN ('logseq', 'visionflow', 'ontology', 'default')),
    time_step REAL DEFAULT 0.016,
    damping REAL DEFAULT 0.85,
    repulsion_strength REAL DEFAULT 500.0,
    attraction_strength REAL DEFAULT 0.01,
    max_velocity REAL DEFAULT 100.0,
    convergence_threshold REAL DEFAULT 0.001,
    settings_json TEXT, -- Full settings as JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Namespace mappings
CREATE TABLE IF NOT EXISTS namespaces (
    prefix TEXT PRIMARY KEY,
    namespace_uri TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Schema versioning
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_settings_updated ON settings(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_physics_updated ON physics_settings(updated_at DESC);
```

### Knowledge Graph Database Schema

```sql
-- Nodes table
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
    color_rgba INTEGER DEFAULT 4294967295, -- White
    is_pinned INTEGER DEFAULT 0,
    is_visible INTEGER DEFAULT 1,
    properties TEXT, -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Edges table
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT,
    weight REAL DEFAULT 1.0,
    properties TEXT, -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

-- File metadata
CREATE TABLE IF NOT EXISTS file_metadata (
    metadata_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    last_modified TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- File topics
CREATE TABLE IF NOT EXISTS file_topics (
    metadata_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    PRIMARY KEY (metadata_id, topic),
    FOREIGN KEY (metadata_id) REFERENCES file_metadata(metadata_id) ON DELETE CASCADE
);

-- Graph statistics (cached)
CREATE TABLE IF NOT EXISTS graph_statistics (
    key TEXT PRIMARY KEY,
    value_integer INTEGER,
    value_float REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_metadata ON nodes(metadata_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_file_topics_topic ON file_topics(topic);
```

### Ontology Database Schema

```sql
-- Ontologies table
CREATE TABLE IF NOT EXISTS ontologies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_iri TEXT UNIQUE NOT NULL,
    version TEXT,
    source_url TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- OWL classes
CREATE TABLE IF NOT EXISTS owl_classes (
    iri TEXT PRIMARY KEY,
    label TEXT,
    description TEXT,
    parent_classes TEXT, -- JSON array
    properties TEXT, -- JSON object
    source_file TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- OWL properties
CREATE TABLE IF NOT EXISTS owl_properties (
    iri TEXT PRIMARY KEY,
    label TEXT,
    property_type TEXT CHECK(property_type IN ('ObjectProperty', 'DataProperty', 'AnnotationProperty')),
    domain TEXT, -- JSON array
    range TEXT, -- JSON array
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- OWL axioms
CREATE TABLE IF NOT EXISTS owl_axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    axiom_type TEXT CHECK(axiom_type IN ('SubClassOf', 'EquivalentClass', 'DisjointWith', 'ObjectPropertyAssertion', 'DataPropertyAssertion')),
    subject TEXT NOT NULL,
    object TEXT NOT NULL,
    annotations TEXT, -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Inference results
CREATE TABLE IF NOT EXISTS inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    inferred_axioms TEXT NOT NULL, -- JSON array
    inference_time_ms INTEGER,
    reasoner_version TEXT
);

-- Validation reports
CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    is_valid INTEGER,
    errors TEXT, -- JSON array
    warnings TEXT -- JSON array
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);
CREATE INDEX IF NOT EXISTS idx_owl_properties_type ON owl_properties(property_type);
CREATE INDEX IF NOT EXISTS idx_owl_axioms_subject ON owl_axioms(subject);
CREATE INDEX IF NOT EXISTS idx_owl_axioms_type ON owl_axioms(axiom_type);
CREATE INDEX IF NOT EXISTS idx_inference_timestamp ON inference_results(timestamp DESC);
```

---

## Database Initialization

### Automatic Initialization

Databases are automatically initialized on application startup:

```rust
// src/app_state.rs
impl AppState {
    pub fn new() -> Self {
        // Initialize settings database
        let settings_db = Arc::new(
            DatabaseService::new("data/settings.db")
                .expect("Failed to initialize settings database")
        );
        settings_db.initialize_schema()
            .expect("Failed to initialize settings schema");

        // Initialize knowledge graph database
        let graph_db = Arc::new(
            DatabaseService::new("data/knowledge_graph.db")
                .expect("Failed to initialize knowledge graph database")
        );
        graph_db.initialize_schema()
            .expect("Failed to initialize graph schema");

        // Initialize ontology database
        let ontology_db = Arc::new(
            DatabaseService::new("data/ontology.db")
                .expect("Failed to initialize ontology database")
        );
        ontology_db.initialize_schema()
            .expect("Failed to initialize ontology schema");

        // ... rest of initialization
    }
}
```

### Manual Initialization

```bash
# Initialize all databases
cargo run --bin init-databases

# Initialize specific database
cargo run --bin init-databases -- --database settings
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
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open: {}", e))?;

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
// Automatically cached by rusqlite
let mut stmt = conn.prepare_cached(
    "SELECT * FROM nodes WHERE id = ?"
)?;
```

#### Transactions for Batch Operations
```rust
let tx = conn.transaction()?;
for node in nodes {
    tx.execute("INSERT INTO nodes ...", params![...])?;
}
tx.commit()?;
```

---

## Migration Procedures

### Schema Migrations

```bash
# Run all migrations
cargo run --bin migrate --from-version 1 --to-version 2

# Dry-run (preview changes)
cargo run --bin migrate --from-version 1 --to-version 2 --dry-run
```

### Data Migrations

```bash
# Migrate from YAML to database
cargo run --bin migrate-data --source yaml --target database

# Migrate specific domain
cargo run --bin migrate-data --source yaml --target database --domain settings
```

---

## Backup and Restore

### Automated Backups

```bash
#!/bin/bash
# backup-databases.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="data/backups/$DATE"

mkdir -p "$BACKUP_DIR"

# Backup all databases
sqlite3 data/settings.db ".backup $BACKUP_DIR/settings.db"
sqlite3 data/knowledge_graph.db ".backup $BACKUP_DIR/knowledge_graph.db"
sqlite3 data/ontology.db ".backup $BACKUP_DIR/ontology.db"

echo "Backup complete: $BACKUP_DIR"
```

### Restore from Backup

```bash
#!/bin/bash
# restore-databases.sh BACKUP_DIR

BACKUP_DIR=$1

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: restore-databases.sh BACKUP_DIR"
    exit 1
fi

# Stop application first!
# systemctl stop visionflow

# Restore databases
cp "$BACKUP_DIR/settings.db" data/settings.db
cp "$BACKUP_DIR/knowledge_graph.db" data/knowledge_graph.db
cp "$BACKUP_DIR/ontology.db" data/ontology.db

# Restart application
# systemctl start visionflow

echo "Restore complete from: $BACKUP_DIR"
```

---

## Performance Tuning

### Analyze Query Performance

```sql
-- Enable query logging
PRAGMA query_only = OFF;

-- Analyze query plan
EXPLAIN QUERY PLAN
SELECT n.* FROM nodes n
INNER JOIN edges e ON n.id = e.source_id
WHERE n.type = 'concept';

-- Analyze statistics
ANALYZE;

-- View index usage
SELECT * FROM sqlite_stat1;
```

### Vacuum Database

```bash
# Reclaim unused space and optimize
sqlite3 data/knowledge_graph.db "VACUUM;"

# Analyze statistics for query optimizer
sqlite3 data/knowledge_graph.db "ANALYZE;"
```

### Monitor Database Size

```bash
# Check database sizes
du -h data/*.db

# Check page count and size
sqlite3 data/knowledge_graph.db "PRAGMA page_count; PRAGMA page_size;"

# Check cache hit rate
sqlite3 data/knowledge_graph.db "PRAGMA cache_spill; PRAGMA cache_size;"
```

---

## Troubleshooting

### Common Issues

#### Issue: "Database is locked"
**Cause:** Multiple writers without WAL mode
**Solution:** Ensure WAL mode is enabled (done automatically)

#### Issue: "Disk I/O error"
**Cause:** Insufficient disk space or permissions
**Solution:** Check disk space and file permissions

#### Issue: "Corrupt database"
**Cause:** Unexpected shutdown or disk failure
**Solution:** Restore from backup or run integrity check

```bash
sqlite3 data/settings.db "PRAGMA integrity_check;"
```

---

## Additional Resources

- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [WAL Mode Guide](https://www.sqlite.org/wal.html)
- [Performance Tuning](https://www.sqlite.org/optoverview.html)

---

**Document Maintained By:** VisionFlow Database Team
**Last Review:** 2025-10-22
**Next Review:** 2025-11-22
