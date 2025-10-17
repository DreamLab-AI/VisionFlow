# Data Storage Architecture

**Version**: 2.0.0
**Last Updated**: 2025-10-17
**Status**: Production-Ready with SQLite Migration

## Overview

VisionFlow's data storage architecture has evolved to support both file-based and database-backed persistence. The system now includes SQLite integration for ontology data alongside the existing file-based storage for graph data, providing a hybrid approach optimized for different data types and access patterns.

## Storage Systems

### 1. File-Based Storage (Graph Data)

Primary storage for knowledge graph data using versioned JSON files.

**Location**: `./data/graphs/`

**Structure**:
```
data/
└── graphs/
    ├── current.json          # Active graph state
    ├── backup_20251017_123456.json
    └── snapshots/
        └── snapshot_uuid.json
```

**Graph Data Schema**:
```json
{
  "version": "2.0",
  "timestamp": "2025-10-17T12:34:56Z",
  "nodes": [
    {
      "id": "node_uuid",
      "labels": ["Person"],
      "properties": {
        "name": "Alice",
        "email": "alice@example.com"
      },
      "position": {
        "x": 100.5,
        "y": 200.3,
        "z": 0.0
      }
    }
  ],
  "edges": [
    {
      "id": "edge_uuid",
      "source": "node1_uuid",
      "target": "node2_uuid",
      "relationship_type": "knows",
      "properties": {
        "since": "2020-01-01"
      }
    }
  ],
  "metadata": {
    "node_count": 150,
    "edge_count": 300,
    "last_modified": "2025-10-17T12:34:56Z"
  }
}
```

**Characteristics**:
- Atomic file operations with backup
- Git-friendly format for version control
- Human-readable JSON
- Fast bulk read/write operations
- ~5-10ms for typical graphs (<10K nodes)

**Use Cases**:
- Primary graph state persistence
- Snapshot/restore functionality
- Version control integration
- Backup and disaster recovery

### 2. SQLite Database (Ontology Data)

Structured storage for ontology axioms, validation reports, and inference data.

**Location**: `./data/ontology.db`

**Schema**:

```sql
-- Ontology metadata and content
CREATE TABLE ontologies (
    id TEXT PRIMARY KEY,
    content_hash TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    format TEXT NOT NULL,  -- 'functional' or 'owlxml'
    axiom_count INTEGER NOT NULL,
    loaded_at TIMESTAMP NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    INDEX idx_content_hash (content_hash),
    INDEX idx_loaded_at (loaded_at)
);

-- Validation reports
CREATE TABLE validation_reports (
    id TEXT PRIMARY KEY,
    ontology_id TEXT NOT NULL,
    graph_signature TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    duration_ms INTEGER NOT NULL,
    total_triples INTEGER NOT NULL,
    violations_count INTEGER NOT NULL,
    inferences_count INTEGER NOT NULL,
    report_data BLOB NOT NULL,  -- Serialized ValidationReport
    FOREIGN KEY (ontology_id) REFERENCES ontologies(id) ON DELETE CASCADE,
    INDEX idx_ontology_id (ontology_id),
    INDEX idx_graph_signature (graph_signature),
    INDEX idx_timestamp (timestamp)
);

-- Violations for efficient querying
CREATE TABLE violations (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    severity TEXT NOT NULL,  -- 'Error', 'Warning', 'Info'
    rule TEXT NOT NULL,
    message TEXT NOT NULL,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (report_id) REFERENCES validation_reports(id) ON DELETE CASCADE,
    INDEX idx_report_id (report_id),
    INDEX idx_severity (severity),
    INDEX idx_rule (rule)
);

-- Inferred triples
CREATE TABLE inferred_triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    is_literal BOOLEAN NOT NULL,
    datatype TEXT,
    language TEXT,
    FOREIGN KEY (report_id) REFERENCES validation_reports(id) ON DELETE CASCADE,
    INDEX idx_report_id (report_id),
    INDEX idx_subject (subject),
    INDEX idx_predicate (predicate)
);

-- Cache metadata
CREATE TABLE cache_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    INDEX idx_expires_at (expires_at)
);

-- System health metrics
CREATE TABLE health_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    loaded_ontologies INTEGER NOT NULL,
    cached_reports INTEGER NOT NULL,
    validation_queue_size INTEGER NOT NULL,
    cache_hit_rate REAL NOT NULL,
    avg_validation_time_ms REAL NOT NULL,
    active_jobs INTEGER NOT NULL,
    memory_usage_mb REAL NOT NULL,
    INDEX idx_timestamp (timestamp)
);
```

**Characteristics**:
- ACID transactions for data integrity
- Efficient querying with indexes
- ~1-5ms for typical queries
- Automatic cleanup of expired entries
- Support for concurrent reads

**Use Cases**:
- Ontology axiom storage
- Validation report persistence
- Violation and inference tracking
- Cache management
- System health monitoring

### 3. In-Memory Caches (Runtime)

High-performance caching using DashMap for concurrent access.

**Ontology Cache**:
```rust
Arc<DashMap<String, CachedOntology>>

CachedOntology {
    id: String,
    content_hash: String,
    ontology: SetOntology<Arc<str>>,  // Parsed horned-owl ontology
    axiom_count: usize,
    loaded_at: DateTime<Utc>,
    ttl_seconds: u64,
}
```

**Validation Cache**:
```rust
Arc<DashMap<String, ValidationReport>>
```

**Characteristics**:
- Lock-free concurrent access
- Configurable TTL (default: 3600s)
- Automatic eviction on expiry
- ~100ns access time
- Memory efficient with Arc sharing

**Use Cases**:
- Hot data access
- Reducing database queries
- Session data
- Temporary computation results

## Migration from File-Based to SQLite

### Migration Path

The ontology system has migrated from file-based storage to SQLite for improved query performance and data integrity.

**Before (File-Based)**:
```
ontology/
├── axioms/
│   ├── ontology_abc123.owl
│   └── ontology_def456.owl
├── reports/
│   ├── report_xyz789.json
│   └── report_uvw012.json
└── mapping.toml
```

**After (SQLite)**:
```
data/
├── ontology.db           # SQLite database
└── ontology/
    └── mapping.toml      # Configuration only
```

### Migration Benefits

| Aspect | File-Based | SQLite | Improvement |
|--------|------------|--------|-------------|
| Query Performance | O(n) scan | O(log n) indexed | 10-100x faster |
| Concurrent Access | File locks | Row-level locks | Better concurrency |
| Data Integrity | Manual sync | ACID transactions | Guaranteed consistency |
| Query Capabilities | None | SQL queries | Rich querying |
| Backup | File copy | SQLite backup | Atomic snapshots |

### Backward Compatibility

The system maintains backward compatibility with existing file-based ontologies:

1. **Auto-migration**: On first load, file-based ontologies are imported into SQLite
2. **Dual-mode**: Can read from both sources during transition
3. **Export**: SQLite data can be exported back to files for archival

**Migration Script** (Executed automatically on startup):
```rust
async fn migrate_file_ontologies_to_sqlite() -> Result<()> {
    let db = SqliteConnection::open("./data/ontology.db")?;
    let ontology_dir = Path::new("./ontology/axioms");

    if ontology_dir.exists() {
        for entry in fs::read_dir(ontology_dir)? {
            let path = entry?.path();
            if path.extension() == Some(OsStr::new("owl")) {
                let content = fs::read_to_string(&path)?;
                let ontology_id = import_ontology_to_db(&db, content).await?;
                info!("Migrated {} to {}", path.display(), ontology_id);
            }
        }
    }

    Ok(())
}
```

## Data Flow Architecture

### Write Path

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /api/graph/data
       ▼
┌──────────────────┐
│ GraphServiceActor│
└──────┬───────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌─────────────┐      ┌──────────────┐
│ Memory      │      │ File System  │
│ (In-memory) │      │ (JSON)       │
└─────────────┘      └──────────────┘
       │                     │
       │ Ontology validation │
       ▼                     │
┌──────────────────┐         │
│ OntologyActor    │         │
└──────┬───────────┘         │
       │                     │
       ├─────────────────────┘
       │
       ▼
┌──────────────────┐
│ SQLite           │
│ (ontology.db)    │
└──────────────────┘
```

### Read Path

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ GET /api/graph/data
       ▼
┌──────────────────┐
│ GraphServiceActor│
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│ Memory Cache│───── Cache Hit ────┐
│ (DashMap)   │                    │
└──────┬──────┘                    │
       │ Cache Miss                │
       ▼                           │
┌──────────────┐                   │
│ File System  │                   │
│ (JSON)       │                   │
└──────┬───────┘                   │
       │                           │
       └────────────┬──────────────┘
                    │
                    ▼
              ┌──────────┐
              │  Client  │
              └──────────┘
```

### Validation Path

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /api/ontology/validate
       ▼
┌──────────────────┐
│ OntologyActor    │
└──────┬───────────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌─────────────┐  ┌─────────────┐
│ Memory Cache│  │   SQLite    │
│ (Check)     │  │ (Load)      │
└──────┬──────┘  └──────┬──────┘
       │                │
       │ Miss           │ Hit
       │                │
       └────────┬───────┘
                ▼
       ┌──────────────────┐
       │ OwlValidatorSvc  │
       │ - Parse graph    │
       │ - Validate       │
       │ - Infer          │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ ValidationReport │
       └────────┬─────────┘
                │
                ├──────────────┐
                │              │
                ▼              ▼
       ┌─────────────┐  ┌─────────────┐
       │ Memory Cache│  │   SQLite    │
       │ (Store)     │  │ (Persist)   │
       └─────────────┘  └─────────────┘
```

## Performance Characteristics

### Storage Performance

| Operation | Storage Type | Avg Time | P95 Time | P99 Time |
|-----------|-------------|----------|----------|----------|
| Graph Load | File (JSON) | 15ms | 35ms | 65ms |
| Graph Save | File (JSON) | 25ms | 50ms | 85ms |
| Ontology Load | SQLite | 5ms | 12ms | 20ms |
| Ontology Save | SQLite | 8ms | 18ms | 30ms |
| Report Query | SQLite | 2ms | 5ms | 10ms |
| Cache Hit | DashMap | 0.1ms | 0.2ms | 0.5ms |

### Storage Capacity

| Data Type | Typical Size | Max Recommended | Notes |
|-----------|-------------|-----------------|-------|
| Graph JSON | 1-10 MB | 100 MB | Use compression for large graphs |
| Ontology | 10-500 KB | 10 MB | Larger ontologies need optimization |
| Validation Report | 5-50 KB | 1 MB | Depends on violation count |
| SQLite DB | 10-100 MB | 10 GB | Includes all historical data |
| Memory Cache | 50-200 MB | 2 GB | Per process |

### Concurrent Access

| Operation | File-Based | SQLite | DashMap |
|-----------|------------|--------|---------|
| Concurrent Reads | Limited | Excellent | Excellent |
| Concurrent Writes | Serial | Good | Excellent |
| Read-Write Mix | Poor | Good | Excellent |

## Backup and Recovery

### Backup Strategy

**Automated Backups**:
```
backup/
├── graphs/
│   ├── daily/
│   │   ├── 2025-10-17_current.json
│   │   └── 2025-10-16_current.json
│   └── weekly/
│       └── 2025-week-42_current.json
└── ontology/
    ├── daily/
    │   ├── 2025-10-17_ontology.db
    │   └── 2025-10-16_ontology.db
    └── weekly/
        └── 2025-week-42_ontology.db
```

**Backup Schedule**:
- **Graph data**: Every 6 hours + before major operations
- **SQLite database**: Daily + before schema migrations
- **Retention**: 7 daily + 4 weekly + 12 monthly

**Backup Command**:
```bash
# Graph backup
cp data/graphs/current.json backup/graphs/daily/$(date +%Y-%m-%d)_current.json

# SQLite backup (online backup)
sqlite3 data/ontology.db ".backup backup/ontology/daily/$(date +%Y-%m-%d)_ontology.db"
```

### Recovery Procedures

**Graph Recovery**:
```rust
// Restore from backup
async fn restore_graph(backup_path: &Path) -> Result<()> {
    let backup_data = fs::read_to_string(backup_path)?;
    let graph: GraphData = serde_json::from_str(&backup_data)?;

    // Validate structure
    validate_graph_integrity(&graph)?;

    // Write to current
    let current_path = Path::new("./data/graphs/current.json");
    fs::write(current_path, backup_data)?;

    Ok(())
}
```

**SQLite Recovery**:
```bash
# Restore from backup
cp backup/ontology/daily/2025-10-17_ontology.db data/ontology.db

# Verify integrity
sqlite3 data/ontology.db "PRAGMA integrity_check;"

# Restart service
systemctl restart visionflow
```

## Maintenance

### Cleanup Operations

**SQLite Vacuum** (Monthly):
```sql
-- Reclaim space and optimize
VACUUM;

-- Update statistics
ANALYZE;
```

**Cache Cleanup** (Hourly):
```rust
async fn cleanup_expired_caches() {
    // Remove expired ontologies
    ontology_cache.retain(|_, v| {
        let age = Utc::now().signed_duration_since(v.loaded_at);
        age.num_seconds() < v.ttl_seconds as i64
    });

    // Remove expired validation reports
    validation_cache.retain(|_, v| {
        let age = Utc::now().signed_duration_since(v.timestamp);
        age.num_seconds() < 3600
    });
}
```

**File Cleanup** (Weekly):
```bash
#!/bin/bash
# Remove old snapshots
find data/graphs/snapshots -type f -mtime +30 -delete

# Remove old backups
find backup/graphs/daily -type f -mtime +7 -delete
find backup/ontology/daily -type f -mtime +7 -delete
```

### Database Optimization

**Index Analysis**:
```sql
-- Check index usage
SELECT name, tbl_name, sql
FROM sqlite_master
WHERE type = 'index';

-- Query plan analysis
EXPLAIN QUERY PLAN
SELECT * FROM validation_reports
WHERE ontology_id = 'ontology_abc123'
ORDER BY timestamp DESC
LIMIT 10;
```

**Performance Tuning**:
```sql
-- Optimize for write performance
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB cache

-- Optimize for read performance
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;  -- 256MB memory map
```

## Monitoring

### Storage Metrics

**File System Metrics**:
```rust
async fn get_storage_metrics() -> StorageMetrics {
    let graph_size = fs::metadata("./data/graphs/current.json")?.len();
    let db_size = fs::metadata("./data/ontology.db")?.len();

    StorageMetrics {
        graph_size_mb: graph_size as f64 / 1_048_576.0,
        db_size_mb: db_size as f64 / 1_048_576.0,
        cache_size_mb: estimate_cache_size(),
        total_size_mb: (graph_size + db_size) as f64 / 1_048_576.0,
    }
}
```

**Database Metrics**:
```sql
-- Table sizes
SELECT
    name,
    SUM(pgsize) / 1024.0 / 1024.0 AS size_mb
FROM dbstat
GROUP BY name
ORDER BY size_mb DESC;

-- Cache statistics
SELECT
    cache_size,
    page_count,
    page_size,
    (cache_size * page_size) / 1024.0 / 1024.0 AS cache_mb
FROM pragma_cache_size, pragma_page_count, pragma_page_size;
```

**Alerting Thresholds**:
- Graph file size > 50 MB: Warning
- SQLite database > 1 GB: Warning
- Cache memory > 1 GB: Warning
- Disk usage > 80%: Critical

## Migration Guide

### For Existing Deployments

**Step 1: Backup Current Data**
```bash
# Backup graph data
cp -r data/graphs backup/pre-migration/

# Backup ontology files if they exist
cp -r ontology backup/pre-migration/
```

**Step 2: Update Application**
```bash
git pull origin main
cargo build --release --features ontology
```

**Step 3: Run Migration**
```bash
# Automatic migration on first start
./target/release/webxr

# Monitor logs for migration progress
tail -f logs/visionflow.log | grep -i migration
```

**Step 4: Verify Migration**
```bash
# Check SQLite database
sqlite3 data/ontology.db "SELECT COUNT(*) FROM ontologies;"

# Check graph data
curl http://localhost:8080/api/graph/data | jq '.nodes | length'

# Check ontology health
curl http://localhost:8080/api/ontology/health | jq
```

**Step 5: Cleanup (Optional)**
```bash
# Archive old file-based ontologies
tar -czf backup/ontology-files.tar.gz ontology/axioms/
rm -rf ontology/axioms/
```

### Rollback Procedure

If issues arise:

```bash
# Stop service
systemctl stop visionflow

# Restore from backup
cp -r backup/pre-migration/graphs data/
cp -r backup/pre-migration/ontology .

# Remove SQLite database
rm data/ontology.db

# Restart with previous version
git checkout <previous-commit>
cargo build --release
./target/release/webxr
```

## Future Enhancements

### Planned Improvements

1. **Distributed Storage** (Q1 2026)
   - PostgreSQL cluster for graph data
   - Distributed caching with Redis
   - Multi-region replication

2. **Time-Series Support** (Q2 2026)
   - Historical graph state tracking
   - Temporal queries
   - Change stream API

3. **Compression** (Q3 2026)
   - LZ4 compression for graph JSON
   - SQLite page compression
   - Streaming decompression

4. **Sharding** (Q4 2026)
   - Graph partitioning by domain
   - Distributed ontology validation
   - Cross-shard queries

## References

- [System Architecture Overview](./system-overview.md)
- [Ontology System](../features/ontology-system.md)
- [API Reference](../api/ontology-endpoints.md)
- [Deployment Guide](../deployment/production.md)
