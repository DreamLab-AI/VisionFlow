# Unified Database Architecture

**Status:** ✅ ACTIVE (as of November 2, 2025)
**Database:** `/data/unified.db` (single SQLite database with WAL mode)

---

## Overview

VisionFlow uses a **single unified database** (`unified.db`) that consolidates all domain tables previously spread across three separate databases. This unified architecture provides:

- **Atomic transactions** across all domains
- **Simplified backup/restore** (single file)
- **Reduced connection overhead** (one connection pool)
- **Foreign key integrity** across all tables
- **Easier development** and testing

---

## Database Schema

### Core Domain Tables

**Settings Domain**
- `physics_settings` - GPU physics simulation configuration
- `constraint_settings` - Ontology-based constraints
- `rendering_settings` - 3D visualization parameters
- `constraint_profiles` - Saved constraint configurations

**Knowledge Graph Domain**
- `graph_nodes` - Graph vertices with position, velocity, metadata
- `graph_edges` - Graph edges with source/target relationships
- `graph_clusters` - Leiden clustering results
- `graph_stats` - Performance metrics and statistics
- `file_metadata` - Markdown file tracking with SHA1 checksums
- `node_view` - Optimized read view for API queries

**Ontology Domain**
- `owl_classes` - OWL class definitions with IRIs
- `owl_properties` - Object and data properties
- `owl_axioms` - SubClassOf, DisjointClasses, etc.
- `owl_class_hierarchy` - Parent-child relationships
- `owl_individuals` - Class instances
- `namespaces` - IRI namespace prefixes
- `ontologies` - Ontology metadata
- `inference_results` - Cached reasoning results

**Constraint Domain**
- `active_constraints` - Physics constraints derived from ontology
- `pathfinding_cache` - GPU shortest-path results

---

## Migration from Legacy Architecture

### Previous Architecture (DEPRECATED)

VisionFlow previously used **three separate databases**:
- `settings.db` - Application configuration
- `knowledge_graph.db` - Graph data
- `ontology.db` - OWL ontologies

**Issues with Dual/Triple Database Design:**
- ❌ No cross-database foreign keys
- ❌ No atomic transactions across domains
- ❌ Multiple connection pools
- ❌ Complex backup/restore procedures
- ❌ Schema synchronization challenges

### Current Architecture (ACTIVE)

**Single Unified Database** (`unified.db`):
- ✅ All domain tables in one database
- ✅ Foreign key constraints work across domains
- ✅ Atomic transactions across all operations
- ✅ Single backup file
- ✅ Simplified connection management

---

## Implementation Details

### Repository Layer

**Rust Repositories:**
- `UnifiedGraphRepository` - Manages graph_nodes, graph_edges, file_metadata
- `UnifiedOntologyRepository` - Manages owl_classes, owl_properties, owl_axioms
- `UnifiedSettingsRepository` - Manages all settings tables

All repositories connect to `data/unified.db` via SQLite connection pool.

### Schema Initialization

Schema defined in: `migration/unified_schema.sql`

**Automatic Initialization:**
```rust
// On first startup, tables are created automatically
UnifiedGraphRepository::create_schema(&conn).await?;
UnifiedOntologyRepository::create_schema(&conn).await?;
// ... etc
```

**Manual Initialization:**
```bash
sqlite3 data/unified.db < migration/unified_schema.sql
```

---

## Data Pipeline

```
GitHub API (jjohare/logseq)
   ↓
GitHubSyncService
   ├─ Detects ### OntologyBlock sections in markdown
   ├─ Uses OntologyParser to extract OWL classes
   └─ Uses KnowledgeGraphParser to extract nodes/edges
      ↓
UnifiedGraphRepository → unified.db (graph_nodes, graph_edges, file_metadata)
UnifiedOntologyRepository → unified.db (owl_classes, owl_axioms)
      ↓
GraphServiceActor (GPU-ready)
   └─ 7 tier-1 CUDA kernels for physics simulation
      ↓
REST API /api/graph/data
   └─ Returns JSON with nodes, edges, metadata
      ↓
Client Visualization (localhost:4000)
   └─ 3D graph with ontology-driven physics
```

---

## Environment Variables

**Database Configuration:**
- `DATABASE_URL=sqlite:///path/to/unified.db` - Database connection string
- `FORCE_FULL_SYNC=1` - Bypass SHA1 filtering for full re-sync

**Legacy Variables (DEPRECATED):**
- ❌ `SETTINGS_DATABASE_URL` - No longer used
- ❌ `KNOWLEDGE_GRAPH_DATABASE_URL` - No longer used
- ❌ `ONTOLOGY_DATABASE_URL` - No longer used

---

## Archived Legacy Databases

**Location:** `data/archive/`

Legacy databases have been archived for historical reference:
- `data/archive/knowledge_graph.db` (superseded by unified.db)
- `data/archive/ontology.db` (superseded by unified.db)
- `data/archive/settings.db` (superseded by unified.db)

**DO NOT USE** these archived databases. They are retained only for:
- Data recovery in emergencies
- Historical reference
- Migration verification

---

## Schema Management

### View Current Schema

```bash
# Connect to database
sqlite3 data/unified.db

# List all tables
.tables

# View specific table schema
.schema owl_classes
.schema graph_nodes

# Count rows
SELECT COUNT(*) FROM owl_classes;
SELECT COUNT(*) FROM graph_nodes;
```

### Backup Database

```bash
# Simple file copy (server must be stopped)
cp data/unified.db data/unified.db.backup

# Online backup (server running)
sqlite3 data/unified.db ".backup data/unified.db.backup"
```

### Restore from Backup

```bash
cp data/unified.db.backup data/unified.db
```

---

## Performance Optimization

**WAL Mode Enabled:**
```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-64000;  -- 64MB cache
```

**Indexes Created:**
- All primary keys
- Foreign key columns
- Frequently queried columns (iri, file_sha1, metadata_id)

**Query Optimization:**
- Materialized view `node_view` for fast API queries
- Compound indexes on (x,y,z) for spatial queries
- B-tree indexes on text fields for fast lookups

---

## Troubleshooting

### Database Locked Error

**Symptom:** `database is locked` errors
**Cause:** Multiple writers or long-running transaction
**Solution:**
```bash
# Check for open connections
lsof data/unified.db

# Restart server to release locks
docker-compose restart
```

### Schema Migration Failed

**Symptom:** Column not found errors
**Cause:** Schema out of sync with code
**Solution:**
```bash
# Drop database and recreate (DATA LOSS WARNING)
rm data/unified.db
# Server will auto-create schema on next startup
```

---

## Future Enhancements

**Planned Improvements:**
- PostgreSQL adapter for enterprise deployments
- Read replicas for horizontal scaling
- Partitioning large tables (>1M rows)
- Query result caching with Redis
- Full-text search with FTS5

---

**Last Updated:** 2025-11-02
**Version:** 2.0.0 (Unified Database Architecture)
