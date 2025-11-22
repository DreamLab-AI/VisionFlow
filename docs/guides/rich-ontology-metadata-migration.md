# Rich Ontology Metadata Migration Guide

**Date**: 2025-11-22
**Schema Version**: 2
**Migration**: `002_rich_ontology_metadata.sql`

## Executive Summary

This migration extends VisionFlow's ontology storage schema to support rich metadata discovered in the ontology parsing system. The enhanced schema enables:

- **17+ metadata fields** for comprehensive ontology management
- **Quality metrics** (quality_score, authority_score) for ranking and filtering
- **Domain classification** (source_domain, bridges_to_domain) for cross-domain analysis
- **Semantic relationships** (has-part, uses, enables, requires) for graph traversal
- **OWL2 properties** (owl:physicality, owl:role) for advanced reasoning
- **Backward compatibility** with existing data

## Architecture Overview

### Current Schema (Version 1)

```
owl_classes:
  - ontology_id, class_iri (PK)
  - label, comment, parent_class_iri
  - is_deprecated

owl_properties:
  - ontology_id, property_iri (PK)
  - property_type, label, comment
  - domain_class_iri, range_class_iri
  - is_functional, is_transitive, etc.
```

### Enhanced Schema (Version 2)

```
owl_classes:
  # Core identification
  - term_id, preferred_term

  # Classification
  - source_domain, version, type

  # Quality metrics
  - status, maturity
  - quality_score (0.0-1.0)
  - authority_score (0.0-1.0)
  - public_access, content_status

  # OWL2 properties
  - owl_physicality (physical/virtual/abstract)
  - owl_role (agent/patient/instrument)

  # Domain relationships
  - belongs_to_domain
  - bridges_to_domain

  # Source tracking
  - source_file, file_sha1
  - markdown_content, last_synced

  # Extensibility
  - additional_metadata (JSON)

owl_relationships:
  - source_class_iri, relationship_type, target_class_iri
  - confidence, is_inferred

owl_axioms:
  - axiom_id, axiom_type
  - subject_iri, object_iri
  - is_inferred, confidence, reasoning_rule
  - axiom_content (Clojure syntax)

owl_cross_references:
  - source_class_iri, target_reference
  - reference_type
```

## Migration Process

### Step 1: Backup Your Database

```bash
# Backup production database
cp /path/to/unified.db /path/to/unified.db.backup_$(date +%Y%m%d_%H%M%S)

# Verify backup
sqlite3 /path/to/unified.db.backup_* "SELECT COUNT(*) FROM owl_classes;"
```

### Step 2: Test Migration

```bash
# Run test migration script
cd /home/user/VisionFlow
./scripts/test_rich_ontology_migration.sh

# Review test results
sqlite3 scripts/test_unified.db "SELECT * FROM owl_classes_with_quality LIMIT 5;"
```

### Step 3: Apply Migration to Production

```bash
# Apply migration
sqlite3 /path/to/unified.db < scripts/migrations/002_rich_ontology_metadata.sql

# Verify migration
sqlite3 /path/to/unified.db <<EOF
SELECT version FROM schema_version;
SELECT COUNT(*) FROM owl_classes;
SELECT COUNT(*) FROM owl_relationships;
PRAGMA foreign_key_check;
EOF
```

### Step 4: Verify Data Integrity

```bash
# Check that all data was preserved
sqlite3 /path/to/unified.db <<EOF
.mode column
.headers on

-- Verify class count
SELECT 'Classes preserved: ' || COUNT(*) FROM owl_classes;

-- Check for NULL required fields
SELECT COUNT(*) as classes_with_nulls
FROM owl_classes
WHERE class_iri IS NULL OR ontology_id IS NULL;

-- Verify indexes
SELECT COUNT(*) as index_count
FROM sqlite_master
WHERE type='index' AND tbl_name LIKE 'owl%';

-- Verify foreign keys
PRAGMA foreign_key_check;
EOF
```

## Schema Details

### owl_classes Table

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| ontology_id | TEXT | NO | Ontology identifier | 'default' |
| class_iri | TEXT | NO | Class IRI (PK with ontology_id) | 'bc:SmartContract' |
| term_id | TEXT | YES | Ontology term ID | 'BC-0478' |
| preferred_term | TEXT | YES | Canonical term name | 'Smart Contract' |
| label | TEXT | YES | Display label | 'Smart Contract' |
| comment | TEXT | YES | Description | 'A self-executing contract' |
| parent_class_iri | TEXT | YES | Parent class | 'bc:DigitalAsset' |
| source_domain | TEXT | YES | Domain classification | 'blockchain', 'ai', 'metaverse' |
| version | TEXT | YES | Version number | '2.0' |
| type | TEXT | YES | Entity type | 'Protocol', 'Asset', 'Agent' |
| status | TEXT | YES | Lifecycle status | 'draft', 'review', 'approved', 'deprecated' |
| maturity | TEXT | YES | Maturity level | 'experimental', 'beta', 'stable' |
| quality_score | REAL | YES | Quality metric (0.0-1.0) | 0.95 |
| authority_score | REAL | YES | Authority metric (0.0-1.0) | 0.88 |
| public_access | INTEGER | YES | Publishing flag (0/1) | 1 |
| content_status | TEXT | YES | Workflow state | 'pending', 'published' |
| owl_physicality | TEXT | YES | Physicality | 'physical', 'virtual', 'abstract' |
| owl_role | TEXT | YES | Role classification | 'agent', 'patient', 'instrument' |
| belongs_to_domain | TEXT | YES | Primary domain | 'blockchain' |
| bridges_to_domain | TEXT | YES | Cross-domain bridge | 'ai' |
| source_file | TEXT | YES | Original file path | '/path/to/file.md' |
| file_sha1 | TEXT | YES | File content hash | 'abc123...' |
| markdown_content | TEXT | YES | Full markdown content | '# Smart Contract\n...' |
| last_synced | TIMESTAMP | YES | Last GitHub sync | '2025-11-22 10:30:00' |
| additional_metadata | TEXT | YES | JSON for unknown fields | '{"custom_field": "value"}' |

### owl_relationships Table

| Column | Type | Nullable | Description | Example |
|--------|------|----------|-------------|---------|
| ontology_id | TEXT | NO | Ontology identifier | 'default' |
| source_class_iri | TEXT | NO | Source class | 'bc:DeFiProtocol' |
| relationship_type | TEXT | NO | Relationship type | 'has-part', 'uses', 'enables', 'requires' |
| target_class_iri | TEXT | NO | Target class | 'bc:SmartContract' |
| confidence | REAL | NO | Confidence score (0.0-1.0) | 1.0 |
| is_inferred | INTEGER | NO | Inferred flag (0/1) | 0 |

### Common Relationship Types

- `subclass-of` - Class hierarchy (A is a type of B)
- `has-part` - Composition (A contains B)
- `uses` - Dependency (A uses B)
- `enables` - Capability (A enables B)
- `requires` - Requirement (A requires B)
- `bridges-to` - Cross-domain bridge (A connects to domain B)

## Indexes

The migration creates optimized indexes for common query patterns:

### owl_classes Indexes

```sql
-- Unique constraint (allows foreign key references)
CREATE UNIQUE INDEX idx_owl_classes_iri_unique ON owl_classes(class_iri);

-- Common queries
CREATE INDEX idx_owl_classes_ontology ON owl_classes(ontology_id);
CREATE INDEX idx_owl_classes_parent ON owl_classes(parent_class_iri);
CREATE INDEX idx_owl_classes_label ON owl_classes(label);

-- Rich metadata
CREATE INDEX idx_owl_classes_term_id ON owl_classes(term_id);
CREATE INDEX idx_owl_classes_source_domain ON owl_classes(source_domain);
CREATE INDEX idx_owl_classes_status ON owl_classes(status);
CREATE INDEX idx_owl_classes_quality ON owl_classes(quality_score DESC);
CREATE INDEX idx_owl_classes_physicality ON owl_classes(owl_physicality);
CREATE INDEX idx_owl_classes_sha1 ON owl_classes(file_sha1);
```

### owl_relationships Indexes

```sql
CREATE INDEX idx_owl_rel_source ON owl_relationships(source_class_iri);
CREATE INDEX idx_owl_rel_target ON owl_relationships(target_class_iri);
CREATE INDEX idx_owl_rel_type ON owl_relationships(relationship_type);
CREATE INDEX idx_owl_rel_inferred ON owl_relationships(is_inferred);
```

## Views

The migration creates useful views for common queries:

### owl_classes_with_quality

```sql
CREATE VIEW owl_classes_with_quality AS
SELECT
    class_iri,
    label,
    source_domain,
    quality_score,
    authority_score,
    status,
    maturity,
    COALESCE(quality_score, 0.0) * COALESCE(authority_score, 0.0) as combined_score
FROM owl_classes
WHERE quality_score IS NOT NULL OR authority_score IS NOT NULL
ORDER BY combined_score DESC;
```

**Usage**:
```sql
-- Get top quality classes
SELECT * FROM owl_classes_with_quality LIMIT 10;

-- Filter by domain
SELECT * FROM owl_classes_with_quality
WHERE source_domain = 'blockchain';
```

### owl_cross_domain_bridges

```sql
CREATE VIEW owl_cross_domain_bridges AS
SELECT
    class_iri,
    label,
    belongs_to_domain,
    bridges_to_domain,
    source_domain
FROM owl_classes
WHERE bridges_to_domain IS NOT NULL;
```

**Usage**:
```sql
-- Find all cross-domain bridges
SELECT * FROM owl_cross_domain_bridges;

-- Find blockchain-to-AI bridges
SELECT * FROM owl_cross_domain_bridges
WHERE belongs_to_domain = 'blockchain' AND bridges_to_domain = 'ai';
```

### owl_relationship_graph

```sql
CREATE VIEW owl_relationship_graph AS
SELECT
    r.source_class_iri,
    sc.label as source_label,
    r.relationship_type,
    r.target_class_iri,
    tc.label as target_label,
    r.confidence,
    r.is_inferred
FROM owl_relationships r
LEFT JOIN owl_classes sc ON r.source_class_iri = sc.class_iri
LEFT JOIN owl_classes tc ON r.target_class_iri = tc.class_iri;
```

**Usage**:
```sql
-- Get all relationships with labels
SELECT * FROM owl_relationship_graph;

-- Find all 'has-part' relationships
SELECT * FROM owl_relationship_graph
WHERE relationship_type = 'has-part';
```

## Rust Adapter Usage

The new `SqliteOntologyRepository` adapter supports rich metadata:

### Basic Usage

```rust
use visionflow::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
use visionflow::ports::ontology_repository::OntologyRepository;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create repository
    let repo = SqliteOntologyRepository::new("unified.db")?;

    // Add a class with rich metadata (using extended struct)
    let class = OwlClassExtended {
        iri: "bc:DeFiProtocol".to_string(),
        term_id: Some("BC-1234".to_string()),
        preferred_term: Some("Decentralized Finance Protocol".to_string()),
        label: Some("DeFi Protocol".to_string()),
        description: Some("A protocol for decentralized financial services".to_string()),
        source_domain: Some("blockchain".to_string()),
        quality_score: Some(0.95),
        authority_score: Some(0.88),
        status: Some("approved".to_string()),
        maturity: Some("stable".to_string()),
        owl_physicality: Some("virtual".to_string()),
        owl_role: Some("agent".to_string()),
        bridges_to_domain: Some("ai".to_string()),
        // ... other fields
    };

    // Query by quality
    let high_quality = repo.query_classes_by_quality(0.8).await?;
    println!("Found {} high-quality classes", high_quality.len());

    Ok(())
}
```

### Adding Relationships

```rust
use visionflow::adapters::sqlite_ontology_repository::OwlRelationship;

// Create a relationship
let relationship = OwlRelationship {
    source_class_iri: "bc:DeFiProtocol".to_string(),
    relationship_type: "has-part".to_string(),
    target_class_iri: "bc:SmartContract".to_string(),
    confidence: 1.0,
    is_inferred: false,
};

// Add to database
repo.add_relationship(&relationship).await?;

// Query relationships
let rels = repo.get_relationships("bc:DeFiProtocol").await?;
for rel in rels {
    println!("{} --[{}]--> {}",
        rel.source_class_iri, rel.relationship_type, rel.target_class_iri);
}
```

## Performance Considerations

### Query Optimization

The migration includes indexes for common query patterns:

```sql
-- Fast quality filtering
SELECT * FROM owl_classes WHERE quality_score >= 0.8;

-- Fast domain filtering
SELECT * FROM owl_classes WHERE source_domain = 'blockchain';

-- Fast relationship traversal
SELECT * FROM owl_relationships WHERE source_class_iri = 'bc:DeFiProtocol';
```

### Batch Operations

For large imports, use transactions:

```rust
// In Rust adapter
self.base.execute_transaction(|tx| {
    for class in &classes {
        // Insert class...
    }
    Ok(())
}).await?;
```

## Backward Compatibility

The migration preserves all existing data:

1. **Schema preserves old columns**: All existing columns remain
2. **New columns are nullable**: No data loss
3. **Backward-compatible queries**: Old queries continue to work
4. **Adapter compatibility**: The `OwlClass` struct is preserved

### Converting Between Formats

```rust
// Old format to new
let extended = SqliteOntologyRepository::owl_class_to_extended(&old_class);

// New format to old (for legacy code)
let old_class = SqliteOntologyRepository::extended_to_owl_class(&extended);
```

## Rollback Procedure

If you need to rollback:

```bash
# Restore from backup
cp /path/to/unified.db.backup_TIMESTAMP /path/to/unified.db

# Verify restoration
sqlite3 /path/to/unified.db "SELECT version FROM schema_version;"
# Should show: 1
```

## Troubleshooting

### Foreign Key Violations

```sql
-- Check for violations
PRAGMA foreign_key_check;

-- If violations exist, find them
SELECT * FROM owl_relationships r
LEFT JOIN owl_classes c ON r.source_class_iri = c.class_iri
WHERE c.class_iri IS NULL;
```

### Missing Indexes

```sql
-- Verify all indexes exist
SELECT name, tbl_name FROM sqlite_master
WHERE type='index' AND tbl_name LIKE 'owl%'
ORDER BY tbl_name, name;

-- Should see 14+ indexes
```

### Performance Issues

```sql
-- Analyze tables
ANALYZE owl_classes;
ANALYZE owl_relationships;
ANALYZE owl_properties;

-- Rebuild indexes if needed
REINDEX owl_classes;
REINDEX owl_relationships;
```

## Next Steps

1. **Run migration test**: `./scripts/test_rich_ontology_migration.sh`
2. **Backup production database**: `cp unified.db unified.db.backup`
3. **Apply migration**: `sqlite3 unified.db < scripts/migrations/002_rich_ontology_metadata.sql`
4. **Update Rust code**: Use `SqliteOntologyRepository` with rich metadata
5. **Populate metadata**: Import ontology data with quality scores

## References

- **Migration SQL**: `/home/user/VisionFlow/scripts/migrations/002_rich_ontology_metadata.sql`
- **Test Script**: `/home/user/VisionFlow/scripts/test_rich_ontology_migration.sh`
- **Rust Adapter**: `/home/user/VisionFlow/src/adapters/sqlite_ontology_repository.rs`
- **Ontology Parser**: `/home/user/VisionFlow/multi-agent-docker/skills/ontology-core/src/ontology_parser.py`
- **Architecture Docs**: `/home/user/VisionFlow/docs/concepts/architecture/ontology-storage-architecture.md`

## Support

For issues or questions:
1. Check test migration output: `sqlite3 scripts/test_unified.db`
2. Review migration logs
3. Verify foreign key constraints: `PRAGMA foreign_key_check;`
4. Check schema version: `SELECT * FROM schema_version;`
