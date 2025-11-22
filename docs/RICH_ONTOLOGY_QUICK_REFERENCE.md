# Rich Ontology Metadata - Quick Reference Card

## Schema Version 2 - New Fields

### owl_classes Table

#### Core Identification
```sql
term_id          TEXT      -- "BC-0478"
preferred_term   TEXT      -- "Smart Contract"
```

#### Classification
```sql
source_domain    TEXT      -- blockchain/ai/metaverse/rb/dt
version          TEXT      -- "2.0"
type             TEXT      -- Protocol/Asset/Agent
```

#### Quality Metrics
```sql
status           TEXT      -- draft/review/approved/deprecated
maturity         TEXT      -- experimental/beta/stable
quality_score    REAL      -- 0.0-1.0
authority_score  REAL      -- 0.0-1.0
public_access    INTEGER   -- 0/1
content_status   TEXT      -- pending/published
```

#### OWL2 Properties
```sql
owl_physicality  TEXT      -- physical/virtual/abstract
owl_role         TEXT      -- agent/patient/instrument
```

#### Domain Relationships
```sql
belongs_to_domain   TEXT   -- Primary domain
bridges_to_domain   TEXT   -- Cross-domain bridge
```

#### Source Tracking
```sql
source_file         TEXT   -- File path
file_sha1           TEXT   -- Content hash
markdown_content    TEXT   -- Full markdown
last_synced         TIMESTAMP
```

#### Extensibility
```sql
additional_metadata TEXT   -- JSON blob
```

## Common Queries

### Top Quality Classes
```sql
SELECT class_iri, label, quality_score, authority_score
FROM owl_classes_with_quality
LIMIT 10;
```

### Domain Filtering
```sql
SELECT class_iri, label, quality_score
FROM owl_classes
WHERE source_domain = 'blockchain'
  AND quality_score >= 0.8;
```

### Cross-Domain Bridges
```sql
SELECT * FROM owl_cross_domain_bridges
WHERE belongs_to_domain = 'blockchain'
  AND bridges_to_domain = 'ai';
```

### Relationship Traversal
```sql
SELECT * FROM owl_relationship_graph
WHERE source_class_iri = 'bc:DeFiProtocol'
  AND relationship_type = 'has-part';
```

### Mature Stable Classes
```sql
SELECT class_iri, label, maturity, status
FROM owl_classes
WHERE maturity = 'stable'
  AND status = 'approved'
ORDER BY quality_score DESC;
```

## Rust Usage

### Create Repository
```rust
use visionflow::adapters::sqlite_ontology_repository::SqliteOntologyRepository;

let repo = SqliteOntologyRepository::new("unified.db")?;
```

### Add Class with Metadata
```rust
let class = OwlClassExtended {
    iri: "bc:DeFiProtocol".to_string(),
    term_id: Some("BC-1234".to_string()),
    quality_score: Some(0.95),
    authority_score: Some(0.88),
    source_domain: Some("blockchain".to_string()),
    status: Some("approved".to_string()),
    maturity: Some("stable".to_string()),
    // ...
};
```

### Query by Quality
```rust
let high_quality = repo.query_classes_by_quality(0.8).await?;
```

### Add Relationship
```rust
let rel = OwlRelationship {
    source_class_iri: "bc:DeFiProtocol".to_string(),
    relationship_type: "has-part".to_string(),
    target_class_iri: "bc:SmartContract".to_string(),
    confidence: 1.0,
    is_inferred: false,
};
repo.add_relationship(&rel).await?;
```

### Get Relationships
```rust
let rels = repo.get_relationships("bc:DeFiProtocol").await?;
```

## Migration Commands

### Test Migration
```bash
./scripts/test_rich_ontology_migration.sh
```

### Backup Database
```bash
cp unified.db unified.db.backup_$(date +%Y%m%d_%H%M%S)
```

### Apply Migration
```bash
sqlite3 unified.db < scripts/migrations/002_rich_ontology_metadata.sql
```

### Verify Migration
```sql
SELECT version FROM schema_version;  -- Should be 2
SELECT COUNT(*) FROM owl_classes;
PRAGMA foreign_key_check;            -- Should be empty
```

## Relationship Types

| Type | Description | Example |
|------|-------------|---------|
| `subclass-of` | Class hierarchy | Vehicle ⊆ Transport |
| `has-part` | Composition | Car has Engine |
| `uses` | Dependency | App uses API |
| `enables` | Capability | API enables Integration |
| `requires` | Requirement | Feature requires License |
| `bridges-to` | Cross-domain | Blockchain bridges-to AI |

## Views

### owl_classes_with_quality
Classes with quality metrics and combined score

### owl_cross_domain_bridges
Classes that bridge multiple domains

### owl_relationship_graph
Semantic relationships with human-readable labels

## Indexes (18 total)

- **owl_classes**: 10 indexes (iri, ontology, parent, label, term_id, domain, status, quality, physicality, sha1)
- **owl_relationships**: 4 indexes (source, target, type, inferred)
- **owl_properties**: 4 indexes (iri, type, domain, range)

## Troubleshooting

### Check Schema Version
```sql
SELECT * FROM schema_version;
```

### Verify Data
```sql
SELECT COUNT(*) FROM owl_classes;
SELECT COUNT(*) FROM owl_relationships;
```

### Check Foreign Keys
```sql
PRAGMA foreign_key_check;
```

### Rebuild Indexes
```sql
REINDEX owl_classes;
ANALYZE;
```

## Files

- Migration: `scripts/migrations/002_rich_ontology_metadata.sql`
- Adapter: `src/adapters/sqlite_ontology_repository.rs`
- Test: `scripts/test_rich_ontology_migration.sh`
- Docs: `docs/guides/rich-ontology-metadata-migration.md`
