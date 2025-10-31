# VisionFlow Ontology Database Schema Reference

## Quick Reference

### Correct INSERT Patterns

#### Adding an OWL Class
```rust
conn.execute(
    "INSERT OR REPLACE INTO owl_classes (ontology_id, class_iri, label, comment, file_sha1)
     VALUES (?1, ?2, ?3, ?4, ?5)",
    params!["default", &class.iri, &class.label, &class.description, &class.file_sha1]
)
```

**Column Names**:
- ✅ `ontology_id` (required, use "default")
- ✅ `class_iri` (NOT `iri`)
- ✅ `label` (optional)
- ✅ `comment` (NOT `description`)
- ✅ `file_sha1` (optional)

#### Adding Class Hierarchy
```rust
conn.execute(
    "INSERT INTO owl_class_hierarchy (class_iri, parent_iri) VALUES (?1, ?2)",
    params![&class.iri, parent_iri]
)
```

**Important**: Both `class_iri` and `parent_iri` must exist in `owl_classes.class_iri`

#### Adding an OWL Property
```rust
conn.execute(
    "INSERT OR REPLACE INTO owl_properties (ontology_id, property_iri, property_type, label)
     VALUES (?1, ?2, ?3, ?4)",
    params!["default", &property.iri, property_type_str, &property.label]
)
```

**Column Names**:
- ✅ `ontology_id` (required, use "default")
- ✅ `property_iri` (NOT `iri`)
- ✅ `property_type` (one of: ObjectProperty, DataProperty, AnnotationProperty)
- ✅ `label` (optional)

### Correct SELECT Patterns

#### Querying OWL Classes
```rust
conn.query_row(
    "SELECT class_iri, label, comment, file_sha1 FROM owl_classes WHERE class_iri = ?1",
    params![iri_owned],
    |row| {
        let class_iri: String = row.get(0)?;  // NOT iri
        let label: Option<String> = row.get(1)?;
        let comment: Option<String> = row.get(2)?;  // NOT description
        let file_sha1: Option<String> = row.get(3)?;
        Ok((class_iri, label, comment, file_sha1))
    }
)
```

#### Querying Class Parents
```rust
conn.prepare("SELECT parent_iri FROM owl_class_hierarchy WHERE class_iri = ?1")?
    .query_map(params![&iri], |row| row.get(0))?
    .collect::<Result<Vec<String>, _>>()?
```

#### Querying OWL Properties
```rust
conn.query_row(
    "SELECT property_iri, label, property_type FROM owl_properties WHERE property_iri = ?1",
    params![iri_owned],
    |row| {
        let property_iri: String = row.get(0)?;  // NOT iri
        let label: Option<String> = row.get(1)?;
        let property_type: String = row.get(2)?;
        Ok((property_iri, label, property_type))
    }
)
```

## Critical Points

### 1. Unique Index Requirement
```sql
-- REQUIRED for foreign keys to work!
CREATE UNIQUE INDEX idx_owl_classes_iri_unique ON owl_classes(class_iri);
```
Without this index, foreign keys in `owl_class_hierarchy` will fail because `owl_classes` has a composite primary key.

### 2. Foreign Key Enforcement
```rust
// MUST disable before bulk operations
conn.execute("PRAGMA foreign_keys = OFF", [])?;

// Do bulk operations...

// Re-enable after
conn.execute("PRAGMA foreign_keys = ON", [])?;
```

### 3. Default Ontology
All classes and properties must reference an ontology:
```sql
INSERT OR IGNORE INTO ontologies (
    ontology_id,
    source_path,
    source_type,
    content_hash,
    title
)
VALUES (
    'default',
    'default',
    'embedded',
    'default-ontology',
    'Default Ontology'
);
```

## Common Mistakes to Avoid

### ❌ WRONG Column Names
```rust
// DON'T use these column names!
conn.execute(
    "INSERT INTO owl_classes (iri, description) VALUES (?1, ?2)",  // WRONG!
    params![&class.iri, &class.description]
)
```

### ✅ CORRECT Column Names
```rust
conn.execute(
    "INSERT INTO owl_classes (ontology_id, class_iri, comment) VALUES (?1, ?2, ?3)",
    params!["default", &class.iri, &class.description]
)
```

### ❌ WRONG Foreign Key References
```sql
-- DON'T reference 'iri' column (doesn't exist!)
FOREIGN KEY (class_iri) REFERENCES owl_classes(iri)  -- WRONG!
```

### ✅ CORRECT Foreign Key References
```sql
-- DO reference 'class_iri' column (with unique index!)
FOREIGN KEY (class_iri) REFERENCES owl_classes(class_iri)  -- CORRECT!
```

### ❌ WRONG Primary Key Assumption
```rust
// DON'T assume 'iri' is the primary key!
conn.query_row(
    "SELECT * FROM owl_classes WHERE iri = ?1",  // WRONG! iri doesn't exist
    params![class_iri]
)
```

### ✅ CORRECT Primary Key Usage
```rust
// DO use the composite primary key OR class_iri (which has unique index)
conn.query_row(
    "SELECT * FROM owl_classes WHERE class_iri = ?1",  // CORRECT!
    params![class_iri]
)

// OR for specific ontology:
conn.query_row(
    "SELECT * FROM owl_classes WHERE ontology_id = ?1 AND class_iri = ?2",
    params![ontology_id, class_iri]
)
```

## Schema Validation Queries

### Check Foreign Key Constraints
```sql
-- Should return empty if all constraints are valid
PRAGMA foreign_key_check;
```

### Verify Unique Index Exists
```sql
-- Should show idx_owl_classes_iri_unique with unique=1
PRAGMA index_list('owl_classes');
```

### Count Records
```sql
SELECT
    (SELECT COUNT(*) FROM owl_classes) as classes,
    (SELECT COUNT(*) FROM owl_class_hierarchy) as hierarchies,
    (SELECT COUNT(*) FROM owl_properties) as properties;
```

### Find Orphaned Hierarchies
```sql
-- Should return empty if all foreign keys are valid
SELECT DISTINCT h.class_iri
FROM owl_class_hierarchy h
LEFT JOIN owl_classes c ON h.class_iri = c.class_iri
WHERE c.class_iri IS NULL;
```

## Migration Checklist

When modifying the schema:
- [ ] Update schema creation in `sqlite_ontology_repository.rs`
- [ ] Update all INSERT statements with correct column names
- [ ] Update all SELECT queries with correct column names
- [ ] Create migration script in `/scripts/`
- [ ] Test migration on actual database
- [ ] Run `PRAGMA foreign_key_check`
- [ ] Verify unique indexes exist
- [ ] Update tests
- [ ] Document changes

## Testing

Run the standalone test:
```bash
./tests/standalone_ontology_schema_test.sh
```

Expected output:
```
✅ Schema created successfully
✅ INSERT operations successful
✅ Hierarchy INSERT successful
✅ Foreign key constraints working correctly
✅ Foreign key check passed
✅ All schema tests passed
```
