# Ontology Schema Fixes - Summary

## Overview
Fixed critical database schema mismatches in VisionFlow's ontology repository that were preventing incremental saves from working correctly.

## Problems Identified

### 1. Foreign Key Mismatches in `owl_class_hierarchy`
**Issue**: Foreign keys referenced `owl_classes(iri)` but the column is actually `class_iri`
```sql
-- WRONG (old schema)
FOREIGN KEY (class_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE

-- CORRECT (new schema)
FOREIGN KEY (class_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE
```

### 2. Composite Primary Key Reference Issue
**Issue**: `owl_classes` has composite primary key `(ontology_id, class_iri)` but foreign keys only reference `class_iri`

**Solution**: Created unique index on `class_iri` to allow single-column foreign key references
```sql
CREATE UNIQUE INDEX idx_owl_classes_iri_unique ON owl_classes(class_iri);
```

### 3. Schema Mismatch Between Code and Database

**Old Schema in Code** (sqlite_ontology_repository.rs):
```sql
CREATE TABLE owl_classes (
    iri TEXT PRIMARY KEY,  -- WRONG!
    label TEXT,
    ...
)
```

**Actual Database Schema**:
```sql
CREATE TABLE owl_classes (
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,  -- Correct column name
    label TEXT,
    comment TEXT,
    parent_class_iri TEXT,
    is_deprecated INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_sha1 TEXT,
    PRIMARY KEY (ontology_id, class_iri)
)
```

### 4. INSERT Statement Column Name Errors
**Issue**: Code used wrong column names in INSERT statements
- Used `iri` instead of `class_iri`
- Used `description` instead of `comment`
- Missing `ontology_id` in INSERTs

## Files Modified

### 1. `/home/devuser/workspace/project/src/adapters/sqlite_ontology_repository.rs`
**Changes**:
- Updated schema creation to match actual database (lines 33-127)
- Fixed `ontologies` table schema with all required columns
- Added unique index `idx_owl_classes_iri_unique` on `class_iri`
- Fixed foreign key constraints in `owl_class_hierarchy`
- Updated all INSERT statements to use correct column names:
  - `ontology_id, class_iri, label, comment, file_sha1` (not iri, description)
- Updated all SELECT queries to use `class_iri` column

### 2. `/home/devuser/workspace/project/schema/ontology_metadata_db.sql`
**Created**: Copy of `ontology_db.sql` to match expected filename in database_service.rs

### 3. `/home/devuser/workspace/project/scripts/fix-ontology-schema.sql`
**Created**: Initial migration script (superseded by v2)

### 4. `/home/devuser/workspace/project/scripts/fix-ontology-schema-v2.sql`
**Created**: Final migration script that:
- Disables foreign keys before schema changes
- Creates unique index on `class_iri`
- Drops and recreates `owl_class_hierarchy` with correct foreign keys
- Preserves existing data
- Re-enables foreign keys
- Validates constraints

### 5. `/home/devuser/workspace/project/tests/standalone_ontology_schema_test.sh`
**Created**: Comprehensive standalone test that validates:
- Schema creation
- INSERT operations with correct column names
- Foreign key constraints work properly
- Hierarchy relationships
- Property storage

## Migration Applied

The migration was successfully applied to:
- `/home/devuser/workspace/project/data/ontology.db`

**Verification**:
```bash
sqlite3 /home/devuser/workspace/project/data/ontology.db "PRAGMA foreign_key_check;"
# Returns empty (no violations) ✅
```

## Testing Results

### Standalone Test Results
```
✅ Schema created successfully
✅ INSERT operations successful
✅ Hierarchy INSERT successful
✅ Foreign key constraints working correctly
✅ Foreign key check passed (no violations)
✅ Property INSERT successful
✅ All schema tests passed
```

### Test Coverage
- [x] Schema creation with correct column names
- [x] INSERT statements use correct columns
- [x] Foreign key constraints properly enforced
- [x] Unique index allows single-column FK references
- [x] Class hierarchy relationships work
- [x] Property storage works
- [x] No foreign key violations

## Actual Database Schema (Verified)

### ontologies
```sql
CREATE TABLE ontologies (
    ontology_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('file', 'url', 'embedded')),
    base_iri TEXT,
    version_iri TEXT,
    title TEXT,
    description TEXT,
    author TEXT,
    version TEXT,
    content_hash TEXT NOT NULL,
    axiom_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,
    property_count INTEGER DEFAULT 0,
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_validated_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### owl_classes
```sql
CREATE TABLE owl_classes (
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,
    label TEXT,
    comment TEXT,
    parent_class_iri TEXT,
    is_deprecated INTEGER DEFAULT 0 CHECK (is_deprecated IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_sha1 TEXT,
    PRIMARY KEY (ontology_id, class_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

-- CRITICAL: Unique index required for foreign key references
CREATE UNIQUE INDEX idx_owl_classes_iri_unique ON owl_classes(class_iri);
```

### owl_class_hierarchy
```sql
CREATE TABLE owl_class_hierarchy (
    class_iri TEXT NOT NULL,
    parent_iri TEXT NOT NULL,
    PRIMARY KEY (class_iri, parent_iri),
    FOREIGN KEY (class_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE,
    FOREIGN KEY (parent_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE
);
```

### owl_properties
```sql
CREATE TABLE owl_properties (
    ontology_id TEXT NOT NULL,
    property_iri TEXT NOT NULL,
    property_type TEXT NOT NULL CHECK (property_type IN ('ObjectProperty', 'DataProperty', 'AnnotationProperty')),
    label TEXT,
    comment TEXT,
    domain_class_iri TEXT,
    range_class_iri TEXT,
    is_functional INTEGER DEFAULT 0 CHECK (is_functional IN (0, 1)),
    is_inverse_functional INTEGER DEFAULT 0 CHECK (is_inverse_functional IN (0, 1)),
    is_symmetric INTEGER DEFAULT 0 CHECK (is_symmetric IN (0, 1)),
    is_transitive INTEGER DEFAULT 0 CHECK (is_transitive IN (0, 1)),
    inverse_property_iri TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ontology_id, property_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);
```

## Key Insights

1. **Unique Index Critical**: When a table has a composite primary key but you want to reference just one column in a foreign key, you MUST create a unique index on that column.

2. **PRAGMA foreign_keys**: SQLite has foreign keys disabled by default. The code correctly sets `PRAGMA foreign_keys = OFF` before schema changes and `PRAGMA foreign_keys = ON` after.

3. **Column Name Consistency**: The actual database uses:
   - `class_iri` (not `iri`)
   - `comment` (not `description`)
   - `property_iri` (not `iri`)
   - Composite primary keys `(ontology_id, class_iri)`

## Impact

These fixes enable:
- ✅ Incremental ontology saves to work correctly
- ✅ Foreign key integrity enforcement
- ✅ Proper class hierarchy relationships
- ✅ Correct property storage
- ✅ Database schema consistency between code and actual DB

## Future Considerations

1. **Schema Versioning**: Consider adding schema version tracking to detect mismatches
2. **Migration Framework**: Implement proper migration tooling for schema evolution
3. **Integration Tests**: Add integration tests that validate schema against actual database
4. **Type Safety**: Consider using sqlx compile-time query checking to catch column name errors

## Related Issues

- Incremental saves were failing due to foreign key violations
- INSERT statements were using incorrect column names
- Foreign key constraints were not properly enforced

## Author
Generated by Claude Code on 2025-10-31
