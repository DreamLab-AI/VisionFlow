-- Migration: Fix Ontology Schema Foreign Key Constraints
-- Version: 001
-- Date: 2025-10-31
-- Description: Correct foreign key references in owl_class_hierarchy table
--              and align column names with actual schema

-- ===========================================================================
-- STEP 1: Backup Current Schema
-- ===========================================================================
-- IMPORTANT: Create a backup of ontology.db BEFORE running this migration
-- Command: cp ontology.db ontology.db.backup.$(date +%Y%m%d_%H%M%S)

-- ===========================================================================
-- STEP 2: Create New Tables with Corrected Schema
-- ===========================================================================

-- New owl_classes table with composite primary key
CREATE TABLE IF NOT EXISTS owl_classes_new (
    ontology_id TEXT NOT NULL DEFAULT 'default',
    class_iri TEXT NOT NULL,
    label TEXT,
    comment TEXT,
    file_sha1 TEXT,
    last_synced DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ontology_id, class_iri)
) STRICT;

-- Indexes for owl_classes_new
CREATE INDEX IF NOT EXISTS idx_owl_classes_new_label
    ON owl_classes_new(label);

CREATE INDEX IF NOT EXISTS idx_owl_classes_new_sha1
    ON owl_classes_new(file_sha1);

CREATE INDEX IF NOT EXISTS idx_owl_classes_new_ontology
    ON owl_classes_new(ontology_id);

-- New owl_class_hierarchy with CORRECT foreign keys
CREATE TABLE IF NOT EXISTS owl_class_hierarchy_new (
    ontology_id TEXT NOT NULL DEFAULT 'default',
    class_iri TEXT NOT NULL,
    parent_iri TEXT NOT NULL,
    PRIMARY KEY (ontology_id, class_iri, parent_iri),

    -- CORRECT: References composite primary key (ontology_id, class_iri)
    FOREIGN KEY (ontology_id, class_iri)
        REFERENCES owl_classes_new(ontology_id, class_iri)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    -- CORRECT: References composite primary key for parent
    FOREIGN KEY (ontology_id, parent_iri)
        REFERENCES owl_classes_new(ontology_id, class_iri)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) STRICT;

-- Indexes for owl_class_hierarchy_new
CREATE INDEX IF NOT EXISTS idx_hierarchy_new_class
    ON owl_class_hierarchy_new(ontology_id, class_iri);

CREATE INDEX IF NOT EXISTS idx_hierarchy_new_parent
    ON owl_class_hierarchy_new(ontology_id, parent_iri);

-- New owl_properties table with composite primary key
CREATE TABLE IF NOT EXISTS owl_properties_new (
    ontology_id TEXT NOT NULL DEFAULT 'default',
    property_iri TEXT NOT NULL,
    label TEXT,
    property_type TEXT NOT NULL CHECK(
        property_type IN ('ObjectProperty', 'DataProperty', 'AnnotationProperty')
    ),
    domain TEXT,  -- JSON array of class IRIs
    range TEXT,   -- JSON array of class IRIs or datatypes
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ontology_id, property_iri)
) STRICT;

-- Indexes for owl_properties_new
CREATE INDEX IF NOT EXISTS idx_properties_new_type
    ON owl_properties_new(property_type);

CREATE INDEX IF NOT EXISTS idx_properties_new_ontology
    ON owl_properties_new(ontology_id);

-- owl_axioms table with ontology_id added
CREATE TABLE IF NOT EXISTS owl_axioms_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL DEFAULT 'default',
    axiom_type TEXT NOT NULL CHECK(
        axiom_type IN (
            'SubClassOf',
            'EquivalentClass',
            'DisjointWith',
            'ObjectPropertyAssertion',
            'DataPropertyAssertion'
        )
    ),
    subject TEXT NOT NULL,
    object TEXT NOT NULL,
    annotations TEXT,  -- JSON
    is_inferred BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) STRICT;

-- Indexes for owl_axioms_new
CREATE INDEX IF NOT EXISTS idx_axioms_new_ontology
    ON owl_axioms_new(ontology_id);

CREATE INDEX IF NOT EXISTS idx_axioms_new_subject
    ON owl_axioms_new(subject);

CREATE INDEX IF NOT EXISTS idx_axioms_new_type
    ON owl_axioms_new(axiom_type);

CREATE INDEX IF NOT EXISTS idx_axioms_new_subject_object
    ON owl_axioms_new(subject, object);

-- ===========================================================================
-- STEP 3: Migrate Existing Data
-- ===========================================================================

-- Migrate owl_classes data
INSERT INTO owl_classes_new (
    ontology_id,
    class_iri,
    label,
    comment,
    file_sha1,
    last_synced,
    created_at,
    updated_at
)
SELECT
    COALESCE(ontology_id, 'default') AS ontology_id,
    COALESCE(class_iri, iri) AS class_iri,  -- Handle both column names
    label,
    COALESCE(comment, description) AS comment,  -- Handle both column names
    file_sha1,
    last_synced,
    created_at,
    updated_at
FROM owl_classes
WHERE EXISTS (SELECT 1 FROM owl_classes LIMIT 1);

-- Migrate owl_class_hierarchy data
INSERT INTO owl_class_hierarchy_new (
    ontology_id,
    class_iri,
    parent_iri
)
SELECT
    COALESCE(h.ontology_id, 'default') AS ontology_id,
    h.class_iri,
    h.parent_iri
FROM owl_class_hierarchy h
WHERE EXISTS (SELECT 1 FROM owl_class_hierarchy LIMIT 1)
  -- Only migrate if both class and parent exist in new owl_classes table
  AND EXISTS (
      SELECT 1 FROM owl_classes_new c1
      WHERE c1.class_iri = h.class_iri
        AND c1.ontology_id = COALESCE(h.ontology_id, 'default')
  )
  AND EXISTS (
      SELECT 1 FROM owl_classes_new c2
      WHERE c2.class_iri = h.parent_iri
        AND c2.ontology_id = COALESCE(h.ontology_id, 'default')
  );

-- Migrate owl_properties data
INSERT INTO owl_properties_new (
    ontology_id,
    property_iri,
    label,
    property_type,
    domain,
    range,
    created_at,
    updated_at
)
SELECT
    COALESCE(ontology_id, 'default') AS ontology_id,
    COALESCE(property_iri, iri) AS property_iri,  -- Handle both column names
    label,
    property_type,
    domain,
    range,
    created_at,
    updated_at
FROM owl_properties
WHERE EXISTS (SELECT 1 FROM owl_properties LIMIT 1);

-- Migrate owl_axioms data
INSERT INTO owl_axioms_new (
    id,
    ontology_id,
    axiom_type,
    subject,
    object,
    annotations,
    is_inferred,
    created_at
)
SELECT
    id,
    COALESCE(ontology_id, 'default') AS ontology_id,
    axiom_type,
    subject,
    object,
    annotations,
    is_inferred,
    created_at
FROM owl_axioms
WHERE EXISTS (SELECT 1 FROM owl_axioms LIMIT 1);

-- ===========================================================================
-- STEP 4: Verification
-- ===========================================================================

-- Verify row counts match
SELECT
    'owl_classes' AS table_name,
    (SELECT COUNT(*) FROM owl_classes) AS old_count,
    (SELECT COUNT(*) FROM owl_classes_new) AS new_count,
    CASE
        WHEN (SELECT COUNT(*) FROM owl_classes) = (SELECT COUNT(*) FROM owl_classes_new)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status
UNION ALL
SELECT
    'owl_class_hierarchy',
    (SELECT COUNT(*) FROM owl_class_hierarchy),
    (SELECT COUNT(*) FROM owl_class_hierarchy_new),
    CASE
        WHEN (SELECT COUNT(*) FROM owl_class_hierarchy) >= (SELECT COUNT(*) FROM owl_class_hierarchy_new)
        THEN 'PASS'  -- May be less due to orphaned references
        ELSE 'FAIL'
    END
UNION ALL
SELECT
    'owl_properties',
    (SELECT COUNT(*) FROM owl_properties),
    (SELECT COUNT(*) FROM owl_properties_new),
    CASE
        WHEN (SELECT COUNT(*) FROM owl_properties) = (SELECT COUNT(*) FROM owl_properties_new)
        THEN 'PASS'
        ELSE 'FAIL'
    END
UNION ALL
SELECT
    'owl_axioms',
    (SELECT COUNT(*) FROM owl_axioms),
    (SELECT COUNT(*) FROM owl_axioms_new),
    CASE
        WHEN (SELECT COUNT(*) FROM owl_axioms) = (SELECT COUNT(*) FROM owl_axioms_new)
        THEN 'PASS'
        ELSE 'FAIL'
    END;

-- ===========================================================================
-- STEP 5: Swap Tables (ATOMIC OPERATION)
-- ===========================================================================

-- WARNING: This is the point of no return. Ensure backups exist!

BEGIN IMMEDIATE TRANSACTION;

-- Drop old tables
DROP TABLE IF EXISTS owl_class_hierarchy;
DROP TABLE IF EXISTS owl_axioms;
DROP TABLE IF EXISTS owl_properties;
DROP TABLE IF EXISTS owl_classes;

-- Rename new tables to original names
ALTER TABLE owl_classes_new RENAME TO owl_classes;
ALTER TABLE owl_class_hierarchy_new RENAME TO owl_class_hierarchy;
ALTER TABLE owl_properties_new RENAME TO owl_properties;
ALTER TABLE owl_axioms_new RENAME TO owl_axioms;

-- Recreate indexes with original names
DROP INDEX IF EXISTS idx_owl_classes_new_label;
DROP INDEX IF EXISTS idx_owl_classes_new_sha1;
DROP INDEX IF EXISTS idx_owl_classes_new_ontology;
DROP INDEX IF EXISTS idx_hierarchy_new_class;
DROP INDEX IF EXISTS idx_hierarchy_new_parent;
DROP INDEX IF EXISTS idx_properties_new_type;
DROP INDEX IF EXISTS idx_properties_new_ontology;
DROP INDEX IF EXISTS idx_axioms_new_ontology;
DROP INDEX IF EXISTS idx_axioms_new_subject;
DROP INDEX IF EXISTS idx_axioms_new_type;
DROP INDEX IF EXISTS idx_axioms_new_subject_object;

CREATE INDEX idx_owl_classes_label ON owl_classes(label);
CREATE INDEX idx_owl_classes_sha1 ON owl_classes(file_sha1);
CREATE INDEX idx_owl_classes_ontology ON owl_classes(ontology_id);
CREATE INDEX idx_hierarchy_class ON owl_class_hierarchy(ontology_id, class_iri);
CREATE INDEX idx_hierarchy_parent ON owl_class_hierarchy(ontology_id, parent_iri);
CREATE INDEX idx_properties_type ON owl_properties(property_type);
CREATE INDEX idx_properties_ontology ON owl_properties(ontology_id);
CREATE INDEX idx_axioms_ontology ON owl_axioms(ontology_id);
CREATE INDEX idx_axioms_subject ON owl_axioms(subject);
CREATE INDEX idx_axioms_type ON owl_axioms(axiom_type);
CREATE INDEX idx_axioms_subject_object ON owl_axioms(subject, object);

COMMIT;

-- ===========================================================================
-- STEP 6: Enable Foreign Keys and Verify Integrity
-- ===========================================================================

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Verify foreign key integrity
PRAGMA foreign_key_check;

-- ===========================================================================
-- STEP 7: Optimize Database
-- ===========================================================================

-- Analyze tables for query optimizer
ANALYZE owl_classes;
ANALYZE owl_class_hierarchy;
ANALYZE owl_properties;
ANALYZE owl_axioms;

-- Vacuum to reclaim space and optimize
VACUUM;

-- ===========================================================================
-- STEP 8: Post-Migration Verification
-- ===========================================================================

-- Verify schema
SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'owl_classes';
SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'owl_class_hierarchy';

-- Verify foreign keys work
SELECT * FROM pragma_foreign_key_list('owl_class_hierarchy');

-- Test insert with foreign key constraint
-- This should succeed if a class exists
-- INSERT INTO owl_class_hierarchy (ontology_id, class_iri, parent_iri)
-- VALUES ('default', 'http://test.com/Class1', 'http://test.com/Class2');

-- This should fail if parent doesn't exist
-- INSERT INTO owl_class_hierarchy (ontology_id, class_iri, parent_iri)
-- VALUES ('default', 'http://test.com/Class1', 'http://nonexistent.com/Class');

-- ===========================================================================
-- ROLLBACK SCRIPT (In case of issues)
-- ===========================================================================

-- To rollback this migration:
-- 1. Stop the application
-- 2. Run: cp ontology.db.backup.YYYYMMDD_HHMMSS ontology.db
-- 3. Restart the application
-- 4. Investigate the issue before re-attempting migration

-- ===========================================================================
-- MIGRATION COMPLETE
-- ===========================================================================

-- Log migration completion
INSERT INTO schema_migrations (version, applied_at)
VALUES ('001', CURRENT_TIMESTAMP);

SELECT 'Migration 001 completed successfully' AS status;
