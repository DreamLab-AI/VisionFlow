-- VisionFlow Ontology Database Schema Migration
-- Fixes foreign key constraints in owl_class_hierarchy table
-- Date: 2025-10-31

-- Begin transaction for atomic changes
BEGIN TRANSACTION;

-- Step 1: Ensure default ontology exists (using actual schema)
INSERT OR IGNORE INTO ontologies (
    ontology_id,
    source_path,
    source_type,
    content_hash,
    title,
    description
)
VALUES (
    'default',
    'default',
    'embedded',
    'default-ontology',
    'Default Ontology',
    'Default ontology for VisionFlow incremental saves'
);

-- Step 3: Backup existing hierarchy data
CREATE TEMPORARY TABLE owl_class_hierarchy_backup AS
SELECT * FROM owl_class_hierarchy;

-- Step 4: Drop the incorrect owl_class_hierarchy table
DROP TABLE IF EXISTS owl_class_hierarchy;

-- Step 5: Recreate owl_class_hierarchy with correct foreign keys
-- CRITICAL FIX: Foreign keys now reference class_iri column (not iri)
CREATE TABLE owl_class_hierarchy (
    class_iri TEXT NOT NULL,
    parent_iri TEXT NOT NULL,
    PRIMARY KEY (class_iri, parent_iri),
    FOREIGN KEY (class_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE,
    FOREIGN KEY (parent_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE
);

-- Step 6: Restore data from backup
INSERT INTO owl_class_hierarchy (class_iri, parent_iri)
SELECT class_iri, parent_iri FROM owl_class_hierarchy_backup;

-- Step 7: Drop temporary backup table
DROP TABLE owl_class_hierarchy_backup;

-- Step 8: Verify foreign key constraints are working
-- This will fail if there are orphaned references
PRAGMA foreign_keys = ON;
PRAGMA foreign_key_check;

-- Step 9: Ensure all necessary indexes exist
CREATE INDEX IF NOT EXISTS idx_owl_classes_iri ON owl_classes(class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_classes_parent ON owl_classes(parent_class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);
CREATE INDEX IF NOT EXISTS idx_owl_classes_sha1 ON owl_classes(file_sha1);
CREATE INDEX IF NOT EXISTS idx_owl_properties_iri ON owl_properties(property_iri);
CREATE INDEX IF NOT EXISTS idx_owl_properties_type ON owl_properties(property_type);

-- Commit transaction
COMMIT;

-- Display schema verification
.echo on
SELECT '=== Schema Migration Complete ===' AS status;
SELECT 'owl_classes count: ' || COUNT(*) FROM owl_classes;
SELECT 'owl_class_hierarchy count: ' || COUNT(*) FROM owl_class_hierarchy;
SELECT 'owl_properties count: ' || COUNT(*) FROM owl_properties;
.schema owl_class_hierarchy
