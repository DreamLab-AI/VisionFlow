-- VisionFlow Rich Ontology Metadata Migration
-- Extends owl_classes, owl_properties, and owl_axioms with comprehensive metadata
-- Date: 2025-11-22
-- Schema Version: 2

-- CRITICAL: Must disable foreign keys before making schema changes
PRAGMA foreign_keys = OFF;

-- Begin transaction for atomic changes
BEGIN TRANSACTION;

-- =================================================================
-- STEP 1: Update schema version
-- =================================================================
UPDATE schema_version SET version = 2, updated_at = CURRENT_TIMESTAMP WHERE id = 1;

-- =================================================================
-- STEP 2: Backup existing tables
-- =================================================================
DROP TABLE IF EXISTS owl_classes_backup;
DROP TABLE IF EXISTS owl_properties_backup;
DROP TABLE IF EXISTS owl_axioms_backup;

CREATE TABLE owl_classes_backup AS SELECT * FROM owl_classes;
CREATE TABLE owl_properties_backup AS SELECT * FROM owl_properties;

-- =================================================================
-- STEP 3: Drop and recreate owl_classes with rich metadata
-- =================================================================
DROP TABLE IF EXISTS owl_classes;

CREATE TABLE owl_classes (
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,

    -- Core identification (from parser)
    term_id TEXT,                    -- Ontology term ID (e.g., BC-0478)
    preferred_term TEXT,             -- Canonical term name

    -- Basic metadata (existing)
    label TEXT,
    comment TEXT,
    parent_class_iri TEXT,
    is_deprecated INTEGER DEFAULT 0 CHECK (is_deprecated IN (0, 1)),

    -- Classification metadata
    source_domain TEXT,              -- Domain (blockchain/ai/metaverse/rb/dt)
    version TEXT,                    -- Version number
    type TEXT,                       -- Entity type

    -- Quality metrics
    status TEXT,                     -- Lifecycle status (draft/review/approved/deprecated)
    maturity TEXT,                   -- Maturity level (experimental/beta/stable)
    quality_score REAL,              -- Quality metric (0.0-1.0)
    authority_score REAL,            -- Authority metric (0.0-1.0)
    public_access INTEGER DEFAULT 0 CHECK (public_access IN (0, 1)),
    content_status TEXT,             -- Workflow state

    -- OWL2 properties
    owl_physicality TEXT,            -- Physicality classification (physical/virtual/abstract)
    owl_role TEXT,                   -- Role classification (agent/patient/instrument)

    -- Domain relationships
    belongs_to_domain TEXT,          -- Primary domain link
    bridges_to_domain TEXT,          -- Cross-domain bridge

    -- Source tracking
    source_file TEXT,                -- Original file path
    file_sha1 TEXT,                  -- File content hash
    markdown_content TEXT,           -- Full markdown content
    last_synced TIMESTAMP,           -- Last GitHub sync

    -- Additional metadata (JSON for extensibility)
    additional_metadata TEXT,        -- JSON blob for unknown fields

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, class_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,

    -- Constraints for scores
    CHECK (quality_score IS NULL OR (quality_score >= 0.0 AND quality_score <= 1.0)),
    CHECK (authority_score IS NULL OR (authority_score >= 0.0 AND authority_score <= 1.0))
);

-- =================================================================
-- STEP 4: Create owl_relationships table for semantic relationships
-- =================================================================
DROP TABLE IF EXISTS owl_relationships;

CREATE TABLE owl_relationships (
    ontology_id TEXT NOT NULL,
    source_class_iri TEXT NOT NULL,
    relationship_type TEXT NOT NULL,  -- has-part, uses, enables, requires, subclass-of, etc.
    target_class_iri TEXT NOT NULL,

    -- Relationship metadata
    confidence REAL DEFAULT 1.0,      -- Confidence score (0.0-1.0)
    is_inferred INTEGER DEFAULT 0 CHECK (is_inferred IN (0, 1)),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, source_class_iri, relationship_type, target_class_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    FOREIGN KEY (source_class_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE,
    FOREIGN KEY (target_class_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE,

    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- =================================================================
-- STEP 5: Update owl_properties with additional metadata
-- =================================================================
DROP TABLE IF EXISTS owl_properties;

CREATE TABLE owl_properties (
    ontology_id TEXT NOT NULL,
    property_iri TEXT NOT NULL,
    property_type TEXT NOT NULL CHECK (property_type IN ('ObjectProperty', 'DataProperty', 'AnnotationProperty')),

    -- Basic metadata
    label TEXT,
    comment TEXT,
    domain_class_iri TEXT,
    range_class_iri TEXT,

    -- Property characteristics
    is_functional INTEGER DEFAULT 0 CHECK (is_functional IN (0, 1)),
    is_inverse_functional INTEGER DEFAULT 0 CHECK (is_inverse_functional IN (0, 1)),
    is_symmetric INTEGER DEFAULT 0 CHECK (is_symmetric IN (0, 1)),
    is_transitive INTEGER DEFAULT 0 CHECK (is_transitive IN (0, 1)),
    inverse_property_iri TEXT,

    -- Quality metrics (new)
    quality_score REAL,
    authority_score REAL,

    -- Source tracking
    source_file TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, property_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,

    CHECK (quality_score IS NULL OR (quality_score >= 0.0 AND quality_score <= 1.0)),
    CHECK (authority_score IS NULL OR (authority_score >= 0.0 AND authority_score <= 1.0))
);

-- =================================================================
-- STEP 6: Create owl_axioms table (if not exists)
-- =================================================================
DROP TABLE IF EXISTS owl_axioms;

CREATE TABLE owl_axioms (
    ontology_id TEXT NOT NULL,
    axiom_id TEXT NOT NULL,
    axiom_type TEXT NOT NULL,        -- SubClassOf, DisjointWith, etc.

    -- Subject/Object
    subject_iri TEXT NOT NULL,
    object_iri TEXT,

    -- Axiom metadata
    is_inferred INTEGER DEFAULT 0 CHECK (is_inferred IN (0, 1)),
    confidence REAL DEFAULT 1.0,     -- 1.0 for asserted, 0.3-0.9 for inferred
    reasoning_rule TEXT,             -- Which rule produced this inference

    -- OWL axiom content (serialized)
    axiom_content TEXT,              -- Clojure/functional syntax

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, axiom_id),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,

    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- =================================================================
-- STEP 7: Create owl_cross_references table for WikiLinks
-- =================================================================
DROP TABLE IF EXISTS owl_cross_references;

CREATE TABLE owl_cross_references (
    ontology_id TEXT NOT NULL,
    source_class_iri TEXT NOT NULL,
    target_reference TEXT NOT NULL,  -- WikiLink target
    reference_type TEXT DEFAULT 'wiki',  -- wiki, external, doi, etc.

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, source_class_iri, target_reference),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    FOREIGN KEY (source_class_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE
);

-- =================================================================
-- STEP 8: Restore data from backup with column mapping
-- =================================================================

-- Restore owl_classes (map old columns to new schema)
INSERT INTO owl_classes (
    ontology_id, class_iri, label, comment, parent_class_iri,
    is_deprecated, created_at
)
SELECT
    ontology_id, class_iri, label, comment, parent_class_iri,
    is_deprecated, created_at
FROM owl_classes_backup;

-- Restore owl_properties
INSERT INTO owl_properties (
    ontology_id, property_iri, property_type, label, comment,
    domain_class_iri, range_class_iri,
    is_functional, is_inverse_functional, is_symmetric, is_transitive,
    inverse_property_iri, created_at
)
SELECT
    ontology_id, property_iri, property_type, label, comment,
    domain_class_iri, range_class_iri,
    is_functional, is_inverse_functional, is_symmetric, is_transitive,
    inverse_property_iri, created_at
FROM owl_properties_backup;

-- =================================================================
-- STEP 9: Create indexes for common query patterns
-- =================================================================

-- owl_classes indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_owl_classes_iri_unique ON owl_classes(class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_classes_ontology ON owl_classes(ontology_id);
CREATE INDEX IF NOT EXISTS idx_owl_classes_parent ON owl_classes(parent_class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);
CREATE INDEX IF NOT EXISTS idx_owl_classes_term_id ON owl_classes(term_id);
CREATE INDEX IF NOT EXISTS idx_owl_classes_source_domain ON owl_classes(source_domain);
CREATE INDEX IF NOT EXISTS idx_owl_classes_status ON owl_classes(status);
CREATE INDEX IF NOT EXISTS idx_owl_classes_quality ON owl_classes(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_owl_classes_physicality ON owl_classes(owl_physicality);
CREATE INDEX IF NOT EXISTS idx_owl_classes_sha1 ON owl_classes(file_sha1);

-- owl_relationships indexes
CREATE INDEX IF NOT EXISTS idx_owl_rel_source ON owl_relationships(source_class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_rel_target ON owl_relationships(target_class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_rel_type ON owl_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_owl_rel_inferred ON owl_relationships(is_inferred);

-- owl_properties indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_owl_properties_iri ON owl_properties(property_iri);
CREATE INDEX IF NOT EXISTS idx_owl_properties_type ON owl_properties(property_type);
CREATE INDEX IF NOT EXISTS idx_owl_properties_domain ON owl_properties(domain_class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_properties_range ON owl_properties(range_class_iri);

-- owl_axioms indexes
CREATE INDEX IF NOT EXISTS idx_owl_axioms_type ON owl_axioms(axiom_type);
CREATE INDEX IF NOT EXISTS idx_owl_axioms_subject ON owl_axioms(subject_iri);
CREATE INDEX IF NOT EXISTS idx_owl_axioms_object ON owl_axioms(object_iri);
CREATE INDEX IF NOT EXISTS idx_owl_axioms_inferred ON owl_axioms(is_inferred);

-- owl_cross_references indexes
CREATE INDEX IF NOT EXISTS idx_owl_xref_source ON owl_cross_references(source_class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_xref_target ON owl_cross_references(target_reference);

-- =================================================================
-- STEP 10: Create views for common queries
-- =================================================================

-- View for classes with quality metrics
DROP VIEW IF EXISTS owl_classes_with_quality;
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

-- View for cross-domain bridges
DROP VIEW IF EXISTS owl_cross_domain_bridges;
CREATE VIEW owl_cross_domain_bridges AS
SELECT
    class_iri,
    label,
    belongs_to_domain,
    bridges_to_domain,
    source_domain
FROM owl_classes
WHERE bridges_to_domain IS NOT NULL;

-- View for relationship graph
DROP VIEW IF EXISTS owl_relationship_graph;
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

-- =================================================================
-- STEP 11: Drop backup tables
-- =================================================================
DROP TABLE IF EXISTS owl_classes_backup;
DROP TABLE IF EXISTS owl_properties_backup;
DROP TABLE IF EXISTS owl_axioms_backup;

-- Commit transaction
COMMIT;

-- Re-enable foreign keys
PRAGMA foreign_keys = ON;

-- =================================================================
-- STEP 12: Verify migration
-- =================================================================
.echo on
SELECT '=== Schema Migration V2 Complete ===' AS status;
SELECT 'Schema version: ' || version FROM schema_version;
SELECT 'owl_classes count: ' || COUNT(*) FROM owl_classes;
SELECT 'owl_properties count: ' || COUNT(*) FROM owl_properties;
SELECT 'owl_relationships count: ' || COUNT(*) FROM owl_relationships;
SELECT 'owl_axioms count: ' || COUNT(*) FROM owl_axioms;
SELECT '=== Checking Foreign Key Constraints ===' AS status;
PRAGMA foreign_key_check;
