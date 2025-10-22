-- =================================================================
-- VisionFlow Ontology Database v2 (ontology.db)
-- =================================================================
-- Purpose: OWL ontology structures from GitHub with reasoning and validation
-- Version: 2.0.0
-- Created: 2025-10-22
-- Enhanced with: inference results, validation, metrics, GitHub sync
-- =================================================================

-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- =================================================================
-- SCHEMA VERSION TRACKING
-- =================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR IGNORE INTO schema_version (id, version, description)
VALUES (1, 2, 'Three-database system - Enhanced ontology database with reasoning');

-- =================================================================
-- ONTOLOGY METADATA TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS ontologies (
    ontology_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('file', 'url', 'github', 'embedded')),

    -- OWL Ontology IRIs
    base_iri TEXT,
    version_iri TEXT,

    -- Metadata
    title TEXT,
    description TEXT,
    author TEXT,
    version TEXT,
    license TEXT,

    -- Content tracking
    content_hash TEXT NOT NULL,

    -- Statistics
    axiom_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,
    property_count INTEGER DEFAULT 0,
    individual_count INTEGER DEFAULT 0,

    -- Status
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),

    -- Timestamps
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_validated_at DATETIME,
    last_reasoned_at DATETIME,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ontologies_source ON ontologies(source_path);
CREATE INDEX IF NOT EXISTS idx_ontologies_hash ON ontologies(content_hash);
CREATE INDEX IF NOT EXISTS idx_ontologies_type ON ontologies(source_type);
CREATE INDEX IF NOT EXISTS idx_ontologies_active ON ontologies(is_active);

-- =================================================================
-- OWL CLASSES TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS owl_classes (
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,

    -- Labels and descriptions
    label TEXT,
    comment TEXT,
    description TEXT,

    -- Properties as JSON
    properties TEXT NOT NULL DEFAULT '{}',

    -- Source tracking
    source_file TEXT,
    line_number INTEGER,

    -- Status
    is_deprecated INTEGER DEFAULT 0 CHECK (is_deprecated IN (0, 1)),
    deprecation_reason TEXT,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, class_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_owl_classes_iri ON owl_classes(class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);
CREATE INDEX IF NOT EXISTS idx_owl_classes_source ON owl_classes(source_file);
CREATE INDEX IF NOT EXISTS idx_owl_classes_deprecated ON owl_classes(is_deprecated);

-- =================================================================
-- OWL CLASS HIERARCHY TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS owl_class_hierarchy (
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,
    parent_iri TEXT NOT NULL,

    -- Hierarchy metadata
    depth_level INTEGER DEFAULT 0,
    path_to_root TEXT, -- JSON array of IRIs from this class to root

    -- Whether this relationship was inferred
    is_inferred INTEGER NOT NULL DEFAULT 0 CHECK (is_inferred IN (0, 1)),

    PRIMARY KEY (ontology_id, class_iri, parent_iri),
    FOREIGN KEY (ontology_id, class_iri) REFERENCES owl_classes(ontology_id, class_iri) ON DELETE CASCADE,
    FOREIGN KEY (ontology_id, parent_iri) REFERENCES owl_classes(ontology_id, class_iri) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_hierarchy_class ON owl_class_hierarchy(class_iri);
CREATE INDEX IF NOT EXISTS idx_hierarchy_parent ON owl_class_hierarchy(parent_iri);
CREATE INDEX IF NOT EXISTS idx_hierarchy_depth ON owl_class_hierarchy(depth_level);
CREATE INDEX IF NOT EXISTS idx_hierarchy_inferred ON owl_class_hierarchy(is_inferred);

-- =================================================================
-- OWL PROPERTIES TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS owl_properties (
    ontology_id TEXT NOT NULL,
    property_iri TEXT NOT NULL,

    -- Property metadata
    label TEXT,
    comment TEXT,
    property_type TEXT NOT NULL CHECK (property_type IN ('ObjectProperty', 'DataProperty', 'AnnotationProperty')),

    -- Domain and range as JSON arrays of IRIs
    domain TEXT NOT NULL DEFAULT '[]',
    range TEXT NOT NULL DEFAULT '[]',

    -- Property characteristics
    is_functional INTEGER DEFAULT 0 CHECK (is_functional IN (0, 1)),
    is_inverse_functional INTEGER DEFAULT 0 CHECK (is_inverse_functional IN (0, 1)),
    is_symmetric INTEGER DEFAULT 0 CHECK (is_symmetric IN (0, 1)),
    is_asymmetric INTEGER DEFAULT 0 CHECK (is_asymmetric IN (0, 1)),
    is_transitive INTEGER DEFAULT 0 CHECK (is_transitive IN (0, 1)),
    is_reflexive INTEGER DEFAULT 0 CHECK (is_reflexive IN (0, 1)),
    is_irreflexive INTEGER DEFAULT 0 CHECK (is_irreflexive IN (0, 1)),

    -- Inverse property
    inverse_property_iri TEXT,

    -- Status
    is_deprecated INTEGER DEFAULT 0 CHECK (is_deprecated IN (0, 1)),

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, property_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_owl_properties_iri ON owl_properties(property_iri);
CREATE INDEX IF NOT EXISTS idx_owl_properties_type ON owl_properties(property_type);
CREATE INDEX IF NOT EXISTS idx_owl_properties_label ON owl_properties(label);
CREATE INDEX IF NOT EXISTS idx_owl_properties_deprecated ON owl_properties(is_deprecated);

-- =================================================================
-- OWL AXIOMS TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS owl_axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,

    -- Axiom type
    axiom_type TEXT NOT NULL CHECK (axiom_type IN (
        'SubClassOf',
        'EquivalentClass',
        'DisjointWith',
        'DisjointUnion',
        'ObjectPropertyAssertion',
        'DataPropertyAssertion',
        'ClassAssertion',
        'SameIndividual',
        'DifferentIndividuals',
        'NegativeObjectPropertyAssertion',
        'NegativeDataPropertyAssertion'
    )),

    -- Axiom components
    subject TEXT NOT NULL,
    predicate TEXT, -- For property assertions
    object TEXT NOT NULL,

    -- Additional axiom data as JSON
    axiom_data TEXT DEFAULT '{}',

    -- Annotations as JSON
    annotations TEXT NOT NULL DEFAULT '{}',

    -- Inference tracking
    is_inferred INTEGER NOT NULL DEFAULT 0 CHECK (is_inferred IN (0, 1)),
    inferred_from TEXT, -- JSON array of axiom IDs that led to this inference
    inference_rule TEXT, -- Rule that generated this inference
    confidence REAL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_axioms_ontology ON owl_axioms(ontology_id);
CREATE INDEX IF NOT EXISTS idx_axioms_type ON owl_axioms(axiom_type);
CREATE INDEX IF NOT EXISTS idx_axioms_subject ON owl_axioms(subject);
CREATE INDEX IF NOT EXISTS idx_axioms_object ON owl_axioms(object);
CREATE INDEX IF NOT EXISTS idx_axioms_inferred ON owl_axioms(is_inferred);
CREATE INDEX IF NOT EXISTS idx_axioms_predicate ON owl_axioms(predicate);

-- =================================================================
-- OWL DISJOINT CLASSES TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS owl_disjoint_classes (
    ontology_id TEXT NOT NULL,
    class_iri_1 TEXT NOT NULL,
    class_iri_2 TEXT NOT NULL,

    -- Whether this was inferred
    is_inferred INTEGER NOT NULL DEFAULT 0 CHECK (is_inferred IN (0, 1)),

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (ontology_id, class_iri_1, class_iri_2),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    CHECK (class_iri_1 < class_iri_2) -- Ensure ordering to prevent duplicates
);

CREATE INDEX IF NOT EXISTS idx_disjoint_class1 ON owl_disjoint_classes(class_iri_1);
CREATE INDEX IF NOT EXISTS idx_disjoint_class2 ON owl_disjoint_classes(class_iri_2);

-- =================================================================
-- ONTOLOGY GRAPH NODES TABLE (For Visualization)
-- =================================================================

CREATE TABLE IF NOT EXISTS ontology_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,
    iri TEXT NOT NULL,
    label TEXT NOT NULL,
    node_type TEXT NOT NULL CHECK (node_type IN ('class', 'property', 'individual', 'datatype')),

    -- Position data (3D coordinates)
    x REAL NOT NULL DEFAULT 0.0,
    y REAL NOT NULL DEFAULT 0.0,
    z REAL NOT NULL DEFAULT 0.0,

    -- Visual properties
    color TEXT,
    size REAL DEFAULT 15.0,
    shape TEXT DEFAULT 'sphere',

    -- Metadata as JSON
    metadata TEXT NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (ontology_id, iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ontology_nodes_iri ON ontology_nodes(iri);
CREATE INDEX IF NOT EXISTS idx_ontology_nodes_type ON ontology_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_ontology_nodes_ontology ON ontology_nodes(ontology_id);

-- =================================================================
-- ONTOLOGY GRAPH EDGES TABLE (For Visualization)
-- =================================================================

CREATE TABLE IF NOT EXISTS ontology_edges (
    id TEXT PRIMARY KEY,
    ontology_id TEXT NOT NULL,
    source INTEGER NOT NULL,
    target INTEGER NOT NULL,
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'subClassOf',
        'equivalentClass',
        'disjointWith',
        'objectProperty',
        'dataProperty',
        'annotation',
        'instance'
    )),
    weight REAL NOT NULL DEFAULT 1.0,

    -- Edge metadata as JSON
    metadata TEXT DEFAULT '{}',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    FOREIGN KEY (source) REFERENCES ontology_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES ontology_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ontology_edges_source ON ontology_edges(source);
CREATE INDEX IF NOT EXISTS idx_ontology_edges_target ON ontology_edges(target);
CREATE INDEX IF NOT EXISTS idx_ontology_edges_type ON ontology_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_ontology_edges_ontology ON ontology_edges(ontology_id);

-- =================================================================
-- INFERENCE RESULTS TABLE (New in v2)
-- =================================================================

CREATE TABLE IF NOT EXISTS inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,

    -- Inference session metadata
    session_id TEXT UNIQUE NOT NULL,
    reasoner_name TEXT NOT NULL DEFAULT 'whelk-rs',
    reasoner_version TEXT NOT NULL,

    -- Performance metrics
    inference_time_ms INTEGER NOT NULL,
    memory_used_mb REAL,

    -- Statistics
    inferred_axiom_count INTEGER NOT NULL,
    total_axiom_count INTEGER NOT NULL,
    new_subsumptions INTEGER DEFAULT 0,

    -- Reasoning profile
    reasoning_profile TEXT DEFAULT 'EL' CHECK (reasoning_profile IN ('EL', 'RL', 'QL', 'DL', 'FULL')),

    -- Complete inference data as JSON
    result_data TEXT NOT NULL,

    -- Consistency check result
    is_consistent INTEGER NOT NULL DEFAULT 1 CHECK (is_consistent IN (0, 1)),
    inconsistency_explanation TEXT,

    -- Status
    status TEXT DEFAULT 'complete' CHECK (status IN ('running', 'complete', 'failed', 'timeout')),
    error_message TEXT,

    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_inference_ontology ON inference_results(ontology_id);
CREATE INDEX IF NOT EXISTS idx_inference_timestamp ON inference_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_inference_session ON inference_results(session_id);
CREATE INDEX IF NOT EXISTS idx_inference_status ON inference_results(status);

-- =================================================================
-- VALIDATION REPORTS TABLE (New in v2)
-- =================================================================

CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,

    -- Validation metadata
    validation_type TEXT NOT NULL CHECK (validation_type IN (
        'structural',
        'semantic',
        'consistency',
        'completeness',
        'syntax'
    )),

    -- Results
    is_valid INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
    errors TEXT NOT NULL DEFAULT '[]', -- JSON array of error objects
    warnings TEXT NOT NULL DEFAULT '[]', -- JSON array of warning objects
    info TEXT NOT NULL DEFAULT '[]', -- JSON array of info messages

    -- Statistics
    error_count INTEGER NOT NULL DEFAULT 0,
    warning_count INTEGER NOT NULL DEFAULT 0,
    info_count INTEGER NOT NULL DEFAULT 0,

    -- Validation rules applied
    rules_applied TEXT DEFAULT '[]', -- JSON array of rule names

    -- Performance
    validation_time_ms INTEGER,

    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_validation_ontology ON validation_reports(ontology_id);
CREATE INDEX IF NOT EXISTS idx_validation_type ON validation_reports(validation_type);
CREATE INDEX IF NOT EXISTS idx_validation_timestamp ON validation_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_validation_valid ON validation_reports(is_valid);

-- =================================================================
-- ONTOLOGY METRICS TABLE (New in v2)
-- =================================================================

CREATE TABLE IF NOT EXISTS ontology_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,

    -- Structural metrics
    class_count INTEGER NOT NULL,
    property_count INTEGER NOT NULL,
    object_property_count INTEGER NOT NULL DEFAULT 0,
    data_property_count INTEGER NOT NULL DEFAULT 0,
    axiom_count INTEGER NOT NULL,
    individual_count INTEGER NOT NULL DEFAULT 0,

    -- Complexity metrics
    max_depth INTEGER NOT NULL,
    average_depth REAL NOT NULL,
    average_branching_factor REAL NOT NULL,
    max_branching_factor INTEGER DEFAULT 0,

    -- Richness metrics
    relationship_richness REAL NOT NULL, -- ratio of properties to classes
    attribute_richness REAL NOT NULL, -- average properties per class
    inheritance_richness REAL NOT NULL, -- average parents per class

    -- Additional metrics
    axiom_class_ratio REAL, -- axioms per class
    property_usage REAL, -- percentage of properties actually used

    -- Graph metrics
    connected_components INTEGER DEFAULT 1,
    average_path_length REAL,
    clustering_coefficient REAL,

    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_ontology ON ontology_metrics(ontology_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON ontology_metrics(timestamp);

-- =================================================================
-- GITHUB SYNC METADATA TABLE (New in v2)
-- =================================================================

CREATE TABLE IF NOT EXISTS github_sync_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT,

    -- Source information
    repository TEXT NOT NULL,
    owner TEXT NOT NULL,
    branch TEXT NOT NULL DEFAULT 'main',
    file_path TEXT NOT NULL,

    -- GitHub API metadata
    last_commit_sha TEXT,
    last_tree_sha TEXT,
    etag TEXT, -- For efficient API caching

    -- Sync status
    last_sync_timestamp DATETIME,
    next_sync_scheduled DATETIME,
    sync_status TEXT NOT NULL DEFAULT 'pending' CHECK (sync_status IN (
        'pending',
        'syncing',
        'success',
        'failed',
        'conflict'
    )),

    -- Sync strategy
    sync_mode TEXT DEFAULT 'pull' CHECK (sync_mode IN ('pull', 'push', 'bidirectional')),
    auto_sync_enabled INTEGER DEFAULT 1 CHECK (auto_sync_enabled IN (0, 1)),
    sync_interval_minutes INTEGER DEFAULT 60,

    -- Statistics
    files_processed INTEGER DEFAULT 0,
    classes_imported INTEGER DEFAULT 0,
    properties_imported INTEGER DEFAULT 0,
    axioms_imported INTEGER DEFAULT 0,

    -- Change tracking
    files_added INTEGER DEFAULT 0,
    files_modified INTEGER DEFAULT 0,
    files_deleted INTEGER DEFAULT 0,

    -- Error tracking
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (repository, owner, branch, file_path),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_github_sync_status ON github_sync_metadata(sync_status);
CREATE INDEX IF NOT EXISTS idx_github_sync_updated ON github_sync_metadata(updated_at);
CREATE INDEX IF NOT EXISTS idx_github_sync_repo ON github_sync_metadata(repository, owner);
CREATE INDEX IF NOT EXISTS idx_github_sync_ontology ON github_sync_metadata(ontology_id);
CREATE INDEX IF NOT EXISTS idx_github_sync_auto ON github_sync_metadata(auto_sync_enabled);

-- =================================================================
-- NAMESPACES TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS namespaces (
    prefix TEXT PRIMARY KEY,
    namespace_iri TEXT NOT NULL UNIQUE,
    is_default INTEGER DEFAULT 0 CHECK (is_default IN (0, 1)),
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_namespaces_iri ON namespaces(namespace_iri);

-- =================================================================
-- INITIALIZATION DATA
-- =================================================================

BEGIN TRANSACTION;

-- Insert OWL Thing (top of hierarchy)
INSERT OR IGNORE INTO ontologies (ontology_id, source_path, source_type, base_iri, title, content_hash)
VALUES ('owl-base', 'embedded', 'embedded', 'http://www.w3.org/2002/07/owl#', 'OWL Base Ontology', 'embedded-base');

INSERT OR IGNORE INTO owl_classes (ontology_id, class_iri, label, comment, properties)
VALUES
    ('owl-base', 'http://www.w3.org/2002/07/owl#Thing',
     'Thing',
     'The class of OWL individuals',
     '{"type": "owl:Class", "isTopLevel": true}'),
    ('owl-base', 'http://www.w3.org/2002/07/owl#Nothing',
     'Nothing',
     'The empty class',
     '{"type": "owl:Class", "isBottomLevel": true}');

-- Insert OWL built-in properties
INSERT OR IGNORE INTO owl_properties (ontology_id, property_iri, label, property_type, domain, range)
VALUES
    ('owl-base', 'http://www.w3.org/2000/01/rdf-schema#subClassOf',
     'subClassOf',
     'ObjectProperty',
     '["http://www.w3.org/2002/07/owl#Class"]',
     '["http://www.w3.org/2002/07/owl#Class"]'),

    ('owl-base', 'http://www.w3.org/2002/07/owl#equivalentClass',
     'equivalentClass',
     'ObjectProperty',
     '["http://www.w3.org/2002/07/owl#Class"]',
     '["http://www.w3.org/2002/07/owl#Class"]'),

    ('owl-base', 'http://www.w3.org/2002/07/owl#disjointWith',
     'disjointWith',
     'ObjectProperty',
     '["http://www.w3.org/2002/07/owl#Class"]',
     '["http://www.w3.org/2002/07/owl#Class"]');

-- Insert standard namespaces
INSERT OR IGNORE INTO namespaces (prefix, namespace_iri, is_default, description) VALUES
    ('owl', 'http://www.w3.org/2002/07/owl#', 1, 'OWL namespace'),
    ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 1, 'RDF namespace'),
    ('rdfs', 'http://www.w3.org/2000/01/rdf-schema#', 1, 'RDF Schema namespace'),
    ('xsd', 'http://www.w3.org/2001/XMLSchema#', 1, 'XML Schema namespace'),
    ('dc', 'http://purl.org/dc/elements/1.1/', 0, 'Dublin Core namespace'),
    ('foaf', 'http://xmlns.com/foaf/0.1/', 0, 'Friend of a Friend namespace'),
    ('skos', 'http://www.w3.org/2004/02/skos/core#', 0, 'SKOS namespace');

-- Initialize metrics for base ontology
INSERT INTO ontology_metrics (
    ontology_id,
    class_count, property_count, object_property_count, data_property_count,
    axiom_count, individual_count,
    max_depth, average_depth, average_branching_factor,
    relationship_richness, attribute_richness, inheritance_richness
) VALUES (
    'owl-base',
    2, 3, 3, 0,
    0, 0,
    0, 0.0, 0.0,
    1.5, 0.0, 0.0
);

COMMIT;

-- =================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =================================================================

CREATE TRIGGER IF NOT EXISTS update_ontologies_timestamp
AFTER UPDATE ON ontologies
FOR EACH ROW
BEGIN
    UPDATE ontologies SET updated_at = CURRENT_TIMESTAMP WHERE ontology_id = NEW.ontology_id;
END;

CREATE TRIGGER IF NOT EXISTS update_owl_classes_timestamp
AFTER UPDATE ON owl_classes
FOR EACH ROW
BEGIN
    UPDATE owl_classes SET updated_at = CURRENT_TIMESTAMP
    WHERE ontology_id = NEW.ontology_id AND class_iri = NEW.class_iri;
END;

CREATE TRIGGER IF NOT EXISTS update_owl_properties_timestamp
AFTER UPDATE ON owl_properties
FOR EACH ROW
BEGIN
    UPDATE owl_properties SET updated_at = CURRENT_TIMESTAMP
    WHERE ontology_id = NEW.ontology_id AND property_iri = NEW.property_iri;
END;

CREATE TRIGGER IF NOT EXISTS update_ontology_nodes_timestamp
AFTER UPDATE ON ontology_nodes
FOR EACH ROW
BEGIN
    UPDATE ontology_nodes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_github_sync_timestamp
AFTER UPDATE ON github_sync_metadata
FOR EACH ROW
BEGIN
    UPDATE github_sync_metadata SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- =================================================================
-- VIEWS FOR CONVENIENT QUERYING
-- =================================================================

-- View for ontology summary
CREATE VIEW IF NOT EXISTS v_ontology_summary AS
SELECT
    o.ontology_id,
    o.title,
    o.base_iri,
    o.class_count,
    o.property_count,
    o.axiom_count,
    o.last_validated_at,
    o.last_reasoned_at,
    (SELECT COUNT(*) FROM inference_results ir WHERE ir.ontology_id = o.ontology_id) as inference_count,
    (SELECT COUNT(*) FROM validation_reports vr WHERE vr.ontology_id = o.ontology_id) as validation_count
FROM ontologies o
WHERE o.is_active = 1;

-- View for class hierarchy depth
CREATE VIEW IF NOT EXISTS v_class_depth AS
SELECT
    ontology_id,
    class_iri,
    MAX(depth_level) as max_depth
FROM owl_class_hierarchy
GROUP BY ontology_id, class_iri;

-- View for latest validation status
CREATE VIEW IF NOT EXISTS v_latest_validation AS
SELECT
    ontology_id,
    validation_type,
    is_valid,
    error_count,
    warning_count,
    timestamp
FROM validation_reports vr1
WHERE timestamp = (
    SELECT MAX(timestamp)
    FROM validation_reports vr2
    WHERE vr2.ontology_id = vr1.ontology_id
    AND vr2.validation_type = vr1.validation_type
);

-- View for GitHub sync status
CREATE VIEW IF NOT EXISTS v_github_sync_status AS
SELECT
    repository,
    owner,
    branch,
    file_path,
    sync_status,
    last_sync_timestamp,
    next_sync_scheduled,
    auto_sync_enabled,
    error_message
FROM github_sync_metadata
ORDER BY last_sync_timestamp DESC;

-- =================================================================
-- VACUUM AND OPTIMIZE
-- =================================================================

PRAGMA optimize;
