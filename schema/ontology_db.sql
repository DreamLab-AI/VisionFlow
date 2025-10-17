-- ================================================================
-- Ontology Database Schema for Knowledge Graph System
-- ================================================================
-- Replaces YAML/TOML file-based storage with SQLite for:
-- - Settings and configuration
-- - Ontology framework metadata
-- - Markdown file metadata with ontology blocks
-- - Graph generation optimization
-- ================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;

-- ================================================================
-- CORE SETTINGS TABLES
-- ================================================================

-- Application settings hierarchy (replaces settings.yaml)
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    parent_key TEXT,
    value_type TEXT NOT NULL CHECK(value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    value_text TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER CHECK(value_boolean IN (0, 1)),
    value_json TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_key) REFERENCES settings(key) ON DELETE CASCADE
);

CREATE INDEX idx_settings_key ON settings(key);
CREATE INDEX idx_settings_parent_key ON settings(parent_key);
CREATE INDEX idx_settings_updated_at ON settings(updated_at DESC);

-- Physics settings (extracted from nested YAML for fast access)
CREATE TABLE IF NOT EXISTS physics_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL UNIQUE DEFAULT 'default',
    -- Core physics parameters
    damping REAL NOT NULL DEFAULT 0.95,
    dt REAL NOT NULL DEFAULT 0.016,
    iterations INTEGER NOT NULL DEFAULT 100,
    max_velocity REAL NOT NULL DEFAULT 1.0,
    max_force REAL NOT NULL DEFAULT 100.0,
    repel_k REAL NOT NULL DEFAULT 50.0,
    spring_k REAL NOT NULL DEFAULT 0.005,
    mass_scale REAL NOT NULL DEFAULT 1.0,
    boundary_damping REAL NOT NULL DEFAULT 0.95,
    temperature REAL NOT NULL DEFAULT 0.01,
    gravity REAL NOT NULL DEFAULT 0.0001,
    bounds_size REAL NOT NULL DEFAULT 500.0,
    enable_bounds INTEGER NOT NULL DEFAULT 1 CHECK(enable_bounds IN (0, 1)),
    -- GPU kernel parameters
    rest_length REAL NOT NULL DEFAULT 50.0,
    repulsion_cutoff REAL NOT NULL DEFAULT 50.0,
    repulsion_softening_epsilon REAL NOT NULL DEFAULT 0.0001,
    center_gravity_k REAL NOT NULL DEFAULT 0.0,
    grid_cell_size REAL NOT NULL DEFAULT 50.0,
    warmup_iterations INTEGER NOT NULL DEFAULT 100,
    cooling_rate REAL NOT NULL DEFAULT 0.001,
    -- Constraint parameters
    constraint_ramp_frames INTEGER NOT NULL DEFAULT 60,
    constraint_max_force_per_node REAL NOT NULL DEFAULT 50.0,
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_physics_profile_name ON physics_settings(profile_name);

-- ================================================================
-- ONTOLOGY FRAMEWORK TABLES
-- ================================================================

-- Ontology definitions (loaded OWL files)
CREATE TABLE IF NOT EXISTS ontologies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('file', 'url', 'embedded')),
    base_iri TEXT NOT NULL,
    version_iri TEXT,
    title TEXT,
    description TEXT,
    author TEXT,
    version TEXT,
    content_hash TEXT NOT NULL, -- Blake3 hash of ontology content
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_validated_at TIMESTAMP,
    axiom_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,
    property_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ontologies_ontology_id ON ontologies(ontology_id);
CREATE INDEX idx_ontologies_source_path ON ontologies(source_path);
CREATE INDEX idx_ontologies_content_hash ON ontologies(content_hash);
CREATE INDEX idx_ontologies_parsed_at ON ontologies(parsed_at DESC);

-- OWL Classes from ontologies
CREATE TABLE IF NOT EXISTS owl_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,
    label TEXT,
    comment TEXT,
    parent_class_iri TEXT,
    is_deprecated INTEGER DEFAULT 0 CHECK(is_deprecated IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    UNIQUE(ontology_id, class_iri)
);

CREATE INDEX idx_owl_classes_ontology_id ON owl_classes(ontology_id);
CREATE INDEX idx_owl_classes_class_iri ON owl_classes(class_iri);
CREATE INDEX idx_owl_classes_parent ON owl_classes(parent_class_iri);

-- OWL Properties (object and data properties)
CREATE TABLE IF NOT EXISTS owl_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,
    property_iri TEXT NOT NULL,
    property_type TEXT NOT NULL CHECK(property_type IN ('object', 'data', 'annotation')),
    label TEXT,
    comment TEXT,
    domain_iri TEXT,
    range_iri TEXT,
    inverse_of_iri TEXT,
    is_functional INTEGER DEFAULT 0 CHECK(is_functional IN (0, 1)),
    is_inverse_functional INTEGER DEFAULT 0 CHECK(is_inverse_functional IN (0, 1)),
    is_transitive INTEGER DEFAULT 0 CHECK(is_transitive IN (0, 1)),
    is_symmetric INTEGER DEFAULT 0 CHECK(is_symmetric IN (0, 1)),
    is_asymmetric INTEGER DEFAULT 0 CHECK(is_asymmetric IN (0, 1)),
    is_reflexive INTEGER DEFAULT 0 CHECK(is_reflexive IN (0, 1)),
    is_irreflexive INTEGER DEFAULT 0 CHECK(is_irreflexive IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    UNIQUE(ontology_id, property_iri)
);

CREATE INDEX idx_owl_properties_ontology_id ON owl_properties(ontology_id);
CREATE INDEX idx_owl_properties_property_iri ON owl_properties(property_iri);
CREATE INDEX idx_owl_properties_property_type ON owl_properties(property_type);
CREATE INDEX idx_owl_properties_domain ON owl_properties(domain_iri);
CREATE INDEX idx_owl_properties_range ON owl_properties(range_iri);

-- Disjoint class relationships
CREATE TABLE IF NOT EXISTS owl_disjoint_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,
    class_iri_1 TEXT NOT NULL,
    class_iri_2 TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    UNIQUE(ontology_id, class_iri_1, class_iri_2)
);

CREATE INDEX idx_owl_disjoint_ontology_id ON owl_disjoint_classes(ontology_id);
CREATE INDEX idx_owl_disjoint_class1 ON owl_disjoint_classes(class_iri_1);
CREATE INDEX idx_owl_disjoint_class2 ON owl_disjoint_classes(class_iri_2);

-- Equivalent class relationships
CREATE TABLE IF NOT EXISTS owl_equivalent_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,
    class_iri_1 TEXT NOT NULL,
    class_iri_2 TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
    UNIQUE(ontology_id, class_iri_1, class_iri_2)
);

CREATE INDEX idx_owl_equivalent_ontology_id ON owl_equivalent_classes(ontology_id);
CREATE INDEX idx_owl_equivalent_class1 ON owl_equivalent_classes(class_iri_1);
CREATE INDEX idx_owl_equivalent_class2 ON owl_equivalent_classes(class_iri_2);

-- ================================================================
-- MAPPING CONFIGURATION TABLES (replaces mapping.toml)
-- ================================================================

-- Namespace prefixes
CREATE TABLE IF NOT EXISTS namespaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prefix TEXT NOT NULL UNIQUE,
    namespace_iri TEXT NOT NULL,
    is_default INTEGER DEFAULT 0 CHECK(is_default IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_namespaces_prefix ON namespaces(prefix);
CREATE INDEX idx_namespaces_iri ON namespaces(namespace_iri);

-- Class mappings (graph labels to OWL classes)
CREATE TABLE IF NOT EXISTS class_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_label TEXT NOT NULL UNIQUE,
    owl_class_iri TEXT NOT NULL,
    rdfs_label TEXT,
    rdfs_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_class_mappings_label ON class_mappings(graph_label);
CREATE INDEX idx_class_mappings_iri ON class_mappings(owl_class_iri);

-- Property mappings (graph edge types to OWL properties)
CREATE TABLE IF NOT EXISTS property_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_property TEXT NOT NULL UNIQUE,
    owl_property_iri TEXT NOT NULL,
    property_type TEXT NOT NULL CHECK(property_type IN ('object', 'data')),
    rdfs_label TEXT,
    rdfs_comment TEXT,
    rdfs_domain TEXT,
    rdfs_range TEXT,
    inverse_property_iri TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_property_mappings_property ON property_mappings(graph_property);
CREATE INDEX idx_property_mappings_iri ON property_mappings(owl_property_iri);
CREATE INDEX idx_property_mappings_type ON property_mappings(property_type);

-- IRI templates for node/edge generation
CREATE TABLE IF NOT EXISTS iri_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('node', 'edge', 'property', 'class')),
    label_or_type TEXT NOT NULL,
    template TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, label_or_type)
);

CREATE INDEX idx_iri_templates_entity_type ON iri_templates(entity_type);
CREATE INDEX idx_iri_templates_label ON iri_templates(label_or_type);

-- ================================================================
-- MARKDOWN METADATA TABLES (replaces metadata.json)
-- ================================================================

-- File metadata with graph positioning
CREATE TABLE IF NOT EXISTS file_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL DEFAULT 0,
    sha1 TEXT NOT NULL,
    file_blob_sha TEXT,
    node_id TEXT NOT NULL,
    node_size REAL NOT NULL DEFAULT 1.0,
    hyperlink_count INTEGER NOT NULL DEFAULT 0,
    perplexity_link TEXT,
    -- Timestamps
    last_modified TIMESTAMP NOT NULL,
    last_content_change TIMESTAMP,
    last_commit TIMESTAMP,
    last_perplexity_process TIMESTAMP,
    change_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_file_metadata_file_name ON file_metadata(file_name);
CREATE INDEX idx_file_metadata_node_id ON file_metadata(node_id);
CREATE INDEX idx_file_metadata_sha1 ON file_metadata(sha1);
CREATE INDEX idx_file_metadata_last_modified ON file_metadata(last_modified DESC);

-- Topic counts per file
CREATE TABLE IF NOT EXISTS file_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_name) REFERENCES file_metadata(file_name) ON DELETE CASCADE,
    UNIQUE(file_name, topic)
);

CREATE INDEX idx_file_topics_file_name ON file_topics(file_name);
CREATE INDEX idx_file_topics_topic ON file_topics(topic);
CREATE INDEX idx_file_topics_count ON file_topics(count DESC);

-- Ontology blocks extracted from markdown frontmatter
CREATE TABLE IF NOT EXISTS ontology_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    block_type TEXT NOT NULL CHECK(block_type IN ('etsi_domain', 'ontology_definition', 'property_schema', 'constraint', 'axiom')),
    -- ETSI domain fields
    etsi_domain TEXT,
    etsi_subdomain TEXT,
    -- Ontology definition fields
    ontology_class TEXT,
    ontology_property TEXT,
    ontology_range TEXT,
    ontology_cardinality TEXT,
    -- Additional metadata
    raw_yaml TEXT, -- Original YAML block for preservation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_name) REFERENCES file_metadata(file_name) ON DELETE CASCADE
);

CREATE INDEX idx_ontology_blocks_file_name ON ontology_blocks(file_name);
CREATE INDEX idx_ontology_blocks_block_type ON ontology_blocks(block_type);
CREATE INDEX idx_ontology_blocks_etsi_domain ON ontology_blocks(etsi_domain);
CREATE INDEX idx_ontology_blocks_ontology_class ON ontology_blocks(ontology_class);

-- ================================================================
-- PHYSICS CONSTRAINT TABLES (ontology to physics translation)
-- ================================================================

-- Constraint groups (replaces ontology_physics.toml)
CREATE TABLE IF NOT EXISTS constraint_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_name TEXT NOT NULL UNIQUE,
    kernel_name TEXT NOT NULL,
    physics_type TEXT NOT NULL CHECK(physics_type IN ('repulsion', 'attraction', 'alignment', 'symmetry', 'constraint')),
    default_strength REAL NOT NULL DEFAULT 0.5,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    batch_size INTEGER DEFAULT 1000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_constraint_groups_group_name ON constraint_groups(group_name);
CREATE INDEX idx_constraint_groups_physics_type ON constraint_groups(physics_type);

-- Generated constraints from ontology axioms
CREATE TABLE IF NOT EXISTS ontology_constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    constraint_group_id INTEGER NOT NULL,
    ontology_id TEXT NOT NULL,
    axiom_type TEXT NOT NULL,
    source_iri TEXT NOT NULL,
    target_iri TEXT,
    strength REAL NOT NULL DEFAULT 0.5,
    distance_threshold REAL,
    force_multiplier REAL DEFAULT 1.0,
    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (constraint_group_id) REFERENCES constraint_groups(id) ON DELETE CASCADE,
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX idx_ontology_constraints_group_id ON ontology_constraints(constraint_group_id);
CREATE INDEX idx_ontology_constraints_ontology_id ON ontology_constraints(ontology_id);
CREATE INDEX idx_ontology_constraints_axiom_type ON ontology_constraints(axiom_type);
CREATE INDEX idx_ontology_constraints_active ON ontology_constraints(active);

-- ================================================================
-- VALIDATION AND INFERENCE TABLES
-- ================================================================

-- Validation reports
CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL UNIQUE,
    ontology_id TEXT NOT NULL,
    graph_signature TEXT NOT NULL,
    validation_mode TEXT NOT NULL CHECK(validation_mode IN ('quick', 'full', 'incremental')),
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed')),
    violation_count INTEGER DEFAULT 0,
    inference_count INTEGER DEFAULT 0,
    duration_ms INTEGER,
    error_message TEXT,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX idx_validation_reports_report_id ON validation_reports(report_id);
CREATE INDEX idx_validation_reports_ontology_id ON validation_reports(ontology_id);
CREATE INDEX idx_validation_reports_graph_signature ON validation_reports(graph_signature);
CREATE INDEX idx_validation_reports_status ON validation_reports(status);
CREATE INDEX idx_validation_reports_completed_at ON validation_reports(completed_at DESC);

-- Validation violations
CREATE TABLE IF NOT EXISTS validation_violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('error', 'warning', 'info')),
    violation_type TEXT NOT NULL,
    subject_iri TEXT,
    predicate_iri TEXT,
    object_iri TEXT,
    message TEXT NOT NULL,
    suggested_fix TEXT,
    fix_confidence REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES validation_reports(report_id) ON DELETE CASCADE
);

CREATE INDEX idx_validation_violations_report_id ON validation_violations(report_id);
CREATE INDEX idx_validation_violations_severity ON validation_violations(severity);
CREATE INDEX idx_validation_violations_violation_type ON validation_violations(violation_type);
CREATE INDEX idx_validation_violations_subject ON validation_violations(subject_iri);

-- Inferred triples from reasoning
CREATE TABLE IF NOT EXISTS inferred_triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    subject_iri TEXT NOT NULL,
    predicate_iri TEXT NOT NULL,
    object_iri TEXT NOT NULL,
    is_literal INTEGER DEFAULT 0 CHECK(is_literal IN (0, 1)),
    datatype_iri TEXT,
    confidence REAL DEFAULT 1.0,
    derivation_rule TEXT,
    applied INTEGER DEFAULT 0 CHECK(applied IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES validation_reports(report_id) ON DELETE CASCADE
);

CREATE INDEX idx_inferred_triples_report_id ON inferred_triples(report_id);
CREATE INDEX idx_inferred_triples_subject ON inferred_triples(subject_iri);
CREATE INDEX idx_inferred_triples_predicate ON inferred_triples(predicate_iri);
CREATE INDEX idx_inferred_triples_applied ON inferred_triples(applied);

-- ================================================================
-- PERFORMANCE OPTIMIZATION TABLES
-- ================================================================

-- Graph node cache for fast lookups
CREATE TABLE IF NOT EXISTS graph_node_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL UNIQUE,
    node_type TEXT,
    labels TEXT, -- JSON array of labels
    properties TEXT, -- JSON object
    last_validated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_graph_node_cache_node_id ON graph_node_cache(node_id);
CREATE INDEX idx_graph_node_cache_node_type ON graph_node_cache(node_type);
CREATE INDEX idx_graph_node_cache_updated_at ON graph_node_cache(updated_at DESC);

-- Graph edge cache for relationship queries
CREATE TABLE IF NOT EXISTS graph_edge_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id TEXT NOT NULL UNIQUE,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    properties TEXT, -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES graph_node_cache(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES graph_node_cache(node_id) ON DELETE CASCADE
);

CREATE INDEX idx_graph_edge_cache_edge_id ON graph_edge_cache(edge_id);
CREATE INDEX idx_graph_edge_cache_source ON graph_edge_cache(source_node_id);
CREATE INDEX idx_graph_edge_cache_target ON graph_edge_cache(target_node_id);
CREATE INDEX idx_graph_edge_cache_relationship ON graph_edge_cache(relationship_type);

-- ================================================================
-- USER MANAGEMENT AND AUTHENTICATION
-- ================================================================

-- Users table with Nostr pubkey authentication
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nostr_pubkey TEXT NOT NULL UNIQUE,
    username TEXT,
    is_power_user INTEGER DEFAULT 0 CHECK(is_power_user IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_nostr_pubkey ON users(nostr_pubkey);
CREATE INDEX idx_users_is_power_user ON users(is_power_user);
CREATE INDEX idx_users_last_seen ON users(last_seen DESC);

-- Per-user settings overrides
CREATE TABLE IF NOT EXISTS user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK(value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    value_text TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER CHECK(value_boolean IN (0, 1)),
    value_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, key)
);

CREATE INDEX idx_user_settings_user_id ON user_settings(user_id);
CREATE INDEX idx_user_settings_key ON user_settings(key);
CREATE INDEX idx_user_settings_updated_at ON user_settings(updated_at DESC);

-- Settings audit log for tracking changes
CREATE TABLE IF NOT EXISTS settings_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    key TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    action TEXT NOT NULL CHECK(action IN ('create', 'update', 'delete')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX idx_settings_audit_user_id ON settings_audit_log(user_id);
CREATE INDEX idx_settings_audit_key ON settings_audit_log(key);
CREATE INDEX idx_settings_audit_timestamp ON settings_audit_log(timestamp DESC);
CREATE INDEX idx_settings_audit_action ON settings_audit_log(action);

-- ================================================================
-- SYSTEM METADATA
-- ================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    version INTEGER NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 2);

-- ================================================================
-- TRIGGERS FOR UPDATED_AT AUTOMATION
-- ================================================================

CREATE TRIGGER IF NOT EXISTS update_settings_timestamp
AFTER UPDATE ON settings
FOR EACH ROW
BEGIN
    UPDATE settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_physics_settings_timestamp
AFTER UPDATE ON physics_settings
FOR EACH ROW
BEGIN
    UPDATE physics_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_ontologies_timestamp
AFTER UPDATE ON ontologies
FOR EACH ROW
BEGIN
    UPDATE ontologies SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_file_metadata_timestamp
AFTER UPDATE ON file_metadata
FOR EACH ROW
BEGIN
    UPDATE file_metadata SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_ontology_blocks_timestamp
AFTER UPDATE ON ontology_blocks
FOR EACH ROW
BEGIN
    UPDATE ontology_blocks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_graph_node_cache_timestamp
AFTER UPDATE ON graph_node_cache
FOR EACH ROW
BEGIN
    UPDATE graph_node_cache SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_graph_edge_cache_timestamp
AFTER UPDATE ON graph_edge_cache
FOR EACH ROW
BEGIN
    UPDATE graph_edge_cache SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_user_settings_timestamp
AFTER UPDATE ON user_settings
FOR EACH ROW
BEGIN
    UPDATE user_settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_users_last_seen
AFTER UPDATE ON users
FOR EACH ROW
WHEN NEW.last_seen = OLD.last_seen
BEGIN
    UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ================================================================
-- SAMPLE QUERIES
-- ================================================================

-- Query 1: Get all physics settings for a profile
-- SELECT * FROM physics_settings WHERE profile_name = 'default';

-- Query 2: Find all files with ETSI domain metadata
-- SELECT fm.*, ob.etsi_domain, ob.etsi_subdomain
-- FROM file_metadata fm
-- JOIN ontology_blocks ob ON fm.file_name = ob.file_name
-- WHERE ob.etsi_domain IS NOT NULL;

-- Query 3: Get constraint groups by physics type
-- SELECT * FROM constraint_groups WHERE physics_type = 'repulsion' AND enabled = 1;

-- Query 4: Find validation violations by severity
-- SELECT vr.report_id, vr.ontology_id, vv.severity, vv.violation_type, vv.message
-- FROM validation_reports vr
-- JOIN validation_violations vv ON vr.report_id = vv.report_id
-- WHERE vv.severity = 'error'
-- ORDER BY vr.completed_at DESC;

-- Query 5: Get all disjoint class pairs for an ontology
-- SELECT dc.class_iri_1, dc.class_iri_2
-- FROM owl_disjoint_classes dc
-- WHERE dc.ontology_id = 'my-ontology-id';

-- Query 6: Graph generation query - all nodes with metadata
-- SELECT
--     fm.node_id,
--     fm.file_name,
--     fm.node_size,
--     fm.hyperlink_count,
--     GROUP_CONCAT(ft.topic || ':' || ft.count) as topics
-- FROM file_metadata fm
-- LEFT JOIN file_topics ft ON fm.file_name = ft.file_name
-- GROUP BY fm.node_id;

-- Query 7: Fast lookup of node type mapping
-- SELECT cm.owl_class_iri
-- FROM class_mappings cm
-- WHERE cm.graph_label = 'Person';

-- Query 8: Get all inferred triples not yet applied
-- SELECT subject_iri, predicate_iri, object_iri, confidence
-- FROM inferred_triples
-- WHERE applied = 0 AND confidence > 0.8
-- ORDER BY confidence DESC;

-- ================================================================
-- END OF SCHEMA
-- ================================================================
