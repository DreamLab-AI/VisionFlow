-- VisionFlow Database Schema
-- SQLite schema for settings, ontology metadata, and physics configuration

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 1);

-- =================================================================
-- SETTINGS TABLES
-- =================================================================

-- General application settings with flexible value storage
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_type TEXT NOT NULL CHECK (value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    value_text TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER CHECK (value_boolean IN (0, 1)),
    value_json TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);

-- Physics simulation settings profiles
CREATE TABLE IF NOT EXISTS physics_settings (
    profile_name TEXT PRIMARY KEY,

    -- Core physics parameters
    damping REAL NOT NULL DEFAULT 0.9,
    dt REAL NOT NULL DEFAULT 0.016,
    iterations INTEGER NOT NULL DEFAULT 1,
    max_velocity REAL NOT NULL DEFAULT 50.0,
    max_force REAL NOT NULL DEFAULT 100.0,
    repel_k REAL NOT NULL DEFAULT 500.0,
    spring_k REAL NOT NULL DEFAULT 150.0,
    mass_scale REAL NOT NULL DEFAULT 1.0,
    boundary_damping REAL NOT NULL DEFAULT 0.5,
    temperature REAL NOT NULL DEFAULT 1.0,
    gravity REAL NOT NULL DEFAULT 0.0,
    bounds_size REAL NOT NULL DEFAULT 1000.0,
    enable_bounds INTEGER NOT NULL DEFAULT 1 CHECK (enable_bounds IN (0, 1)),

    -- CUDA kernel parameters
    rest_length REAL NOT NULL DEFAULT 50.0,
    repulsion_cutoff REAL NOT NULL DEFAULT 300.0,
    repulsion_softening_epsilon REAL NOT NULL DEFAULT 1.0,
    center_gravity_k REAL NOT NULL DEFAULT 0.1,
    grid_cell_size REAL NOT NULL DEFAULT 100.0,
    warmup_iterations INTEGER NOT NULL DEFAULT 100,
    cooling_rate REAL NOT NULL DEFAULT 0.95,

    -- Constraint parameters
    constraint_ramp_frames INTEGER NOT NULL DEFAULT 60,
    constraint_max_force_per_node REAL NOT NULL DEFAULT 50.0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default physics profile
INSERT OR IGNORE INTO physics_settings (profile_name) VALUES ('default');

-- =================================================================
-- ONTOLOGY METADATA TABLES
-- =================================================================

-- Ontology metadata
CREATE TABLE IF NOT EXISTS ontologies (
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

CREATE INDEX IF NOT EXISTS idx_ontologies_source ON ontologies(source_path);
CREATE INDEX IF NOT EXISTS idx_ontologies_hash ON ontologies(content_hash);

-- OWL class definitions
CREATE TABLE IF NOT EXISTS owl_classes (
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,
    label TEXT,
    comment TEXT,
    parent_class_iri TEXT,
    is_deprecated INTEGER DEFAULT 0 CHECK (is_deprecated IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ontology_id, class_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_owl_classes_iri ON owl_classes(class_iri);
CREATE INDEX IF NOT EXISTS idx_owl_classes_parent ON owl_classes(parent_class_iri);

-- OWL properties (object and data properties)
CREATE TABLE IF NOT EXISTS owl_properties (
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

CREATE INDEX IF NOT EXISTS idx_owl_properties_iri ON owl_properties(property_iri);
CREATE INDEX IF NOT EXISTS idx_owl_properties_type ON owl_properties(property_type);

-- Disjoint class pairs
CREATE TABLE IF NOT EXISTS owl_disjoint_classes (
    ontology_id TEXT NOT NULL,
    class_iri_1 TEXT NOT NULL,
    class_iri_2 TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ontology_id, class_iri_1, class_iri_2),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

-- =================================================================
-- FILE METADATA TABLES
-- =================================================================

-- Markdown file metadata
CREATE TABLE IF NOT EXISTS file_metadata (
    file_name TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    sha1 TEXT,
    file_blob_sha TEXT,
    node_id TEXT,
    node_size INTEGER,
    hyperlink_count INTEGER DEFAULT 0,
    perplexity_link TEXT,
    last_modified TIMESTAMP,
    last_content_change TIMESTAMP,
    last_commit TIMESTAMP,
    last_perplexity_process TIMESTAMP,
    change_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_file_metadata_path ON file_metadata(file_path);
CREATE INDEX IF NOT EXISTS idx_file_metadata_modified ON file_metadata(last_modified);

-- File topic counts
CREATE TABLE IF NOT EXISTS file_topics (
    file_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (file_name, topic),
    FOREIGN KEY (file_name) REFERENCES file_metadata(file_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_file_topics_topic ON file_topics(topic);

-- =================================================================
-- MAPPING CONFIGURATION TABLES
-- =================================================================

-- Namespace prefix mappings
CREATE TABLE IF NOT EXISTS namespaces (
    prefix TEXT PRIMARY KEY,
    namespace_iri TEXT NOT NULL,
    is_default INTEGER DEFAULT 0 CHECK (is_default IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Class mappings (graph label to OWL class)
CREATE TABLE IF NOT EXISTS class_mappings (
    graph_label TEXT PRIMARY KEY,
    owl_class_iri TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Property mappings (graph property to OWL property)
CREATE TABLE IF NOT EXISTS property_mappings (
    graph_property TEXT PRIMARY KEY,
    owl_property_iri TEXT NOT NULL,
    property_type TEXT NOT NULL CHECK (property_type IN ('ObjectProperty', 'DataProperty')),
    rdfs_domain TEXT,
    rdfs_range TEXT,
    inverse_property_iri TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
