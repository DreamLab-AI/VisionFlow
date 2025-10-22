-- VisionFlow Ontology Metadata Database Schema
-- SQLite schema for OWL ontology metadata and mappings

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 1);

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
