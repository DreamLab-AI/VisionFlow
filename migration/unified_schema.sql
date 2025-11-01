-- ==============================================================================
-- UNIFIED DATABASE SCHEMA (unified.db)
-- ==============================================================================
-- Integration of: knowledge_graph.db + ontology.db + control center settings
-- Target: Single source of truth for VisionFlow system
-- Date: 2025-10-31
-- Version: 1.0
-- ==============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456; -- 256MB mmap

-- ==============================================================================
-- ONTOLOGY CORE: Classes, Properties, Axioms
-- ==============================================================================

-- OWL Classes (from ontology.db)
CREATE TABLE owl_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iri TEXT UNIQUE NOT NULL,                    -- Full IRI (e.g., http://example.org/ont#Person)
    local_name TEXT NOT NULL,                    -- Short name (e.g., Person)
    namespace_id INTEGER,                        -- References namespaces table
    label TEXT,
    comment TEXT,
    deprecated BOOLEAN DEFAULT 0,

    -- Hierarchy linkage
    parent_class_iri TEXT,                       -- Direct parent (SubClassOf)

    -- Content tracking
    markdown_content TEXT,                       -- Source markdown block
    file_sha1 TEXT,                              -- Checksum for cache invalidation
    source_file TEXT,                            -- Markdown file path

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (parent_class_iri) REFERENCES owl_classes(iri) ON DELETE SET NULL,
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id) ON DELETE SET NULL
);

CREATE INDEX idx_owl_classes_iri ON owl_classes(iri);
CREATE INDEX idx_owl_classes_local_name ON owl_classes(local_name);
CREATE INDEX idx_owl_classes_parent ON owl_classes(parent_class_iri);
CREATE INDEX idx_owl_classes_namespace ON owl_classes(namespace_id);
CREATE INDEX idx_owl_classes_checksum ON owl_classes(file_sha1);

-- OWL Properties (from ontology.db)
CREATE TABLE owl_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iri TEXT UNIQUE NOT NULL,
    local_name TEXT NOT NULL,
    namespace_id INTEGER,

    -- Property type
    property_type TEXT NOT NULL CHECK(property_type IN ('object', 'datatype', 'annotation')),

    -- Domain and Range
    domain_class_iri TEXT,                       -- Domain restriction
    range_class_iri TEXT,                        -- Range restriction

    -- OWL characteristics
    is_functional BOOLEAN DEFAULT 0,             -- Max 1 value per subject
    is_inverse_functional BOOLEAN DEFAULT 0,     -- Max 1 subject per value
    is_transitive BOOLEAN DEFAULT 0,             -- P(a,b) ∧ P(b,c) → P(a,c)
    is_symmetric BOOLEAN DEFAULT 0,              -- P(a,b) → P(b,a)
    is_asymmetric BOOLEAN DEFAULT 0,             -- P(a,b) → ¬P(b,a)
    is_reflexive BOOLEAN DEFAULT 0,              -- ∀x P(x,x)
    is_irreflexive BOOLEAN DEFAULT 0,            -- ∀x ¬P(x,x)

    -- Metadata
    label TEXT,
    comment TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (domain_class_iri) REFERENCES owl_classes(iri) ON DELETE SET NULL,
    FOREIGN KEY (range_class_iri) REFERENCES owl_classes(iri) ON DELETE SET NULL,
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id) ON DELETE SET NULL
);

CREATE INDEX idx_owl_properties_iri ON owl_properties(iri);
CREATE INDEX idx_owl_properties_type ON owl_properties(property_type);
CREATE INDEX idx_owl_properties_domain ON owl_properties(domain_class_iri);
CREATE INDEX idx_owl_properties_range ON owl_properties(range_class_iri);

-- OWL Axioms (from ontology.db + enhanced)
CREATE TABLE owl_axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Axiom type (SubClassOf, DisjointClasses, SameAs, etc.)
    axiom_type TEXT NOT NULL CHECK(axiom_type IN (
        'SubClassOf',
        'DisjointClasses',
        'EquivalentClasses',
        'DisjointUnion',
        'SubObjectPropertyOf',
        'EquivalentProperties',
        'DisjointProperties',
        'InverseProperties',
        'ObjectPropertyDomain',
        'ObjectPropertyRange',
        'FunctionalProperty',
        'InverseFunctionalProperty',
        'TransitiveProperty',
        'SymmetricProperty',
        'AsymmetricProperty',
        'ReflexiveProperty',
        'IrreflexiveProperty',
        'SameIndividual',
        'DifferentIndividuals',
        'ClassAssertion',
        'PropertyAssertion',
        'NegativePropertyAssertion'
    )),

    -- Entity references (flexible for different axiom types)
    subject_id INTEGER,                          -- Source entity
    object_id INTEGER,                           -- Target entity
    property_id INTEGER,                         -- Property (if applicable)

    -- Physics constraint parameters (NEW - for GPU translation)
    strength REAL DEFAULT 1.0,                   -- Constraint strength (0-1)
    priority INTEGER DEFAULT 5,                  -- Priority (1=high, 10=low)
    distance REAL,                               -- Ideal spatial distance
    user_defined BOOLEAN DEFAULT 0,              -- User override flag

    -- Metadata
    graph_id INTEGER,                            -- Graph/ontology context
    metadata TEXT,                               -- JSON metadata

    -- Inference tracking
    inferred BOOLEAN DEFAULT 0,                  -- Inferred by reasoner?
    inference_method TEXT,                       -- Reasoning algorithm
    source_axiom_id INTEGER,                     -- Axiom that triggered inference

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
    FOREIGN KEY (source_axiom_id) REFERENCES owl_axioms(id) ON DELETE SET NULL
);

CREATE INDEX idx_owl_axioms_type ON owl_axioms(axiom_type);
CREATE INDEX idx_owl_axioms_subject ON owl_axioms(subject_id);
CREATE INDEX idx_owl_axioms_object ON owl_axioms(object_id);
CREATE INDEX idx_owl_axioms_graph ON owl_axioms(graph_id);
CREATE INDEX idx_owl_axioms_inferred ON owl_axioms(inferred);
CREATE INDEX idx_owl_axioms_priority ON owl_axioms(priority);

-- Class Hierarchy (optimized traversal)
CREATE TABLE owl_class_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subclass_iri TEXT NOT NULL,
    superclass_iri TEXT NOT NULL,
    graph_id INTEGER,
    distance INTEGER DEFAULT 1,                  -- Tree distance (1=direct)
    inferred BOOLEAN DEFAULT 0,

    UNIQUE(subclass_iri, superclass_iri, graph_id),

    FOREIGN KEY (subclass_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE,
    FOREIGN KEY (superclass_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE,
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);

CREATE INDEX idx_class_hierarchy_sub ON owl_class_hierarchy(subclass_iri);
CREATE INDEX idx_class_hierarchy_super ON owl_class_hierarchy(superclass_iri);
CREATE INDEX idx_class_hierarchy_distance ON owl_class_hierarchy(distance);

-- Individuals (instances)
CREATE TABLE owl_individuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iri TEXT UNIQUE NOT NULL,
    local_name TEXT NOT NULL,
    class_iri TEXT,                              -- Instance of which class
    graph_id INTEGER,
    metadata TEXT,                               -- JSON metadata

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (class_iri) REFERENCES owl_classes(iri) ON DELETE SET NULL,
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);

CREATE INDEX idx_owl_individuals_iri ON owl_individuals(iri);
CREATE INDEX idx_owl_individuals_class ON owl_individuals(class_iri);
CREATE INDEX idx_owl_individuals_graph ON owl_individuals(graph_id);

-- Namespaces
CREATE TABLE namespaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prefix TEXT UNIQUE NOT NULL,                 -- e.g., "foaf", "rdf", "owl"
    uri TEXT UNIQUE NOT NULL,                    -- e.g., "http://xmlns.com/foaf/0.1/"
    default_namespace BOOLEAN DEFAULT 0
);

CREATE INDEX idx_namespaces_prefix ON namespaces(prefix);

-- ==============================================================================
-- GRAPH NODES & EDGES: Visualization + Physics State
-- ==============================================================================

-- Graph Nodes (from knowledge_graph.db + ontology linkage)
CREATE TABLE graph_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metadata_id TEXT NOT NULL UNIQUE,            -- Stable identifier
    label TEXT NOT NULL,

    -- 3D Position (physics state)
    x REAL NOT NULL DEFAULT 0.0,
    y REAL NOT NULL DEFAULT 0.0,
    z REAL NOT NULL DEFAULT 0.0,

    -- Velocity (physics state)
    vx REAL NOT NULL DEFAULT 0.0,
    vy REAL NOT NULL DEFAULT 0.0,
    vz REAL NOT NULL DEFAULT 0.0,

    -- Acceleration (physics state)
    ax REAL NOT NULL DEFAULT 0.0,
    ay REAL NOT NULL DEFAULT 0.0,
    az REAL NOT NULL DEFAULT 0.0,

    -- Physical properties
    mass REAL NOT NULL DEFAULT 1.0,
    charge REAL NOT NULL DEFAULT 1.0,

    -- ONTOLOGY LINKAGE (NEW)
    owl_class_iri TEXT,                          -- Links to owl_classes(iri)
    owl_individual_iri TEXT,                     -- Links to owl_individuals(iri)

    -- Visual properties
    color TEXT,
    size REAL DEFAULT 10.0,
    opacity REAL DEFAULT 1.0,

    -- Node type
    node_type TEXT DEFAULT 'page',

    -- Pinning (user constraints)
    is_pinned INTEGER NOT NULL DEFAULT 0,
    pin_x REAL,
    pin_y REAL,
    pin_z REAL,

    -- Metadata
    metadata TEXT NOT NULL DEFAULT '{}',         -- JSON
    source_file TEXT,
    file_path TEXT,

    -- Graph association
    graph_id INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (owl_class_iri) REFERENCES owl_classes(iri) ON DELETE SET NULL,
    FOREIGN KEY (owl_individual_iri) REFERENCES owl_individuals(iri) ON DELETE SET NULL,
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);

CREATE INDEX idx_graph_nodes_metadata_id ON graph_nodes(metadata_id);
CREATE INDEX idx_graph_nodes_label ON graph_nodes(label);
CREATE INDEX idx_graph_nodes_owl_class ON graph_nodes(owl_class_iri);
CREATE INDEX idx_graph_nodes_owl_individual ON graph_nodes(owl_individual_iri);
CREATE INDEX idx_graph_nodes_type ON graph_nodes(node_type);
CREATE INDEX idx_graph_nodes_pinned ON graph_nodes(is_pinned);
CREATE INDEX idx_graph_nodes_graph ON graph_nodes(graph_id);
CREATE INDEX idx_graph_nodes_position ON graph_nodes(x, y, z); -- Spatial queries

-- Graph Edges (from knowledge_graph.db)
CREATE TABLE graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,

    -- Edge type
    relation_type TEXT NOT NULL DEFAULT 'relates_to',

    -- Physics
    weight REAL NOT NULL DEFAULT 1.0,            -- Spring stiffness
    rest_length REAL,                            -- Ideal edge length

    -- Metadata
    metadata TEXT DEFAULT '{}',                  -- JSON

    graph_id INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);

CREATE INDEX idx_graph_edges_source ON graph_edges(source_id);
CREATE INDEX idx_graph_edges_target ON graph_edges(target_id);
CREATE INDEX idx_graph_edges_type ON graph_edges(relation_type);
CREATE INDEX idx_graph_edges_graph ON graph_edges(graph_id);
CREATE INDEX idx_graph_edges_pair ON graph_edges(source_id, target_id);

-- Graphs (container)
CREATE TABLE graphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,

    -- Ontology linkage
    ontology_iri TEXT,                           -- Base ontology IRI

    -- Statistics (denormalized for performance)
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,

    -- Metadata
    metadata TEXT DEFAULT '{}',                  -- JSON

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_graphs_name ON graphs(name);

-- ==============================================================================
-- GRAPH CLUSTERING: Preserved from knowledge_graph.db
-- ==============================================================================

CREATE TABLE graph_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,                 -- Cluster number (0, 1, 2, ...)
    node_id INTEGER NOT NULL,                    -- Node in this cluster

    -- Cluster metadata
    cluster_label TEXT,
    cluster_algorithm TEXT,                      -- 'kmeans', 'dbscan', 'louvain', etc.

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
    FOREIGN KEY (node_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
);

CREATE INDEX idx_clusters_graph ON graph_clusters(graph_id);
CREATE INDEX idx_clusters_cluster_id ON graph_clusters(cluster_id);
CREATE INDEX idx_clusters_node ON graph_clusters(node_id);
CREATE INDEX idx_clusters_algorithm ON graph_clusters(cluster_algorithm);

-- ==============================================================================
-- PATHFINDING CACHE: Preserved from knowledge_graph.db
-- ==============================================================================

CREATE TABLE pathfinding_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,

    -- SSSP result
    distance REAL NOT NULL,                      -- Shortest path distance
    path TEXT,                                   -- JSON array of node IDs

    -- Cache metadata
    algorithm TEXT DEFAULT 'sssp',               -- 'sssp', 'apsp', 'landmark'
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl_seconds INTEGER DEFAULT 3600,            -- 1-hour default TTL

    UNIQUE(graph_id, source_id, target_id),

    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
);

CREATE INDEX idx_pathfinding_source ON pathfinding_cache(source_id);
CREATE INDEX idx_pathfinding_target ON pathfinding_cache(target_id);
CREATE INDEX idx_pathfinding_graph ON pathfinding_cache(graph_id);
CREATE INDEX idx_pathfinding_computed ON pathfinding_cache(computed_at);

-- ==============================================================================
-- INFERENCE RESULTS: Cached reasoning output
-- ==============================================================================

CREATE TABLE inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Input axiom that triggered inference
    input_axiom_id INTEGER,

    -- Inferred axiom
    inferred_axiom_type TEXT NOT NULL,
    inferred_subject_id INTEGER,
    inferred_object_id INTEGER,

    -- Inference metadata
    confidence REAL DEFAULT 1.0,                 -- Confidence score (0-1)
    reasoning_method TEXT,                       -- e.g., 'transitivity', 'subsumption'
    proof_chain TEXT,                            -- JSON array of axiom IDs

    -- Cache metadata
    ontology_checksum TEXT,                      -- SHA1 of source ontology
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (input_axiom_id) REFERENCES owl_axioms(id) ON DELETE CASCADE
);

CREATE INDEX idx_inference_input ON inference_results(input_axiom_id);
CREATE INDEX idx_inference_type ON inference_results(inferred_axiom_type);
CREATE INDEX idx_inference_checksum ON inference_results(ontology_checksum);

-- ==============================================================================
-- FILE METADATA: GitHub sync tracking for incremental updates
-- ==============================================================================

CREATE TABLE file_metadata (
    file_name TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,

    -- File attributes
    file_size INTEGER,
    file_extension TEXT,

    -- Content hash
    sha1 TEXT,
    content_hash TEXT,

    -- GitHub metadata (if applicable)
    file_blob_sha TEXT,
    github_node_id TEXT,

    -- Statistics
    node_count INTEGER DEFAULT 0,
    hyperlink_count INTEGER DEFAULT 0,
    block_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,

    -- Processing metadata
    perplexity_link TEXT,
    processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'complete', 'error')),
    error_message TEXT,

    -- Timestamps
    last_modified DATETIME,
    last_content_change DATETIME,
    last_commit DATETIME,
    last_perplexity_process DATETIME,
    last_processed DATETIME,
    change_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_file_metadata_path ON file_metadata(file_path);
CREATE INDEX idx_file_metadata_modified ON file_metadata(last_modified);
CREATE INDEX idx_file_metadata_extension ON file_metadata(file_extension);
CREATE INDEX idx_file_metadata_status ON file_metadata(processing_status);
CREATE INDEX idx_file_metadata_hash ON file_metadata(content_hash);
CREATE INDEX idx_file_metadata_sha ON file_metadata(file_blob_sha);

CREATE TRIGGER update_file_metadata_timestamp
AFTER UPDATE ON file_metadata
FOR EACH ROW
BEGIN
    UPDATE file_metadata SET updated_at = CURRENT_TIMESTAMP WHERE file_name = NEW.file_name;
END;

-- ==============================================================================
-- CONTROL CENTER SETTINGS: NEW (Physics & Rendering Configuration)
-- ==============================================================================

-- Physics Settings
CREATE TABLE physics_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT UNIQUE NOT NULL,           -- e.g., "default", "high-performance", "precision"

    -- Time stepping
    dt REAL DEFAULT 0.2,                         -- Time step (5 fps)
    max_iterations INTEGER DEFAULT 300,

    -- Force parameters
    spring_k REAL DEFAULT 0.01,                  -- Spring stiffness
    repel_k REAL DEFAULT 500.0,                  -- Repulsion strength
    center_gravity_k REAL DEFAULT 0.001,         -- Centering force

    -- Damping
    damping REAL DEFAULT 0.85,                   -- Velocity decay
    warmup_damping REAL DEFAULT 0.5,             -- Initial damping (first N iterations)
    warmup_iterations INTEGER DEFAULT 50,

    -- Limits
    max_velocity REAL DEFAULT 50.0,
    max_force REAL DEFAULT 15.0,
    boundary_limit REAL DEFAULT 5000.0,
    boundary_damping REAL DEFAULT 0.3,

    -- Spatial grid
    grid_cell_size REAL DEFAULT 100.0,

    -- Mass scaling
    mass_scale REAL DEFAULT 1.0,                 -- Global mass multiplier

    -- Stability gates
    enable_stability_gates BOOLEAN DEFAULT 1,
    stability_threshold REAL DEFAULT 0.001,      -- Kinetic energy threshold
    stability_min_iterations INTEGER DEFAULT 600,

    is_default BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_physics_profile ON physics_settings(profile_name);

-- Constraint Settings (NEW)
CREATE TABLE constraint_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT UNIQUE NOT NULL,

    -- Constraint activation
    enable_progressive_activation BOOLEAN DEFAULT 1,
    constraint_ramp_frames INTEGER DEFAULT 60,   -- Frames to fully activate

    -- LOD (Level of Detail) thresholds
    lod_near_distance REAL DEFAULT 100.0,        -- All constraints active
    lod_medium_distance REAL DEFAULT 500.0,      -- Priority ≤ 5 active
    lod_far_distance REAL DEFAULT 1000.0,        -- Priority ≤ 3 active

    -- Priority weights
    priority_1_weight REAL DEFAULT 10.0,         -- User overrides
    priority_2_weight REAL DEFAULT 5.0,          -- Identity constraints
    priority_3_weight REAL DEFAULT 2.5,          -- Disjoint constraints
    priority_4_weight REAL DEFAULT 1.5,          -- Hierarchy constraints
    priority_5_weight REAL DEFAULT 1.0,          -- Default

    -- Activation frames (for progressive ramping)
    activation_frame_offset INTEGER DEFAULT 0,   -- Frame to start activation

    is_default BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_constraint_profile ON constraint_settings(profile_name);

-- Rendering Settings (NEW)
CREATE TABLE rendering_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT UNIQUE NOT NULL,

    -- Node rendering
    node_size_scale REAL DEFAULT 1.0,            -- Global node size multiplier
    node_lod_near REAL DEFAULT 10.0,             -- Full detail distance
    node_lod_medium REAL DEFAULT 50.0,           -- Medium detail
    node_lod_far REAL DEFAULT 200.0,             -- Low detail (billboard)
    node_lod_cull REAL DEFAULT 1000.0,           -- Culling distance

    -- Edge rendering
    edge_thickness REAL DEFAULT 1.0,
    edge_lod_near REAL DEFAULT 50.0,
    edge_lod_far REAL DEFAULT 500.0,

    -- Label rendering
    label_distance_min REAL DEFAULT 10.0,        -- Always visible if closer
    label_distance_max REAL DEFAULT 100.0,       -- Always hidden if farther
    label_importance_threshold REAL DEFAULT 0.5, -- Semantic importance cutoff

    -- Colors
    default_node_color TEXT DEFAULT '#3498db',
    default_edge_color TEXT DEFAULT '#95a5a6',
    highlight_color TEXT DEFAULT '#e74c3c',

    -- Performance
    max_visible_nodes INTEGER DEFAULT 5000,
    max_visible_edges INTEGER DEFAULT 10000,

    is_default BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rendering_profile ON rendering_settings(profile_name);

-- User Profiles (NEW - Save/Load Configurations)
CREATE TABLE constraint_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,                                -- User identifier
    profile_name TEXT NOT NULL,

    -- Profile references
    physics_profile TEXT,
    constraint_profile TEXT,
    rendering_profile TEXT,

    -- Metadata
    description TEXT,
    is_public BOOLEAN DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (physics_profile) REFERENCES physics_settings(profile_name),
    FOREIGN KEY (constraint_profile) REFERENCES constraint_settings(profile_name),
    FOREIGN KEY (rendering_profile) REFERENCES rendering_settings(profile_name)
);

CREATE INDEX idx_profiles_user ON constraint_profiles(user_id);
CREATE INDEX idx_profiles_name ON constraint_profiles(profile_name);

-- ==============================================================================
-- DEFAULT DATA: Initial configuration
-- ==============================================================================

-- Default physics settings
INSERT INTO physics_settings (
    profile_name, dt, spring_k, repel_k, center_gravity_k,
    damping, max_velocity, max_force, grid_cell_size,
    enable_stability_gates, is_default
) VALUES (
    'default', 0.2, 0.01, 500.0, 0.001,
    0.85, 50.0, 15.0, 100.0,
    1, 1
);

-- Default constraint settings
INSERT INTO constraint_settings (
    profile_name, enable_progressive_activation, constraint_ramp_frames,
    lod_near_distance, lod_medium_distance, lod_far_distance,
    is_default
) VALUES (
    'default', 1, 60,
    100.0, 500.0, 1000.0,
    1
);

-- Default rendering settings
INSERT INTO rendering_settings (
    profile_name, node_size_scale, edge_thickness,
    label_distance_min, label_distance_max,
    max_visible_nodes, max_visible_edges,
    is_default
) VALUES (
    'default', 1.0, 1.0,
    10.0, 100.0,
    5000, 10000,
    1
);

-- Common namespaces
INSERT INTO namespaces (prefix, uri, default_namespace) VALUES
    ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 0),
    ('rdfs', 'http://www.w3.org/2000/01/rdf-schema#', 0),
    ('owl', 'http://www.w3.org/2002/07/owl#', 0),
    ('xsd', 'http://www.w3.org/2001/XMLSchema#', 0),
    ('foaf', 'http://xmlns.com/foaf/0.1/', 0),
    ('dc', 'http://purl.org/dc/elements/1.1/', 0);

-- ==============================================================================
-- TRIGGERS: Auto-update timestamps and denormalized counts
-- ==============================================================================

-- Update timestamps
CREATE TRIGGER update_owl_classes_timestamp
AFTER UPDATE ON owl_classes
FOR EACH ROW
BEGIN
    UPDATE owl_classes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_graph_nodes_timestamp
AFTER UPDATE ON graph_nodes
FOR EACH ROW
BEGIN
    UPDATE graph_nodes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update graph statistics
CREATE TRIGGER update_graph_node_count_insert
AFTER INSERT ON graph_nodes
FOR EACH ROW
BEGIN
    UPDATE graphs SET node_count = node_count + 1 WHERE id = NEW.graph_id;
END;

CREATE TRIGGER update_graph_node_count_delete
AFTER DELETE ON graph_nodes
FOR EACH ROW
BEGIN
    UPDATE graphs SET node_count = node_count - 1 WHERE id = OLD.graph_id;
END;

CREATE TRIGGER update_graph_edge_count_insert
AFTER INSERT ON graph_edges
FOR EACH ROW
BEGIN
    UPDATE graphs SET edge_count = edge_count + 1 WHERE id = NEW.graph_id;
END;

CREATE TRIGGER update_graph_edge_count_delete
AFTER DELETE ON graph_edges
FOR EACH ROW
BEGIN
    UPDATE graphs SET edge_count = edge_count - 1 WHERE id = OLD.graph_id;
END;

-- ==============================================================================
-- VIEWS: Common queries
-- ==============================================================================

-- Active constraints (user-defined + inferred with priority)
CREATE VIEW active_constraints AS
SELECT
    a.id,
    a.axiom_type,
    a.subject_id,
    a.object_id,
    a.strength,
    a.priority,
    a.distance,
    a.user_defined,
    CASE
        WHEN a.user_defined = 1 THEN 1
        WHEN a.inferred = 1 THEN 6
        ELSE a.priority
    END AS effective_priority
FROM owl_axioms a
WHERE a.axiom_type IN (
    'SubClassOf',
    'DisjointClasses',
    'SameIndividual',
    'FunctionalProperty'
)
ORDER BY effective_priority ASC;

-- Node positions with ontology metadata
CREATE VIEW node_view AS
SELECT
    n.id,
    n.metadata_id,
    n.label,
    n.x, n.y, n.z,
    n.vx, n.vy, n.vz,
    n.mass,
    n.owl_class_iri,
    c.label AS class_label,
    c.parent_class_iri,
    n.node_type,
    n.is_pinned
FROM graph_nodes n
LEFT JOIN owl_classes c ON n.owl_class_iri = c.iri;

-- Graph statistics
CREATE VIEW graph_stats AS
SELECT
    g.id AS graph_id,
    g.name,
    g.node_count,
    g.edge_count,
    COUNT(DISTINCT c.cluster_id) AS cluster_count,
    COUNT(DISTINCT a.id) AS constraint_count
FROM graphs g
LEFT JOIN graph_clusters c ON g.id = c.graph_id
LEFT JOIN owl_axioms a ON g.id = a.graph_id
GROUP BY g.id;

-- ==============================================================================
-- PERFORMANCE HINTS
-- ==============================================================================

-- Analyze tables for query optimizer
ANALYZE;

-- ==============================================================================
-- SCHEMA VALIDATION QUERIES (for testing)
-- ==============================================================================

-- Verify foreign key integrity
-- PRAGMA foreign_key_check;

-- Count tables
-- SELECT COUNT(*) AS table_count FROM sqlite_master WHERE type='table';

-- Check indexes
-- SELECT name, tbl_name FROM sqlite_master WHERE type='index' ORDER BY tbl_name;

-- ==============================================================================
-- END OF SCHEMA
-- ==============================================================================
