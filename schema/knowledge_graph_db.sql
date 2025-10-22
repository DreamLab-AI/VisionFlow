-- =================================================================
-- VisionFlow Knowledge Graph Database (knowledge_graph.db)
-- =================================================================
-- Purpose: Main graph structure from local markdown files with physics simulation
-- Version: 2.0.0
-- Created: 2025-10-22
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
VALUES (1, 2, 'Three-database system - Knowledge graph database');

-- =================================================================
-- NODES TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metadata_id TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,

    -- Position data (3D coordinates)
    x REAL NOT NULL DEFAULT 0.0,
    y REAL NOT NULL DEFAULT 0.0,
    z REAL NOT NULL DEFAULT 0.0,

    -- Velocity data (for physics simulation)
    vx REAL NOT NULL DEFAULT 0.0,
    vy REAL NOT NULL DEFAULT 0.0,
    vz REAL NOT NULL DEFAULT 0.0,

    -- Acceleration data
    ax REAL NOT NULL DEFAULT 0.0,
    ay REAL NOT NULL DEFAULT 0.0,
    az REAL NOT NULL DEFAULT 0.0,

    -- Physical properties
    mass REAL NOT NULL DEFAULT 1.0,
    charge REAL NOT NULL DEFAULT 1.0,

    -- Visual properties
    color TEXT,
    size REAL DEFAULT 10.0,
    opacity REAL DEFAULT 1.0 CHECK (opacity >= 0.0 AND opacity <= 1.0),

    -- Node type classification
    node_type TEXT DEFAULT 'page' CHECK (node_type IN ('page', 'tag', 'block', 'concept', 'journal')),

    -- Pinning and constraints
    is_pinned INTEGER NOT NULL DEFAULT 0 CHECK (is_pinned IN (0, 1)),
    pin_x REAL,
    pin_y REAL,
    pin_z REAL,

    -- Metadata as JSON
    metadata TEXT NOT NULL DEFAULT '{}',

    -- Source file information
    source_file TEXT,
    file_path TEXT,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_modified DATETIME
);

CREATE INDEX IF NOT EXISTS idx_nodes_metadata_id ON nodes(metadata_id);
CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_source_file ON nodes(source_file);
CREATE INDEX IF NOT EXISTS idx_nodes_updated_at ON nodes(updated_at);
CREATE INDEX IF NOT EXISTS idx_nodes_pinned ON nodes(is_pinned);

-- Spatial index for efficient proximity queries
CREATE INDEX IF NOT EXISTS idx_nodes_spatial_xy ON nodes(x, y);
CREATE INDEX IF NOT EXISTS idx_nodes_spatial_xyz ON nodes(x, y, z);

-- =================================================================
-- EDGES TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source INTEGER NOT NULL,
    target INTEGER NOT NULL,

    -- Edge properties
    weight REAL NOT NULL DEFAULT 1.0,
    edge_type TEXT DEFAULT 'link' CHECK (edge_type IN ('link', 'tag', 'parent', 'reference', 'related')),

    -- Visual properties
    color TEXT,
    opacity REAL DEFAULT 1.0 CHECK (opacity >= 0.0 AND opacity <= 1.0),

    -- Bidirectional flag
    is_bidirectional INTEGER NOT NULL DEFAULT 0 CHECK (is_bidirectional IN (0, 1)),

    -- Edge metadata as JSON
    metadata TEXT DEFAULT '{}',

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE,

    -- Prevent duplicate edges
    UNIQUE (source, target, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_source_target ON edges(source, target);
CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);

-- =================================================================
-- NODE PROPERTIES TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS node_properties (
    node_id INTEGER NOT NULL,
    property_key TEXT NOT NULL,
    property_value TEXT NOT NULL,
    property_type TEXT NOT NULL CHECK (property_type IN ('string', 'integer', 'float', 'boolean', 'datetime')),

    PRIMARY KEY (node_id, property_key),
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_node_props_key ON node_properties(property_key);
CREATE INDEX IF NOT EXISTS idx_node_props_value ON node_properties(property_value);
CREATE INDEX IF NOT EXISTS idx_node_props_type ON node_properties(property_type);

-- =================================================================
-- FILE METADATA TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS file_metadata (
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

CREATE INDEX IF NOT EXISTS idx_file_metadata_path ON file_metadata(file_path);
CREATE INDEX IF NOT EXISTS idx_file_metadata_modified ON file_metadata(last_modified);
CREATE INDEX IF NOT EXISTS idx_file_metadata_extension ON file_metadata(file_extension);
CREATE INDEX IF NOT EXISTS idx_file_metadata_status ON file_metadata(processing_status);
CREATE INDEX IF NOT EXISTS idx_file_metadata_hash ON file_metadata(content_hash);

-- =================================================================
-- FILE TOPICS TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS file_topics (
    file_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    confidence REAL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),

    PRIMARY KEY (file_name, topic),
    FOREIGN KEY (file_name) REFERENCES file_metadata(file_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_file_topics_topic ON file_topics(topic);
CREATE INDEX IF NOT EXISTS idx_file_topics_count ON file_topics(count);
CREATE INDEX IF NOT EXISTS idx_file_topics_confidence ON file_topics(confidence);

-- =================================================================
-- GRAPH METADATA TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS graph_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    value_type TEXT DEFAULT 'string' CHECK (value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    description TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_graph_metadata_type ON graph_metadata(value_type);

-- =================================================================
-- GRAPH SNAPSHOTS TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS graph_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_name TEXT UNIQUE NOT NULL,

    -- Snapshot metadata
    description TEXT,
    snapshot_type TEXT DEFAULT 'manual' CHECK (snapshot_type IN ('manual', 'automatic', 'scheduled')),

    -- Compressed graph data
    snapshot_data TEXT NOT NULL, -- Compressed JSON of full graph

    -- Statistics
    node_count INTEGER NOT NULL,
    edge_count INTEGER NOT NULL,
    file_count INTEGER NOT NULL DEFAULT 0,

    -- Size metrics
    uncompressed_size INTEGER,
    compressed_size INTEGER,
    compression_ratio REAL,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'system'
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date ON graph_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_name ON graph_snapshots(snapshot_name);
CREATE INDEX IF NOT EXISTS idx_snapshots_type ON graph_snapshots(snapshot_type);

-- =================================================================
-- GRAPH CLUSTERS TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS graph_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_name TEXT NOT NULL,

    -- Cluster properties
    node_count INTEGER NOT NULL DEFAULT 0,
    avg_degree REAL,
    density REAL,

    -- Centroid position
    centroid_x REAL,
    centroid_y REAL,
    centroid_z REAL,

    -- Visual properties
    color TEXT,

    -- Metadata
    metadata TEXT DEFAULT '{}',

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clusters_name ON graph_clusters(cluster_name);

-- =================================================================
-- NODE CLUSTER MEMBERSHIP TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS node_cluster_membership (
    node_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    membership_score REAL DEFAULT 1.0 CHECK (membership_score >= 0.0 AND membership_score <= 1.0),

    PRIMARY KEY (node_id, cluster_id),
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES graph_clusters(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cluster_membership_node ON node_cluster_membership(node_id);
CREATE INDEX IF NOT EXISTS idx_cluster_membership_cluster ON node_cluster_membership(cluster_id);

-- =================================================================
-- GRAPH ANALYTICS TABLE
-- =================================================================

CREATE TABLE IF NOT EXISTS graph_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Metric type
    metric_name TEXT NOT NULL,
    metric_category TEXT CHECK (metric_category IN ('centrality', 'community', 'structure', 'dynamics')),

    -- Results as JSON
    results TEXT NOT NULL,

    -- Statistics
    computation_time_ms INTEGER,

    -- Timestamps
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analytics_metric ON graph_analytics(metric_name);
CREATE INDEX IF NOT EXISTS idx_analytics_category ON graph_analytics(metric_category);
CREATE INDEX IF NOT EXISTS idx_analytics_time ON graph_analytics(computed_at);

-- =================================================================
-- INITIALIZATION DATA
-- =================================================================

BEGIN TRANSACTION;

-- Initialize graph metadata
INSERT OR IGNORE INTO graph_metadata (key, value, value_type, description) VALUES
    ('node_count', '0', 'integer', 'Total number of nodes in graph'),
    ('edge_count', '0', 'integer', 'Total number of edges in graph'),
    ('last_full_rebuild', datetime('now'), 'string', 'Last full graph rebuild timestamp'),
    ('graph_version', '2', 'integer', 'Graph schema version'),
    ('source_type', 'local_markdown', 'string', 'Source of graph data'),
    ('physics_enabled', 'true', 'boolean', 'Physics simulation enabled'),
    ('current_profile', 'default', 'string', 'Current physics profile'),
    ('auto_layout', 'true', 'boolean', 'Automatic layout enabled');

COMMIT;

-- =================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =================================================================

-- Update node timestamp on modification
CREATE TRIGGER IF NOT EXISTS update_nodes_timestamp
AFTER UPDATE ON nodes
FOR EACH ROW
BEGIN
    UPDATE nodes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update file metadata timestamp
CREATE TRIGGER IF NOT EXISTS update_file_metadata_timestamp
AFTER UPDATE ON file_metadata
FOR EACH ROW
BEGIN
    UPDATE file_metadata SET updated_at = CURRENT_TIMESTAMP WHERE file_name = NEW.file_name;
END;

-- Update cluster timestamp
CREATE TRIGGER IF NOT EXISTS update_cluster_timestamp
AFTER UPDATE ON graph_clusters
FOR EACH ROW
BEGIN
    UPDATE graph_clusters SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Automatically update node count in graph_metadata
CREATE TRIGGER IF NOT EXISTS update_node_count_on_insert
AFTER INSERT ON nodes
BEGIN
    UPDATE graph_metadata
    SET value = CAST((SELECT COUNT(*) FROM nodes) AS TEXT)
    WHERE key = 'node_count';
END;

CREATE TRIGGER IF NOT EXISTS update_node_count_on_delete
AFTER DELETE ON nodes
BEGIN
    UPDATE graph_metadata
    SET value = CAST((SELECT COUNT(*) FROM nodes) AS TEXT)
    WHERE key = 'node_count';
END;

-- Automatically update edge count in graph_metadata
CREATE TRIGGER IF NOT EXISTS update_edge_count_on_insert
AFTER INSERT ON edges
BEGIN
    UPDATE graph_metadata
    SET value = CAST((SELECT COUNT(*) FROM edges) AS TEXT)
    WHERE key = 'edge_count';
END;

CREATE TRIGGER IF NOT EXISTS update_edge_count_on_delete
AFTER DELETE ON edges
BEGIN
    UPDATE graph_metadata
    SET value = CAST((SELECT COUNT(*) FROM edges) AS TEXT)
    WHERE key = 'edge_count';
END;

-- =================================================================
-- VIEWS FOR CONVENIENT QUERYING
-- =================================================================

-- View for graph statistics
CREATE VIEW IF NOT EXISTS v_graph_stats AS
SELECT
    (SELECT COUNT(*) FROM nodes) as total_nodes,
    (SELECT COUNT(*) FROM edges) as total_edges,
    (SELECT COUNT(*) FROM file_metadata) as total_files,
    (SELECT COUNT(*) FROM graph_clusters) as total_clusters,
    (SELECT AVG(degree) FROM (
        SELECT COUNT(*) as degree
        FROM edges
        GROUP BY source
    )) as avg_degree,
    (SELECT value FROM graph_metadata WHERE key = 'last_full_rebuild') as last_rebuild;

-- View for node degree centrality
CREATE VIEW IF NOT EXISTS v_node_degrees AS
SELECT
    n.id,
    n.metadata_id,
    n.label,
    COUNT(DISTINCT e_out.id) as out_degree,
    COUNT(DISTINCT e_in.id) as in_degree,
    COUNT(DISTINCT e_out.id) + COUNT(DISTINCT e_in.id) as total_degree
FROM nodes n
LEFT JOIN edges e_out ON n.id = e_out.source
LEFT JOIN edges e_in ON n.id = e_in.target
GROUP BY n.id;

-- View for file processing status
CREATE VIEW IF NOT EXISTS v_file_status AS
SELECT
    file_name,
    file_path,
    processing_status,
    node_count,
    hyperlink_count,
    last_processed,
    error_message
FROM file_metadata
ORDER BY last_processed DESC;

-- View for pinned nodes
CREATE VIEW IF NOT EXISTS v_pinned_nodes AS
SELECT
    id,
    metadata_id,
    label,
    pin_x,
    pin_y,
    pin_z,
    updated_at
FROM nodes
WHERE is_pinned = 1;

-- =================================================================
-- UTILITY FUNCTIONS (stored as metadata for Rust implementation)
-- =================================================================

-- Store SQL snippets for common operations
INSERT OR IGNORE INTO graph_metadata (key, value, value_type, description) VALUES
    ('query_neighbors', 'SELECT * FROM nodes WHERE id IN (SELECT target FROM edges WHERE source = ?)', 'string', 'Query to get node neighbors'),
    ('query_hub_nodes', 'SELECT * FROM v_node_degrees ORDER BY total_degree DESC LIMIT 10', 'string', 'Query to get hub nodes'),
    ('query_isolated_nodes', 'SELECT * FROM nodes WHERE id NOT IN (SELECT DISTINCT source FROM edges UNION SELECT DISTINCT target FROM edges)', 'string', 'Query for isolated nodes');

-- =================================================================
-- VACUUM AND OPTIMIZE
-- =================================================================

PRAGMA optimize;
