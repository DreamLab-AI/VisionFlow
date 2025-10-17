// examples/database_migration_example.rs
//! Example migration script showing how to migrate from TOML/YAML to SQLite
//!
//! Run with: cargo run --example database_migration_example

use std::collections::HashMap;
use std::fs;
use std::path::Path;

// Note: This is a standalone example. In production, these would use the actual service
// and have proper error handling.

fn main() {
    println!("=== SQLite Database Migration Example ===\n");

    // Example 1: Migrate dev_config.toml
    migrate_dev_config_example();

    // Example 2: Migrate mapping.toml
    migrate_mapping_config_example();

    // Example 3: Migrate physics constraints
    migrate_physics_constraints_example();

    // Example 4: Migrate file metadata
    migrate_file_metadata_example();

    println!("\n=== Migration Examples Complete ===");
}

fn migrate_dev_config_example() {
    println!("--- Example 1: Migrate dev_config.toml ---");

    // Simulated TOML parsing (in production, use actual toml crate)
    let example_physics_params = r#"
[physics]
force_epsilon = 1e-8
spring_length_multiplier = 5.0
rest_length = 100.0
repulsion_cutoff = 150.0
repulsion_softening_epsilon = 0.0001
center_gravity_k = 0.001
grid_cell_size = 50.0
warmup_iterations = 100
cooling_rate = 0.001
max_force = 50.0
max_velocity = 150.0
    "#;

    println!("Source TOML:\n{}", example_physics_params);

    // SQL INSERT statement that would be executed
    let sql = r#"
INSERT INTO physics_settings (
    profile_name, rest_length, repulsion_cutoff,
    repulsion_softening_epsilon, center_gravity_k, grid_cell_size,
    warmup_iterations, cooling_rate, max_force, max_velocity
) VALUES (
    'default',          -- profile_name
    100.0,              -- rest_length
    150.0,              -- repulsion_cutoff
    0.0001,             -- repulsion_softening_epsilon
    0.001,              -- center_gravity_k
    50.0,               -- grid_cell_size
    100,                -- warmup_iterations
    0.001,              -- cooling_rate
    50.0,               -- max_force
    150.0               -- max_velocity
);
    "#;

    println!("\nGenerated SQL:\n{}", sql);
    println!("✓ Would insert physics settings into database\n");
}

fn migrate_mapping_config_example() {
    println!("--- Example 2: Migrate mapping.toml ---");

    let example_mapping = r#"
[namespaces]
owl = "http://www.w3.org/2002/07/owl#"
rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
foaf = "http://xmlns.com/foaf/0.1/"

[class_mappings.person]
owl_class = "foaf:Person"
rdfs_label = "Person"

[object_property_mappings.employedBy]
owl_property = "vocab:employedBy"
rdfs_domain = "foaf:Person"
rdfs_range = "vocab:Company"
owl_inverse_of = "vocab:employs"
    "#;

    println!("Source TOML:\n{}", example_mapping);

    // Multiple SQL statements for different tables
    let namespace_sql = r#"
-- Migrate namespaces
INSERT INTO namespaces (prefix, namespace_iri) VALUES
    ('owl', 'http://www.w3.org/2002/07/owl#'),
    ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
    ('foaf', 'http://xmlns.com/foaf/0.1/');
    "#;

    let class_mapping_sql = r#"
-- Migrate class mappings
INSERT INTO class_mappings (graph_label, owl_class_iri, rdfs_label)
VALUES ('person', 'foaf:Person', 'Person');
    "#;

    let property_mapping_sql = r#"
-- Migrate property mappings
INSERT INTO property_mappings (
    graph_property, owl_property_iri, property_type,
    rdfs_domain, rdfs_range, inverse_property_iri
) VALUES (
    'employedBy',
    'vocab:employedBy',
    'object',
    'foaf:Person',
    'vocab:Company',
    'vocab:employs'
);
    "#;

    println!("\nGenerated SQL:");
    println!("{}", namespace_sql);
    println!("{}", class_mapping_sql);
    println!("{}", property_mapping_sql);
    println!("✓ Would insert mapping configuration into database\n");
}

fn migrate_physics_constraints_example() {
    println!("--- Example 3: Migrate ontology_physics.toml ---");

    let example_constraints = r#"
[constraint_groups.disjoint_classes]
enabled = true
kernel_name = "apply_disjoint_classes_kernel"
default_strength = 0.8
physics_type = "repulsion"

[constraint_groups.subclass_hierarchy]
enabled = true
kernel_name = "apply_subclass_hierarchy_kernel"
default_strength = 0.6
physics_type = "alignment"
    "#;

    println!("Source TOML:\n{}", example_constraints);

    let sql = r#"
INSERT INTO constraint_groups (group_name, kernel_name, physics_type, default_strength, enabled)
VALUES
    ('disjoint_classes', 'apply_disjoint_classes_kernel', 'repulsion', 0.8, 1),
    ('subclass_hierarchy', 'apply_subclass_hierarchy_kernel', 'alignment', 0.6, 1);
    "#;

    println!("\nGenerated SQL:\n{}", sql);
    println!("✓ Would insert constraint groups into database\n");
}

fn migrate_file_metadata_example() {
    println!("--- Example 4: Migrate file metadata ---");

    // Simulated in-memory metadata (in production, this would be from MetadataStore)
    let example_metadata = r#"
{
    "example.md": {
        "file_name": "example.md",
        "file_size": 2048,
        "sha1": "abc123def456",
        "node_id": "1",
        "node_size": 1.5,
        "hyperlink_count": 5,
        "last_modified": "2025-10-17T12:00:00Z",
        "topic_counts": {
            "rust": 10,
            "ontology": 5,
            "database": 3
        }
    }
}
    "#;

    println!("Source metadata JSON:\n{}", example_metadata);

    let file_sql = r#"
INSERT INTO file_metadata (
    file_name, file_path, file_size, sha1, node_id, node_size,
    hyperlink_count, last_modified
) VALUES (
    'example.md',
    './markdown/example.md',
    2048,
    'abc123def456',
    '1',
    1.5,
    5,
    '2025-10-17T12:00:00Z'
);
    "#;

    let topics_sql = r#"
INSERT INTO file_topics (file_name, topic, count) VALUES
    ('example.md', 'rust', 10),
    ('example.md', 'ontology', 5),
    ('example.md', 'database', 3);
    "#;

    println!("\nGenerated SQL for file metadata:\n{}", file_sql);
    println!("\nGenerated SQL for topics:\n{}", topics_sql);
    println!("✓ Would insert file metadata into database\n");
}

// ================================================================
// QUERY EXAMPLES
// ================================================================

fn example_queries() {
    println!("--- Example Queries ---\n");

    println!("Query 1: Get physics settings for default profile");
    println!(r#"
SELECT damping, dt, max_velocity, rest_length
FROM physics_settings
WHERE profile_name = 'default';
    "#);

    println!("\nQuery 2: Find all files with specific ETSI domain");
    println!(r#"
SELECT fm.file_name, fm.node_id, ob.etsi_domain, ob.etsi_subdomain
FROM file_metadata fm
JOIN ontology_blocks ob ON fm.file_name = ob.file_name
WHERE ob.etsi_domain = 'NFV'
ORDER BY fm.file_name;
    "#);

    println!("\nQuery 3: Get most popular topics across all files");
    println!(r#"
SELECT topic, SUM(count) as total_count
FROM file_topics
GROUP BY topic
ORDER BY total_count DESC
LIMIT 10;
    "#);

    println!("\nQuery 4: Find validation violations for recent reports");
    println!(r#"
SELECT
    vr.report_id,
    vr.ontology_id,
    vr.completed_at,
    vv.severity,
    vv.violation_type,
    COUNT(*) as violation_count
FROM validation_reports vr
JOIN validation_violations vv ON vr.report_id = vv.report_id
WHERE vr.status = 'completed'
  AND vr.completed_at > datetime('now', '-7 days')
GROUP BY vr.report_id, vr.ontology_id, vv.severity, vv.violation_type
ORDER BY vr.completed_at DESC;
    "#);

    println!("\nQuery 5: Graph generation - nodes with metadata");
    println!(r#"
SELECT
    fm.node_id,
    fm.file_name,
    fm.node_size,
    fm.hyperlink_count,
    GROUP_CONCAT(ft.topic || ':' || ft.count) as topics
FROM file_metadata fm
LEFT JOIN file_topics ft ON fm.file_name = ft.file_name
GROUP BY fm.node_id, fm.file_name, fm.node_size, fm.hyperlink_count
ORDER BY fm.node_id;
    "#);

    println!("\nQuery 6: Get active constraints for physics simulation");
    println!(r#"
SELECT
    cg.group_name,
    cg.kernel_name,
    cg.physics_type,
    cg.default_strength,
    COUNT(oc.id) as constraint_count
FROM constraint_groups cg
LEFT JOIN ontology_constraints oc ON cg.id = oc.constraint_group_id
WHERE cg.enabled = 1 AND (oc.active = 1 OR oc.active IS NULL)
GROUP BY cg.id
ORDER BY cg.physics_type;
    "#);
}

// ================================================================
// TRANSACTION EXAMPLE
// ================================================================

fn transaction_example() {
    println!("--- Transaction Example: Atomic Ontology Load ---\n");

    println!("Transaction ensures all-or-nothing semantics:");
    println!(r#"
BEGIN TRANSACTION;

-- 1. Insert ontology metadata
INSERT INTO ontologies (ontology_id, source_path, source_type, base_iri, content_hash)
VALUES ('onto-123', '/path/to/ontology.owl', 'file', 'http://example.org/', 'hash123');

-- 2. Insert classes
INSERT INTO owl_classes (ontology_id, class_iri, label) VALUES
    ('onto-123', 'ex:Person', 'Person'),
    ('onto-123', 'ex:Company', 'Company');

-- 3. Insert properties
INSERT INTO owl_properties (ontology_id, property_iri, property_type, rdfs_domain, rdfs_range) VALUES
    ('onto-123', 'ex:employs', 'object', 'ex:Company', 'ex:Person');

-- 4. Insert disjoint relationships
INSERT INTO owl_disjoint_classes (ontology_id, class_iri_1, class_iri_2) VALUES
    ('onto-123', 'ex:Person', 'ex:Company');

COMMIT;
    "#);

    println!("✓ All inserts succeed or entire transaction rolls back\n");
}

// ================================================================
// INDEX USAGE EXAMPLE
// ================================================================

fn index_usage_example() {
    println!("--- Index Usage Example ---\n");

    println!("Explain query plan to verify index usage:");
    println!(r#"
EXPLAIN QUERY PLAN
SELECT * FROM file_metadata WHERE file_name = 'example.md';

-- Expected output:
-- SEARCH file_metadata USING INDEX idx_file_metadata_file_name (file_name=?)
    "#);

    println!("\nCompound index for complex queries:");
    println!(r#"
EXPLAIN QUERY PLAN
SELECT * FROM validation_violations
WHERE report_id = 'report-123' AND severity = 'error'
ORDER BY created_at DESC;

-- Expected:
-- Uses idx_validation_violations_report_id and filters on severity
    "#);
}
