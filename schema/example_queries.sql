-- ================================================================
-- Example INSERT and SELECT Queries for Ontology Database
-- ================================================================
-- This file demonstrates common database operations for the
-- ontology knowledge graph system.
-- ================================================================

-- ================================================================
-- EXAMPLE INSERTS
-- ================================================================

-- Insert a new physics settings profile
INSERT INTO physics_settings (
    profile_name, damping, dt, iterations, max_velocity, max_force,
    repel_k, spring_k, rest_length, repulsion_cutoff,
    repulsion_softening_epsilon, center_gravity_k, grid_cell_size,
    warmup_iterations, cooling_rate
) VALUES (
    'high_performance',  -- Custom profile for large graphs
    0.98,                -- Higher damping for stability
    0.008,               -- Smaller time step
    200,                 -- More iterations
    2.0,                 -- Higher max velocity
    150.0,               -- Higher max force
    75.0,                -- Stronger repulsion
    0.003,               -- Weaker springs
    75.0,                -- Larger rest length
    100.0,               -- Larger repulsion cutoff
    0.0001,
    0.001,
    75.0,                -- Larger grid cells
    150,
    0.002
);

-- Insert namespace prefixes
INSERT INTO namespaces (prefix, namespace_iri, is_default) VALUES
    ('owl', 'http://www.w3.org/2002/07/owl#', 0),
    ('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 0),
    ('rdfs', 'http://www.w3.org/2000/01/rdf-schema#', 0),
    ('xsd', 'http://www.w3.org/2001/XMLSchema#', 0),
    ('foaf', 'http://xmlns.com/foaf/0.1/', 0),
    ('dc', 'http://purl.org/dc/elements/1.1/', 0),
    ('ex', 'http://example.org/', 1);

-- Insert ontology metadata
INSERT INTO ontologies (
    ontology_id, source_path, source_type, base_iri, version_iri,
    title, description, author, version, content_hash,
    axiom_count, class_count, property_count
) VALUES (
    'corporate-ontology-v1',
    '/ontologies/corporate.owl',
    'file',
    'http://example.org/corporate#',
    'http://example.org/corporate/v1.0',
    'Corporate Knowledge Graph Ontology',
    'Formal ontology for corporate structure and relationships',
    'Ontology Team',
    '1.0',
    'blake3_hash_of_content_here',
    150,  -- Total axioms
    25,   -- Total classes
    40    -- Total properties
);

-- Insert OWL classes for the corporate ontology
INSERT INTO owl_classes (ontology_id, class_iri, label, comment, parent_class_iri) VALUES
    ('corporate-ontology-v1', 'ex:Person', 'Person', 'An individual human being', NULL),
    ('corporate-ontology-v1', 'ex:Employee', 'Employee', 'A person employed by a company', 'ex:Person'),
    ('corporate-ontology-v1', 'ex:Manager', 'Manager', 'An employee with management responsibilities', 'ex:Employee'),
    ('corporate-ontology-v1', 'ex:Executive', 'Executive', 'A senior management employee', 'ex:Manager'),
    ('corporate-ontology-v1', 'ex:Organization', 'Organization', 'A structured group of people', NULL),
    ('corporate-ontology-v1', 'ex:Company', 'Company', 'A commercial business organization', 'ex:Organization'),
    ('corporate-ontology-v1', 'ex:Department', 'Department', 'A division within a company', 'ex:Organization'),
    ('corporate-ontology-v1', 'ex:Project', 'Project', 'A temporary endeavor with specific goals', NULL),
    ('corporate-ontology-v1', 'ex:Document', 'Document', 'A written or digital record', NULL),
    ('corporate-ontology-v1', 'ex:Contract', 'Contract', 'A legal agreement', 'ex:Document');

-- Insert OWL properties
INSERT INTO owl_properties (
    ontology_id, property_iri, property_type, label, comment,
    rdfs_domain, rdfs_range, inverse_of_iri,
    is_functional, is_inverse_functional, is_transitive, is_symmetric
) VALUES
    ('corporate-ontology-v1', 'ex:employs', 'object', 'employs',
     'Relationship between company and employee',
     'ex:Company', 'ex:Employee', 'ex:employedBy', 0, 0, 0, 0),
    ('corporate-ontology-v1', 'ex:employedBy', 'object', 'employed by',
     'Relationship between employee and company',
     'ex:Employee', 'ex:Company', 'ex:employs', 1, 0, 0, 0),
    ('corporate-ontology-v1', 'ex:manages', 'object', 'manages',
     'Relationship between manager and employee',
     'ex:Manager', 'ex:Employee', 'ex:managedBy', 0, 0, 0, 0),
    ('corporate-ontology-v1', 'ex:partOf', 'object', 'part of',
     'Relationship between parts and wholes',
     'ex:Organization', 'ex:Organization', NULL, 0, 0, 1, 0),
    ('corporate-ontology-v1', 'ex:collaboratesWith', 'object', 'collaborates with',
     'Symmetric collaboration relationship',
     'ex:Person', 'ex:Person', NULL, 0, 0, 0, 1),
    ('corporate-ontology-v1', 'ex:name', 'data', 'name',
     'The name of an entity',
     'ex:Person', 'xsd:string', NULL, 1, 0, 0, 0),
    ('corporate-ontology-v1', 'ex:email', 'data', 'email',
     'Email address',
     'ex:Person', 'xsd:string', NULL, 0, 1, 0, 0),
    ('corporate-ontology-v1', 'ex:startDate', 'data', 'start date',
     'Date when something began',
     'ex:Employee', 'xsd:dateTime', NULL, 1, 0, 0, 0),
    ('corporate-ontology-v1', 'ex:salary', 'data', 'salary',
     'Annual salary amount',
     'ex:Employee', 'xsd:decimal', NULL, 1, 0, 0, 0),
    ('corporate-ontology-v1', 'ex:age', 'data', 'age',
     'Age in years',
     'ex:Person', 'xsd:nonNegativeInteger', NULL, 1, 0, 0, 0);

-- Insert disjoint class relationships
INSERT INTO owl_disjoint_classes (ontology_id, class_iri_1, class_iri_2) VALUES
    ('corporate-ontology-v1', 'ex:Person', 'ex:Organization'),
    ('corporate-ontology-v1', 'ex:Person', 'ex:Document'),
    ('corporate-ontology-v1', 'ex:Organization', 'ex:Document'),
    ('corporate-ontology-v1', 'ex:Company', 'ex:Department'),
    ('corporate-ontology-v1', 'ex:Project', 'ex:Organization');

-- Insert class mappings (graph labels to OWL classes)
INSERT INTO class_mappings (graph_label, owl_class_iri, rdfs_label, rdfs_comment) VALUES
    ('person', 'ex:Person', 'Person', 'Individual human'),
    ('employee', 'ex:Employee', 'Employee', 'Company employee'),
    ('manager', 'ex:Manager', 'Manager', 'Management role'),
    ('company', 'ex:Company', 'Company', 'Business entity'),
    ('department', 'ex:Department', 'Department', 'Company division'),
    ('project', 'ex:Project', 'Project', 'Work initiative'),
    ('document', 'ex:Document', 'Document', 'Written record');

-- Insert property mappings (graph edges to OWL properties)
INSERT INTO property_mappings (
    graph_property, owl_property_iri, property_type,
    rdfs_label, rdfs_domain, rdfs_range, inverse_property_iri
) VALUES
    ('employs', 'ex:employs', 'object', 'employs', 'ex:Company', 'ex:Employee', 'ex:employedBy'),
    ('employed_by', 'ex:employedBy', 'object', 'employed by', 'ex:Employee', 'ex:Company', 'ex:employs'),
    ('manages', 'ex:manages', 'object', 'manages', 'ex:Manager', 'ex:Employee', 'ex:managedBy'),
    ('works_on', 'ex:worksOn', 'object', 'works on', 'ex:Employee', 'ex:Project', NULL),
    ('collaborates', 'ex:collaboratesWith', 'object', 'collaborates with', 'ex:Person', 'ex:Person', NULL),
    ('name', 'ex:name', 'data', 'name', 'ex:Person', 'xsd:string', NULL),
    ('email', 'ex:email', 'data', 'email', 'ex:Person', 'xsd:string', NULL),
    ('start_date', 'ex:startDate', 'data', 'start date', 'ex:Employee', 'xsd:dateTime', NULL),
    ('salary', 'ex:salary', 'data', 'salary', 'ex:Employee', 'xsd:decimal', NULL);

-- Insert IRI templates
INSERT INTO iri_templates (entity_type, label_or_type, template, description) VALUES
    ('node', 'person', '{base_iri}person/{id}', 'IRI template for person nodes'),
    ('node', 'company', '{base_iri}company/{id}', 'IRI template for company nodes'),
    ('node', 'project', '{base_iri}project/{id}', 'IRI template for project nodes'),
    ('edge', 'employs', '{base_iri}employment/{source_id}-{target_id}', 'IRI template for employment edges'),
    ('edge', 'manages', '{base_iri}management/{source_id}-{target_id}', 'IRI template for management edges');

-- Insert constraint groups
INSERT INTO constraint_groups (
    group_name, kernel_name, physics_type, default_strength, enabled, batch_size
) VALUES
    ('disjoint_classes', 'apply_disjoint_classes_kernel', 'repulsion', 0.8, 1, 1000),
    ('subclass_hierarchy', 'apply_subclass_hierarchy_kernel', 'alignment', 0.6, 1, 1000),
    ('sameas_colocate', 'apply_sameas_colocate_kernel', 'attraction', 0.9, 1, 500),
    ('inverse_symmetry', 'apply_inverse_symmetry_kernel', 'symmetry', 0.7, 1, 500),
    ('functional_cardinality', 'apply_functional_cardinality_kernel', 'constraint', 1.0, 1, 1000);

-- Insert file metadata (sample markdown files)
INSERT INTO file_metadata (
    file_name, file_path, file_size, sha1, node_id, node_size,
    hyperlink_count, last_modified, change_count
) VALUES
    ('introduction.md', './markdown/introduction.md', 2048, 'sha1_intro', '1', 1.2, 5, '2025-10-15 10:00:00', 3),
    ('architecture.md', './markdown/architecture.md', 4096, 'sha1_arch', '2', 1.5, 12, '2025-10-16 14:30:00', 7),
    ('api-reference.md', './markdown/api-reference.md', 8192, 'sha1_api', '3', 2.0, 25, '2025-10-17 09:15:00', 15),
    ('deployment.md', './markdown/deployment.md', 3072, 'sha1_deploy', '4', 1.3, 8, '2025-10-10 16:45:00', 2),
    ('performance.md', './markdown/performance.md', 5120, 'sha1_perf', '5', 1.8, 18, '2025-10-14 11:20:00', 5);

-- Insert file topics
INSERT INTO file_topics (file_name, topic, count) VALUES
    ('introduction.md', 'overview', 10),
    ('introduction.md', 'getting-started', 8),
    ('introduction.md', 'concepts', 5),
    ('architecture.md', 'design', 20),
    ('architecture.md', 'components', 15),
    ('architecture.md', 'actors', 12),
    ('architecture.md', 'ontology', 8),
    ('api-reference.md', 'api', 50),
    ('api-reference.md', 'endpoints', 30),
    ('api-reference.md', 'validation', 12),
    ('deployment.md', 'docker', 10),
    ('deployment.md', 'kubernetes', 8),
    ('deployment.md', 'configuration', 15),
    ('performance.md', 'benchmarks', 20),
    ('performance.md', 'optimization', 18),
    ('performance.md', 'gpu', 15),
    ('performance.md', 'cuda', 10);

-- Insert ontology blocks (extracted from markdown frontmatter)
INSERT INTO ontology_blocks (
    file_name, block_type, etsi_domain, etsi_subdomain,
    ontology_class, raw_yaml
) VALUES
    ('architecture.md', 'etsi_domain', 'NFV', 'MANO', NULL,
     'etsi_domain: NFV\netsi_subdomain: MANO'),
    ('api-reference.md', 'ontology_definition', NULL, NULL, 'ex:APIEndpoint',
     'ontology_class: ex:APIEndpoint\nontology_property: ex:hasPath'),
    ('deployment.md', 'etsi_domain', '5G', 'Core', NULL,
     'etsi_domain: 5G\netsi_subdomain: Core');

-- Insert validation report
INSERT INTO validation_reports (
    report_id, ontology_id, graph_signature, validation_mode,
    status, violation_count, inference_count, duration_ms,
    started_at, completed_at
) VALUES (
    'report-2025-10-17-001',
    'corporate-ontology-v1',
    'blake3_graph_hash',
    'full',
    'completed',
    5,
    12,
    850,
    '2025-10-17 10:00:00',
    '2025-10-17 10:00:01'
);

-- Insert validation violations
INSERT INTO validation_violations (
    report_id, severity, violation_type, subject_iri, predicate_iri, object_iri,
    message, suggested_fix, fix_confidence
) VALUES
    ('report-2025-10-17-001', 'error', 'domain_violation',
     'ex:node123', 'ex:employs', 'ex:node456',
     'Subject ex:node123 is not of type ex:Company as required by domain of ex:employs',
     'Change type of ex:node123 to ex:Company or use different property',
     0.95),
    ('report-2025-10-17-001', 'warning', 'disjoint_violation',
     'ex:node789', 'rdf:type', 'ex:Person',
     'Individual ex:node789 is both ex:Person and ex:Company which are disjoint',
     'Remove one of the type assertions',
     0.85),
    ('report-2025-10-17-001', 'warning', 'cardinality_violation',
     'ex:employee42', 'ex:employedBy', 'ex:company2',
     'ex:employee42 has multiple values for functional property ex:employedBy',
     'Keep only one ex:employedBy relationship',
     0.90);

-- Insert inferred triples
INSERT INTO inferred_triples (
    report_id, subject_iri, predicate_iri, object_iri,
    is_literal, confidence, derivation_rule
) VALUES
    ('report-2025-10-17-001', 'ex:manager1', 'rdf:type', 'ex:Employee',
     0, 1.0, 'SubClassOf(ex:Manager, ex:Employee)'),
    ('report-2025-10-17-001', 'ex:manager1', 'rdf:type', 'ex:Person',
     0, 1.0, 'SubClassOf(ex:Employee, ex:Person)'),
    ('report-2025-10-17-001', 'ex:company1', 'ex:employs', 'ex:employee5',
     0, 0.95, 'InverseOf(ex:employedBy, ex:employs)');

-- ================================================================
-- EXAMPLE SELECT QUERIES
-- ================================================================

-- Query 1: Get physics settings for a profile
SELECT
    profile_name,
    damping,
    dt,
    max_velocity,
    rest_length,
    repulsion_cutoff,
    warmup_iterations
FROM physics_settings
WHERE profile_name = 'default';

-- Query 2: Get all classes in an ontology with their hierarchy
SELECT
    c1.class_iri,
    c1.label,
    c1.comment,
    c2.class_iri as parent_class,
    c2.label as parent_label
FROM owl_classes c1
LEFT JOIN owl_classes c2 ON c1.parent_class_iri = c2.class_iri
WHERE c1.ontology_id = 'corporate-ontology-v1'
ORDER BY c1.class_iri;

-- Query 3: Get all properties with their domains and ranges
SELECT
    property_iri,
    property_type,
    label,
    rdfs_domain,
    rdfs_range,
    inverse_of_iri,
    CASE WHEN is_functional = 1 THEN 'Functional' ELSE '' END as characteristics
FROM owl_properties
WHERE ontology_id = 'corporate-ontology-v1'
ORDER BY property_type, property_iri;

-- Query 4: Find all disjoint class pairs
SELECT
    dc.class_iri_1,
    c1.label as class_1_label,
    dc.class_iri_2,
    c2.label as class_2_label
FROM owl_disjoint_classes dc
JOIN owl_classes c1 ON dc.ontology_id = c1.ontology_id AND dc.class_iri_1 = c1.class_iri
JOIN owl_classes c2 ON dc.ontology_id = c2.ontology_id AND dc.class_iri_2 = c2.class_iri
WHERE dc.ontology_id = 'corporate-ontology-v1';

-- Query 5: Get namespace expansion for prefixed IRI
SELECT prefix || ':localname' as prefixed, namespace_iri || 'localname' as expanded
FROM namespaces
WHERE prefix = 'ex';

-- Query 6: Map graph label to OWL class
SELECT
    graph_label,
    owl_class_iri,
    rdfs_label,
    rdfs_comment
FROM class_mappings
WHERE graph_label IN ('person', 'employee', 'company');

-- Query 7: Get all file metadata with topic counts
SELECT
    fm.file_name,
    fm.node_id,
    fm.node_size,
    fm.hyperlink_count,
    fm.last_modified,
    GROUP_CONCAT(ft.topic || ':' || ft.count, ', ') as topics
FROM file_metadata fm
LEFT JOIN file_topics ft ON fm.file_name = ft.file_name
GROUP BY fm.file_name
ORDER BY fm.node_id;

-- Query 8: Find files by topic
SELECT
    fm.file_name,
    fm.node_id,
    ft.topic,
    ft.count
FROM file_metadata fm
JOIN file_topics ft ON fm.file_name = ft.file_name
WHERE ft.topic = 'ontology'
ORDER BY ft.count DESC;

-- Query 9: Get files with ETSI domain annotations
SELECT
    fm.file_name,
    fm.node_id,
    ob.etsi_domain,
    ob.etsi_subdomain,
    ob.block_type
FROM file_metadata fm
JOIN ontology_blocks ob ON fm.file_name = ob.file_name
WHERE ob.etsi_domain IS NOT NULL
ORDER BY ob.etsi_domain, ob.etsi_subdomain;

-- Query 10: Get active constraint groups by physics type
SELECT
    group_name,
    kernel_name,
    physics_type,
    default_strength,
    batch_size
FROM constraint_groups
WHERE enabled = 1 AND physics_type = 'repulsion'
ORDER BY default_strength DESC;

-- Query 11: Get validation report summary with violations
SELECT
    vr.report_id,
    vr.ontology_id,
    vr.validation_mode,
    vr.status,
    vr.violation_count,
    vr.inference_count,
    vr.duration_ms,
    COUNT(DISTINCT vv.id) as actual_violations
FROM validation_reports vr
LEFT JOIN validation_violations vv ON vr.report_id = vv.report_id
WHERE vr.status = 'completed'
GROUP BY vr.report_id
ORDER BY vr.completed_at DESC
LIMIT 10;

-- Query 12: Get violations by severity for a report
SELECT
    severity,
    violation_type,
    subject_iri,
    message,
    suggested_fix,
    fix_confidence
FROM validation_violations
WHERE report_id = 'report-2025-10-17-001'
ORDER BY
    CASE severity
        WHEN 'error' THEN 1
        WHEN 'warning' THEN 2
        WHEN 'info' THEN 3
    END,
    fix_confidence DESC;

-- Query 13: Get unapplied inferences with high confidence
SELECT
    subject_iri,
    predicate_iri,
    object_iri,
    confidence,
    derivation_rule
FROM inferred_triples
WHERE applied = 0 AND confidence > 0.9
ORDER BY confidence DESC
LIMIT 20;

-- Query 14: Graph generation query - nodes with all metadata
SELECT
    fm.node_id,
    fm.file_name,
    fm.node_size,
    fm.hyperlink_count,
    fm.last_modified,
    ob.etsi_domain,
    ob.ontology_class,
    GROUP_CONCAT(DISTINCT ft.topic) as all_topics
FROM file_metadata fm
LEFT JOIN file_topics ft ON fm.file_name = ft.file_name
LEFT JOIN ontology_blocks ob ON fm.file_name = ob.file_name
GROUP BY fm.node_id
ORDER BY fm.node_id;

-- Query 15: Performance - most recently modified files
SELECT
    file_name,
    node_id,
    file_size,
    last_modified,
    change_count
FROM file_metadata
ORDER BY last_modified DESC
LIMIT 100;

-- ================================================================
-- ANALYTICAL QUERIES
-- ================================================================

-- Analytics 1: Topic distribution across all files
SELECT
    topic,
    COUNT(DISTINCT file_name) as file_count,
    SUM(count) as total_mentions,
    AVG(count) as avg_mentions_per_file,
    MAX(count) as max_mentions
FROM file_topics
GROUP BY topic
ORDER BY total_mentions DESC;

-- Analytics 2: Ontology complexity metrics
SELECT
    ontology_id,
    title,
    axiom_count,
    class_count,
    property_count,
    CAST(axiom_count AS FLOAT) / NULLIF(class_count, 0) as axioms_per_class,
    CAST(property_count AS FLOAT) / NULLIF(class_count, 0) as properties_per_class
FROM ontologies
ORDER BY axiom_count DESC;

-- Analytics 3: Validation report success rate
SELECT
    ontology_id,
    COUNT(*) as total_validations,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    AVG(duration_ms) as avg_duration_ms,
    AVG(violation_count) as avg_violations,
    AVG(inference_count) as avg_inferences
FROM validation_reports
GROUP BY ontology_id;

-- Analytics 4: File metadata growth over time
SELECT
    DATE(last_modified) as date,
    COUNT(*) as files_modified,
    SUM(file_size) as total_size_modified
FROM file_metadata
WHERE last_modified > datetime('now', '-30 days')
GROUP BY DATE(last_modified)
ORDER BY date;

-- Analytics 5: Constraint group effectiveness
SELECT
    cg.group_name,
    cg.physics_type,
    cg.default_strength,
    COUNT(oc.id) as active_constraints
FROM constraint_groups cg
LEFT JOIN ontology_constraints oc ON cg.id = oc.constraint_group_id AND oc.active = 1
WHERE cg.enabled = 1
GROUP BY cg.id
ORDER BY active_constraints DESC;

-- ================================================================
-- MAINTENANCE QUERIES
-- ================================================================

-- Maintenance 1: Database size statistics
SELECT
    'Total Size' as metric,
    (page_count * page_size) / 1024.0 / 1024.0 as size_mb
FROM pragma_page_count(), pragma_page_size()
UNION ALL
SELECT
    'Page Size',
    page_size / 1024.0
FROM pragma_page_size();

-- Maintenance 2: Table row counts
SELECT
    'physics_settings' as table_name, COUNT(*) as row_count FROM physics_settings
UNION ALL SELECT 'ontologies', COUNT(*) FROM ontologies
UNION ALL SELECT 'owl_classes', COUNT(*) FROM owl_classes
UNION ALL SELECT 'owl_properties', COUNT(*) FROM owl_properties
UNION ALL SELECT 'namespaces', COUNT(*) FROM namespaces
UNION ALL SELECT 'class_mappings', COUNT(*) FROM class_mappings
UNION ALL SELECT 'property_mappings', COUNT(*) FROM property_mappings
UNION ALL SELECT 'file_metadata', COUNT(*) FROM file_metadata
UNION ALL SELECT 'file_topics', COUNT(*) FROM file_topics
UNION ALL SELECT 'ontology_blocks', COUNT(*) FROM ontology_blocks
UNION ALL SELECT 'constraint_groups', COUNT(*) FROM constraint_groups
UNION ALL SELECT 'validation_reports', COUNT(*) FROM validation_reports
ORDER BY row_count DESC;

-- Maintenance 3: Index usage statistics
SELECT * FROM sqlite_stat1 ORDER BY tbl, idx;

-- Maintenance 4: Integrity check
PRAGMA integrity_check;

-- Maintenance 5: Foreign key check
PRAGMA foreign_key_check;

-- ================================================================
-- END OF EXAMPLE QUERIES
-- ================================================================
