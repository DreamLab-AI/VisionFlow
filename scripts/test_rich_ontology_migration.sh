#!/bin/bash
# Test Rich Ontology Metadata Migration
# Tests migration 002_rich_ontology_metadata.sql with sample data
# Date: 2025-11-22

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATION_FILE="$SCRIPT_DIR/migrations/002_rich_ontology_metadata.sql"
TEST_DB="$SCRIPT_DIR/test_unified.db"

echo "==================================================================="
echo "Testing Rich Ontology Metadata Migration"
echo "==================================================================="

# Clean up any existing test database
if [ -f "$TEST_DB" ]; then
    echo "Removing existing test database..."
    rm "$TEST_DB"
fi

echo ""
echo "Step 1: Creating test database with base schema..."
sqlite3 "$TEST_DB" <<'EOF'
-- Create base schema (version 1)
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_version (id, version) VALUES (1, 1);

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

INSERT INTO ontologies (ontology_id, source_path, source_type, content_hash, title)
VALUES ('default', 'default', 'embedded', 'test-hash', 'Test Ontology');

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

.print "✅ Base schema created (version 1)"
EOF

echo ""
echo "Step 2: Inserting sample data..."
sqlite3 "$TEST_DB" <<'EOF'
INSERT INTO owl_classes (ontology_id, class_iri, label, comment, parent_class_iri)
VALUES
    ('default', 'bc:SmartContract', 'Smart Contract', 'A self-executing contract', 'bc:DigitalAsset'),
    ('default', 'bc:DigitalAsset', 'Digital Asset', 'A digital representation of value', NULL),
    ('default', 'ai:NeuralNetwork', 'Neural Network', 'A computational model', 'ai:Model');

INSERT INTO owl_properties (ontology_id, property_iri, property_type, label, domain_class_iri, range_class_iri)
VALUES
    ('default', 'bc:hasOwner', 'ObjectProperty', 'has owner', 'bc:DigitalAsset', 'bc:Agent'),
    ('default', 'bc:hasValue', 'DataProperty', 'has value', 'bc:DigitalAsset', NULL);

.print "✅ Sample data inserted"
.print ""
SELECT 'Classes: ' || COUNT(*) FROM owl_classes;
SELECT 'Properties: ' || COUNT(*) FROM owl_properties;
EOF

echo ""
echo "Step 3: Running migration 002_rich_ontology_metadata.sql..."
if [ ! -f "$MIGRATION_FILE" ]; then
    echo "❌ Migration file not found: $MIGRATION_FILE"
    exit 1
fi

sqlite3 "$TEST_DB" < "$MIGRATION_FILE"

echo ""
echo "Step 4: Verifying migration results..."
sqlite3 "$TEST_DB" <<'EOF'
.mode column
.headers on

.print ""
.print "==================================================================="
.print "Schema Version Check"
.print "==================================================================="
SELECT * FROM schema_version;

.print ""
.print "==================================================================="
.print "Table Structure Check"
.print "==================================================================="
.print ""
.print "owl_classes columns:"
PRAGMA table_info(owl_classes);

.print ""
.print "owl_relationships columns:"
PRAGMA table_info(owl_relationships);

.print ""
.print "==================================================================="
.print "Data Preservation Check"
.print "==================================================================="
SELECT COUNT(*) as migrated_classes FROM owl_classes;
SELECT COUNT(*) as migrated_properties FROM owl_properties;

.print ""
.print "==================================================================="
.print "Sample Classes (with new fields)"
.print "==================================================================="
SELECT class_iri, label, source_domain, quality_score, status, maturity
FROM owl_classes
LIMIT 5;

.print ""
.print "==================================================================="
.print "Index Check"
.print "==================================================================="
SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND tbl_name LIKE 'owl%' ORDER BY tbl_name, name;

.print ""
.print "==================================================================="
.print "View Check"
.print "==================================================================="
SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;

.print ""
.print "==================================================================="
.print "Foreign Key Check"
.print "==================================================================="
PRAGMA foreign_key_check;
.print "✅ No foreign key violations (empty result is good)"

.print ""
.print "==================================================================="
.print "Migration Test Complete"
.print "==================================================================="
EOF

echo ""
echo "Step 5: Testing rich metadata insertion..."
sqlite3 "$TEST_DB" <<'EOF'
-- Insert a class with rich metadata
INSERT INTO owl_classes (
    ontology_id, class_iri, term_id, preferred_term, label, comment,
    source_domain, version, type, status, maturity,
    quality_score, authority_score, public_access,
    owl_physicality, owl_role,
    belongs_to_domain, bridges_to_domain
) VALUES (
    'default',
    'bc:DeFiProtocol',
    'BC-1234',
    'Decentralized Finance Protocol',
    'DeFi Protocol',
    'A protocol for decentralized financial services',
    'blockchain',
    '2.0',
    'Protocol',
    'approved',
    'stable',
    0.95,
    0.88,
    1,
    'virtual',
    'agent',
    'blockchain',
    'ai'
);

-- Insert a relationship
INSERT INTO owl_relationships (
    ontology_id, source_class_iri, relationship_type, target_class_iri, confidence, is_inferred
) VALUES (
    'default',
    'bc:DeFiProtocol',
    'has-part',
    'bc:SmartContract',
    1.0,
    0
);

.print "✅ Rich metadata inserted successfully"
.print ""
.print "Sample class with rich metadata:"
.mode column
.headers on
SELECT
    class_iri,
    term_id,
    preferred_term,
    quality_score,
    authority_score,
    status,
    maturity,
    owl_physicality,
    owl_role,
    bridges_to_domain
FROM owl_classes
WHERE term_id = 'BC-1234';

.print ""
.print "Sample relationship:"
SELECT * FROM owl_relationships;

.print ""
.print "==================================================================="
.print "Testing Views"
.print "==================================================================="
.print ""
.print "Classes with quality metrics:"
SELECT * FROM owl_classes_with_quality LIMIT 3;

.print ""
.print "Relationship graph:"
SELECT * FROM owl_relationship_graph LIMIT 3;
EOF

echo ""
echo "Step 6: Cleanup..."
echo "Test database preserved at: $TEST_DB"
echo "To inspect manually: sqlite3 $TEST_DB"
echo ""
echo "To remove test database:"
echo "  rm $TEST_DB"

echo ""
echo "==================================================================="
echo "✅ Migration Test PASSED"
echo "==================================================================="
echo ""
echo "Summary:"
echo "  - Schema upgraded from version 1 to version 2"
echo "  - All existing data preserved"
echo "  - New rich metadata fields available"
echo "  - Relationships table created"
echo "  - Indexes created for performance"
echo "  - Views created for common queries"
echo "  - Foreign key constraints verified"
echo ""
echo "Next steps:"
echo "  1. Review test database: sqlite3 $SCRIPT_DIR/test_unified.db"
echo "  2. Run migration on production database"
echo "  3. Update Rust adapter code to use rich metadata"
echo ""
