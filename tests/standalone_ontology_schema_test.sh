#!/bin/bash
# Standalone test for ontology schema fixes
# Tests the database directly using sqlite3

set -e

echo "=== Testing Ontology Schema Fixes ==="
echo ""

# Create a temporary test database
TEST_DB=$(mktemp /tmp/ontology_test.XXXXXX.db)
trap "rm -f $TEST_DB" EXIT

echo "Test database: $TEST_DB"
echo ""

# Apply the schema from our Rust code (simulated)
echo "Step 1: Creating schema..."
sqlite3 "$TEST_DB" <<'EOF'
-- Create ontologies table
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

-- Insert default ontology
INSERT INTO ontologies (
    ontology_id,
    source_path,
    source_type,
    content_hash,
    title,
    description
)
VALUES (
    'default',
    'default',
    'embedded',
    'default-ontology',
    'Default Ontology',
    'Default ontology for VisionFlow incremental saves'
);

-- Create owl_classes table with composite primary key
CREATE TABLE owl_classes (
    ontology_id TEXT NOT NULL,
    class_iri TEXT NOT NULL,
    label TEXT,
    comment TEXT,
    parent_class_iri TEXT,
    is_deprecated INTEGER DEFAULT 0 CHECK (is_deprecated IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_sha1 TEXT,
    PRIMARY KEY (ontology_id, class_iri),
    FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
);

-- CRITICAL: Create unique index to allow foreign key references
CREATE UNIQUE INDEX idx_owl_classes_iri_unique ON owl_classes(class_iri);
CREATE INDEX idx_owl_classes_parent ON owl_classes(parent_class_iri);
CREATE INDEX idx_owl_classes_label ON owl_classes(label);
CREATE INDEX idx_owl_classes_sha1 ON owl_classes(file_sha1);

-- Create owl_class_hierarchy with correct foreign keys
CREATE TABLE owl_class_hierarchy (
    class_iri TEXT NOT NULL,
    parent_iri TEXT NOT NULL,
    PRIMARY KEY (class_iri, parent_iri),
    FOREIGN KEY (class_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE,
    FOREIGN KEY (parent_iri) REFERENCES owl_classes(class_iri) ON DELETE CASCADE
);

-- Create owl_properties table
CREATE TABLE owl_properties (
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

CREATE INDEX idx_owl_properties_iri ON owl_properties(property_iri);
CREATE INDEX idx_owl_properties_type ON owl_properties(property_type);

-- Enable foreign keys
PRAGMA foreign_keys = ON;
EOF

echo "✅ Schema created successfully"
echo ""

# Test 2: Insert a class with correct column names
echo "Step 2: Testing INSERT with correct column names..."
sqlite3 "$TEST_DB" <<'EOF'
PRAGMA foreign_keys = ON;

INSERT INTO owl_classes (ontology_id, class_iri, label, comment, file_sha1)
VALUES ('default', 'http://example.org/TestClass', 'Test Class', 'A test class', 'abc123');

INSERT INTO owl_classes (ontology_id, class_iri, label, comment, file_sha1)
VALUES ('default', 'http://example.org/ParentClass', 'Parent Class', 'A parent class', 'def456');
EOF

echo "✅ INSERT operations successful"
echo ""

# Test 3: Insert hierarchy with correct foreign keys
echo "Step 3: Testing hierarchy INSERT..."
sqlite3 "$TEST_DB" <<'EOF'
PRAGMA foreign_keys = ON;

INSERT INTO owl_class_hierarchy (class_iri, parent_iri)
VALUES ('http://example.org/TestClass', 'http://example.org/ParentClass');
EOF

echo "✅ Hierarchy INSERT successful"
echo ""

# Test 4: Test foreign key constraints work
echo "Step 4: Testing foreign key constraints..."
if sqlite3 "$TEST_DB" "PRAGMA foreign_keys = ON; INSERT INTO owl_class_hierarchy (class_iri, parent_iri) VALUES ('http://example.org/NonExistent', 'http://example.org/TestClass');" 2>&1 | grep -q "FOREIGN KEY"; then
    echo "✅ Foreign key constraints working correctly (rejected invalid insert)"
else
    echo "❌ Foreign key constraints NOT working! (allowed invalid insert)"
    exit 1
fi
echo ""

# Test 5: Verify foreign key check passes
echo "Step 5: Running foreign key check..."
FK_CHECK=$(sqlite3 "$TEST_DB" "PRAGMA foreign_key_check;")
if [ -z "$FK_CHECK" ]; then
    echo "✅ Foreign key check passed (no violations)"
else
    echo "❌ Foreign key violations found:"
    echo "$FK_CHECK"
    exit 1
fi
echo ""

# Test 6: Query the data
echo "Step 6: Querying inserted data..."
sqlite3 "$TEST_DB" <<'EOF'
.mode column
.headers on
SELECT class_iri, label, comment FROM owl_classes;
EOF
echo ""

sqlite3 "$TEST_DB" <<'EOF'
.mode column
.headers on
SELECT * FROM owl_class_hierarchy;
EOF
echo ""

# Test 7: Insert property
echo "Step 7: Testing property INSERT..."
sqlite3 "$TEST_DB" <<'EOF'
PRAGMA foreign_keys = ON;

INSERT INTO owl_properties (ontology_id, property_iri, property_type, label)
VALUES ('default', 'http://example.org/testProperty', 'ObjectProperty', 'Test Property');
EOF

echo "✅ Property INSERT successful"
echo ""

# Summary
echo "=== All Schema Tests Passed! ==="
echo ""
echo "Summary:"
sqlite3 "$TEST_DB" "SELECT 'Classes: ' || COUNT(*) FROM owl_classes;"
sqlite3 "$TEST_DB" "SELECT 'Hierarchies: ' || COUNT(*) FROM owl_class_hierarchy;"
sqlite3 "$TEST_DB" "SELECT 'Properties: ' || COUNT(*) FROM owl_properties;"
echo ""
echo "✅ Schema is correct and all operations work properly"
