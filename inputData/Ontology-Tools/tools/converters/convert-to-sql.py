#!/usr/bin/env python3
"""
Convert Logseq Ontology Blocks to PostgreSQL SQL Schema
========================================================

Parses Logseq markdown files with OntologyBlock sections and generates a complete
PostgreSQL database schema with:
- concepts table: All concepts with full IRIs and metadata
- properties table: All properties and their values
- relationships table: All relationships between concepts
- domains table: Domain classification lookup

Features:
- Foreign keys for referential integrity
- Indexes on IRIs and term-ids for performance
- INSERT statements for all parsed data
- Support for all 6 domains: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies

Uses the shared ontology_block_parser library for consistent parsing.

Usage:
    python convert-to-sql.py --input <pages_dir> --output <schema.sql>
    python convert-to-sql.py --input mainKnowledgeGraph/pages --output ontology-schema.sql

    # Import to PostgreSQL:
    psql -d database_name -f ontology-schema.sql

Requirements:
    - Logseq pages directory with OntologyBlock sections
    - ontology_block_parser library (lib/ontology_block_parser.py)
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Any

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG


def sanitize_sql(value: Any) -> str:
    """
    Sanitize value for SQL.
    Escapes single quotes and handles None values.
    """
    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        value = "; ".join(str(v) for v in value)

    # Escape single quotes for SQL
    value_str = str(value).replace("'", "''").replace('\n', ' ').replace('\r', '')
    return f"'{value_str}'"


def write_schema_header(f):
    """Write SQL schema header."""
    f.write("-- ============================================================================\n")
    f.write("-- Logseq Ontology PostgreSQL Schema\n")
    f.write("-- Auto-generated from Logseq markdown files with OntologyBlock sections\n")
    f.write("-- ============================================================================\n")
    f.write("-- \n")
    f.write("-- Supports 6 domains:\n")
    for domain, config in DOMAIN_CONFIG.items():
        f.write(f"--   - {config['full_name']} ({domain})\n")
    f.write("-- \n")
    f.write("-- Import with: psql -d database_name -f ontology-schema.sql\n")
    f.write("-- ============================================================================\n\n")


def write_drop_tables(f):
    """Drop existing tables."""
    f.write("-- Drop existing tables (CASCADE to handle foreign keys)\n")
    f.write("DROP TABLE IF EXISTS relationships CASCADE;\n")
    f.write("DROP TABLE IF EXISTS properties CASCADE;\n")
    f.write("DROP TABLE IF EXISTS concepts CASCADE;\n")
    f.write("DROP TABLE IF EXISTS domains CASCADE;\n\n")


def write_domains_table(f):
    """Create domains table."""
    f.write("-- ============================================================================\n")
    f.write("-- Domains Table: Classification of ontology domains\n")
    f.write("-- ============================================================================\n\n")

    f.write("CREATE TABLE domains (\n")
    f.write("    domain_code VARCHAR(10) PRIMARY KEY,\n")
    f.write("    domain_name VARCHAR(100) NOT NULL,\n")
    f.write("    namespace VARCHAR(500) NOT NULL,\n")
    f.write("    prefix VARCHAR(10) NOT NULL,\n")
    f.write("    description TEXT\n")
    f.write(");\n\n")

    f.write("-- Insert domain definitions\n")
    for domain, config in DOMAIN_CONFIG.items():
        f.write(f"INSERT INTO domains (domain_code, domain_name, namespace, prefix) VALUES\n")
        f.write(f"    ('{domain}', '{config['full_name']}', '{config['namespace']}', '{config['prefix']}');\n")
    f.write("\n")


def write_concepts_table(f):
    """Create concepts table."""
    f.write("-- ============================================================================\n")
    f.write("-- Concepts Table: All ontology concepts with metadata\n")
    f.write("-- ============================================================================\n\n")

    f.write("CREATE TABLE concepts (\n")
    f.write("    term_id VARCHAR(100) PRIMARY KEY,\n")
    f.write("    iri TEXT NOT NULL,\n")
    f.write("    preferred_term VARCHAR(500) NOT NULL,\n")
    f.write("    definition TEXT,\n")
    f.write("    status VARCHAR(50),\n")
    f.write("    source_domain VARCHAR(10),\n")
    f.write("    physicality VARCHAR(50),\n")
    f.write("    role VARCHAR(50),\n")
    f.write("    maturity VARCHAR(50),\n")
    f.write("    quality_score DECIMAL(3,2),\n")
    f.write("    authority_score DECIMAL(3,2),\n")
    f.write("    public_access BOOLEAN,\n")
    f.write("    last_updated DATE,\n")
    f.write("    version VARCHAR(20),\n")
    f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n")
    f.write("    FOREIGN KEY (source_domain) REFERENCES domains(domain_code)\n")
    f.write(");\n\n")

    # Create indexes
    f.write("-- Indexes for performance\n")
    f.write("CREATE INDEX idx_concepts_iri ON concepts(iri);\n")
    f.write("CREATE INDEX idx_concepts_domain ON concepts(source_domain);\n")
    f.write("CREATE INDEX idx_concepts_status ON concepts(status);\n")
    f.write("CREATE INDEX idx_concepts_physicality ON concepts(physicality);\n")
    f.write("CREATE INDEX idx_concepts_role ON concepts(role);\n\n")


def write_properties_table(f):
    """Create properties table."""
    f.write("-- ============================================================================\n")
    f.write("-- Properties Table: All properties from all concepts\n")
    f.write("-- ============================================================================\n\n")

    f.write("CREATE TABLE properties (\n")
    f.write("    id SERIAL PRIMARY KEY,\n")
    f.write("    term_id VARCHAR(100) NOT NULL,\n")
    f.write("    property_name VARCHAR(100) NOT NULL,\n")
    f.write("    property_value TEXT,\n")
    f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n")
    f.write("    FOREIGN KEY (term_id) REFERENCES concepts(term_id) ON DELETE CASCADE\n")
    f.write(");\n\n")

    # Create indexes
    f.write("-- Indexes for performance\n")
    f.write("CREATE INDEX idx_properties_term_id ON properties(term_id);\n")
    f.write("CREATE INDEX idx_properties_name ON properties(property_name);\n\n")


def write_relationships_table(f):
    """Create relationships table."""
    f.write("-- ============================================================================\n")
    f.write("-- Relationships Table: All relationships between concepts\n")
    f.write("-- ============================================================================\n\n")

    f.write("CREATE TABLE relationships (\n")
    f.write("    id SERIAL PRIMARY KEY,\n")
    f.write("    source_term_id VARCHAR(100) NOT NULL,\n")
    f.write("    relationship_type VARCHAR(100) NOT NULL,\n")
    f.write("    target_term TEXT NOT NULL,\n")
    f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n")
    f.write("    FOREIGN KEY (source_term_id) REFERENCES concepts(term_id) ON DELETE CASCADE\n")
    f.write(");\n\n")

    # Create indexes
    f.write("-- Indexes for performance\n")
    f.write("CREATE INDEX idx_relationships_source ON relationships(source_term_id);\n")
    f.write("CREATE INDEX idx_relationships_type ON relationships(relationship_type);\n")
    f.write("CREATE INDEX idx_relationships_target ON relationships(target_term);\n\n")


def write_concepts_data(blocks: List[OntologyBlock], f) -> int:
    """Write INSERT statements for concepts."""
    f.write("-- ============================================================================\n")
    f.write("-- Insert Concepts Data\n")
    f.write("-- ============================================================================\n\n")

    count = 0
    for block in blocks:
        quality_score = sanitize_sql(block.quality_score)
        authority_score = sanitize_sql(block.authority_score)
        public_access = sanitize_sql(block.public_access)

        f.write("INSERT INTO concepts (\n")
        f.write("    term_id, iri, preferred_term, definition, status,\n")
        f.write("    source_domain, physicality, role, maturity,\n")
        f.write("    quality_score, authority_score, public_access,\n")
        f.write("    last_updated, version\n")
        f.write(") VALUES (\n")
        f.write(f"    {sanitize_sql(block.term_id)},\n")
        f.write(f"    {sanitize_sql(block.get_full_iri())},\n")
        f.write(f"    {sanitize_sql(block.preferred_term)},\n")
        f.write(f"    {sanitize_sql(block.definition)},\n")
        f.write(f"    {sanitize_sql(block.status)},\n")
        f.write(f"    {sanitize_sql(block.source_domain)},\n")
        f.write(f"    {sanitize_sql(block.owl_physicality)},\n")
        f.write(f"    {sanitize_sql(block.owl_role)},\n")
        f.write(f"    {sanitize_sql(block.maturity)},\n")
        f.write(f"    {quality_score},\n")
        f.write(f"    {authority_score},\n")
        f.write(f"    {public_access},\n")
        f.write(f"    {sanitize_sql(block.last_updated)},\n")
        f.write(f"    {sanitize_sql(block.version)}\n")
        f.write(");\n\n")
        count += 1

    return count


def write_properties_data(blocks: List[OntologyBlock], f) -> int:
    """Write INSERT statements for properties."""
    f.write("-- ============================================================================\n")
    f.write("-- Insert Properties Data\n")
    f.write("-- ============================================================================\n\n")

    count = 0
    for block in blocks:
        term_id = sanitize_sql(block.term_id)

        # Alt terms
        for alt_term in block.alt_terms:
            f.write(f"INSERT INTO properties (term_id, property_name, property_value) VALUES\n")
            f.write(f"    ({term_id}, 'alt-term', {sanitize_sql(alt_term)});\n")
            count += 1

        # Sources
        for source in block.source:
            f.write(f"INSERT INTO properties (term_id, property_name, property_value) VALUES\n")
            f.write(f"    ({term_id}, 'source', {sanitize_sql(source)});\n")
            count += 1

        # Belongs to domain
        for domain in block.belongs_to_domain:
            f.write(f"INSERT INTO properties (term_id, property_name, property_value) VALUES\n")
            f.write(f"    ({term_id}, 'belongs-to-domain', {sanitize_sql(domain)});\n")
            count += 1

        # Implemented in layer
        for layer in block.implemented_in_layer:
            f.write(f"INSERT INTO properties (term_id, property_name, property_value) VALUES\n")
            f.write(f"    ({term_id}, 'implemented-in-layer', {sanitize_sql(layer)});\n")
            count += 1

        # Scope note
        if block.scope_note:
            f.write(f"INSERT INTO properties (term_id, property_name, property_value) VALUES\n")
            f.write(f"    ({term_id}, 'scope-note', {sanitize_sql(block.scope_note)});\n")
            count += 1

        # Domain-specific extensions
        for ext_name, ext_value in block.domain_extensions.items():
            f.write(f"INSERT INTO properties (term_id, property_name, property_value) VALUES\n")
            f.write(f"    ({term_id}, {sanitize_sql(ext_name)}, {sanitize_sql(ext_value)});\n")
            count += 1

    if count > 0:
        f.write("\n")

    return count


def write_relationships_data(blocks: List[OntologyBlock], f) -> int:
    """Write INSERT statements for relationships."""
    f.write("-- ============================================================================\n")
    f.write("-- Insert Relationships Data\n")
    f.write("-- ============================================================================\n\n")

    count = 0
    for block in blocks:
        term_id = sanitize_sql(block.term_id)

        # Standard relationships
        for parent in block.is_subclass_of:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'is-subclass-of', {sanitize_sql(parent)});\n")
            count += 1

        for part in block.has_part:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'has-part', {sanitize_sql(part)});\n")
            count += 1

        for whole in block.is_part_of:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'is-part-of', {sanitize_sql(whole)});\n")
            count += 1

        for req in block.requires:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'requires', {sanitize_sql(req)});\n")
            count += 1

        for dep in block.depends_on:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'depends-on', {sanitize_sql(dep)});\n")
            count += 1

        for enabled in block.enables:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'enables', {sanitize_sql(enabled)});\n")
            count += 1

        for related in block.relates_to:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'relates-to', {sanitize_sql(related)});\n")
            count += 1

        # Other relationships
        for rel_type, targets in block.other_relationships.items():
            for target in targets:
                f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
                f.write(f"    ({term_id}, {sanitize_sql(rel_type)}, {sanitize_sql(target)});\n")
                count += 1

        # Cross-domain bridges
        for bridge in block.bridges_to:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'bridges-to', {sanitize_sql(bridge)});\n")
            count += 1

        for bridge in block.bridges_from:
            f.write(f"INSERT INTO relationships (source_term_id, relationship_type, target_term) VALUES\n")
            f.write(f"    ({term_id}, 'bridges-from', {sanitize_sql(bridge)});\n")
            count += 1

    if count > 0:
        f.write("\n")

    return count


def write_useful_views(f):
    """Write useful database views."""
    f.write("-- ============================================================================\n")
    f.write("-- Useful Views for Querying\n")
    f.write("-- ============================================================================\n\n")

    # View: Concept hierarchy with labels
    f.write("-- View: Concept hierarchy with readable labels\n")
    f.write("CREATE VIEW v_concept_hierarchy AS\n")
    f.write("SELECT \n")
    f.write("    c.term_id AS concept_id,\n")
    f.write("    c.preferred_term AS concept_term,\n")
    f.write("    r.target_term AS parent_term,\n")
    f.write("    c.source_domain,\n")
    f.write("    d.domain_name\n")
    f.write("FROM concepts c\n")
    f.write("LEFT JOIN relationships r ON c.term_id = r.source_term_id AND r.relationship_type = 'is-subclass-of'\n")
    f.write("LEFT JOIN domains d ON c.source_domain = d.domain_code;\n\n")

    # View: Concept with all properties
    f.write("-- View: Concepts with aggregated properties\n")
    f.write("CREATE VIEW v_concept_properties AS\n")
    f.write("SELECT \n")
    f.write("    c.term_id,\n")
    f.write("    c.preferred_term,\n")
    f.write("    c.source_domain,\n")
    f.write("    COUNT(DISTINCT p.id) AS property_count,\n")
    f.write("    COUNT(DISTINCT r.id) AS relationship_count\n")
    f.write("FROM concepts c\n")
    f.write("LEFT JOIN properties p ON c.term_id = p.term_id\n")
    f.write("LEFT JOIN relationships r ON c.term_id = r.source_term_id\n")
    f.write("GROUP BY c.term_id, c.preferred_term, c.source_domain;\n\n")

    # View: Domain statistics
    f.write("-- View: Domain statistics\n")
    f.write("CREATE VIEW v_domain_statistics AS\n")
    f.write("SELECT \n")
    f.write("    d.domain_code,\n")
    f.write("    d.domain_name,\n")
    f.write("    COUNT(DISTINCT c.term_id) AS concept_count,\n")
    f.write("    AVG(c.quality_score) AS avg_quality_score,\n")
    f.write("    COUNT(DISTINCT r.id) AS relationship_count\n")
    f.write("FROM domains d\n")
    f.write("LEFT JOIN concepts c ON d.domain_code = c.source_domain\n")
    f.write("LEFT JOIN relationships r ON c.term_id = r.source_term_id\n")
    f.write("GROUP BY d.domain_code, d.domain_name;\n\n")


def write_example_queries(f):
    """Write example SQL queries as comments."""
    f.write("-- ============================================================================\n")
    f.write("-- Example Queries\n")
    f.write("-- ============================================================================\n\n")

    f.write("-- Find all concepts in AI domain\n")
    f.write("-- SELECT * FROM concepts WHERE source_domain = 'ai';\n\n")

    f.write("-- Find all subclasses of a specific concept\n")
    f.write("-- SELECT c.* FROM concepts c\n")
    f.write("-- JOIN relationships r ON c.term_id = r.source_term_id\n")
    f.write("-- WHERE r.relationship_type = 'is-subclass-of' AND r.target_term = 'Artificial Intelligence';\n\n")

    f.write("-- Find concepts with high quality scores\n")
    f.write("-- SELECT * FROM concepts WHERE quality_score >= 0.8 ORDER BY quality_score DESC;\n\n")

    f.write("-- Count concepts by domain\n")
    f.write("-- SELECT * FROM v_domain_statistics ORDER BY concept_count DESC;\n\n")

    f.write("-- Find all properties for a specific concept\n")
    f.write("-- SELECT * FROM properties WHERE term_id = 'AI-0850';\n\n")

    f.write("-- Find cross-domain bridges\n")
    f.write("-- SELECT * FROM relationships WHERE relationship_type IN ('bridges-to', 'bridges-from');\n\n")


def export_to_sql(blocks: List[OntologyBlock], output_file: Path) -> dict:
    """
    Export all ontology blocks to SQL schema file.

    Args:
        blocks: List of parsed OntologyBlock objects
        output_file: Path to output SQL file

    Returns:
        Dictionary with statistics
    """
    stats = {}

    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        write_schema_header(f)

        # Drop existing tables
        write_drop_tables(f)

        # Create tables
        write_domains_table(f)
        write_concepts_table(f)
        write_properties_table(f)
        write_relationships_table(f)

        # Insert data
        print("\nGenerating SQL INSERT statements:")
        print("-" * 80)

        stats['concepts'] = write_concepts_data(blocks, f)
        print(f"✓ Concepts: {stats['concepts']} records")

        stats['properties'] = write_properties_data(blocks, f)
        print(f"✓ Properties: {stats['properties']} records")

        stats['relationships'] = write_relationships_data(blocks, f)
        print(f"✓ Relationships: {stats['relationships']} records")

        # Create views
        write_useful_views(f)

        # Add example queries
        write_example_queries(f)

    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Convert Logseq Ontology Blocks to PostgreSQL SQL schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input pages --output ontology-schema.sql
  %(prog)s --input mainKnowledgeGraph/pages --output /tmp/ontology.sql

Import to PostgreSQL:
  psql -d database_name -f ontology-schema.sql
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        type=Path,
        help='Input directory containing Logseq markdown files'
    )

    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        help='Output SQL schema file'
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.is_dir():
        print(f"Error: Input path is not a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("Logseq Ontology to PostgreSQL SQL Converter")
    print("=" * 80)
    print(f"Input directory: {args.input}")
    print(f"Output file: {args.output}")

    # Parse all ontology blocks
    print("\nParsing ontology blocks...")
    parser_obj = OntologyBlockParser()
    blocks = parser_obj.parse_directory(args.input)

    if not blocks:
        print("\n⚠️  No ontology blocks found in input directory!")
        sys.exit(1)

    print(f"✓ Found {len(blocks)} ontology blocks")

    # Statistics by domain
    domain_counts = defaultdict(int)
    for block in blocks:
        domain = block.get_domain()
        if domain:
            domain_counts[domain] += 1

    print("\nConcepts by domain:")
    for domain, count in sorted(domain_counts.items()):
        domain_name = DOMAIN_CONFIG[domain]['full_name'] if domain in DOMAIN_CONFIG else domain
        print(f"  {domain_name}: {count}")

    # Export to SQL
    stats = export_to_sql(blocks, args.output)

    # Summary
    print("\n" + "=" * 80)
    print("Export Summary:")
    print("-" * 80)
    for table, count in stats.items():
        print(f"  {table}: {count} records")

    # File size
    file_size = args.output.stat().st_size
    print(f"\nSQL schema file saved: {args.output}")
    print(f"File size: {file_size / 1024:.2f} KB")
    print(f"\nImport with: psql -d database_name -f {args.output.name}")
    print("=" * 80)


if __name__ == '__main__':
    main()
