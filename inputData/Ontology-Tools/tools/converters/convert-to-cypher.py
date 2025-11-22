#!/usr/bin/env python3
"""
Convert Ontology Blocks to Neo4j Cypher format.
Creates property graph with nodes for concepts and relationships.
Supports all 6 domains: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies.

Usage:
    python convert-to-cypher.py --input mainKnowledgeGraph/pages --output output/ontology.cypher
    python convert-to-cypher.py --input mainKnowledgeGraph/pages/rb-0001-robot.md --output output/robot.cypher

Then import to Neo4j:
    cypher-shell -u neo4j -p [password] < output/ontology.cypher
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG


def sanitize_string(text):
    """Sanitize string for Cypher (escape quotes and backslashes)."""
    if not text:
        return ""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', '\\n')


def get_domain_label(block: OntologyBlock) -> str:
    """Get domain label for node."""
    domain = block.get_domain()
    if domain and domain in DOMAIN_CONFIG:
        return DOMAIN_CONFIG[domain]['full_name'].replace(' ', '')
    return 'UnknownDomain'


def write_cypher_header(f):
    """Write Cypher file header with setup commands."""
    f.write("// ========================================================================\n")
    f.write("// Ontology Import Script for Neo4j\n")
    f.write("// Generated from Logseq Ontology Blocks\n")
    f.write("// Supports: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies\n")
    f.write("// ========================================================================\n")
    f.write("// Usage: cypher-shell -u neo4j -p [password] < ontology.cypher\n")
    f.write("// ========================================================================\n\n")

    f.write("// Clear existing data (CAUTION: Deletes all nodes and relationships!)\n")
    f.write("// MATCH (n) DETACH DELETE n;\n\n")

    f.write("// Create constraints and indexes\n")
    f.write("CREATE CONSTRAINT concept_iri IF NOT EXISTS FOR (c:Concept) REQUIRE c.iri IS UNIQUE;\n")
    f.write("CREATE INDEX concept_term_id IF NOT EXISTS FOR (c:Concept) ON (c.term_id);\n")
    f.write("CREATE INDEX concept_preferred_term IF NOT EXISTS FOR (c:Concept) ON (c.preferred_term);\n")
    f.write("CREATE INDEX concept_domain IF NOT EXISTS FOR (c:Concept) ON (c.domain);\n")
    f.write("CREATE INDEX concept_status IF NOT EXISTS FOR (c:Concept) ON (c.status);\n\n")


def write_concept_node(f, block: OntologyBlock):
    """Write Cypher CREATE statement for a concept node."""
    iri = block.get_full_iri()
    if not iri:
        print(f"Warning: Skipping {block.term_id} - no IRI found", file=sys.stderr)
        return

    domain_label = get_domain_label(block)

    # Start CREATE statement with multiple labels
    f.write(f"CREATE (c{sanitize_string(block.term_id)}:Concept:{domain_label} {{\n")

    # Core properties
    f.write(f"  iri: '{sanitize_string(iri)}',\n")
    f.write(f"  term_id: '{sanitize_string(block.term_id)}',\n")
    f.write(f"  preferred_term: '{sanitize_string(block.preferred_term or '')}',\n")
    f.write(f"  definition: '{sanitize_string(block.definition or '')}',\n")

    # Domain and classification
    domain = block.get_domain()
    f.write(f"  domain: '{domain or 'unknown'}',\n")
    f.write(f"  source_domain: '{sanitize_string(block.source_domain or '')}',\n")

    # OWL properties
    f.write(f"  owl_class: '{sanitize_string(block.owl_class or '')}',\n")
    f.write(f"  owl_physicality: '{sanitize_string(block.owl_physicality or '')}',\n")
    f.write(f"  owl_role: '{sanitize_string(block.owl_role or '')}',\n")

    # Metadata
    f.write(f"  status: '{sanitize_string(block.status or '')}',\n")
    f.write(f"  public_access: {str(block.public_access).lower()},\n")
    f.write(f"  last_updated: '{sanitize_string(block.last_updated or '')}',\n")

    # Optional properties
    if block.version:
        f.write(f"  version: '{sanitize_string(block.version)}',\n")
    if block.maturity:
        f.write(f"  maturity: '{sanitize_string(block.maturity)}',\n")
    if block.quality_score is not None:
        f.write(f"  quality_score: {block.quality_score},\n")
    if block.authority_score is not None:
        f.write(f"  authority_score: {block.authority_score},\n")
    if block.cross_domain_links is not None:
        f.write(f"  cross_domain_links: {block.cross_domain_links},\n")
    if block.scope_note:
        f.write(f"  scope_note: '{sanitize_string(block.scope_note)}',\n")

    # Lists as arrays
    if block.alt_terms:
        terms_str = ', '.join(f"'{sanitize_string(t)}'" for t in block.alt_terms)
        f.write(f"  alt_terms: [{terms_str}],\n")
    if block.source:
        sources_str = ', '.join(f"'{sanitize_string(s)}'" for s in block.source)
        f.write(f"  sources: [{sources_str}],\n")
    if block.belongs_to_domain:
        domains_str = ', '.join(f"'{sanitize_string(d)}'" for d in block.belongs_to_domain)
        f.write(f"  belongs_to_domains: [{domains_str}],\n")
    if block.implemented_in_layer:
        layers_str = ', '.join(f"'{sanitize_string(l)}'" for l in block.implemented_in_layer)
        f.write(f"  implemented_in_layers: [{layers_str}],\n")

    # Domain-specific extensions
    if block.domain_extensions:
        for key, value in block.domain_extensions.items():
            safe_key = key.replace('-', '_')
            if isinstance(value, (int, float, bool)):
                f.write(f"  {safe_key}: {value},\n")
            else:
                f.write(f"  {safe_key}: '{sanitize_string(str(value))}',\n")

    # File path for reference
    f.write(f"  source_file: '{sanitize_string(str(block.file_path))}'\n")
    f.write("});\n\n")


def write_relationships(f, blocks: list[OntologyBlock]):
    """Write Cypher MATCH/CREATE statements for relationships."""
    f.write("\n// ========================================================================\n")
    f.write("// Relationships\n")
    f.write("// ========================================================================\n\n")

    # Create a mapping of term_id to ensure we can find nodes
    term_map = {block.term_id: block for block in blocks if block.term_id}

    for block in blocks:
        term_id = sanitize_string(block.term_id)

        # is-subclass-of relationships
        if block.is_subclass_of:
            f.write(f"// Subclass relationships for {block.term_id}\n")
            for parent in block.is_subclass_of:
                parent_safe = sanitize_string(parent)
                f.write(f"MATCH (child:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(parent:Concept {{preferred_term: '{parent_safe}'}})\n")
                f.write("CREATE (child)-[:IS_SUBCLASS_OF]->(parent);\n\n")

        # has-part relationships
        if block.has_part:
            for part in block.has_part:
                part_safe = sanitize_string(part)
                f.write(f"MATCH (whole:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(part:Concept {{preferred_term: '{part_safe}'}})\n")
                f.write("CREATE (whole)-[:HAS_PART]->(part);\n\n")

        # is-part-of relationships
        if block.is_part_of:
            for whole in block.is_part_of:
                whole_safe = sanitize_string(whole)
                f.write(f"MATCH (part:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(whole:Concept {{preferred_term: '{whole_safe}'}})\n")
                f.write("CREATE (part)-[:IS_PART_OF]->(whole);\n\n")

        # requires relationships
        if block.requires:
            for req in block.requires:
                req_safe = sanitize_string(req)
                f.write(f"MATCH (source:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(target:Concept {{preferred_term: '{req_safe}'}})\n")
                f.write("CREATE (source)-[:REQUIRES]->(target);\n\n")

        # depends-on relationships
        if block.depends_on:
            for dep in block.depends_on:
                dep_safe = sanitize_string(dep)
                f.write(f"MATCH (source:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(target:Concept {{preferred_term: '{dep_safe}'}})\n")
                f.write("CREATE (source)-[:DEPENDS_ON]->(target);\n\n")

        # enables relationships
        if block.enables:
            for enabled in block.enables:
                enabled_safe = sanitize_string(enabled)
                f.write(f"MATCH (source:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(target:Concept {{preferred_term: '{enabled_safe}'}})\n")
                f.write("CREATE (source)-[:ENABLES]->(target);\n\n")

        # relates-to relationships
        if block.relates_to:
            for related in block.relates_to:
                related_safe = sanitize_string(related)
                f.write(f"MATCH (source:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(target:Concept {{preferred_term: '{related_safe}'}})\n")
                f.write("CREATE (source)-[:RELATES_TO]->(target);\n\n")

        # bridges-to relationships (cross-domain)
        if block.bridges_to:
            for bridge in block.bridges_to:
                # Parse "target via property" format
                if ' via ' in bridge:
                    target, rel_type = bridge.split(' via ', 1)
                    target_safe = sanitize_string(target.strip())
                    rel_type_safe = sanitize_string(rel_type.strip().upper().replace('-', '_'))
                    f.write(f"MATCH (source:Concept {{term_id: '{term_id}'}}), ")
                    f.write(f"(target:Concept {{preferred_term: '{target_safe}'}})\n")
                    f.write(f"CREATE (source)-[:BRIDGES_TO {{via: '{rel_type_safe}'}}]->(target);\n\n")

        # Other custom relationships
        for rel_name, targets in block.other_relationships.items():
            rel_type = rel_name.upper().replace('-', '_')
            for target in targets:
                target_safe = sanitize_string(target)
                f.write(f"MATCH (source:Concept {{term_id: '{term_id}'}}), ")
                f.write(f"(target:Concept {{preferred_term: '{target_safe}'}})\n")
                f.write(f"CREATE (source)-[:{rel_type}]->(target);\n\n")


def write_useful_queries(f):
    """Write example queries at the end of the file."""
    f.write("\n// ========================================================================\n")
    f.write("// Useful Queries\n")
    f.write("// ========================================================================\n\n")

    f.write("// Count concepts by domain:\n")
    f.write("// MATCH (c:Concept) RETURN c.domain, count(c) ORDER BY count(c) DESC;\n\n")

    f.write("// Find all AI concepts:\n")
    f.write("// MATCH (c:ArtificialIntelligence) RETURN c.term_id, c.preferred_term, c.definition LIMIT 25;\n\n")

    f.write("// Find all Blockchain concepts:\n")
    f.write("// MATCH (c:Blockchain) RETURN c.term_id, c.preferred_term, c.definition LIMIT 25;\n\n")

    f.write("// Find class hierarchy (subclass relationships):\n")
    f.write("// MATCH path = (child:Concept)-[:IS_SUBCLASS_OF*]->(parent:Concept)\n")
    f.write("// RETURN path LIMIT 50;\n\n")

    f.write("// Find top-level concepts (no parent):\n")
    f.write("// MATCH (c:Concept) WHERE NOT (c)-[:IS_SUBCLASS_OF]->()\n")
    f.write("// RETURN c.term_id, c.preferred_term, c.domain;\n\n")

    f.write("// Find cross-domain bridges:\n")
    f.write("// MATCH (source:Concept)-[r:BRIDGES_TO]->(target:Concept)\n")
    f.write("// WHERE source.domain <> target.domain\n")
    f.write("// RETURN source.term_id, source.domain, type(r), target.term_id, target.domain, r.via LIMIT 25;\n\n")

    f.write("// Find concepts by maturity level:\n")
    f.write("// MATCH (c:Concept {maturity: 'Mature'}) RETURN c.term_id, c.preferred_term LIMIT 25;\n\n")

    f.write("// Find high quality concepts (quality_score >= 0.8):\n")
    f.write("// MATCH (c:Concept) WHERE c.quality_score >= 0.8\n")
    f.write("// RETURN c.term_id, c.preferred_term, c.quality_score ORDER BY c.quality_score DESC;\n\n")

    f.write("// Full-text search in definitions:\n")
    f.write("// MATCH (c:Concept) WHERE c.definition CONTAINS 'learning'\n")
    f.write("// RETURN c.term_id, c.preferred_term, c.definition LIMIT 10;\n\n")


def convert_to_cypher(input_path: Path, output_file: Path):
    """Convert ontology blocks to Cypher format."""
    parser = OntologyBlockParser()

    # Parse input (file or directory)
    if input_path.is_file():
        print(f"Parsing file: {input_path}")
        blocks = [parser.parse_file(input_path)]
        blocks = [b for b in blocks if b is not None]
    elif input_path.is_dir():
        print(f"Parsing directory: {input_path}")
        blocks = parser.parse_directory(input_path)
    else:
        print(f"Error: {input_path} is neither a file nor directory", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(blocks)} ontology blocks")

    # Validate blocks
    blocks_with_errors = 0
    for block in blocks:
        errors = block.validate()
        if errors:
            blocks_with_errors += 1
            if blocks_with_errors <= 5:  # Show first 5 errors
                print(f"Warning: {block.term_id} has validation errors:", file=sys.stderr)
                for error in errors[:3]:
                    print(f"  - {error}", file=sys.stderr)

    if blocks_with_errors > 0:
        print(f"\nWarning: {blocks_with_errors} blocks have validation errors", file=sys.stderr)

    # Statistics by domain
    domain_counts = defaultdict(int)
    for block in blocks:
        domain = block.get_domain()
        if domain:
            domain_counts[domain] += 1

    print("\nStatistics by domain:")
    for domain, count in sorted(domain_counts.items()):
        domain_name = DOMAIN_CONFIG[domain]['full_name'] if domain in DOMAIN_CONFIG else domain
        print(f"  {domain_name}: {count}")

    # Write Cypher file
    print(f"\nGenerating Cypher script: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        write_cypher_header(f)

        f.write("// ========================================================================\n")
        f.write("// Concept Nodes\n")
        f.write("// ========================================================================\n\n")

        for i, block in enumerate(blocks, 1):
            write_concept_node(f, block)
            if i % 50 == 0:
                print(f"  Written {i}/{len(blocks)} nodes...")

        write_relationships(f, blocks)
        write_useful_queries(f)

    # Final statistics
    file_size_kb = output_file.stat().st_size / 1024
    print(f"\n✅ Cypher script generated successfully!")
    print(f"   Output: {output_file}")
    print(f"   File size: {file_size_kb:.2f} KB")
    print(f"   Total concepts: {len(blocks)}")
    print(f"   Domains: {len(domain_counts)}")
    print(f"\n📘 Import to Neo4j with:")
    print(f"   cypher-shell -u neo4j -p [password] < {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Ontology Blocks to Neo4j Cypher format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert directory of markdown files
  python convert-to-cypher.py --input mainKnowledgeGraph/pages --output output/ontology.cypher

  # Convert single file
  python convert-to-cypher.py --input mainKnowledgeGraph/pages/rb-0001-robot.md --output output/robot.cypher

  # Import to Neo4j
  cypher-shell -u neo4j -p password < output/ontology.cypher
        """
    )
    parser.add_argument('--input', type=Path, required=True,
                       help='Input markdown file or directory')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output Cypher script file')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert_to_cypher(args.input, args.output)


if __name__ == '__main__':
    main()
