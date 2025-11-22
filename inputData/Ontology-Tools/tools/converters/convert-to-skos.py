#!/usr/bin/env python3
"""
Convert Ontology Blocks to SKOS Format
=======================================

Converts markdown files with OntologyBlock sections to SKOS (Simple Knowledge Organization System) format.
Maps OWL classes to SKOS concepts for use in thesaurus applications.
Supports all 6 domains: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies.

Uses the shared ontology_block_parser library for consistent parsing.

Usage:
    python convert-to-skos.py --input <input_path> --output <output_file>
    python convert-to-skos.py --input mainKnowledgeGraph/pages/ --output ontology.ttl
    python convert-to-skos.py --input mainKnowledgeGraph/pages/rb-*.md --output robotics.ttl

Features:
    - Converts to SKOS concept schemes
    - Uses skos:Concept, skos:prefLabel, skos:definition
    - Handles hierarchies with skos:broader/narrower
    - Supports all 6 domains as separate concept schemes
    - Outputs Turtle (TTL) format
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict
from datetime import date

# Add lib directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG, STANDARD_NAMESPACES


# Map full domain names to prefix keys
DOMAIN_NAME_TO_PREFIX = {
    'artificial intelligence': 'ai',
    'ai': 'ai',
    'blockchain': 'bc',
    'bc': 'bc',
    'robotics': 'rb',
    'rb': 'rb',
    'metaverse': 'mv',
    'mv': 'mv',
    'telecollaboration': 'tc',
    'tc': 'tc',
    'disruptive technologies': 'dt',
    'dt': 'dt',
}


def normalize_domain(domain: Optional[str]) -> Optional[str]:
    """
    Normalize domain name to standard prefix.

    Args:
        domain: Domain name (e.g., "robotics" or "rb")

    Returns:
        Standard domain prefix (e.g., "rb") or None
    """
    if not domain:
        return None
    domain_lower = domain.lower().strip()
    return DOMAIN_NAME_TO_PREFIX.get(domain_lower, domain_lower)


def write_turtle_header(f, domain_prefixes: Set[str]):
    """
    Write Turtle file header with namespace prefixes.

    Args:
        f: File handle
        domain_prefixes: Set of domain prefixes used in the file
    """
    f.write("# Ontology Concept Schemes - SKOS Format\n")
    f.write("# Generated from OntologyBlock metadata\n")
    f.write(f"# Generated: {date.today().isoformat()}\n\n")

    # Write standard namespace prefixes
    for prefix, uri in STANDARD_NAMESPACES.items():
        f.write(f"@prefix {prefix}: <{uri}> .\n")

    # Write domain namespace prefixes
    for domain in sorted(domain_prefixes):
        if domain in DOMAIN_CONFIG:
            f.write(f"@prefix {domain}: <{DOMAIN_CONFIG[domain]['namespace']}> .\n")

    f.write("\n")


def convert_reference_to_qname(reference: str, block: OntologyBlock) -> str:
    """
    Convert a reference (e.g., [[TermName]] or term-id) to a qualified name.

    Args:
        reference: Reference string
        block: Source OntologyBlock for context

    Returns:
        Qualified name string (e.g., "ai:ConceptName")
    """
    # Remove wiki-link brackets if present
    ref = reference.strip()
    if ref.startswith('[[') and ref.endswith(']]'):
        ref = ref[2:-2]

    # If already a full URI, return as-is (will be wrapped in <>)
    if ref.startswith('http://') or ref.startswith('https://'):
        return f"<{ref}>"

    # If it's already a prefixed name (e.g., "ai:Concept"), return as-is
    if ':' in ref:
        prefix = ref.split(':', 1)[0]
        if prefix in DOMAIN_CONFIG or prefix in STANDARD_NAMESPACES:
            return ref

    # Otherwise, assume it's from the same domain as the block
    domain = normalize_domain(block.get_domain())
    if domain and domain in DOMAIN_CONFIG:
        # Clean the reference to be a valid local name
        clean_ref = ref.replace(' ', '').replace('-', '')
        return f"{domain}:{clean_ref}"

    # Fallback: return as-is
    return ref


def escape_turtle_string(s: str) -> str:
    """
    Escape special characters in Turtle string literals.

    Args:
        s: String to escape

    Returns:
        Escaped string
    """
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')


def write_concept_schemes(f, blocks: List[OntologyBlock]):
    """
    Write SKOS ConceptScheme definitions for each domain.

    Args:
        f: File handle
        blocks: List of OntologyBlock instances
    """
    f.write("# ============================================================\n")
    f.write("# Concept Schemes by Domain\n")
    f.write("# ============================================================\n\n")

    # Get domains present in the data
    domains_present = set()
    for block in blocks:
        domain = normalize_domain(block.get_domain())
        if domain:
            domains_present.add(domain)

    # Create a concept scheme for each domain
    for domain in sorted(domains_present):
        if domain not in DOMAIN_CONFIG:
            continue

        config = DOMAIN_CONFIG[domain]
        scheme_uri = f"{domain}:ConceptScheme"

        f.write(f"{scheme_uri}\n")
        f.write("    a skos:ConceptScheme ;\n")
        f.write(f'    dcterms:title "{config["full_name"]} Concept Scheme"@en ;\n')
        f.write(f'    dcterms:description "Hierarchical organization of {config["full_name"].lower()} concepts"@en ;\n')
        f.write(f'    dcterms:created "{date.today().isoformat()}"^^xsd:date ;\n')
        f.write('    dcterms:creator "Metaverse Ontology Project" .\n\n')


def write_concepts(f, blocks: List[OntologyBlock]):
    """
    Write SKOS Concept definitions.

    Args:
        f: File handle
        blocks: List of OntologyBlock instances
    """
    f.write("# ============================================================\n")
    f.write("# Concepts\n")
    f.write("# ============================================================\n\n")

    for block in blocks:
        domain = normalize_domain(block.get_domain())
        if not domain or domain not in DOMAIN_CONFIG:
            continue

        # Get concept URI from owl:class or construct from term-id
        if block.owl_class and ':' in block.owl_class:
            concept_qname = block.owl_class
        elif block.term_id:
            # Clean term-id to make valid local name
            local_name = block.term_id.replace('-', '').replace(' ', '')
            concept_qname = f"{domain}:{local_name}"
        else:
            continue

        f.write(f"{concept_qname}\n")
        f.write("    a skos:Concept ;\n")

        # Link to concept scheme
        f.write(f"    skos:inScheme {domain}:ConceptScheme ;\n")

        # skos:notation - the term ID
        if block.term_id:
            f.write(f'    skos:notation "{block.term_id}"^^xsd:string ;\n')

        # skos:prefLabel - the preferred term
        if block.preferred_term:
            escaped_term = escape_turtle_string(block.preferred_term)
            f.write(f'    skos:prefLabel "{escaped_term}"@en ;\n')

        # skos:altLabel - alternative terms
        if block.alt_terms:
            for alt_term in block.alt_terms:
                escaped_alt = escape_turtle_string(alt_term)
                f.write(f'    skos:altLabel "{escaped_alt}"@en ;\n')

        # skos:definition - the definition
        if block.definition:
            # Truncate very long definitions for readability
            definition = block.definition
            if len(definition) > 500:
                definition = definition[:497] + "..."
            escaped_def = escape_turtle_string(definition)
            f.write(f'    skos:definition "{escaped_def}"@en ;\n')

        # skos:scopeNote - additional scope information
        if block.scope_note:
            escaped_scope = escape_turtle_string(block.scope_note)
            f.write(f'    skos:scopeNote "{escaped_scope}"@en ;\n')

        # skos:broader - parent classes
        if block.is_subclass_of:
            for parent in block.is_subclass_of:
                parent_qname = convert_reference_to_qname(parent, block)
                f.write(f"    skos:broader {parent_qname} ;\n")

        # Additional metadata as custom properties
        if block.status:
            f.write(f'    dcterms:status "{block.status}" ;\n')

        if block.maturity:
            f.write(f'    dcterms:maturity "{block.maturity}" ;\n')

        if block.source:
            for source in block.source:
                escaped_source = escape_turtle_string(source)
                f.write(f'    dcterms:source "{escaped_source}" ;\n')

        # Related concepts
        if block.relates_to:
            for related in block.relates_to:
                related_qname = convert_reference_to_qname(related, block)
                f.write(f"    skos:related {related_qname} ;\n")

        # File source
        f.write(f'    dcterms:fileSource "{block.file_path}" .\n\n')


def write_narrower_relationships(f, blocks: List[OntologyBlock]):
    """
    Write inverse narrower relationships (children pointing from parents).

    Args:
        f: File handle
        blocks: List of OntologyBlock instances
    """
    f.write("# ============================================================\n")
    f.write("# Inverse Narrower Relationships\n")
    f.write("# ============================================================\n\n")

    # Build parent-to-children mapping
    parent_to_children = defaultdict(list)

    for block in blocks:
        domain = normalize_domain(block.get_domain())
        if not domain or domain not in DOMAIN_CONFIG:
            continue

        # Get concept URI
        if block.owl_class and ':' in block.owl_class:
            concept_qname = block.owl_class
        elif block.term_id:
            local_name = block.term_id.replace('-', '').replace(' ', '')
            concept_qname = f"{domain}:{local_name}"
        else:
            continue

        # Map each parent to this child
        for parent in block.is_subclass_of:
            parent_qname = convert_reference_to_qname(parent, block)
            parent_to_children[parent_qname].append(concept_qname)

    # Write narrower relationships
    for parent_qname, children in sorted(parent_to_children.items()):
        f.write(f"{parent_qname}\n")
        for i, child_qname in enumerate(children):
            if i == len(children) - 1:
                f.write(f"    skos:narrower {child_qname} .\n")
            else:
                f.write(f"    skos:narrower {child_qname} ;\n")
        f.write("\n")


def write_semantic_relations(f, blocks: List[OntologyBlock]):
    """
    Write additional semantic relationships as SKOS properties.

    Args:
        f: File handle
        blocks: List of OntologyBlock instances
    """
    f.write("# ============================================================\n")
    f.write("# Additional Semantic Relationships\n")
    f.write("# ============================================================\n\n")

    for block in blocks:
        domain = normalize_domain(block.get_domain())
        if not domain or domain not in DOMAIN_CONFIG:
            continue

        # Get concept URI
        if block.owl_class and ':' in block.owl_class:
            concept_qname = block.owl_class
        elif block.term_id:
            local_name = block.term_id.replace('-', '').replace(' ', '')
            concept_qname = f"{domain}:{local_name}"
        else:
            continue

        relationships = []

        # has_part -> dcterms:hasPart
        if block.has_part:
            for part in block.has_part:
                part_qname = convert_reference_to_qname(part, block)
                relationships.append(f"    dcterms:hasPart {part_qname}")

        # is_part_of -> dcterms:isPartOf
        if block.is_part_of:
            for whole in block.is_part_of:
                whole_qname = convert_reference_to_qname(whole, block)
                relationships.append(f"    dcterms:isPartOf {whole_qname}")

        # requires -> dcterms:requires
        if block.requires:
            for req in block.requires:
                req_qname = convert_reference_to_qname(req, block)
                relationships.append(f"    dcterms:requires {req_qname}")

        # depends_on -> custom property
        if block.depends_on:
            for dep in block.depends_on:
                dep_qname = convert_reference_to_qname(dep, block)
                relationships.append(f"    {domain}:dependsOn {dep_qname}")

        # enables -> custom property
        if block.enables:
            for enables in block.enables:
                enables_qname = convert_reference_to_qname(enables, block)
                relationships.append(f"    {domain}:enables {enables_qname}")

        # Write if we have any relationships
        if relationships:
            f.write(f"{concept_qname}\n")
            for i, rel in enumerate(relationships):
                if i == len(relationships) - 1:
                    f.write(f"{rel} .\n")
                else:
                    f.write(f"{rel} ;\n")
            f.write("\n")


def convert_blocks_to_skos(blocks: List[OntologyBlock], output_path: Path):
    """
    Convert multiple OntologyBlocks to a SKOS Turtle file.

    Args:
        blocks: List of OntologyBlock instances
        output_path: Output file path
    """
    # Get all domain prefixes used
    domain_prefixes = set()
    for block in blocks:
        domain = normalize_domain(block.get_domain())
        if domain:
            domain_prefixes.add(domain)

    # Write Turtle file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header with namespace prefixes
        write_turtle_header(f, domain_prefixes)

        # Concept schemes by domain
        write_concept_schemes(f, blocks)

        # Main concept definitions
        write_concepts(f, blocks)

        # Inverse narrower relationships
        write_narrower_relationships(f, blocks)

        # Additional semantic relationships
        write_semantic_relations(f, blocks)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Convert OntologyBlocks to SKOS format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all files in a directory
  python convert-to-skos.py --input mainKnowledgeGraph/pages/ --output ontology.ttl

  # Convert a single file
  python convert-to-skos.py --input pages/rb-0100.md --output robotics-term.ttl

  # Convert files matching a pattern
  python convert-to-skos.py --input "pages/AI-*.md" --output ai-ontology.ttl
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input path: directory, file, or glob pattern'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output SKOS Turtle file path (.ttl)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate blocks before conversion (show warnings for incomplete blocks)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OntologyBlock to SKOS Conversion")
    print("=" * 80)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Parse input
    parser_obj = OntologyBlockParser()
    blocks = []

    if input_path.is_file():
        print(f"\nParsing file: {input_path}")
        block = parser_obj.parse_file(input_path)
        if block:
            blocks.append(block)
    elif input_path.is_dir():
        print(f"\nParsing directory: {input_path}")
        blocks = parser_obj.parse_directory(input_path)
    else:
        # Try glob pattern
        print(f"\nParsing pattern: {args.input}")
        from glob import glob
        for file_path in glob(args.input):
            block = parser_obj.parse_file(Path(file_path))
            if block:
                blocks.append(block)

    if not blocks:
        print("\n❌ Error: No valid ontology blocks found")
        sys.exit(1)

    print(f"\n✅ Successfully parsed {len(blocks)} ontology blocks")

    # Statistics by domain
    domain_counts = defaultdict(int)
    for block in blocks:
        domain = normalize_domain(block.get_domain())
        if domain:
            domain_counts[domain] += 1

    print("\n📊 Statistics by domain:")
    for domain, count in sorted(domain_counts.items()):
        domain_name = DOMAIN_CONFIG[domain]['full_name'] if domain in DOMAIN_CONFIG else domain
        print(f"   {domain_name}: {count}")

    # Validation
    if args.validate:
        print("\n🔍 Validating blocks...")
        total_errors = 0
        blocks_with_errors = 0
        for block in blocks:
            errors = block.validate()
            if errors:
                blocks_with_errors += 1
                total_errors += len(errors)
                print(f"\n⚠️  {block.term_id}:")
                for error in errors:
                    print(f"   - {error}")

        if blocks_with_errors > 0:
            print(f"\n⚠️  Validation: {blocks_with_errors}/{len(blocks)} blocks have errors")
            print(f"   (Total {total_errors} errors)")
        else:
            print("\n✅ Validation: All blocks valid")

    # Convert to SKOS
    print("\n🔄 Converting to SKOS...")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_blocks_to_skos(blocks, output_path)

    print(f"\n✅ SKOS file saved: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"   Total concepts: {len(blocks)}")
    print(f"   Concept schemes: {len(domain_counts)}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
