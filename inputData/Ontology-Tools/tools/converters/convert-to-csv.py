#!/usr/bin/env python3
"""
Convert Logseq Ontology Blocks to CSV Format
==============================================

Parses Logseq markdown files with OntologyBlock sections and exports to multiple CSV files:
- concepts.csv: All concepts with full IRIs and metadata
- properties.csv: All properties and their values
- relationships.csv: All relationships between concepts
- domains.csv: Domain classification for all concepts

Supports all 6 domains: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies.

Uses the shared ontology_block_parser library for consistent parsing.

Usage:
    python convert-to-csv.py --input <pages_dir> --output <output_dir>
    python convert-to-csv.py --input mainKnowledgeGraph/pages --output exports/csv

Requirements:
    - Logseq pages directory with OntologyBlock sections
    - ontology_block_parser library (lib/ontology_block_parser.py)
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_block_parser import OntologyBlock, DOMAIN_CONFIG
from ontology_loader import OntologyLoader


def sanitize_csv(value: Any) -> str:
    """Sanitize value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value).strip()


def generate_concepts_csv(blocks: List[OntologyBlock], output_dir: Path) -> int:
    """
    Generate concepts.csv with all concepts and their metadata.

    Columns: term_id, iri, preferred_term, definition, status, domain,
             physicality, role, maturity, quality_score
    """
    concepts_file = output_dir / 'concepts.csv'

    fieldnames = [
        'term_id', 'iri', 'preferred_term', 'definition',
        'status', 'source_domain', 'physicality', 'role',
        'maturity', 'quality_score', 'authority_score',
        'public_access', 'last_updated', 'version'
    ]

    with open(concepts_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for block in blocks:
            writer.writerow({
                'term_id': sanitize_csv(block.term_id),
                'iri': sanitize_csv(block.get_full_iri()),
                'preferred_term': sanitize_csv(block.preferred_term),
                'definition': sanitize_csv(block.definition),
                'status': sanitize_csv(block.status),
                'source_domain': sanitize_csv(block.source_domain),
                'physicality': sanitize_csv(block.owl_physicality),
                'role': sanitize_csv(block.owl_role),
                'maturity': sanitize_csv(block.maturity),
                'quality_score': sanitize_csv(block.quality_score),
                'authority_score': sanitize_csv(block.authority_score),
                'public_access': sanitize_csv(block.public_access),
                'last_updated': sanitize_csv(block.last_updated),
                'version': sanitize_csv(block.version)
            })

    return len(blocks)


def generate_properties_csv(blocks: List[OntologyBlock], output_dir: Path) -> int:
    """
    Generate properties.csv with all properties from all concepts.

    Columns: term_id, property_name, property_value
    """
    properties_file = output_dir / 'properties.csv'
    row_count = 0

    fieldnames = ['term_id', 'property_name', 'property_value']

    with open(properties_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for block in blocks:
            term_id = block.term_id

            # Alt terms
            for alt_term in block.alt_terms:
                writer.writerow({
                    'term_id': term_id,
                    'property_name': 'alt-term',
                    'property_value': sanitize_csv(alt_term)
                })
                row_count += 1

            # Sources
            for source in block.source:
                writer.writerow({
                    'term_id': term_id,
                    'property_name': 'source',
                    'property_value': sanitize_csv(source)
                })
                row_count += 1

            # Belongs to domain
            for domain in block.belongs_to_domain:
                writer.writerow({
                    'term_id': term_id,
                    'property_name': 'belongs-to-domain',
                    'property_value': sanitize_csv(domain)
                })
                row_count += 1

            # Implemented in layer
            for layer in block.implemented_in_layer:
                writer.writerow({
                    'term_id': term_id,
                    'property_name': 'implemented-in-layer',
                    'property_value': sanitize_csv(layer)
                })
                row_count += 1

            # Scope note
            if block.scope_note:
                writer.writerow({
                    'term_id': term_id,
                    'property_name': 'scope-note',
                    'property_value': sanitize_csv(block.scope_note)
                })
                row_count += 1

            # Domain-specific extensions
            for ext_name, ext_value in block.domain_extensions.items():
                writer.writerow({
                    'term_id': term_id,
                    'property_name': ext_name,
                    'property_value': sanitize_csv(ext_value)
                })
                row_count += 1

    return row_count


def generate_relationships_csv(blocks: List[OntologyBlock], output_dir: Path) -> int:
    """
    Generate relationships.csv with all relationships between concepts.

    Columns: source_term_id, relationship_type, target_term
    """
    relationships_file = output_dir / 'relationships.csv'
    row_count = 0

    fieldnames = ['source_term_id', 'relationship_type', 'target_term']

    with open(relationships_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for block in blocks:
            term_id = block.term_id

            # Standard relationships
            for parent in block.is_subclass_of:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'is-subclass-of',
                    'target_term': sanitize_csv(parent)
                })
                row_count += 1

            for part in block.has_part:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'has-part',
                    'target_term': sanitize_csv(part)
                })
                row_count += 1

            for whole in block.is_part_of:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'is-part-of',
                    'target_term': sanitize_csv(whole)
                })
                row_count += 1

            for req in block.requires:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'requires',
                    'target_term': sanitize_csv(req)
                })
                row_count += 1

            for dep in block.depends_on:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'depends-on',
                    'target_term': sanitize_csv(dep)
                })
                row_count += 1

            for enabled in block.enables:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'enables',
                    'target_term': sanitize_csv(enabled)
                })
                row_count += 1

            for related in block.relates_to:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'relates-to',
                    'target_term': sanitize_csv(related)
                })
                row_count += 1

            # Other relationships
            for rel_type, targets in block.other_relationships.items():
                for target in targets:
                    writer.writerow({
                        'source_term_id': term_id,
                        'relationship_type': rel_type,
                        'target_term': sanitize_csv(target)
                    })
                    row_count += 1

            # Cross-domain bridges
            for bridge in block.bridges_to:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'bridges-to',
                    'target_term': sanitize_csv(bridge)
                })
                row_count += 1

            for bridge in block.bridges_from:
                writer.writerow({
                    'source_term_id': term_id,
                    'relationship_type': 'bridges-from',
                    'target_term': sanitize_csv(bridge)
                })
                row_count += 1

    return row_count


def generate_domains_csv(blocks: List[OntologyBlock], output_dir: Path) -> int:
    """
    Generate domains.csv with domain classification for all concepts.

    Columns: term_id, domain, namespace, full_name
    """
    domains_file = output_dir / 'domains.csv'

    fieldnames = ['term_id', 'domain_code', 'domain_name', 'namespace']

    with open(domains_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for block in blocks:
            domain = block.get_domain()
            if domain and domain in DOMAIN_CONFIG:
                writer.writerow({
                    'term_id': block.term_id,
                    'domain_code': domain,
                    'domain_name': DOMAIN_CONFIG[domain]['full_name'],
                    'namespace': DOMAIN_CONFIG[domain]['namespace']
                })

    return len(blocks)


def export_to_csv(blocks: List[OntologyBlock], output_dir: Path) -> Dict[str, int]:
    """
    Export all ontology blocks to CSV files.

    Args:
        blocks: List of parsed OntologyBlock objects
        output_dir: Directory to write CSV files

    Returns:
        Dictionary with row counts for each file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    print("\nGenerating CSV files:")
    print("-" * 80)

    # Generate concepts.csv
    concepts_count = generate_concepts_csv(blocks, output_dir)
    stats['concepts'] = concepts_count
    print(f"✓ concepts.csv: {concepts_count} concepts")

    # Generate properties.csv
    props_count = generate_properties_csv(blocks, output_dir)
    stats['properties'] = props_count
    print(f"✓ properties.csv: {props_count} property values")

    # Generate relationships.csv
    rels_count = generate_relationships_csv(blocks, output_dir)
    stats['relationships'] = rels_count
    print(f"✓ relationships.csv: {rels_count} relationships")

    # Generate domains.csv
    domains_count = generate_domains_csv(blocks, output_dir)
    stats['domains'] = domains_count
    print(f"✓ domains.csv: {domains_count} domain classifications")

    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Convert Logseq Ontology Blocks to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input pages --output exports/csv
  %(prog)s --input mainKnowledgeGraph/pages --output /tmp/csv_export
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
        help='Output directory for CSV files'
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
    print("Logseq Ontology to CSV Converter")
    print("=" * 80)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")

    # Load all ontology blocks
    print("\nLoading ontology blocks...")
    loader = OntologyLoader(cache_size=200)
    blocks = loader.load_directory(args.input, progress=True)

    if not blocks:
        print("\n⚠️  No ontology blocks found in input directory!")
        sys.exit(1)

    print(f"✓ Loaded {len(blocks)} ontology blocks")

    # Print cache statistics
    cache_stats = loader.get_cache_stats()
    print(f"   Cache performance: {cache_stats['hit_rate']:.1%} hit rate")

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

    # Export to CSV
    stats = export_to_csv(blocks, args.output)

    # Summary
    print("\n" + "=" * 80)
    print("Export Summary:")
    print("-" * 80)
    for file_type, count in stats.items():
        print(f"  {file_type}.csv: {count} rows")
    print(f"\nCSV files saved to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
