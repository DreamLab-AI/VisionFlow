#!/usr/bin/env python3
"""
Generate individual page JSON files for the React app.
Creates /api/pages/*.json from Logseq markdown files.

Updated to use shared ontology_block_parser library.
Supports all 6 domains: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add lib directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG


def extract_wiki_links(content: str) -> List[str]:
    """Extract all wiki links from content."""
    links = re.findall(r'\[\[([^\]]+)\]\]', content)
    return list(set(links))  # Deduplicate


def clean_content(content: str) -> str:
    """Clean markdown content for display, removing metadata."""
    # Remove YAML front matter
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

    # Remove properties lines (key:: value at start of file)
    lines = content.split('\n')
    cleaned_lines = []
    in_content = False

    for line in lines:
        # Start content after first heading or non-property line
        if line.startswith('#') or (line.strip() and '::' not in line):
            in_content = True

        if in_content:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def build_ontology_metadata(block: OntologyBlock) -> Dict[str, Any]:
    """Build comprehensive ontology metadata from parsed block."""
    metadata = {
        # Core identification
        'term_id': block.term_id,
        'preferred_term': block.preferred_term,
        'alt_terms': block.alt_terms,
        'iri': block.get_full_iri(),

        # Domain and classification
        'source_domain': block.source_domain,
        'domain': block.get_domain(),
        'domain_full_name': DOMAIN_CONFIG.get(block.get_domain(), {}).get('full_name', '') if block.get_domain() else '',

        # Definition
        'definition': block.definition,
        'scope_note': block.scope_note,

        # Status and quality
        'status': block.status,
        'maturity': block.maturity,
        'version': block.version,
        'public_access': block.public_access,
        'last_updated': block.last_updated,
        'authority_score': block.authority_score,
        'quality_score': block.quality_score,
        'cross_domain_links': block.cross_domain_links,

        # OWL semantic classification
        'owl_class': block.owl_class,
        'owl_physicality': block.owl_physicality,
        'owl_role': block.owl_role,
        'owl_inferred_class': block.owl_inferred_class,

        # Relationships
        'is_subclass_of': block.is_subclass_of,
        'has_part': block.has_part,
        'is_part_of': block.is_part_of,
        'requires': block.requires,
        'depends_on': block.depends_on,
        'enables': block.enables,
        'relates_to': block.relates_to,

        # Cross-domain bridges
        'bridges_to': block.bridges_to,
        'bridges_from': block.bridges_from,

        # Domain-specific extensions
        'domain_extensions': block.domain_extensions,

        # Additional metadata
        'belongs_to_domain': block.belongs_to_domain,
        'implemented_in_layer': block.implemented_in_layer,
        'source': block.source,
    }

    # Add other relationships
    if block.other_relationships:
        metadata['other_relationships'] = dict(block.other_relationships)

    return metadata


def process_page(md_file: Path, parser: OntologyBlockParser) -> Dict[str, Any]:
    """Process a single markdown page into page data."""
    content = md_file.read_text(encoding='utf-8')

    # Parse ontology block using shared library
    block = parser.parse_file(md_file)

    # Extract wiki links for navigation
    wiki_links = extract_wiki_links(content)

    # Get page title
    title = block.preferred_term if block and block.preferred_term else md_file.stem

    # Clean content for display
    display_content = clean_content(content)

    # Build page data
    page_data = {
        'id': md_file.stem,
        'title': title,
        'content': display_content,
        'backlinks': [],  # Will be populated by backlink analysis
        'wiki_links': wiki_links,
    }

    # Add comprehensive ontology metadata if available
    if block and block.term_id:
        page_data['ontology'] = build_ontology_metadata(block)

        # Add validation results
        validation_errors = block.validate()
        page_data['ontology']['validation'] = {
            'is_valid': len(validation_errors) == 0,
            'errors': validation_errors
        }

    return page_data


def build_backlinks_map(pages_dir: Path) -> Dict[str, List[str]]:
    """Build a map of page -> list of pages that link to it."""
    backlinks = {}

    # First pass: collect all wiki links
    for md_file in pages_dir.glob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        wiki_links = extract_wiki_links(content)
        page_name = md_file.stem

        for link in wiki_links:
            if link not in backlinks:
                backlinks[link] = []
            backlinks[link].append(page_name)

    return backlinks


def build_domain_index(pages_dir: Path, parser: OntologyBlockParser) -> Dict[str, List[str]]:
    """Build an index of pages by domain."""
    domain_index = {domain: [] for domain in DOMAIN_CONFIG.keys()}
    domain_index['uncategorized'] = []

    for md_file in sorted(pages_dir.glob('*.md')):
        try:
            block = parser.parse_file(md_file)
            if block and block.term_id:
                domain = block.get_domain()
                if domain and domain in DOMAIN_CONFIG:
                    domain_index[domain].append(md_file.stem)
                else:
                    domain_index['uncategorized'].append(md_file.stem)
        except Exception:
            pass

    return domain_index


def main():
    """Generate page API files with domain organization."""
    if len(sys.argv) < 3:
        print("Usage: python generate_page_api.py <pages-dir> <output-dir>")
        print("\nGenerates JSON API for React app with:")
        print("  - Full IRIs for all terms")
        print("  - Domain-organized structure")
        print("  - Comprehensive ontology metadata")
        print("  - Search-friendly format")
        print("  - Support for all 6 domains")
        sys.exit(1)

    pages_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not pages_dir.exists():
        print(f"Error: Pages directory not found: {pages_dir}")
        sys.exit(1)

    print(f"🔧 Generating Page API")
    print(f"{'='*60}")
    print(f"Source: {pages_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Initialize parser
    parser = OntologyBlockParser()

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_output = output_dir / 'pages'
    pages_output.mkdir(exist_ok=True)

    # Build backlinks map
    print("📊 Building backlinks map...")
    backlinks_map = build_backlinks_map(pages_dir)
    print(f"   Found {sum(len(v) for v in backlinks_map.values())} backlinks\n")

    # Build domain index
    print("🗂️  Building domain index...")
    domain_index = build_domain_index(pages_dir, parser)

    # Print domain statistics
    print("\n📈 Domain Statistics:")
    for domain_key in sorted(DOMAIN_CONFIG.keys()):
        domain_name = DOMAIN_CONFIG[domain_key]['full_name']
        count = len(domain_index[domain_key])
        print(f"   {domain_name}: {count} pages")
    print(f"   Uncategorized: {len(domain_index['uncategorized'])} pages\n")

    # Process each page
    print("📄 Processing pages...")
    processed = 0
    errors = 0
    pages_by_domain = {domain: [] for domain in DOMAIN_CONFIG.keys()}
    pages_by_domain['uncategorized'] = []

    for md_file in sorted(pages_dir.glob('*.md')):
        try:
            page_data = process_page(md_file, parser)

            # Add backlinks
            page_name = md_file.stem
            page_data['backlinks'] = backlinks_map.get(page_name, [])

            # Write individual page JSON file
            output_file = pages_output / f"{md_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)

            # Track by domain
            if 'ontology' in page_data and page_data['ontology'].get('domain'):
                domain = page_data['ontology']['domain']
                if domain in pages_by_domain:
                    pages_by_domain[domain].append({
                        'id': page_data['id'],
                        'title': page_data['title'],
                        'term_id': page_data['ontology'].get('term_id'),
                        'iri': page_data['ontology'].get('iri'),
                    })
            else:
                pages_by_domain['uncategorized'].append({
                    'id': page_data['id'],
                    'title': page_data['title'],
                })

            processed += 1
            domain_label = page_data.get('ontology', {}).get('domain', 'none')
            print(f"   ✓ {page_data['title']} [{domain_label}]")

        except Exception as e:
            errors += 1
            print(f"   ✗ Error processing {md_file.name}: {e}")

    # Write domain index
    print("\n🗂️  Writing domain index...")
    index_file = output_dir / 'domain-index.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            'domains': {
                domain: {
                    'name': DOMAIN_CONFIG[domain]['full_name'],
                    'prefix': DOMAIN_CONFIG[domain]['prefix'],
                    'namespace': DOMAIN_CONFIG[domain]['namespace'],
                    'pages': pages_by_domain[domain],
                    'count': len(pages_by_domain[domain])
                }
                for domain in DOMAIN_CONFIG.keys()
            },
            'uncategorized': {
                'pages': pages_by_domain['uncategorized'],
                'count': len(pages_by_domain['uncategorized'])
            },
            'total_pages': processed
        }, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Page API Generation Complete")
    print(f"{'='*60}")
    print(f"📊 Results:")
    print(f"   ✅ Processed: {processed} pages")
    print(f"   ❌ Errors: {errors}")
    print(f"   📁 Output: {output_dir}")
    print(f"   📑 Individual pages: {pages_output}")
    print(f"   🗂️  Domain index: {index_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
