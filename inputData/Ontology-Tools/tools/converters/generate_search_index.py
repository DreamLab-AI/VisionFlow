#!/usr/bin/env python3
"""
Generate search index from Logseq markdown pages.
Creates /api/search-index.json for the React app's search functionality.

Updated to use shared ontology_block_parser library.
Supports all 6 domains with full IRI support and domain facets.
Optimized for fuzzy search with aliases and comprehensive metadata.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Add lib directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG


def extract_text_content(content: str, max_length: int = 500) -> str:
    """Extract plain text from markdown, removing syntax."""
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)

    # Remove wiki links but keep text
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove YAML blocks
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)

    # Remove OntologyBlock
    text = re.sub(r'### OntologyBlock.*?(?=###|\Z)', '', text, flags=re.DOTALL)

    # Clean whitespace
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

    return text[:max_length]  # Limit content


def build_search_terms(block: OntologyBlock, content: str) -> Set[str]:
    """Build comprehensive set of search terms for fuzzy matching."""
    terms = set()

    # Add primary term
    if block.preferred_term:
        terms.add(block.preferred_term.lower())
        # Add individual words
        terms.update(block.preferred_term.lower().split())

    # Add alternative terms
    for alt_term in block.alt_terms:
        terms.add(alt_term.lower())
        terms.update(alt_term.lower().split())

    # Add term ID (both with and without domain prefix)
    if block.term_id:
        terms.add(block.term_id.lower())
        # Add numeric part
        if '-' in block.term_id:
            terms.add(block.term_id.split('-', 1)[1].lower())

    # Add domain
    if block.source_domain:
        terms.add(block.source_domain.lower())

    domain = block.get_domain()
    if domain:
        terms.add(domain)
        if domain in DOMAIN_CONFIG:
            terms.add(DOMAIN_CONFIG[domain]['full_name'].lower())

    # Add key terms from definition
    if block.definition:
        # Extract significant words (3+ characters, not common words)
        words = re.findall(r'\b[a-z]{3,}\b', block.definition.lower())
        common_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was'}
        terms.update(w for w in words if w not in common_words)

    return terms


def build_search_document(block: OntologyBlock, content: str, file_stem: str) -> Dict[str, Any]:
    """Build comprehensive search document with all searchable metadata."""
    # Get page title
    title = block.preferred_term if block.preferred_term else file_stem

    # Extract text content (limit for search preview)
    text_content = extract_text_content(content, max_length=300)

    # Build search terms for fuzzy matching
    search_terms = list(build_search_terms(block, content))

    # Build base document
    doc = {
        'id': file_stem,
        'title': title,
        'content': text_content,

        # Core ontology data for display and filtering
        'term_id': block.term_id,
        'preferred_term': block.preferred_term,
        'alt_terms': block.alt_terms,
        'definition': block.definition,

        # Full IRI for linking
        'iri': block.get_full_iri(),

        # Domain facets for filtering
        'domain': block.get_domain(),
        'domain_name': DOMAIN_CONFIG.get(block.get_domain(), {}).get('full_name', '') if block.get_domain() else '',
        'source_domain': block.source_domain,

        # Search optimization
        'search_terms': search_terms,

        # Quality and status for ranking
        'authority_score': block.authority_score if block.authority_score is not None else 0.5,
        'quality_score': block.quality_score if block.quality_score is not None else 0.5,
        'maturity': block.maturity,
        'status': block.status,

        # Classification for faceted search
        'owl_class': block.owl_class,
        'owl_physicality': block.owl_physicality,
        'owl_role': block.owl_role,

        # Relationships for contextual search
        'is_subclass_of': block.is_subclass_of,
        'relates_to': block.relates_to[:10] if block.relates_to else [],  # Limit for size

        # Cross-domain linking
        'cross_domain_links': block.cross_domain_links if block.cross_domain_links is not None else 0,
        'belongs_to_domain': block.belongs_to_domain,

        # Additional metadata
        'last_updated': block.last_updated,
        'public_access': block.public_access,
    }

    return doc


def process_pages(pages_dir: str, parser: OntologyBlockParser) -> List[Dict[str, Any]]:
    """Process all markdown pages and create search documents."""
    documents = []
    pages_path = Path(pages_dir)

    if not pages_path.exists():
        print(f"Warning: Pages directory not found: {pages_dir}")
        return documents

    # Track statistics
    stats = {
        'total': 0,
        'with_ontology': 0,
        'by_domain': {domain: 0 for domain in DOMAIN_CONFIG.keys()},
        'no_domain': 0
    }

    # Process each markdown file
    for md_file in sorted(pages_path.glob('*.md')):
        stats['total'] += 1

        try:
            content = md_file.read_text(encoding='utf-8')

            # Parse ontology block using shared library
            block = parser.parse_file(md_file)

            # Skip if no ontology block
            if not block or not block.term_id:
                print(f"   ⊘ Skipped (no ontology): {md_file.stem}")
                continue

            stats['with_ontology'] += 1

            # Track domain statistics
            domain = block.get_domain()
            if domain and domain in DOMAIN_CONFIG:
                stats['by_domain'][domain] += 1
            else:
                stats['no_domain'] += 1

            # Build search document
            doc = build_search_document(block, content, md_file.stem)

            documents.append(doc)
            domain_label = f"[{domain}]" if domain else "[no-domain]"
            print(f"   ✓ Indexed: {doc['title']} {domain_label}")

        except Exception as e:
            print(f"   ✗ Error processing {md_file.name}: {e}")

    # Print statistics
    print(f"\n{'='*60}")
    print(f"📊 Indexing Statistics:")
    print(f"   Total files: {stats['total']}")
    print(f"   With ontology: {stats['with_ontology']}")
    print(f"   Without ontology: {stats['total'] - stats['with_ontology']}")
    print(f"\n   By domain:")
    for domain_key in sorted(DOMAIN_CONFIG.keys()):
        domain_name = DOMAIN_CONFIG[domain_key]['full_name']
        count = stats['by_domain'][domain_key]
        print(f"      {domain_name}: {count}")
    print(f"      Uncategorized: {stats['no_domain']}")
    print(f"{'='*60}\n")

    return documents


def build_facets(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build facet metadata for search UI."""
    facets = {
        'domains': {},
        'physicality': {},
        'roles': {},
        'maturity': {},
        'status': {}
    }

    for doc in documents:
        # Domain facets
        if doc.get('domain'):
            domain = doc['domain']
            if domain not in facets['domains']:
                facets['domains'][domain] = {
                    'name': doc.get('domain_name', domain),
                    'count': 0
                }
            facets['domains'][domain]['count'] += 1

        # Physicality facets
        if doc.get('owl_physicality'):
            phys = doc['owl_physicality']
            facets['physicality'][phys] = facets['physicality'].get(phys, 0) + 1

        # Role facets
        if doc.get('owl_role'):
            role = doc['owl_role']
            facets['roles'][role] = facets['roles'].get(role, 0) + 1

        # Maturity facets
        if doc.get('maturity'):
            mat = doc['maturity']
            facets['maturity'][mat] = facets['maturity'].get(mat, 0) + 1

        # Status facets
        if doc.get('status'):
            status = doc['status']
            facets['status'][status] = facets['status'].get(status, 0) + 1

    return facets


def main():
    """Generate comprehensive search index with fuzzy search support."""
    if len(sys.argv) < 3:
        print("Usage: python generate_search_index.py <pages-dir> <output-file>")
        print("\nGenerates search index with:")
        print("  - Full IRIs for linking")
        print("  - Preferred terms and aliases")
        print("  - Comprehensive definitions")
        print("  - Domain facets for filtering")
        print("  - Fuzzy search optimization")
        print("  - Support for all 6 domains")
        sys.exit(1)

    pages_dir = sys.argv[1]
    output_file = sys.argv[2]

    print(f"🔍 Generating Search Index")
    print(f"{'='*60}")
    print(f"Source: {pages_dir}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Initialize parser
    parser = OntologyBlockParser()

    # Process pages
    print("📄 Processing pages...\n")
    documents = process_pages(pages_dir, parser)

    if not documents:
        print("\n⚠️  Warning: No documents indexed!")
        sys.exit(1)

    # Build facets
    print("🏷️  Building search facets...")
    facets = build_facets(documents)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build complete search index with metadata
    search_index = {
        'version': '2.0',
        'generated_at': None,  # Could add timestamp if needed
        'total_documents': len(documents),
        'facets': facets,
        'documents': documents
    }

    # Write search index
    print(f"💾 Writing search index to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(search_index, f, indent=2, ensure_ascii=False)

    # Statistics
    print(f"\n{'='*60}")
    print(f"✅ Search Index Generation Complete")
    print(f"{'='*60}")
    print(f"📊 Results:")
    print(f"   Documents indexed: {len(documents)}")
    print(f"   Domains: {len(facets['domains'])}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    if documents:
        # Calculate average quality
        avg_authority = sum(d.get('authority_score', 0) for d in documents) / len(documents)
        avg_quality = sum(d.get('quality_score', 0) for d in documents) / len(documents)
        print(f"\n   Quality Metrics:")
        print(f"      Average authority: {avg_authority:.2f}")
        print(f"      Average quality: {avg_quality:.2f}")

        print(f"\n   Sample documents:")
        print(f"      First: {documents[0]['title']}")
        print(f"      Last: {documents[-1]['title']}")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
