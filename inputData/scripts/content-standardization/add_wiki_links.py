#!/usr/bin/env python3
"""
Apply high-confidence wiki links to content.
Uses corpus index to add [[Term]] links for known concepts.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys


def load_corpus_index(index_path: Path) -> Dict:
    """Load the corpus index."""
    return json.loads(index_path.read_text())


def extract_existing_links(content: str) -> Set[str]:
    """Extract all existing [[links]] from content."""
    return set(re.findall(r'\[\[([^\]]+)\]\]', content))


def extract_ontology_block(content: str) -> str:
    """Extract ontology block to avoid linking within it."""
    match = re.search(
        r'- ### OntologyBlock.*?(?=\n##|\n###[^#]|\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    return match.group(0) if match else ""


def should_link_term(
    term: str,
    content: str,
    existing_links: Set[str],
    ontology_block: str,
    high_confidence_only: bool = True
) -> bool:
    """Determine if a term should be linked."""

    # Don't link if already linked
    if term in existing_links:
        return False

    # Don't link very short terms (too generic)
    if len(term) < 4:
        return False

    # Don't link common words (even if in corpus)
    common_words = {
        'data', 'system', 'model', 'network', 'process', 'method',
        'user', 'value', 'node', 'layer', 'agent', 'token', 'block'
    }
    if term.lower() in common_words:
        return False

    # For high confidence, require case-insensitive match
    if high_confidence_only:
        # Check if term appears (case-insensitive for better coverage)
        pattern = r'\b' + re.escape(term) + r'\b'
        if not re.search(pattern, content, re.IGNORECASE):
            return False

    return True


def add_wiki_links_to_section(
    section: str,
    terms: List[str],
    existing_links: Set[str],
    ontology_block: str,
    max_links_per_section: int = 5
) -> Tuple[str, int]:
    """Add wiki links to a section. Returns (modified_section, links_added)."""

    modified = section
    links_added = 0
    terms_linked_this_section = set()

    # Sort terms by length (longest first to avoid substring issues)
    sorted_terms = sorted(terms, key=len, reverse=True)

    for term in sorted_terms:
        if links_added >= max_links_per_section:
            break

        if term in terms_linked_this_section:
            continue

        if not should_link_term(term, section, existing_links, ontology_block):
            continue

        # Create pattern for first unlinked occurrence
        # Match term as whole word, case-insensitive
        pattern = r'\b' + re.escape(term) + r'\b'

        # Find first match not already inside [[...]]
        match = re.search(pattern, modified, re.IGNORECASE)
        count = 0

        if match:
            # Check if this match is inside an existing link
            start = match.start()
            end = match.end()

            # Look backwards for [[ and forwards for ]]
            before = modified[max(0, start-100):start]
            after = modified[end:min(len(modified), end+100)]

            # If we find [[ before and ]] after without intervening ]] or [[, skip
            has_open_before = '[[' in before and ']]' not in before[before.rfind('[['):]
            has_close_after = ']]' in after and '[[' not in after[:after.find(']]')]

            if not (has_open_before and has_close_after):
                # This is not inside a link, so we can link it
                new_section = modified[:start] + f'[[{term}]]' + modified[end:]
                modified = new_section
                count = 1

        if count > 0:
            modified = new_section
            links_added += count
            terms_linked_this_section.add(term)
            existing_links.add(term)

    return modified, links_added


def add_wiki_links(
    content: str,
    corpus_index: Dict,
    high_confidence_only: bool = True
) -> Tuple[str, int]:
    """Add wiki links to content. Returns (modified_content, total_links_added)."""

    # Extract ontology block (don't modify it)
    ontology_block = extract_ontology_block(content)

    # Get existing links
    existing_links = extract_existing_links(content)

    # Get all terms from corpus index
    # Handle nested structure with 'terms' key
    terms_data = corpus_index.get('terms', corpus_index)

    all_terms = set()
    for term_name, term_data in terms_data.items():
        # Add the term name itself
        all_terms.add(term_name)
        # Add alternative terms if present
        all_terms.update(term_data.get('alt_terms', []))

    # Filter to high-confidence terms (longer, more specific)
    if high_confidence_only:
        # Multi-word terms OR single-word technical terms (long and specific)
        terms = [t for t in all_terms if (len(t) >= 8 and ' ' in t) or len(t) >= 12]
    else:
        terms = list(all_terms)

    # Split content into sections
    sections = re.split(r'(##[^#\n].*?\n)', content)

    total_links_added = 0
    modified_sections = []

    for i, section in enumerate(sections):
        # Skip ontology block
        if ontology_block and ontology_block in section:
            modified_sections.append(section)
            continue

        # Skip headers
        if section.startswith('##'):
            modified_sections.append(section)
            continue

        # Add links to content sections
        modified_section, links_added = add_wiki_links_to_section(
            section,
            terms,
            existing_links,
            ontology_block,
            max_links_per_section=3  # Conservative for first pass
        )

        modified_sections.append(modified_section)
        total_links_added += links_added

    return ''.join(modified_sections), total_links_added


def process_file(
    file_path: Path,
    corpus_index: Dict,
    dry_run: bool = True
) -> Dict:
    """Process a single file. Returns stats."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content

        modified_content, links_added = add_wiki_links(
            content,
            corpus_index,
            high_confidence_only=True
        )

        if links_added > 0 and not dry_run:
            file_path.write_text(modified_content, encoding='utf-8')

        return {
            'file': str(file_path.name),
            'links_added': links_added,
            'success': True
        }

    except Exception as e:
        return {
            'file': str(file_path.name),
            'links_added': 0,
            'success': False,
            'error': str(e)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add high-confidence wiki links')
    parser.add_argument('--directory', default='../../mainKnowledgeGraph/pages')
    parser.add_argument('--corpus-index', default='corpus_index.json')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--report', help='Output report JSON')
    parser.add_argument('--min-links', type=int, default=1, help='Minimum links to show in report')

    args = parser.parse_args()

    # Load corpus index
    corpus_index_path = Path(args.corpus_index)
    if not corpus_index_path.exists():
        print(f"Error: Corpus index not found: {corpus_index_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus index from {corpus_index_path}...")
    corpus_index = load_corpus_index(corpus_index_path)
    terms_data = corpus_index.get('terms', corpus_index)
    print(f"Loaded {len(terms_data)} terms from corpus")

    # Process files
    pages_dir = Path(args.directory)
    markdown_files = list(pages_dir.rglob("*.md"))
    print(f"Processing {len(markdown_files)} files...")

    results = []
    total_links = 0
    files_modified = 0

    for file_path in markdown_files:
        result = process_file(file_path, corpus_index, dry_run=not args.apply)
        results.append(result)

        if result['links_added'] > 0:
            total_links += result['links_added']
            files_modified += 1
            if result['links_added'] >= args.min_links:
                status = "✓" if args.apply else "→"
                print(f"{status} {result['file']}: +{result['links_added']} links")

    print(f"\n{'=' * 60}")
    print(f"Total files processed: {len(markdown_files)}")
    print(f"Files with new links: {files_modified}")
    print(f"Total links added: {total_links}")
    print(f"{'=' * 60}")

    if not args.apply:
        print("\nDry run mode. Use --apply to make changes.")

    # Save report
    if args.report:
        report = {
            'summary': {
                'total_files': len(markdown_files),
                'files_modified': files_modified,
                'total_links_added': total_links,
                'dry_run': not args.apply
            },
            'files': [r for r in results if r['links_added'] > 0]
        }
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\n✓ Report saved to {args.report}")


if __name__ == "__main__":
    main()
