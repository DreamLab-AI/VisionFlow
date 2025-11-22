#!/usr/bin/env python3
"""
Add semantic parent relationships to AI domain orphan concepts.
Reads mapping from ai-orphan-taxonomy-mapping.md and updates files.

Updated to use shared ontology_block_parser library for better compatibility.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Ontology-Tools' / 'tools' / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock

# Default paths - can be overridden via command line
DEFAULT_PAGES_DIR = Path(__file__).parent.parent / 'mainKnowledgeGraph' / 'pages'
DEFAULT_MAPPING_FILE = Path(__file__).parent.parent / 'mainKnowledgeGraph' / 'docs' / 'ai-orphan-taxonomy-mapping.md'

def parse_mapping_file(mapping_path: Path) -> Dict[str, str]:
    """Parse the taxonomy mapping document to extract concept->parent mappings."""
    mappings = {}

    with open(mapping_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all table rows with mappings
    # Format: | Concept | Term ID | Recommended Parent | Justification |
    table_pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'

    for match in re.finditer(table_pattern, content):
        concept = match.group(1).strip()
        parent = match.group(3).strip()

        # Skip header rows
        if concept in ['Concept', '------', '---'] or parent in ['Recommended Parent', '---', '------']:
            continue

        # Normalize concept names (handle variations)
        if concept and parent:
            mappings[concept] = parent

            # Also add normalized versions
            # Convert spaces to various formats
            normalized = concept.replace(' ', '')
            if normalized != concept:
                mappings[normalized] = parent

            # Add hyphenated version
            hyphenated = concept.replace(' ', '-')
            if hyphenated != concept:
                mappings[hyphenated] = parent

    return mappings

def get_concept_from_filename(filename: str) -> str:
    """Extract concept name from filename."""
    # Remove .md extension
    name = filename.replace('.md', '')
    # Handle URL encoding
    name = name.replace('%3A', ':').replace('%2F', '/')
    return name

def normalize_for_lookup(name: str) -> List[str]:
    """Generate multiple normalized versions for lookup."""
    variants = [name]

    # Remove parenthetical suffixes like "(AI-0431)"
    base = re.sub(r'\s*\([^)]+\)\s*$', '', name)
    if base != name:
        variants.append(base)

    # Without spaces
    variants.append(name.replace(' ', ''))
    variants.append(base.replace(' ', ''))

    # With hyphens instead of spaces
    variants.append(name.replace(' ', '-'))
    variants.append(base.replace(' ', '-'))

    # Handle AI- prefixed terms
    if name.startswith('AI-'):
        # Extract the descriptive part after the ID
        parts = name.split('-', 2)
        if len(parts) >= 3:
            desc = parts[2].replace('-', ' ').title()
            variants.append(desc)
            variants.append(desc.replace(' ', ''))

    return variants

def find_parent_for_concept(concept: str, mappings: Dict[str, str]) -> Optional[str]:
    """Find parent for a concept using various normalization strategies."""
    variants = normalize_for_lookup(concept)

    for variant in variants:
        if variant in mappings:
            return mappings[variant]
        # Case-insensitive lookup
        for key, value in mappings.items():
            if key.lower() == variant.lower():
                return value

    return None

def has_is_subclass_of(content: str) -> bool:
    """Check if file already has is-subclass-of relationship."""
    return bool(re.search(r'is-subclass-of::', content, re.IGNORECASE))

def find_relationships_section(content: str) -> Tuple[int, int]:
    """Find the Relationships section in the content."""
    # Look for ### Relationships or ## Relationships or #### Relationships
    match = re.search(r'^(#{2,4})\s*Relationships?\s*$', content, re.MULTILINE | re.IGNORECASE)
    if match:
        start = match.end()
        # Find the next section header of same or higher level
        level = len(match.group(1))
        next_section = re.search(rf'^#{{{1},{level}}}\s+\w', content[start:], re.MULTILINE)
        if next_section:
            end = start + next_section.start()
        else:
            end = len(content)
        return start, end
    return -1, -1

def add_parent_relationship(content: str, parent: str) -> str:
    """Add is-subclass-of relationship to content."""
    parent_line = f"- is-subclass-of:: [[{parent}]]"

    # Try to find Relationships section
    start, end = find_relationships_section(content)

    if start != -1:
        # Insert after the section header
        section_content = content[start:end]

        # Check if there's already content in the section
        lines = section_content.split('\n')
        insert_pos = start

        # Find first non-empty line after header
        for i, line in enumerate(lines):
            if line.strip() and i > 0:
                # Insert before existing content
                insert_pos = start + sum(len(l) + 1 for l in lines[:i])
                break
        else:
            # Section is empty or only whitespace
            insert_pos = start + 1

        # Insert the parent relationship
        return content[:insert_pos] + '\n' + parent_line + content[insert_pos:]

    else:
        # No Relationships section - create one
        # Look for a good insertion point (after properties, before main content)

        # Check for OWL Axioms section
        owl_match = re.search(r'^#{2,4}\s*OWL\s*Axiom', content, re.MULTILINE | re.IGNORECASE)

        # Check for CrossDomainBridges
        bridge_match = re.search(r'^#{2,4}\s*CrossDomainBridges', content, re.MULTILINE)

        # Check for Definition section
        def_match = re.search(r'^#{2,4}\s*Definition', content, re.MULTILINE | re.IGNORECASE)

        # Determine insertion point
        if owl_match:
            insert_pos = owl_match.start()
        elif bridge_match:
            insert_pos = bridge_match.start()
        elif def_match:
            insert_pos = def_match.start()
        else:
            # Insert after front matter / properties
            # Look for first ## or ### that's not meta/properties
            section_match = re.search(r'^#{2,3}\s+(?!Properties|Metadata)', content, re.MULTILINE)
            if section_match:
                insert_pos = section_match.start()
            else:
                # Just append at end
                insert_pos = len(content)

        # Create the Relationships section
        new_section = f"\n### Relationships\n{parent_line}\n\n"

        return content[:insert_pos] + new_section + content[insert_pos:]

def is_ai_related_block(block: OntologyBlock) -> bool:
    """Determine if block is AI-related based on term_id and source_domain."""
    # Check term_id
    if block.term_id and block.term_id.startswith('AI-'):
        return True

    # Check source_domain
    if block.source_domain and 'ai' in block.source_domain.lower():
        return True

    # Check domain detection
    domain = block.get_domain()
    if domain == 'ai':
        return True

    return False

def process_files(pages_dir: Path, mapping_file: Path, dry_run: bool = False) -> Tuple[int, int, List[str]]:
    """Process all AI files and add parent relationships."""
    # Load mappings
    mappings = parse_mapping_file(mapping_file)
    print(f"Loaded {len(mappings)} concept mappings")

    # Parse ontology blocks
    parser = OntologyBlockParser()
    print(f"Parsing ontology blocks from {pages_dir}...")
    blocks = parser.parse_directory(pages_dir)
    print(f"Parsed {len(blocks)} blocks")

    # Filter to AI-related blocks
    ai_blocks = [b for b in blocks if is_ai_related_block(b)]
    print(f"Found {len(ai_blocks)} AI-related blocks\n")

    updated_count = 0
    skipped_count = 0
    updated_files = []

    for block in ai_blocks:
        try:
            with open(block.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {block.file_path.name}: {e}")
            continue

        # Check if already has is-subclass-of
        if block.is_subclass_of:
            skipped_count += 1
            continue

        # Get concept name and find parent
        concept = block.preferred_term or block.term_id or ''
        parent = find_parent_for_concept(concept, mappings)

        if not parent:
            # Try additional lookups for common patterns
            base_concept = re.sub(r'\s*\([^)]+\)\s*$', '', concept)
            parent = find_parent_for_concept(base_concept, mappings)

        if not parent:
            # Special handling for specific patterns
            if 'Edge' in concept and 'AI' in concept:
                parent = 'AIApplications'
            elif 'Privacy' in concept:
                parent = 'AIGovernance'
            elif 'Governance' in concept and 'AI' in concept:
                parent = 'AIGovernance'
            elif 'Safety' in concept and ('AI' in concept or 'Fine' in concept):
                parent = 'AISafety'
            elif 'Risk' in concept and 'AI' in concept:
                parent = 'AIGovernance'

        if parent:
            new_content = add_parent_relationship(content, parent)

            if not dry_run:
                with open(block.file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

            updated_count += 1
            updated_files.append(f"{block.file_path.name} -> {parent}")
            print(f"Updated: {block.file_path.name} -> {parent}")
        else:
            print(f"No mapping found for: {concept}")

    return updated_count, skipped_count, updated_files

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Add semantic parent relationships to AI domain orphan concepts'
    )
    parser.add_argument(
        '--pages-dir',
        type=Path,
        default=DEFAULT_PAGES_DIR,
        help='Directory containing ontology markdown files'
    )
    parser.add_argument(
        '--mapping-file',
        type=Path,
        default=DEFAULT_MAPPING_FILE,
        help='Taxonomy mapping file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - No files will be modified\n")

    # Validate paths
    if not args.pages_dir.exists():
        print(f"Error: Pages directory not found: {args.pages_dir}")
        sys.exit(1)

    if not args.mapping_file.exists():
        print(f"Error: Mapping file not found: {args.mapping_file}")
        sys.exit(1)

    updated, skipped, files = process_files(args.pages_dir, args.mapping_file, dry_run=args.dry_run)

    print(f"\n{'Would update' if args.dry_run else 'Updated'}: {updated} files")
    print(f"Skipped (already have parent): {skipped} files")

    if updated > 0:
        print(f"\nFiles {'to be ' if args.dry_run else ''}updated:")
        for f in files[:50]:
            print(f"  {f}")
        if len(files) > 50:
            print(f"  ... and {len(files) - 50} more")

if __name__ == "__main__":
    main()
