#!/usr/bin/env python3
"""
Add Missing rdfs:comment to OWL Classes

Reads source Logseq files to extract definitions and adds them as
rdfs:comment annotations to OWL classes in batches.

Updated to use shared ontology_block_parser library.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal
    from rdflib.term import URIRef
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Install with: pip install rdflib")
    sys.exit(1)

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Ontology-Tools' / 'tools' / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock


class CommentAdder:
    """Adds missing rdfs:comment annotations to OWL ontology."""

    def __init__(self, ttl_file: Path, pages_dir: Path):
        self.ttl_file = ttl_file
        self.pages_dir = pages_dir
        self.graph = Graph()
        self.parser = OntologyBlockParser()
        self.blocks_by_id = {}  # term_id -> OntologyBlock
        self.DC = Namespace("http://purl.org/dc/elements/1.1/")

    def load_ontology(self) -> bool:
        """Load the OWL ontology."""
        try:
            print(f"Loading ontology: {self.ttl_file}")
            self.graph.parse(self.ttl_file, format='turtle')
            print(f"✓ Loaded {len(self.graph):,} triples\n")
            return True
        except Exception as e:
            print(f"✗ Failed to load ontology: {e}")
            return False

    def load_source_data(self) -> bool:
        """Load source Logseq pages to get definitions."""
        try:
            print(f"Parsing source files: {self.pages_dir}")
            blocks = self.parser.parse_directory(self.pages_dir)

            # Build lookup by term_id
            for block in blocks:
                if block.term_id:
                    self.blocks_by_id[block.term_id] = block

            print(f"✓ Parsed {len(blocks)} ontology blocks\n")
            return True
        except Exception as e:
            print(f"✗ Failed to parse source files: {e}")
            return False

    def extract_first_paragraph(self, file_path: Path) -> str:
        """Extract first substantial paragraph from markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip OntologyBlock header
            lines = content.split('\n')
            in_ontology_block = False
            paragraphs = []
            current = []

            for line in lines:
                # Skip OntologyBlock section
                if '### OntologyBlock' in line or 'OntologyBlock' in line:
                    in_ontology_block = True
                    continue
                if in_ontology_block and line.strip().startswith('##'):
                    in_ontology_block = False
                if in_ontology_block:
                    continue

                # Skip headings
                if line.strip().startswith('#'):
                    continue

                # Collect lines
                if line.strip():
                    # Remove bullet points and clean
                    cleaned = line.strip().lstrip('-').strip()
                    if cleaned and not cleaned.startswith('[[') and len(cleaned) > 20:
                        current.append(cleaned)
                elif current:
                    paragraphs.append(' '.join(current))
                    current = []
                    if len(paragraphs) >= 2:  # Got enough
                        break

            if current:
                paragraphs.append(' '.join(current))

            # Return first substantial paragraph
            for para in paragraphs:
                if len(para) > 50:
                    return para

            return ""
        except Exception:
            return ""

    def find_missing_comments(self) -> List[Tuple[URIRef, str, str]]:
        """
        Find classes without rdfs:comment.

        Returns:
            List of (class_uri, identifier, definition) tuples
        """
        missing = []

        for cls in self.graph.subjects(RDF.type, OWL.Class):
            # Check if has rdfs:comment
            if list(self.graph.objects(cls, RDFS.comment)):
                continue

            # Get identifier to lookup in source
            identifiers = list(self.graph.objects(cls, self.DC.identifier))
            if not identifiers:
                continue

            identifier = str(identifiers[0])

            # Lookup in parsed blocks
            block = self.blocks_by_id.get(identifier)
            if not block:
                continue

            # Try explicit definition first
            definition = block.definition

            # If no definition, extract from file content
            if not definition and block.file_path:
                definition = self.extract_first_paragraph(block.file_path)

            if not definition:
                continue

            missing.append((cls, identifier, definition))

        return missing

    def add_comments_batch(self, batch: List[Tuple[URIRef, str, str]]) -> int:
        """
        Add rdfs:comment to a batch of classes.

        Returns:
            Number of comments added
        """
        added = 0

        for cls_uri, identifier, definition in batch:
            # Clean definition
            clean_def = definition.strip()

            # Remove common prefixes
            prefixes_to_remove = [
                "### Primary Definition",
                "## Definition",
                "# Definition",
                "**Definition:**",
                "Definition:",
            ]

            for prefix in prefixes_to_remove:
                if clean_def.startswith(prefix):
                    clean_def = clean_def[len(prefix):].strip()

            # Remove leading dashes/bullets
            clean_def = clean_def.lstrip('- ').strip()

            # Limit length
            if len(clean_def) > 500:
                # Try to cut at sentence boundary
                sentences = clean_def[:500].split('. ')
                if len(sentences) > 1:
                    clean_def = '. '.join(sentences[:-1]) + '.'
                else:
                    clean_def = clean_def[:500].strip() + '...'

            # Add to graph
            if clean_def:
                self.graph.add((cls_uri, RDFS.comment, Literal(clean_def, lang='en')))
                added += 1

        return added

    def save_ontology(self, output_file: Path):
        """Save updated ontology."""
        try:
            print(f"\nSaving updated ontology to: {output_file}")
            self.graph.serialize(destination=str(output_file), format='turtle')
            print(f"✓ Saved {len(self.graph):,} triples")
        except Exception as e:
            print(f"✗ Failed to save: {e}")
            raise

    def process(self, output_file: Path, batch_size: int = 100, dry_run: bool = False) -> Dict[str, int]:
        """
        Process ontology to add missing comments.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_classes': 0,
            'missing_comments': 0,
            'comments_added': 0,
            'no_definition': 0,
        }

        # Count total classes
        stats['total_classes'] = len(list(self.graph.subjects(RDF.type, OWL.Class)))

        # Find classes missing comments
        missing = self.find_missing_comments()
        stats['missing_comments'] = len(missing)

        print(f"Classes without comments: {len(missing)}")
        print(f"Batch size: {batch_size}\n")

        if dry_run:
            print("DRY RUN - No changes will be made\n")
            print("Sample definitions that would be added:\n")
            for cls_uri, identifier, definition in missing[:5]:
                print(f"{identifier}:")
                print(f"  {definition[:100]}...")
                print()
            return stats

        # Process in batches
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(missing) + batch_size - 1) // batch_size

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} classes)...")
            added = self.add_comments_batch(batch)
            stats['comments_added'] += added
            print(f"  ✓ Added {added} comments")

        # Save if changes were made
        if stats['comments_added'] > 0:
            self.save_ontology(output_file)

        return stats


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Add missing rdfs:comment annotations to OWL ontology'
    )
    parser.add_argument(
        'ttl_file',
        type=Path,
        help='Input TTL ontology file'
    )
    parser.add_argument(
        'pages_dir',
        type=Path,
        help='Directory containing source Logseq pages'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file (default: overwrite input)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    if not args.ttl_file.exists():
        print(f"Error: TTL file not found: {args.ttl_file}")
        sys.exit(1)

    if not args.pages_dir.exists():
        print(f"Error: Pages directory not found: {args.pages_dir}")
        sys.exit(1)

    output_file = args.output or args.ttl_file

    print("=" * 80)
    print("Add Missing rdfs:comment Annotations")
    print("=" * 80)
    print(f"Input: {args.ttl_file}")
    print(f"Source: {args.pages_dir}")
    print(f"Output: {output_file}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print("=" * 80)
    print()

    # Process
    adder = CommentAdder(args.ttl_file, args.pages_dir)

    if not adder.load_ontology():
        sys.exit(1)

    if not adder.load_source_data():
        sys.exit(1)

    stats = adder.process(output_file, args.batch_size, args.dry_run)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total classes: {stats['total_classes']}")
    print(f"Missing comments: {stats['missing_comments']}")
    print(f"Comments {'would be ' if args.dry_run else ''}added: {stats['comments_added']}")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")

    print("=" * 80)


if __name__ == '__main__':
    main()
