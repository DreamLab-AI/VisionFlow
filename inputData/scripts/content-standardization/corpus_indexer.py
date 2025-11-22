#!/usr/bin/env python3
"""
Corpus Indexer for Wiki Link Enhancement

Scans knowledge graph to build searchable index of all terms,
enabling intelligent wiki link suggestions.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class CorpusIndexer:
    """Index all terms in the knowledge graph for wiki linking."""

    def __init__(self, pages_dir: Path):
        self.pages_dir = Path(pages_dir)
        self.term_index: Dict[str, Dict] = {}
        self.alt_terms_index: Dict[str, str] = {}  # alt-term → preferred-term
        self.term_frequencies: Dict[str, int] = defaultdict(int)

    def build_index(self) -> Dict:
        """
        Build complete index of all terms.

        Returns:
            dict: Index with term metadata
        """
        print(f"Scanning {self.pages_dir} for terms...")

        md_files = list(self.pages_dir.glob("**/*.md"))
        print(f"Found {len(md_files)} markdown files")

        for md_file in md_files:
            if md_file.name.startswith('.'):
                continue

            try:
                self._index_file(md_file)
            except Exception as e:
                print(f"Warning: Error indexing {md_file.name}: {e}")

        print(f"\nIndex built:")
        print(f"  - {len(self.term_index)} preferred terms")
        print(f"  - {len(self.alt_terms_index)} alternative terms")

        return {
            'terms': self.term_index,
            'alt_terms': self.alt_terms_index,
            'frequencies': dict(self.term_frequencies)
        }

    def _index_file(self, file_path: Path):
        """Index a single markdown file."""
        content = file_path.read_text(encoding='utf-8')

        # Extract OntologyBlock
        ontology_match = re.search(
            r'- ### OntologyBlock\s+id::\s*(\S+).*?(?=\n- |\n###|\Z)',
            content,
            re.DOTALL
        )

        if not ontology_match:
            return

        ontology_block = ontology_match.group(0)

        # Extract key fields
        term_id = self._extract_field(ontology_block, 'term-id')
        preferred_term = self._extract_field(ontology_block, 'preferred-term')
        domain = self._extract_field(ontology_block, 'source-domain')
        status = self._extract_field(ontology_block, 'status')
        quality_score = self._extract_field(ontology_block, 'quality-score')
        definition = self._extract_field(ontology_block, 'definition')

        if not preferred_term:
            # Fallback to filename
            preferred_term = file_path.stem.replace('-', ' ')

        # Extract alternative terms
        alt_terms = self._extract_alt_terms(ontology_block)

        # Store in index
        self.term_index[preferred_term] = {
            'term_id': term_id,
            'file': str(file_path.relative_to(self.pages_dir.parent)),
            'domain': domain,
            'status': status,
            'quality_score': float(quality_score) if quality_score else 0.0,
            'definition': definition[:200] if definition else '',
            'alt_terms': alt_terms
        }

        # Index alternative terms
        for alt in alt_terms:
            self.alt_terms_index[alt.lower()] = preferred_term

        # Also index the preferred term itself
        self.alt_terms_index[preferred_term.lower()] = preferred_term

    def _extract_field(self, text: str, field: str) -> str:
        """Extract a field value from ontology block."""
        pattern = rf'{field}::\s*(.+?)(?=\n|$)'
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ''

    def _extract_alt_terms(self, text: str) -> List[str]:
        """Extract alternative terms from ontology block."""
        alt_terms = []

        # Look for alt-terms field
        pattern = r'alt-terms::\s*(.+?)(?=\n\s*-|$)'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            alt_text = match.group(1)
            # Split by commas or newlines
            terms = re.split(r'[,\n]', alt_text)
            for term in terms:
                term = term.strip()
                if term:
                    alt_terms.append(term)

        return alt_terms

    def find_term(self, text: str) -> Tuple[str, float]:
        """
        Find best matching term for given text.

        Args:
            text: Text to find term for

        Returns:
            tuple: (preferred_term, confidence_score)
        """
        text_lower = text.lower().strip()

        # Exact match in alt terms
        if text_lower in self.alt_terms_index:
            preferred = self.alt_terms_index[text_lower]
            return (preferred, 1.0)

        # Partial match (fuzzy)
        best_match = None
        best_score = 0.0

        for alt_term, preferred in self.alt_terms_index.items():
            # Simple similarity: ratio of matching characters
            if text_lower in alt_term or alt_term in text_lower:
                score = min(len(text_lower), len(alt_term)) / max(len(text_lower), len(alt_term))
                if score > best_score:
                    best_score = score
                    best_match = preferred

        return (best_match, best_score) if best_match else (None, 0.0)

    def get_linkable_terms(self, confidence_threshold: float = 0.8) -> List[str]:
        """
        Get list of terms suitable for wiki linking.

        Args:
            confidence_threshold: Minimum quality score

        Returns:
            list: Sorted list of linkable terms
        """
        linkable = []

        for term, info in self.term_index.items():
            if info['quality_score'] >= confidence_threshold:
                linkable.append(term)

        return sorted(linkable)

    def save_index(self, output_file: Path):
        """Save index to JSON file."""
        index_data = {
            'terms': self.term_index,
            'alt_terms': self.alt_terms_index,
            'frequencies': dict(self.term_frequencies),
            'stats': {
                'total_terms': len(self.term_index),
                'total_alt_terms': len(self.alt_terms_index)
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)

        print(f"Index saved to {output_file}")

    @staticmethod
    def load_index(index_file: Path) -> 'CorpusIndexer':
        """Load index from JSON file."""
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create indexer instance
        indexer = CorpusIndexer(Path('.'))
        indexer.term_index = data['terms']
        indexer.alt_terms_index = data['alt_terms']
        indexer.term_frequencies = defaultdict(int, data.get('frequencies', {}))

        return indexer


def main():
    """Build corpus index."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python corpus_indexer.py <pages_directory> [output_file]")
        sys.exit(1)

    pages_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('corpus_index.json')

    indexer = CorpusIndexer(pages_dir)
    indexer.build_index()
    indexer.save_index(output_file)

    print(f"\nIndex statistics:")
    print(f"  Total terms: {len(indexer.term_index)}")
    print(f"  Total alternatives: {len(indexer.alt_terms_index)}")
    print(f"  Linkable terms (quality ≥ 0.8): {len(indexer.get_linkable_terms())}")


if __name__ == '__main__':
    main()
