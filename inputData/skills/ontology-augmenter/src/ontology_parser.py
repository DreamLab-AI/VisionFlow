#!/usr/bin/env python3
"""
Ontology Parser Module

DEPRECATED: This module is maintained for backward compatibility only.
Please use the shared ontology_block_parser library instead:
  Location: /Ontology-Tools/tools/lib/ontology_block_parser.py

This wrapper provides compatibility by re-exporting from the shared library
and providing a compatibility layer for legacy OntologyTerm interface.

Parses Logseq markdown files with OntologyBlock headers to extract:
- Metadata (term-id, preferred-term, definition, source-domain, maturity)
- OWL axioms from clojure code blocks
- Relationships between terms
- Cross-references and wiki-style links

Builds in-memory ontology maps for navigation and augmentation.
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Import from shared library
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'Ontology-Tools' / 'tools' / 'lib'))

from ontology_block_parser import (
    OntologyBlockParser as SharedParser,
    OntologyBlock,
    DOMAIN_CONFIG,
    STANDARD_NAMESPACES
)


@dataclass
class OntologyTerm:
    """
    DEPRECATED: Legacy interface for compatibility.
    Please use OntologyBlock from shared library instead.

    Represents a single ontology term with all its metadata.
    This is a compatibility wrapper around OntologyBlock.
    """

    file_path: Path
    term_id: Optional[str] = None
    preferred_term: Optional[str] = None
    definition: Optional[str] = None
    source_domain: Optional[str] = None
    maturity: Optional[str] = None
    authority_score: Optional[float] = None
    status: Optional[str] = None
    source: Optional[str] = None
    owl_class: Optional[str] = None
    physicality: Optional[str] = None
    role: Optional[str] = None

    # Relationships extracted from markdown
    relationships: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    # OWL axioms from code blocks
    owl_axioms: List[str] = field(default_factory=list)

    # Cross-references found in content (wiki-style [[links]])
    cross_references: Set[str] = field(default_factory=set)

    # Raw OntologyBlock content
    raw_block: str = ""

    @classmethod
    def from_ontology_block(cls, block: OntologyBlock) -> 'OntologyTerm':
        """Create OntologyTerm from OntologyBlock for backward compatibility."""
        # Combine all relationships into single dict
        relationships = defaultdict(list)
        relationships['is-subclass-of'] = block.is_subclass_of
        relationships['has-part'] = block.has_part
        relationships['is-part-of'] = block.is_part_of
        relationships['requires'] = block.requires
        relationships['depends-on'] = block.depends_on
        relationships['enables'] = block.enables
        relationships['relates-to'] = block.relates_to

        # Add other relationships
        for rel_name, targets in block.other_relationships.items():
            relationships[rel_name] = targets

        # Extract cross-references (approximate from relationships)
        cross_refs = set()
        for targets in relationships.values():
            cross_refs.update(targets)

        return cls(
            file_path=block.file_path,
            term_id=block.term_id,
            preferred_term=block.preferred_term,
            definition=block.definition,
            source_domain=block.source_domain,
            maturity=block.maturity,
            authority_score=block.authority_score,
            status=block.status,
            source=", ".join(block.source) if block.source else None,
            owl_class=block.owl_class,
            physicality=block.owl_physicality,
            role=block.owl_role,
            relationships=dict(relationships),
            owl_axioms=block.owl_axioms,
            cross_references=cross_refs,
            raw_block=block.raw_block
        )


class OntologyParser:
    """
    DEPRECATED: Legacy interface for compatibility.
    Please use OntologyBlockParser from shared library instead.

    Parser for Logseq ontology markdown files.

    Extracts structured metadata from OntologyBlock headers and builds
    an in-memory map of terms with their relationships and locations.

    This is a compatibility wrapper around SharedParser (OntologyBlockParser).
    """

    def __init__(self):
        """Initialize the ontology parser."""
        self.shared_parser = SharedParser()
        self.term_map: Dict[str, OntologyTerm] = {}
        self.file_to_term: Dict[Path, str] = {}
        self.cross_reference_graph: Dict[str, Set[str]] = defaultdict(set)

    def extract_ontology_block(self, content: str) -> str:
        """
        Extract ONLY the OntologyBlock header section.
        Delegates to shared parser.
        """
        return self.shared_parser.extract_ontology_block(content)

    def extract_metadata(self, block: str) -> Dict[str, str]:
        """
        Extract metadata fields from OntologyBlock.

        Parses logseq-style properties:
        - term-id:: value
        - preferred-term:: value
        - etc.

        Args:
            block: OntologyBlock content

        Returns:
            Dictionary of metadata key-value pairs
        """
        metadata = {}

        # Pattern matches: "- property:: value" (with flexible whitespace)
        patterns = {
            'term_id': r'^\s*-\s*term-id::\s*(.+)$',
            'preferred_term': r'^\s*-\s*preferred-term::\s*(.+)$',
            'definition': r'^\s*-\s*definition::\s*(.+)$',
            'source_domain': r'^\s*-\s*source-domain::\s*(.+)$',
            'maturity': r'^\s*-\s*maturity::\s*(.+)$',
            'status': r'^\s*-\s*status::\s*(.+)$',
            'source': r'^\s*-\s*source::\s*(.+)$',
            'authority_score': r'^\s*-\s*authority-score::\s*(.+)$',
            'owl_class': r'^\s*-\s*owl:class::\s*(.+)$',
            'owl_physicality': r'^\s*-\s*owl:physicality::\s*(.+)$',
            'owl_role': r'^\s*-\s*owl:role::\s*(.+)$',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, block, re.MULTILINE)
            if match:
                metadata[key] = match.group(1).strip()

        return metadata

    def extract_relationships(self, block: str) -> Dict[str, List[str]]:
        """
        Extract relationships from #### Relationships section.

        Looks for a section like:
        - #### Relationships
          - has-part:: [[Component1]], [[Component2]]
          - uses:: [[Tool1]]

        Args:
            block: OntologyBlock content

        Returns:
            Dictionary mapping relationship names to lists of target terms
        """
        relationships = defaultdict(list)

        # Find the Relationships section
        rel_section = re.search(
            r'-\s*####\s*Relationships\s*\n(.*?)(?=-\s*####|\Z)',
            block,
            re.DOTALL | re.IGNORECASE
        )

        if not rel_section:
            return relationships

        # Parse each relationship line
        for line in rel_section.group(1).strip().split('\n'):
            # Match: "- property:: [[target1]], [[target2]]"
            match = re.match(r'-\s*([\w-]+)::\s*(.*)', line.strip())
            if match:
                prop_name = match.group(1).strip()
                targets_text = match.group(2)

                # Extract all [[wiki-style]] references
                targets = re.findall(r'\[\[([^\]]+)\]\]', targets_text)
                if targets:
                    relationships[prop_name].extend(t.strip() for t in targets)

        return dict(relationships)

    def extract_owl_axioms(self, block: str) -> List[str]:
        """
        Extract OWL axioms from clojure/owl code blocks.

        Args:
            block: OntologyBlock or full file content

        Returns:
            List of OWL axiom code blocks
        """
        # Match ```clojure or ```owl code blocks
        axioms = re.findall(
            r'```(?:clojure|owl)\s*\n(.*?)\n```',
            block,
            re.DOTALL
        )
        return [axiom.strip() for axiom in axioms]

    def extract_cross_references(self, content: str) -> Set[str]:
        """
        Extract all wiki-style [[references]] from content.

        Args:
            content: Full markdown content

        Returns:
            Set of referenced term names
        """
        references = re.findall(r'\[\[([^\]]+)\]\]', content)
        return set(ref.strip() for ref in references)

    def parse_file(self, file_path: Path) -> Optional[OntologyTerm]:
        """
        Parse a single markdown file to extract ontology term.
        Uses shared parser and converts to legacy OntologyTerm format.
        """
        # Use shared parser
        block = self.shared_parser.parse_file(file_path)
        if not block:
            return None

        # Convert to legacy format
        return OntologyTerm.from_ontology_block(block)

    def build_ontology_map(self, pages_dir: Path, pattern: str = "*.md") -> int:
        """
        Scan directory and build complete ontology map.
        Uses shared parser for parsing, maintains legacy interface.
        """
        pages_path = Path(pages_dir)
        if not pages_path.exists():
            raise FileNotFoundError(f"Directory not found: {pages_dir}")

        # Use shared parser to get blocks
        blocks = self.shared_parser.parse_directory(pages_path, pattern)

        # Convert to legacy format
        for block in blocks:
            if not block.term_id:
                continue

            term = OntologyTerm.from_ontology_block(block)

            # Store by term_id
            self.term_map[term.term_id] = term

            # Store file-to-term mapping
            self.file_to_term[block.file_path] = term.term_id

            # Store by preferred_term if available
            if term.preferred_term:
                self.term_map[term.preferred_term] = term

        # Build cross-reference graph
        self._build_cross_reference_graph()

        return len(blocks)

    def _build_cross_reference_graph(self):
        """Build bidirectional cross-reference graph."""
        for term_id, term in self.term_map.items():
            if not isinstance(term_id, str) or not term_id.startswith(
                ('AI-', 'BC-', 'MV-', 'RB-', 'DT-')
            ):
                continue

            for ref in term.cross_references:
                self.cross_reference_graph[term_id].add(ref)

            for rel_type, targets in term.relationships.items():
                for target in targets:
                    self.cross_reference_graph[term_id].add(target)

    def get_term_location(self, term_identifier: str) -> Optional[Path]:
        """
        Find the file path where a term is defined.

        Args:
            term_identifier: Term ID (BC-0123) or preferred term name

        Returns:
            Path to file containing term, or None if not found
        """
        term = self.term_map.get(term_identifier)
        return term.file_path if term else None

    def get_term(self, term_identifier: str) -> Optional[OntologyTerm]:
        """
        Get full OntologyTerm object by ID or name.

        Args:
            term_identifier: Term ID (BC-0123) or preferred term name

        Returns:
            OntologyTerm object or None if not found
        """
        return self.term_map.get(term_identifier)

    def get_related_terms(self, term_identifier: str) -> Set[str]:
        """
        Get all terms referenced by or referencing this term.

        Args:
            term_identifier: Term ID or preferred term name

        Returns:
            Set of related term identifiers
        """
        term = self.get_term(term_identifier)
        if not term:
            return set()

        related = set()

        # Add terms this term references
        related.update(term.cross_references)
        for targets in term.relationships.values():
            related.update(targets)

        # Add terms that reference this term
        if term.term_id:
            for other_id, refs in self.cross_reference_graph.items():
                if term.term_id in refs or (term.preferred_term and term.preferred_term in refs):
                    related.add(other_id)

        return related

    def get_terms_by_domain(self, domain: str) -> List[OntologyTerm]:
        """
        Get all terms from a specific domain.

        Args:
            domain: Domain name (e.g., "blockchain", "ai", "metaverse")

        Returns:
            List of OntologyTerm objects in that domain
        """
        return [
            term for term_id, term in self.term_map.items()
            if isinstance(term_id, str) and term_id.startswith(
                ('AI-', 'BC-', 'MV-', 'RB-', 'DT-')
            ) and term.source_domain and term.source_domain.lower() == domain.lower()
        ]

    def get_terms_by_maturity(self, maturity: str) -> List[OntologyTerm]:
        """
        Get all terms with specific maturity level.

        Args:
            maturity: Maturity level (e.g., "emerging", "mature", "established")

        Returns:
            List of OntologyTerm objects at that maturity level
        """
        return [
            term for term_id, term in self.term_map.items()
            if isinstance(term_id, str) and term_id.startswith(
                ('AI-', 'BC-', 'MV-', 'RB-', 'DT-')
            ) and term.maturity and term.maturity.lower() == maturity.lower()
        ]

    def find_orphan_terms(self) -> List[str]:
        """
        Find terms with no incoming or outgoing references.

        Returns:
            List of term IDs for orphaned terms
        """
        orphans = []

        for term_id, term in self.term_map.items():
            if not isinstance(term_id, str) or not term_id.startswith(
                ('AI-', 'BC-', 'MV-', 'RB-', 'DT-')
            ):
                continue

            # Check if term has any outgoing references
            has_outgoing = bool(term.cross_references or any(term.relationships.values()))

            # Check if term has any incoming references
            has_incoming = any(
                term_id in refs or (term.preferred_term and term.preferred_term in refs)
                for refs in self.cross_reference_graph.values()
            )

            if not has_outgoing and not has_incoming:
                orphans.append(term_id)

        return orphans

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the parsed ontology.

        Returns:
            Dictionary with counts of various ontology elements
        """
        # Count unique term IDs (exclude duplicate entries by preferred_term)
        unique_terms = {
            term_id for term_id in self.term_map.keys()
            if isinstance(term_id, str) and term_id.startswith(
                ('AI-', 'BC-', 'MV-', 'RB-', 'DT-')
            )
        }

        stats = {
            'total_terms': len(unique_terms),
            'files_processed': len(self.file_to_term),
            'total_relationships': sum(
                len(rels) for term in self.term_map.values()
                for rels in term.relationships.values()
            ),
            'terms_with_owl_axioms': sum(
                1 for term in self.term_map.values() if term.owl_axioms
            ),
            'total_cross_references': sum(
                len(term.cross_references) for term in self.term_map.values()
            ),
        }

        # Domain breakdown
        for prefix in ['AI-', 'BC-', 'MV-', 'RB-', 'DT-']:
            count = sum(1 for tid in unique_terms if tid.startswith(prefix))
            stats[f'{prefix[:-1].lower()}_terms'] = count

        return stats


def main():
    """Example usage of OntologyParser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ontology_parser.py <pages_directory>")
        sys.exit(1)

    pages_dir = Path(sys.argv[1])

    parser = OntologyParser()
    print(f"Scanning {pages_dir}...")

    count = parser.build_ontology_map(pages_dir)
    print(f"\n✅ Parsed {count} ontology terms")

    stats = parser.get_statistics()
    print("\n📊 Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Example queries
    print("\n🔍 Example queries:")

    # Find blockchain terms
    bc_terms = parser.get_terms_by_domain("blockchain")
    print(f"   Blockchain terms: {len(bc_terms)}")

    # Find mature terms
    mature = parser.get_terms_by_maturity("mature")
    print(f"   Mature terms: {len(mature)}")

    # Find orphans
    orphans = parser.find_orphan_terms()
    print(f"   Orphaned terms: {len(orphans)}")


if __name__ == '__main__':
    main()
