#!/usr/bin/env python3
"""
Shared Ontology Block Parser Library
=====================================

Parses the canonical ontology block format from Logseq markdown files.
Supports all 6 domains: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies.

Based on canonical-ontology-block.md specification v1.0.0

Usage:
    from ontology_block_parser import OntologyBlockParser, OntologyBlock

    parser = OntologyBlockParser()
    block = parser.parse_file('path/to/file.md')

    if block:
        print(f"Term ID: {block.term_id}")
        print(f"IRI: {block.get_full_iri()}")
        print(f"Domain: {block.get_domain()}")
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# Domain configuration
DOMAIN_CONFIG = {
    'ai': {
        'prefix': 'AI-',
        'namespace': 'http://narrativegoldmine.com/ai#',
        'full_name': 'Artificial Intelligence',
        'extension_properties': ['algorithm-type', 'computational-complexity']
    },
    'bc': {
        'prefix': 'BC-',
        'namespace': 'http://narrativegoldmine.com/blockchain#',
        'full_name': 'Blockchain',
        'extension_properties': ['consensus-mechanism', 'decentralization-level']
    },
    'rb': {
        'prefix': 'RB-',
        'namespace': 'http://narrativegoldmine.com/robotics#',
        'full_name': 'Robotics',
        'extension_properties': ['physicality', 'autonomy-level']
    },
    'mv': {
        'prefix': 'MV-',
        'namespace': 'http://narrativegoldmine.com/metaverse#',
        'full_name': 'Metaverse',
        'extension_properties': ['immersion-level', 'interaction-mode']
    },
    'tc': {
        'prefix': 'TC-',
        'namespace': 'http://narrativegoldmine.com/telecollaboration#',
        'full_name': 'Telecollaboration',
        'extension_properties': ['collaboration-type', 'communication-mode']
    },
    'dt': {
        'prefix': 'DT-',
        'namespace': 'http://narrativegoldmine.com/disruptive-tech#',
        'full_name': 'Disruptive Technologies',
        'extension_properties': ['disruption-level', 'maturity-stage']
    }
}

# Standard namespaces
STANDARD_NAMESPACES = {
    'owl': 'http://www.w3.org/2002/07/owl#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
    'dcterms': 'http://purl.org/dc/terms/',
    'skos': 'http://www.w3.org/2004/02/skos/core#'
}


@dataclass
class OntologyBlock:
    """
    Represents a parsed ontology block with all metadata and relationships.
    """

    # File location
    file_path: Path

    # === Tier 1: Required Properties ===
    # Identification
    ontology: bool = True
    term_id: Optional[str] = None
    preferred_term: Optional[str] = None
    source_domain: Optional[str] = None
    status: Optional[str] = None
    public_access: Optional[bool] = None
    last_updated: Optional[str] = None

    # Definition
    definition: Optional[str] = None

    # Semantic Classification
    owl_class: Optional[str] = None  # Full IRI format (e.g., "ai:LargeLanguageModel")
    owl_physicality: Optional[str] = None
    owl_role: Optional[str] = None

    # Relationships
    is_subclass_of: List[str] = field(default_factory=list)

    # === Tier 2: Recommended Properties ===
    # Identification (Tier 2)
    alt_terms: List[str] = field(default_factory=list)
    version: Optional[str] = None
    quality_score: Optional[float] = None
    cross_domain_links: Optional[int] = None

    # Definition (Tier 2)
    maturity: Optional[str] = None
    source: List[str] = field(default_factory=list)
    authority_score: Optional[float] = None
    scope_note: Optional[str] = None

    # Semantic Classification (Tier 2)
    owl_inferred_class: Optional[str] = None
    belongs_to_domain: List[str] = field(default_factory=list)

    # Relationships (Tier 2)
    has_part: List[str] = field(default_factory=list)
    is_part_of: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    enables: List[str] = field(default_factory=list)
    relates_to: List[str] = field(default_factory=list)

    # === Additional properties ===
    implemented_in_layer: List[str] = field(default_factory=list)

    # Cross-domain bridges
    bridges_to: List[str] = field(default_factory=list)
    bridges_from: List[str] = field(default_factory=list)

    # OWL axioms from code blocks
    owl_axioms: List[str] = field(default_factory=list)

    # Domain-specific extension properties
    domain_extensions: Dict[str, Any] = field(default_factory=dict)

    # All other relationships not explicitly modeled
    other_relationships: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    # Raw content
    raw_block: str = ""

    def get_domain(self) -> Optional[str]:
        """Detect domain from term-id, source-domain, or namespace."""
        # Try source-domain first
        if self.source_domain:
            return self.source_domain.lower()

        # Try term-id prefix
        if self.term_id:
            for domain_key, config in DOMAIN_CONFIG.items():
                if self.term_id.startswith(config['prefix']):
                    return domain_key

        # Try namespace in owl_class
        if self.owl_class and ':' in self.owl_class:
            prefix = self.owl_class.split(':')[0]
            if prefix in DOMAIN_CONFIG:
                return prefix

        return None

    def get_full_iri(self) -> Optional[str]:
        """
        Extract full IRI from owl:class property.
        Returns the complete URI, not just namespace:localname.
        """
        if not self.owl_class:
            return None

        # If already a full URI, return it
        if self.owl_class.startswith('http://') or self.owl_class.startswith('https://'):
            return self.owl_class

        # Parse namespace:localname format
        if ':' in self.owl_class:
            prefix, localname = self.owl_class.split(':', 1)

            # Check domain namespaces
            if prefix in DOMAIN_CONFIG:
                return DOMAIN_CONFIG[prefix]['namespace'] + localname

            # Check standard namespaces
            if prefix in STANDARD_NAMESPACES:
                return STANDARD_NAMESPACES[prefix] + localname

        # Default: assume it's from the source domain
        domain = self.get_domain()
        if domain and domain in DOMAIN_CONFIG:
            return DOMAIN_CONFIG[domain]['namespace'] + self.owl_class

        return None

    def get_namespace_prefix_declarations(self) -> Dict[str, str]:
        """Generate namespace prefix declarations for this term."""
        declarations = {}

        # Add domain namespace
        domain = self.get_domain()
        if domain and domain in DOMAIN_CONFIG:
            declarations[domain] = DOMAIN_CONFIG[domain]['namespace']

        # Add standard namespaces
        declarations.update(STANDARD_NAMESPACES)

        return declarations

    def validate(self) -> List[str]:
        """
        Validate that all Tier 1 (required) properties are present.
        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Tier 1 required properties
        if not self.term_id:
            errors.append("Missing required property: term-id")

        if not self.preferred_term:
            errors.append("Missing required property: preferred-term")

        if not self.source_domain:
            errors.append("Missing required property: source-domain")

        if not self.status:
            errors.append("Missing required property: status")

        if self.public_access is None:
            errors.append("Missing required property: public-access")

        if not self.last_updated:
            errors.append("Missing required property: last-updated")

        if not self.definition:
            errors.append("Missing required property: definition")

        if not self.owl_class:
            errors.append("Missing required property: owl:class")

        if not self.owl_physicality:
            errors.append("Missing required property: owl:physicality")

        if not self.owl_role:
            errors.append("Missing required property: owl:role")

        if not self.is_subclass_of:
            errors.append("Missing required property: is-subclass-of (at least one parent class)")

        # Validate term-id format
        if self.term_id:
            domain = self.get_domain()
            if domain and domain in DOMAIN_CONFIG:
                expected_prefix = DOMAIN_CONFIG[domain]['prefix']
                if not self.term_id.startswith(expected_prefix):
                    errors.append(f"term-id '{self.term_id}' doesn't match domain '{domain}' (expected {expected_prefix})")

        # Validate namespace consistency
        if self.owl_class and ':' in self.owl_class:
            prefix = self.owl_class.split(':')[0]
            domain = self.get_domain()
            if domain and prefix != domain:
                errors.append(f"owl:class namespace '{prefix}' doesn't match source-domain '{domain}'")

        return errors


class OntologyBlockParser:
    """
    Parser for canonical ontology block format.
    """

    def __init__(self):
        """Initialize the parser."""
        pass

    def extract_ontology_block(self, content: str) -> str:
        """
        Extract the OntologyBlock section from markdown content.
        Stops at the first ## heading after the block.
        """
        match = re.search(
            r'-\s*###\s*OntologyBlock\s*\n(.*?)(?=\n-\s*##[^#]|\Z)',
            content,
            re.DOTALL
        )
        return match.group(1).strip() if match else ""

    def extract_property(self, block: str, property_name: str, is_list: bool = False) -> Any:
        """
        Extract a property value from the ontology block.

        Args:
            block: The ontology block content
            property_name: The property name (e.g., 'term-id', 'owl:class')
            is_list: Whether the property can have multiple values

        Returns:
            Property value (string, list, or None)
        """
        # Escape special regex characters in property name
        escaped_name = re.escape(property_name)

        # Pattern for property with wiki-links
        pattern_wiki = rf'^\s*-\s*{escaped_name}::\s*(.+)$'

        matches = list(re.finditer(pattern_wiki, block, re.MULTILINE))

        if not matches:
            return [] if is_list else None

        if is_list:
            # Extract all wiki-links from all matches
            all_values = []
            for match in matches:
                value_text = match.group(1).strip()
                # Extract [[wiki-style]] links
                wiki_links = re.findall(r'\[\[([^\]]+)\]\]', value_text)
                if wiki_links:
                    all_values.extend(link.strip() for link in wiki_links)
                elif value_text and not value_text.startswith('[['):
                    # Plain value without wiki-links
                    all_values.append(value_text)
            return all_values
        else:
            # Return first match
            value = matches[0].group(1).strip()
            # Remove wiki-link brackets if present
            value = re.sub(r'^\[\[(.+)\]\]$', r'\1', value)
            return value

    def extract_relationships_section(self, block: str) -> Dict[str, List[str]]:
        """
        Extract all relationships from the #### Relationships section.
        """
        relationships = defaultdict(list)

        # Find Relationships section
        rel_match = re.search(
            r'-\s*####\s*Relationships\s*\n(.*?)(?=-\s*####|\Z)',
            block,
            re.DOTALL | re.IGNORECASE
        )

        if not rel_match:
            return dict(relationships)

        rel_section = rel_match.group(1)

        # Parse each relationship line
        for line in rel_section.strip().split('\n'):
            match = re.match(r'-\s*([\w-]+)::\s*(.*)', line.strip())
            if match:
                prop_name = match.group(1).strip()
                value_text = match.group(2)

                # Extract wiki-links
                wiki_links = re.findall(r'\[\[([^\]]+)\]\]', value_text)
                if wiki_links:
                    relationships[prop_name].extend(link.strip() for link in wiki_links)

        return dict(relationships)

    def extract_owl_axioms(self, content: str) -> List[str]:
        """
        Extract OWL axioms from ```clojure or ```owl code blocks.
        """
        axioms = re.findall(
            r'```(?:clojure|owl)\s*\n(.*?)\n```',
            content,
            re.DOTALL
        )
        return [axiom.strip() for axiom in axioms]

    def extract_domain_extensions(self, block: str, domain: str) -> Dict[str, Any]:
        """
        Extract domain-specific extension properties.
        """
        extensions = {}

        if domain not in DOMAIN_CONFIG:
            return extensions

        # Get domain-specific properties
        for prop in DOMAIN_CONFIG[domain]['extension_properties']:
            value = self.extract_property(block, prop)
            if value:
                extensions[prop] = value

        return extensions

    def parse_file(self, file_path: Path) -> Optional[OntologyBlock]:
        """
        Parse a single markdown file and extract the ontology block.

        Args:
            file_path: Path to the markdown file

        Returns:
            OntologyBlock object if valid ontology file, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            return None

        # Extract ontology block
        block_content = self.extract_ontology_block(content)
        if not block_content:
            return None

        # Create block object
        block = OntologyBlock(file_path=file_path, raw_block=block_content)

        # === Extract Tier 1 Properties ===
        # Identification
        block.term_id = self.extract_property(block_content, 'term-id')
        block.preferred_term = self.extract_property(block_content, 'preferred-term')
        block.source_domain = self.extract_property(block_content, 'source-domain')
        block.status = self.extract_property(block_content, 'status')

        public_access_str = self.extract_property(block_content, 'public-access')
        if public_access_str:
            block.public_access = public_access_str.lower() == 'true'

        block.last_updated = self.extract_property(block_content, 'last-updated')

        # Definition
        block.definition = self.extract_property(block_content, 'definition')

        # Semantic Classification
        block.owl_class = self.extract_property(block_content, 'owl:class')
        block.owl_physicality = self.extract_property(block_content, 'owl:physicality')
        block.owl_role = self.extract_property(block_content, 'owl:role')

        # === Extract Tier 2 Properties ===
        block.alt_terms = self.extract_property(block_content, 'alt-terms', is_list=True)
        block.version = self.extract_property(block_content, 'version')

        quality_str = self.extract_property(block_content, 'quality-score')
        if quality_str:
            try:
                block.quality_score = float(quality_str)
            except ValueError:
                pass

        cross_domain_str = self.extract_property(block_content, 'cross-domain-links')
        if cross_domain_str:
            try:
                block.cross_domain_links = int(cross_domain_str)
            except ValueError:
                pass

        block.maturity = self.extract_property(block_content, 'maturity')
        block.source = self.extract_property(block_content, 'source', is_list=True)

        authority_str = self.extract_property(block_content, 'authority-score')
        if authority_str:
            try:
                block.authority_score = float(authority_str)
            except ValueError:
                pass

        block.scope_note = self.extract_property(block_content, 'scope-note')

        block.owl_inferred_class = self.extract_property(block_content, 'owl:inferred-class')
        block.belongs_to_domain = self.extract_property(block_content, 'belongsToDomain', is_list=True)
        block.implemented_in_layer = self.extract_property(block_content, 'implementedInLayer', is_list=True)

        # === Extract Relationships ===
        relationships = self.extract_relationships_section(block_content)

        # Map known relationships
        block.is_subclass_of = relationships.get('is-subclass-of', [])
        block.has_part = relationships.get('has-part', [])
        block.is_part_of = relationships.get('is-part-of', [])
        block.requires = relationships.get('requires', [])
        block.depends_on = relationships.get('depends-on', [])
        block.enables = relationships.get('enables', [])
        block.relates_to = relationships.get('relates-to', [])

        # Store other relationships
        known_rels = {'is-subclass-of', 'has-part', 'is-part-of', 'requires',
                      'depends-on', 'enables', 'relates-to'}
        for rel_name, targets in relationships.items():
            if rel_name not in known_rels:
                block.other_relationships[rel_name] = targets

        # === Extract Cross-Domain Bridges ===
        bridge_to_matches = re.findall(
            r'bridges-to::\s*\[\[([^\]]+)\]\]\s*via\s+(\w+)',
            block_content
        )
        block.bridges_to = [f"{target} via {rel}" for target, rel in bridge_to_matches]

        bridge_from_matches = re.findall(
            r'bridges-from::\s*\[\[([^\]]+)\]\]\s*via\s+(\w+)',
            block_content
        )
        block.bridges_from = [f"{target} via {rel}" for target, rel in bridge_from_matches]

        # === Extract OWL Axioms ===
        block.owl_axioms = self.extract_owl_axioms(content)

        # === Extract Domain Extensions ===
        domain = block.get_domain()
        if domain:
            block.domain_extensions = self.extract_domain_extensions(block_content, domain)

        return block

    def parse_directory(self, pages_dir: Path, pattern: str = "*.md") -> List[OntologyBlock]:
        """
        Parse all markdown files in a directory.

        Args:
            pages_dir: Directory containing markdown files
            pattern: Glob pattern for files to process

        Returns:
            List of successfully parsed OntologyBlock objects
        """
        pages_path = Path(pages_dir)
        if not pages_path.exists():
            raise FileNotFoundError(f"Directory not found: {pages_dir}")

        md_files = sorted(pages_path.glob(pattern))
        blocks = []

        for md_file in md_files:
            block = self.parse_file(md_file)
            if block and block.term_id:
                blocks.append(block)

        return blocks


def main():
    """Example usage of OntologyBlockParser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ontology_block_parser.py <file_or_directory>")
        print("\nExample: python ontology_block_parser.py mainKnowledgeGraph/pages/")
        sys.exit(1)

    path = Path(sys.argv[1])
    parser = OntologyBlockParser()

    if path.is_file():
        # Parse single file
        print(f"Parsing file: {path}")
        block = parser.parse_file(path)

        if block:
            print(f"\n✅ Successfully parsed: {block.preferred_term}")
            print(f"   Term ID: {block.term_id}")
            print(f"   Domain: {block.get_domain()}")
            print(f"   IRI: {block.get_full_iri()}")
            print(f"   Status: {block.status}")

            # Validation
            errors = block.validate()
            if errors:
                print(f"\n⚠️  Validation errors:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print(f"\n✅ Validation: PASSED")
        else:
            print("❌ No ontology block found in file")

    elif path.is_dir():
        # Parse directory
        print(f"Parsing directory: {path}")
        blocks = parser.parse_directory(path)

        print(f"\n✅ Successfully parsed {len(blocks)} ontology blocks")

        # Statistics by domain
        domain_counts = defaultdict(int)
        for block in blocks:
            domain = block.get_domain()
            if domain:
                domain_counts[domain] += 1

        print("\n📊 Statistics by domain:")
        for domain, count in sorted(domain_counts.items()):
            domain_name = DOMAIN_CONFIG[domain]['full_name'] if domain in DOMAIN_CONFIG else domain
            print(f"   {domain_name}: {count}")

        # Validation summary
        total_errors = 0
        blocks_with_errors = 0
        for block in blocks:
            errors = block.validate()
            if errors:
                blocks_with_errors += 1
                total_errors += len(errors)

        print(f"\n✅ Validation summary:")
        print(f"   Blocks with errors: {blocks_with_errors}/{len(blocks)}")
        print(f"   Total errors: {total_errors}")

    else:
        print(f"Error: {path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
