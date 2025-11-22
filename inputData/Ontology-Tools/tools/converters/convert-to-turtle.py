#!/usr/bin/env python3
"""
Convert Logseq Ontology to OWL2 DL Turtle Format
=================================================

Reads markdown files with canonical ontology blocks and generates
OWL2 DL compliant Turtle (.ttl) output with:
- Proper class definitions
- Full IRI declarations
- Namespace prefix declarations
- Property restrictions
- Multi-domain support (AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Tech)

Usage:
    python convert-to-turtle.py <input_dir_or_file> <output_file.ttl>

Example:
    python convert-to-turtle.py mainKnowledgeGraph/pages/ output/ontology.ttl
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ontology_block_parser import (
    OntologyBlock,
    DOMAIN_CONFIG,
    STANDARD_NAMESPACES
)
from lib.ontology_loader import OntologyLoader


class TTLConverter:
    """Converts parsed ontology blocks to OWL2 DL Turtle format."""

    def __init__(self):
        # Use the unified loader instead of parser directly
        self.loader = OntologyLoader(cache_size=200, strict_validation=False)
        self.blocks: List[OntologyBlock] = []
        self.all_properties: Set[str] = set()
        self.statistics = defaultdict(int)

    def load_blocks(self, input_path: Path):
        """Load ontology blocks from file or directory."""
        if input_path.is_file():
            block = self.loader.load_file(input_path)
            if block and block.term_id:
                self.blocks.append(block)
        elif input_path.is_dir():
            self.blocks = self.loader.load_directory(input_path, progress=True)
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        print(f"Loaded {len(self.blocks)} ontology blocks")

        # Print cache statistics
        cache_stats = self.loader.get_cache_stats()
        print(f"Cache performance: {cache_stats['hit_rate']:.1%} hit rate ({cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses)")

        # Collect statistics
        for block in self.blocks:
            domain = block.get_domain()
            if domain:
                self.statistics[f'{domain}_terms'] += 1
            self.statistics['total_terms'] += 1

            # Collect all relationship properties
            for rel_name in block.other_relationships.keys():
                self.all_properties.add(rel_name)

            if block.has_part:
                self.all_properties.add('has-part')
            if block.requires:
                self.all_properties.add('requires')
            if block.depends_on:
                self.all_properties.add('depends-on')
            if block.enables:
                self.all_properties.add('enables')
            if block.relates_to:
                self.all_properties.add('relates-to')

    def sanitize_literal(self, value: str) -> str:
        """Escape special characters in literals."""
        if not value:
            return ""
        value = value.replace('\\', '\\\\')
        value = value.replace('"', '\\"')
        value = value.replace('\n', '\\n')
        value = value.replace('\r', '\\r')
        return value

    def generate_header(self) -> str:
        """Generate Turtle header with prefixes and ontology metadata."""
        ttl = "# ==============================================================================\n"
        ttl += "# Multi-Domain Ontology - OWL2 DL Turtle Format\n"
        ttl += "# ==============================================================================\n"
        ttl += "#\n"
        ttl += f"# Generated: {datetime.now().isoformat()}\n"
        ttl += f"# Total Terms: {self.statistics['total_terms']}\n"
        ttl += "#\n"
        ttl += "# Domain Distribution:\n"

        for domain_key, config in DOMAIN_CONFIG.items():
            count = self.statistics.get(f'{domain_key}_terms', 0)
            ttl += f"#   {config['full_name']}: {count} terms\n"

        ttl += "#\n"
        ttl += "# ==============================================================================\n\n"

        # Add namespace prefixes
        all_namespaces = {}
        all_namespaces.update(STANDARD_NAMESPACES)

        # Add domain namespaces
        for domain_key, config in DOMAIN_CONFIG.items():
            all_namespaces[domain_key] = config['namespace']

        for prefix, uri in sorted(all_namespaces.items()):
            ttl += f"@prefix {prefix}: <{uri}> .\n"

        ttl += "\n"

        # Ontology declaration
        ttl += "# ==============================================================================\n"
        ttl += "# ONTOLOGY DECLARATION\n"
        ttl += "# ==============================================================================\n\n"

        ttl += "<http://narrativegoldmine.com/ontology> a owl:Ontology ;\n"
        ttl += '    rdfs:label "Multi-Domain Knowledge Ontology"@en ;\n'
        ttl += '    rdfs:comment "Comprehensive ontology covering AI, Blockchain, Robotics, Metaverse, Telecollaboration, and Disruptive Technologies."@en ;\n'
        ttl += '    owl:versionInfo "1.0.0" ;\n'
        ttl += f'    dcterms:created "{datetime.now().date().isoformat()}"^^xsd:date ;\n'
        ttl += '    dcterms:creator "Logseq Ontology Tools" .\n\n'

        return ttl

    def convert_block_to_ttl(self, block: OntologyBlock) -> str:
        """Convert a single ontology block to Turtle format."""
        ttl = f"# {block.preferred_term or block.term_id}\n"

        # Get full IRI
        iri = block.owl_class or f"{block.get_domain()}:{block.term_id}"

        ttl += f"{iri} a owl:Class"

        # SubClassOf relationships
        if block.is_subclass_of:
            for parent in block.is_subclass_of:
                # Try to resolve parent to a proper IRI
                parent_clean = parent.replace(' ', '').replace('-', '')
                ttl += f" ;\n    rdfs:subClassOf {block.get_domain()}:{parent_clean}"

        # Labels
        if block.preferred_term:
            ttl += f' ;\n    rdfs:label "{self.sanitize_literal(block.preferred_term)}"@en'

        # Definition as comment
        if block.definition:
            ttl += f' ;\n    rdfs:comment "{self.sanitize_literal(block.definition)}"@en'

        # SKOS preferred label
        if block.preferred_term:
            ttl += f' ;\n    skos:prefLabel "{self.sanitize_literal(block.preferred_term)}"@en'

        # Metadata annotations
        if block.term_id:
            ttl += f' ;\n    dcterms:identifier "{block.term_id}"'

        if block.source_domain:
            ttl += f' ;\n    dcterms:subject "{block.source_domain}"'

        if block.maturity:
            ttl += f' ;\n    dcterms:type "{block.maturity}"'

        if block.authority_score is not None:
            ttl += f' ;\n    dcterms:conformsTo "{block.authority_score}"^^xsd:decimal'

        # Physicality and role
        if block.owl_physicality:
            ttl += f' ;\n    owl:hasValue "{block.owl_physicality}"'

        if block.owl_role:
            ttl += f' ;\n    rdf:type owl:{block.owl_role}Class'

        ttl += " .\n\n"

        # Property restrictions
        restrictions_ttl = ""

        # has-part restrictions
        for part in block.has_part:
            part_clean = part.replace(' ', '').replace('-', '')
            restrictions_ttl += f"{iri} rdfs:subClassOf [\n"
            restrictions_ttl += f"    a owl:Restriction ;\n"
            restrictions_ttl += f"    owl:onProperty {block.get_domain()}:hasPart ;\n"
            restrictions_ttl += f"    owl:someValuesFrom {block.get_domain()}:{part_clean}\n"
            restrictions_ttl += f"] .\n\n"

        # requires restrictions
        for req in block.requires:
            req_clean = req.replace(' ', '').replace('-', '')
            restrictions_ttl += f"{iri} rdfs:subClassOf [\n"
            restrictions_ttl += f"    a owl:Restriction ;\n"
            restrictions_ttl += f"    owl:onProperty {block.get_domain()}:requires ;\n"
            restrictions_ttl += f"    owl:someValuesFrom {block.get_domain()}:{req_clean}\n"
            restrictions_ttl += f"] .\n\n"

        # enables restrictions
        for capability in block.enables:
            cap_clean = capability.replace(' ', '').replace('-', '')
            restrictions_ttl += f"{iri} rdfs:subClassOf [\n"
            restrictions_ttl += f"    a owl:Restriction ;\n"
            restrictions_ttl += f"    owl:onProperty {block.get_domain()}:enables ;\n"
            restrictions_ttl += f"    owl:someValuesFrom {block.get_domain()}:{cap_clean}\n"
            restrictions_ttl += f"] .\n\n"

        ttl += restrictions_ttl

        return ttl

    def generate_property_declarations(self) -> str:
        """Generate OWL property declarations."""
        ttl = "# ==============================================================================\n"
        ttl += "# OBJECT PROPERTY DECLARATIONS\n"
        ttl += "# ==============================================================================\n\n"

        # Standard relationship properties
        standard_props = {
            'has-part': {
                'label': 'has part',
                'inverse': 'is-part-of',
                'transitive': True,
                'comment': 'Indicates compositional part-whole relationships'
            },
            'is-part-of': {
                'label': 'is part of',
                'inverse': 'has-part',
                'transitive': True,
                'comment': 'Inverse of has-part'
            },
            'requires': {
                'label': 'requires',
                'inverse': 'is-required-by',
                'comment': 'Indicates technical dependencies and prerequisites'
            },
            'depends-on': {
                'label': 'depends on',
                'transitive': True,
                'comment': 'Indicates logical or functional dependencies'
            },
            'enables': {
                'label': 'enables',
                'inverse': 'is-enabled-by',
                'comment': 'Indicates capabilities and functionalities provided'
            },
            'relates-to': {
                'label': 'relates to',
                'symmetric': True,
                'comment': 'General semantic association'
            }
        }

        for prop_name, prop_info in standard_props.items():
            # Determine property type
            prop_types = ['owl:ObjectProperty']
            if prop_info.get('transitive'):
                prop_types.append('owl:TransitiveProperty')
            if prop_info.get('symmetric'):
                prop_types.append('owl:SymmetricProperty')

            ttl += f":{prop_name} a {', '.join(prop_types)} ;\n"
            ttl += f'    rdfs:label "{prop_info["label"]}"@en ;\n'
            ttl += f'    rdfs:comment "{prop_info["comment"]}"@en'

            if prop_info.get('inverse'):
                ttl += f' ;\n    owl:inverseOf :{prop_info["inverse"]}'

            ttl += " .\n\n"

        # Additional properties found in blocks
        for prop in sorted(self.all_properties):
            if prop not in standard_props:
                clean_label = prop.replace('-', ' ')
                ttl += f":{prop} a owl:ObjectProperty ;\n"
                ttl += f'    rdfs:label "{clean_label}"@en .\n\n'

        return ttl

    def convert_to_ttl(self, output_file: Path):
        """Generate complete Turtle file."""
        print(f"Converting {len(self.blocks)} blocks to Turtle...")

        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(self.generate_header())

            # Write class definitions
            f.write("# ==============================================================================\n")
            f.write("# CLASS DEFINITIONS\n")
            f.write("# ==============================================================================\n\n")

            for block in sorted(self.blocks, key=lambda b: b.term_id or ''):
                f.write(self.convert_block_to_ttl(block))

            # Write property declarations
            f.write(self.generate_property_declarations())

        file_size = output_file.stat().st_size
        print(f"\n✅ Turtle file generated: {output_file}")
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        print(f"   Classes: {len(self.blocks)}")
        print(f"   Properties: {len(self.all_properties)}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Convert Logseq ontology markdown to OWL2 DL Turtle format'
    )
    parser.add_argument('input', type=str,
                        help='Input markdown file or directory')
    parser.add_argument('output', type=str,
                        help='Output TTL file path')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_file = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Logseq to OWL2 DL Turtle Converter")
    print("=" * 80)
    print()

    converter = TTLConverter()

    # Load blocks
    print(f"Loading ontology blocks from: {input_path}")
    converter.load_blocks(input_path)

    if not converter.blocks:
        print("Error: No valid ontology blocks found")
        sys.exit(1)

    # Convert to TTL
    converter.convert_to_ttl(output_file)

    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
