#!/usr/bin/env python3
"""
Convert Logseq Hybrid Ontology to OWL2 TTL Format

Parses mainKnowledgeGraph/pages directory and generates OWL2-compliant TTL
with proper class definitions, properties, and metadata.

Updated to use shared ontology_block_parser library.
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple

# Add shared library to path for ontology_block_parser import
sys.path.insert(0, str(Path(__file__).parent.parent / 'Ontology-Tools' / 'tools' / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG


class OWL2Converter:
    """Converts parsed Logseq ontology to OWL2 TTL format."""

    def __init__(self, parser: OntologyBlockParser, blocks: List[OntologyBlock]):
        self.parser = parser
        self.blocks = blocks
        # Build a lookup map by term_id
        self.term_map = {block.term_id: block for block in blocks if block.term_id}
        self.namespaces = {
            '': 'http://narrativegoldmine.com/core#',  # Default namespace
            'mv': 'http://narrativegoldmine.com/core#',
            'bc': 'http://narrativegoldmine.com/blockchain#',
            'blockchain': 'http://narrativegoldmine.com/blockchain#',
            'ai': 'http://narrativegoldmine.com/ai#',
            'aigo': 'http://narrativegoldmine.com/ai#',
            'ml': 'http://narrativegoldmine.com/ai#',
            'btcai': 'http://narrativegoldmine.com/ai#',
            'rb': 'http://narrativegoldmine.com/robotics#',
            'dt': 'http://narrativegoldmine.com/disruptive-tech#',
            'sc': 'http://narrativegoldmine.com/supply-chain#',
            'tele': 'http://narrativegoldmine.com/telecommunications#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'dcterms': 'http://purl.org/dc/terms/',
            'skos': 'http://www.w3.org/2004/02/skos/core#',
        }
        self.object_properties: Set[str] = set()
        self.data_properties: Set[str] = set()
        self.relationship_triples: List[Tuple[str, str, str]] = []  # (subject, property, object)
        self.owl_restrictions: Dict[str, List[Tuple[str, str, str]]] = {}  # uri -> [(property, restriction_type, target)]

        # Property characteristics
        self.transitive_properties = {
            'depends-on', 'is-subclass-of', 'is-part-of', 'derives-from'
        }
        self.inverse_property_pairs = [
            ('has-part', 'is-part-of'),
            ('enables', 'is-enabled-by'),
            ('implements', 'is-implemented-by'),
            ('requires', 'is-required-by'),
            ('produces', 'is-produced-by'),
            ('contains', 'is-contained-in'),
        ]

    def extract_owl_restrictions(self, content: str) -> List[Tuple[str, str, str]]:
        """
        Extract OWL restrictions from #### OWL Restrictions sections.

        Parses lines like:
        - hasSensingCapability some SensorSystem
        - is-part-of some Autonomousagent
        - hasControlSystem some RobotController

        Returns list of (property, restriction_type, target) tuples.
        """
        restrictions = []

        # Find OWL Restrictions sections
        restriction_sections = re.findall(
            r'-\s*####\s*OWL Restrictions\s*\n(.*?)(?=-\s*####|\n-\s*##|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )

        for section in restriction_sections:
            # Parse each restriction line
            for line in section.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Remove leading dash if present
                if line.startswith('-'):
                    line = line[1:].strip()

                # Skip lines with :: (those are metadata, not restrictions)
                if '::' in line:
                    continue

                # Match pattern: property restriction_type Target
                # e.g., "hasSensingCapability some SensorSystem"
                match = re.match(r'(\w[\w-]*)\s+(some|only|min|max|exactly)\s+(\w+)', line)
                if match:
                    prop = match.group(1)
                    restr_type = match.group(2)
                    target = match.group(3)
                    restrictions.append((prop, restr_type, target))

        return restrictions

    def generate_header(self) -> str:
        """Generate OWL2 TTL header with prefixes and ontology declaration."""
        # Count terms by domain
        domain_counts = {}
        for domain_key in DOMAIN_CONFIG.keys():
            count = sum(1 for b in self.blocks if b.get_domain() == domain_key)
            domain_counts[domain_key] = count

        header = f"""# ==============================================================================
# Logseq Hybrid Knowledge Graph - OWL2 Ontology
# ==============================================================================
#
# Generated from mainKnowledgeGraph/pages directory
# Total Terms: {len(self.blocks)}
# Generation Date: {datetime.now().isoformat()}
#
# This ontology integrates knowledge from multiple domains:
# - Artificial Intelligence (AI): {domain_counts.get('ai', 0)} terms
# - Blockchain (BC): {domain_counts.get('bc', 0)} terms
# - Robotics (RB): {domain_counts.get('rb', 0)} terms
# - Metaverse (MV): {domain_counts.get('mv', 0)} terms
# - Telecollaboration (TC): {domain_counts.get('tc', 0)} terms
# - Disruptive Tech (DT): {domain_counts.get('dt', 0)} terms
#
# ==============================================================================

"""
        # Add namespace prefixes
        for prefix, uri in sorted(self.namespaces.items()):
            header += f"@prefix {prefix}: <{uri}> .\n"

        header += f"""
# ==============================================================================
# ONTOLOGY DECLARATION
# ==============================================================================

<{self.namespaces['mv']}> a owl:Ontology ;
    rdfs:label "Logseq Hybrid Knowledge Graph Ontology"@en ;
    rdfs:comment "Comprehensive ontology extracted from Logseq knowledge graph covering metaverse, blockchain, AI, robotics, and disruptive technologies."@en ;
    owl:versionInfo "1.0.0" ;
    dc:created "{datetime.now().date().isoformat()}"^^xsd:date ;
    dc:creator "Logseq Knowledge Graph" ;
    dcterms:license <https://creativecommons.org/licenses/by/4.0/> .

"""
        return header

    def extract_owl_class_from_axioms(self, axioms: List[str]) -> Optional[str]:
        """Extract owl:Class declaration from OWL axioms."""
        for axiom in axioms:
            # Look for Declaration(Class(:ClassName))
            match = re.search(r'Declaration\(Class\(:(\w+)\)\)', axiom)
            if match:
                return match.group(1)
        return None

    def extract_subclass_relations(self, axioms: List[str]) -> List[Tuple[str, str]]:
        """Extract SubClassOf relations from OWL axioms.

        Returns list of (prefix, classname) tuples.
        """
        subclasses = []
        for axiom in axioms:
            # Look for SubClassOf(:Class1 prefix:Class2) or SubClassOf(:Class1 :Class2)
            matches = re.finditer(r'SubClassOf\(:\w+\s+(?:(\w+):)?(\w+)\)', axiom)
            for match in matches:
                prefix = match.group(1) if match.group(1) else ''  # Default prefix
                classname = match.group(2)
                subclasses.append((prefix, classname))
        return subclasses

    def extract_properties_from_axioms(self, axioms: List[str]) -> Dict[str, List[str]]:
        """Extract object and data properties from OWL axioms."""
        properties = {'object': [], 'data': []}

        for axiom in axioms:
            # ObjectSomeValuesFrom(:propertyName :TargetClass)
            obj_matches = re.finditer(r'ObjectSomeValuesFrom\(:(\w+)\s+', axiom)
            for match in obj_matches:
                prop_name = match.group(1)
                properties['object'].append(prop_name)
                self.object_properties.add(prop_name)

            # DataSomeValuesFrom(:propertyName
            data_matches = re.finditer(r'DataSomeValuesFrom\(:(\w+)\s+', axiom)
            for match in data_matches:
                prop_name = match.group(1)
                properties['data'].append(prop_name)
                self.data_properties.add(prop_name)

        return properties

    def normalize_term_uri(self, block: OntologyBlock) -> str:
        """Generate normalized URI for term based on domain."""
        # If owl_class is already set, use it directly
        if block.owl_class:
            # Already in namespace:localname format
            return block.owl_class

        # Determine prefix from domain
        domain = block.get_domain()
        prefix = domain if domain else 'mv'  # default to metaverse

        # Use preferred term as class name
        class_name = block.preferred_term or block.term_id or 'UnknownTerm'
        # Clean class name - replace & with And, remove spaces, special chars
        class_name = class_name.replace('&', 'And')
        class_name = re.sub(r'[^\w]', '', class_name.replace(' ', ''))

        return f"{prefix}:{class_name}"

    def get_namespace_for_concept(self, concept_name: str) -> str:
        """Determine the appropriate namespace for a concept based on its name or domain."""
        # Known blockchain concepts
        blockchain_concepts = {
            'Blockchain', 'Bitcoin', 'Ethereum', 'SmartContract', 'Cryptocurrency',
            'DeFi', 'NFT', 'Token', 'Wallet', 'Stablecoin', 'Mining', 'Block',
            'Transaction', 'Consensus', 'ProofofWork', 'ProofofStake', 'Hash',
            'Merkle', 'Mempool', 'UTXOModel', 'Halving', 'DifficultyAdjustment'
        }

        # Known AI concepts
        ai_concepts = {
            'ArtificialIntelligence', 'MachineLearning', 'DeepLearning', 'NeuralNetwork',
            'Transformer', 'GenerativeAI', 'NLP', 'ComputerVision', 'Algorithm',
            'Model', 'Training', 'Inference', 'Embedding', 'Attention', 'Layer',
            'Optimizer', 'LossFunction', 'Gradient', 'Backpropagation'
        }

        # Known robotics concepts
        robotics_concepts = {
            'Robot', 'Robotics', 'Manipulator', 'Actuator', 'Sensor', 'Controller',
            'Kinematics', 'Dynamics', 'PathPlanning', 'SLAM', 'HumanRobotInteraction',
            'SwarmRobotics', 'RobotLearning', 'MotionPlanning'
        }

        # Normalize concept name
        concept_clean = concept_name.replace(' ', '').replace('-', '')

        if concept_clean in blockchain_concepts:
            return 'bc'
        elif concept_clean in ai_concepts:
            return 'ai'
        elif concept_clean in robotics_concepts:
            return 'rb'
        else:
            # Try to look up the term in term_map (by term_id or preferred_term)
            block = self.term_map.get(concept_name)
            if block:
                domain = block.get_domain()
                if domain:
                    return domain

            return 'mv'  # Default to mv namespace

    def convert_term_to_owl(self, block: OntologyBlock) -> str:
        """Convert single OntologyBlock to OWL2 TTL format."""
        uri = self.normalize_term_uri(block)

        ttl = f"# {block.preferred_term or block.term_id}\n"
        ttl += f"{uri} a owl:Class"

        # Extract subclass relations from axioms
        subclasses = self.extract_subclass_relations(block.owl_axioms)

        # Also check is-subclass-of relationships
        if block.is_subclass_of:
            for parent in block.is_subclass_of:
                # Convert wiki-link to URI reference
                parent_clean = re.sub(r'[^\w]', '', parent.replace(' ', ''))
                # Determine namespace based on parent concept domain
                parent_prefix = self.get_namespace_for_concept(parent_clean)
                subclasses.append((parent_prefix, parent_clean))

        if subclasses:
            for prefix, parent in subclasses:
                if prefix:
                    ttl += f" ;\n    rdfs:subClassOf {prefix}:{parent}"
                else:
                    ttl += f" ;\n    rdfs:subClassOf mv:{parent}"

        # Add labels and comments
        if block.preferred_term:
            ttl += f' ;\n    rdfs:label "{block.preferred_term}"@en'

        if block.definition:
            # Clean definition - escape quotes
            clean_def = block.definition.replace('"', '\\"')
            ttl += f' ;\n    rdfs:comment "{clean_def}"@en'

        # Add SKOS preferred label
        if block.preferred_term:
            ttl += f' ;\n    skos:prefLabel "{block.preferred_term}"@en'

        # Add metadata annotations
        if block.term_id:
            ttl += f' ;\n    dc:identifier "{block.term_id}"'

        if block.source_domain:
            ttl += f' ;\n    dcterms:subject "{block.source_domain}"'

        if block.maturity:
            ttl += f' ;\n    dcterms:type "{block.maturity}"'

        if block.source:
            ttl += f' ;\n    dc:source "{", ".join(block.source)}"'

        # Add Tier 2 relationship properties
        relationships = {
            'has-part': block.has_part,
            'is-part-of': block.is_part_of,
            'requires': block.requires,
            'depends-on': block.depends_on,
            'enables': block.enables,
            'relates-to': block.relates_to,
        }

        # Add other relationships from other_relationships dict
        for rel_name, targets in block.other_relationships.items():
            relationships[rel_name] = targets

        for prop, targets in relationships.items():
            if not targets:
                continue

            # Add property to set
            self.object_properties.add(prop)

            # Add property assertions
            for target in targets:
                # Try to resolve target to actual class URI
                target_block = self.term_map.get(target)
                if target_block:
                    # Use the actual class URI
                    target_uri = self.normalize_term_uri(target_block)
                    ttl += f' ;\n    mv:{prop} {target_uri}'
                # If target not found, skip it (don't create broken links)

        ttl += " .\n\n"

        return ttl

    def generate_property_declarations(self) -> str:
        """Generate OWL property declarations with characteristics."""
        ttl = "# ==============================================================================\n"
        ttl += "# OBJECT PROPERTIES WITH CHARACTERISTICS\n"
        ttl += "# ==============================================================================\n\n"

        # Create reverse lookup for inverse properties
        inverse_lookup = {}
        for prop1, prop2 in self.inverse_property_pairs:
            inverse_lookup[prop1] = prop2
            inverse_lookup[prop2] = prop1

        for prop in sorted(self.object_properties):
            # Determine property characteristics
            is_transitive = prop in self.transitive_properties
            inverse_of = inverse_lookup.get(prop)

            # Build property declaration
            if is_transitive:
                ttl += f"mv:{prop} a owl:ObjectProperty, owl:TransitiveProperty ;\n"
            else:
                ttl += f"mv:{prop} a owl:ObjectProperty ;\n"

            ttl += f'    rdfs:label "{prop}"@en'

            # Add inverse if exists
            if inverse_of:
                ttl += f' ;\n    owl:inverseOf mv:{inverse_of}'

            # Add domain/range for known properties
            if prop in ['has-part', 'contains']:
                ttl += ' ;\n    rdfs:domain owl:Thing ;\n    rdfs:range owl:Thing'
            elif prop in ['implements', 'requires', 'enables']:
                ttl += ' ;\n    rdfs:domain owl:Thing ;\n    rdfs:range owl:Thing'

            ttl += " .\n\n"

        ttl += "# ==============================================================================\n"
        ttl += "# DATA PROPERTIES\n"
        ttl += "# ==============================================================================\n\n"

        for prop in sorted(self.data_properties):
            ttl += f"mv:{prop} a owl:DatatypeProperty ;\n"
            ttl += f'    rdfs:label "{prop}"@en .\n\n'

        return ttl

    def generate_restriction_triples(self, uri: str, restrictions: List[Tuple[str, str, str]]) -> str:
        """Generate OWL restriction triples for a class."""
        if not restrictions:
            return ""

        ttl = ""
        for prop, restr_type, target in restrictions:
            # Add property to object properties
            self.object_properties.add(prop)

            # Determine target URI prefix based on target name
            target_uri = self._resolve_target_uri(target)

            # Generate restriction based on type
            if restr_type == 'some':
                ttl += f"\n{uri} rdfs:subClassOf [\n"
                ttl += f"    a owl:Restriction ;\n"
                ttl += f"    owl:onProperty mv:{prop} ;\n"
                ttl += f"    owl:someValuesFrom {target_uri}\n"
                ttl += f"] .\n"
            elif restr_type == 'only':
                ttl += f"\n{uri} rdfs:subClassOf [\n"
                ttl += f"    a owl:Restriction ;\n"
                ttl += f"    owl:onProperty mv:{prop} ;\n"
                ttl += f"    owl:allValuesFrom {target_uri}\n"
                ttl += f"] .\n"
            elif restr_type in ['min', 'max', 'exactly']:
                # For cardinality restrictions, target is a number
                try:
                    card_value = int(target)
                    if restr_type == 'min':
                        ttl += f"\n{uri} rdfs:subClassOf [\n"
                        ttl += f"    a owl:Restriction ;\n"
                        ttl += f"    owl:onProperty mv:{prop} ;\n"
                        ttl += f'    owl:minCardinality "{card_value}"^^xsd:nonNegativeInteger\n'
                        ttl += f"] .\n"
                    elif restr_type == 'max':
                        ttl += f"\n{uri} rdfs:subClassOf [\n"
                        ttl += f"    a owl:Restriction ;\n"
                        ttl += f"    owl:onProperty mv:{prop} ;\n"
                        ttl += f'    owl:maxCardinality "{card_value}"^^xsd:nonNegativeInteger\n'
                        ttl += f"] .\n"
                    elif restr_type == 'exactly':
                        ttl += f"\n{uri} rdfs:subClassOf [\n"
                        ttl += f"    a owl:Restriction ;\n"
                        ttl += f"    owl:onProperty mv:{prop} ;\n"
                        ttl += f'    owl:cardinality "{card_value}"^^xsd:nonNegativeInteger\n'
                        ttl += f"] .\n"
                except ValueError:
                    # If target is not a number, treat as class restriction
                    target_uri = self._resolve_target_uri(target)
                    if restr_type == 'min':
                        ttl += f"\n{uri} rdfs:subClassOf [\n"
                        ttl += f"    a owl:Restriction ;\n"
                        ttl += f"    owl:onProperty mv:{prop} ;\n"
                        ttl += f"    owl:minQualifiedCardinality 1 ;\n"
                        ttl += f"    owl:onClass {target_uri}\n"
                        ttl += f"] .\n"

        return ttl

    def _resolve_target_uri(self, target: str) -> str:
        """Resolve a target class name to its full URI."""
        # Try to find the block in term_map
        block = self.term_map.get(target)
        if block:
            return self.normalize_term_uri(block)

        # Default to mv: prefix for unknown targets
        # Clean the target name
        clean_target = re.sub(r'[^\w]', '', target)
        return f"mv:{clean_target}"

    def generate_disjointness_constraints(self) -> str:
        """Generate disjointness constraints between major domain classes."""
        ttl = "\n# ==============================================================================\n"
        ttl += "# DISJOINTNESS CONSTRAINTS\n"
        ttl += "# ==============================================================================\n\n"

        # Domain-level disjointness
        disjoint_pairs = [
            # Physical vs Virtual entities
            ('rb:PhysicalRobot', 'mv:VirtualEntity'),
            # AI vs Physical
            ('ai:ArtificialIntelligence', 'rb:PhysicalRobot'),
            # Different currency types
            ('bc:Cryptocurrency', 'mv:FiatCurrency'),
            # Robot types
            ('rb:IndustrialRobot', 'rb:ServiceRobot'),
            # Blockchain types
            ('bc:PublicBlockchain', 'bc:PrivateBlockchain'),
        ]

        for class1, class2 in disjoint_pairs:
            ttl += f"{class1} owl:disjointWith {class2} .\n"

        # Major domain disjointness
        ttl += "\n# Cross-domain disjointness\n"
        ttl += "[] a owl:AllDisjointClasses ;\n"
        ttl += "    owl:members ( bc:BlockchainPrimitive rb:RoboticComponent ai:AIAlgorithm ) .\n"

        return ttl

    def generate_owl2_ttl(self, output_path: Path):
        """Generate complete OWL2 TTL file."""
        print(f"Generating OWL2 TTL from {len(self.blocks)} blocks...")

        # Extract properties from axioms first
        for block in self.blocks:
            self.extract_properties_from_axioms(block.owl_axioms)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(self.generate_header())

            # Write class definitions (this will collect object properties)
            f.write("# ==============================================================================\n")
            f.write("# CLASS DEFINITIONS\n")
            f.write("# ==============================================================================\n\n")

            processed = 0
            relationship_count = 0
            restriction_count = 0
            restriction_ttl_buffer = []  # Collect restriction triples to write after classes

            for block in sorted(self.blocks, key=lambda b: b.term_id or ''):
                f.write(self.convert_term_to_owl(block))
                processed += 1

                # Count relationships
                all_rels = [
                    block.has_part, block.is_part_of, block.requires,
                    block.depends_on, block.enables, block.relates_to
                ]
                relationship_count += sum(len(rels) for rels in all_rels)

                # Add other relationships
                for targets in block.other_relationships.values():
                    relationship_count += len(targets)

                # Extract and generate OWL restrictions from file content
                try:
                    with open(block.file_path, 'r', encoding='utf-8') as content_file:
                        content = content_file.read()
                        restrictions = self.extract_owl_restrictions(content)
                        if restrictions:
                            uri = self.normalize_term_uri(block)
                            restriction_ttl = self.generate_restriction_triples(uri, restrictions)
                            if restriction_ttl:
                                restriction_ttl_buffer.append(restriction_ttl)
                                restriction_count += len(restrictions)
                except Exception as e:
                    pass  # Skip restriction extraction if file read fails

            print(f"✓ Generated {processed} OWL class definitions")
            print(f"✓ Included {relationship_count} relationship assertions")

            # Write OWL restrictions section
            if restriction_ttl_buffer:
                f.write("\n# ==============================================================================\n")
                f.write("# OWL RESTRICTIONS (Existential and Universal Quantification)\n")
                f.write("# ==============================================================================\n")
                for restriction_block in restriction_ttl_buffer:
                    f.write(restriction_block)
                print(f"✓ Generated {restriction_count} OWL restriction axioms")

            # Write property declarations with characteristics
            if self.object_properties or self.data_properties:
                f.write(self.generate_property_declarations())
                print(f"✓ Declared {len(self.object_properties)} object properties with characteristics")

            # Write disjointness constraints
            f.write(self.generate_disjointness_constraints())
            print(f"✓ Added disjointness constraints")


def main():
    """Main conversion workflow."""
    print("=" * 80)
    print("Logseq to OWL2 TTL Converter")
    print("=" * 80)
    print()

    # Setup paths
    project_root = Path(__file__).parent.parent
    pages_dir = project_root / 'mainKnowledgeGraph' / 'pages'
    output_dir = project_root / 'ontology-output'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'logseq-knowledge-graph.ttl'

    # Parse ontology using shared parser
    print(f"Parsing ontology from: {pages_dir}")
    parser = OntologyBlockParser()
    blocks = parser.parse_directory(pages_dir)

    print(f"✓ Parsed {len(blocks)} ontology blocks\n")

    # Show statistics by domain
    print("Statistics by domain:")
    domain_counts = {}
    for domain_key, config in DOMAIN_CONFIG.items():
        count = sum(1 for b in blocks if b.get_domain() == domain_key)
        domain_counts[domain_key] = count
        print(f"  {config['full_name']}: {count}")
    print()

    # Convert to OWL2
    converter = OWL2Converter(parser, blocks)
    converter.generate_owl2_ttl(output_file)

    print(f"\n✓ OWL2 TTL written to: {output_file}")
    print(f"  File size: {output_file.stat().st_size:,} bytes")

    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)

    return output_file


if __name__ == '__main__':
    output_file = main()
    sys.exit(0)
