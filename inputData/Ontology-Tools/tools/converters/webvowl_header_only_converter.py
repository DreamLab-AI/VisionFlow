#!/usr/bin/env python3
"""
WebVOWL Ontology Generator from OntologyBlock Headers
Generates WebVOWL-compatible TTL from ontology block metadata and relationships.
Supports all 6 domains with distinct namespaces and colors.

Usage:
    python webvowl_header_only_converter.py --pages-dir mainKnowledgeGraph/pages --output outputs/ontology.ttl
"""

import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, URIRef, Literal, BNode

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG, STANDARD_NAMESPACES

# Define domain namespaces
DOMAIN_NAMESPACES = {
    'dt': Namespace(DOMAIN_CONFIG['dt']['namespace']),
    'ai': Namespace(DOMAIN_CONFIG['ai']['namespace']),
    'bc': Namespace(DOMAIN_CONFIG['bc']['namespace']),
    'mv': Namespace(DOMAIN_CONFIG['mv']['namespace']),
    'rb': Namespace(DOMAIN_CONFIG['rb']['namespace']),
    'tc': Namespace(DOMAIN_CONFIG['tc']['namespace'])
}

# Standard namespaces
DCTERMS = Namespace(STANDARD_NAMESPACES['dcterms'])
PROV = Namespace("http://www.w3.org/ns/prov#")


class WebVOWLOntologyBuilder:
    """Build WebVOWL-compatible RDF graph from ontology blocks."""

    def __init__(self):
        self.graph = Graph()
        self.setup_namespaces()
        self.term_map = {}  # Map preferred terms to URIs
        self.object_properties = set()
        self.data_properties = set()
        self.stats = defaultdict(int)
        self.bridges = []  # Track cross-domain relationships

    def setup_namespaces(self):
        """Setup RDF namespaces."""
        # Bind domain namespaces
        for prefix, ns in DOMAIN_NAMESPACES.items():
            self.graph.bind(prefix, ns)

        # Bind standard namespaces
        self.graph.bind('owl', OWL)
        self.graph.bind('rdf', RDF)
        self.graph.bind('rdfs', RDFS)
        self.graph.bind('xsd', XSD)
        self.graph.bind('dcterms', DCTERMS)
        self.graph.bind('prov', PROV)

    def get_namespace(self, domain: str) -> Namespace:
        """Get namespace for a domain."""
        return DOMAIN_NAMESPACES.get(domain, DOMAIN_NAMESPACES['dt'])

    def to_camel_case(self, text: str) -> str:
        """Convert text to PascalCase for URI fragments."""
        # Remove code patterns like BC-0123, AI-0456
        text = re.sub(r'^[A-Z]{2,}-?\d+[-_\s]*', '', text)
        # Remove invalid characters
        text = re.sub(r'[()%\[\]{}]', '', text)
        # Convert to PascalCase
        words = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).split()
        result = ''.join(word.capitalize() for word in words if word)
        # Remove leading digits
        result = re.sub(r'^[\d\s]+', '', result)
        result = re.sub(r'[^a-zA-Z0-9]', '', result)
        return result if result else 'Term'

    def to_property_name(self, text: str) -> str:
        """Convert text to camelCase for properties."""
        words = re.sub(r'[^a-zA-Z0-9\s-]', '', text).split()
        if not words:
            return 'relatedTo'
        return words[0].lower() + ''.join(w.capitalize() for w in words[1:])

    def resolve_term_uri(self, term_name: str, source_domain: str = None) -> URIRef:
        """Resolve term name to URI, using domain context if available."""
        term_name = term_name.lstrip(':')

        # Check if already mapped
        if term_name in self.term_map:
            return self.term_map[term_name]

        # Use source domain if provided
        if source_domain:
            ns = self.get_namespace(source_domain)
            return ns[self.to_camel_case(term_name)]

        # Try to infer from keywords
        keywords = {
            'bc': ['blockchain', 'crypto', 'bitcoin', 'ethereum', 'mining', 'consensus',
                   'smart', 'contract', 'dao', 'nft', 'token', 'ledger', 'defi'],
            'rb': ['robot', 'sensor', 'actuator', 'manipulation', 'gripper', 'arm',
                   'humanoid', 'mobile', 'industrial', 'servo'],
            'mv': ['virtual', 'avatar', 'metaverse', 'spatial', 'digital', 'npc',
                   'immersive', 'vr', 'ar', 'xr'],
            'ai': ['learning', 'neural', 'training', 'model', 'ai', 'algorithm',
                   'network', 'deep', 'machine', 'intelligence', 'fairness', 'bias',
                   'ethics', 'privacy', 'governance'],
            'tc': ['collaboration', 'communication', 'remote', 'distributed', 'team',
                   'teleconference', 'telepresence', 'virtual meeting']
        }

        term_lower = term_name.lower()
        for domain_key, kws in keywords.items():
            if any(kw in term_lower for kw in kws):
                ns = self.get_namespace(domain_key)
                return ns[self.to_camel_case(term_name)]

        # Default to disruptive tech namespace
        return DOMAIN_NAMESPACES['dt'][self.to_camel_case(term_name)]

    def process_block(self, block: OntologyBlock) -> bool:
        """Process a single ontology block and add to graph."""
        # Get full IRI
        full_iri = block.get_full_iri()
        if not full_iri:
            print(f"Warning: Skipping {block.term_id} - no IRI", file=sys.stderr)
            return False

        term_uri = URIRef(full_iri)

        # Store mapping
        if block.preferred_term:
            camel = self.to_camel_case(block.preferred_term)
            self.term_map[camel] = term_uri
            self.term_map[block.term_id] = term_uri
            self.term_map[block.preferred_term] = term_uri

        # Add class declaration
        self.graph.add((term_uri, RDF.type, OWL.Class))
        self.stats['classes'] += 1

        # Add term-id as dcterms:identifier
        if block.term_id:
            self.graph.add((term_uri, DCTERMS.identifier, Literal(block.term_id)))

        # Add label
        if block.preferred_term:
            self.graph.add((term_uri, RDFS.label, Literal(block.preferred_term, lang='en')))
            self.stats['labels'] += 1

        # Add definition as comment
        if block.definition:
            self.graph.add((term_uri, RDFS.comment, Literal(block.definition, lang='en')))
            self.stats['comments'] += 1

        # Add metadata
        domain = block.get_domain()
        if domain:
            domain_ns = self.get_namespace(domain)
            self.graph.add((term_uri, domain_ns.domain, Literal(domain)))

        if block.maturity:
            self.graph.add((term_uri, DOMAIN_NAMESPACES['dt'].maturity, Literal(block.maturity)))

        if block.authority_score is not None:
            self.graph.add((term_uri, DOMAIN_NAMESPACES['dt'].authorityScore,
                          Literal(block.authority_score, datatype=XSD.decimal)))

        # Process is-subclass-of relationships
        for parent in block.is_subclass_of:
            parent_uri = self.resolve_term_uri(parent, domain)
            # Ensure parent is declared as class
            self.graph.add((parent_uri, RDF.type, OWL.Class))
            self.graph.add((parent_uri, RDFS.label, Literal(parent, lang='en')))
            # Add subClassOf relationship
            self.graph.add((term_uri, RDFS.subClassOf, parent_uri))
            self.stats['subclass'] += 1

        # Process belongs_to_domain
        for belongs_domain in block.belongs_to_domain:
            domain_uri = self.resolve_term_uri(belongs_domain, domain)
            self.graph.add((domain_uri, RDF.type, OWL.Class))
            self.graph.add((domain_uri, RDFS.label, Literal(belongs_domain, lang='en')))
            self.graph.add((term_uri, RDFS.subClassOf, domain_uri))
            self.stats['subclass'] += 1

        # Process other relationships
        self._process_relationships(block, term_uri, domain)

        # Process OWL axioms
        for axiom in block.owl_axioms:
            self._parse_owl_axiom(axiom, term_uri, block.term_id)

        return True

    def _process_relationships(self, block: OntologyBlock, term_uri: URIRef, domain: str):
        """Process all relationships from an ontology block."""
        relationship_map = {
            'has_part': 'hasPart',
            'is_part_of': 'isPartOf',
            'requires': 'requires',
            'depends_on': 'dependsOn',
            'enables': 'enables',
            'relates_to': 'relatesTo'
        }

        for attr, prop_name in relationship_map.items():
            targets = getattr(block, attr, [])
            if not targets:
                continue

            # Create property if not exists
            prop_uri = DOMAIN_NAMESPACES['dt'][prop_name]
            if prop_uri not in self.object_properties:
                self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop_uri, RDFS.label, Literal(prop_name.replace('_', ' '))))
                self.object_properties.add(prop_uri)

            for target in targets:
                target_uri = self.resolve_term_uri(target, domain)

                # Create named restriction class
                term_local = str(term_uri).split('#')[-1].split('/')[-1]
                prop_local = prop_name
                target_local = str(target_uri).split('#')[-1].split('/')[-1]
                restriction_name = f"{term_local}_{prop_local}_{target_local}_Restriction"
                restriction = self.resolve_term_uri(restriction_name, domain)

                self.graph.add((restriction, RDF.type, OWL.Class))
                self.graph.add((restriction, RDF.type, OWL.Restriction))
                self.graph.add((term_uri, RDFS.subClassOf, restriction))
                self.graph.add((restriction, OWL.onProperty, prop_uri))
                self.graph.add((restriction, OWL.someValuesFrom, target_uri))
                self.stats['restrictions'] += 1

                # Track cross-domain bridges
                self._track_bridge(term_uri, target_uri, prop_uri, block.term_id)

        # Process other custom relationships
        for rel_name, targets in block.other_relationships.items():
            prop_uri = DOMAIN_NAMESPACES['dt'][self.to_property_name(rel_name)]

            if prop_uri not in self.object_properties:
                self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
                self.graph.add((prop_uri, RDFS.label, Literal(rel_name.replace('-', ' '))))
                self.object_properties.add(prop_uri)

            for target in targets:
                target_uri = self.resolve_term_uri(target, domain)

                # Create restriction
                term_local = str(term_uri).split('#')[-1].split('/')[-1]
                prop_local = self.to_property_name(rel_name)
                target_local = str(target_uri).split('#')[-1].split('/')[-1]
                restriction_name = f"{term_local}_{prop_local}_{target_local}_Restriction"
                restriction = self.resolve_term_uri(restriction_name, domain)

                self.graph.add((restriction, RDF.type, OWL.Class))
                self.graph.add((restriction, RDF.type, OWL.Restriction))
                self.graph.add((term_uri, RDFS.subClassOf, restriction))
                self.graph.add((restriction, OWL.onProperty, prop_uri))
                self.graph.add((restriction, OWL.someValuesFrom, target_uri))
                self.stats['restrictions'] += 1

                self._track_bridge(term_uri, target_uri, prop_uri, block.term_id)

    def _parse_owl_axiom(self, owl_code: str, term_uri: URIRef, term_id: str):
        """Parse OWL functional syntax axioms."""
        # Class declarations from (Declaration (Class :ClassName))
        for match in re.finditer(r'Declaration\s*\(\s*Class\s+:(\w+)\s*\)', owl_code):
            cls_uri = self.resolve_term_uri(match.group(1))
            self.graph.add((cls_uri, RDF.type, OWL.Class))
            self.graph.add((cls_uri, RDFS.label, Literal(match.group(1), lang='en')))
            self.stats['classes'] += 1

        # SubClassOf (simple)
        for match in re.finditer(r'SubClassOf\s+:(\w+)\s+:(\w+)\s*\)', owl_code):
            child = self.resolve_term_uri(match.group(1))
            parent = self.resolve_term_uri(match.group(2))
            self.graph.add((child, RDF.type, OWL.Class))
            self.graph.add((parent, RDF.type, OWL.Class))
            self.graph.add((child, RDFS.subClassOf, parent))
            self.stats['subclass'] += 1

        # ObjectSomeValuesFrom restrictions
        pattern = r'SubClassOf\s+:(\w+)\s+\(\s*ObjectSomeValuesFrom\s+:(\w+)\s+:(\w+)\s*\)'
        for match in re.finditer(pattern, owl_code):
            subject = self.resolve_term_uri(match.group(1))
            prop = self.resolve_term_uri(match.group(2))
            obj = self.resolve_term_uri(match.group(3))

            # Create named restriction
            subject_local = str(subject).split('#')[-1].split('/')[-1]
            prop_local = str(prop).split('#')[-1].split('/')[-1]
            obj_local = str(obj).split('#')[-1].split('/')[-1]
            restriction_name = f"{subject_local}_{prop_local}_{obj_local}_Restriction"
            restriction = self.resolve_term_uri(restriction_name)

            self.graph.add((restriction, RDF.type, OWL.Class))
            self.graph.add((restriction, RDF.type, OWL.Restriction))
            self.graph.add((subject, RDFS.subClassOf, restriction))
            self.graph.add((restriction, OWL.onProperty, prop))
            self.graph.add((restriction, OWL.someValuesFrom, obj))
            self.stats['restrictions'] += 1
            self.object_properties.add(prop)

            self._track_bridge(subject, obj, prop, term_id)

        # ObjectProperty declarations
        for match in re.finditer(r'Declaration\s*\(\s*ObjectProperty\s+:(\w+)\s*\)', owl_code):
            prop = self.resolve_term_uri(match.group(1))
            self.graph.add((prop, RDF.type, OWL.ObjectProperty))
            self.object_properties.add(prop)
            self.stats['obj_props'] += 1

        # DataProperty declarations
        for match in re.finditer(r'Declaration\s*\(\s*DataProperty\s+:(\w+)\s*\)', owl_code):
            prop = self.resolve_term_uri(match.group(1))
            self.graph.add((prop, RDF.type, OWL.DatatypeProperty))
            self.data_properties.add(prop)
            self.stats['data_props'] += 1

    def _track_bridge(self, subject_uri: URIRef, object_uri: URIRef, property_uri: URIRef, term_id: str):
        """Track potential cross-domain bridges."""
        subject_ns = str(subject_uri).split('#')[0]
        object_ns = str(object_uri).split('#')[0]

        if subject_ns != object_ns:
            self.bridges.append({
                'source': subject_uri,
                'target': object_uri,
                'property': property_uri,
                'term_id': term_id
            })

    def add_cross_domain_bridges(self):
        """Generate explicit bridge axioms for cross-domain relationships."""
        for bridge in self.bridges:
            bridge_node = BNode()
            self.graph.add((bridge_node, RDF.type, DOMAIN_NAMESPACES['dt'].CrossDomainBridge))
            self.graph.add((bridge_node, DOMAIN_NAMESPACES['dt'].bridgeSource, bridge['source']))
            self.graph.add((bridge_node, DOMAIN_NAMESPACES['dt'].bridgeTarget, bridge['target']))
            self.graph.add((bridge_node, DOMAIN_NAMESPACES['dt'].bridgeProperty, bridge['property']))
            self.stats['bridges'] += 1

    def add_ontology_metadata(self):
        """Add ontology-level metadata."""
        ont = URIRef("http://narrativegoldmine.com/ontology/unified")

        self.graph.add((ont, RDF.type, OWL.Ontology))
        self.graph.add((ont, RDFS.label,
                       Literal("Narrative Goldmine Unified Ontology", lang='en')))
        self.graph.add((ont, RDFS.comment, Literal(
            "Multi-domain ontology integrating AI, Blockchain, Metaverse, Robotics, "
            "Telecollaboration, and Disruptive Technologies. Generated from standardized OntologyBlock headers.",
            lang='en')))
        self.graph.add((ont, DCTERMS.created, Literal("2025-11-21", datatype=XSD.date)))
        self.graph.add((ont, OWL.versionInfo, Literal("2.0.0")))
        self.graph.add((ont, DCTERMS.creator, Literal("Logseq Ontology Tools")))

    def save_turtle(self, output_file: Path):
        """Save as Turtle with statistics."""
        self.add_ontology_metadata()
        self.add_cross_domain_bridges()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(output_file), format='turtle')

        # Statistics
        triples = len(self.graph)
        classes = len(list(self.graph.subjects(RDF.type, OWL.Class)))

        print(f"\n✅ Generated: {output_file}", file=sys.stderr)
        print(f"\n📊 Statistics:", file=sys.stderr)
        print(f"   Total Triples: {triples:,}", file=sys.stderr)
        print(f"   Classes: {classes:,}", file=sys.stderr)
        print(f"   Object Properties: {len(self.object_properties):,}", file=sys.stderr)
        print(f"   Data Properties: {len(self.data_properties):,}", file=sys.stderr)
        print(f"   SubClassOf: {self.stats['subclass']:,}", file=sys.stderr)
        print(f"   Restrictions: {self.stats['restrictions']:,}", file=sys.stderr)
        print(f"   Cross-Domain Bridges: {self.stats['bridges']:,}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Generate WebVOWL TTL from OntologyBlock headers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert directory
  python webvowl_header_only_converter.py --pages-dir mainKnowledgeGraph/pages --output outputs/ontology.ttl

  # View in WebVOWL
  # 1. Upload TTL to http://vowl.visualdataweb.org/webvowl.html
  # 2. Or convert to JSON: python ttl_to_webvowl_json.py --input outputs/ontology.ttl --output outputs/ontology.json
        """
    )
    parser.add_argument('--pages-dir', type=Path, default='../mainKnowledgeGraph/pages',
                       help='Logseq pages directory')
    parser.add_argument('--output', type=Path, default='outputs/ontology-header-only.ttl',
                       help='Output TTL file')

    args = parser.parse_args()

    if not args.pages_dir.exists():
        print(f"Error: Pages directory not found: {args.pages_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse ontology blocks
    print(f"Parsing ontology blocks from: {args.pages_dir}")
    ontology_parser = OntologyBlockParser()
    blocks = ontology_parser.parse_directory(args.pages_dir)

    print(f"Found {len(blocks)} ontology blocks")

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

    # Build WebVOWL graph
    print(f"\nBuilding WebVOWL graph...")
    builder = WebVOWLOntologyBuilder()

    processed = 0
    for block in blocks:
        if builder.process_block(block):
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(blocks)} blocks...")

    # Save output
    builder.save_turtle(args.output)

    print(f"\n✅ Conversion complete!")
    print(f"   Output: {args.output}")
    print(f"   Processed: {processed}/{len(blocks)} blocks")


if __name__ == '__main__':
    main()
