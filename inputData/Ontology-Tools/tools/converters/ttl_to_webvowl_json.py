#!/usr/bin/env python3
"""
Convert Turtle OWL to WebVOWL JSON format.
Pure Python implementation - no Java required.
Works with TTL files generated from ontology_block_parser.

WebVOWL is a visualization tool for OWL ontologies:
http://vowl.visualdataweb.org/webvowl.html

Usage:
    python ttl_to_webvowl_json.py --input ontology.ttl --output ontology.json

Then visualize:
    1. Go to http://vowl.visualdataweb.org/webvowl.html
    2. Click "Ontology" menu
    3. Select "Select ontology file"
    4. Upload the generated JSON file
"""

import json
import sys
import argparse
from pathlib import Path
from rdflib import Graph, RDF, RDFS, OWL
from collections import defaultdict


# Domain colors for WebVOWL visualization
DOMAIN_COLORS = {
    'ai': '#4CAF50',           # Green for AI
    'bc': '#2196F3',           # Blue for Blockchain
    'rb': '#FF9800',           # Orange for Robotics
    'mv': '#9C27B0',           # Purple for Metaverse
    'tc': '#00BCD4',           # Cyan for Telecollaboration
    'dt': '#F44336'            # Red for Disruptive Tech
}


def get_domain_from_iri(iri_str: str) -> str:
    """Extract domain from IRI namespace."""
    if 'artificial-intelligence' in iri_str or '/ai#' in iri_str:
        return 'ai'
    elif 'blockchain' in iri_str or '/bc#' in iri_str:
        return 'bc'
    elif 'robotics' in iri_str or '/rb#' in iri_str:
        return 'rb'
    elif 'metaverse' in iri_str or '/mv#' in iri_str:
        return 'mv'
    elif 'telecollaboration' in iri_str or '/tc#' in iri_str:
        return 'tc'
    elif 'disruptive-tech' in iri_str or '/dt#' in iri_str:
        return 'dt'
    return 'unknown'


def ttl_to_webvowl(ttl_file: Path, output_file: Path):
    """Convert TTL ontology to WebVOWL JSON format."""

    # Load TTL
    print(f"Loading {ttl_file}...", file=sys.stderr)
    g = Graph()
    try:
        g.parse(str(ttl_file), format='turtle')
    except Exception as e:
        print(f"Error parsing TTL file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(g)} triples", file=sys.stderr)

    # WebVOWL JSON structure
    vowl = {
        "header": {
            "languages": ["en"],
            "title": {"en": "Narrative Goldmine Ontology"},
            "description": {
                "en": "Multi-domain ontology with AI, Blockchain, Metaverse, Robotics, "
                      "Telecollaboration, and Disruptive Technologies"
            },
            "version": "2.0"
        },
        "namespace": [],
        "class": [],
        "classAttribute": [],
        "property": [],
        "propertyAttribute": []
    }

    # Track nodes and properties
    classes = {}
    properties = {}
    class_id = 0
    prop_id = 0

    # Extract namespaces
    for prefix, ns in g.namespaces():
        vowl["namespace"].append({
            "prefix": prefix,
            "iri": str(ns)
        })

    # Extract classes
    print("Extracting classes...", file=sys.stderr)
    for cls in g.subjects(RDF.type, OWL.Class):
        # Skip blank nodes and restrictions
        if not (isinstance(cls, str) or str(cls).startswith('http')):
            continue

        # Skip restriction classes (identified by _Restriction suffix)
        if str(cls).endswith('_Restriction') or str(cls).endswith('Restriction'):
            continue

        class_id += 1
        label = g.value(cls, RDFS.label)
        comment = g.value(cls, RDFS.comment)
        iri_str = str(cls)

        classes[iri_str] = class_id

        # Add class node
        vowl["class"].append({
            "id": str(class_id),
            "type": "owl:Class"
        })

        # Extract base IRI and fragment
        if '#' in iri_str:
            base_iri, fragment = iri_str.rsplit('#', 1)
        elif '/' in iri_str:
            base_iri, fragment = iri_str.rsplit('/', 1)
        else:
            base_iri = "http://narrativegoldmine.com/ontology"
            fragment = iri_str

        # Get domain for coloring
        domain = get_domain_from_iri(iri_str)
        color = DOMAIN_COLORS.get(domain, '#808080')

        attrs = {
            "id": str(class_id),
            "iri": iri_str,
            "baseIri": base_iri,
            "attributes": ["external"] if domain == 'unknown' else []
        }

        # Add color for domain visualization
        attrs["attributes"].append("colored")
        attrs["backgroundColor"] = color

        if label:
            attrs["label"] = {"en": str(label)}
        else:
            attrs["label"] = {"en": fragment}

        if comment:
            # Truncate long comments for better visualization
            comment_text = str(comment)
            if len(comment_text) > 200:
                comment_text = comment_text[:197] + "..."
            attrs["comment"] = {"en": comment_text}

        vowl["classAttribute"].append(attrs)

    print(f"Extracted {len(classes)} classes", file=sys.stderr)

    # Start property IDs AFTER all class IDs to avoid overlap
    prop_id = class_id

    # Extract object properties
    print("Extracting object properties...", file=sys.stderr)
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        if not (isinstance(prop, str) or str(prop).startswith('http')):
            continue

        prop_id += 1
        label = g.value(prop, RDFS.label)
        domain = g.value(prop, RDFS.domain)
        range_val = g.value(prop, RDFS.range)
        prop_iri = str(prop)

        properties[prop_iri] = prop_id

        # Add property node
        vowl["property"].append({
            "id": str(prop_id),
            "type": "owl:objectProperty"
        })

        # Extract base IRI
        if '#' in prop_iri:
            prop_base, prop_frag = prop_iri.rsplit('#', 1)
        elif '/' in prop_iri:
            prop_base, prop_frag = prop_iri.rsplit('/', 1)
        else:
            prop_base = "http://narrativegoldmine.com/ontology"
            prop_frag = prop_iri

        attrs = {
            "id": str(prop_id),
            "iri": prop_iri,
            "baseIri": prop_base,
            "attributes": ["object"]
        }

        # Add domain/range (use first class if not specified)
        if domain and str(domain) in classes:
            attrs["domain"] = str(classes[str(domain)])
        else:
            attrs["domain"] = "1"  # Default to first class

        if range_val and str(range_val) in classes:
            attrs["range"] = str(classes[str(range_val)])
        else:
            attrs["range"] = "1"  # Default to first class

        if label:
            attrs["label"] = {"en": str(label)}
        else:
            attrs["label"] = {"en": prop_frag}

        vowl["propertyAttribute"].append(attrs)

    # Extract datatype properties
    print("Extracting datatype properties...", file=sys.stderr)
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        if not (isinstance(prop, str) or str(prop).startswith('http')):
            continue

        prop_id += 1
        label = g.value(prop, RDFS.label)
        domain = g.value(prop, RDFS.domain)
        prop_iri = str(prop)

        properties[prop_iri] = prop_id

        # Add property node
        vowl["property"].append({
            "id": str(prop_id),
            "type": "owl:datatypeProperty"
        })

        # Extract base IRI
        if '#' in prop_iri:
            prop_base, prop_frag = prop_iri.rsplit('#', 1)
        elif '/' in prop_iri:
            prop_base, prop_frag = prop_iri.rsplit('/', 1)
        else:
            prop_base = "http://narrativegoldmine.com/ontology"
            prop_frag = prop_iri

        attrs = {
            "id": str(prop_id),
            "iri": prop_iri,
            "baseIri": prop_base,
            "attributes": ["datatype"]
        }

        # Add domain
        if domain and str(domain) in classes:
            attrs["domain"] = str(classes[str(domain)])
        else:
            attrs["domain"] = "1"  # Default to first class

        if label:
            attrs["label"] = {"en": str(label)}
        else:
            attrs["label"] = {"en": prop_frag}

        vowl["propertyAttribute"].append(attrs)

    print(f"Extracted {len(properties)} properties", file=sys.stderr)

    # Extract subClassOf relationships
    print("Extracting subClassOf relationships...", file=sys.stderr)
    subclass_count = 0
    for subj, obj in g.subject_objects(RDFS.subClassOf):
        # Skip if either is a restriction or blank node
        subj_str = str(subj)
        obj_str = str(obj)

        if not (subj_str.startswith('http') and obj_str.startswith('http')):
            continue

        # Skip restriction classes
        if '_Restriction' in subj_str or '_Restriction' in obj_str:
            continue

        if subj_str in classes and obj_str in classes:
            prop_id += 1
            subclass_count += 1

            # Add to property array
            vowl["property"].append({
                "id": str(prop_id),
                "type": "rdfs:subClassOf"
            })

            # Add to propertyAttribute array
            vowl["propertyAttribute"].append({
                "id": str(prop_id),
                "domain": str(classes[subj_str]),
                "range": str(classes[obj_str]),
                "attributes": ["anonymous", "object"]
            })

    print(f"Extracted {subclass_count} subClassOf relationships", file=sys.stderr)

    # Save JSON
    print(f"\nWriting {output_file}...", file=sys.stderr)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vowl, f, indent=2, ensure_ascii=False)

    file_size_kb = output_file.stat().st_size / 1024

    print(f"\n✅ Generated WebVOWL JSON successfully!", file=sys.stderr)
    print(f"   Output: {output_file}", file=sys.stderr)
    print(f"   File size: {file_size_kb:.2f} KB", file=sys.stderr)
    print(f"   Classes: {len(classes)}", file=sys.stderr)
    print(f"   Properties: {len(properties)}", file=sys.stderr)
    print(f"   SubClassOf relationships: {subclass_count}", file=sys.stderr)
    print(f"\n📊 Visualize at: http://vowl.visualdataweb.org/webvowl.html", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Turtle OWL to WebVOWL JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert TTL to WebVOWL JSON
  python ttl_to_webvowl_json.py --input ontology.ttl --output ontology.json

  # Full workflow: Markdown -> TTL -> JSON
  python webvowl_header_only_converter.py --pages-dir mainKnowledgeGraph/pages --output outputs/ontology.ttl
  python ttl_to_webvowl_json.py --input outputs/ontology.ttl --output outputs/ontology.json

Visualization:
  1. Go to http://vowl.visualdataweb.org/webvowl.html
  2. Click "Ontology" menu -> "Select ontology file"
  3. Upload the generated JSON file

Domain Colors:
  - AI (Green): #4CAF50
  - Blockchain (Blue): #2196F3
  - Robotics (Orange): #FF9800
  - Metaverse (Purple): #9C27B0
  - Telecollaboration (Cyan): #00BCD4
  - Disruptive Tech (Red): #F44336
        """
    )
    parser.add_argument('--input', type=Path, required=True,
                       help='Input TTL file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSON file')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ttl_to_webvowl(args.input, args.output)


if __name__ == '__main__':
    main()
