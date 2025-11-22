#!/usr/bin/env python3
"""
Convert Ontology Blocks to JSON-LD Format
==========================================

Converts markdown files with OntologyBlock sections to JSON-LD format.
Supports all 6 domains: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies.

Uses the shared ontology_block_parser library for consistent parsing.

Usage:
    python convert-to-jsonld.py --input <input_path> --output <output_file>
    python convert-to-jsonld.py --input mainKnowledgeGraph/pages/ --output ontology.jsonld
    python convert-to-jsonld.py --input mainKnowledgeGraph/pages/rb-0100*.md --output robotics.jsonld

Features:
    - Generates JSON-LD with proper @context
    - Includes full IRIs for all entities
    - Supports all 6 domains
    - Uses schema.org and OWL vocabularies
    - Validates ontology blocks before conversion
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add lib directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG, STANDARD_NAMESPACES


# Map full domain names to prefix keys
DOMAIN_NAME_TO_PREFIX = {
    'artificial intelligence': 'ai',
    'ai': 'ai',
    'blockchain': 'bc',
    'bc': 'bc',
    'robotics': 'rb',
    'rb': 'rb',
    'metaverse': 'mv',
    'mv': 'mv',
    'telecollaboration': 'tc',
    'tc': 'tc',
    'disruptive technologies': 'dt',
    'dt': 'dt',
}


def normalize_domain(domain: Optional[str]) -> Optional[str]:
    """
    Normalize domain name to standard prefix.

    Args:
        domain: Domain name (e.g., "robotics" or "rb")

    Returns:
        Standard domain prefix (e.g., "rb") or None
    """
    if not domain:
        return None
    domain_lower = domain.lower().strip()
    return DOMAIN_NAME_TO_PREFIX.get(domain_lower, domain_lower)


def create_jsonld_context() -> Dict[str, Any]:
    """
    Create JSON-LD @context with all namespace mappings.
    Includes all 6 domain namespaces plus standard vocabularies.
    """
    context = {
        "@context": {
            # Standard namespaces
            "owl": STANDARD_NAMESPACES['owl'],
            "rdfs": STANDARD_NAMESPACES['rdfs'],
            "rdf": STANDARD_NAMESPACES['rdf'],
            "xsd": STANDARD_NAMESPACES['xsd'],
            "dcterms": STANDARD_NAMESPACES['dcterms'],
            "skos": STANDARD_NAMESPACES['skos'],

            # Schema.org
            "schema": "http://schema.org/",

            # Domain namespaces
            **{domain: config['namespace'] for domain, config in DOMAIN_CONFIG.items()},

            # Common OWL/RDF terms
            "Class": "owl:Class",
            "Property": "rdf:Property",
            "ObjectProperty": "owl:ObjectProperty",
            "DatatypeProperty": "owl:DatatypeProperty",

            # Common properties with @id type
            "label": {"@id": "rdfs:label"},
            "comment": {"@id": "rdfs:comment"},
            "definition": {"@id": "skos:definition"},
            "prefLabel": {"@id": "skos:prefLabel"},
            "altLabel": {"@id": "skos:altLabel"},
            "subClassOf": {"@id": "rdfs:subClassOf", "@type": "@id"},
            "domain": {"@id": "rdfs:domain", "@type": "@id"},
            "range": {"@id": "rdfs:range", "@type": "@id"},
            "type": {"@id": "rdf:type", "@type": "@id"},

            # Schema.org terms
            "name": {"@id": "schema:name"},
            "description": {"@id": "schema:description"},
            "version": {"@id": "schema:version"},
            "dateModified": {"@id": "schema:dateModified"},

            # Custom ontology properties
            "termId": {"@id": "dcterms:identifier"},
            "preferredTerm": {"@id": "skos:prefLabel"},
            "sourceDomain": {"@id": "dcterms:subject"},
            "status": {"@id": "schema:status"},
            "publicAccess": {"@id": "schema:isAccessibleForFree", "@type": "xsd:boolean"},
            "lastUpdated": {"@id": "dcterms:modified", "@type": "xsd:date"},
            "maturity": {"@id": "schema:creativeWorkStatus"},
            "qualityScore": {"@id": "schema:ratingValue", "@type": "xsd:float"},
            "authorityScore": {"@id": "schema:aggregateRating", "@type": "xsd:float"},
            "scopeNote": {"@id": "skos:scopeNote"},

            # Relationship properties
            "hasPart": {"@id": "dcterms:hasPart", "@type": "@id"},
            "isPartOf": {"@id": "dcterms:isPartOf", "@type": "@id"},
            "requires": {"@id": "schema:requires", "@type": "@id"},
            "dependsOn": {"@id": "schema:softwareRequirements", "@type": "@id"},
            "enables": {"@id": "schema:enables", "@type": "@id"},
            "relatesTo": {"@id": "dcterms:relation", "@type": "@id"},
        }
    }
    return context


def block_to_jsonld(block: OntologyBlock) -> Dict[str, Any]:
    """
    Convert an OntologyBlock to a JSON-LD object.

    Args:
        block: OntologyBlock instance

    Returns:
        JSON-LD object dictionary
    """
    # Get full IRI for this entity
    full_iri = block.get_full_iri()
    if not full_iri:
        # Fallback to term-id if no owl:class
        domain = normalize_domain(block.get_domain())
        if domain and domain in DOMAIN_CONFIG:
            full_iri = DOMAIN_CONFIG[domain]['namespace'] + (block.term_id or 'Unknown')

    obj = {
        "@id": full_iri,
        "@type": "owl:Class"
    }

    # Tier 1 Properties - Identification
    if block.term_id:
        obj["termId"] = block.term_id

    if block.preferred_term:
        obj["prefLabel"] = {"@value": block.preferred_term, "@language": "en"}
        obj["label"] = {"@value": block.preferred_term, "@language": "en"}

    if block.source_domain:
        obj["sourceDomain"] = block.source_domain

    if block.status:
        obj["status"] = block.status

    if block.public_access is not None:
        obj["publicAccess"] = block.public_access

    if block.last_updated:
        obj["lastUpdated"] = block.last_updated

    # Tier 1 Properties - Definition
    if block.definition:
        obj["definition"] = {"@value": block.definition, "@language": "en"}
        obj["comment"] = {"@value": block.definition, "@language": "en"}

    # Tier 1 Properties - Semantic Classification
    if block.owl_physicality:
        obj["owl:physicality"] = block.owl_physicality

    if block.owl_role:
        obj["owl:role"] = block.owl_role

    # Tier 1 Properties - Relationships
    if block.is_subclass_of:
        if len(block.is_subclass_of) == 1:
            obj["subClassOf"] = {"@id": convert_reference_to_iri(block.is_subclass_of[0], block)}
        else:
            obj["subClassOf"] = [{"@id": convert_reference_to_iri(ref, block)} for ref in block.is_subclass_of]

    # Tier 2 Properties - Identification
    if block.alt_terms:
        obj["altLabel"] = [{"@value": term, "@language": "en"} for term in block.alt_terms]

    if block.version:
        obj["version"] = block.version

    if block.quality_score is not None:
        obj["qualityScore"] = block.quality_score

    if block.cross_domain_links is not None:
        obj["crossDomainLinks"] = block.cross_domain_links

    # Tier 2 Properties - Definition
    if block.maturity:
        obj["maturity"] = block.maturity

    if block.source:
        obj["source"] = block.source

    if block.authority_score is not None:
        obj["authorityScore"] = block.authority_score

    if block.scope_note:
        obj["scopeNote"] = {"@value": block.scope_note, "@language": "en"}

    # Tier 2 Properties - Semantic Classification
    if block.owl_inferred_class:
        obj["owl:inferredClass"] = block.owl_inferred_class

    if block.belongs_to_domain:
        obj["belongsToDomain"] = [convert_reference_to_iri(ref, block) for ref in block.belongs_to_domain]

    if block.implemented_in_layer:
        obj["implementedInLayer"] = block.implemented_in_layer

    # Tier 2 Properties - Relationships
    if block.has_part:
        obj["hasPart"] = [{"@id": convert_reference_to_iri(ref, block)} for ref in block.has_part]

    if block.is_part_of:
        obj["isPartOf"] = [{"@id": convert_reference_to_iri(ref, block)} for ref in block.is_part_of]

    if block.requires:
        obj["requires"] = [{"@id": convert_reference_to_iri(ref, block)} for ref in block.requires]

    if block.depends_on:
        obj["dependsOn"] = [{"@id": convert_reference_to_iri(ref, block)} for ref in block.depends_on]

    if block.enables:
        obj["enables"] = [{"@id": convert_reference_to_iri(ref, block)} for ref in block.enables]

    if block.relates_to:
        obj["relatesTo"] = [{"@id": convert_reference_to_iri(ref, block)} for ref in block.relates_to]

    # Cross-domain bridges
    if block.bridges_to:
        obj["bridgesTo"] = block.bridges_to

    if block.bridges_from:
        obj["bridgesFrom"] = block.bridges_from

    # Domain-specific extensions
    if block.domain_extensions:
        for prop_name, prop_value in block.domain_extensions.items():
            obj[f"domain:{prop_name}"] = prop_value

    # Other relationships
    for rel_name, targets in block.other_relationships.items():
        obj[rel_name] = [convert_reference_to_iri(ref, block) for ref in targets]

    # Add file source metadata
    obj["schema:fileSource"] = str(block.file_path)

    return obj


def convert_reference_to_iri(reference: str, block: OntologyBlock) -> str:
    """
    Convert a reference (e.g., [[TermName]] or term-id) to a full IRI.

    Args:
        reference: Reference string
        block: Source OntologyBlock for context

    Returns:
        Full IRI string
    """
    # Remove wiki-link brackets if present
    ref = reference.strip()
    if ref.startswith('[[') and ref.endswith(']]'):
        ref = ref[2:-2]

    # If already a full URI, return as-is
    if ref.startswith('http://') or ref.startswith('https://'):
        return ref

    # If it's a prefixed name (e.g., "ai:Concept"), expand it
    if ':' in ref:
        prefix, localname = ref.split(':', 1)
        if prefix in DOMAIN_CONFIG:
            return DOMAIN_CONFIG[prefix]['namespace'] + localname
        if prefix in STANDARD_NAMESPACES:
            return STANDARD_NAMESPACES[prefix] + localname

    # Otherwise, assume it's from the same domain as the block
    domain = normalize_domain(block.get_domain())
    if domain and domain in DOMAIN_CONFIG:
        return DOMAIN_CONFIG[domain]['namespace'] + ref

    # Fallback: return as-is
    return ref


def convert_blocks_to_jsonld(blocks: List[OntologyBlock]) -> Dict[str, Any]:
    """
    Convert multiple OntologyBlocks to a JSON-LD document.

    Args:
        blocks: List of OntologyBlock instances

    Returns:
        Complete JSON-LD document
    """
    context = create_jsonld_context()

    graph = []
    for block in blocks:
        try:
            obj = block_to_jsonld(block)
            graph.append(obj)
        except Exception as e:
            print(f"Warning: Failed to convert {block.term_id}: {e}", file=sys.stderr)
            continue

    jsonld_doc = {
        "@context": context["@context"],
        "@graph": graph
    }

    return jsonld_doc


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Convert OntologyBlocks to JSON-LD format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all files in a directory
  python convert-to-jsonld.py --input mainKnowledgeGraph/pages/ --output ontology.jsonld

  # Convert a single file
  python convert-to-jsonld.py --input pages/rb-0100.md --output robotics-term.jsonld

  # Convert files matching a pattern
  python convert-to-jsonld.py --input "pages/AI-*.md" --output ai-ontology.jsonld
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input path: directory, file, or glob pattern'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON-LD file path'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate blocks before conversion (show warnings for incomplete blocks)'
    )

    parser.add_argument(
        '--pretty',
        action='store_true',
        default=True,
        help='Pretty-print JSON output (default: True)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OntologyBlock to JSON-LD Conversion")
    print("=" * 80)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Parse input
    parser_obj = OntologyBlockParser()
    blocks = []

    if input_path.is_file():
        print(f"\nParsing file: {input_path}")
        block = parser_obj.parse_file(input_path)
        if block:
            blocks.append(block)
    elif input_path.is_dir():
        print(f"\nParsing directory: {input_path}")
        blocks = parser_obj.parse_directory(input_path)
    else:
        # Try glob pattern
        print(f"\nParsing pattern: {args.input}")
        from glob import glob
        for file_path in glob(args.input):
            block = parser_obj.parse_file(Path(file_path))
            if block:
                blocks.append(block)

    if not blocks:
        print("\n❌ Error: No valid ontology blocks found")
        sys.exit(1)

    print(f"\n✅ Successfully parsed {len(blocks)} ontology blocks")

    # Statistics by domain
    domain_counts = defaultdict(int)
    for block in blocks:
        domain = normalize_domain(block.get_domain())
        if domain:
            domain_counts[domain] += 1

    print("\n📊 Statistics by domain:")
    for domain, count in sorted(domain_counts.items()):
        domain_name = DOMAIN_CONFIG[domain]['full_name'] if domain in DOMAIN_CONFIG else domain
        print(f"   {domain_name}: {count}")

    # Validation
    if args.validate:
        print("\n🔍 Validating blocks...")
        total_errors = 0
        blocks_with_errors = 0
        for block in blocks:
            errors = block.validate()
            if errors:
                blocks_with_errors += 1
                total_errors += len(errors)
                print(f"\n⚠️  {block.term_id}:")
                for error in errors:
                    print(f"   - {error}")

        if blocks_with_errors > 0:
            print(f"\n⚠️  Validation: {blocks_with_errors}/{len(blocks)} blocks have errors")
            print(f"   (Total {total_errors} errors)")
        else:
            print("\n✅ Validation: All blocks valid")

    # Convert to JSON-LD
    print("\n🔄 Converting to JSON-LD...")
    jsonld_doc = convert_blocks_to_jsonld(blocks)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(jsonld_doc, f, indent=2, ensure_ascii=False)
        else:
            json.dump(jsonld_doc, f, ensure_ascii=False)

    print(f"\n✅ JSON-LD file saved: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"   Total entities: {len(jsonld_doc['@graph'])}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
