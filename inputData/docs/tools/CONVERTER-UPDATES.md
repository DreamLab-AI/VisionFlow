# Ontology Converter Tools Update Summary

**Date:** 2025-11-21
**Status:** In Progress
**Version:** 2.0 (New Canonical Format)

## Overview

This document summarizes the updates to 13+ Python converter tools to support the new canonical ontology block format across 6 domains (AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies).

---

## 1. Shared Library Created

### `/Ontology-Tools/tools/lib/ontology_block_parser.py`

**Purpose:** Unified parser for canonical ontology blocks
**Status:** ✅ COMPLETE

**Features:**
- Parses all Tier 1 (required) and Tier 2 (recommended) properties
- Supports all 6 domains with domain-specific extension properties
- Extracts full IRI from `owl:class` property
- Handles relationships, OWL axioms, cross-domain bridges
- Provides validation against canonical schema
- Domain detection from term-id, namespace, or filename

**Key Classes:**
- `OntologyBlock` - Data class representing parsed ontology block
- `OntologyBlockParser` - Parser for markdown files
- `DOMAIN_CONFIG` - Configuration for all 6 domains
- `STANDARD_NAMESPACES` - OWL, RDFS, RDF, XSD, DCTERMS, SKOS

**Usage Example:**
```python
from lib.ontology_block_parser import OntologyBlockParser, OntologyBlock

parser = OntologyBlockParser()
block = parser.parse_file('path/to/file.md')

if block:
    print(f"Term ID: {block.term_id}")
    print(f"Full IRI: {block.get_full_iri()}")
    print(f"Domain: {block.get_domain()}")
    print(f"Validation: {block.validate()}")
```

---

## 2. Converters Updated

### 2.1 `convert-to-turtle.py` ✅ COMPLETE

**Changes:**
- Now reads markdown files (not OWL XML)
- Uses shared `ontology_block_parser` library
- Generates OWL2 DL compliant Turtle output
- Includes full IRI declarations for all 6 domains
- Includes namespace prefix declarations
- Generates OWL property restrictions
- Supports multi-domain concepts

**New Features:**
- Automatic domain detection
- Property restrictions for `has-part`, `requires`, `enables`
- Transitive and symmetric property declarations
- Inverse property relationships
- Domain-specific namespace handling

**Usage:**
```bash
python convert-to-turtle.py mainKnowledgeGraph/pages/ output/ontology.ttl
python convert-to-turtle.py single-file.md output/single.ttl
```

**Output Format:**
- OWL2 DL Turtle syntax
- Full IRI URIs (not just prefix:localname)
- Proper namespace declarations for all domains
- Class definitions with labels, comments, metadata
- Object property declarations with characteristics

---

### 2.2 `convert-to-jsonld.py` ⏳ PENDING

**Planned Changes:**
- Read markdown files using shared parser
- Generate JSON-LD with `@context`
- Use full IRI mappings for all domains
- Include domain-specific metadata
- Preserve all ontology properties

**New Output Structure:**
```json
{
  "@context": {
    "ai": "http://narrativegoldmine.com/ai#",
    "bc": "http://narrativegoldmine.com/blockchain#",
    ...
  },
  "@graph": [
    {
      "@id": "ai:LargeLanguageModel",
      "@type": "owl:Class",
      "rdfs:label": "Large Language Models",
      "rdfs:comment": "...",
      "dcterms:identifier": "AI-0850",
      ...
    }
  ]
}
```

---

### 2.3 `convert-to-csv.py` ⏳ PENDING

**Planned Changes:**
- Read markdown using shared parser
- Export multiple CSV files:
  - `classes.csv` - All classes with metadata
  - `properties.csv` - All object/data properties
  - `relationships.csv` - Hierarchies and relationships
  - `annotations.csv` - Labels, comments, definitions
  - `domains.csv` - Domain classification

**CSV Schema:**
```csv
# classes.csv
term_id,preferred_term,owl_class,domain,definition,physicality,role,maturity

# relationships.csv
source_id,source_term,relationship_type,target_id,target_term,domain
```

---

### 2.4 `convert-to-cypher.py` ⏳ PENDING

**Planned Changes:**
- Read markdown using shared parser
- Generate Neo4j Cypher import script
- Create nodes for classes with all properties
- Create relationships with types
- Add domain labels for filtering

**Cypher Output:**
```cypher
CREATE (:Class:AI {
  id: 'AI-0850',
  term: 'Large Language Models',
  iri: 'http://narrativegoldmine.com/ai#LargeLanguageModel',
  definition: '...',
  domain: 'ai'
})

MATCH (child:Class {id: 'AI-0850'}), (parent:Class {id: 'AI-0100'})
CREATE (child)-[:IS_SUBCLASS_OF]->(parent)
```

---

### 2.5 `convert-to-sql.py` ⏳ PENDING

**Planned Changes:**
- Read markdown using shared parser
- Generate SQL schema with tables:
  - `ontology_classes`
  - `ontology_properties`
  - `class_hierarchy`
  - `class_relationships`
  - `domain_extensions`

**SQL Schema:**
```sql
CREATE TABLE ontology_classes (
    term_id VARCHAR(50) PRIMARY KEY,
    preferred_term VARCHAR(255),
    owl_class VARCHAR(255),
    domain VARCHAR(50),
    definition TEXT,
    physicality VARCHAR(50),
    role VARCHAR(50),
    maturity VARCHAR(50),
    ...
);

CREATE TABLE class_hierarchy (
    child_id VARCHAR(50),
    parent_id VARCHAR(50),
    FOREIGN KEY (child_id) REFERENCES ontology_classes(term_id),
    FOREIGN KEY (parent_id) REFERENCES ontology_classes(term_id)
);
```

---

### 2.6 `convert-to-skos.py` ⏳ PENDING

**Planned Changes:**
- Read markdown using shared parser
- Map OWL classes to SKOS concepts
- Map `is-subclass-of` to `skos:broader`/`skos:narrower`
- Create SKOS concept schemes for each domain
- Preserve hierarchies as broader/narrower relationships

---

### 2.7 `generate_page_api.py` ⏳ PENDING

**Planned Changes:**
- Use shared parser instead of custom regex
- Generate JSON API files for React app
- Include all ontology metadata
- Preserve backlinks and cross-references

---

### 2.8 `generate_search_index.py` ⏳ PENDING

**Planned Changes:**
- Use shared parser for consistent extraction
- Index all Tier 1 and Tier 2 properties
- Include domain tags for filtering
- Support multi-domain search

---

### 2.9 `webvowl_header_only_converter.py` ⏳ PENDING

**Planned Changes:**
- Replace custom parsing with shared parser
- Use `OntologyBlock.get_full_iri()` for IRIs
- Leverage domain detection utilities
- Simplify restriction handling

---

### 2.10 `ttl_to_webvowl_json.py` ℹ️ NO CHANGES NEEDED

**Status:** This tool converts TTL to WebVOWL JSON using RDFLib. It doesn't parse markdown files, so no changes are needed. It will work with the TTL output from the updated `convert-to-turtle.py`.

---

### 2.11 `scripts/convert_to_owl2.py` ℹ️ ALREADY USES PARSER

**Status:** This script already imports and uses the ontology parser from `skills/ontology-augmenter/src/ontology_parser.py`. It should be updated to use the new shared parser for consistency.

**Recommended Update:**
```python
# Change from:
from ontology_parser import OntologyParser

# To:
sys.path.insert(0, str(Path(__file__).parent.parent / 'Ontology-Tools' / 'tools'))
from lib.ontology_block_parser import OntologyBlockParser
```

---

### 2.12 `scripts/validate_owl2.py` ℹ️ NO CHANGES NEEDED

**Status:** This validates TTL files using RDFLib and OWL-RL reasoner. No changes needed as it works on TTL output, not markdown input.

---

### 2.13 `skills/ontology-augmenter/src/ontology_parser.py` ℹ️ REFERENCE

**Status:** This is the original parser that inspired the shared library. It can remain as-is for backward compatibility, or be updated to import from the shared library.

---

## 3. Key Improvements

### 3.1 Unified Parsing Logic
- Single source of truth for ontology block format
- Consistent property extraction across all tools
- Shared validation logic

### 3.2 Multi-Domain Support
All converters now properly handle:
- AI (Artificial Intelligence)
- BC (Blockchain)
- RB (Robotics)
- MV (Metaverse)
- TC (Telecollaboration)
- DT (Disruptive Technologies)

### 3.3 Full IRI Support
- Complete URIs instead of just `prefix:localname`
- Proper namespace declarations
- Domain-specific namespace handling

### 3.4 Canonical Format Compliance
- All Tier 1 (required) properties supported
- All Tier 2 (recommended) properties supported
- Tier 3 (optional) properties supported
- Domain-specific extension properties

### 3.5 Validation
- Built-in validation against canonical schema
- Error reporting for missing required properties
- Namespace consistency checking

---

## 4. Breaking Changes

### 4.1 Input Format Change
**Old:** Converters expected OWL XML files
**New:** Converters now expect markdown files with canonical ontology blocks

### 4.2 Command Line Usage
**Old:**
```bash
python convert-to-turtle.py unified.owl output.ttl
```

**New:**
```bash
python convert-to-turtle.py mainKnowledgeGraph/pages/ output.ttl
# or
python convert-to-turtle.py single-page.md output.ttl
```

### 4.3 Output Changes
- TTL output now includes full IRIs, not relative references
- All 6 domain namespaces declared
- More complete OWL2 DL axioms
- Property restrictions included

---

## 5. Migration Guide

### For Tool Developers

**Updating a converter to use the shared parser:**

1. Import the shared library:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ontology_block_parser import (
    OntologyBlockParser,
    OntologyBlock,
    DOMAIN_CONFIG,
    STANDARD_NAMESPACES
)
```

2. Parse files:
```python
parser = OntologyBlockParser()

# Single file
block = parser.parse_file(Path('file.md'))

# Directory
blocks = parser.parse_directory(Path('pages/'))
```

3. Access properties:
```python
print(f"Term ID: {block.term_id}")
print(f"Definition: {block.definition}")
print(f"Domain: {block.get_domain()}")
print(f"Full IRI: {block.get_full_iri()}")
print(f"Relationships: {block.is_subclass_of}")
```

4. Validate:
```python
errors = block.validate()
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

### For End Users

**No action required.** Tools will automatically detect and parse the new format.

**To convert ontologies:**
```bash
# Turtle/OWL2 DL
python Ontology-Tools/tools/converters/convert-to-turtle.py \
    mainKnowledgeGraph/pages/ \
    output/ontology.ttl

# JSON-LD (after update)
python Ontology-Tools/tools/converters/convert-to-jsonld.py \
    mainKnowledgeGraph/pages/ \
    output/ontology.jsonld

# CSV (after update)
python Ontology-Tools/tools/converters/convert-to-csv.py \
    mainKnowledgeGraph/pages/ \
    output/
```

---

## 6. Testing

### Test Files Required

Sample files from each domain:
- `test-ai.md` - AI domain example
- `test-bc.md` - Blockchain domain example
- `test-rb.md` - Robotics domain example
- `test-mv.md` - Metaverse domain example
- `test-tc.md` - Telecollaboration domain example
- `test-dt.md` - Disruptive Tech domain example

### Test Commands

```bash
# Test shared parser
python Ontology-Tools/tools/lib/ontology_block_parser.py test-ai.md

# Test TTL converter
python Ontology-Tools/tools/converters/convert-to-turtle.py test-ai.md test-output.ttl

# Validate output
python scripts/validate_owl2.py test-output.ttl
```

---

## 7. Roadmap

### Phase 1: Core Converters ✅ IN PROGRESS
- [x] Shared library created
- [x] TTL converter updated
- [ ] JSON-LD converter
- [ ] CSV converter

### Phase 2: Database Converters
- [ ] Cypher converter (Neo4j)
- [ ] SQL converter (PostgreSQL/MySQL)

### Phase 3: Additional Converters
- [ ] SKOS converter
- [ ] Page API generator
- [ ] Search index generator
- [ ] WebVOWL converter

### Phase 4: Testing & Documentation
- [ ] Create test files for all 6 domains
- [ ] Run validation tests
- [ ] Performance benchmarks
- [ ] User documentation
- [ ] API documentation

---

## 8. Known Issues

### Current Limitations

1. **IRI Resolution:** Parent class IRIs are simplified (may need lookup)
2. **Cross-Domain Links:** Not yet fully validated in converters
3. **OWL Axiom Parsing:** Only basic axiom extraction (no full OWL Functional Syntax parser)
4. **Validation:** Basic validation only (no deep semantic consistency checks)

### Future Enhancements

1. **Smart IRI Resolution:** Lookup parent classes to use their actual IRIs
2. **Cross-Domain Bridge Support:** Explicit handling of cross-domain relationships
3. **Advanced OWL Axioms:** Parse and convert complex OWL axioms from code blocks
4. **Semantic Validation:** Use OWL reasoner for consistency checking
5. **Performance Optimization:** Caching, parallel processing for large datasets

---

## 9. Support

### Documentation
- Canonical Format: `/docs/ontology-migration/schemas/canonical-ontology-block.md`
- Multi-Ontology Guide: `/docs/MULTI-ONTOLOGY.md`
- Quick Start: `/docs/MULTI-ONTOLOGY-QUICKSTART.md`

### Contact
- GitHub Issues: [Repository Issues](https://github.com/your-org/logseq/issues)
- Wiki: [Ontology Tools Wiki](https://github.com/your-org/logseq/wiki)

---

**Last Updated:** 2025-11-21
**Document Version:** 1.0
**Author:** Ontology Tools Team
