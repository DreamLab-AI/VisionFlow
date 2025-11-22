# Tools Directory

This directory contains utilities, scripts, and tools for ontology development, validation, and conversion.

## Directory Structure

```
tools/
├── conversion/          # Format conversion tools
├── converters/          # Legacy and specialized converters
├── generators/          # Ontology generation and enhancement tools
├── utilities/           # General-purpose utility scripts
├── validation/          # Multi-level validation framework
└── validators/          # Legacy and specialized validators
```

## Conversion Tools

### conversion/
- `convert_owl_to_ttl.py` - Convert OWL/XML to Turtle format

### converters/
- `convert-to-csv.py` - Export ontology to CSV format
- `convert-to-cypher.py` - Export to Neo4j Cypher queries
- `convert-to-jsonld.py` - Convert to JSON-LD format
- `convert-to-skos.py` - Transform to SKOS vocabulary
- `convert-to-sql.py` - Export to SQL schema
- `convert-to-turtle.py` - Convert to Turtle format
- `simple-format-converter.py` - Multi-format converter

## Generation Tools

### generators/
- `generate_unified_ontology.py` - Create unified meta-ontology
- `generate_combined_ontology.py` - Combine domain ontologies
- `generate-foundational-terms.py` - Generate base vocabulary
- `complete_crypto_ontology.py` - Cryptocurrency ontology generator
- `aggregate-owl-files.py` - Merge multiple OWL files
- `enhance_p1_owl.py` - Enhance ontology with metadata

## Validation Tools

### validation/
Multi-level validation framework:
- `level1_syntactic_validator.py` - Syntax and structure validation
- `level2_semantic_validator.py` - Semantic consistency checks
- `level3_quality_metrics.py` - Quality and completeness metrics
- `level4_competency_validator.py` - Competency question validation
- `level5_statistics_reporter.py` - Comprehensive statistics
- `run_all_validations.py` - Execute full validation suite

### validators/
- `validate-owl-syntax.py` - OWL syntax validation
- `verify-ontology.py` - Ontology verification
- `check-and-validate.py` - Quick validation check
- `watch-and-validate.sh` - Continuous validation monitoring

## Utility Scripts

### utilities/
- `ontology-validator.py` - General ontology validator
- `ofn_to_owl_xml.py` - Convert OWL Functional Syntax to XML
- `comprehensive_owl_to_turtle.py` - Advanced OWL to Turtle converter
- `watch-and-validate.sh` - File watcher with auto-validation

## Usage Examples

### Convert OWL to Turtle
```bash
python tools/conversion/convert_owl_to_ttl.py input.owl output.ttl
```

### Run Full Validation
```bash
python tools/validation/run_all_validations.py \
  --input ontologies/unified/disruptive-technologies-meta-ontology-v1.0.0.ttl \
  --output validation-report.json
```

### Generate Unified Ontology
```bash
python tools/generators/generate_unified_ontology.py \
  --ai ontologies/artificial-intelligence/schemas/ai-v1.0.0.ttl \
  --blockchain ontologies/blockchain/schemas/blockchain-v1.0.0.ttl \
  --metaverse ontologies/metaverse/schemas/metaverse-v1.0.0.ttl \
  --robotics ontologies/robotics/schemas/robotics-v1.0.0.ttl \
  --output ontologies/unified/disruptive-technologies-meta-ontology-v1.0.0.ttl
```

### Convert to JSON-LD
```bash
python tools/converters/convert-to-jsonld.py \
  --input ontologies/metaverse/schemas/metaverse-v1.0.0.ttl \
  --output metaverse-v1.0.0.jsonld
```

## Requirements

Most tools require:
- Python 3.8+
- rdflib
- owlrl
- pyshacl

Install dependencies:
```bash
pip install rdflib owlrl pyshacl
```

## Validation Levels

The validation framework uses a 5-level approach:

1. **Syntactic** - RDF/OWL syntax correctness
2. **Semantic** - Logical consistency, no contradictions
3. **Quality** - Completeness, documentation, standards
4. **Competency** - Domain coverage via test queries
5. **Statistics** - Metrics and reporting

## Contributing

When adding new tools:
1. Place in appropriate subdirectory
2. Include docstrings and help text
3. Add usage examples to this README
4. Update requirements if adding dependencies

## Shared Libraries

### Core Libraries (Use These!)

All converter and validator tools should use these shared libraries for consistent behavior:

#### ontology_loader.py
High-performance loader with caching, filtering, and batch processing.

```python
from ontology_loader import OntologyLoader

loader = OntologyLoader(cache_size=128)
blocks = loader.load_directory(
    Path('mainKnowledgeGraph/pages/'),
    domain='ai',
    progress=True
)
stats = loader.get_statistics(blocks)
```

**Features**:
- LRU caching for repeated loads
- Domain and pattern filtering
- Batch processing with progress bars
- Statistics generation
- Error handling and recovery

#### ontology_block_parser.py
Parses canonical ontology block format from Logseq markdown files.

```python
from ontology_block_parser import OntologyBlockParser

parser = OntologyBlockParser()
block = parser.parse_file(Path('file.md'))
print(f"Term ID: {block.term_id}")
print(f"IRI: {block.get_full_iri()}")
```

**Features**:
- Supports all 6 domains (AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Tech)
- Extracts all properties (Tier 1-3)
- Parses OWL axioms and relationships
- Namespace resolution
- Validation and error detection

See **API Reference** for complete documentation: `/docs/API-REFERENCE.md`

## Quick Reference

### Most Common Tasks

**Convert to RDF Turtle**:
```bash
python converters/convert-to-turtle.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output ontology.ttl
```

**Export to CSV**:
```bash
python converters/convert-to-csv.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output ontology.csv
```

**Generate WebVOWL JSON**:
```bash
# First convert to Turtle
python converters/convert-to-turtle.py --input ../../mainKnowledgeGraph/pages/ --output ontology.ttl
# Then to WebVOWL
python converters/ttl_to_webvowl_json.py ontology.ttl webvowl.json
```

**Validate Ontology**:
```bash
cd ../../scripts/ontology-migration
node cli.js validate
```

## Related Documentation

Comprehensive documentation is available in the `/docs/` directory:

- **Tooling Overview** (`/docs/TOOLING-OVERVIEW.md`)
  - Complete map of all tools
  - Input/output formats
  - Dependencies between tools
  - Architecture diagrams

- **Tool Workflows** (`/docs/TOOL-WORKFLOWS.md`)
  - Step-by-step guides for common workflows
  - From markdown to Neo4j graph
  - From markdown to WebVOWL visualization
  - Validating and publishing ontology
  - Adding new concepts
  - Batch migration and standardization

- **User Guide** (`/docs/USER-GUIDE.md`)
  - For non-developers
  - Installation instructions
  - Basic workflows
  - Troubleshooting
  - FAQ

- **Developer Guide** (`/docs/DEVELOPER-GUIDE.md`)
  - How to add new tools
  - How to extend existing tools
  - Shared libraries usage
  - Testing requirements
  - Code standards

- **API Reference** (`/docs/API-REFERENCE.md`)
  - Python API (ontology_loader, ontology_block_parser)
  - Rust API (WebVOWL WASM)
  - JavaScript API (migration tools)
  - CLI reference

## License

See LICENSE file in the repository root.
