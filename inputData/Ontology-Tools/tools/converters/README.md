# Ontology Block Converters

This directory contains Python converters that transform OntologyBlock markdown files into standard semantic web formats.

## Converters

### 1. convert-to-jsonld.py
Converts OntologyBlocks to JSON-LD format with proper @context and full IRIs.

### 2. convert-to-skos.py
Converts OntologyBlocks to SKOS (Simple Knowledge Organization System) format in Turtle syntax.

## Quick Start

### JSON-LD Conversion

```bash
# Convert all ontology files
cd /home/user/logseq/Ontology-Tools/tools/converters
python3 convert-to-jsonld.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output ontology.jsonld

# Convert with validation
python3 convert-to-jsonld.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output ontology.jsonld \
  --validate
```

### SKOS Conversion

```bash
# Convert all ontology files
python3 convert-to-skos.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output ontology.ttl

# Convert with validation
python3 convert-to-skos.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output ontology.ttl \
  --validate
```

## Supported Domains

Both converters support all 6 domains:
- Artificial Intelligence (ai)
- Blockchain (bc)
- Robotics (rb)
- Metaverse (mv)
- Telecollaboration (tc)
- Disruptive Technologies (dt)

## CLI Arguments

- `--input PATH`: Input path (file, directory, or glob pattern) **[required]**
- `--output PATH`: Output file path **[required]**
- `--validate`: Validate blocks before conversion (shows warnings)
- `--pretty`: Pretty-print JSON output (JSON-LD only, default: True)

## Dependencies

- Python 3.7+
- Standard library only (no external packages)
- Parser library: `../lib/ontology_block_parser.py` (included)

## Documentation

See `/home/user/logseq/docs/converter-updates-summary.md` for complete documentation.
