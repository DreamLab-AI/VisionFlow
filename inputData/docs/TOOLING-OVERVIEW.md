# Tooling Overview - Logseq Ontology Ecosystem

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Purpose**: Complete map of all tools, their functions, inputs/outputs, and dependencies

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tool Categories](#tool-categories)
3. [Tool Catalog](#tool-catalog)
4. [Input/Output Formats](#inputoutput-formats)
5. [Dependencies Between Tools](#dependencies-between-tools)
6. [Quick Reference](#quick-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LOGSEQ ONTOLOGY TOOLING ECOSYSTEM                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐        ┌───────────────┐        ┌───────────────┐
│   PARSERS &   │        │  CONVERTERS   │        │  VALIDATORS   │
│    LOADERS    │        │               │        │               │
├───────────────┤        ├───────────────┤        ├───────────────┤
│ • Parser Lib  │───────▶│ • Turtle      │◀──────│ • Syntax      │
│ • Loader Lib  │        │ • JSON-LD     │       │ • Semantic    │
│ • Scanner     │        │ • CSV         │       │ • Quality     │
└───────┬───────┘        │ • SQL         │       └───────────────┘
        │                │ • Cypher      │                │
        │                │ • SKOS        │                │
        │                │ • WebVOWL     │                │
        │                └───────┬───────┘                │
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  GENERATORS &  │
                        │   PROCESSORS   │
                        ├────────────────┤
                        │ • Unified Gen  │
                        │ • Enhancers    │
                        │ • Aggregators  │
                        │ • Migration    │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  PUBLISHING    │
                        │     TOOLS      │
                        ├────────────────┤
                        │ • WasmVOWL     │
                        │ • Search Index │
                        │ • Page API     │
                        └────────────────┘
```

---

## Tool Categories

### 1. **Core Libraries** (Shared Functionality)
Foundation libraries used by all other tools for consistent behavior.

### 2. **Parsers & Scanners** (Input Processing)
Extract and analyze ontology data from markdown files.

### 3. **Converters** (Format Transformation)
Transform ontology data into various output formats.

### 4. **Validators** (Quality Assurance)
Ensure ontology correctness, consistency, and quality.

### 5. **Generators** (Content Creation)
Create and enhance ontology files and structures.

### 6. **Migration Tools** (Batch Processing)
Standardize and migrate large sets of ontology files.

### 7. **Publishing Tools** (Visualization & API)
Publish ontologies for web consumption and visualization.

### 8. **Utility Scripts** (Maintenance)
Support scripts for specific maintenance tasks.

---

## Tool Catalog

### 1. Core Libraries

#### 1.1 Ontology Block Parser (`ontology_block_parser.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/lib/ontology_block_parser.py`

**Purpose**: Parse canonical ontology block format from Logseq markdown files

**Features**:
- Supports all 6 domains (AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Tech)
- Extracts all properties (Tier 1-3)
- Parses OWL axioms and relationships
- Namespace resolution
- Validation and error detection

**Inputs**:
- Markdown file path or content string

**Outputs**:
- `OntologyBlock` object with structured data
- Validation errors and warnings

**Used By**: ALL converter tools, validators, generators, migration tools

**Example**:
```python
from ontology_block_parser import OntologyBlockParser

parser = OntologyBlockParser()
block = parser.parse_file('mainKnowledgeGraph/pages/AI-0001-term.md')
print(f"Term ID: {block.term_id}")
print(f"IRI: {block.get_full_iri()}")
```

---

#### 1.2 Ontology Loader (`ontology_loader.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/lib/ontology_loader.py`

**Purpose**: High-performance loader with caching, filtering, and batch processing

**Features**:
- LRU caching for repeated loads
- Domain and pattern filtering
- Batch processing with progress reporting
- Statistics generation
- Error handling and recovery

**Inputs**:
- File path or directory path
- Optional filters (domain, pattern)

**Outputs**:
- Single `OntologyBlock` or list of blocks
- `LoaderStatistics` object

**Used By**: Converters, validators, generators, migration tools

**Example**:
```python
from ontology_loader import OntologyLoader

loader = OntologyLoader(cache_size=100)

# Load directory with filtering
blocks = loader.load_directory(
    Path('mainKnowledgeGraph/pages/'),
    domain='ai',
    progress=True
)

# Get statistics
stats = loader.get_statistics(blocks)
print(f"Loaded {stats.total_blocks} blocks")
```

---

### 2. Parsers & Scanners

#### 2.1 Ontology Migration Scanner (`scanner.js`)
**Location**: `/home/user/logseq/scripts/ontology-migration/scanner.js`

**Purpose**: Scan all markdown files and generate inventory report

**Features**:
- Identifies files with ontology blocks
- Classifies by pattern (pattern1-6)
- Detects domain
- Identifies issues (namespace errors, naming issues)

**Inputs**:
- Source directory path

**Outputs**:
- `file-inventory.json` report

**CLI**:
```bash
node scripts/ontology-migration/cli.js scan
```

**Output Format**:
```json
{
  "totalFiles": 1709,
  "filesWithOntology": 1450,
  "patternDistribution": {...},
  "domainDistribution": {...},
  "issues": {...}
}
```

---

#### 2.2 Ontology Migration Parser (`parser.js`)
**Location**: `/home/user/logseq/scripts/ontology-migration/parser.js`

**Purpose**: Extract and parse ontology blocks from files

**Features**:
- Extracts ontology block
- Parses all properties and relationships
- Identifies OWL axioms
- Detects namespace usage
- Analyzes issues

**Inputs**:
- File path
- File content

**Outputs**:
- Parsed block object
- Issues array

**Used By**: Migration batch processor, updater

---

### 3. Converters

#### 3.1 Turtle Converter (`convert-to-turtle.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/convert-to-turtle.py`

**Purpose**: Convert ontology blocks to RDF Turtle format

**Inputs**:
- Single markdown file OR directory of files
- Optional: domain filter

**Outputs**:
- `.ttl` file(s) in RDF Turtle format

**CLI**:
```bash
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/AI-0001-term.md \
  --output ontology.ttl
```

**Dependencies**:
- `ontology_loader.py`
- `ontology_block_parser.py`
- `rdflib`

---

#### 3.2 JSON-LD Converter (`convert-to-jsonld.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/convert-to-jsonld.py`

**Purpose**: Convert ontology blocks to JSON-LD format

**Inputs**:
- Single markdown file OR directory of files

**Outputs**:
- `.jsonld` file(s)

**CLI**:
```bash
python Ontology-Tools/tools/converters/convert-to-jsonld.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.jsonld
```

**Dependencies**:
- `ontology_loader.py`
- `rdflib`

---

#### 3.3 CSV Converter (`convert-to-csv.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/convert-to-csv.py`

**Purpose**: Export ontology data to CSV for spreadsheet analysis

**Inputs**:
- Directory of markdown files

**Outputs**:
- `.csv` file with all properties

**CLI**:
```bash
python Ontology-Tools/tools/converters/convert-to-csv.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.csv
```

**Output Columns**:
- term_id, preferred_term, definition, source_domain, status, maturity, etc.

**Dependencies**:
- `ontology_loader.py`
- `csv` module

---

#### 3.4 Neo4j Cypher Converter (`convert-to-cypher.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/convert-to-cypher.py`

**Purpose**: Generate Neo4j Cypher queries for graph database import

**Inputs**:
- Directory of markdown files

**Outputs**:
- `.cypher` file with CREATE statements

**CLI**:
```bash
python Ontology-Tools/tools/converters/convert-to-cypher.py \
  --input mainKnowledgeGraph/pages/ \
  --output import.cypher
```

**Output Format**:
```cypher
CREATE (n:Class {term_id: 'AI-0001', label: 'MachineLearning', ...})
CREATE (n1)-[:subClassOf]->(n2)
```

**Dependencies**:
- `ontology_loader.py`

---

#### 3.5 SQL Converter (`convert-to-sql.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/convert-to-sql.py`

**Purpose**: Export to SQL schema and INSERT statements

**Inputs**:
- Directory of markdown files

**Outputs**:
- `.sql` file with schema and data

**CLI**:
```bash
python Ontology-Tools/tools/converters/convert-to-sql.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.sql
```

**Output Structure**:
- CREATE TABLE statements
- INSERT statements for all data
- Foreign key relationships

**Dependencies**:
- `ontology_loader.py`

---

#### 3.6 SKOS Converter (`convert-to-skos.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/convert-to-skos.py`

**Purpose**: Transform to SKOS vocabulary format

**Inputs**:
- Directory of markdown files

**Outputs**:
- `.ttl` file in SKOS format

**CLI**:
```bash
python Ontology-Tools/tools/converters/convert-to-skos.py \
  --input mainKnowledgeGraph/pages/ \
  --output skos-vocabulary.ttl
```

**SKOS Mappings**:
- Classes → skos:Concept
- preferredTerm → skos:prefLabel
- alternativeTerms → skos:altLabel
- definition → skos:definition
- subClassOf → skos:broader

**Dependencies**:
- `ontology_loader.py`
- `rdflib`

---

#### 3.7 WebVOWL JSON Converter (`ttl_to_webvowl_json.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/ttl_to_webvowl_json.py`

**Purpose**: Convert Turtle to WebVOWL JSON format for visualization

**Inputs**:
- `.ttl` file

**Outputs**:
- WebVOWL-compatible JSON

**CLI**:
```bash
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  ontology.ttl \
  webvowl.json
```

**Output Format**: WebVOWL specification JSON (see VOWL-SPEC.md)

**Dependencies**:
- `rdflib`
- `owlrl`

---

#### 3.8 WebVOWL Header-Only Converter (`webvowl_header_only_converter.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/webvowl_header_only_converter.py`

**Purpose**: Convert to WebVOWL format showing only top-level classes

**Inputs**:
- Directory of markdown files

**Outputs**:
- WebVOWL JSON with filtered hierarchy

**CLI**:
```bash
python Ontology-Tools/tools/converters/webvowl_header_only_converter.py \
  --input mainKnowledgeGraph/pages/ \
  --output header-only.json \
  --max-depth 2
```

**Features**:
- Filters hierarchy by depth
- Reduces visual clutter
- Maintains relationships at top level

**Dependencies**:
- `ontology_loader.py`

---

#### 3.9 Search Index Generator (`generate_search_index.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/generate_search_index.py`

**Purpose**: Generate search index for web interface

**Inputs**:
- Directory of markdown files

**Outputs**:
- `search-index.json` for client-side search

**CLI**:
```bash
python Ontology-Tools/tools/converters/generate_search_index.py \
  --input mainKnowledgeGraph/pages/ \
  --output public/search-index.json
```

**Output Format**:
```json
{
  "terms": [
    {
      "id": "AI-0001",
      "label": "MachineLearning",
      "definition": "...",
      "domain": "ai",
      "keywords": ["learning", "ML", "artificial intelligence"]
    }
  ]
}
```

**Dependencies**:
- `ontology_loader.py`

---

#### 3.10 Page API Generator (`generate_page_api.py`)
**Location**: `/home/user/logseq/Ontology-Tools/tools/converters/generate_page_api.py`

**Purpose**: Generate REST API-style JSON for each ontology page

**Inputs**:
- Directory of markdown files

**Outputs**:
- Individual JSON files per term

**CLI**:
```bash
python Ontology-Tools/tools/converters/generate_page_api.py \
  --input mainKnowledgeGraph/pages/ \
  --output api/pages/
```

**Output Structure**:
```
api/pages/
  ai-0001.json
  ai-0002.json
  bc-0001.json
  ...
```

**Dependencies**:
- `ontology_loader.py`

---

### 4. Validators

#### 4.1 OWL2 Validator (`validate_owl2.py`)
**Location**: `/home/user/logseq/scripts/validate_owl2.py`

**Purpose**: Validate OWL2 compliance and syntax

**Inputs**:
- `.ttl` or `.owl` file

**Outputs**:
- Validation report with errors and warnings

**CLI**:
```bash
python scripts/validate_owl2.py ontology.ttl
```

**Checks**:
- OWL2 syntax correctness
- Namespace declarations
- Class and property definitions
- Axiom validity

**Dependencies**:
- `owlrl`
- `rdflib`

---

### 5. Generators

#### 5.1 Ontology Migration Generator (`generator.js`)
**Location**: `/home/user/logseq/scripts/ontology-migration/generator.js`

**Purpose**: Generate canonical ontology blocks from parsed data

**Features**:
- Reads canonical schema
- Fixes namespace issues (mv: → rb:)
- Converts class names to CamelCase
- Normalizes status/maturity values
- Applies domain templates

**Inputs**:
- Parsed block object

**Outputs**:
- Canonical ontology block text

**Used By**: Batch processor, updater

---

### 6. Migration Tools

#### 6.1 Ontology Migration CLI (`cli.js`)
**Location**: `/home/user/logseq/scripts/ontology-migration/cli.js`

**Purpose**: Command-line interface for batch ontology migration

**Commands**:
- `scan` - Generate file inventory
- `preview` - Preview transformations
- `process` - Run batch migration
- `validate` - Validate results
- `test` - Test single file
- `domain` - Process specific domain
- `pattern` - Process specific pattern
- `rollback` - Restore from backups
- `stats` - Show statistics

**Full Documentation**: See `/home/user/logseq/scripts/ontology-migration/README.md`

---

#### 6.2 Batch Processor (`batch-process.js`)
**Location**: `/home/user/logseq/scripts/ontology-migration/batch-process.js`

**Purpose**: Orchestrate full pipeline with checkpointing

**Features**:
- Processes files in batches (default: 100)
- Creates backups automatically
- Handles errors gracefully
- Progress checkpointing
- Resumable operations

**Used By**: CLI processor command

---

#### 6.3 File Updater (`updater.js`)
**Location**: `/home/user/logseq/scripts/ontology-migration/updater.js`

**Purpose**: Update markdown files with canonical blocks

**Features**:
- Creates timestamped backups
- Replaces ontology block
- Preserves content below block
- Validates after update (optional)

**Used By**: Batch processor

---

#### 6.4 Format Validator (`validator.js`)
**Location**: `/home/user/logseq/scripts/ontology-migration/validator.js`

**Purpose**: Validate canonical format compliance

**Checks**:
- Required properties present
- OWL syntax correctness
- Namespace correctness
- Format consistency

**Outputs**:
- Validation score (0-100)
- Issues and warnings

---

### 7. Publishing Tools

#### 7.1 WasmVOWL (Rust/WASM)
**Location**: `/home/user/logseq/publishing-tools/WasmVOWL/rust-wasm/`

**Purpose**: High-performance ontology visualization using Rust/WebAssembly

**Components**:
- **Rust WASM Module**: Physics engine and graph layout
  - `ontology/`: OWL parsing
  - `graph/`: Graph structures
  - `layout/`: Force-directed layout (Barnes-Hut algorithm)
  - `bindings/`: JavaScript API

**Performance**: 10-100x faster than JavaScript implementation

**Build**:
```bash
cd publishing-tools/WasmVOWL/rust-wasm
wasm-pack build --target web --release
```

**NPM Package**: `@dreamlab-ai/webvowl-wasm` v0.3.3

**Full Documentation**:
- `/home/user/logseq/publishing-tools/WasmVOWL/README.md`
- `/home/user/logseq/publishing-tools/WasmVOWL/rust-wasm/README.md`

---

#### 7.2 WasmVOWL React Frontend
**Location**: `/home/user/logseq/publishing-tools/WasmVOWL/modern/`

**Purpose**: Modern React Three Fiber UI for 3D ontology visualization

**Features**:
- 3D graph visualization
- Interactive node selection
- Metadata display panel
- Hardware-accelerated rendering (WebGL)
- 15-30 FPS with 1,700+ nodes

**Tech Stack**:
- React Three Fiber
- TypeScript
- Zustand (state management)
- Vite (build tool)

**Development**:
```bash
cd publishing-tools/WasmVOWL/modern
npm install
npm run dev  # http://localhost:5173
```

**Deployment**:
- Live at: https://narrativegoldmine.com
- Auto-deploys from main branch via GitHub Actions

---

### 8. Utility Scripts

#### 8.1 Convert to OWL2 (`convert_to_owl2.py`)
**Location**: `/home/user/logseq/scripts/convert_to_owl2.py`

**Purpose**: Convert ontology files to OWL2 format

**CLI**:
```bash
python scripts/convert_to_owl2.py input.ttl output.owl
```

---

#### 8.2 Add Missing Comments (`add_missing_comments.py`)
**Location**: `/home/user/logseq/scripts/add_missing_comments.py`

**Purpose**: Add rdfs:comment to classes missing documentation

**CLI**:
```bash
python scripts/add_missing_comments.py ontology.ttl
```

---

#### 8.3 Add AI Orphan Parents (`add_ai_orphan_parents.py`)
**Location**: `/home/user/logseq/scripts/add_ai_orphan_parents.py`

**Purpose**: Fix orphaned classes by adding parent relationships

**CLI**:
```bash
python scripts/add_ai_orphan_parents.py ontology.ttl
```

---

## Input/Output Formats

### Input Formats

| Format | Extension | Tools That Accept |
|--------|-----------|-------------------|
| Logseq Markdown | `.md` | Parser, Loader, All Converters, Migration Tools |
| RDF Turtle | `.ttl` | WebVOWL Converter, Validators |
| OWL/XML | `.owl` | Validators, Converters |

### Output Formats

| Format | Extension | Use Case | Generated By |
|--------|-----------|----------|--------------|
| RDF Turtle | `.ttl` | Standard RDF format | Turtle Converter |
| JSON-LD | `.jsonld` | Linked data format | JSON-LD Converter |
| CSV | `.csv` | Spreadsheet analysis | CSV Converter |
| SQL | `.sql` | Relational databases | SQL Converter |
| Cypher | `.cypher` | Neo4j graph DB | Cypher Converter |
| SKOS | `.ttl` | Vocabulary format | SKOS Converter |
| WebVOWL JSON | `.json` | Visualization | WebVOWL Converters |
| Search Index | `.json` | Web search | Search Index Generator |
| Page API | `.json` | REST-style API | Page API Generator |

---

## Dependencies Between Tools

### Dependency Graph

```
ontology_block_parser.py (CORE)
         │
         ├─► ontology_loader.py (CORE)
         │            │
         │            ├─► convert-to-turtle.py
         │            ├─► convert-to-jsonld.py
         │            ├─► convert-to-csv.py
         │            ├─► convert-to-cypher.py
         │            ├─► convert-to-sql.py
         │            ├─► convert-to-skos.py
         │            ├─► webvowl_header_only_converter.py
         │            ├─► generate_search_index.py
         │            └─► generate_page_api.py
         │
         └─► scanner.js (Migration)
                  │
                  └─► parser.js (Migration)
                           │
                           └─► generator.js (Migration)
                                    │
                                    └─► updater.js (Migration)
                                             │
                                             └─► batch-process.js (Migration)

convert-to-turtle.py ──► ttl_to_webvowl_json.py ──► WasmVOWL

webvowl_header_only_converter.py ──► WasmVOWL
```

### External Dependencies

#### Python Tools
- `rdflib` - RDF manipulation
- `owlrl` - OWL reasoning
- `pyshacl` - SHACL validation

#### JavaScript Tools
- `node.js` v14+ - Runtime
- No external npm dependencies (migration tools)

#### Rust Tools
- `wasm-pack` - WASM compilation
- `serde` - Serialization
- `wasm-bindgen` - JS bindings

---

## Quick Reference

### Most Common Workflows

1. **Markdown → Turtle**:
   ```bash
   python Ontology-Tools/tools/converters/convert-to-turtle.py \
     --input mainKnowledgeGraph/pages/ \
     --output ontology.ttl
   ```

2. **Markdown → WebVOWL Visualization**:
   ```bash
   # Step 1: Convert to Turtle
   python Ontology-Tools/tools/converters/convert-to-turtle.py \
     --input mainKnowledgeGraph/pages/ \
     --output ontology.ttl

   # Step 2: Convert to WebVOWL JSON
   python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
     ontology.ttl webvowl.json

   # Step 3: Load in WasmVOWL
   # Copy webvowl.json to publishing-tools/WasmVOWL/modern/public/data/
   ```

3. **Batch Migration**:
   ```bash
   cd scripts/ontology-migration
   node cli.js scan
   node cli.js preview 10
   node cli.js process --live --validate
   ```

4. **Validate Ontology**:
   ```bash
   python scripts/validate_owl2.py ontology.ttl
   ```

5. **Generate Search Index**:
   ```bash
   python Ontology-Tools/tools/converters/generate_search_index.py \
     --input mainKnowledgeGraph/pages/ \
     --output public/search-index.json
   ```

### Tool Selection Guide

**Need to...**
- **Parse markdown files?** → Use `ontology_loader.py`
- **Convert to RDF?** → Use `convert-to-turtle.py`
- **Visualize ontology?** → Use WasmVOWL
- **Export to database?** → Use `convert-to-sql.py` or `convert-to-cypher.py`
- **Migrate/standardize files?** → Use migration CLI
- **Validate format?** → Use `validator.js` or `validate_owl2.py`
- **Generate API?** → Use `generate_page_api.py`
- **Create search?** → Use `generate_search_index.py`

---

## Related Documentation

- **User Guide**: `/home/user/logseq/docs/USER-GUIDE.md`
- **Developer Guide**: `/home/user/logseq/docs/DEVELOPER-GUIDE.md`
- **Workflows**: `/home/user/logseq/docs/TOOL-WORKFLOWS.md`
- **API Reference**: `/home/user/logseq/docs/API-REFERENCE.md`
- **Migration Tools**: `/home/user/logseq/scripts/ontology-migration/README.md`
- **WasmVOWL**: `/home/user/logseq/publishing-tools/WasmVOWL/README.md`
- **Ontology Tools**: `/home/user/logseq/Ontology-Tools/tools/README.md`

---

**Maintainer**: Claude Code Agent
**Repository**: DreamLab-AI/knowledgeGraph
**License**: See LICENSE in repository root
