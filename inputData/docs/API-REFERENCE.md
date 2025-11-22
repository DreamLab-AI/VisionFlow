# API Reference - Complete API Documentation

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Purpose**: Complete API reference for Python, Rust, and JavaScript

---

## Table of Contents

1. [Python API](#python-api)
   - [OntologyBlockParser](#ontologyblockparser)
   - [OntologyLoader](#ontologyloader)
   - [OntologyBlock](#ontologyblock)
2. [Rust API](#rust-api)
   - [WebVowl](#webvowl-struct)
   - [Ontology Parser](#ontology-parser)
   - [Graph API](#graph-api)
   - [Layout API](#layout-api)
3. [JavaScript API](#javascript-api)
   - [WASM Bindings](#wasm-bindings)
   - [Migration Tools](#migration-tools)
4. [CLI Reference](#cli-reference)
   - [Converter Tools](#converter-tools-cli)
   - [Migration Tools](#migration-tools-cli)
   - [Validation Tools](#validation-tools-cli)

---

## Python API

### OntologyBlockParser

**Module**: `Ontology-Tools/tools/lib/ontology_block_parser.py`

Parse canonical ontology blocks from Logseq markdown files.

#### Class: `OntologyBlockParser`

```python
from ontology_block_parser import OntologyBlockParser

parser = OntologyBlockParser()
```

#### Methods

##### `parse_file(file_path: Path) -> Optional[OntologyBlock]`

Parse a single markdown file.

**Parameters**:
- `file_path` (Path): Path to markdown file

**Returns**:
- `OntologyBlock` if parsing succeeds
- `None` if file has no ontology block

**Example**:
```python
from pathlib import Path
from ontology_block_parser import OntologyBlockParser

parser = OntologyBlockParser()
block = parser.parse_file(Path('mainKnowledgeGraph/pages/AI-0001-ml.md'))

if block:
    print(f"Term ID: {block.term_id}")
    print(f"Label: {block.preferred_term}")
    print(f"IRI: {block.get_full_iri()}")
```

##### `parse_content(content: str, file_path: Path) -> Optional[OntologyBlock]`

Parse markdown content string.

**Parameters**:
- `content` (str): Markdown content
- `file_path` (Path): Path for reference (used in errors)

**Returns**:
- `OntologyBlock` or `None`

**Example**:
```python
content = """
- ontology:: true
- term-id:: AI-0001
- preferred-term:: MachineLearning
"""

block = parser.parse_content(content, Path('test.md'))
```

##### `extract_block(content: str) -> str`

Extract raw ontology block from markdown.

**Parameters**:
- `content` (str): Full markdown content

**Returns**:
- Raw ontology block text

**Example**:
```python
markdown = open('file.md').read()
block_text = parser.extract_block(markdown)
print(block_text)
```

---

### OntologyLoader

**Module**: `Ontology-Tools/tools/lib/ontology_loader.py`

High-performance loader with caching and batch processing.

#### Class: `OntologyLoader`

```python
from ontology_loader import OntologyLoader

loader = OntologyLoader(cache_size=128, strict_validation=False)
```

**Parameters**:
- `cache_size` (int): Maximum number of files to cache (default: 128)
- `strict_validation` (bool): If True, skip blocks with errors (default: False)

#### Methods

##### `load_file(path: Path, use_cache: bool = True) -> Optional[OntologyBlock]`

Load a single file.

**Parameters**:
- `path` (Path): File path
- `use_cache` (bool): Use LRU cache (default: True)

**Returns**:
- `OntologyBlock` or `None`

**Example**:
```python
from pathlib import Path
from ontology_loader import OntologyLoader

loader = OntologyLoader()
block = loader.load_file(Path('mainKnowledgeGraph/pages/AI-0001-ml.md'))

if block:
    print(f"Loaded: {block.preferred_term}")
```

##### `load_directory(directory: Path, domain: str = None, pattern: str = None, progress: bool = False) -> List[OntologyBlock]`

Load all ontology files from a directory.

**Parameters**:
- `directory` (Path): Directory path
- `domain` (str): Filter by domain (ai, bc, rb, mv, tc, dt)
- `pattern` (str): Regex pattern for term IDs
- `progress` (bool): Show progress bar

**Returns**:
- List of `OntologyBlock` objects

**Example**:
```python
from pathlib import Path
from ontology_loader import OntologyLoader

loader = OntologyLoader(cache_size=256)

# Load all blocks
blocks = loader.load_directory(
    Path('mainKnowledgeGraph/pages/'),
    progress=True
)
print(f"Loaded {len(blocks)} blocks")

# Load AI domain only
ai_blocks = loader.load_directory(
    Path('mainKnowledgeGraph/pages/'),
    domain='ai',
    progress=True
)

# Load specific pattern
pattern_blocks = loader.load_directory(
    Path('mainKnowledgeGraph/pages/'),
    pattern=r'^AI-0[0-9]{3}',
    progress=True
)
```

##### `load_batch(paths: List[Path], progress: bool = False) -> List[OntologyBlock]`

Load multiple specific files.

**Parameters**:
- `paths` (List[Path]): List of file paths
- `progress` (bool): Show progress bar

**Returns**:
- List of `OntologyBlock` objects

**Example**:
```python
paths = [
    Path('mainKnowledgeGraph/pages/AI-0001-ml.md'),
    Path('mainKnowledgeGraph/pages/AI-0002-dl.md'),
    Path('mainKnowledgeGraph/pages/AI-0003-nlp.md')
]

blocks = loader.load_batch(paths, progress=True)
```

##### `get_statistics(blocks: List[OntologyBlock]) -> LoaderStatistics`

Get statistics for loaded blocks.

**Parameters**:
- `blocks` (List[OntologyBlock]): List of blocks

**Returns**:
- `LoaderStatistics` object

**Example**:
```python
blocks = loader.load_directory(Path('mainKnowledgeGraph/pages/'))
stats = loader.get_statistics(blocks)

print(f"Total blocks: {stats.total_blocks}")
print(f"By domain: {stats.by_domain}")
print(f"By status: {stats.by_status}")
print(f"Load time: {stats.load_time} seconds")
print(f"Cache hit rate: {stats.cache_hits / (stats.cache_hits + stats.cache_misses)}")
```

##### `filter_blocks(blocks: List[OntologyBlock], **criteria) -> List[OntologyBlock]`

Filter blocks by criteria.

**Parameters**:
- `blocks` (List[OntologyBlock]): Blocks to filter
- `**criteria`: Filter criteria (domain, status, maturity_level, etc.)

**Returns**:
- Filtered list of blocks

**Example**:
```python
# Filter by domain and status
published_ai = loader.filter_blocks(
    blocks,
    domain='ai',
    status='published'
)

# Filter by maturity
mature_blocks = loader.filter_blocks(
    blocks,
    maturity_level='mature'
)
```

#### Class: `LoaderStatistics`

Statistics for loaded blocks.

**Attributes**:
- `total_blocks` (int): Total blocks loaded
- `by_domain` (Dict[str, int]): Count by domain
- `by_status` (Dict[str, int]): Count by status
- `by_maturity` (Dict[str, int]): Count by maturity level
- `validation_errors` (int): Number of validation errors
- `load_time` (float): Time taken to load (seconds)
- `cache_hits` (int): Number of cache hits
- `cache_misses` (int): Number of cache misses

**Methods**:

##### `to_dict() -> Dict[str, Any]`

Convert to dictionary for JSON serialization.

**Example**:
```python
import json

stats = loader.get_statistics(blocks)
stats_json = json.dumps(stats.to_dict(), indent=2)
print(stats_json)
```

---

### OntologyBlock

**Module**: `Ontology-Tools/tools/lib/ontology_block_parser.py`

Dataclass representing a parsed ontology block.

#### Attributes

##### Tier 1: Required Properties

```python
block.ontology: bool                    # Always True
block.term_id: str                      # e.g., "AI-0001"
block.preferred_term: str               # e.g., "MachineLearning"
block.source_domain: str                # e.g., "ai"
block.status: str                       # draft, published, deprecated
block.public_access: bool               # True/False
block.last_updated: str                 # ISO date: "2025-11-21"
block.definition: str                   # Concept definition
```

##### OWL Classification

```python
block.owl_class_uri: str                # Full URI
block.subclass_of: List[str]            # Parent classes
block.equivalent_class: List[str]       # Equivalent classes
block.disjoint_with: List[str]          # Disjoint classes
```

##### Tier 2: Detailed Semantics

```python
block.alternative_terms: List[str]      # Alternative labels
block.dc_subject: List[str]             # Subject categories
block.maturity_level: str               # emerging, developing, mature
block.see_also: List[str]               # Related concepts
block.version_info: str                 # Version (e.g., "v1.0.0")
```

##### Tier 3: Extended Properties

```python
block.extension_properties: Dict        # Domain-specific properties
block.use_cases: List[str]              # Use case examples
block.references: List[str]             # Citations
block.examples: List[str]               # Example instances
```

#### Methods

##### `get_full_iri() -> str`

Get the full IRI for this concept.

**Returns**:
- Full IRI string

**Example**:
```python
iri = block.get_full_iri()
# "http://narrativegoldmine.com/ai#MachineLearning"
```

##### `get_domain() -> str`

Get the domain code (ai, bc, rb, mv, tc, dt).

**Returns**:
- Domain code string

**Example**:
```python
domain = block.get_domain()  # "ai"
```

##### `get_namespace() -> str`

Get the namespace URI for this domain.

**Returns**:
- Namespace URI

**Example**:
```python
namespace = block.get_namespace()
# "http://narrativegoldmine.com/ai#"
```

##### `validate() -> Dict[str, Any]`

Validate the block for completeness and correctness.

**Returns**:
- Dictionary with validation results

**Example**:
```python
validation = block.validate()
print(f"Score: {validation['score']}")
print(f"Errors: {validation['errors']}")
print(f"Warnings: {validation['warnings']}")
```

##### `to_dict() -> Dict[str, Any]`

Convert to dictionary for JSON serialization.

**Returns**:
- Dictionary representation

**Example**:
```python
import json

block_dict = block.to_dict()
json_str = json.dumps(block_dict, indent=2)
```

##### `to_rdf_graph() -> rdflib.Graph`

Convert to RDF graph.

**Returns**:
- rdflib.Graph object

**Example**:
```python
from rdflib import Graph

rdf_graph = block.to_rdf_graph()
print(rdf_graph.serialize(format='turtle'))
```

---

## Rust API

### WebVowl Struct

**Module**: `publishing-tools/WasmVOWL/rust-wasm/src/lib.rs`

Main WASM interface for ontology visualization.

#### Struct: `WebVowl`

```rust
use webvowl_wasm::WebVowl;

let mut webvowl = WebVowl::new();
```

#### Methods

##### `new() -> Self`

Create a new WebVowl instance.

**Example**:
```rust
let webvowl = WebVowl::new();
```

##### `load_ontology(&mut self, json: String) -> Result<(), JsValue>`

Load ontology from JSON string.

**Parameters**:
- `json` (String): WebVOWL JSON format

**Returns**:
- `Result<(), JsValue>`

**Example**:
```rust
let json_data = std::fs::read_to_string("ontology.json")?;
webvowl.load_ontology(json_data)?;
```

##### `init_simulation(&mut self)`

Initialize force-directed layout simulation.

**Example**:
```rust
webvowl.init_simulation();
```

##### `tick(&mut self) -> bool`

Perform one simulation step.

**Returns**:
- `true` if simulation is still running, `false` if finished

**Example**:
```rust
while webvowl.tick() {
    // Update visualization
}
```

##### `run_simulation(&mut self, iterations: usize)`

Run simulation for N iterations.

**Parameters**:
- `iterations` (usize): Number of iterations

**Example**:
```rust
webvowl.run_simulation(100);
```

##### `is_finished(&self) -> bool`

Check if simulation has converged.

**Returns**:
- `true` if finished, `false` if still running

**Example**:
```rust
if webvowl.is_finished() {
    println!("Simulation complete");
}
```

##### `get_alpha(&self) -> f64`

Get current simulation energy (alpha).

**Returns**:
- Alpha value (0.0 = finished, 1.0 = maximum energy)

**Example**:
```rust
let energy = webvowl.get_alpha();
println!("Simulation energy: {:.2}", energy);
```

##### `set_center(&mut self, x: f64, y: f64)`

Set center position for layout.

**Parameters**:
- `x` (f64): X coordinate
- `y` (f64): Y coordinate

**Example**:
```rust
webvowl.set_center(400.0, 300.0);
```

##### `set_link_distance(&mut self, distance: f64)`

Set target distance between connected nodes.

**Parameters**:
- `distance` (f64): Link distance

**Example**:
```rust
webvowl.set_link_distance(250.0);
```

##### `set_charge_strength(&mut self, strength: f64)`

Set node repulsion strength.

**Parameters**:
- `strength` (f64): Charge strength (negative = repulsion)

**Example**:
```rust
webvowl.set_charge_strength(-2000.0);
```

##### `get_graph_data(&self) -> String`

Get current graph data with node positions.

**Returns**:
- JSON string with graph data

**Example**:
```rust
let graph_json = webvowl.get_graph_data();
```

##### `get_node_count(&self) -> usize`

Get number of nodes.

**Returns**:
- Node count

**Example**:
```rust
let count = webvowl.get_node_count();
println!("Nodes: {}", count);
```

##### `get_edge_count(&self) -> usize`

Get number of edges.

**Returns**:
- Edge count

**Example**:
```rust
let count = webvowl.get_edge_count();
println!("Edges: {}", count);
```

##### `get_statistics(&self) -> String`

Get graph statistics as JSON.

**Returns**:
- JSON string with statistics

**Example**:
```rust
let stats = webvowl.get_statistics();
println!("{}", stats);
```

##### `check_node_click(&self, x: f64, y: f64) -> Option<String>`

Check if a node was clicked at coordinates.

**Parameters**:
- `x` (f64): X coordinate
- `y` (f64): Y coordinate

**Returns**:
- `Some(node_id)` if clicked, `None` otherwise

**Example**:
```rust
if let Some(node_id) = webvowl.check_node_click(150.0, 200.0) {
    println!("Clicked node: {}", node_id);
}
```

##### `get_node_details(&self, node_id: &str) -> Result<String, JsValue>`

Get detailed information about a node.

**Parameters**:
- `node_id` (&str): Node ID

**Returns**:
- JSON string with node details

**Example**:
```rust
let details = webvowl.get_node_details("AI-0001")?;
println!("{}", details);
```

##### `get_metadata(&self) -> String`

Get ontology metadata (title, description, authors, etc.).

**Returns**:
- JSON string with metadata

**Example**:
```rust
let metadata = webvowl.get_metadata();
println!("{}", metadata);
```

---

### Ontology Parser

**Module**: `publishing-tools/WasmVOWL/rust-wasm/src/ontology/parser.rs`

#### Function: `parse_owl_json`

```rust
pub fn parse_owl_json(json: &str) -> Result<Ontology, ParseError>
```

Parse WebVOWL JSON format.

**Parameters**:
- `json` (&str): JSON string

**Returns**:
- `Result<Ontology, ParseError>`

**Example**:
```rust
use webvowl_wasm::ontology::parser::parse_owl_json;

let json_data = r#"{"classes": [...], "properties": [...]}"#;
let ontology = parse_owl_json(json_data)?;
```

---

### Graph API

**Module**: `publishing-tools/WasmVOWL/rust-wasm/src/graph/`

#### Struct: `GraphBuilder`

```rust
use webvowl_wasm::graph::GraphBuilder;

let builder = GraphBuilder::new();
```

#### Methods

##### `add_node(&mut self, id: String, label: String) -> &mut Self`

Add a node to the graph.

**Example**:
```rust
builder.add_node("AI-0001".to_string(), "MachineLearning".to_string());
```

##### `add_edge(&mut self, source: String, target: String, label: String) -> &mut Self`

Add an edge to the graph.

**Example**:
```rust
builder.add_edge("AI-0002".to_string(), "AI-0001".to_string(), "subClassOf".to_string());
```

##### `build(self) -> Graph`

Build the graph.

**Example**:
```rust
let graph = builder
    .add_node("AI-0001".to_string(), "MachineLearning".to_string())
    .add_edge("AI-0002".to_string(), "AI-0001".to_string(), "subClassOf".to_string())
    .build();
```

---

### Layout API

**Module**: `publishing-tools/WasmVOWL/rust-wasm/src/layout/`

#### Struct: `ForceSimulation`

```rust
use webvowl_wasm::layout::ForceSimulation;

let mut simulation = ForceSimulation::new(graph);
```

#### Methods

##### `new(graph: Graph) -> Self`

Create new simulation.

##### `with_link_distance(mut self, distance: f64) -> Self`

Set link distance (builder pattern).

**Example**:
```rust
let simulation = ForceSimulation::new(graph)
    .with_link_distance(250.0)
    .with_charge_strength(-2000.0);
```

##### `tick(&mut self) -> bool`

Perform one simulation step.

**Returns**:
- `true` if still running

**Example**:
```rust
while simulation.tick() {
    // Continue...
}
```

##### `get_alpha(&self) -> f64`

Get simulation energy.

---

## JavaScript API

### WASM Bindings

**Module**: Published as `@dreamlab-ai/webvowl-wasm` on npm

#### Installation

```bash
npm install @dreamlab-ai/webvowl-wasm
```

#### Usage

```javascript
import init, { WebVowl } from '@dreamlab-ai/webvowl-wasm';

async function main() {
    // Initialize WASM module
    await init();

    // Create instance
    const webvowl = new WebVowl();

    // Load ontology
    const ontologyJson = await fetch('ontology.json').then(r => r.json());
    webvowl.loadOntology(JSON.stringify(ontologyJson));

    // Configure
    webvowl.setCenter(400, 300);
    webvowl.setLinkDistance(250);
    webvowl.setChargeStrength(-2000);

    // Run simulation
    webvowl.initSimulation();
    webvowl.runSimulation(100);

    // Get results
    const graphData = JSON.parse(webvowl.getGraphData());
    const stats = JSON.parse(webvowl.getStatistics());

    console.log('Nodes:', stats.node_count);
    console.log('Edges:', stats.edge_count);
}

main();
```

#### Animation Loop

```javascript
function animate() {
    if (!webvowl.isFinished()) {
        webvowl.tick();

        const graphData = JSON.parse(webvowl.getGraphData());
        updateVisualization(graphData);

        requestAnimationFrame(animate);
    } else {
        console.log('Simulation complete');
    }
}

webvowl.initSimulation();
animate();
```

#### Interactive Features

```javascript
// Node click detection
canvas.addEventListener('click', (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const nodeId = webvowl.checkNodeClick(x, y);

    if (nodeId) {
        const details = JSON.parse(webvowl.getNodeDetails(nodeId));
        console.log('Clicked node:', details);
        showNodeDetailsPanel(details);
    }
});

// Get metadata
const metadata = JSON.parse(webvowl.getMetadata());
console.log('Ontology:', metadata.title);
console.log('Version:', metadata.version);
console.log('Authors:', metadata.authors);
```

---

### Migration Tools

**Module**: `scripts/ontology-migration/`

#### Scanner

```javascript
const scanner = require('./scanner');

// Scan directory
const inventory = scanner.scanDirectory('/path/to/pages/');

console.log('Total files:', inventory.totalFiles);
console.log('With ontology:', inventory.filesWithOntology);
console.log('By domain:', inventory.domainDistribution);
```

#### Parser

```javascript
const parser = require('./parser');

// Parse file
const result = parser.parseFile('/path/to/file.md');

if (result.success) {
    console.log('Term ID:', result.block.termId);
    console.log('Label:', result.block.preferredTerm);
    console.log('Domain:', result.block.sourceDomain);
} else {
    console.error('Errors:', result.errors);
}

// Parse content
const content = fs.readFileSync('file.md', 'utf8');
const result2 = parser.parseContent(content, 'file.md');
```

#### Generator

```javascript
const generator = require('./generator');

// Generate canonical block
const parsedBlock = parser.parseFile('file.md').block;
const canonicalBlock = generator.generateCanonicalBlock(parsedBlock, {
    fixNamespaces: true,
    normalizeCasing: true
});

console.log(canonicalBlock);
```

#### Validator

```javascript
const validator = require('./validator');

// Validate file
const result = validator.validateFile('/path/to/file.md');

console.log('Score:', result.score);
console.log('Errors:', result.errors);
console.log('Warnings:', result.warnings);
console.log('Passed:', result.passed);
```

---

## CLI Reference

### Converter Tools CLI

All converters in `Ontology-Tools/tools/converters/` follow this pattern:

#### Common Options

```bash
--input DIR         Input directory or file (required)
--output FILE       Output file path (required)
--domain DOMAIN     Filter by domain (ai, bc, rb, mv, tc, dt)
--verbose           Enable verbose logging
--help              Show help message
```

#### convert-to-turtle.py

```bash
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl \
  [--domain ai] \
  [--verbose]
```

#### convert-to-jsonld.py

```bash
python Ontology-Tools/tools/converters/convert-to-jsonld.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.jsonld \
  [--domain ai] \
  [--verbose]
```

#### convert-to-csv.py

```bash
python Ontology-Tools/tools/converters/convert-to-csv.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.csv \
  [--domain ai] \
  [--verbose]
```

#### convert-to-cypher.py

```bash
python Ontology-Tools/tools/converters/convert-to-cypher.py \
  --input mainKnowledgeGraph/pages/ \
  --output neo4j-import.cypher \
  [--domain ai] \
  [--verbose]
```

#### convert-to-sql.py

```bash
python Ontology-Tools/tools/converters/convert-to-sql.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.sql \
  [--domain ai] \
  [--verbose]
```

#### convert-to-skos.py

```bash
python Ontology-Tools/tools/converters/convert-to-skos.py \
  --input mainKnowledgeGraph/pages/ \
  --output skos-vocabulary.ttl \
  [--domain ai] \
  [--verbose]
```

#### ttl_to_webvowl_json.py

```bash
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  INPUT.ttl \
  OUTPUT.json
```

#### generate_search_index.py

```bash
python Ontology-Tools/tools/converters/generate_search_index.py \
  --input mainKnowledgeGraph/pages/ \
  --output search-index.json \
  [--domain ai] \
  [--verbose]
```

#### generate_page_api.py

```bash
python Ontology-Tools/tools/converters/generate_page_api.py \
  --input mainKnowledgeGraph/pages/ \
  --output api/pages/ \
  [--domain ai] \
  [--verbose]
```

---

### Migration Tools CLI

**Location**: `scripts/ontology-migration/cli.js`

#### scan

```bash
node scripts/ontology-migration/cli.js scan
```

Scan all markdown files and generate inventory report.

#### preview

```bash
node scripts/ontology-migration/cli.js preview [NUMBER]

# Examples:
node cli.js preview 10
node cli.js preview 50
```

Preview transformations without making changes.

#### process

```bash
node scripts/ontology-migration/cli.js process [OPTIONS]

# Options:
#   --live         Perform actual updates (default: dry-run)
#   --batch=N      Set batch size (default: 100)
#   --validate     Run validation after processing
#   --no-backup    Disable backup creation (NOT recommended)

# Examples:
node cli.js process                    # Dry-run
node cli.js process --live             # Live update
node cli.js process --live --batch=50  # Live with batch size 50
```

#### test

```bash
node scripts/ontology-migration/cli.js test FILE_PATH

# Example:
node cli.js test mainKnowledgeGraph/pages/AI-0001-ml.md
```

Test transformation on a single file.

#### validate

```bash
node scripts/ontology-migration/cli.js validate
```

Validate all ontology blocks.

#### domain

```bash
node scripts/ontology-migration/cli.js domain DOMAIN [OPTIONS]

# Domains: ai, blockchain, robotics, metaverse

# Examples:
node cli.js domain robotics --live
node cli.js domain ai --live --validate
```

Process specific domain only.

#### stats

```bash
node scripts/ontology-migration/cli.js stats
```

Show statistics and reports.

#### rollback

```bash
node scripts/ontology-migration/cli.js rollback
```

Restore files from backups.

---

### Validation Tools CLI

#### validate_owl2.py

```bash
python scripts/validate_owl2.py ONTOLOGY_FILE

# Example:
python scripts/validate_owl2.py ontology.ttl
```

Validate OWL2 compliance.

---

## Related Documentation

- **Tooling Overview**: `/home/user/logseq/docs/TOOLING-OVERVIEW.md`
- **Workflows**: `/home/user/logseq/docs/TOOL-WORKFLOWS.md`
- **Developer Guide**: `/home/user/logseq/docs/DEVELOPER-GUIDE.md`
- **User Guide**: `/home/user/logseq/docs/USER-GUIDE.md`

---

**Maintainer**: Claude Code Agent
**Last Updated**: 2025-11-21
**Version**: 1.0.0
