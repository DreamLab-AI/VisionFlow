# Tool Workflows - Common Use Cases

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Purpose**: Step-by-step guides for common workflows combining multiple tools

---

## Table of Contents

1. [From Markdown to Neo4j Graph](#1-from-markdown-to-neo4j-graph)
2. [From Markdown to WebVOWL Visualization](#2-from-markdown-to-webvowl-visualization)
3. [Validating and Publishing Ontology](#3-validating-and-publishing-ontology)
4. [Adding New Concepts](#4-adding-new-concepts)
5. [Batch Migration and Standardization](#5-batch-migration-and-standardization)
6. [Creating a Search-Enabled Web Interface](#6-creating-a-search-enabled-web-interface)
7. [Exporting to Multiple Formats](#7-exporting-to-multiple-formats)
8. [Quality Assurance Pipeline](#8-quality-assurance-pipeline)
9. [Development and Testing Workflow](#9-development-and-testing-workflow)
10. [Production Deployment Workflow](#10-production-deployment-workflow)

---

## 1. From Markdown to Neo4j Graph

**Goal**: Import ontology data from Logseq markdown files into Neo4j graph database

### Prerequisites
- Python 3.8+
- Neo4j database running
- Ontology markdown files in `mainKnowledgeGraph/pages/`

### Steps

#### Step 1: Convert Markdown to Cypher
```bash
cd /home/user/logseq

python Ontology-Tools/tools/converters/convert-to-cypher.py \
  --input mainKnowledgeGraph/pages/ \
  --output neo4j-import.cypher \
  --domain all
```

**Output**: `neo4j-import.cypher` file with CREATE statements

#### Step 2: Review Generated Cypher (Optional)
```bash
head -n 50 neo4j-import.cypher
```

Expected output:
```cypher
// Create nodes
CREATE (n1:Class {term_id: 'AI-0001', label: 'MachineLearning', ...})
CREATE (n2:Class {term_id: 'AI-0002', label: 'DeepLearning', ...})

// Create relationships
MATCH (n1:Class {term_id: 'AI-0002'}), (n2:Class {term_id: 'AI-0001'})
CREATE (n1)-[:subClassOf]->(n2)
```

#### Step 3: Import into Neo4j

**Option A: Using Neo4j Browser**
1. Open Neo4j Browser: http://localhost:7474
2. Copy contents of `neo4j-import.cypher`
3. Paste and execute in query window

**Option B: Using cypher-shell**
```bash
cat neo4j-import.cypher | cypher-shell -u neo4j -p password
```

**Option C: Batch import (for large datasets)**
```bash
# Split into batches of 1000 statements
split -l 1000 neo4j-import.cypher cypher-batch-

# Import each batch
for file in cypher-batch-*; do
  cat "$file" | cypher-shell -u neo4j -p password
done
```

#### Step 4: Verify Import
```cypher
// Count nodes
MATCH (n:Class) RETURN count(n)

// Count relationships
MATCH ()-[r:subClassOf]->() RETURN count(r)

// Query specific domain
MATCH (n:Class) WHERE n.domain = 'ai' RETURN n LIMIT 10
```

#### Step 5: Create Indexes (Performance)
```cypher
CREATE INDEX class_term_id FOR (n:Class) ON (n.term_id);
CREATE INDEX class_label FOR (n:Class) ON (n.label);
CREATE INDEX class_domain FOR (n:Class) ON (n.domain);
```

### Expected Results
- ✅ All ontology nodes imported as `:Class` nodes
- ✅ Relationships preserved (subClassOf, relatedTo, etc.)
- ✅ Queryable by term_id, label, domain
- ✅ Full-text search on definitions

### Troubleshooting

**Problem**: "Syntax error near line X"
- **Solution**: Check that input markdown files are properly formatted
- Run validator first: `node scripts/ontology-migration/cli.js validate`

**Problem**: Duplicate nodes
- **Solution**: Clear database before re-importing:
  ```cypher
  MATCH (n) DETACH DELETE n
  ```

---

## 2. From Markdown to WebVOWL Visualization

**Goal**: Visualize ontology as interactive 3D graph on the web

### Prerequisites
- Node.js v18+
- Rust and wasm-pack installed
- Python 3.8+

### Steps

#### Step 1: Convert Markdown to Turtle
```bash
cd /home/user/logseq

python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl \
  --domain all
```

**Output**: `ontology.ttl` in RDF Turtle format

#### Step 2: Convert Turtle to WebVOWL JSON
```bash
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  ontology.ttl \
  webvowl-data.json
```

**Output**: `webvowl-data.json` in WebVOWL specification format

#### Step 3: (Optional) Create Header-Only View
For large ontologies, create a simplified top-level view:
```bash
python Ontology-Tools/tools/converters/webvowl_header_only_converter.py \
  --input mainKnowledgeGraph/pages/ \
  --output webvowl-header.json \
  --max-depth 2
```

#### Step 4: Build WASM Module
```bash
cd publishing-tools/WasmVOWL/rust-wasm

# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Build WASM
wasm-pack build --target web --release
```

**Output**: `pkg/` directory with WASM binary and JS bindings

#### Step 5: Copy Data to Frontend
```bash
cd /home/user/logseq

# Copy JSON data
cp webvowl-data.json publishing-tools/WasmVOWL/modern/public/data/ontology.json

# Optional: Copy header-only view
cp webvowl-header.json publishing-tools/WasmVOWL/modern/public/data/ontology-header.json
```

#### Step 6: Install Frontend Dependencies
```bash
cd publishing-tools/WasmVOWL/modern
npm install
```

#### Step 7: Run Development Server
```bash
npm run dev
```

**Output**: Development server at http://localhost:5173

#### Step 8: View Visualization
1. Open http://localhost:5173 in browser
2. Ontology loads automatically from `public/data/ontology.json`
3. Interact with 3D graph:
   - **Mouse drag**: Rotate camera
   - **Mouse wheel**: Zoom in/out
   - **Click node**: View details
   - **Ctrl+Click**: Multi-select nodes

#### Step 9: Build for Production
```bash
npm run build
```

**Output**: `dist/` directory with optimized static files

#### Step 10: Deploy
```bash
# Option A: Deploy to GitHub Pages (automatic via Actions)
git add dist/
git commit -m "Update visualization"
git push

# Option B: Deploy to custom server
rsync -avz dist/ user@server:/var/www/ontology/

# Option C: Serve locally
npm run preview  # http://localhost:4173
```

### Expected Results
- ✅ Interactive 3D graph visualization
- ✅ 15-30 FPS with 1,700+ nodes
- ✅ Node click detection and details panel
- ✅ Hardware-accelerated rendering (WebGL)
- ✅ Live at https://narrativegoldmine.com

### Performance Tips
- For >5,000 nodes: Use header-only view
- Adjust physics parameters in `rust-wasm/src/layout/simulation.rs`
- Enable level-of-detail (LOD) for very large graphs

---

## 3. Validating and Publishing Ontology

**Goal**: Ensure quality and consistency before publishing

### Prerequisites
- Python 3.8+ with rdflib, owlrl, pyshacl
- Node.js for migration tools

### Steps

#### Step 1: Validate Markdown Format
```bash
cd /home/user/logseq/scripts/ontology-migration

node cli.js validate
```

**Output**: `validation-report.json` with scores and issues

**Expected**: Average score > 90

#### Step 2: Fix Issues (if needed)
```bash
# Preview fixes
node cli.js preview 10

# Run dry-run
node cli.js process

# Apply fixes
node cli.js process --live --validate
```

#### Step 3: Convert to Turtle
```bash
cd /home/user/logseq

python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl
```

#### Step 4: Validate OWL2 Syntax
```bash
python scripts/validate_owl2.py ontology.ttl
```

**Expected output**:
```
✓ Valid OWL2 syntax
✓ All namespaces declared
✓ No dangling references
✓ Classes: 1450
✓ Properties: 380
```

#### Step 5: Generate All Export Formats
```bash
# JSON-LD
python Ontology-Tools/tools/converters/convert-to-jsonld.py \
  --input ontology.ttl \
  --output ontology.jsonld

# CSV for analysis
python Ontology-Tools/tools/converters/convert-to-csv.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.csv

# SKOS vocabulary
python Ontology-Tools/tools/converters/convert-to-skos.py \
  --input mainKnowledgeGraph/pages/ \
  --output skos-vocabulary.ttl
```

#### Step 6: Generate Web Assets
```bash
# Search index
python Ontology-Tools/tools/converters/generate_search_index.py \
  --input mainKnowledgeGraph/pages/ \
  --output publishing-tools/WasmVOWL/modern/public/search-index.json

# Page API
python Ontology-Tools/tools/converters/generate_page_api.py \
  --input mainKnowledgeGraph/pages/ \
  --output publishing-tools/WasmVOWL/modern/public/api/pages/

# WebVOWL visualization
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  ontology.ttl \
  publishing-tools/WasmVOWL/modern/public/data/ontology.json
```

#### Step 7: Build and Test Frontend
```bash
cd publishing-tools/WasmVOWL/rust-wasm
wasm-pack build --target web --release

cd ../modern
npm install
npm run build
npm run preview
```

#### Step 8: Final Validation
```bash
# Test locally
open http://localhost:4173

# Verify:
# - Visualization loads
# - Search works
# - All domains visible
# - Node details show correctly
```

#### Step 9: Commit and Push
```bash
cd /home/user/logseq

git add ontology.ttl ontology.jsonld ontology.csv
git add publishing-tools/WasmVOWL/modern/public/
git commit -m "chore: publish validated ontology v1.0.0"
git push
```

#### Step 10: Deploy (Auto via GitHub Actions)
Push to main branch triggers automatic deployment to GitHub Pages.

### Validation Checklist
- [ ] All markdown files score > 85 in validation
- [ ] OWL2 syntax passes validation
- [ ] No orphaned classes (all have parents)
- [ ] All namespaces properly declared
- [ ] Search index generated and tested
- [ ] WebVOWL visualization renders correctly
- [ ] All export formats generated successfully

---

## 4. Adding New Concepts

**Goal**: Add new ontology concepts following best practices

### Prerequisites
- Understanding of the domain ontology
- Text editor or Logseq
- Validation tools installed

### Steps

#### Step 1: Choose Domain and ID
Determine the domain and next available ID:
```bash
# List existing IDs in domain
ls mainKnowledgeGraph/pages/AI-* | tail -5

# Example output:
# AI-0403-term.md
# AI-0404-term.md
# AI-0405-term.md

# Next ID: AI-0406
```

#### Step 2: Create Markdown File
```bash
cd /home/user/logseq/mainKnowledgeGraph/pages

cat > AI-0406-transformer-architecture.md << 'EOF'
- ontology:: true
- term-id:: AI-0406
- preferred-term:: TransformerArchitecture
- source-domain:: ai
- status:: published
- public-access:: true
- last-updated:: 2025-11-21

# Tier 1: Definition
- definition::
  - A neural network architecture based on self-attention mechanisms, introduced in "Attention is All You Need" (2017)

# Tier 1: OWL Classification
- owl:Class::
  - uri: http://narrativegoldmine.com/ai#TransformerArchitecture
- rdfs:subClassOf::
  - [[AI-0207 EncoderDecoderArchitecture]]
- owl:equivalentClass::
  - (none)
- owl:disjointWith::
  - [[AI-0150 ConvolutionalNeuralNetwork]]

# Tier 2: Detailed Semantics
- skos:altLabel::
  - Transformer Model
  - Attention-Based Architecture
- dc:subject::
  - Deep Learning
  - Natural Language Processing
- maturity-level:: mature

# Tier 2: Additional Classification
- rdfs:seeAlso::
  - [[AI-0210 AttentionMechanism]]
  - [[AI-0350 BERT]]
- owl:versionInfo:: v1.0.0

# Tier 3: Extended Properties
- algorithm-type:: architecture
- computational-complexity:: O(n²d)
- use-cases::
  - Machine Translation
  - Language Modeling
  - Question Answering

# Tier 3: Documentation
- references::
  - Vaswani et al. "Attention is All You Need" (2017)
- examples::
  - BERT, GPT, T5, BART
EOF
```

#### Step 3: Validate the New File
```bash
cd /home/user/logseq/scripts/ontology-migration

node cli.js test mainKnowledgeGraph/pages/AI-0406-transformer-architecture.md
```

**Expected output**:
```
✓ Parsing successful
✓ All required properties present
✓ OWL syntax valid
✓ Namespace correct
✓ Parent class exists
Validation Score: 98/100
```

#### Step 4: Check for Issues
```bash
# Common issues:
# - Typos in property names
# - Missing parent class
# - Incorrect namespace
# - Missing required fields

# Fix issues and re-validate
```

#### Step 5: Update Parent Class
Add reference in parent class file:
```bash
# Edit AI-0207-encoder-decoder-architecture.md
# Add to rdfs:seeAlso:
  - [[AI-0406 TransformerArchitecture]]
```

#### Step 6: Convert to Turtle (Test)
```bash
cd /home/user/logseq

python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/AI-0406-transformer-architecture.md \
  --output test-output.ttl
```

#### Step 7: Validate OWL2
```bash
python scripts/validate_owl2.py test-output.ttl
```

**Expected**: No errors

#### Step 8: Commit Changes
```bash
git add mainKnowledgeGraph/pages/AI-0406-transformer-architecture.md
git add mainKnowledgeGraph/pages/AI-0207-encoder-decoder-architecture.md
git commit -m "feat(ai): add TransformerArchitecture concept (AI-0406)"
git push
```

#### Step 9: Regenerate Exports (Optional)
```bash
# Full pipeline - see Workflow #3 "Validating and Publishing"
```

### Best Practices
- ✅ Use CamelCase for class names
- ✅ Start definitions with capital letter
- ✅ Always specify parent class (avoid orphans)
- ✅ Include at least 2 alternative terms
- ✅ Add dc:subject for categorization
- ✅ Reference related concepts in rdfs:seeAlso
- ✅ Include maturity-level
- ✅ Provide real-world examples
- ✅ Cite authoritative references

---

## 5. Batch Migration and Standardization

**Goal**: Migrate 1,700+ ontology files to canonical format

### Prerequisites
- Node.js v14+
- Backup of current state
- At least 30-40 minutes for full pipeline

### Steps

#### Step 1: Create Safety Backup
```bash
cd /home/user/logseq

# Git commit current state
git add -A
git commit -m "checkpoint: before batch migration"

# Optional: Create tarball backup
tar -czf mainKnowledgeGraph-backup-$(date +%Y%m%d).tar.gz mainKnowledgeGraph/
```

#### Step 2: Scan All Files
```bash
cd scripts/ontology-migration

node cli.js scan
```

**Output**: `docs/ontology-migration/reports/file-inventory.json`

**Review**:
```bash
cat ../../docs/ontology-migration/reports/file-inventory.json | jq '.summary'
```

#### Step 3: Preview Transformations
```bash
# Preview first 20 files
node cli.js preview 20
```

**Review output carefully**:
- Check namespace fixes (mv: → rb:)
- Verify class name conversions (lowercase → CamelCase)
- Ensure parent classes preserved

#### Step 4: Test on Single File
```bash
# Test on a robotics file (namespace fix)
node cli.js test mainKnowledgeGraph/pages/rb-0010-aerial-robot.md

# Test on an AI file (already correct)
node cli.js test mainKnowledgeGraph/pages/AI-0001-machine-learning.md
```

#### Step 5: Validate Current State
```bash
node cli.js validate
```

**Baseline scores**: Note for comparison after migration

#### Step 6: Dry-Run Full Pipeline
```bash
node cli.js process
```

**Expected time**: ~5-10 minutes for dry-run

**Output**: Simulated changes, no actual modifications

#### Step 7: Pilot Run (Robotics Domain Only)
```bash
# Process only robotics (most critical fixes)
node cli.js domain robotics --live --validate
```

**Expected**:
- ~250 robotics files updated
- Namespace mv: → rb: fixed
- Backups created in `docs/ontology-migration/backups/`

#### Step 8: Verify Pilot Results
```bash
# Check a few files manually
git diff mainKnowledgeGraph/pages/rb-0010-aerial-robot.md

# Check validation scores
node cli.js stats
```

#### Step 9: Full Migration (All Domains)
```bash
node cli.js process --live --validate --batch=100
```

**Expected time**: ~30-40 minutes

**Monitor progress**:
```bash
# In another terminal
watch -n 5 cat ../../docs/ontology-migration/reports/checkpoint.json
```

#### Step 10: Final Validation
```bash
node cli.js validate
```

**Compare scores**: Should improve by 10-20 points

#### Step 11: Review Reports
```bash
# Summary statistics
node cli.js stats

# Detailed reports
cat ../../docs/ontology-migration/reports/final-report.json | jq
```

#### Step 12: Test Exports
```bash
cd /home/user/logseq

# Test conversion
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output test-migrated.ttl

# Validate
python scripts/validate_owl2.py test-migrated.ttl
```

#### Step 13: Commit Changes
```bash
git add -A
git commit -m "refactor: standardize all ontology blocks to canonical format

- Fixed 250 namespace errors (mv: → rb:)
- Converted 1,420 class names to CamelCase
- Normalized status and maturity values
- Removed 180 duplicate sections
- Average validation score: 87 → 94

Processed: 1,450 files
Backups: docs/ontology-migration/backups/
Reports: docs/ontology-migration/reports/"
git push
```

### Rollback (If Needed)
```bash
cd scripts/ontology-migration

# Restore from backups
node cli.js rollback

# OR restore from git
cd /home/user/logseq
git reset --hard HEAD~1
```

### Success Criteria
- ✅ All files processed without errors
- ✅ Average validation score > 90
- ✅ Namespace errors fixed
- ✅ OWL2 validation passes
- ✅ Backups created for all files
- ✅ Reports generated successfully

---

## 6. Creating a Search-Enabled Web Interface

**Goal**: Build searchable ontology browser with visualization

### Prerequisites
- Completed Workflow #2 (WebVOWL setup)
- Generated search index

### Steps

#### Step 1: Generate Search Index
```bash
cd /home/user/logseq

python Ontology-Tools/tools/converters/generate_search_index.py \
  --input mainKnowledgeGraph/pages/ \
  --output publishing-tools/WasmVOWL/modern/public/search-index.json
```

**Output**: `search-index.json` with all terms

#### Step 2: Generate Page API
```bash
python Ontology-Tools/tools/converters/generate_page_api.py \
  --input mainKnowledgeGraph/pages/ \
  --output publishing-tools/WasmVOWL/modern/public/api/pages/
```

**Output**: Individual JSON files per term

#### Step 3: Verify JSON Files
```bash
# Check search index
cat publishing-tools/WasmVOWL/modern/public/search-index.json | jq '.terms | length'
# Expected: 1450

# Check page API
ls publishing-tools/WasmVOWL/modern/public/api/pages/ | wc -l
# Expected: 1450
```

#### Step 4: Build Frontend
```bash
cd publishing-tools/WasmVOWL/modern
npm run build
```

#### Step 5: Test Search Locally
```bash
npm run preview
# Open http://localhost:4173
```

**Test search features**:
1. Search by term (e.g., "machine learning")
2. Filter by domain (AI, Blockchain, Robotics, Metaverse)
3. Click search result → view details
4. Click "View in Graph" → navigate to node

#### Step 6: Deploy
```bash
cd /home/user/logseq

git add publishing-tools/WasmVOWL/modern/public/
git commit -m "feat: add search index and page API"
git push
```

**Auto-deploys to**: https://narrativegoldmine.com

### Search Features
- ✅ Full-text search across all terms
- ✅ Filter by domain
- ✅ Fuzzy matching
- ✅ Keyword highlighting
- ✅ Deep linking to graph nodes
- ✅ Client-side (no backend required)

---

## 7. Exporting to Multiple Formats

**Goal**: Generate all export formats for distribution

### Steps

```bash
cd /home/user/logseq

# RDF Turtle
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output exports/ontology.ttl

# JSON-LD
python Ontology-Tools/tools/converters/convert-to-jsonld.py \
  --input mainKnowledgeGraph/pages/ \
  --output exports/ontology.jsonld

# CSV (for Excel/spreadsheets)
python Ontology-Tools/tools/converters/convert-to-csv.py \
  --input mainKnowledgeGraph/pages/ \
  --output exports/ontology.csv

# SQL (relational databases)
python Ontology-Tools/tools/converters/convert-to-sql.py \
  --input mainKnowledgeGraph/pages/ \
  --output exports/ontology.sql

# Neo4j Cypher (graph databases)
python Ontology-Tools/tools/converters/convert-to-cypher.py \
  --input mainKnowledgeGraph/pages/ \
  --output exports/neo4j-import.cypher

# SKOS vocabulary
python Ontology-Tools/tools/converters/convert-to-skos.py \
  --input mainKnowledgeGraph/pages/ \
  --output exports/skos-vocabulary.ttl

# WebVOWL JSON
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  exports/ontology.ttl \
  exports/webvowl.json

# Compress for distribution
cd exports
tar -czf ontology-exports-$(date +%Y%m%d).tar.gz *.ttl *.jsonld *.csv *.sql *.cypher *.json
```

### Distribution
```bash
# Upload to GitHub releases
gh release create v1.0.0 exports/ontology-exports-*.tar.gz \
  --title "Ontology Export Package v1.0.0" \
  --notes "Complete ontology in multiple formats"
```

---

## 8. Quality Assurance Pipeline

**Goal**: Comprehensive validation before release

### Steps

```bash
cd /home/user/logseq

# 1. Validate markdown format
cd scripts/ontology-migration
node cli.js validate
cd ../..

# 2. Convert to Turtle
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output qa/ontology.ttl

# 3. Validate OWL2 syntax
python scripts/validate_owl2.py qa/ontology.ttl > qa/owl2-validation.txt

# 4. Check for orphaned classes
python scripts/add_ai_orphan_parents.py qa/ontology.ttl --check-only > qa/orphans-report.txt

# 5. Check for missing comments
python scripts/add_missing_comments.py qa/ontology.ttl --check-only > qa/comments-report.txt

# 6. Generate statistics
python Ontology-Tools/tools/converters/convert-to-csv.py \
  --input mainKnowledgeGraph/pages/ \
  --output qa/ontology-full.csv

# 7. Analyze with spreadsheet
# Open qa/ontology-full.csv in Excel/LibreOffice
# Create pivot tables for domain/status/maturity analysis

# 8. Test visualization
cd publishing-tools/WasmVOWL/rust-wasm
wasm-pack build --target web --release
cd ../modern
npm run build
npm run preview
# Manual test in browser

# 9. Review all reports
cat qa/*.txt

# 10. Generate final QA report
echo "QA Report - $(date)" > qa/QA-REPORT.md
echo "===================" >> qa/QA-REPORT.md
echo "" >> qa/QA-REPORT.md
echo "## Validation Summary" >> qa/QA-REPORT.md
echo "- Markdown validation: PASS" >> qa/QA-REPORT.md
echo "- OWL2 validation: PASS" >> qa/QA-REPORT.md
echo "- Orphan check: PASS" >> qa/QA-REPORT.md
echo "- Visualization test: PASS" >> qa/QA-REPORT.md
```

---

## 9. Development and Testing Workflow

**Goal**: Develop and test new converter tool

### Example: Creating a new converter

```bash
cd /home/user/logseq/Ontology-Tools/tools/converters

# 1. Create new converter file
cat > convert-to-graphml.py << 'EOF'
#!/usr/bin/env python3
"""
Convert ontology to GraphML format
"""

from pathlib import Path
from ontology_loader import OntologyLoader
import xml.etree.ElementTree as ET

def convert_to_graphml(input_path, output_path):
    """Convert ontology to GraphML"""
    loader = OntologyLoader()
    blocks = loader.load_directory(input_path)

    # Create GraphML structure
    graphml = ET.Element('graphml')
    graph = ET.SubElement(graphml, 'graph')

    # Add nodes
    for block in blocks:
        node = ET.SubElement(graph, 'node', id=block.term_id)
        # Add data...

    # Write output
    tree = ET.ElementTree(graphml)
    tree.write(output_path)

if __name__ == '__main__':
    # CLI implementation...
    pass
EOF

# 2. Test on sample data
python convert-to-graphml.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output test-outputs/test.graphml

# 3. Validate output
# Load test.graphml in graph tool (e.g., yEd, Gephi)

# 4. Add to documentation
# Update Ontology-Tools/tools/README.md

# 5. Commit
git add convert-to-graphml.py
git commit -m "feat: add GraphML converter"
```

---

## 10. Production Deployment Workflow

**Goal**: Deploy updates to production

### Steps

```bash
cd /home/user/logseq

# 1. Quality checks (see Workflow #8)

# 2. Update version numbers
# Edit package.json, Cargo.toml, etc.

# 3. Generate all exports (see Workflow #7)

# 4. Build WASM
cd publishing-tools/WasmVOWL/rust-wasm
wasm-pack build --target web --release

# 5. Publish WASM to npm (if updated)
cd pkg
npm publish --access public

# 6. Update frontend
cd ../../modern
npm install  # Get latest WASM
npm run build

# 7. Test production build
npm run preview
# Manual testing

# 8. Commit and tag
cd /home/user/logseq
git add -A
git commit -m "release: v1.1.0"
git tag v1.1.0
git push
git push --tags

# 9. GitHub Actions auto-deploys to production

# 10. Verify deployment
curl -I https://narrativegoldmine.com
# Check 200 status

# 11. Create GitHub release
gh release create v1.1.0 \
  --title "Release v1.1.0" \
  --notes "See CHANGELOG.md" \
  exports/ontology-exports-*.tar.gz
```

---

## Related Documentation

- **Tooling Overview**: `/home/user/logseq/docs/TOOLING-OVERVIEW.md`
- **User Guide**: `/home/user/logseq/docs/USER-GUIDE.md`
- **Developer Guide**: `/home/user/logseq/docs/DEVELOPER-GUIDE.md`
- **API Reference**: `/home/user/logseq/docs/API-REFERENCE.md`

---

**Maintainer**: Claude Code Agent
**Last Updated**: 2025-11-21
**Version**: 1.0.0
