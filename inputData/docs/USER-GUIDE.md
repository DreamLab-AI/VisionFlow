# User Guide - Logseq Ontology Tools

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Audience**: Non-developers, ontology authors, researchers

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Basic Workflows](#basic-workflows)
5. [Working with Ontology Files](#working-with-ontology-files)
6. [Converting to Different Formats](#converting-to-different-formats)
7. [Viewing Your Ontology](#viewing-your-ontology)
8. [Validating Your Work](#validating-your-work)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Introduction

### What is This?

This toolset helps you work with ontology data stored in Logseq markdown files. It provides:

- **Format converters**: Export to RDF, CSV, JSON, SQL, and more
- **Validators**: Check your ontology for errors
- **Visualization**: View your ontology as an interactive 3D graph
- **Migration tools**: Standardize large sets of files
- **Search tools**: Generate searchable web interfaces

### Who Is This For?

- **Ontology Authors**: Creating and maintaining ontology concepts
- **Researchers**: Analyzing ontology data
- **Data Scientists**: Exporting ontology for analysis
- **Knowledge Engineers**: Publishing ontologies on the web

### No Coding Required

All tools can be used from the command line with simple commands. No programming knowledge needed!

---

## Installation

### Prerequisites

#### For All Users

1. **Python 3.8 or higher**
   ```bash
   # Check if Python is installed
   python3 --version

   # If not installed, download from: https://www.python.org/downloads/
   ```

2. **Required Python Packages**
   ```bash
   # Install required packages
   pip install rdflib owlrl pyshacl
   ```

#### For Visualization (Optional)

3. **Node.js 18 or higher**
   ```bash
   # Check if Node.js is installed
   node --version

   # If not installed, download from: https://nodejs.org/
   ```

4. **Rust (for building visualization)**
   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Install wasm-pack
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

### Verify Installation

```bash
# Check Python
python3 --version

# Check pip packages
pip list | grep rdflib
pip list | grep owlrl

# Check Node.js (if using visualization)
node --version

# Check Rust (if building visualization)
rustc --version
wasm-pack --version
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/DreamLab-AI/knowledgeGraph.git
cd knowledgeGraph
```

### 2. Your First Conversion

Convert your ontology to RDF Turtle format:

```bash
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output my-ontology.ttl
```

**Output**: A file called `my-ontology.ttl` containing your ontology in RDF format

### 3. View the Results

```bash
# On Linux/Mac
head -n 50 my-ontology.ttl

# On Windows
type my-ontology.ttl | more
```

---

## Basic Workflows

### Workflow 1: Export to Excel/CSV

**Goal**: Open your ontology in Excel or Google Sheets

```bash
# Convert to CSV
python Ontology-Tools/tools/converters/convert-to-csv.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.csv

# Open in Excel or Google Sheets
```

**What You Get**: A spreadsheet with columns for:
- Term ID
- Preferred Term
- Definition
- Domain
- Status
- Parent Classes
- And more...

### Workflow 2: View as Interactive Graph

**Goal**: See your ontology as a visual network

**Step 1: Generate Visualization Data**
```bash
# First, convert to Turtle
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl

# Then, convert to WebVOWL format
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  ontology.ttl \
  webvowl.json
```

**Step 2: View in Browser**

Visit our live visualization: **https://narrativegoldmine.com**

Or run locally:
```bash
cd publishing-tools/WasmVOWL/modern
npm install
npm run dev
# Open http://localhost:5173
```

### Workflow 3: Check for Errors

**Goal**: Validate your ontology files

```bash
# Validate markdown files
cd scripts/ontology-migration
node cli.js validate

# Validate OWL syntax
cd ../..
python scripts/validate_owl2.py my-ontology.ttl
```

**What You Get**: A report showing:
- Validation scores (0-100)
- Errors that must be fixed
- Warnings that should be fixed
- Statistics about your ontology

---

## Working with Ontology Files

### Understanding File Structure

Each ontology concept is stored in a markdown file like this:

```markdown
- ontology:: true
- term-id:: AI-0001
- preferred-term:: MachineLearning
- source-domain:: ai
- status:: published
- public-access:: true
- last-updated:: 2025-11-21

# Tier 1: Definition
- definition::
  - The study of computer algorithms that improve through experience

# Tier 1: OWL Classification
- owl:Class::
  - uri: http://narrativegoldmine.com/ai#MachineLearning
- rdfs:subClassOf::
  - [[AI-0000 ArtificialIntelligence]]

# Tier 2: Detailed Semantics
- skos:altLabel::
  - ML
  - Machine Learning
- dc:subject::
  - Artificial Intelligence
  - Computer Science
```

### Creating a New Concept

**Step 1: Choose a Term ID**

Each domain has its own ID range:
- **AI-xxxx**: Artificial Intelligence
- **BC-xxxx**: Blockchain
- **RB-xxxx**: Robotics
- **MV-xxxx**: Metaverse
- **TC-xxxx**: Telecollaboration
- **DT-xxxx**: Disruptive Technologies

Find the next available ID:
```bash
ls mainKnowledgeGraph/pages/AI-* | tail -5
# Shows: AI-0403, AI-0404, AI-0405
# Next ID: AI-0406
```

**Step 2: Create the File**

Create a new file: `mainKnowledgeGraph/pages/AI-0406-my-concept.md`

Use this template:
```markdown
- ontology:: true
- term-id:: AI-0406
- preferred-term:: MyConcept
- source-domain:: ai
- status:: draft
- public-access:: true
- last-updated:: 2025-11-21

# Tier 1: Definition
- definition::
  - A clear, concise definition of your concept

# Tier 1: OWL Classification
- owl:Class::
  - uri: http://narrativegoldmine.com/ai#MyConcept
- rdfs:subClassOf::
  - [[AI-0000 ParentConcept]]

# Tier 2: Detailed Semantics
- skos:altLabel::
  - Alternative Name 1
  - Alternative Name 2
- dc:subject::
  - Subject Category 1
  - Subject Category 2
- maturity-level:: emerging
```

**Step 3: Validate Your File**

```bash
cd scripts/ontology-migration
node cli.js test mainKnowledgeGraph/pages/AI-0406-my-concept.md
```

**Step 4: Save and Commit**

```bash
git add mainKnowledgeGraph/pages/AI-0406-my-concept.md
git commit -m "Add new concept: MyConcept (AI-0406)"
```

### Editing Existing Concepts

1. Open the file in any text editor
2. Make your changes
3. Validate the file (see Step 3 above)
4. Save and commit

**Common Changes**:
- Update definition
- Add alternative terms
- Add related concepts
- Change status (draft → published)
- Update maturity level

---

## Converting to Different Formats

### RDF Turtle (.ttl)

**Use Case**: Standard semantic web format

```bash
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl
```

**Options**:
```bash
# Convert only AI domain
python ... --domain ai --output ai-only.ttl

# Verbose output
python ... --verbose
```

### JSON-LD (.jsonld)

**Use Case**: JSON format for linked data

```bash
python Ontology-Tools/tools/converters/convert-to-jsonld.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.jsonld
```

### CSV (.csv)

**Use Case**: Excel, Google Sheets, data analysis

```bash
python Ontology-Tools/tools/converters/convert-to-csv.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.csv
```

**Opening the CSV**:
- Double-click to open in Excel
- Import into Google Sheets
- Open in LibreOffice Calc

### SQL (.sql)

**Use Case**: Import into MySQL, PostgreSQL, SQLite

```bash
python Ontology-Tools/tools/converters/convert-to-sql.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.sql
```

**Importing**:
```bash
# MySQL
mysql -u user -p database < ontology.sql

# PostgreSQL
psql -U user -d database -f ontology.sql

# SQLite
sqlite3 ontology.db < ontology.sql
```

### Neo4j Cypher (.cypher)

**Use Case**: Import into Neo4j graph database

```bash
python Ontology-Tools/tools/converters/convert-to-cypher.py \
  --input mainKnowledgeGraph/pages/ \
  --output neo4j-import.cypher
```

**Importing**:
```bash
# Using cypher-shell
cat neo4j-import.cypher | cypher-shell -u neo4j -p password
```

### SKOS Vocabulary (.ttl)

**Use Case**: Taxonomy and vocabulary management

```bash
python Ontology-Tools/tools/converters/convert-to-skos.py \
  --input mainKnowledgeGraph/pages/ \
  --output skos-vocabulary.ttl
```

### WebVOWL JSON (.json)

**Use Case**: Web visualization

```bash
# First convert to Turtle
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl

# Then convert to WebVOWL
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  ontology.ttl \
  webvowl.json
```

---

## Viewing Your Ontology

### Option 1: Online Viewer (Easiest)

Visit **https://narrativegoldmine.com** to view the full ontology visualization.

**Features**:
- 3D interactive graph
- Click nodes to see details
- Search for concepts
- Filter by domain
- Hardware-accelerated rendering

### Option 2: Local Viewer

**Setup (One Time)**:
```bash
cd publishing-tools/WasmVOWL/rust-wasm
wasm-pack build --target web --release

cd ../modern
npm install
```

**Run**:
```bash
cd publishing-tools/WasmVOWL/modern
npm run dev
```

**Open**: http://localhost:5173

### Using the Visualization

**Navigation**:
- **Drag**: Rotate the view
- **Scroll**: Zoom in/out
- **Click node**: View details
- **Ctrl+Click**: Select multiple nodes

**Search**:
- Type in search box
- Filter by domain
- Click result to jump to node

**Details Panel**:
- Shows all properties
- Lists relationships
- Copy IRI or label
- Navigate to related nodes

---

## Validating Your Work

### Quick Validation

Check a single file:
```bash
cd scripts/ontology-migration
node cli.js test mainKnowledgeGraph/pages/AI-0406-my-concept.md
```

**Output**:
```
✓ Parsing successful
✓ All required properties present
✓ OWL syntax valid
✓ Namespace correct
✓ Parent class exists
Validation Score: 95/100
```

### Full Validation

Validate all files:
```bash
cd scripts/ontology-migration
node cli.js validate
```

**Output**: `docs/ontology-migration/reports/validation-report.json`

**View Report**:
```bash
node cli.js stats
```

### OWL2 Validation

Validate semantic correctness:
```bash
# First convert to Turtle
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl

# Validate
python scripts/validate_owl2.py ontology.ttl
```

### Understanding Validation Scores

| Score | Meaning | Action |
|-------|---------|--------|
| 95-100 | Excellent | Ready to publish |
| 85-94 | Good | Minor improvements recommended |
| 70-84 | Fair | Address warnings |
| <70 | Needs Work | Fix errors before publishing |

### Common Issues and Fixes

**Issue**: Missing parent class
```markdown
# Bad
- rdfs:subClassOf::
  - (none)

# Good
- rdfs:subClassOf::
  - [[AI-0000 ParentConcept]]
```

**Issue**: Short definition
```markdown
# Bad
- definition::
  - ML algorithm

# Good
- definition::
  - A machine learning algorithm that learns from data to make predictions
```

**Issue**: Missing alternative terms
```markdown
# Bad
- skos:altLabel::
  - (none)

# Good
- skos:altLabel::
  - ML
  - Machine Learning
```

---

## Troubleshooting

### Problem: "Python not found"

**Solution**:
```bash
# Install Python from https://www.python.org/downloads/
# Or use python3 instead of python:
python3 --version
```

### Problem: "Module not found: rdflib"

**Solution**:
```bash
pip install rdflib owlrl pyshacl
```

### Problem: "No ontology blocks found"

**Solution**:
- Check that you're pointing to the correct directory
- Verify files have `ontology:: true` in them
- Check file permissions

### Problem: "Permission denied"

**Solution**:
```bash
# On Linux/Mac, make scripts executable
chmod +x Ontology-Tools/tools/converters/*.py
chmod +x scripts/ontology-migration/*.js
```

### Problem: "Validation score low"

**Solution**:
1. Run validation: `node cli.js validate`
2. Check report: `node cli.js stats`
3. Fix reported errors
4. Re-validate

### Problem: "Visualization not loading"

**Solution**:
```bash
# Rebuild WASM
cd publishing-tools/WasmVOWL/rust-wasm
wasm-pack build --target web --release

# Reinstall frontend
cd ../modern
npm install
npm run dev
```

### Problem: "Out of memory"

**Solution**:
```bash
# For large ontologies, process in batches
python convert-to-turtle.py --domain ai --output ai.ttl
python convert-to-turtle.py --domain bc --output bc.ttl
# etc.
```

### Getting Help

1. Check documentation in `/docs/`
2. Search existing GitHub issues
3. Create a new issue with:
   - Command you ran
   - Full error message
   - Your environment (OS, Python version)

---

## FAQ

### Q: Do I need to know programming?

**A**: No! All tools work from the command line with simple commands. Just copy and paste the examples.

### Q: Can I use this on Windows?

**A**: Yes! Install Python and use Command Prompt or PowerShell. Some commands may differ slightly (use `type` instead of `cat`, for example).

### Q: How do I export my ontology for publication?

**A**: Use the Turtle converter to create an RDF file:
```bash
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl
```

### Q: How do I search for a specific concept?

**A**: Use the web visualization at https://narrativegoldmine.com and use the search box.

### Q: Can I edit files in Logseq?

**A**: Yes! Logseq is the native editor for these files. Changes sync automatically.

### Q: What's the difference between domains?

**A**: Domains organize concepts by subject area:
- **AI**: Artificial Intelligence concepts
- **BC**: Blockchain concepts
- **RB**: Robotics concepts
- **MV**: Metaverse concepts
- **TC**: Telecollaboration concepts
- **DT**: Disruptive Technologies concepts

### Q: How do I add a new domain?

**A**: Contact the maintainers. Adding domains requires updating configuration files.

### Q: Can I convert only one domain?

**A**: Yes! Use the `--domain` flag:
```bash
python convert-to-turtle.py --domain ai --output ai-only.ttl
```

### Q: What format should I use for publication?

**A**: RDF Turtle (.ttl) is the standard format for semantic web publication.

### Q: How do I cite this ontology?

**A**: Use the ontology IRI and version:
```
Narrative Goldmine Ontology. Version 1.0.0.
http://narrativegoldmine.com/
```

### Q: Can I import my ontology into Protégé?

**A**: Yes! Convert to Turtle or OWL/XML format, then open in Protégé.

### Q: How often should I validate?

**A**: Validate after making changes, before committing, and before publishing.

---

## Next Steps

### For Ontology Authors
1. Read: Creating a New Concept (above)
2. Practice: Add a test concept
3. Validate: Check your work
4. Commit: Save changes to git

### For Researchers
1. Export to CSV for analysis
2. Import into your preferred tool
3. Use visualization for exploration
4. Cite in your publications

### For Publishers
1. Follow Workflow 3: Validating and Publishing
2. Generate all export formats
3. Build web visualization
4. Deploy to your server

### Learning More

- **Tooling Overview**: See `/docs/TOOLING-OVERVIEW.md` for complete tool catalog
- **Workflows**: See `/docs/TOOL-WORKFLOWS.md` for detailed workflows
- **Developer Guide**: See `/docs/DEVELOPER-GUIDE.md` if you want to extend tools
- **API Reference**: See `/docs/API-REFERENCE.md` for programmatic access

---

## Getting Support

### Documentation
- User Guide (this document)
- Tool Workflows: `/docs/TOOL-WORKFLOWS.md`
- Tooling Overview: `/docs/TOOLING-OVERVIEW.md`

### Issues and Bugs
- GitHub Issues: https://github.com/DreamLab-AI/knowledgeGraph/issues

### Community
- GitHub Discussions
- Project Wiki

---

**Maintainer**: Claude Code Agent
**Last Updated**: 2025-11-21
**Version**: 1.0.0
**License**: See LICENSE file
