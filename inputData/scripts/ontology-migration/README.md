# Ontology Block Migration Pipeline

Automated pipeline for batch processing and standardizing 1,709 ontology blocks in Logseq markdown files.

## 🎯 Overview

This pipeline transforms existing ontology blocks across 6 different patterns into a canonical, standardized format. It handles critical fixes including:

- **Namespace corrections** (mv: → rb: for robotics files)
- **Class naming standardization** (lowercase → CamelCase)
- **Status/maturity normalization**
- **Duplicate section removal**
- **Format consistency** (tabs → spaces, indentation)

## 📋 Features

- ✅ **Safe Processing**: Dry-run mode by default, automatic backups
- ✅ **Batch Processing**: Handles large file sets in configurable batches
- ✅ **Error Handling**: Graceful error recovery and detailed logging
- ✅ **Progress Tracking**: Checkpointing and resumable operations
- ✅ **Validation**: Pre and post-processing validation
- ✅ **Rollback**: Restore from backups if needed
- ✅ **Domain/Pattern Filtering**: Process specific subsets
- ✅ **Detailed Reports**: JSON reports for all operations

## 🚀 Quick Start

### Installation

```bash
cd /home/user/logseq/scripts/ontology-migration

# Ensure Node.js is installed (v14+ recommended)
node --version

# Make CLI executable
chmod +x cli.js
```

### Basic Workflow

```bash
# 1. Scan all files and generate inventory
node cli.js scan

# 2. Preview transformations (first 10 files)
node cli.js preview 10

# 3. Test on a single file
node cli.js test /path/to/file.md

# 4. Run full migration (DRY RUN first!)
node cli.js process

# 5. When satisfied, run LIVE update
node cli.js process --live

# 6. Validate results
node cli.js validate

# 7. Check statistics
node cli.js stats
```

## 📚 Commands

### scan
Scan all markdown files and generate inventory report.

```bash
node cli.js scan
```

**Output**: `docs/ontology-migration/reports/file-inventory.json`

### preview
Preview transformations without making changes.

```bash
node cli.js preview [number]

# Examples:
node cli.js preview 10    # Preview first 10 files
node cli.js preview 50    # Preview first 50 files
```

### process
Process all files (batch migration).

```bash
node cli.js process [options]

# Options:
#   --live         Perform actual updates (default: dry-run)
#   --batch=N      Set batch size (default: 100)
#   --validate     Run validation after processing
#   --no-backup    Disable backup creation (NOT recommended)

# Examples:
node cli.js process                    # Dry-run
node cli.js process --live             # Live update
node cli.js process --live --batch=50  # Live with batch size 50
node cli.js process --live --validate  # Live with validation
```

### validate
Validate ontology blocks against canonical format.

```bash
node cli.js validate
```

**Output**: `docs/ontology-migration/reports/validation-report.json`

### test
Test transformation on a single file.

```bash
node cli.js test <file-path>

# Example:
node cli.js test mainKnowledgeGraph/pages/rb-0010-aerial-robot.md
```

### domain
Process specific domain only.

```bash
node cli.js domain <domain> [options]

# Domains: ai, blockchain, robotics, metaverse

# Examples:
node cli.js domain robotics --live     # Process only robotics files
node cli.js domain ai --live           # Process only AI files
```

### pattern
Process specific pattern only.

```bash
node cli.js pattern <pattern> [options]

# Patterns: pattern1, pattern2, pattern3, pattern4, pattern5, pattern6

# Example:
node cli.js pattern pattern3 --live    # Process only pattern3 (robotics simplified)
```

### rollback
Restore files from backups.

```bash
node cli.js rollback
```

**⚠️  WARNING**: This will overwrite current files with backups!

### stats
Show current statistics and reports.

```bash
node cli.js stats
```

## 🗂️ Directory Structure

```
scripts/ontology-migration/
├── cli.js                 # Command-line interface
├── scanner.js             # File scanner and inventory generator
├── parser.js              # Ontology block parser
├── generator.js           # Canonical block generator
├── updater.js             # File updater with backup
├── validator.js           # Format validator
├── batch-process.js       # Batch processing orchestrator
├── config.json            # Configuration
└── README.md              # This file

docs/ontology-migration/
├── reports/               # Generated reports
│   ├── file-inventory.json
│   ├── validation-report.json
│   ├── update-report.json
│   ├── final-report.json
│   └── checkpoint.json
├── backups/               # File backups (created automatically)
├── schemas/               # Canonical format schemas
└── analysis/              # Analysis documents
    ├── block-patterns-catalog.md
    ├── semantic-patterns-analysis.md
    └── best-practices-research.md

mainKnowledgeGraph/pages/  # Source files (1,709 .md files)
```

## ⚙️ Configuration

Edit `config.json` to customize behavior:

```json
{
  "sourceDirectory": "/home/user/logseq/mainKnowledgeGraph/pages",
  "backupDirectory": "/home/user/logseq/docs/ontology-migration/backups",
  "reportsDirectory": "/home/user/logseq/docs/ontology-migration/reports",
  "batchSize": 100,
  "dryRun": true,
  "verboseLogging": true,
  "createBackups": true,
  "validateAfterUpdate": true,
  ...
}
```

## 🔧 Pipeline Components

### 1. Scanner (`scanner.js`)
- Scans all .md files in source directory
- Identifies files with ontology blocks
- Classifies by pattern (pattern1-6)
- Detects domain (ai, blockchain, robotics, metaverse)
- Identifies issues (namespace errors, naming issues, etc.)
- Generates inventory report

### 2. Parser (`parser.js`)
- Extracts ontology block from file
- Parses all properties and relationships
- Identifies OWL axioms
- Detects namespace usage
- Analyzes issues with specific fixes
- Preserves content below ontology block

### 3. Generator (`generator.js`)
- Reads canonical schema
- Generates new ontology block
- Fixes namespace issues (mv: → rb:)
- Converts class names to CamelCase
- Normalizes status and maturity values
- Applies domain-appropriate template
- Preserves essential existing data

### 4. Updater (`updater.js`)
- Creates backup before modification
- Replaces old block with new canonical block
- Preserves file content below ontology block
- Maintains file metadata
- Validates after update (optional)
- Provides rollback capability

### 5. Validator (`validator.js`)
- Checks canonical format compliance
- Verifies required properties present
- Validates OWL syntax
- Checks namespace correctness
- Calculates validation score (0-100)
- Reports issues and warnings

### 6. Batch Processor (`batch-process.js`)
- Orchestrates full pipeline
- Processes files in batches
- Handles errors gracefully
- Creates progress checkpoints
- Generates comprehensive reports
- Supports domain/pattern filtering

## 📊 Reports Generated

### File Inventory (`file-inventory.json`)
```json
{
  "totalFiles": 1709,
  "filesWithOntology": 1450,
  "patternDistribution": {
    "pattern1": 580,
    "pattern2": 362,
    "pattern3": 290,
    ...
  },
  "domainDistribution": {
    "ai": 400,
    "blockchain": 300,
    "robotics": 250,
    "metaverse": 500
  },
  "issues": {
    "namespaceErrors": [...],
    "namingIssues": [...],
    "duplicateSections": [...]
  },
  "fileInventory": [...]
}
```

### Validation Report (`validation-report.json`)
```json
{
  "summary": {
    "totalValidated": 1450,
    "passed": 1380,
    "failed": 70,
    "warnings": 250
  },
  "averageScore": 92.5,
  "validations": [...]
}
```

### Update Report (`update-report.json`)
```json
{
  "processed": 1450,
  "updated": 1420,
  "skipped": 15,
  "errors": 15,
  "backupsCreated": 1420
}
```

## 🛡️ Safety Features

### 1. Dry-Run Mode (Default)
All operations run in dry-run mode by default. No files are modified unless `--live` flag is used.

### 2. Automatic Backups
Before modifying any file, a timestamped backup is created in `docs/ontology-migration/backups/`.

### 3. Rollback Capability
Restore all files from backups using `node cli.js rollback`.

### 4. Progress Checkpointing
Pipeline state is saved after each batch. Operations can be resumed if interrupted.

### 5. Validation Checks
Pre and post-processing validation ensures data integrity.

### 6. Error Handling
Graceful error recovery prevents batch failures from corrupting data.

## 🧪 Testing Workflow

### Test Single File
```bash
# Test transformation on one file
node cli.js test mainKnowledgeGraph/pages/rb-0010-aerial-robot.md
```

**Output**:
- ✅ Parsing result
- ✅ Generated canonical block preview
- ✅ Validation results with score
- ✅ Errors and warnings

### Preview Batch
```bash
# Preview first 10 transformations
node cli.js preview 10
```

### Dry-Run Full Pipeline
```bash
# Simulate full migration without changes
node cli.js process
```

### Validate Current State
```bash
# Check current ontology blocks
node cli.js validate
```

## 🚀 Production Workflow

### Phase 1: Analysis (Complete)
✅ Patterns cataloged
✅ Issues identified
✅ Canonical format defined

### Phase 2: Setup & Testing
```bash
# 1. Scan files
node cli.js scan

# 2. Test on sample files
node cli.js test mainKnowledgeGraph/pages/rb-0010-aerial-robot.md
node cli.js test mainKnowledgeGraph/pages/AI-0207-encoder-decoder-architecture.md

# 3. Preview batch
node cli.js preview 20

# 4. Validate current state
node cli.js validate
```

### Phase 3: Pilot Run (Recommended)
```bash
# Process robotics domain only (critical namespace fix)
node cli.js domain robotics --live --validate

# Check results
node cli.js stats
```

### Phase 4: Full Migration
```bash
# Run full pipeline with validation
node cli.js process --live --validate

# Monitor progress in checkpoint.json
watch -n 5 cat docs/ontology-migration/reports/checkpoint.json
```

### Phase 5: Validation & Verification
```bash
# Validate all files
node cli.js validate

# Review reports
node cli.js stats

# Manual spot-checks
```

## 🔄 Resuming Interrupted Operations

If the pipeline is interrupted:

```bash
# Check last checkpoint
cat docs/ontology-migration/reports/pipeline-checkpoint.json

# Resume from checkpoint (future feature)
node cli.js process --resume
```

For now, the pipeline will skip already-processed files automatically.

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure you're in the correct directory:
```bash
cd /home/user/logseq/scripts/ontology-migration
node cli.js scan
```

### Issue: "Permission denied"
**Solution**: Make scripts executable:
```bash
chmod +x cli.js scanner.js parser.js generator.js updater.js validator.js batch-process.js
```

### Issue: "Out of memory"
**Solution**: Reduce batch size:
```bash
node --max-old-space-size=4096 cli.js process --live --batch=50
```

### Issue: "Files not found"
**Solution**: Check source directory in config.json matches your installation.

## 📈 Performance

**Estimated Processing Time**:
- Scan: ~2-3 minutes (1,709 files)
- Parse: ~30 seconds per 100 files
- Generate: ~10 seconds per 100 files
- Update: ~1 minute per 100 files (with backups)
- Validate: ~30 seconds per 100 files

**Total Pipeline**: ~30-40 minutes for 1,709 files (batch size 100)

## 📝 Best Practices

1. **Always test first**: Use `test` and `preview` commands before live runs
2. **Keep backups enabled**: Don't use `--no-backup` unless absolutely necessary
3. **Start with dry-run**: Never skip dry-run validation
4. **Process by domain**: Consider processing domains separately for easier rollback
5. **Monitor progress**: Check checkpoint.json during long runs
6. **Validate results**: Always run validation after updates
7. **Review reports**: Check generated reports for issues
8. **Keep originals**: Maintain git commits before major changes

## 🔗 Integration with Claude Flow

Store results in swarm memory:

```bash
# After processing
npx claude-flow@alpha hooks post-edit \
  --file "scripts/ontology-migration/cli.js" \
  --memory-key "swarm/coder/processor"

npx claude-flow@alpha hooks notify \
  --message "Processing pipeline built and tested: ${stats}"
```

## 📚 Additional Resources

- **Analysis Documents**: `docs/ontology-migration/analysis/`
- **Pattern Catalog**: `docs/ontology-migration/analysis/block-patterns-catalog.md`
- **Semantic Analysis**: `docs/ontology-migration/analysis/semantic-patterns-analysis.md`
- **Reports**: `docs/ontology-migration/reports/`

## 🤝 Contributing

When modifying the pipeline:

1. Update relevant scripts
2. Test thoroughly with sample files
3. Update this README
4. Run full validation suite
5. Commit changes with descriptive messages

## 📚 Related Documentation

Comprehensive documentation is available in `/docs/`:

- **Tooling Overview** (`/docs/TOOLING-OVERVIEW.md`)
  - Complete map of all tools including migration tools
  - How migration tools fit into the ecosystem
  - Dependencies and architecture

- **Tool Workflows** (`/docs/TOOL-WORKFLOWS.md`)
  - Workflow #5: Batch Migration and Standardization (complete guide)
  - Integration with other tools
  - Production deployment workflow

- **User Guide** (`/docs/USER-GUIDE.md`)
  - For non-developers using migration tools
  - Step-by-step instructions
  - Troubleshooting common issues

- **Developer Guide** (`/docs/DEVELOPER-GUIDE.md`)
  - Extending the migration pipeline
  - Adding new transforms
  - Testing requirements

- **API Reference** (`/docs/API-REFERENCE.md`)
  - JavaScript API for migration tools
  - Scanner, Parser, Generator APIs
  - CLI reference

## 🔗 Integration with Other Tools

### After Migration

Once migration is complete, use these tools:

**Convert to RDF**:
```bash
cd /home/user/logseq
python Ontology-Tools/tools/converters/convert-to-turtle.py \
  --input mainKnowledgeGraph/pages/ \
  --output ontology.ttl
```

**Validate OWL2**:
```bash
python scripts/validate_owl2.py ontology.ttl
```

**Generate Web Visualization**:
```bash
# Convert to WebVOWL format
python Ontology-Tools/tools/converters/ttl_to_webvowl_json.py \
  ontology.ttl \
  webvowl.json

# Copy to frontend
cp webvowl.json publishing-tools/WasmVOWL/modern/public/data/ontology.json
```

**Generate Search Index**:
```bash
python Ontology-Tools/tools/converters/generate_search_index.py \
  --input mainKnowledgeGraph/pages/ \
  --output publishing-tools/WasmVOWL/modern/public/search-index.json
```

### Shared Libraries

The migration tools use JavaScript, but converters can access the same data using Python libraries:

```python
from ontology_loader import OntologyLoader

loader = OntologyLoader()
blocks = loader.load_directory(Path('mainKnowledgeGraph/pages/'))
# Now use for any conversion or analysis
```

See `/docs/API-REFERENCE.md` for complete Python API documentation.

## 📄 License

Part of the Logseq Knowledge Graph Ontology Standardization Project.

---

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Maintainer**: Claude Code Agent
**Status**: Ready for testing and deployment
