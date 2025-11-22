# Ontology Format Audit Tool - Updates & Documentation

**Version:** 1.0.0
**Date:** 2025-11-21
**Status:** Complete
**Tool Location:** `/home/user/logseq/Ontology-Tools/tools/audit/`

---

## Executive Summary

The canonical ontology format audit tool has been completely redesigned to validate markdown files against the **Canonical Ontology Block Schema v1.0.0**. The new tool performs comprehensive format validation, IRI uniqueness checking, and generates detailed compliance reports.

### Key Improvements

- **Format-Focused Validation**: Replaces semantic graph analysis with structural format validation
- **Canonical Compliance**: Validates 12 Tier 1 required properties across all files
- **IRI Tracking**: Detects duplicate IRIs and format violations across all files
- **Domain Analysis**: Organizes findings by domain (AI, Blockchain, Robotics, Metaverse)
- **JSON Reporting**: Generates detailed JSON reports with file-by-file validation results
- **Actionable Recommendations**: Provides specific guidance for remediation

---

## What the Tool Validates

### 1. Ontology Block Presence & Position

**Checks:**
- Files MUST contain exactly ONE `### OntologyBlock` section
- Block MUST be first major content block (warnings if not)
- Detects multiple blocks per file (critical error)

**Errors Reported:**
- "No OntologyBlock found in file"
- "Multiple OntologyBlocks found (N)"
- Warning: "OntologyBlock is not first content block in file"

---

### 2. Tier 1 Required Properties

**All 12 properties MUST be present:**

| Property | Format | Validation |
|----------|--------|-----------|
| `ontology` | boolean | Must be `true` |
| `term-id` | string | Format: `PREFIX-NNNN` (AI, BC, RB) or numeric (20xxx) |
| `preferred-term` | string | Human-readable title |
| `source-domain` | enum | One of: ai, blockchain, robotics, metaverse, general |
| `status` | enum | One of: draft, in-progress, complete, deprecated |
| `public-access` | boolean | Must be `true` or `false` (not "yes"/"no") |
| `last-updated` | date | ISO 8601 format: YYYY-MM-DD |
| `definition` | string | 2-5 sentences with [[wiki links]] |
| `owl:class` | IRI | Format: `namespace:ClassName` (PascalCase) |
| `owl:physicality` | enum | One of: PhysicalEntity, VirtualEntity, AbstractEntity, HybridEntity |
| `owl:role` | enum | One of: Object, Process, Agent, Quality, Relation, Concept |
| `is-subclass-of` | page link | At least one parent class reference |

**Example Error:**
```
Missing required properties: last-updated, owl:role, is-subclass-of
```

---

### 3. Term-ID Format Validation

**Valid Formats:**
- Domain-prefixed: `AI-0850`, `BC-0026`, `RB-0010` (4-digit suffix)
- Metaverse numeric: `20150`, `20151` (5-digit, starts with 20)

**Invalid Formats:**
- Missing hyphen: `AI0850`
- Wrong suffix length: `AI-85` or `AI-08500`
- Invalid prefix: `MB-0010`, `XY-1234`
- Non-numeric suffix: `AI-085A`

**Example Error:**
```
Invalid term-id format: BC-0026-ALT
```

---

### 4. Namespace-Domain Consistency

**Required Mapping:**

| Domain | Expected Namespace | Example IRI |
|--------|-------------------|-----------|
| `ai` | `ai:` | `ai:LargeLanguageModel` |
| `blockchain` | `bc:` | `bc:ConsensusMechanism` |
| `robotics` | `rb:` | `rb:AerialRobot` |
| `metaverse` | `mv:` | `mv:GameEngine` |

**Validation:**
- If `source-domain:: ai`, then `owl:class::` MUST start with `ai:`
- Catches critical robotics migration errors (e.g., `mv:rb...` → should be `rb:...`)

**Example Error:**
```
Namespace mismatch: expected rb but got mv for domain robotics
```

---

### 5. Property Value Validation

**Status Values:**
- Accepted: `draft`, `in-progress`, `complete`, `deprecated`
- Rejected: `working`, `finished`, `published`, `unknown`

**Public-Access Values:**
- Accepted: `true`, `false`
- Rejected: `yes`, `no`, `1`, `0`, `public`, `private`

**Date Format:**
- Accepted: `2025-11-21`, `2024-01-01`
- Rejected: `11/21/2025`, `21-11-2025`, `Nov 21, 2025`

**Physicality Values:**
- `PhysicalEntity` - Physical objects, hardware, robots
- `VirtualEntity` - Software, algorithms, digital systems
- `AbstractEntity` - Concepts, theories, methodologies
- `HybridEntity` - Cyber-physical systems

**Role Values:**
- `Object` - Things, entities, artifacts
- `Process` - Activities, methods, operations
- `Agent` - Autonomous actors with goals
- `Quality` - Attributes, properties, characteristics
- `Relation` - Connections, links, associations
- `Concept` - Abstract ideas, categories

---

### 6. Class Naming (PascalCase)

**Valid Names:**
- `LargeLanguageModel`
- `ConsensusMechanism`
- `AerialRobot`
- `GameEngine`

**Invalid Names:**
- `largelanguagemodel` (not PascalCase)
- `large-language-model` (contains hyphens)
- `Large_Language_Model` (contains underscores)
- `large language model` (contains spaces)

**Note:** Tool warns but doesn't fail on non-PascalCase names (should be fixed but not blocking).

---

### 7. Filename-TermID Consistency

**Validation:**
- Filename SHOULD contain the term-id
- Files named: `AI-0850-*` should have `term-id:: AI-0850`
- Tools warns if filename and term-id don't match

**Example Warning:**
```
Filename 'custom-name.md' does not match term-id 'AI-0850'
```

---

### 8. IRI Uniqueness

**Checks:**
- Every IRI (owl:class value) should be unique across all files
- Detects duplicate IRIs and lists affected files
- Validates IRI format: `namespace:ClassName`

**Output in Report:**
```json
"duplicate_iris": [
  {
    "iri": "ai:LargeLanguageModel",
    "files": ["file1.md", "file2.md"]
  }
]
```

---

## Using the Audit Tool

### Building the Tool

```bash
cd /home/user/logseq/Ontology-Tools/tools/audit

# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

### Running the Tool

```bash
# Basic usage (scans default directory)
./target/release/ontology-audit

# Custom pages directory
./target/release/ontology-audit --pages /path/to/pages

# Custom output location
./target/release/ontology-audit --output /path/to/report.json

# All options together
./target/release/ontology-audit \
  --pages mainKnowledgeGraph/pages \
  --output outputs/audit-report.json
```

### Command-Line Options

```
OPTIONS:
  -p, --pages <PAGES>
      Path to markdown pages directory
      [default: mainKnowledgeGraph/pages]

  -o, --output <OUTPUT>
      Output JSON report path
      [default: outputs/ontology-format-audit-report.json]

  --check-iri-uniqueness <CHECK_IRI_UNIQUENESS>
      Check for IRI uniqueness across all files
      [default: true]

  -h, --help
      Print help

  -V, --version
      Print version
```

---

## Understanding the Report

### Console Output Summary

```
📊 Format Compliance:
   Total Files Scanned: 6
   With OntologyBlock: 5
   Format Compliant: 1
   Compliance Rate: 16.7%

📁 Files Per Domain:
   robotics: 2 files
   ai: 1 files
   blockchain: 1 files
   metaverse: 1 files

🔗 IRI Analysis:
   Total IRIs: 5
   Unique IRIs: 5
   Duplicate IRIs: 0

⚠️  Issues Found:
   Multiple blocks per file: 1
   Block not first: 5
   Missing required properties: 2
   Invalid term-ids: 1
   Namespace mismatches: 1
   Invalid public-access: 1
   Domain classification errors: 0
   Malformed blocks: 1

💡 Recommendations:
   1. CRITICAL: Less than 50% of files are format compliant...
   2. ACTION: Fix identified issues...
   [etc.]
```

### JSON Report Structure

```json
{
  "summary": {
    "total_files_scanned": 6,
    "files_with_ontology_block": 5,
    "format_compliant_files": 1,
    "compliance_percentage": 16.7,
    "files_per_domain": {
      "ai": 1,
      "robotics": 2,
      "blockchain": 1,
      "metaverse": 1
    }
  },
  "format_validation": [
    {
      "file_path": "/path/to/file.md",
      "file_name": "AI-0850-valid.md",
      "is_valid": true,
      "ontology_block_count": 1,
      "block_position": 2,
      "term_id": "AI-0850",
      "domain": "ai",
      "iri": "ai:LargeLanguageModel",
      "errors": [],
      "warnings": [
        "OntologyBlock is not first content block in file"
      ]
    }
  ],
  "iri_analysis": {
    "total_iris": 5,
    "unique_iris": 5,
    "duplicate_iris": [],
    "iri_format_errors": []
  },
  "files_by_domain": {
    "ai": ["AI-0850-valid.md"],
    "robotics": ["RB-0010-missing-properties.md", "RB-0020-namespace-mismatch.md"],
    "blockchain": ["BC-0026-multiple-blocks.md"],
    "metaverse": ["MV-20150-invalid-date.md"]
  },
  "issues_summary": {
    "multiple_blocks_per_file": ["BC-0026-multiple-blocks.md"],
    "block_not_first": [
      "RB-0010-missing-properties.md",
      "RB-0020-namespace-mismatch.md",
      ...
    ],
    "missing_required_properties": [
      {
        "file": "RB-0010-missing-properties.md",
        "missing_properties": ["last-updated", "owl:role", "is-subclass-of"]
      }
    ],
    "invalid_term_ids": [
      {
        "file": "BC-0026-multiple-blocks.md",
        "term_id": "BC-0026-ALT",
        "reason": "Invalid term-id format: BC-0026-ALT"
      }
    ],
    "namespace_mismatches": [
      {
        "file": "RB-0020-namespace-mismatch.md",
        "expected_namespace": "rb",
        "actual_namespace": "mv"
      }
    ],
    "invalid_public_access": ["MV-20150-invalid-date.md"],
    "domain_classification_errors": [],
    "malformed_blocks": ["no-ontology-block.md"]
  },
  "recommendations": [
    "CRITICAL: Less than 50% of files are format compliant. Prioritize migration immediately.",
    "ACTION: 1 files have multiple OntologyBlocks. Each file should have exactly one block.",
    "ACTION: 2 files are missing required Tier 1 properties...",
    ...
  ]
}
```

---

## Test Results

### Sample Test Files Created

Located in `/home/user/logseq/Ontology-Tools/sample_test_files/`:

| File | Status | Issues |
|------|--------|--------|
| `AI-0850-valid.md` | VALID | None (only position warning) |
| `BC-0026-multiple-blocks.md` | INVALID | Multiple blocks, invalid term-id |
| `RB-0010-missing-properties.md` | INVALID | Missing 3 properties |
| `RB-0020-namespace-mismatch.md` | INVALID | Namespace mismatch (mv: instead of rb:) |
| `MV-20150-invalid-date.md` | INVALID | Invalid date, invalid public-access |
| `no-ontology-block.md` | INVALID | No OntologyBlock |

### Test Run Results

```
✅ Format Audit Complete!

📊 Summary:
   Total Files: 6
   With OntologyBlock: 5
   Format Compliant: 1 (16.7%)

📁 By Domain:
   AI: 1, Blockchain: 1, Robotics: 2, Metaverse: 1

⚠️  Issues:
   Multiple blocks: 1
   Block not first: 5
   Missing properties: 2 files
   Invalid term-ids: 1
   Namespace mismatches: 1
   Invalid public-access: 1
   No block: 1

💡 Recommendations: 7 actions identified
```

---

## Key Features

### 1. Comprehensive Validation

The tool checks **12 categories** of validation:

- Ontology block presence and count
- Block position in file
- All 12 Tier 1 required properties
- Property value types and constraints
- Term-ID format compliance
- Domain-namespace mapping
- IRI uniqueness
- PascalCase naming conventions
- ISO 8601 date format
- Filename-term-id consistency

### 2. Domain Organization

Automatically categorizes files by domain:
- **AI Domain**: `ai:` namespace, `AI-XXXX` term-ids
- **Blockchain**: `bc:` namespace, `BC-XXXX` term-ids
- **Robotics**: `rb:` namespace, `RB-XXXX` term-ids
- **Metaverse**: `mv:` namespace, numeric term-ids (20xxx)

### 3. IRI Collision Detection

- Tracks all IRIs across all files
- Detects and lists duplicates
- Reports format errors in IRIs
- Helps ensure uniqueness constraint

### 4. Actionable Recommendations

- Severity-based messaging (CRITICAL, WARNING, ACTION)
- Specific counts of affected files
- Clear remediation guidance
- References to migration rules

### 5. JSON Report Export

- Machine-readable JSON format
- File-by-file validation results
- Detailed error/warning messages
- Domain and IRI analysis
- Suitable for automated processing

---

## Usage Workflow

### For Regular Audits

```bash
# 1. Run audit weekly
./target/release/ontology-audit \
  --pages mainKnowledgeGraph/pages \
  --output outputs/audit-$(date +%Y%m%d).json

# 2. Review console summary for quick overview

# 3. Analyze JSON for detailed findings

# 4. Generate migration plan based on recommendations

# 5. Track compliance improvement over time
```

### For Migration Verification

```bash
# After Phase 1 (Critical Fixes)
./target/release/ontology-audit

# Check if namespace mismatches are resolved
# Verify all term-ids are in correct format
# Confirm compliance percentage increased

# Re-run after each phase to track progress
```

### For Continuous Monitoring

```bash
# Add to CI/CD pipeline
cargo build --release && \
./target/release/ontology-audit --output report.json && \
# Check if compliance_percentage >= target (e.g., 95%)
```

---

## Interpreting Common Issues

### Issue: "Multiple OntologyBlocks found (2)"

**Cause:** File has more than one `### OntologyBlock` section

**Fix:** Merge blocks into single canonical block or split into separate files

**Migration Rule:** Rule 4.2 covers this scenario

### Issue: "Namespace mismatch: expected rb but got mv"

**Cause:** Robotics file using wrong namespace prefix

**Example:** `owl:class:: mv:AerialRobot` should be `owl:class:: rb:AerialRobot`

**Fix:** Update namespace to match domain

**Migration Rule:** Rule 1.1 (Critical Robotics Namespace Fix)

### Issue: "Invalid term-id format: BC-0026-ALT"

**Cause:** Term-ID doesn't follow required format

**Valid Formats:**
- `AI-0850`, `BC-0026`, `RB-0010` (domain prefixed)
- `20150`, `20151` (metaverse numeric)

**Invalid:**
- `BC-0026-ALT` (has non-numeric suffix)
- `BC-26` (insufficient digits)
- `BC0026` (missing hyphen)

**Fix:** Conform to format rules

**Migration Rule:** Rule 2.1

### Issue: "Invalid public-access value: yes"

**Cause:** Boolean property using non-boolean value

**Valid Values:** `true`, `false`

**Invalid Values:** `yes`, `no`, `1`, `0`, `"true"` (quoted)

**Fix:** Use exact values `true` or `false`

### Issue: "OntologyBlock is not first content block"

**Cause:** Non-empty content exists before the OntologyBlock

**Warning Level:** Non-critical (doesn't fail validation)

**Fix:** Move OntologyBlock to top of file (after YAML frontmatter)

---

## Differences from Previous Tool

### Old Tool (Semantic Audit)

- Focused on RDF/TTL graph analysis
- Checked connectivity and isolated nodes
- Analyzed OWL axiom richness
- Detected missing OWL blocks
- Generated quality metrics

### New Tool (Format Audit)

- Focuses on markdown file structure
- Validates format compliance
- Checks IRI uniqueness
- Tracks domain classification
- Generates migration guidance

### Integration

- New tool is **complementary**, not replacement
- Use format audit for **migration tracking**
- Use semantic audit for **ontology quality**
- Run both in different phases

---

## Future Enhancements

### Planned Features

1. **Automated Fixing**
   - Auto-convert invalid public-access values
   - Auto-convert dates to ISO format
   - Auto-generate missing term-ids

2. **Advanced IRI Analysis**
   - Detect semantic duplicates (not just exact matches)
   - Suggest IRI consolidation
   - Validate IRI URIs resolve

3. **Owl Axiom Validation**
   - Parse and validate OWL Functional Syntax
   - Run reasoner checks
   - Detect unsatisfiable classes

4. **Cross-Domain Analysis**
   - Validate cross-domain bridges
   - Check consistency of relationships
   - Detect orphaned references

5. **Compliance Tracking**
   - Historical compliance graphs
   - Trend analysis
   - Compliance targets by domain

---

## Building the Tool from Source

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs/)
- Git

### Build Steps

```bash
# Clone or navigate to repository
cd /home/user/logseq/Ontology-Tools/tools/audit

# Build debug version (fast, larger binary)
cargo build

# Build release version (slower build, optimized binary)
cargo build --release

# Run tests (if any exist)
cargo test

# Create documentation
cargo doc --open
```

### Binary Location

- Debug: `./target/debug/ontology-audit`
- Release: `./target/release/ontology-audit`

### Dependencies

All dependencies are specified in `Cargo.toml`:

```toml
[dependencies]
anyhow = "1.0"          # Error handling
clap = "4.5"            # CLI argument parsing
serde = "1.0"           # Serialization
serde_json = "1.0"      # JSON support
walkdir = "2.4"         # Directory traversal
regex = "1.12"          # Pattern matching
```

---

## Documentation References

- **Canonical Format Spec:** `/home/user/logseq/docs/ontology-migration/schemas/canonical-ontology-block.md`
- **Migration Rules:** `/home/user/logseq/docs/ontology-migration/schemas/migration-rules.md`
- **Source Code:** `/home/user/logseq/Ontology-Tools/tools/audit/src/main.rs`
- **Test Files:** `/home/user/logseq/Ontology-Tools/sample_test_files/`

---

## Support & Reporting Issues

### Getting Help

1. Check this documentation first
2. Review sample test files for examples
3. Examine generated JSON reports for details
4. Refer to canonical format spec for requirements

### Reporting Bugs

Include in bug report:
- Command used to run tool
- Input file (or minimal reproduction)
- Expected behavior
- Actual behavior
- Console output

### Suggesting Improvements

Submit enhancement requests with:
- Use case description
- Proposed feature
- Example of expected output
- Priority assessment

---

## Changelog

### Version 1.0.0 (2025-11-21)

**Initial Release:**
- Complete redesign for format validation
- 12-category validation system
- JSON report generation
- Domain-based organization
- IRI uniqueness checking
- Actionable recommendations
- Console and file output

**Tested Against:**
- Canonical ontology format spec v1.0.0
- Migration rules v1.0.0
- 6 sample test files covering all major issues

---

**Document Control:**
- **Version**: 1.0.0
- **Status**: Complete & Tested
- **Last Updated**: 2025-11-21
- **Next Review**: After Phase 1 migration completion
- **Maintenance**: Update after each tool version release

