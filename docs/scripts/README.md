---
title: "Documentation Scripts Reference"
description: "Reference guide for documentation validation and automation scripts"
category: reference
tags:
  - automation
  - validation
  - ci-cd
  - tools
updated-date: 2025-12-19
difficulty-level: intermediate
---

# Documentation Scripts Reference

This directory contains automation scripts for validating, maintaining, and generating documentation.

---

## Quick Reference

```bash
# Run all validators
./validate-all.sh

# Validate specific aspects
./validate-links.sh
./validate-frontmatter.sh
./validate-mermaid.sh
./detect-ascii.sh
./validate-coverage.sh

# Generate outputs
./generate-reports.sh
./generate-index.sh
```

---

## Scripts Overview

### Validation Scripts

#### `validate-links.sh`

Validates all internal links and detects orphaned documents.

**Usage:**
```bash
./validate-links.sh
```

**Output:**
- Report file: `/tmp/link-validation-report.txt`
- Exit code: 0 (success), 1 (failures found)

**Checks:**
- Broken internal links
- Orphaned documents (no incoming links)
- Invalid link paths

**Example Output:**
```
=== Link Validation Report ===
Generated: 2025-12-18 10:30:00

### Orphaned Documents (no incoming links)
✓ No orphaned documents found

### Broken Links
✓ No broken links found

### Summary
- Total documents: 45
- Orphaned documents: 0
- Files with broken links: 0
```

---

#### `validate-frontmatter.sh`

Ensures all documents have complete and valid front matter.

**Usage:**
```bash
./validate-frontmatter.sh
```

**Output:**
- Report file: `/tmp/frontmatter-validation-report.txt`
- Exit code: 0 (success), 1 (failures found)

**Required Fields:**
- `title`
- `description`
- `category`
- `tags`
- `version`
- `last_updated`

**Example Output:**
```
=== Front Matter Validation Report ===
Generated: 2025-12-18 10:30:00

### Summary
- Total documents validated: 45
- Documents with issues: 0

✓ All documents have valid front matter
```

---

#### `validate-mermaid.sh`

Validates Mermaid diagram syntax.

**Usage:**
```bash
./validate-mermaid.sh
```

**Output:**
- Report file: `/tmp/mermaid-validation-report.txt`
- Exit code: 0 (success), 1 (failures found)

**Checks:**
- Valid diagram types
- Proper syntax patterns
- Required elements

**Supported Diagram Types:**
- `graph` / `flowchart`
- `sequenceDiagram`
- `classDiagram`
- `stateDiagram`
- `erDiagram`
- `gantt`
- `pie`
- `gitGraph`

---

#### `detect-ascii.sh`

Detects remaining ASCII art diagrams that should be converted to Mermaid.

**Usage:**
```bash
./detect-ascii.sh
```

**Output:**
- Report file: `/tmp/ascii-detection-report.txt`
- Exit code: 0 (none found), 1 (ASCII diagrams detected)

**Detection Patterns:**
- Box drawing characters: `+---+`, `|...|`
- Unicode box drawing: `┌─┐`, `├─┤`, `└─┘`
- Arrow symbols: `▼`, `▲`, `►`, `◄`
- Multiple horizontal/vertical lines

---

#### `validate-coverage.sh`

Validates documentation coverage and completeness.

**Usage:**
```bash
./validate-coverage.sh
```

**Output:**
- Report file: `/tmp/coverage-validation-report.txt`
- Exit code: 0 (complete), 1 (issues found)

**Checks:**
- INDEX.md existence
- Category coverage
- Feature documentation
- Statistics

---

### Build Scripts

#### `validate-all.sh`

Master validation script that runs all validators.

**Usage:**
```bash
./validate-all.sh
```

**Output:**
- Individual reports in `/tmp/docs-validation/`
- Combined report: `/tmp/docs-validation/combined-report.md`
- Exit code: 0 (all passed), 1 (any failures)

**Process:**
1. Runs all validation scripts
2. Collects individual reports
3. Generates combined report
4. Provides summary

**Example Output:**
```
=========================================
Documentation Validation Suite
=========================================

Running: Link Validation...
✓ Link Validation passed

Running: Front Matter Validation...
✓ Front Matter Validation passed

Running: Mermaid Diagram Validation...
✓ Mermaid Diagram Validation passed

Running: ASCII Diagram Detection...
✓ ASCII Diagram Detection passed

Running: Coverage Validation...
✓ Coverage Validation passed

=========================================
Validation Complete
=========================================

Combined report: /tmp/docs-validation/combined-report.md

✓ All validations passed
```

---

#### `generate-reports.sh`

Generates comprehensive documentation metrics and statistics.

**Usage:**
```bash
./generate-reports.sh
```

**Output:**
- Report file: `../DOCUMENTATION_METRICS.md`
- Displays report to stdout

**Metrics Generated:**
- Documents by category
- Total statistics
- Content analysis
- Health metrics
- Link density
- Tag distribution
- Quality metrics
- Recommendations

**Example Sections:**
```markdown
# Documentation Metrics Report

## Overview Statistics
- Total Documents: 45
- Total Words: 125,000
- Total Diagrams: 87
- Average Words per Document: 2,777

## Content Analysis
- Documents updated in last 30 days: 12
- Update frequency: 26% recent
- Total internal links: 234
- Average links per document: 5.2

## Quality Metrics
- Front matter completeness: 100%
- Mermaid diagram adoption: 87 diagrams
```

---

#### `generate-index.sh`

Generates the main INDEX.md navigation file.

**Usage:**
```bash
./generate-index.sh
```

**Output:**
- Creates/updates `../INDEX.md`

**Features:**
- Quick navigation links
- Documentation structure diagram
- Category listings
- Automatic file discovery
- Title extraction from front matter

---

## Environment Variables

All scripts support these environment variables:

```bash
# Documentation root directory
export DOCS_ROOT="/path/to/docs"

# Report output file (validator scripts)
export REPORT_FILE="/custom/path/report.txt"

# Project root directory (coverage script)
export PROJECT_ROOT="/path/to/project"

# Report output directory (validate-all)
export REPORT_DIR="/custom/reports"
```

**Example:**
```bash
DOCS_ROOT=/custom/docs ./validate-links.sh
REPORT_FILE=/tmp/my-report.txt ./validate-frontmatter.sh
```

---

## CI/CD Integration

### GitHub Actions

Scripts are integrated in `.github/workflows/docs-ci.yml`:

**On Pull Request:**
- Run all validation scripts
- Post results as PR comment
- Fail if validation errors found

**On Push to Main:**
- Run full validation suite
- Generate reports and metrics
- Update INDEX.md
- Commit changes

**On Schedule (Daily):**
- Run full validation
- Generate fresh reports
- Publish metrics

### Local Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
cd docs
./scripts/validate-all.sh || {
    echo "Documentation validation failed!"
    echo "Fix issues before committing."
    exit 1
}
```

---

## Troubleshooting

### Script Not Found

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Permission Denied

```bash
# Check file permissions
ls -la scripts/

# Fix if needed
chmod 755 scripts/*.sh
```

### Report Directory Issues

```bash
# Create report directory manually
mkdir -p /tmp/docs-validation
```

### Validation Failures

1. Check individual reports:
   ```bash
   cat /tmp/link-validation-report.txt
   cat /tmp/frontmatter-validation-report.txt
   ```

2. Fix identified issues

3. Re-run validation:
   ```bash
   ./validate-all.sh
   ```

---

## Best Practices

### Regular Validation

Run validation before committing:
```bash
./validate-all.sh && git commit
```

### Weekly Reports

Generate metrics weekly:
```bash
# Add to cron
0 3 * * 1 cd /path/to/docs && ./scripts/generate-reports.sh
```

### Automated Index Updates

Regenerate index after adding documents:
```bash
./generate-index.sh
git add ../INDEX.md
git commit -m "docs: update index"
```

---

## Script Dependencies

### Required Commands

All scripts require:
- `bash` (version 4+)
- `find`
- `grep`
- `awk`
- `sed`
- `wc`

### Optional Commands

For enhanced functionality:
- `git` (for commit hooks)
- `bc` (for calculations in reports)

### Verify Dependencies

```bash
# Check bash version
bash --version

# Verify all required commands
for cmd in find grep awk sed wc; do
    command -v $cmd >/dev/null || echo "Missing: $cmd"
done
```

---

## Contributing

### Adding New Validators

1. Create script in `scripts/` directory:
   ```bash
   scripts/validate-newfeature.sh
   ```

2. Follow template:
   ```bash
   #!/bin/bash
   set -euo pipefail

   DOCS_ROOT="${DOCS_ROOT:-$(dirname "$(dirname "$(realpath "$0")")")}"
   REPORT_FILE="${REPORT_FILE:-/tmp/newfeature-validation-report.txt}"
   EXIT_CODE=0

   # Validation logic here

   cat "$REPORT_FILE"
   exit $EXIT_CODE
   ```

3. Add to `validate-all.sh`:
   ```bash
   VALIDATORS=(
       # ... existing validators
       "validate-newfeature.sh|New Feature Validation"
   )
   ```

4. Add to CI/CD workflow

---

## Support

For issues with scripts:

1. Check script output and error messages
2. Review environment variables
3. Verify dependencies
4. Check file permissions
5. Consult [MAINTENANCE.md](../MAINTENANCE.md)
6. Open an issue on GitHub

---

*Last Updated: 2025-12-18*
*Version: 2.0.0*
