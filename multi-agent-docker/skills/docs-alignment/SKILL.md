---
name: "Documentation Alignment"
description: "Align documentation corpus to codebase with link validation, mermaid diagram verification, ASCII diagram conversion, and working document archival. Use when auditing documentation, ensuring docs match code, validating cross-references, or preparing for release."
---

# Documentation Alignment Skill

## Overview

This skill orchestrates a multi-agent swarm to comprehensively align a project's documentation corpus with its codebase. It validates forward and backward links, ensures all mermaid diagrams render correctly on GitHub, identifies ASCII diagrams for conversion, archives working documents, and scans for stubs, partial code blocks, and TODOs.

## Prerequisites

- Python 3.10+ with pip
- Node.js 18+ (for mermaid validation)
- Git repository with documentation in `/docs` folder
- Access to Claude Code Task tool for swarm orchestration

## What This Skill Does

1. **Link Validation**: Validates all forward and backward links between documentation files
2. **Mermaid Verification**: Ensures all mermaid diagrams render correctly on GitHub
3. **ASCII Detection**: Identifies ASCII-based diagrams requiring conversion to mermaid
4. **Archive Migration**: Moves working documents to archive directory
5. **Code Coverage**: Scans codebase for stubs, partial implementations, and TODOs
6. **Issues Report**: Generates comprehensive report of all findings

---

## Quick Start

### Single Command Execution

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Run full documentation alignment
python scripts/docs_alignment.py --project-root /path/to/project
```

### Swarm Execution (Recommended)

```bash
# Initialize swarm with mesh topology for parallel execution
npx claude-flow@alpha swarm init --topology mesh --agents 8

# Spawn specialised agents
npx claude-flow@alpha agent spawn --type researcher --name "link-validator"
npx claude-flow@alpha agent spawn --type analyst --name "mermaid-checker"
npx claude-flow@alpha agent spawn --type analyst --name "ascii-detector"
npx claude-flow@alpha agent spawn --type coder --name "archiver"
npx claude-flow@alpha agent spawn --type tester --name "stub-scanner"
npx claude-flow@alpha agent spawn --type reviewer --name "readme-integrator"
npx claude-flow@alpha agent spawn --type coordinator --name "report-generator"
npx claude-flow@alpha agent spawn --type coordinator --name "swarm-orchestrator"
```

---

## Swarm Agent Configuration

### Agent Roles and Responsibilities

| Agent | Type | Responsibility |
|-------|------|----------------|
| **link-validator** | researcher | Validate all markdown links, find broken/orphan files |
| **mermaid-checker** | analyst | Verify mermaid diagrams render on GitHub |
| **ascii-detector** | analyst | Find ASCII diagrams, suggest mermaid conversions |
| **archiver** | coder | Move working documents to archive directory |
| **stub-scanner** | tester | Find stubs, TODOs, partial implementations |
| **readme-integrator** | reviewer | Ensure README links to doc corpus |
| **report-generator** | coordinator | Compile findings into issues report |
| **swarm-orchestrator** | coordinator | Coordinate all agents, manage dependencies |

### Claude Code Task Tool Invocation

```javascript
// Spawn all agents in parallel using Claude Code's Task tool
[Single Message - Parallel Agent Execution]:
  Task("Link Validator", `
    Validate all markdown links in the docs/ directory.

    Instructions:
    1. Run: python scripts/validate_links.py --root /path/to/project
    2. Collect forward links (docs pointing to code)
    3. Collect backward links (docs pointing to other docs)
    4. Identify orphan documents (no inbound links)
    5. Identify broken links (target does not exist)
    6. Store results in memory: swarm/link-validator/results

    Expected output: JSON with broken_links, orphan_docs, valid_links arrays
  `, "researcher")

  Task("Mermaid Checker", `
    Verify all mermaid diagrams in documentation.

    Instructions:
    1. Run: python scripts/check_mermaid.py --root /path/to/project/docs
    2. Find all \`\`\`mermaid code blocks
    3. Validate syntax against mermaid.js parser
    4. Check for GitHub rendering compatibility
    5. Store results in memory: swarm/mermaid-checker/results

    Expected output: JSON with valid_diagrams, invalid_diagrams, suggestions
  `, "analyst")

  Task("ASCII Detector", `
    Find ASCII-based diagrams requiring conversion.

    Instructions:
    1. Run: python scripts/detect_ascii.py --root /path/to/project/docs
    2. Detect box-drawing characters (─│┌┐└┘├┤┬┴┼)
    3. Detect arrow patterns (-->, <--, ==>, etc.)
    4. Detect ASCII art patterns
    5. Store results in memory: swarm/ascii-detector/results

    Expected output: JSON with ascii_diagrams array containing file, line, content
  `, "analyst")

  Task("Archiver", `
    Identify and move working documents to archive.

    Instructions:
    1. Run: python scripts/archive_working_docs.py --root /path/to/project
    2. Identify working documents (WORKING_, WIP_, DRAFT_, _NOTES)
    3. Identify implementation notes not in /docs structure
    4. Create /docs/archive directory if needed
    5. Generate move commands (do NOT execute)
    6. Store results in memory: swarm/archiver/results

    Expected output: JSON with working_docs, suggested_moves
  `, "coder")

  Task("Stub Scanner", `
    Find incomplete code and documentation.

    Instructions:
    1. Run: python scripts/scan_stubs.py --root /path/to/project
    2. Find TODO comments in code and docs
    3. Find FIXME, HACK, XXX markers
    4. Find stub functions (unimplemented!(), todo!())
    5. Find partial code blocks (// ... or /* ... */)
    6. Store results in memory: swarm/stub-scanner/results

    Expected output: JSON with todos, fixmes, stubs, partial_blocks
  `, "tester")

  Task("README Integrator", `
    Verify README integration with documentation corpus.

    Instructions:
    1. Read root README.md
    2. Check all links to /docs are valid
    3. Verify documentation index matches actual structure
    4. Identify missing documentation references
    5. Store results in memory: swarm/readme-integrator/results

    Expected output: JSON with readme_links, missing_refs, structure_mismatches
  `, "reviewer")

  Task("Report Generator", `
    Compile all findings into comprehensive issues report.

    Instructions:
    1. Wait for all other agents to complete
    2. Retrieve results from memory: swarm/*/results
    3. Generate DOCUMENTATION_ISSUES.md with sections:
       - Broken Links
       - Orphan Documents
       - Invalid Mermaid Diagrams
       - ASCII Diagrams to Convert
       - Working Documents to Archive
       - Stubs and TODOs
       - README Integration Issues
    4. Include severity levels and suggested fixes
    5. Write to /docs/DOCUMENTATION_ISSUES.md
  `, "coordinator")
```

---

## Step-by-Step Guide

### Step 1: Install Dependencies

```bash
cd multi-agent-docker/skills/docs-alignment
pip install -r scripts/requirements.txt
```

### Step 2: Run Link Validation

```bash
python scripts/validate_links.py \
  --root /home/devuser/workspace/project \
  --docs-dir docs \
  --output link-report.json
```

Expected output:
```json
{
  "total_links": 456,
  "valid_links": 420,
  "broken_links": [
    {"file": "docs/guide.md", "line": 45, "link": "missing.md", "type": "internal"}
  ],
  "orphan_docs": ["docs/old-feature.md"]
}
```

### Step 3: Validate Mermaid Diagrams

```bash
python scripts/check_mermaid.py \
  --root /home/devuser/workspace/project/docs \
  --output mermaid-report.json
```

### Step 4: Detect ASCII Diagrams

```bash
python scripts/detect_ascii.py \
  --root /home/devuser/workspace/project/docs \
  --output ascii-report.json
```

### Step 5: Identify Working Documents

```bash
python scripts/archive_working_docs.py \
  --root /home/devuser/workspace/project \
  --output archive-report.json
```

### Step 6: Scan for Stubs and TODOs

```bash
python scripts/scan_stubs.py \
  --root /home/devuser/workspace/project \
  --output stubs-report.json
```

### Step 7: Generate Issues Report

```bash
python scripts/generate_report.py \
  --link-report link-report.json \
  --mermaid-report mermaid-report.json \
  --ascii-report ascii-report.json \
  --archive-report archive-report.json \
  --stubs-report stubs-report.json \
  --output DOCUMENTATION_ISSUES.md
```

---

## Script Reference

### validate_links.py

Validates all internal and external links in markdown files.

**Features:**
- Forward link validation (docs → code)
- Backward link validation (docs → docs)
- Orphan document detection
- Anchor validation (#section links)
- External URL checking (optional)

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Project root directory | `.` |
| `--docs-dir` | Documentation directory | `docs` |
| `--output` | Output JSON file | `stdout` |
| `--check-external` | Validate external URLs | `false` |
| `--ignore` | Patterns to ignore | `[]` |

### check_mermaid.py

Validates mermaid diagram syntax and GitHub compatibility.

**Features:**
- Syntax validation against mermaid.js
- GitHub rendering compatibility check
- Diagram type detection (flowchart, sequence, etc.)
- Common error detection

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Directory to scan | `.` |
| `--output` | Output JSON file | `stdout` |
| `--strict` | Fail on warnings | `false` |

### detect_ascii.py

Detects ASCII-based diagrams that should be converted to mermaid.

**Detects:**
- Box-drawing characters (─│┌┐└┘)
- Arrow patterns (-->, <--, ==>, |->)
- Table-like structures
- Tree diagrams using pipes and dashes
- Flow indicators (+, *, -, bullets with connectors)

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Directory to scan | `.` |
| `--output` | Output JSON file | `stdout` |
| `--min-lines` | Minimum lines to consider | `3` |

### archive_working_docs.py

Identifies working documents for archival.

**Patterns detected:**
- Files prefixed with WORKING_, WIP_, DRAFT_
- Files suffixed with _NOTES, _WIP, _DRAFT
- Files in /tmp, /scratch, /working directories
- Implementation notes outside /docs structure

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Project root | `.` |
| `--output` | Output JSON file | `stdout` |
| `--dry-run` | Do not move files | `true` |
| `--archive-dir` | Archive destination | `docs/archive` |

### scan_stubs.py

Scans codebase for incomplete implementations.

**Detects:**
- `TODO:` comments
- `FIXME:` markers
- `HACK:` annotations
- `XXX:` flags
- `unimplemented!()` macros (Rust)
- `todo!()` macros (Rust)
- `raise NotImplementedError` (Python)
- `// ...` or `/* ... */` placeholder patterns

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--root` | Project root | `.` |
| `--output` | Output JSON file | `stdout` |
| `--include` | File patterns to include | `*.rs,*.ts,*.py,*.md` |
| `--exclude` | Patterns to exclude | `node_modules,target,.git` |

---

## Output Format

### DOCUMENTATION_ISSUES.md Structure

```markdown
# Documentation Issues Report

Generated: YYYY-MM-DD HH:MM:SS
Project: VisionFlow

## Summary

| Category | Count | Severity |
|----------|-------|----------|
| Broken Links | 12 | High |
| Orphan Documents | 5 | Medium |
| Invalid Mermaid | 3 | Medium |
| ASCII Diagrams | 8 | Low |
| Working Documents | 4 | Low |
| Stubs/TODOs | 45 | Info |

## Broken Links

### High Priority
| File | Line | Link | Status |
|------|------|------|--------|
| docs/guide.md | 45 | ../missing.md | 404 |

## Orphan Documents

Documents with no inbound links:
- docs/deprecated/old-feature.md
- docs/analysis/unused-study.md

## Invalid Mermaid Diagrams

| File | Line | Error |
|------|------|-------|
| docs/arch.md | 120 | Syntax error: unexpected token |

## ASCII Diagrams to Convert

| File | Lines | Preview |
|------|-------|---------|
| docs/flow.md | 45-60 | `┌──────┐` |

## Working Documents to Archive

| Current Location | Suggested Archive Location |
|-----------------|---------------------------|
| docs/WORKING_notes.md | docs/archive/WORKING_notes.md |

## Stubs and TODOs

### Rust Code
| File | Line | Type | Content |
|------|------|------|---------|
| src/main.rs | 123 | TODO | Implement caching |

### TypeScript Code
| File | Line | Type | Content |
|------|------|------|---------|
| client/App.tsx | 45 | FIXME | Race condition |

### Documentation
| File | Line | Type | Content |
|------|------|------|---------|
| docs/guide.md | 78 | TODO | Add examples |
```

---

## UK English Standards

This skill enforces UK English spelling where code compatibility permits:

- colour (not color) in documentation text
- behaviour (not behavior) in documentation text
- organisation (not organization) in documentation text
- analyse (not analyze) in documentation text
- centre (not center) in documentation text

**Exceptions** (code compatibility):
- CSS properties: `color`, `background-color`
- API names: as defined in codebase
- Library references: as published

---

## Integration with Other Skills

Works well with:
- `sparc-methodology` - Use after SPARC completion phase
- `github-code-review` - Include in PR review workflow
- `verification-quality` - Add to quality gates

---

## Troubleshooting

### Issue: Python Script Not Found

**Symptoms**: `ModuleNotFoundError` when running scripts
**Solution**:
```bash
cd multi-agent-docker/skills/docs-alignment
pip install -r scripts/requirements.txt
```

### Issue: Mermaid Validation Fails

**Symptoms**: All diagrams reported as invalid
**Solution**: Ensure Node.js 18+ is installed:
```bash
node --version  # Should be 18.x or higher
npm install -g @mermaid-js/mermaid-cli
```

### Issue: Permission Denied on Archive

**Symptoms**: Cannot create archive directory
**Solution**:
```bash
mkdir -p docs/archive
chmod 755 docs/archive
```

---

## Advanced: Custom Swarm Topology

For large codebases, use hierarchical topology:

```bash
# Hierarchical with queen coordinator
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --agents 12 \
  --strategy adaptive

# Queen agent coordinates all others
npx claude-flow@alpha agent spawn \
  --type coordinator \
  --name "queen-doc-aligner" \
  --capabilities "orchestration,memory,reporting"
```

---

## Resources

- Templates: `resources/templates/`
- Example outputs: `resources/examples/`
- Configuration schemas: `resources/schemas/`

See [Advanced Configuration](docs/ADVANCED.md) for complex scenarios.
See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for common issues.

---

**Created**: 2024-12-02
**Category**: Documentation
**Difficulty**: Intermediate
**Estimated Time**: 15-45 minutes (depending on codebase size)
