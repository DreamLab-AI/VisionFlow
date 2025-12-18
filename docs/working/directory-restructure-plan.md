# Directory Restructure Plan - Implementation Details

**Version:** 1.0
**Date:** 2025-12-18
**Parent Specification:** `UNIFIED_ARCHITECTURE_SPEC.md`
**Status:** READY FOR EXECUTION

---

## Overview

This document provides the complete implementation details for restructuring the VisionFlow documentation from the current 298-file fragmented structure to the unified 7-section Diátaxis-based architecture.

---

## Phase 1: Quick Wins (Days 1-2)

### 1.1 Backup Creation

```bash
#!/bin/bash
# Step 1: Create comprehensive backup

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/home/devuser/workspace/project/docs-backup-${TIMESTAMP}"
DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Creating backup: ${BACKUP_DIR}"
cp -r "${DOCS_ROOT}" "${BACKUP_DIR}"

echo "Creating tarball"
tar -czf "${BACKUP_DIR}.tar.gz" "${DOCS_ROOT}"

echo "Backup complete:"
echo "  Directory: ${BACKUP_DIR}"
echo "  Tarball: ${BACKUP_DIR}.tar.gz"
```

### 1.2 Delete Exact Duplicates

```bash
#!/bin/bash
# Step 2: Remove exact duplicate directories

DOCS_ROOT="/home/devuser/workspace/project/docs"

# Verify duplicates before deletion
echo "Verifying duplicates..."
diff -r "${DOCS_ROOT}/concepts/architecture/core/client.md" \
        "${DOCS_ROOT}/explanations/architecture/core/client.md" || \
        echo "WARNING: Files differ!"

diff -r "${DOCS_ROOT}/concepts/architecture/core/server.md" \
        "${DOCS_ROOT}/explanations/architecture/core/server.md" || \
        echo "WARNING: Files differ!"

# Delete concepts/ directory (exact duplicate of explanations/)
echo "Deleting concepts/ directory..."
rm -rf "${DOCS_ROOT}/concepts/"

# Delete archive/data/pages/ (exact duplicate of archive/data/markdown/)
echo "Deleting archive/data/pages/ directory..."
rm -rf "${DOCS_ROOT}/archive/data/pages/"

echo "Duplicates removed"
git add -A
git commit -m "docs: remove duplicate directories (concepts, archive/data/pages)"
```

### 1.3 Standardize README Files

```bash
#!/bin/bash
# Step 3: Standardize README case and merge redundant

DOCS_ROOT="/home/devuser/workspace/project/docs"

# Standardize to uppercase README.md
echo "Standardizing README case..."
mv "${DOCS_ROOT}/guides/readme.md" "${DOCS_ROOT}/guides/README.md" 2>/dev/null || true
mv "${DOCS_ROOT}/guides/infrastructure/readme.md" "${DOCS_ROOT}/guides/infrastructure/README.md" 2>/dev/null || true
mv "${DOCS_ROOT}/guides/developer/readme.md" "${DOCS_ROOT}/guides/developer/README.md" 2>/dev/null || true
mv "${DOCS_ROOT}/explanations/architecture/gpu/readme.md" "${DOCS_ROOT}/explanations/architecture/gpu/README.md" 2>/dev/null || true
mv "${DOCS_ROOT}/reference/api/readme.md" "${DOCS_ROOT}/reference/api/README.md" 2>/dev/null || true

# Merge redundant READMEs into parent
echo "Merging redundant READMEs..."

# Append infrastructure/README.md content to guides/README.md
cat "${DOCS_ROOT}/guides/infrastructure/README.md" >> "${DOCS_ROOT}/guides/README.md"
echo "" >> "${DOCS_ROOT}/guides/README.md"
rm "${DOCS_ROOT}/guides/infrastructure/README.md"

# Append developer/README.md content to guides/README.md
cat "${DOCS_ROOT}/guides/developer/README.md" >> "${DOCS_ROOT}/guides/README.md"
echo "" >> "${DOCS_ROOT}/guides/README.md"
rm "${DOCS_ROOT}/guides/developer/README.md"

# Append gpu/readme.md content to architecture/README.md
cat "${DOCS_ROOT}/explanations/architecture/gpu/README.md" >> "${DOCS_ROOT}/explanations/architecture/README.md"
echo "" >> "${DOCS_ROOT}/explanations/architecture/README.md"
rm "${DOCS_ROOT}/explanations/architecture/gpu/README.md"

# Append api/readme.md content to reference/README.md
cat "${DOCS_ROOT}/reference/api/README.md" >> "${DOCS_ROOT}/reference/README.md"
echo "" >> "${DOCS_ROOT}/reference/README.md"
rm "${DOCS_ROOT}/reference/api/README.md"

echo "README standardization complete"
git add -A
git commit -m "docs: standardize and merge redundant README files"
```

### 1.4 Archive Completed Working Documents

```bash
#!/bin/bash
# Step 4: Move completed analyses to archive

DOCS_ROOT="/home/devuser/workspace/project/docs"

# Create archive/analysis directory
mkdir -p "${DOCS_ROOT}/archive/analysis"

# Move completed working documents
echo "Archiving completed working documents..."

mv "${DOCS_ROOT}/working/CLIENT_ARCHITECTURE_ANALYSIS.md" \
   "${DOCS_ROOT}/archive/analysis/client-architecture-2025-12.md"

mv "${DOCS_ROOT}/working/HISTORICAL_CONTEXT_RECOVERY.md" \
   "${DOCS_ROOT}/archive/analysis/historical-context-recovery-2025-12.md"

mv "${DOCS_ROOT}/working/LINK_ANALYSIS_COMPLETE.md" \
   "${DOCS_ROOT}/archive/reports/link-analysis-complete-2025-12.md"

mv "${DOCS_ROOT}/working/README-LINK-ANALYSIS.md" \
   "${DOCS_ROOT}/archive/reports/readme-link-analysis-2025-12.md"

echo "Working documents archived"
git add -A
git commit -m "docs: archive completed working documents"
```

---

## Phase 2: Consolidation (Days 3-7)

### 2.1 Create New Directory Structure

```bash
#!/bin/bash
# Step 5: Create complete new directory structure

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Creating new directory structure..."

# Getting Started (Tutorials)
mkdir -p "${DOCS_ROOT}/getting-started"

# Guides subdirectories
mkdir -p "${DOCS_ROOT}/guides/"{features,ontology,infrastructure,developer,ai-models,client}

# Explanations subdirectories
mkdir -p "${DOCS_ROOT}/explanations/architecture/"{core,components,subsystems}
mkdir -p "${DOCS_ROOT}/explanations/"{ontology,physics,patterns}

# Reference subdirectories
mkdir -p "${DOCS_ROOT}/reference/"{api,protocols,database,ontology,client}

# Architecture subdirectories
mkdir -p "${DOCS_ROOT}/architecture/"{decisions,phase-reports}

# Archive analysis subdirectory
mkdir -p "${DOCS_ROOT}/archive/analysis"

echo "Directory structure created"
```

### 2.2 Consolidate API Reference Documentation

```bash
#!/bin/bash
# Step 6: Consolidate API reference into single primary document

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Consolidating API reference documentation..."

# Create consolidated REST API reference
cat > "${DOCS_ROOT}/reference/api/rest-api-complete.md" << 'EOF'
# REST API Complete Reference

**Consolidated from:**
- API_REFERENCE.md
- api-complete-reference.md
- reference/api/rest-api-reference.md

**Last Updated:** 2025-12-18

---

## Table of Contents
1. [Authentication](#authentication)
2. [Graph Endpoints](#graph-endpoints)
3. [Ontology Endpoints](#ontology-endpoints)
4. [Settings Endpoints](#settings-endpoints)
5. [Agent Endpoints](#agent-endpoints)
6. [Error Codes](#error-codes)

---

EOF

# Append content from source files
cat "${DOCS_ROOT}/reference/API_REFERENCE.md" >> "${DOCS_ROOT}/reference/api/rest-api-complete.md"
echo -e "\n---\n" >> "${DOCS_ROOT}/reference/api/rest-api-complete.md"

cat "${DOCS_ROOT}/reference/api-complete-reference.md" >> "${DOCS_ROOT}/reference/api/rest-api-complete.md"
echo -e "\n---\n" >> "${DOCS_ROOT}/reference/api/rest-api-complete.md"

cat "${DOCS_ROOT}/reference/api/rest-api-reference.md" >> "${DOCS_ROOT}/reference/api/rest-api-complete.md"

# Remove source files
rm "${DOCS_ROOT}/reference/API_REFERENCE.md"
rm "${DOCS_ROOT}/reference/api-complete-reference.md"
rm "${DOCS_ROOT}/reference/api/rest-api-reference.md"

echo "API reference consolidated"
git add -A
git commit -m "docs: consolidate API reference into single primary document"
```

### 2.3 Consolidate WebSocket Protocol Documentation

```bash
#!/bin/bash
# Step 7: Consolidate WebSocket protocol specs

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Consolidating WebSocket protocol documentation..."

# Create consolidated binary WebSocket protocol spec
cat > "${DOCS_ROOT}/reference/protocols/binary-websocket.md" << 'EOF'
# Binary WebSocket Protocol Specification

**Consolidated from:**
- websocket-protocol.md
- reference/api/03-websocket.md

**Last Updated:** 2025-12-18

---

## Table of Contents
1. [Protocol Overview](#overview)
2. [Binary Message Format](#binary-format)
3. [Message Types](#message-types)
4. [Connection Lifecycle](#connection-lifecycle)
5. [Error Handling](#error-handling)
6. [Performance Characteristics](#performance)

---

EOF

# Append content from source files
cat "${DOCS_ROOT}/reference/websocket-protocol.md" >> "${DOCS_ROOT}/reference/protocols/binary-websocket.md"
echo -e "\n---\n" >> "${DOCS_ROOT}/reference/protocols/binary-websocket.md"

cat "${DOCS_ROOT}/reference/api/03-websocket.md" >> "${DOCS_ROOT}/reference/protocols/binary-websocket.md"

# Remove source files
rm "${DOCS_ROOT}/reference/websocket-protocol.md"
rm "${DOCS_ROOT}/reference/api/03-websocket.md"

# Keep separate (different purposes):
# - explanations/architecture/components/websocket-protocol.md (architecture)
# - diagrams/infrastructure/websocket/binary-protocol-complete.md (diagrams)
# - guides/developer/websocket-best-practices.md (guide)

echo "WebSocket protocol consolidated"
git add -A
git commit -m "docs: consolidate WebSocket protocol specifications"
```

### 2.4 Organize Guides Directory

```bash
#!/bin/bash
# Step 8: Reorganize guides into subdirectories

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Reorganizing guides directory..."

# Move to features/
mv "${DOCS_ROOT}/guides/semantic-forces-complete.md" "${DOCS_ROOT}/guides/features/semantic-forces.md"
mv "${DOCS_ROOT}/guides/physics-simulation.md" "${DOCS_ROOT}/guides/features/physics-simulation.md"
mv "${DOCS_ROOT}/guides/vircadia-xr-complete-guide.md" "${DOCS_ROOT}/guides/features/xr-immersive.md"

# Move to ontology/
mv "${DOCS_ROOT}/guides/neo4j-integration.md" "${DOCS_ROOT}/guides/ontology/neo4j-setup.md"
mv "${DOCS_ROOT}/guides/ontology-"*.md "${DOCS_ROOT}/guides/ontology/" 2>/dev/null || true

# Move to infrastructure/
mv "${DOCS_ROOT}/guides/docker-"*.md "${DOCS_ROOT}/guides/infrastructure/" 2>/dev/null || true
mv "${DOCS_ROOT}/guides/deployment.md" "${DOCS_ROOT}/guides/infrastructure/deployment.md"

# Move to developer/
mv "${DOCS_ROOT}/guides/testing-guide.md" "${DOCS_ROOT}/guides/developer/testing-guide.md"
mv "${DOCS_ROOT}/guides/developer/test-execution.md" "${DOCS_ROOT}/guides/developer/" 2>/dev/null || true
mv "${DOCS_ROOT}/guides/websocket-best-practices.md" "${DOCS_ROOT}/guides/developer/websocket-best-practices.md"
mv "${DOCS_ROOT}/guides/telemetry-logging.md" "${DOCS_ROOT}/guides/developer/telemetry-logging.md"

# Move to ai-models/
mv "${DOCS_ROOT}/guides/ai-models/"*.md "${DOCS_ROOT}/guides/ai-models/" 2>/dev/null || true

# Move to client/
mkdir -p "${DOCS_ROOT}/guides/client"
# (Create new client guides from existing content)

echo "Guides reorganized"
git add -A
git commit -m "docs: reorganize guides into topic subdirectories"
```

### 2.5 Consolidate Testing Documentation

```bash
#!/bin/bash
# Step 9: Merge testing documentation

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Consolidating testing documentation..."

# Append test-execution.md content to testing-guide.md
cat "${DOCS_ROOT}/guides/developer/test-execution.md" >> "${DOCS_ROOT}/guides/developer/testing-guide.md"

# Remove source file
rm "${DOCS_ROOT}/guides/developer/test-execution.md"

# Keep separate:
# - diagrams/infrastructure/testing/test-architecture.md (diagrams)
# - explanations/architecture/reasoning-tests-summary.md (specialized)

echo "Testing documentation consolidated"
git add -A
git commit -m "docs: consolidate testing documentation"
```

### 2.6 Consolidate Troubleshooting Documentation

```bash
#!/bin/bash
# Step 10: Merge troubleshooting guides

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Consolidating troubleshooting documentation..."

# Append infrastructure troubleshooting to main troubleshooting guide
cat "${DOCS_ROOT}/guides/infrastructure/troubleshooting.md" >> "${DOCS_ROOT}/guides/troubleshooting.md"

# Remove source file
rm "${DOCS_ROOT}/guides/infrastructure/troubleshooting.md"

echo "Troubleshooting documentation consolidated"
git add -A
git commit -m "docs: consolidate troubleshooting documentation"
```

---

## Phase 3: Major Reorganization (Days 8-15)

### 3.1 Consolidate Architecture Documentation

```bash
#!/bin/bash
# Step 11: Reorganize architecture documentation

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Consolidating architecture documentation..."

# Merge ARCHITECTURE_COMPLETE.md into ARCHITECTURE_OVERVIEW.md
cat "${DOCS_ROOT}/ARCHITECTURE_COMPLETE.md" >> "${DOCS_ROOT}/ARCHITECTURE_OVERVIEW.md"
rm "${DOCS_ROOT}/ARCHITECTURE_COMPLETE.md"

# Move subsystem docs to explanations/architecture/subsystems/
mv "${DOCS_ROOT}/architecture/blender-mcp-unified-architecture.md" \
   "${DOCS_ROOT}/explanations/architecture/subsystems/blender-mcp.md"

mv "${DOCS_ROOT}/architecture/visionflow-distributed-systems-assessment.md" \
   "${DOCS_ROOT}/explanations/architecture/subsystems/distributed-systems.md"

# Move phase reports to architecture/phase-reports/
mv "${DOCS_ROOT}/architecture/phase1-completion.md" \
   "${DOCS_ROOT}/architecture/phase-reports/phase1-completion.md"

# Move skill-related to architecture/decisions/
mv "${DOCS_ROOT}/architecture/skill-mcp-classification.md" \
   "${DOCS_ROOT}/architecture/decisions/skill-mcp-classification.md"

mv "${DOCS_ROOT}/architecture/skills-refactoring-plan.md" \
   "${DOCS_ROOT}/architecture/decisions/skills-refactoring-plan.md"

echo "Architecture documentation consolidated"
git add -A
git commit -m "docs: consolidate and organize architecture documentation"
```

### 3.2 Incorporate Archived Specialized Content

```bash
#!/bin/bash
# Step 12: Restore active content from archive

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Restoring active content from archive..."

# Move client TypeScript architecture
mv "${DOCS_ROOT}/archive/specialized/client-typescript-architecture.md" \
   "${DOCS_ROOT}/explanations/architecture/core/client-typescript.md"

# Move client components reference
mv "${DOCS_ROOT}/archive/specialized/client-components-reference.md" \
   "${DOCS_ROOT}/reference/client/components.md"

# Move ontology specialized content if still relevant
# (Manual review required for each file)

echo "Active content restored from archive"
git add -A
git commit -m "docs: restore active specialized content from archive"
```

### 3.3 Organize Ontology Documentation

```bash
#!/bin/bash
# Step 13: Organize ontology documentation into 3 sections

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Organizing ontology documentation..."

# Guides (practical)
mkdir -p "${DOCS_ROOT}/guides/ontology"
mv "${DOCS_ROOT}/guides/neo4j-integration.md" "${DOCS_ROOT}/guides/ontology/neo4j-setup.md"

# Explanations (conceptual)
mkdir -p "${DOCS_ROOT}/explanations/ontology"
# Merge hierarchical visualization
cat "${DOCS_ROOT}/explanations/architecture/hierarchical-visualization.md" \
    >> "${DOCS_ROOT}/explanations/ontology/hierarchical-visualization.md"
rm "${DOCS_ROOT}/explanations/architecture/hierarchical-visualization.md"

# Reference (specifications)
mkdir -p "${DOCS_ROOT}/reference/ontology"
# Create schema.md, api.md, data-model.md from existing content

# Keep audits separate (historical records)
# audits/neo4j-*.md remain in place

echo "Ontology documentation organized"
git add -A
git commit -m "docs: organize ontology documentation into guides/explanations/reference"
```

### 3.4 Organize Client Documentation

```bash
#!/bin/bash
# Step 14: Organize client-side documentation

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Organizing client documentation..."

# Core architecture (explanations)
# explanations/architecture/core/client.md (already exists)
# explanations/architecture/core/client-typescript.md (restored from archive)

# Reference
mkdir -p "${DOCS_ROOT}/reference/client"
# reference/client/components.md (restored from archive)

# Guides
mkdir -p "${DOCS_ROOT}/guides/client"
# Create state-management.md, three-js-rendering.md, xr-integration.md

echo "Client documentation organized"
git add -A
git commit -m "docs: organize client documentation into architecture/reference/guides"
```

### 3.5 Clean Archive Directory

```bash
#!/bin/bash
# Step 15: Review and clean archive directory

DOCS_ROOT="/home/devuser/workspace/project/docs"

echo "Cleaning archive directory..."

# Remove archive/specialized/ (content moved to active docs)
rm -rf "${DOCS_ROOT}/archive/specialized/"

# Review archive/fixes/ - keep only historical quick-references
# (Manual review for each file)

# Remove archive/docs/guides/ if content exists in current guides
# (Verify first with diff)

# Clean archive/working/ temporary files
find "${DOCS_ROOT}/archive/working/" -name "*.tmp" -delete
find "${DOCS_ROOT}/archive/working/" -name "*.bak" -delete

echo "Archive directory cleaned"
git add -A
git commit -m "docs: clean archive directory, remove superseded content"
```

---

## Phase 4: Link Updates & Validation (Days 16-18)

### 4.1 Update All Internal Links

```bash
#!/bin/bash
# Step 16: Run link update script

python3 /home/devuser/workspace/project/docs/scripts/update-links.py

git add -A
git commit -m "docs: update all internal links after restructure"
```

### 4.2 Generate Category Indexes

```bash
#!/bin/bash
# Step 17: Generate README.md indexes for all categories

python3 /home/devuser/workspace/project/docs/scripts/generate-indexes.py

git add -A
git commit -m "docs: generate category indexes"
```

### 4.3 Inject Front Matter

```bash
#!/bin/bash
# Step 18: Add front matter to all documents

python3 /home/devuser/workspace/project/docs/scripts/inject-front-matter.py --execute

git add -A
git commit -m "docs: inject front matter into all documents"
```

### 4.4 Generate Bidirectional Links

```bash
#!/bin/bash
# Step 19: Generate back-links sections

python3 /home/devuser/workspace/project/docs/scripts/generate-links.py

git add -A
git commit -m "docs: generate bidirectional links and related docs sections"
```

### 4.5 Validate Complete IA

```bash
#!/bin/bash
# Step 20: Run comprehensive validation

echo "=== Documentation IA Validation ==="

# Check orphans
echo "Checking for orphaned documents..."
python3 /home/devuser/workspace/project/docs/scripts/validate-links.py --check-orphans

# Check broken links
echo "Checking for broken links..."
python3 /home/devuser/workspace/project/docs/scripts/validate-links.py --check-broken

# Check front matter
echo "Checking front matter consistency..."
python3 /home/devuser/workspace/project/docs/scripts/validate-front-matter.py

# Check directory structure
echo "Checking directory structure..."
python3 /home/devuser/workspace/project/docs/scripts/validate-structure.py

# Generate validation report
echo "Generating validation report..."
python3 /home/devuser/workspace/project/docs/scripts/generate-validation-report.py \
  --output /home/devuser/workspace/project/docs/working/VALIDATION_REPORT.md

echo "=== Validation Complete ==="
echo "See: docs/working/VALIDATION_REPORT.md"
```

---

## Complete File Move Mapping

### Tutorials (Getting Started)

| Source | Destination | Action |
|--------|-------------|--------|
| `tutorials/01-installation.md` | `getting-started/01-installation.md` | MOVE |
| `tutorials/02-first-graph.md` | `getting-started/02-first-graph.md` | MOVE |
| `tutorials/neo4j-quick-start.md` | `getting-started/03-neo4j-quickstart.md` | MOVE |

### Guides

| Source | Destination | Action |
|--------|-------------|--------|
| `guides/semantic-forces-complete.md` | `guides/features/semantic-forces.md` | MOVE |
| `guides/physics-simulation.md` | `guides/features/physics-simulation.md` | MOVE |
| `guides/vircadia-xr-complete-guide.md` | `guides/features/xr-immersive.md` | MOVE |
| `guides/neo4j-integration.md` | `guides/ontology/neo4j-setup.md` | MOVE |
| `guides/deployment.md` | `guides/infrastructure/deployment.md` | MOVE |
| `guides/infrastructure/troubleshooting.md` | `guides/troubleshooting.md` | MERGE |
| `guides/testing-guide.md` | `guides/developer/testing-guide.md` | MOVE |
| `guides/developer/test-execution.md` | `guides/developer/testing-guide.md` | MERGE |
| `guides/websocket-best-practices.md` | `guides/developer/websocket-best-practices.md` | MOVE |
| `guides/telemetry-logging.md` | `guides/developer/telemetry-logging.md` | MOVE |

### Explanations

| Source | Destination | Action |
|--------|-------------|--------|
| `concepts/architecture/core/client.md` | DELETE | DUPLICATE |
| `concepts/architecture/core/server.md` | DELETE | DUPLICATE |
| `architecture/blender-mcp-unified-architecture.md` | `explanations/architecture/subsystems/blender-mcp.md` | MOVE |
| `explanations/architecture/hierarchical-visualization.md` | `explanations/ontology/hierarchical-visualization.md` | MERGE |
| `archive/specialized/client-typescript-architecture.md` | `explanations/architecture/core/client-typescript.md` | RESTORE |

### Reference

| Source | Destination | Action |
|--------|-------------|--------|
| `reference/API_REFERENCE.md` | `reference/api/rest-api-complete.md` | MERGE |
| `reference/api-complete-reference.md` | `reference/api/rest-api-complete.md` | MERGE |
| `reference/api/rest-api-reference.md` | `reference/api/rest-api-complete.md` | MERGE |
| `reference/websocket-protocol.md` | `reference/protocols/binary-websocket.md` | MERGE |
| `reference/api/03-websocket.md` | `reference/protocols/binary-websocket.md` | MERGE |
| `archive/specialized/client-components-reference.md` | `reference/client/components.md` | RESTORE |

### Architecture

| Source | Destination | Action |
|--------|-------------|--------|
| `ARCHITECTURE_COMPLETE.md` | `ARCHITECTURE_OVERVIEW.md` | MERGE |
| `architecture/phase1-completion.md` | `architecture/phase-reports/phase1-completion.md` | MOVE |
| `architecture/skill-mcp-classification.md` | `architecture/decisions/skill-mcp-classification.md` | MOVE |
| `architecture/skills-refactoring-plan.md` | `architecture/decisions/skills-refactoring-plan.md` | MOVE |

### Archive

| Source | Destination | Action |
|--------|-------------|--------|
| `archive/data/pages/*.md` | DELETE | DUPLICATE |
| `archive/specialized/*.md` | MOVE TO ACTIVE | RESTORE |
| `archive/fixes/quick-reference.md` | DELETE | SUPERSEDED |
| `working/CLIENT_ARCHITECTURE_ANALYSIS.md` | `archive/analysis/client-architecture-2025-12.md` | ARCHIVE |

---

## Validation Checklist

### After Each Phase

- [ ] All moved files exist at destination
- [ ] No broken references to moved files
- [ ] Git commit created with clear message
- [ ] Backup verified
- [ ] Rollback tested (on separate branch)

### After Complete Restructure

- [ ] Total file count: 220-230 (from 298)
- [ ] Orphaned files: 0 (from 124)
- [ ] Duplicate groups: 0 (from 14)
- [ ] Front matter coverage: 98%+ (from 72.5%)
- [ ] Link validity: 98%+ (from ~60%)
- [ ] Directory depth: ≤3 levels
- [ ] All categories have README.md indexes
- [ ] All documents have ≥2 inbound links
- [ ] All documents have ≥2 outbound links

---

**Document Status:** READY FOR EXECUTION
**Estimated Time:** 15-18 days (phased approach)
**Next Action:** Begin Phase 1 with backup creation
