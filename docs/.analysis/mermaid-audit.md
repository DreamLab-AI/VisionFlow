# Mermaid Diagram Audit Report

**Generated:** 2026-01-14
**Scope:** `/docs` directory
**Auditor:** Code Review Agent

---

## Summary

| Metric | Count |
|--------|-------|
| **Total Diagrams Found** | 381 |
| **Files Containing Diagrams** | 79 |
| **Valid (GitHub Compatible)** | 358 |
| **Needs Fixing** | 20 |
| **Recommend Removal** | 3 |

---

## Diagram Type Distribution

| Type | Count | GitHub Support | Notes |
|------|-------|----------------|-------|
| `graph TB/LR` | ~180 | Full | Preferred for architecture |
| `sequenceDiagram` | ~85 | Full | Used for data flows |
| `flowchart TB/LR` | ~45 | Full | Similar to graph |
| `stateDiagram-v2` | 18 | Partial | Notes may not render |
| `classDiagram` | ~25 | Full | Used for types/interfaces |
| `mindmap` | 3 | Limited | May timeout on GitHub |
| `gantt` | 11 | Full | Timeline diagrams |
| `erDiagram` | 11 | Full | Database schemas |

---

## Diagrams Needing Fixes

### Critical Issues (Syntax/Rendering Failures)

| File | Line | Issue | Fix Required |
|------|------|-------|--------------|
| `diagrams/data-flow/complete-data-flows.md` | Multiple | 456 `note` blocks - excessive notes in sequence diagrams may cause timeout | Reduce notes per diagram to <20 |
| `diagrams/infrastructure/websocket/binary-protocol-complete.md` | Multiple | 57 `note` blocks - high note density | Split into multiple diagrams |
| `diagrams/server/actors/actor-system-complete.md` | Multiple | 40 `note` blocks + 4 stateDiagram-v2 with complex notes | Simplify state diagram notes |

### Minor Issues (May Render Incorrectly)

| File | Line | Issue | Fix Required |
|------|------|-------|--------------|
| `explanations/architecture/hexagonal-cqrs.md` | 200 | Uses `mindmap` diagram type | Convert to `graph TB` or accept limited support |
| `diagrams/client/rendering/threejs-pipeline-complete.md` | ~967 | Uses `mindmap` for Performance Optimizations | Convert to `graph TB` hierarchy |
| `diagrams/infrastructure/gpu/cuda-architecture-complete.md` | Various | 3 notes in stateDiagram contexts | Verify rendering |
| `explanations/architecture/pipeline-sequence-diagrams.md` | Multiple | 30 `note` blocks in sequence diagrams | Consider splitting |
| `architecture/solid-sidecar-architecture.md` | Various | 5 `note` blocks in state diagrams | Verify rendering |
| `guides/infrastructure/docker-environment.md` | Various | 38 `note` blocks - very high density | Split diagrams |
| `diagrams/mermaid-library/02-data-flow-diagrams.md` | Various | 13 `note` blocks | Acceptable but monitor |
| `diagrams/mermaid-library/04-agent-orchestration.md` | Various | 7 `note` blocks | Acceptable |

### `stateDiagram-v2` Notes Compatibility

Files with stateDiagram-v2 containing `note` blocks (may not render on GitHub):

| File | stateDiagram Count | Notes in States |
|------|-------------------|-----------------|
| `diagrams/server/actors/actor-system-complete.md` | 4 | Yes - complex |
| `diagrams/infrastructure/websocket/binary-protocol-complete.md` | 1 | Yes |
| `diagrams/client/xr/xr-architecture-complete.md` | 1 | Minimal |
| `diagrams/architecture/backend-api-architecture-complete.md` | 1 | Minimal |
| `diagrams/client/rendering/threejs-pipeline-complete.md` | 1 | Minimal |

---

## Complex Diagrams to Simplify

These diagrams exceed 100 nodes or have high complexity that may cause slow rendering:

| File | Diagram Count | Estimated Nodes | Recommendation |
|------|--------------|-----------------|----------------|
| `diagrams/data-flow/complete-data-flows.md` | 10 | 200+ per diagram | Split into 3-4 focused files |
| `diagrams/infrastructure/gpu/cuda-architecture-complete.md` | 26 | 100-150 per diagram | Acceptable - well-organized with subgraphs |
| `diagrams/server/actors/actor-system-complete.md` | 23 | 80-120 per diagram | Acceptable - good subgraph organization |
| `diagrams/client/rendering/threejs-pipeline-complete.md` | 24 | 60-100 per diagram | Acceptable |
| `diagrams/infrastructure/websocket/binary-protocol-complete.md` | 19 | 50-80 per diagram | Acceptable |

---

## Diagrams to Remove

| File | Reason | Alternative |
|------|--------|-------------|
| `docs/scripts/validate-mermaid.sh` | Contains mermaid in bash script heredoc - not a diagram file | N/A - script file |
| `docs/scripts/generate-reports.sh` | Contains mermaid reference in script | N/A - script file |
| `docs/scripts/generate-index.sh` | Contains mermaid reference in script | N/A - script file |

**Note:** These are false positives from the grep search - they reference mermaid syntax but are shell scripts, not documentation.

---

## `mindmap` Diagram Locations

GitHub has limited support for `mindmap` diagrams. Consider converting to `graph TB`:

| File | Description |
|------|-------------|
| `explanations/architecture/hexagonal-cqrs.md` | Architecture overview mindmap |
| `diagrams/client/rendering/threejs-pipeline-complete.md` | Performance optimizations mindmap |
| N/A | (reports/diagram-audit.md is an existing audit file) |

**Recommendation:** Convert mindmaps to hierarchical `graph TB` diagrams for better compatibility.

---

## `gantt` Chart Locations

All gantt charts should render correctly on GitHub:

| File | Purpose |
|------|---------|
| `guides/semantic-features-implementation.md` | Implementation timeline |
| `guides/neo4j-implementation-roadmap.md` | Migration roadmap |
| `explanations/system-overview.md` | System phases |
| `explanations/architecture/data-flow-complete.md` | Data flow phases |
| `diagrams/mermaid-library/00-mermaid-style-guide.md` | Style guide examples |
| `diagrams/mermaid-library/README.md` | Library examples |
| `diagrams/server/actors/actor-system-complete.md` | Actor timing diagrams |
| `diagrams/infrastructure/gpu/cuda-architecture-complete.md` | GPU kernel execution timeline |
| `docs/scripts/validate-mermaid.sh` | (Script reference - not a diagram) |
| `docs/scripts/README.md` | (Script docs - if applicable) |

---

## Style Guide Compliance

The project has a comprehensive style guide at `docs/diagrams/mermaid-library/00-mermaid-style-guide.md`.

### Compliance Summary

| Criterion | Status | Notes |
|-----------|--------|-------|
| Consistent color scheme | Good | Most diagrams use defined palette |
| Node label format | Good | `<br/>` used correctly |
| Subgraph organization | Good | Logical grouping |
| Edge labels | Good | Descriptive labels |
| Complexity limits | Needs Work | Some diagrams exceed 100 nodes |

---

## High-Value Diagram Files

These files contain the most comprehensive and well-structured diagrams:

1. **`diagrams/server/actors/actor-system-complete.md`** - 23 diagrams covering complete actor system
2. **`diagrams/infrastructure/gpu/cuda-architecture-complete.md`** - 26 diagrams for GPU/CUDA architecture
3. **`diagrams/client/rendering/threejs-pipeline-complete.md`** - 24 diagrams for Three.js rendering
4. **`diagrams/infrastructure/websocket/binary-protocol-complete.md`** - 19 diagrams for WebSocket protocol
5. **`diagrams/data-flow/complete-data-flows.md`** - 10 comprehensive data flow diagrams
6. **`ARCHITECTURE_OVERVIEW.md`** - 16 diagrams for system overview
7. **`explanations/architecture/hexagonal-cqrs.md`** - 13 diagrams for CQRS architecture
8. **`explanations/architecture/core/client.md`** - 14 diagrams for client architecture

---

## Recommended Actions

### Priority 1: Fix Critical Issues
1. **Split `diagrams/data-flow/complete-data-flows.md`** into focused sections to reduce note density
2. **Reduce notes** in `diagrams/infrastructure/websocket/binary-protocol-complete.md`
3. **Simplify state diagram notes** in `diagrams/server/actors/actor-system-complete.md`

### Priority 2: Improve Compatibility
1. **Convert mindmaps to graph TB** in:
   - `explanations/architecture/hexagonal-cqrs.md`
   - `diagrams/client/rendering/threejs-pipeline-complete.md`

### Priority 3: General Improvements
1. Review diagrams with >50 notes for potential simplification
2. Ensure all stateDiagram-v2 diagrams render correctly on GitHub
3. Add fallback descriptions for complex diagrams

---

## Validation Commands

```bash
# Install mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Validate all diagrams
find docs -name "*.md" -exec mmdc -i {} -o /dev/null 2>&1 \; | grep -i error

# Generate SVG previews
mmdc -i docs/diagrams/server/actors/actor-system-complete.md -o /tmp/preview.svg

# Count diagrams per file
grep -c '```mermaid' docs/**/*.md | sort -t: -k2 -nr | head -20
```

---

## Existing Validation Infrastructure

The project already has validation scripts:
- `docs/scripts/validate-mermaid.sh` - Mermaid validation script
- `docs/scripts/validate-coverage.sh` - Coverage validation

---

**Audit Status:** Complete
**Next Review:** When making significant diagram changes
