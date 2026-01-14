# Link Validation Report

**Generated**: 2026-01-14
**Scope**: `/home/devuser/workspace/project/docs`

## Summary

| Metric | Count |
|--------|-------|
| Total Markdown Files | 280 |
| Total Internal Links | 2904 |
| Valid Links | 2538 |
| Broken Links | 366 |
| Orphaned Files | 61 |
| Link Health | 87.4% |

## Broken Links by Category

| Category | Count | Fix Strategy |
|----------|-------|--------------|
| Archive References | 49 | Remove or redirect to reports/ |
| Other/Various | 241 | Manual review required |
| Case Sensitivity (readme.md) | 25 | Change to README.md |
| Broken Anchors | 21 | Update or remove anchor |
| Doubled docs/ Prefix | 13 | Remove docs/ prefix |
| DeepSeek Location | 10 | Move to guides/ai-models/ |
| Getting Started | 7 | Change to tutorials/ |

## Critical Broken Links to Fix

### 1. Case Sensitivity Issues (25 links)

These links use lowercase `readme.md` but should use uppercase `README.md`:

| Source File | Broken Link | Correct Target |
|-------------|-------------|----------------|
| README.md | explanations/architecture/gpu/readme.md | explanations/architecture/gpu/README.md |
| README.md | reference/api/readme.md | reference/api/README.md |
| INDEX.md | explanations/architecture/gpu/readme.md | explanations/architecture/gpu/README.md |
| INDEX.md | reference/api/readme.md | reference/api/README.md |
| QUICK_NAVIGATION.md | explanations/architecture/gpu/readme.md | explanations/architecture/gpu/README.md |
| QUICK_NAVIGATION.md | reference/api/readme.md | reference/api/README.md |
| QUICK_NAVIGATION.md | guides/readme.md | guides/README.md |
| guides/navigation-guide.md | guides/readme.md | guides/README.md |
| guides/ai-models/README.md | guides/readme.md | guides/README.md |
| diagrams/mermaid-library/README.md | guides/readme.md | guides/README.md |

### 2. Doubled docs/ Prefix (13 links)

These links incorrectly include a `docs/` prefix:

| Source File | Broken Link | Correct Target |
|-------------|-------------|----------------|
| ARCHITECTURE_COMPLETE.md | docs/diagrams/architecture/backend-api-architecture-complete.md | diagrams/architecture/backend-api-architecture-complete.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/client/rendering/threejs-pipeline-complete.md | diagrams/client/rendering/threejs-pipeline-complete.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/client/state/state-management-complete.md | diagrams/client/state/state-management-complete.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/client/xr/xr-architecture-complete.md | diagrams/client/xr/xr-architecture-complete.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/server/actors/actor-system-complete.md | diagrams/server/actors/actor-system-complete.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/server/agents/agent-system-architecture.md | diagrams/server/agents/agent-system-architecture.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/server/api/rest-api-architecture.md | diagrams/server/api/rest-api-architecture.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/infrastructure/websocket/binary-protocol-complete.md | diagrams/infrastructure/websocket/binary-protocol-complete.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/infrastructure/gpu/cuda-architecture-complete.md | diagrams/infrastructure/gpu/cuda-architecture-complete.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/infrastructure/testing/test-architecture.md | diagrams/infrastructure/testing/test-architecture.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/data-flow/complete-data-flows.md | diagrams/data-flow/complete-data-flows.md |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/cross-reference-matrix.md | diagrams/cross-reference-matrix.md |

### 3. Getting Started to Tutorials (7 links)

The `getting-started/` directory was renamed to `tutorials/`:

| Source File | Broken Link | Correct Target |
|-------------|-------------|----------------|
| OVERVIEW.md | getting-started/01-installation.md | tutorials/01-installation.md |
| OVERVIEW.md | getting-started/02-first-graph-and-agents.md | tutorials/02-first-graph.md |

### 4. DeepSeek Location (10 links)

DeepSeek docs should be in `guides/ai-models/`:

| Source File | Broken Link | Correct Target |
|-------------|-------------|----------------|
| README.md | guides/features/deepseek-verification.md | guides/ai-models/deepseek-verification.md |
| README.md | guides/features/deepseek-deployment.md | guides/ai-models/deepseek-deployment.md |
| INDEX.md | guides/features/deepseek-verification.md | guides/ai-models/deepseek-verification.md |
| INDEX.md | guides/features/deepseek-deployment.md | guides/ai-models/deepseek-deployment.md |
| QUICK_NAVIGATION.md | guides/features/deepseek-verification.md | guides/ai-models/deepseek-verification.md |
| QUICK_NAVIGATION.md | guides/features/deepseek-deployment.md | guides/ai-models/deepseek-deployment.md |

### 5. Archive References (49 links)

These reference an `archive/` directory that no longer exists. Options:
1. Create archive directory structure
2. Remove these links
3. Redirect to reports/ directory

| Source File | Broken Link | Action |
|-------------|-------------|--------|
| README.md | archive/README.md | Remove or create |
| README.md | archive/reports/ | Remove or redirect to reports/ |
| audits/README.md | ../archive/reports/ARCHIVE_INDEX.md | Remove |
| audits/README.md | ../archive/reports/README.md | Remove |
| audits/README.md | ../archive/reports/documentation-alignment-2025-12-02/DOCUMENTATION_ALIGNMENT_COMPLETE.md | Remove |
| audits/README.md | ../archive/reports/documentation-alignment-2025-12-02/DEEPSEEK_SETUP_COMPLETE.md | Remove |
| QA_VALIDATION_FINAL.md | archive/reports/2025-12-02-user-settings-summary.md | Remove |
| QA_VALIDATION_FINAL.md | archive/reports/mermaid-fixes-examples.md | Remove |
| comfyui-integration-design.md | archive/analysis/client-architecture-analysis-2025-12.md | Remove |
| (40 more...) | | |

### 6. Broken Anchor Links (21 links)

These internal anchors do not exist in target files:

| Source File | Broken Link | Issue |
|-------------|-------------|-------|
| PROJECT_CONSOLIDATION_PLAN.md | #2-phase-0-critical-fixes | Anchor not found |
| ARCHITECTURE_OVERVIEW.md | README.md#roadmap | Anchor #roadmap not found |
| NAVIGATION.md | #quick-answers | Anchor not found (2x) |
| DEVELOPER_JOURNEY.md | guides/troubleshooting.md#faq | Anchor #faq not found |
| GETTING_STARTED_WITH_UNIFIED_DOCS.md | ./INDEX.md#faq | Anchor #faq not found |
| INDEX.md | #specialized-content | Anchor not found |
| 01-GETTING_STARTED.md | #devops | Anchor not found |
| (14 more...) | | |

## Orphaned Files (Never Referenced)

These 61 files are not linked from any other document:

### High-Priority Orphans (Reports/Analysis)
| File | Content Summary | Recommended Action |
|------|-----------------|-------------------|
| reports/navigation-design.md | Navigation design proposals | Link from guides/README.md |
| reports/consolidation-plan.md | Documentation consolidation | Link from INDEX.md |
| reports/content-audit.md | Content audit results | Link from reports/README.md |
| reports/diagram-audit.md | Diagram audit results | Link from reports/README.md |
| reports/spelling-audit.md | Spelling audit results | Link from reports/README.md |
| reports/link-validation.md | Link validation results | Link from reports/README.md |
| reports/corpus-analysis.md | Corpus analysis | Link from reports/README.md |
| testing/TESTING_GUIDE.md | Testing documentation | Link from guides/testing-guide.md |

### Architecture Orphans
| File | Content Summary | Recommended Action |
|------|-----------------|-------------------|
| architecture/PROTOCOL_MATRIX.md | Protocol comparison matrix | Link from reference/PROTOCOL_REFERENCE.md |
| architecture/skills-refactoring-plan.md | Skills refactoring | Link from guides/multi-agent-skills.md |
| architecture/user-agent-pod-design.md | User agent design | Link from architecture docs |

### Research Orphans
| File | Content Summary | Recommended Action |
|------|-----------------|-------------------|
| research/QUIC_HTTP3_ANALYSIS.md | QUIC/HTTP3 analysis | Link from technology docs |
| research/graph-visualization-sota-analysis.md | Graph viz research | Link from explanations/ |
| research/threejs-vs-babylonjs-graph-visualization.md | Renderer comparison | Link from client docs |

### Other Orphans
- API_TEST_IMPLEMENTATION.md
- CLIENT_CODE_ANALYSIS.md
- CODE_QUALITY_ANALYSIS.md
- CUDA_KERNEL_ANALYSIS_REPORT.md
- CUDA_KERNEL_AUDIT_REPORT.md
- CUDA_OPTIMIZATION_SUMMARY.md
- GETTING_STARTED_WITH_UNIFIED_DOCS.md
- PROJECT_CONSOLIDATION_PLAN.md
- QUICK_NAVIGATION.md
- SOLID_POD_CREATION.md
- TEST_COVERAGE_ANALYSIS.md
- VALIDATION_CHECKLIST.md
- VISIONFLOW_WARDLEY_ANALYSIS.md
- analysis/DUAL_RENDERER_OVERHEAD_ANALYSIS.md
- architecture/VIRCADIA_BABYLON_CONSOLIDATION_ANALYSIS.md
- architecture/phase1-completion.md
- architecture/skill-mcp-classification.md
- architecture_analysis_report.md
- code-quality-analysis-report.md
- conversion-report.md
- diagrams/ASCII-TO-MERMAID-CONVERSION-REPORT.md
- explanations/architecture/event-driven-architecture.md
- guides/features/MOVED.md
- observability-analysis.md
- phase6-integration-guide.md
- phase6-multiuser-sync-implementation.md
- phase7_broadcast_optimization.md
- phase7_implementation_summary.md
- refactoring_guide.md
- reference/api/API_DESIGN_ANALYSIS.md
- reference/api/API_IMPROVEMENT_TEMPLATES.md
- reports/CODE_COVERAGE_INDEX.md
- reports/CONTENT-AUDIT-QUICK-REFERENCE.md
- reports/DIAGRAM_AUDIT_SUMMARY.md
- reports/LINK_FIX_CHECKLIST.md
- reports/LINK_VALIDATION_SUMMARY.md
- reports/README-AUDIT.md
- reports/UNDOCUMENTED_COMPONENTS.md
- reports/ascii-conversion-archive-batch-report.md
- reports/code-coverage.md
- reports/diataxis-compliance-final-report.md
- reports/frontmatter-quick-reference.md
- reports/frontmatter-remediation-action-items.md
- reports/frontmatter-validation.md
- reports/ia-proposal.md
- scripts/AUTOMATION_COMPLETE.md
- testing/PHASE8_COMPLETION.md

## Link Pattern Issues

### 1. Inconsistent Relative Paths
Some files use `./` prefix while others don't:
- `./neo4j-migration-summary.md` vs `neo4j-migration-summary.md`

### 2. Missing README Links
Several directories have README.md files that aren't linked:
- `guides/user/README.md` (directory doesn't exist)
- `guides/getting-started/` (directory doesn't exist)

### 3. Cross-Directory References
Several files reference paths outside docs/:
- `../deployment/docker-deployment.md` (from CONTRIBUTION.md)
- `../api/rest-api.md` (from CONTRIBUTION.md)
- `../reference/configuration.md` (from CONTRIBUTION.md)

## Repair Script

```bash
#!/bin/bash
# Link Repair Script for VisionFlow Documentation
# Run from /home/devuser/workspace/project/docs

set -e

# 1. Fix case sensitivity (readme.md -> README.md)
echo "Fixing case sensitivity issues..."
find . -name "*.md" -exec sed -i 's|/readme\.md|/README.md|g' {} \;
find . -name "*.md" -exec sed -i 's|(readme\.md)|(README.md)|g' {} \;

# 2. Fix doubled docs/ prefix
echo "Fixing doubled docs/ prefix..."
sed -i 's|(docs/diagrams/|(diagrams/|g' ARCHITECTURE_COMPLETE.md

# 3. Fix getting-started -> tutorials
echo "Fixing getting-started references..."
find . -name "*.md" -exec sed -i 's|getting-started/01-installation\.md|tutorials/01-installation.md|g' {} \;
find . -name "*.md" -exec sed -i 's|getting-started/02-first-graph-and-agents\.md|tutorials/02-first-graph.md|g' {} \;
find . -name "*.md" -exec sed -i 's|getting-started/02-first-graph\.md|tutorials/02-first-graph.md|g' {} \;

# 4. Fix DeepSeek location
echo "Fixing DeepSeek references..."
find . -name "*.md" -exec sed -i 's|guides/features/deepseek-|guides/ai-models/deepseek-|g' {} \;

# 5. Remove archive references (manual review recommended)
echo "Archive references need manual review - see broken links list"

# 6. Remove duplicate README files with wrong case
if [ -f guides/infrastructure/readme.md ] && [ -f guides/infrastructure/README.md ]; then
    echo "Found duplicate readme files in guides/infrastructure/"
fi
if [ -f guides/developer/readme.md ] && [ -f guides/developer/README.md ]; then
    echo "Found duplicate readme files in guides/developer/"
fi

echo "Link repair complete. Run validation again to verify."
```

## Next Steps

1. **Immediate**: Run the repair script to fix automatable issues (85 links)
2. **Short-term**: Review and remove/update archive references (49 links)
3. **Short-term**: Fix broken anchor links by updating headings (21 links)
4. **Medium-term**: Link orphaned files from appropriate parent documents (61 files)
5. **Medium-term**: Review "other" category for additional patterns (241 links)
6. **Ongoing**: Add link validation to CI/CD pipeline

## Files Requiring Most Attention

| File | Broken Links | Priority |
|------|--------------|----------|
| reports/navigation-design.md | 76 | High (orphaned + broken) |
| guides/navigation-guide.md | 18 | High |
| guides/ai-models/README.md | 15 | High |
| ARCHITECTURE_COMPLETE.md | 13 | High |
| QUICK_NAVIGATION.md | 13 | Medium |
| reference/INDEX.md | 11 | Medium |
| diagrams/mermaid-library/README.md | 10 | Medium |
| README.md | 7 | Critical (entry point) |
## Complete Broken Links Table

| Source File | Broken Link | Reason |
|-------------|-------------|--------|
| README.md | guides/features/deepseek-verification.md | File not found |
| README.md | guides/features/deepseek-deployment.md | File not found |
| README.md | explanations/architecture/gpu/readme.md | File not found |
| README.md | reference/api/readme.md | File not found |
| README.md | explanations/architecture/gpu/readme.md | File not found |
| README.md | archive/README.md | File not found |
| README.md | archive/reports/ | File not found |
| PROJECT_CONSOLIDATION_PLAN.md | #2-phase-0-critical-fixes | Anchor not found |
| ARCHITECTURE_OVERVIEW.md | README.md#roadmap | Anchor #roadmap not found in file |
| CONTRIBUTION.md | ../deployment/docker-deployment.md | File not found |
| CONTRIBUTION.md | /docs/deployment/docker-deployment.md | File not found |
| CONTRIBUTION.md | ../deployment/docker-deployment.md#configuration | File not found |
| CONTRIBUTION.md | ../api/rest-api.md | File not found |
| CONTRIBUTION.md | ../reference/configuration.md | File not found |
| CONTRIBUTION.md | ../guides/troubleshooting.md | File not found |
| OVERVIEW.md | getting-started/01-installation.md | File not found |
| OVERVIEW.md | getting-started/02-first-graph-and-agents.md | File not found |
| OVERVIEW.md | getting-started/01-installation.md | File not found |
| gpu-fix-summary.md | multi-agent-docker/development-notes/SESSION_2025-11-15.md | File not found |
| MAINTENANCE.md | ../related-category/related-doc.md | File not found |
| MAINTENANCE.md | ../new-category/new-doc.md | File not found |
| QA_VALIDATION_FINAL.md | archive/reports/2025-12-02-user-settings-summary.md | File not found |
| QA_VALIDATION_FINAL.md | archive/reports/mermaid-fixes-examples.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/architecture/backend-api-architecture-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/client/rendering/threejs-pipeline-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/client/state/state-management-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/client/xr/xr-architecture-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/server/actors/actor-system-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/server/agents/agent-system-architecture.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/server/api/rest-api-architecture.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/infrastructure/websocket/binary-protocol-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/infrastructure/gpu/cuda-architecture-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/infrastructure/database/neo4j-architecture-complete.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/infrastructure/testing/test-architecture.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/data-flow/complete-data-flows.md | File not found |
| ARCHITECTURE_COMPLETE.md | docs/diagrams/cross-reference-matrix.md | File not found |
| NAVIGATION.md | #quick-answers | Anchor not found |
| NAVIGATION.md | #quick-answers | Anchor not found |
| SOLID_POD_CREATION.md | /home/devuser/.claude/plans/composed-singing-stallman.md | File not found |
| SOLID_POD_CREATION.md | ./NOSTR_AUTH.md | File not found |
| DEVELOPER_JOURNEY.md | guides/troubleshooting.md#faq | Anchor #faq not found in file |
| comfyui-integration-design.md | archive/analysis/client-architecture-analysis-2025-12.md | File not found |
| GETTING_STARTED_WITH_UNIFIED_DOCS.md | ./guides/user/README.md | File not found |
| GETTING_STARTED_WITH_UNIFIED_DOCS.md | ./INDEX.md#faq | Anchor #faq not found in file |
| INDEX.md | #specialized-content | Anchor not found |
| INDEX.md | guides/features/deepseek-verification.md | File not found |
| INDEX.md | guides/features/deepseek-deployment.md | File not found |
| INDEX.md | explanations/architecture/gpu/readme.md | File not found |
| INDEX.md | reference/api/readme.md | File not found |
| 01-GETTING_STARTED.md | #devops | Anchor not found |
| 01-GETTING_STARTED.md | guides/getting-started/GETTING_STARTED_USER.md | File not found |
| 01-GETTING_STARTED.md | guides/getting-started/GETTING_STARTED_DEVELOPER.md | File not found |
| 01-GETTING_STARTED.md | guides/getting-started/GETTING_STARTED_ARCHITECT.md | File not found |
| 01-GETTING_STARTED.md | guides/getting-started/GETTING_STARTED_OPERATOR.md | File not found |
| QUICK_NAVIGATION.md | MERMAID_FIXES_STATS.json | File not found |
| QUICK_NAVIGATION.md | guides/readme.md | File not found |
| QUICK_NAVIGATION.md | guides/features/deepseek-deployment.md | File not found |
| QUICK_NAVIGATION.md | guides/features/deepseek-verification.md | File not found |
| QUICK_NAVIGATION.md | explanations/architecture/gpu/readme.md | File not found |
| QUICK_NAVIGATION.md | reference/api/readme.md | File not found |
| QUICK_NAVIGATION.md | archive/README.md | File not found |
| QUICK_NAVIGATION.md | working/CLIENT_ARCHITECTURE_ANALYSIS.md | File not found |
| QUICK_NAVIGATION.md | working/CLIENT_DOCS_SUMMARY.md | File not found |
| QUICK_NAVIGATION.md | working/DEPRECATION_PURGE.md | File not found |
| QUICK_NAVIGATION.md | working/DEPRECATION_PURGE_COMPLETE.md | File not found |
| QUICK_NAVIGATION.md | working/DOCS_ROOT_CLEANUP.md | File not found |
| QUICK_NAVIGATION.md | working/HISTORICAL_CONTEXT_RECOVERY.md | File not found |
| ASCII_DEPRECATION_COMPLETE.md | DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| audits/README.md | ../archive/reports/ARCHIVE_INDEX.md | File not found |
| audits/README.md | ../archive/reports/README.md | File not found |
| audits/README.md | ../archive/reports/documentation-alignment-2025-12-02/DOCUMENTATION_ALIGNMENT_COMPLETE.md | File not found |
| audits/README.md | ../archive/reports/documentation-alignment-2025-12-02/DEEPSEEK_SETUP_COMPLETE.md | File not found |
| audits/ascii-diagram-deprecation-audit.md | ../../diagrams/data-flow/complete-data-flows.md | File not found |
| audits/ascii-diagram-deprecation-audit.md | ../archive/analysis/client-architecture-analysis-2025-12.md | File not found |
| reference/README.md | ./API_REFERENCE.md#authentication--authorization | Anchor #authentication--authorization not found in file |
| reference/README.md | ./API_REFERENCE.md#section-name | Anchor #section-name not found in file |
| reference/implementation-status.md | ../../explanations/architecture/system-overview.md | File not found |
| reference/API_REFERENCE.md | #authentication--authorization | Anchor not found |
| reference/PROTOCOL_REFERENCE.md | ./CONFIGURATION_REFERENCE.md#solid-integration | Anchor #solid-integration not found in file |
| reference/DATABASE_SCHEMA_REFERENCE.md | #relationships--foreign-keys | Anchor not found |
| reference/DATABASE_SCHEMA_REFERENCE.md | #indexes--performance | Anchor not found |
| reference/api-complete-reference.md | ../../explanations/architecture/system-overview.md | File not found |
| reference/api-complete-reference.md | ../../tutorials/01-installation.md | File not found |
| reference/code-quality-status.md | ../guides/developer/05-testing-guide.md | File not found |
| reference/code-quality-status.md | ../../explanations/architecture/system-overview.md | File not found |
| reference/performance-benchmarks.md | ./binary-websocket.md | File not found |
| reference/performance-benchmarks.md | ./reference/api/03-websocket.md | File not found |
| reference/INDEX.md | ./CONFIGURATION_REFERENCE.md#agent-management | Anchor #agent-management not found in file |
| reference/INDEX.md | ./API_REFERENCE.md#authentication--authorization | Anchor #authentication--authorization not found in file |
| reference/INDEX.md | ./DATABASE_SCHEMA_REFERENCE.md#relationships--foreign-keys | Anchor #relationships--foreign-keys not found in file |
| reference/INDEX.md | ./DATABASE_SCHEMA_REFERENCE.md#indexes--performance | Anchor #indexes--performance not found in file |
| reference/INDEX.md | ./CONFIGURATION_REFERENCE.md#logging--monitoring | Anchor #logging--monitoring not found in file |
| reference/INDEX.md | ./DATABASE_SCHEMA_REFERENCE.md#relationships--foreign-keys | Anchor #relationships--foreign-keys not found in file |
| reference/INDEX.md | ./ERROR_REFERENCE.md#common-issues--solutions | Anchor #common-issues--solutions not found in file |
| reference/INDEX.md | ./ERROR_REFERENCE.md#common-issues--solutions | Anchor #common-issues--solutions not found in file |
| reference/INDEX.md | ./API_REFERENCE.md#authentication--authorization | Anchor #authentication--authorization not found in file |
| reference/INDEX.md | ./ERROR_REFERENCE.md#common-issues--solutions | Anchor #common-issues--solutions not found in file |
| reference/INDEX.md | ./API_REFERENCE.md#startrstop-simulation | Anchor #startrstop-simulation not found in file |
| reference/ERROR_REFERENCE.md | #common-issues--solutions | Anchor not found |
| multi-agent-docker/x-fluxagent-adaptation-plan.md | ../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| multi-agent-docker/TERMINAL_GRID.md | development-notes/SESSION_2025-11-15.md | File not found |
| multi-agent-docker/TERMINAL_GRID.md | fixes/GPU_BUILD_STATUS.md | File not found |
| multi-agent-docker/upstream-analysis.md | development-notes/SESSION_2025-11-15.md | File not found |
| multi-agent-docker/SKILLS.md | ../archive/docs/guides/user/working-with-agents.md | File not found |
| multi-agent-docker/SKILLS.md | ../archive/docs/guides/developer/05-testing-guide.md | File not found |
| multi-agent-docker/comfyui-sam3d-setup.md | ../archive/docs/guides/developer/05-testing-guide.md | File not found |
| multi-agent-docker/ANTIGRAVITY.md | fixes/SUMMARY.md | File not found |
| multi-agent-docker/hyprland-migration-summary.md | development-notes/SESSION_2025-11-15.md | File not found |
| multi-agent-docker/hyprland-migration-summary.md | fixes/GPU_BUILD_STATUS.md | File not found |
| analysis/ontology-skills-cluster-analysis.md | ../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| explanations/system-overview.md | ./schemas.md | File not found |
| explanations/system-overview.md | ../reasoning-engine.md | File not found |
| explanations/system-overview.md | ../archive/reports/2025-12-02-user-settings-summary.md | File not found |
| explanations/system-overview.md | ../archive/reports/2025-12-02-restructuring-complete.md | File not found |
| diagrams/README.md | infrastructure/database/neo4j-architecture-complete.md | File not found |
| diagrams/README.md | infrastructure/database/neo4j-architecture-complete.md | File not found |
| diagrams/cross-reference-matrix.md | ../archive/analysis/client-architecture-analysis-2025-12.md | File not found |
| reports/ascii-conversion-archive-batch-report.md | ascii-to-mermaid-conversion.md | File not found |
| reports/ascii-conversion-archive-batch-report.md | ascii-conversion-report.md | File not found |
| reports/ascii-conversion-archive-batch-report.md | documentation-audit-final.md | File not found |
| reports/consolidation-plan.md | ./api/rest-api-complete.md | File not found |
| reports/consolidation-plan.md | ./API_REFERENCE.md#rest-api-endpoints | File not found |
| reports/consolidation-plan.md | ./INDEX.md | File not found |
| reports/consolidation-plan.md | ./CONFIGURATION_REFERENCE.md | File not found |
| reports/consolidation-plan.md | ./ERROR_REFERENCE.md | File not found |
| reports/navigation-design.md | ../INDEX.md#-new-users | Anchor #-new-users not found in file |
| reports/navigation-design.md | ../INDEX.md#-developers | Anchor #-developers not found in file |
| reports/navigation-design.md | ../INDEX.md#-architects | Anchor #-architects not found in file |
| reports/navigation-design.md | ../INDEX.md#-devops-operators | Anchor #-devops-operators not found in file |
| reports/navigation-design.md | 01-installation.md | File not found |
| reports/navigation-design.md | 02-first-graph.md | File not found |
| reports/navigation-design.md | neo4j-quick-start.md | File not found |
| reports/navigation-design.md | OVERVIEW.md | File not found |
| reports/navigation-design.md | tutorials/01-installation.md | File not found |
| reports/navigation-design.md | tutorials/02-first-graph.md | File not found |
| reports/navigation-design.md | INDEX.md#-new-users | File not found |
| reports/navigation-design.md | INDEX.md#-developers | File not found |
| reports/navigation-design.md | INDEX.md#-architects | File not found |
| reports/navigation-design.md | INDEX.md#-devops-operators | File not found |
| reports/navigation-design.md | INDEX.md#search-index | File not found |
| reports/navigation-design.md | INDEX.md#navigation-by-role | File not found |
| reports/navigation-design.md | INDEX.md#navigation-by-role | File not found |
| reports/navigation-design.md | INDEX.md | File not found |
| reports/navigation-design.md | tutorials/ | File not found |
| reports/navigation-design.md | guides/ | File not found |
| reports/navigation-design.md | explanations/ | File not found |
| reports/navigation-design.md | reference/ | File not found |
| reports/navigation-design.md | 01-installation.md | File not found |
| reports/navigation-design.md | 02-first-graph.md | File not found |
| reports/navigation-design.md | neo4j-quick-start.md | File not found |
| reports/navigation-design.md | navigation-guide.md | File not found |
| reports/navigation-design.md | configuration.md | File not found |
| reports/navigation-design.md | troubleshooting.md | File not found |
| reports/navigation-design.md | extending-the-system.md | File not found |
| reports/navigation-design.md | developer/01-development-setup.md | File not found |
| reports/navigation-design.md | developer/02-project-structure.md | File not found |
| reports/navigation-design.md | developer/04-adding-features.md | File not found |
| reports/navigation-design.md | testing-guide.md | File not found |
| reports/navigation-design.md | developer/06-contributing.md | File not found |
| reports/navigation-design.md | developer/02-project-structure.md | File not found |
| reports/navigation-design.md | client/state-management.md | File not found |
| reports/navigation-design.md | neo4j-integration.md | File not found |
| reports/navigation-design.md | features/gpu-optimization.md | File not found |
| reports/navigation-design.md | linked-doc.md | File not found |
| reports/navigation-design.md | task-guide.md | File not found |
| reports/navigation-design.md | foundation.md | File not found |
| reports/navigation-design.md | advanced.md | File not found |
| reports/navigation-design.md | other-system.md | File not found |
| reports/navigation-design.md | INDEX.md#-new-user-getting-started | File not found |
| reports/navigation-design.md | INDEX.md#-developer-getting-started | File not found |
| reports/navigation-design.md | INDEX.md#-architect-getting-started | File not found |
| reports/navigation-design.md | INDEX.md#-devops-getting-started | File not found |
| reports/navigation-design.md | INDEX.md#-api-consumer-getting-started | File not found |
| reports/navigation-design.md | INDEX.md#-security-officer-getting-started | File not found |
| reports/navigation-design.md | INDEX.md#-integration-engineer-getting-started | File not found |
| reports/navigation-design.md | OVERVIEW.md | File not found |
| reports/navigation-design.md | tutorials/01-installation.md | File not found |
| reports/navigation-design.md | tutorials/02-first-graph.md | File not found |
| reports/navigation-design.md | guides/navigation-guide.md | File not found |
| reports/navigation-design.md | tutorials/neo4j-quick-start.md | File not found |
| reports/navigation-design.md | guides/configuration.md | File not found |
| reports/navigation-design.md | DEVELOPER_JOURNEY.md | File not found |
| reports/navigation-design.md | guides/developer/01-development-setup.md | File not found |
| reports/navigation-design.md | guides/developer/02-project-structure.md | File not found |
| reports/navigation-design.md | ARCHITECTURE_OVERVIEW.md | File not found |
| reports/navigation-design.md | concepts/architecture/core/server.md | File not found |
| reports/navigation-design.md | explanations/architecture/core/client.md | File not found |
| reports/navigation-design.md | explanations/architecture/database-architecture.md | File not found |
| reports/navigation-design.md | explanations/architecture/gpu-semantic-forces.md | File not found |
| reports/navigation-design.md | reference/protocols/binary-websocket.md | File not found |
| reports/navigation-design.md | guides/developer/04-adding-features.md | File not found |
| reports/navigation-design.md | guides/testing-guide.md | File not found |
| reports/navigation-design.md | guides/developer/06-contributing.md | File not found |
| reports/navigation-design.md | tutorials/01-installation.md | File not found |
| reports/navigation-design.md | tutorials/02-first-graph.md | File not found |
| reports/navigation-design.md | tutorials/neo4j-quick-start.md | File not found |
| reports/navigation-design.md | guides/navigation-guide.md | File not found |
| reports/navigation-design.md | guides/developer/01-development-setup.md | File not found |
| reports/navigation-design.md | guides/deployment.md | File not found |
| reports/navigation-design.md | path.md | File not found |
| reports/navigation-design.md | path.md | File not found |
| guides/configuration.md | readme.md | File not found |
| guides/configuration.md | readme.md | File not found |
| guides/semantic-features-implementation.md | ../../explanations/architecture/semantic-forces-system.md | File not found |
| guides/semantic-features-implementation.md | ../../explanations/ontology/ontology-typed-system.md | File not found |
| guides/semantic-features-implementation.md | ../../explanations/ontology/intelligent-pathfinding-system.md | File not found |
| guides/semantic-features-implementation.md | ../../explanations/architecture/gpu/readme.md | File not found |
| guides/docker-environment-setup.md | ../../explanations/architecture/multi-agent-system.md | File not found |
| guides/docker-environment-setup.md | ../../explanations/architecture/multi-agent-system.md | File not found |
| guides/contributing.md | ./readme.md | File not found |
| guides/contributing.md | ./readme.md | File not found |
| guides/orchestrating-agents.md | readme.md | File not found |
| guides/ontology-storage-guide.md | ../../explanations/architecture/ontology-storage-architecture.md | File not found |
| guides/telemetry-logging.md | ../guides/readme.md | File not found |
| guides/telemetry-logging.md | ../guides/readme.md | File not found |
| guides/neo4j-implementation-roadmap.md | ../../explanations/architecture/system-overview.md | File not found |
| guides/neo4j-implementation-roadmap.md | ../../explanations/architecture/schemas.md | File not found |
| guides/neo4j-implementation-roadmap.md | ./developer/05-testing-guide.md | File not found |
| guides/index.md | ../archive/docs/guides/xr-setup.md | File not found |
| guides/index.md | ../archive/docs/guides/xr-setup.md | File not found |
| guides/index.md | ../archive/docs/guides/xr-setup.md | File not found |
| guides/index.md | xr-setup.md | File not found |
| guides/security.md | ../guides/readme.md | File not found |
| guides/security.md | ../guides/readme.md | File not found |
| guides/navigation-guide.md | readme.md | File not found |
| guides/navigation-guide.md | ../archive/docs/guides/xr-setup.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/xr-immersive-system.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/hexagonal-cqrs.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/schemas.md | File not found |
| guides/navigation-guide.md | ../reference/binary-websocket.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/gpu/readme.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/gpu/optimizations.md | File not found |
| guides/navigation-guide.md | ../archive/docs/guides/user/working-with-agents.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/multi-agent-system.md | File not found |
| guides/navigation-guide.md | ../archive/docs/guides/xr-setup.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/schemas.md | File not found |
| guides/navigation-guide.md | ../reference/binary-websocket.md | File not found |
| guides/navigation-guide.md | user/xr-setup.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/hexagonal-cqrs.md | File not found |
| guides/navigation-guide.md | ../../explanations/architecture/gpu/ | File not found |
| guides/navigation-guide.md | readme.md | File not found |
| guides/navigation-guide.md | readme.md | File not found |
| guides/troubleshooting.md | ../../tutorials/01-installation.md | File not found |
| guides/troubleshooting.md | readme.md | File not found |
| guides/development-workflow.md | ../../explanations/architecture/ | File not found |
| guides/extending-the-system.md | readme.md | File not found |
| guides/multi-agent-skills.md | ../../explanations/architecture/multi-agent-system.md | File not found |
| architecture/visionflow-distributed-systems-assessment.md | ../archive/reports/mermaid-fixes-examples.md | File not found |
| architecture/visionflow-distributed-systems-assessment.md | ../archive/reports/2025-12-02-restructuring-complete.md | File not found |
| architecture/HEXAGONAL_ARCHITECTURE_STATUS.md | ../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| tutorials/01-installation.md | ../../explanations/architecture/ | File not found |
| tutorials/02-first-graph.md | ../guides/xr-setup.md | File not found |
| tutorials/02-first-graph.md | ../guides/xr-setup.md | File not found |
| tutorials/02-first-graph.md | ../guides/xr-setup.md | File not found |
| reference/protocols/binary-websocket.md | ../diagrams/infrastructure/websocket/binary-protocol-complete.md#3-position-update-v2-21-bytes-per-node | File not found |
| reference/protocols/binary-websocket.md | ../diagrams/infrastructure/websocket/binary-protocol-complete.md#protocol-versions | File not found |
| reference/protocols/binary-websocket.md | ../diagrams/infrastructure/websocket/binary-protocol-complete.md#binary-message-formats | File not found |
| reference/protocols/binary-websocket.md | ../diagrams/infrastructure/websocket/binary-protocol-complete.md#1-message-header-all-messages | File not found |
| reference/api/rest-api-reference.md | ../../../explanations/architecture/ontology-reasoning-pipeline.md | File not found |
| reference/api/rest-api-reference.md | ../../../explanations/architecture/semantic-physics-system.md | File not found |
| reference/api/03-websocket.md | ../binary-websocket.md | File not found |
| explanations/physics/semantic-forces.md | ../../archive/tests/test_README.md | File not found |
| explanations/physics/semantic-forces.md | ../../archive/fixes/borrow-checker.md | File not found |
| explanations/physics/semantic-forces.md | ../../archive/fixes/README.md | File not found |
| explanations/physics/semantic-forces.md | ../../archive/fixes/borrow-checker-summary.md | File not found |
| explanations/ontology/intelligent-pathfinding-system.md | ./explanations/architecture/semantic-forces-system.md | File not found |
| explanations/ontology/intelligent-pathfinding-system.md | ../guides/semantic-features-implementation.md | File not found |
| explanations/ontology/intelligent-pathfinding-system.md | ../guides/semantic-features-implementation.md | File not found |
| explanations/ontology/ontology-typed-system.md | ./explanations/architecture/semantic-forces-system.md | File not found |
| explanations/ontology/ontology-typed-system.md | ../guides/neo4j-integration.md | File not found |
| explanations/ontology/ontology-typed-system.md | ../guides/semantic-features-implementation.md | File not found |
| explanations/ontology/ontology-typed-system.md | ../guides/semantic-features-implementation.md | File not found |
| explanations/architecture/event-driven-architecture.md | ./modular-actor-system.md | File not found |
| explanations/architecture/event-driven-architecture.md | ./hexagonal-cqrs-architecture.md | File not found |
| explanations/architecture/event-driven-architecture.md | ../../concepts/domain-driven-design.md | File not found |
| explanations/architecture/quick-reference.md | ../../diagrams/server/actors/actor-system-complete.md#actor-lifecycle-and-supervision-strategies | Anchor #actor-lifecycle-and-supervision-strategies not found in file |
| explanations/architecture/quick-reference.md | ../../diagrams/server/actors/actor-system-complete.md#message-flow-patterns | Anchor #message-flow-patterns not found in file |
| explanations/architecture/api-handlers-reference.md | schemas.md | File not found |
| explanations/architecture/services-architecture.md | schemas.md | File not found |
| explanations/architecture/database-architecture.md | ../../guides/user-settings.md | File not found |
| explanations/architecture/database-architecture.md | ../../docs/user-settings-implementation-summary.md | File not found |
| explanations/architecture/database-architecture.md | ../../docs/neo4j-user-settings-schema.md | File not found |
| explanations/architecture/database-architecture.md | ../../guides/sqlite-to-neo4j-migration.md | File not found |
| explanations/architecture/database-architecture.md | ../../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| explanations/architecture/ports/03-knowledge-graph-repository.md | ../../../archive/deprecated-patterns/03-architecture-WRONG-STACK.md | File not found |
| explanations/architecture/ports/01-overview.md | ../../../archive/deprecated-patterns/03-architecture-WRONG-STACK.md | File not found |
| diagrams/mermaid-library/README.md | 03-deployment-infrastructure.md#6-backup--disaster-recovery | Anchor #6-backup--disaster-recovery not found in file |
| diagrams/mermaid-library/README.md | /docs/ARCHITECTURE_OVERVIEW.md | File not found |
| diagrams/mermaid-library/README.md | /docs/explanations/architecture/hexagonal-cqrs.md | File not found |
| diagrams/mermaid-library/README.md | /docs/diagrams/server/actors/actor-system-complete.md | File not found |
| diagrams/mermaid-library/README.md | /docs/diagrams/infrastructure/websocket/binary-protocol-complete.md | File not found |
| diagrams/mermaid-library/README.md | /docs/diagrams/infrastructure/gpu/cuda-architecture-complete.md | File not found |
| diagrams/mermaid-library/README.md | /docs/diagrams/infrastructure/database/neo4j-architecture-complete.md | File not found |
| diagrams/mermaid-library/README.md | /docs/DEVELOPER_JOURNEY.md | File not found |
| diagrams/mermaid-library/README.md | /docs/TECHNOLOGY_CHOICES.md | File not found |
| diagrams/mermaid-library/README.md | /docs/QUICK_NAVIGATION.md | File not found |
| diagrams/mermaid-library/00-mermaid-style-guide.md | 01-system-architecture-overview.md#section | Anchor #section not found in file |
| diagrams/mermaid-library/00-mermaid-style-guide.md | 02-data-flow-diagrams.md#github-sync | Anchor #github-sync not found in file |
| diagrams/infrastructure/websocket/binary-protocol-complete.md | #frame-structure--byte-layout | Anchor not found |
| diagrams/infrastructure/websocket/binary-protocol-complete.md | #heartbeat--keepalive | Anchor not found |
| diagrams/infrastructure/websocket/binary-protocol-complete.md | #multi-client-broadcast | Anchor not found |
| diagrams/infrastructure/websocket/binary-protocol-complete.md | #queue-management--backpressure | Anchor not found |
| diagrams/infrastructure/websocket/binary-protocol-complete.md | #error-handling--recovery | Anchor not found |
| diagrams/infrastructure/websocket/binary-protocol-complete.md | ../../../archive/analysis/client-architecture-analysis-2025-12.md | File not found |
| diagrams/infrastructure/gpu/cuda-architecture-complete.md | ../../../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| diagrams/server/actors/actor-system-complete.md | ../../../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| diagrams/server/api/rest-api-architecture.md | #authentication--authorization | Anchor not found |
| diagrams/server/api/rest-api-architecture.md | #rate-limiting--throttling | Anchor not found |
| diagrams/server/api/rest-api-architecture.md | #error-handling--status-codes | Anchor not found |
| diagrams/server/api/rest-api-architecture.md | #request-validation--sanitization | Anchor not found |
| diagrams/server/api/rest-api-architecture.md | ../../../archive/analysis/client-architecture-analysis-2025-12.md | File not found |
| diagrams/client/xr/xr-architecture-complete.md | ../../../archive/analysis/client-architecture-analysis-2025-12.md | File not found |
| diagrams/client/state/state-management-complete.md | #persistence--hydration | Anchor not found |
| diagrams/client/state/state-management-complete.md | ../../../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
| guides/infrastructure/docker-environment.md | ../guides/security.md | File not found |
| guides/infrastructure/readme.md | ../guides/security.md | File not found |
| guides/features/ontology-sync-enhancement.md | ../../archive/README.md | File not found |
| guides/features/settings-authentication.md | ../../archive/deprecated-patterns/README.md | File not found |
| guides/features/settings-authentication.md | ../../archive/deprecated-patterns/03-architecture-WRONG-STACK.md | File not found |
| guides/features/settings-authentication.md | ../../archive/reports/2025-12-02-user-settings-summary.md | File not found |
| guides/developer/websocket-best-practices.md | ../../reference/binary-websocket.md | File not found |
| guides/developer/websocket-best-practices.md | ../../../explanations/architecture/components/websocket-protocol.md | File not found |
| guides/developer/02-project-structure.md | ./03-architecture.md | File not found |
| guides/developer/02-project-structure.md | ./05-testing-guide.md | File not found |
| guides/developer/json-serialization-patterns.md | ../../reference/binary-websocket.md | File not found |
| guides/developer/json-serialization-patterns.md | ../../reference/api/readme.md | File not found |
| guides/developer/04-adding-features.md | ../../archive/docs/guides/developer/05-testing-guide.md | File not found |
| guides/developer/04-adding-features.md | ../../archive/docs/guides/user/working-with-agents.md | File not found |
| guides/developer/01-development-setup.md | ./03-architecture.md | File not found |
| guides/developer/01-development-setup.md | ../../../explanations/architecture/ | File not found |
| guides/developer/readme.md | ./03-architecture.md | File not found |
| guides/developer/readme.md | ./03-architecture.md | File not found |
| guides/developer/readme.md | ./05-testing-guide.md | File not found |
| guides/client/three-js-rendering.md | #known-issues--workarounds | Anchor not found |
| guides/client/three-js-rendering.md | ./websocket-protocol.md | File not found |
| guides/client/state-management.md | ../../archive/docs/guides/developer/05-testing-guide.md | File not found |
| guides/client/state-management.md | ./websocket-protocol.md | File not found |
| guides/client/xr-integration.md | #known-issues--workarounds | Anchor not found |
| guides/client/xr-integration.md | ../../archive/docs/guides/developer/05-testing-guide.md | File not found |
| guides/client/xr-integration.md | ../../archive/docs/guides/user/working-with-agents.md | File not found |
| guides/ai-models/README.md | /multi-agent-docker/skills/deepseek-reasoning/SKILL.md | File not found |
| guides/ai-models/README.md | /multi-agent-docker/skills/perplexity/SKILL.md | File not found |
| guides/ai-models/README.md | /multi-agent-docker/skills/perplexity/docs/templates.md | File not found |
| guides/ai-models/README.md | /docker-compose.unified-with-neo4j.yml | File not found |
| guides/ai-models/README.md | /docs/reference/api-complete-reference.md | File not found |
| guides/ai-models/README.md | /docs/guides/multi-agent-skills.md | File not found |
| guides/ai-models/README.md | /docs/guides/orchestrating-agents.md | File not found |
| guides/ai-models/README.md | /docs/guides/configuration.md | File not found |
| guides/ai-models/README.md | /docs/guides/features/deepseek-verification.md | File not found |
| guides/ai-models/README.md | /docs/guides/features/deepseek-deployment.md | File not found |
| guides/ai-models/README.md | /multi-agent-docker/skills/perplexity/SKILL.md | File not found |
| guides/ai-models/README.md | /multi-agent-docker/skills/deepseek-reasoning/SKILL.md | File not found |
| guides/ai-models/README.md | /docs/guides/docker-environment-setup.md | File not found |
| guides/ai-models/README.md | /multi-agent-docker/README.md | File not found |
| guides/ai-models/README.md | /multi-agent-docker/unified-config/supervisord.unified.conf | File not found |
| guides/ai-models/ragflow-integration.md | ../../archive/docs/guides/user/working-with-agents.md | File not found |
| guides/ai-models/perplexity-integration.md | ../../archive/docs/guides/user/working-with-agents.md | File not found |
| guides/ai-models/perplexity-integration.md | ../../archive/docs/guides/developer/05-testing-guide.md | File not found |
| guides/architecture/actor-system.md | ../debugging/actor-message-tracing.md | File not found |
| guides/architecture/actor-system.md | ../performance/actor-optimization.md | File not found |
| guides/architecture/actor-system.md | ../../archive/docs/guides/developer/05-testing-guide.md | File not found |
| guides/architecture/actor-system.md | ../../archive/docs/guides/user/working-with-agents.md | File not found |
| concepts/architecture/core/client.md | ../../../archive/analysis/client-architecture-analysis-2025-12.md | File not found |
| concepts/architecture/core/server.md | ../../../explanations/gpu-computing.md | File not found |
| concepts/architecture/core/server.md | ../../../guides/user-settings.md | File not found |
| concepts/architecture/core/server.md | ../../../DOCUMENTATION_MODERNIZATION_COMPLETE.md | File not found |
