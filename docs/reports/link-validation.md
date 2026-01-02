---
title: Link Validation Report
generated: 2026-01-02 17:54:25
category: reports
---

# Code Quality Analysis Report - Link Validation

## Summary

| Metric | Value |
|--------|-------|
| Overall Quality Score | 2/10 |
| Files Analysed | 431 |
| Total Internal Links | 3570 |
| Valid Links | 2752 (77%) |
| Broken Links | 818 |
| Anchor Issues | 103 |
| Orphaned Files | 169 |
| External Links | 202 |
| Technical Debt Estimate | 213.1 hours |

## Critical Issues

### Broken Links by Category

| Category | Count | Priority |
|----------|-------|----------|
| Case-sensitive errors (readme.md vs README.md) | 29 | High |
| Path errors (../docs, docs/docs) | 195 | High |
| Missing files (non-archive) | 329 | Medium |
| Archive references | 265 | Low |

### Top 15 Files with Broken Links

| File | Broken Links | Priority |
|------|--------------|----------|
| `archive/reports/consolidation/link-validation-report-2025-12.md` | 245 | Critical |
| `archive/INDEX-QUICK-START-old.md` | 98 | Critical |
| `reports/navigation-design.md` | 72 | Critical |
| `working/hive-spelling-audit.md` | 61 | Critical |
| `working/hive-link-validation.md` | 49 | Critical |
| `architect.md` | 36 | Critical |
| `reports/link-validation.md` | 30 | Critical |
| `guides/navigation-guide.md` | 15 | High |
| `guides/ai-models/README.md` | 15 | High |
| `index.md` | 11 | High |
| `archive/reports/link-fixes-report.md` | 10 | Medium |
| `getting-started.md` | 9 | Medium |
| `diagrams/mermaid-library/README.md` | 9 | Medium |
| `QUICK_NAVIGATION.md` | 8 | Medium |
| `CONTRIBUTION.md` | 6 | Medium |

## Broken Links Detail

### Case-Sensitive Errors (Quick Fix)

These links use `readme.md` instead of `README.md`:

- `archive/reports/consolidation/link-validation-report-2025-12.md:806`: `../../explanations/architecture/gpu/readme.md`
- `archive/reports/consolidation/link-validation-report-2025-12.md:860`: `../../explanations/architecture/gpu/readme.md`
- `working/hive-spelling-audit.md:2329`: `readme.md`
- `working/hive-spelling-audit.md:2331`: `../../explanations/architecture/gpu/readme.md`
- `working/hive-spelling-audit.md:2333`: `../../explanations/architecture/gpu/readme.md`
- `working/hive-link-validation.md:36`: `explanations/architecture/gpu/readme.md`
- `working/hive-link-validation.md:40`: `reference/api/readme.md`
- `working/hive-link-validation.md:44`: `explanations/architecture/gpu/readme.md`
- `working/hive-link-validation.md:164`: `explanations/architecture/gpu/readme.md`
- `working/hive-link-validation.md:168`: `reference/api/readme.md`
- `working/hive-link-validation.md:188`: `guides/readme.md`
- `working/hive-link-validation.md:200`: `explanations/architecture/gpu/readme.md`
- `working/hive-link-validation.md:204`: `reference/api/readme.md`
- `working/HIVE_QUALITY_REPORT.md:147`: `explanations/architecture/gpu/readme.md`
- `working/HIVE_QUALITY_REPORT.md:148`: `reference/api/readme.md`
- `guides/configuration.md:12`: `readme.md`
- `guides/semantic-features-implementation.md:797`: `../../explanations/architecture/gpu/readme.md`
- `guides/contributing.md:147`: `./readme.md`
- `guides/contributing.md:156`: `./readme.md`
- `guides/orchestrating-agents.md:2290`: `readme.md`

*...and 9 more*

### Path Errors (Requires Path Correction)

These links have incorrect path prefixes:

- `CONTRIBUTION.md:345`: `../deployment/docker-deployment.md`
  - Resolved to: `/home/devuser/workspace/project/deployment/docker-deployment.md`
- `CONTRIBUTION.md:350`: `/docs/deployment/docker-deployment.md`
  - Resolved to: `docs/deployment/docker-deployment.md`
- `CONTRIBUTION.md:356`: `../deployment/docker-deployment.md#configuration`
  - Resolved to: `/home/devuser/workspace/project/deployment/docker-deployment.md`
- `CONTRIBUTION.md:393`: `../api/rest-api.md`
  - Resolved to: `/home/devuser/workspace/project/api/rest-api.md`
- `CONTRIBUTION.md:394`: `../reference/configuration.md`
  - Resolved to: `/home/devuser/workspace/project/reference/configuration.md`
- `CONTRIBUTION.md:395`: `../guides/troubleshooting.md`
  - Resolved to: `/home/devuser/workspace/project/guides/troubleshooting.md`
- `MAINTENANCE.md:174`: `../related-category/related-doc.md`
  - Resolved to: `/home/devuser/workspace/project/related-category/related-doc.md`
- `MAINTENANCE.md:245`: `../new-category/new-doc.md`
  - Resolved to: `/home/devuser/workspace/project/new-category/new-doc.md`
- `audits/ascii-diagram-deprecation-audit.md:122`: `../../diagrams/data-flow/complete-data-flows.md`
  - Resolved to: `/home/devuser/workspace/project/diagrams/data-flow/complete-data-flows.md`
- `reference/implementation-status.md:455`: `../../explanations/architecture/system-overview.md`
  - Resolved to: `/home/devuser/workspace/project/explanations/architecture/system-overview.md`
- `reference/api-complete-reference.md:1314`: `../../explanations/architecture/system-overview.md`
  - Resolved to: `/home/devuser/workspace/project/explanations/architecture/system-overview.md`
- `reference/api-complete-reference.md:1315`: `../../tutorials/01-installation.md`
  - Resolved to: `/home/devuser/workspace/project/tutorials/01-installation.md`
- `reference/code-quality-status.md:405`: `../guides/developer/05-testing-guide.md`
  - Resolved to: `guides/developer/05-testing-guide.md`
- `reference/code-quality-status.md:406`: `../../explanations/architecture/system-overview.md`
  - Resolved to: `/home/devuser/workspace/project/explanations/architecture/system-overview.md`
- `reference/protocols/binary-websocket.md:62`: `../diagrams/infrastructure/websocket/binary-protocol-complete.md#3-position-update-v2-21-bytes-per-node`
  - Resolved to: `reference/diagrams/infrastructure/websocket/binary-protocol-complete.md`
- `reference/protocols/binary-websocket.md:116`: `../diagrams/infrastructure/websocket/binary-protocol-complete.md#protocol-versions`
  - Resolved to: `reference/diagrams/infrastructure/websocket/binary-protocol-complete.md`
- `reference/protocols/binary-websocket.md:138`: `../diagrams/infrastructure/websocket/binary-protocol-complete.md#binary-message-formats`
  - Resolved to: `reference/diagrams/infrastructure/websocket/binary-protocol-complete.md`
- `reference/protocols/binary-websocket.md:207`: `../diagrams/infrastructure/websocket/binary-protocol-complete.md#1-message-header-all-messages`
  - Resolved to: `reference/diagrams/infrastructure/websocket/binary-protocol-complete.md`
- `reference/api/rest-api-reference.md:633`: `../../../explanations/architecture/ontology-reasoning-pipeline.md`
  - Resolved to: `/home/devuser/workspace/project/explanations/architecture/ontology-reasoning-pipeline.md`
- `reference/api/rest-api-reference.md:634`: `../../../explanations/architecture/semantic-physics-system.md`
  - Resolved to: `/home/devuser/workspace/project/explanations/architecture/semantic-physics-system.md`

*...and 175 more*

### Missing Files (Requires Content Creation or Link Removal)

- `README.md:264`: [DeepSeek Verification](guides/features/deepseek-verification.md)
- `README.md:265`: [DeepSeek Deployment](guides/features/deepseek-deployment.md)
- `OVERVIEW.md:189`: [Installation Guide](getting-started/01-installation.md)
- `OVERVIEW.md:190`: [First Graph Tutorial](getting-started/02-first-graph-and-agents.md)
- `OVERVIEW.md:239`: [Get Started →](getting-started/01-installation.md)
- `architect.md:14`: [Architecture Overview]($1.md)
- `architect.md:15`: [Technology Choices]($1.md)
- `architect.md:16`: [System Overview]($1.md)
- `architect.md:17`: [Hexagonal CQRS]($1.md)
- `architect.md:25`: [Data Flow Complete]($1.md)
- `architect.md:26`: [Integration Patterns]($1.md)
- `architect.md:27`: [Services Architecture]($1.md)
- `architect.md:28`: [Adapter Patterns]($1.md)
- `architect.md:34`: [Ports Overview]($1.md)
- `architect.md:35`: [Knowledge Graph Repository]($1.md)
- `architect.md:36`: [Ontology Repository]($1.md)
- `architect.md:37`: [GPU Physics Adapter]($1.md)
- `architect.md:43`: [Server Architecture]($1.md)
- `architect.md:43`: [Actor System]($1.md)
- `architect.md:44`: [Client Architecture]($1.md)
- `architect.md:44`: [State Management]($1.md)
- `architect.md:45`: [Database Architecture]($1.md)
- `architect.md:45`: [Schemas]($1.md)
- `architect.md:46`: [GPU Semantic Forces]($1.md)
- `architect.md:46`: [Optimisations]($1.md)
- `architect.md:52`: [Storage Architecture]($1.md)
- `architect.md:52`: [Reasoning Pipeline]($1.md)
- `architect.md:53`: [Semantic Physics System]($1.md)
- `architect.md:53`: [Stress Majorisation]($1.md)
- `architect.md:54`: [Multi-Agent System]($1.md)

*...and 299 more*

## Anchor Issues

103 links point to anchors that do not exist in target files:

- `ARCHITECTURE_OVERVIEW.md:695`: Anchor #roadmap not found in /home/devuser/workspace/project/docs/README.md
- `DEVELOPER_JOURNEY.md:1048`: Anchor #faq not found in /home/devuser/workspace/project/docs/guides/troubleshooting.md
- `GETTING_STARTED_WITH_UNIFIED_DOCS.md:19`: Anchor #faq not found in /home/devuser/workspace/project/docs/INDEX.md
- `INDEX.md:38`: Anchor #specialized-content not found in same file
- `reference/README.md:239`: Anchor #authentication--authorization not found in /home/devuser/workspace/project/docs/reference/API_REFERENCE.md
- `reference/README.md:306`: Anchor #section-name not found in /home/devuser/workspace/project/docs/reference/API_REFERENCE.md
- `reference/API_REFERENCE.md:20`: Anchor #authentication--authorization not found in same file
- `reference/DATABASE_SCHEMA_REFERENCE.md:23`: Anchor #relationships--foreign-keys not found in same file
- `reference/DATABASE_SCHEMA_REFERENCE.md:24`: Anchor #indexes--performance not found in same file
- `reference/INDEX.md:35`: Anchor #agent-management not found in /home/devuser/workspace/project/docs/reference/CONFIGURATION_REFERENCE.md
- `reference/INDEX.md:37`: Anchor #authentication--authorization not found in /home/devuser/workspace/project/docs/reference/API_REFERENCE.md
- `reference/INDEX.md:78`: Anchor #relationships--foreign-keys not found in /home/devuser/workspace/project/docs/reference/DATABASE_SCHEMA_REFERENCE.md
- `reference/INDEX.md:96`: Anchor #indexes--performance not found in /home/devuser/workspace/project/docs/reference/DATABASE_SCHEMA_REFERENCE.md
- `reference/INDEX.md:110`: Anchor #logging--monitoring not found in /home/devuser/workspace/project/docs/reference/CONFIGURATION_REFERENCE.md
- `reference/INDEX.md:150`: Anchor #relationships--foreign-keys not found in /home/devuser/workspace/project/docs/reference/DATABASE_SCHEMA_REFERENCE.md
- `reference/INDEX.md:165`: Anchor #common-issues--solutions not found in /home/devuser/workspace/project/docs/reference/ERROR_REFERENCE.md
- `reference/INDEX.md:192`: Anchor #common-issues--solutions not found in /home/devuser/workspace/project/docs/reference/ERROR_REFERENCE.md
- `reference/INDEX.md:287`: Anchor #authentication--authorization not found in /home/devuser/workspace/project/docs/reference/API_REFERENCE.md
- `reference/INDEX.md:342`: Anchor #common-issues--solutions not found in /home/devuser/workspace/project/docs/reference/ERROR_REFERENCE.md
- `reference/INDEX.md:373`: Anchor #startrstop-simulation not found in /home/devuser/workspace/project/docs/reference/API_REFERENCE.md
- `reference/ERROR_REFERENCE.md:25`: Anchor #common-issues--solutions not found in same file
- `archive/reports/documentation-alignment-2025-12-02/SWARM_EXECUTION_REPORT.md:522`: Anchor #installation-guide not found in same file
- `archive/reports/documentation-alignment-2025-12-02/SWARM_EXECUTION_REPORT.md:525`: Anchor #setup-and-installation not found in same file
- `explanations/architecture/quick-reference.md:24`: Anchor #actor-lifecycle-and-supervision-strategies not found in /home/devuser/workspace/project/docs/diagrams/server/actors/actor-system-complete.md
- `working/hive-spelling-audit.md:930`: Anchor #performance-optimisation not found in same file
- `working/hive-spelling-audit.md:932`: Anchor #performance-optimisation not found in same file
- `working/hive-spelling-audit.md:1161`: Anchor #8-performance-optimisations not found in same file
- `working/hive-spelling-audit.md:1163`: Anchor #8-performance-optimisations not found in same file
- `working/hive-spelling-audit.md:1180`: Anchor #store-catalogue not found in same file
- `working/hive-spelling-audit.md:1182`: Anchor #store-catalogue not found in same file

*...and 73 more*

## Orphaned Files

169 files have no incoming links (excluding README.md and INDEX.md files):

- `API_TEST_IMPLEMENTATION.md`
- `CLIENT_CODE_ANALYSIS.md`
- `CODE_QUALITY_ANALYSIS.md`
- `CUDA_KERNEL_ANALYSIS_REPORT.md`
- `CUDA_KERNEL_AUDIT_REPORT.md`
- `CUDA_OPTIMIZATION_SUMMARY.md`
- `DOCS-MIGRATION-PLAN.md`
- `FINAL_LINK_VERIFICATION.md`
- `GETTING_STARTED_WITH_UNIFIED_DOCS.md`
- `LINK_REPAIR_REPORT.md`
- `LINK_VALIDATION_COMPLETE.md`
- `MIGRATION_LOG.md`
- `PROJECT_CONSOLIDATION_PLAN.md`
- `QUICK_NAVIGATION.md`
- `SOLID_POD_CREATION.md`
- `TEST_COVERAGE_ANALYSIS.md`
- `VALIDATION_CHECKLIST.md`
- `VISIONFLOW_WARDLEY_ANALYSIS.md`
- `_pages/getting-started.md`
- `analysis/index.md`
- `architect.md`
- `architecture/PROTOCOL_MATRIX.md`
- `architecture/VIRCADIA_BABYLON_CONSOLIDATION_ANALYSIS.md`
- `architecture/phase1-completion.md`
- `architecture/skill-mcp-classification.md`
- `architecture/skills-refactoring-plan.md`
- `architecture/user-agent-pod-design.md`
- `architecture_analysis_report.md`
- `archive/ARCHIVE_REPORT.md`
- `archive/INDEX-QUICK-START-old.md`
- `archive/analysis/analysis-summary-2025-12.md`
- `archive/analysis/client-docs-summary-2025-12.md`
- `archive/analysis/historical-context-recovery-2025-12.md`
- `archive/audits/quality-gates-api.md`
- `archive/data/markdown/IMPLEMENTATION-SUMMARY.md`
- `archive/data/markdown/OntologyDefinition.md`
- `archive/data/markdown/implementation-examples.md`
- `archive/data/pages/ComfyWorkFlows.md`
- `archive/data/pages/IMPLEMENTATION-SUMMARY.md`
- `archive/data/pages/OntologyDefinition.md`
- `archive/data/pages/implementation-examples.md`
- `archive/docs/guides/working-with-gui-sandbox.md`
- `archive/fixes/type-corrections-final-summary.md`
- `archive/fixes/type-corrections-progress.md`
- `archive/implementation-logs/stress-majorization-implementation.md`
- `archive/multi-agent-docker/skills/IMPLEMENTATION_GUIDE.md`
- `archive/reports/CLEANUP_SUMMARY.md`
- `archive/reports/ascii-conversion-report.md`
- `archive/reports/ascii-to-mermaid-conversion.md`
- `archive/reports/consolidation/link-fix-suggestions-2025-12.md`

*...and 119 more*

## Refactoring Opportunities

### 1. Consolidate Archive Links
Many broken links originate from archive files. Consider:
- Removing archive files from active documentation
- Fixing links within archive for historical accuracy

### 2. Standardise File Naming
Adopt consistent naming conventions:
- Use `README.md` (uppercase) consistently
- Avoid `readme.md`, `Readme.md` variants

### 3. Remove Deprecated References
Several files reference non-existent DeepSeek and getting-started guides:
- Either create these files or remove references

### 4. Fix Path Structure
Common path issues:
- `docs/docs/` should be `docs/`
- Excessive `../` navigation

## Positive Findings

- 77% of internal links are valid
- Core documentation files (README.md, INDEX.md) are well-linked
- External links are properly formatted
- No security issues with links (no credentials in URLs)

## Recommended Actions

1. **Immediate**: Fix case-sensitive README links (39 items)
2. **Short-term**: Correct path prefix errors (196 items)  
3. **Medium-term**: Address missing file references (317 items)
4. **Long-term**: Review and link orphaned files (169 items)

---

*Report generated automatically by link validation tool*
