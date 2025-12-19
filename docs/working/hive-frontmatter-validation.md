---
title: "YAML Front Matter Validation Report"
description: "**Generated**: 2025-12-19T18:01:25.052907"
category: explanation
tags:
  - documentation
updated-date: 2025-12-19
difficulty-level: intermediate
---

# YAML Front Matter Validation Report

**Generated**: 2025-12-19T18:01:25.052907

## Summary Statistics

- **Total Files**: 299
- **Files With Front Matter**: 145
- **Files Missing Front Matter**: 5
- **Front Matter Compliance**: 48.49%

## Issues Breakdown

### Malformed YAML (149 files)

- `README.md`: YAML parse error: while scanning a block scalar
  in "<unicode string>", line 2, column 14:
    description: > **226 documents** organised us ... 
                 ^
expected a comment or a line break, but found '*'
  in "<unicode string>", line 2, column 16:
    description: > **226 documents** organised usin ... 
                   ^
- `ARCHITECTURE_OVERVIEW.md`: YAML parse error: mapping values are not allowed here
  in "<unicode string>", line 2, column 72:
     ... ee core architectural principles:
                                         ^
- `TECHNOLOGY_CHOICES.md`: YAML parse error: mapping values are not allowed here
  in "<unicode string>", line 1, column 37:
    title: VisionFlow Technology Choices: Design Rationale
                                        ^
- `DEVELOPER_JOURNEY.md`: YAML parse error: mapping values are not allowed here
  in "<unicode string>", line 1, column 36:
    title: VisionFlow Developer Journey: Navigating the Codebase
                                       ^
- `QUICK_NAVIGATION.md`: YAML parse error: while scanning a block scalar
  in "<unicode string>", line 2, column 14:
    description: > **Search tip**: Use Ctrl+F to  ... 
                 ^
expected a comment or a line break, but found '*'
  in "<unicode string>", line 2, column 16:
    description: > **Search tip**: Use Ctrl+F to fi ... 
                   ^
- `audits/neo4j-settings-migration-audit.md`: YAML parse error: while scanning an alias
  in "<unicode string>", line 2, column 14:
    description: **Date**: 2025-11-06 **Auditor** ... 
                 ^
expected alphabetic or numeric character, but found '*'
  in "<unicode string>", line 2, column 15:
    description: **Date**: 2025-11-06 **Auditor**: ... 
                  ^
- `audits/neo4j-migration-action-plan.md`: YAML parse error: while scanning an alias
  in "<unicode string>", line 2, column 14:
    description: **Date**: 2025-11-06 **Priority* ... 
                 ^
expected alphabetic or numeric character, but found '*'
  in "<unicode string>", line 2, column 15:
    description: **Date**: 2025-11-06 **Priority** ... 
                  ^
- `audits/neo4j-migration-summary.md`: YAML parse error: while scanning an alias
  in "<unicode string>", line 2, column 14:
    description: **Date**: 2025-11-06 **Status**: ... 
                 ^
expected alphabetic or numeric character, but found '*'
  in "<unicode string>", line 2, column 15:
    description: **Date**: 2025-11-06 **Status**:  ... 
                  ^
- `reference/physics-implementation.md`: YAML parse error: while scanning an alias
  in "<unicode string>", line 2, column 14:
    description: **Agent**: Semantic Physics Spec ... 
                 ^
expected alphabetic or numeric character, but found '*'
  in "<unicode string>", line 2, column 15:
    description: **Agent**: Semantic Physics Speci ... 
                  ^
- `reference/error-codes.md`: YAML parse error: mapping values are not allowed here
  in "<unicode string>", line 2, column 168:
     ... low a hierarchical naming scheme: `[SYSTEM]-[SEVERITY]-[NUMBER]`.
                                         ^
- ... and 139 more

### Missing Required Fields (6 issues)

- `reference/README.md`: Missing required field: category
- `reference/README.md`: Missing required field: tags
- `reference/PROTOCOL_REFERENCE.md`: Missing required field: category
- `reference/PROTOCOL_REFERENCE.md`: Missing required field: tags
- `reference/INDEX.md`: Missing required field: category
- `reference/INDEX.md`: Missing required field: tags

### Invalid Categories (0 files)

None

### Non-Standard Tags (134 files)

- `DOCUMENTATION_MODERNIZATION_COMPLETE.md`: ['patterns', 'structure', 'rest']
- `CONTRIBUTION.md`: ['contribution', 'standards', 'workflow']
- `OVERVIEW.md`: ['design', 'patterns', 'structure']
- `gpu-fix-summary.md`: ['http', 'ai']
- `MAINTENANCE.md`: ['maintenance', 'procedures', 'quality']
- `ARCHITECTURE_COMPLETE.md`: ['design', 'patterns', 'structure']
- `VALIDATION_CHECKLIST.md`: ['validation', 'quality', 'checklist', 'standards']
- `NAVIGATION.md`: ['navigation', 'search', 'documentation', 'guide']
- `visionflow-architecture-analysis.md`: ['design', 'patterns', 'structure']
- `comfyui-integration-design.md`: ['design', 'structure']
- ... and 124 more

### Files Missing Front Matter (5 files)

- `GETTING_STARTED_WITH_UNIFIED_DOCS.md`
- `working/QUICK_REFERENCE.md`
- `diagrams/mermaid-library/README.md`
- `diagrams/mermaid-library/00-mermaid-style-guide.md`
- `archive/analysis/analysis-summary-2025-12.md`


## Standard Tag Vocabulary

actors, agents, api, architecture, authentication, binary-protocol, caching, client, configuration, cuda, database, deployment, docker, error-handling, features, github, gpu, guides, hexagonal, integration, json, logging, mcp, memory, monitoring, neo4j, ontology, performance, physics, protocol, python, reference, rest-api, rust, schema, security, semantic, server, settings, testing, troubleshooting, typescript, visualization, websocket, xr

## Recommendations

- **Priority**: Add front matter to 5 files
- Fix 6 required field issues
- Standardize 134 files with non-standard tags
- Repair 149 files with malformed YAML
