# Documentation Consolidation Plan

**Version:** 1.0.0
**Date:** 2025-10-12
**Status:** In Progress

## Executive Summary

This document outlines the consolidation of 682+ documentation files across the agentic-flow repository into a standardised, professionally organised structure under `/docs` with UK English spelling.

## Current State Analysis

### Documentation Distribution
- **Total Files:** 682 documentation files
- **Primary Locations:**
  - `/agentic-flow/docs/` - 100+ files (core documentation)
  - `/agentic-flow/.claude/agents/` - 66+ agent definitions
  - `/agentic-flow/.claude/commands/` - 200+ command files
  - `/agentic-flow/docs/archived/` - 75+ legacy validation reports
  - `/agent-booster/` - 40+ files
  - `/docker/cachyos/` - Test framework documentation
  - Various README files scattered across modules

### Quality Assessment
- **Strengths:**
  - Comprehensive coverage of all major features
  - Extensive validation and testing documentation
  - Clear integration guides
  - Well-defined agent and command structures

- **Weaknesses:**
  - Documentation fragmentation across multiple directories
  - Duplicate content (multiple README files, validation summaries)
  - Inconsistent spelling (mixed US/UK English)
  - Archived content mixed with active documentation
  - Lack of clear navigation structure
  - No central index or documentation portal

## Proposed Structure

### New `/docs` Organisation

```
/docs/
├── README.md                          # Main documentation portal
├── GETTING_STARTED.md                 # Quick start guide
├── ARCHITECTURE.md                    # System architecture overview
├── CHANGELOG.md                       # Consolidated changelog
│
├── guides/                            # User guides
│   ├── README.md
│   ├── installation.md
│   ├── configuration.md
│   ├── deployment.md
│   ├── docker-deployment.md
│   ├── gcp-deployment.md
│   └── troubleshooting.md
│
├── architecture/                      # Architecture documentation
│   ├── README.md
│   ├── system-overview.md
│   ├── multi-model-routing.md
│   ├── agent-orchestration.md
│   ├── mcp-integration.md
│   └── cost-optimisation.md
│
├── integrations/                      # Integration guides
│   ├── README.md
│   ├── agent-booster.md
│   ├── claude-flow.md
│   ├── gemini.md
│   ├── openrouter.md
│   ├── onnx.md
│   └── mcp-servers.md
│
├── api/                               # API documentation
│   ├── README.md
│   ├── management-api.md
│   ├── router-api.md
│   ├── agent-sdk.md
│   └── mcp-protocol.md
│
├── agents/                            # Agent system documentation
│   ├── README.md
│   ├── overview.md
│   ├── core-agents.md
│   ├── specialised-agents.md
│   ├── swarm-coordination.md
│   └── agent-development.md
│
├── commands/                          # Command reference
│   ├── README.md
│   ├── quick-reference.md
│   ├── sparc-commands.md
│   ├── swarm-commands.md
│   └── github-commands.md
│
├── testing/                           # Testing documentation
│   ├── README.md
│   ├── unit-testing.md
│   ├── integration-testing.md
│   ├── e2e-testing.md
│   ├── performance-benchmarks.md
│   └── validation.md
│
├── operations/                        # Operations & monitoring
│   ├── README.md
│   ├── monitoring.md
│   ├── logging.md
│   ├── metrics.md
│   ├── alerting.md
│   └── incident-response.md
│
├── development/                       # Developer documentation
│   ├── README.md
│   ├── contributing.md
│   ├── coding-standards.md
│   ├── git-workflow.md
│   └── release-process.md
│
├── reference/                         # Reference materials
│   ├── README.md
│   ├── glossary.md
│   ├── model-comparison.md
│   ├── provider-comparison.md
│   └── cost-calculator.md
│
├── releases/                          # Release notes
│   ├── README.md
│   ├── v1.0.0.md
│   ├── v1.1.0.md
│   ├── v1.2.0.md
│   └── v1.3.0.md
│
├── examples/                          # Code examples
│   ├── README.md
│   ├── basic-agent.md
│   ├── custom-mcp-server.md
│   ├── multi-model-routing.md
│   └── swarm-orchestration.md
│
└── archived/                          # Historical documentation
    ├── README.md
    └── [organised by date/version]
```

## Consolidation Strategy

### Phase 1: Structure Creation (Immediate)
1. Create new `/docs` directory structure
2. Establish documentation templates
3. Define UK English spelling conventions
4. Create main README.md portal

### Phase 2: Core Documentation Migration (Week 1)
1. Consolidate architecture documentation
2. Merge integration guides
3. Standardise API documentation
4. Create unified getting started guide

### Phase 3: Agent & Command Documentation (Week 1-2)
1. Consolidate agent definitions into comprehensive guide
2. Create command reference from 200+ command files
3. Develop quick reference cards
4. Build search index

### Phase 4: Legacy Content Management (Week 2)
1. Archive validation reports by version
2. Remove duplicate content
3. Consolidate changelog entries
4. Clean up redundant README files

### Phase 5: Quality Assurance (Week 2-3)
1. UK English spelling conversion
2. Consistent formatting and style
3. Link validation and fixing
4. Search functionality testing

## UK English Spelling Conventions

### Key Changes
- `optimization` → `optimisation`
- `authorization` → `authorisation`
- `organization` → `organisation`
- `analyze` → `analyse`
- `color` → `colour`
- `behavior` → `behaviour`
- `center` → `centre`
- `initialize` → `initialise`
- `serialize` → `serialise`

### Code vs Documentation
- **Code:** Maintain US English for:
  - Function names
  - Variable names
  - API endpoints
  - Configuration keys

- **Documentation:** Use UK English for:
  - All prose
  - Comments
  - User-facing text
  - Guide content

## Documentation Standards

### File Naming
- Use lowercase with hyphens: `multi-model-routing.md`
- Avoid underscores and spaces
- Be descriptive but concise

### Structure
1. **Title** (H1)
2. **Metadata** (version, date, status)
3. **Table of Contents** (for long documents)
4. **Overview** (summary)
5. **Main Content** (sections)
6. **Examples** (practical demonstrations)
7. **References** (links to related docs)
8. **Changelog** (document history)

### Formatting
- Use GitHub Flavoured Markdown
- Code blocks with language identifiers
- Consistent heading hierarchy
- Links to related documentation
- Version badges where applicable

### Metadata Block
```markdown
---
title: Document Title
version: 1.0.0
date: 2025-10-12
status: Active | Draft | Deprecated
category: Guide | Reference | Tutorial
tags: [tag1, tag2, tag3]
---
```

## Content Consolidation Rules

### Merge Criteria
Consolidate files when:
1. Content overlap exceeds 70%
2. Subject matter is identical
3. Multiple versions exist
4. Content is outdated but has current equivalent

### Archive Criteria
Move to `/docs/archived/` when:
1. Content is version-specific and superseded
2. Validation reports from completed work
3. Historical status reports
4. Deprecated integration guides

### Delete Criteria
Remove entirely when:
1. Content is duplicate with no additional value
2. Information is incorrect and superseded
3. Stub files with no content
4. Temporary documentation

## Migration Checklist

### Per-Document Tasks
- [ ] Convert to UK English spelling
- [ ] Add metadata block
- [ ] Update internal links
- [ ] Add to relevant README
- [ ] Update navigation
- [ ] Add version badge if applicable
- [ ] Review and update examples
- [ ] Validate all code snippets
- [ ] Add related documentation links
- [ ] Update changelog

### Global Tasks
- [ ] Create documentation portal (main README)
- [ ] Build navigation structure
- [ ] Implement search functionality
- [ ] Add version selector
- [ ] Create quick reference cards
- [ ] Generate API documentation
- [ ] Build glossary
- [ ] Create visual diagrams
- [ ] Set up documentation CI/CD
- [ ] Configure documentation hosting

## Success Metrics

### Quantitative
- Total files reduced from 682 to <150
- 100% UK English spelling compliance
- Zero broken links
- <3 levels of nesting in structure
- All documents with metadata

### Qualitative
- Single source of truth for each topic
- Clear navigation paths
- Consistent formatting
- Professional appearance
- Easy discoverability

## Timeline

### Week 1
- Phase 1: Structure creation
- Phase 2: Core documentation migration

### Week 2
- Phase 3: Agent & command documentation
- Phase 4: Legacy content management

### Week 3
- Phase 5: Quality assurance
- Final review and validation

## Risk Management

### Potential Issues
1. **Broken Links:** Use automated link checker
2. **Content Loss:** Maintain git history, archive everything
3. **User Disruption:** Create redirect map, maintain legacy docs temporarily
4. **Inconsistency:** Use automated spelling/style checkers
5. **Scope Creep:** Stick to consolidation, defer rewrites

### Mitigation
- Automated testing in CI/CD
- Staged rollout with redirects
- Documentation review process
- Style guide enforcement
- Regular progress reviews

## Maintenance Plan

### Ongoing
1. **Weekly:** Review new documentation for compliance
2. **Monthly:** Check for broken links
3. **Quarterly:** Update examples and versions
4. **Annually:** Major documentation review

### Responsibilities
- **Technical Writers:** Style and formatting
- **Engineers:** Technical accuracy
- **DevOps:** Documentation infrastructure
- **Community:** Feedback and contributions

## References

- [GitHub Flavoured Markdown Spec](https://github.github.com/gfm/)
- [UK English Style Guide](https://www.gov.uk/guidance/style-guide)
- [Documentation Best Practices](https://www.writethedocs.org/guide/)

## Changelog

### 2025-10-12
- Initial consolidation plan created
- Structure defined
- Migration strategy outlined
