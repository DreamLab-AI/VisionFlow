# Ontology/Knowledge Skills Analysis

**Date**: 2025-12-18
**Analyst**: Claude Code Quality Analyzer
**Focus**: Ontology and knowledge management skills in multi-agent-docker/skills/

---

## Executive Summary

Analyzed 6 ontology/knowledge skills to determine implementation status, MCP integration, dependencies, and migration priorities. Found mixed implementation states requiring strategic consolidation and migration.

### Key Findings

| Skill | Status | MCP | Implementation | Priority |
|-------|--------|-----|----------------|----------|
| **ontology-core** | ✅ Library | ❌ No | Python complete | Keep separate |
| **ontology-enrich** | ✅ Implemented | ❌ No | Python complete | Migrate to MCP |
| **import-to-ontology** | ✅ Implemented | ❌ No | Node.js + Python bridge | Migrate to MCP |
| **web-summary** | ✅ Reference | ✅ FastMCP | Python complete | Reference impl |
| **deepseek-reasoning** | ⚠️ Stub | ✅ MCP SDK | Node.js stub only | Implement client |
| **perplexity** | ⚠️ Stub | ✅ MCP SDK | Node.js stub only | Implement client |

---

## 1. Ontology-Core (Library)

### Classification
**Pure Library** - Not a skill, provides shared functionality

### Implementation Status
✅ **Fully Implemented** (Python)

**Files**:
- `src/ontology_parser.py` (18,978 bytes) - Complete parser
- `src/ontology_modifier.py` (11,450 bytes) - Safe modifications
- `src/owl2_validator.py` (9,008 bytes) - OWL2 validation
- `tests/` - Test suite present

### Features
- Zero-data-loss parsing (17+ fields preserved)
- Immutable operations
- OWL2 validation (namespaces, syntax, axioms)
- Automatic rollback on failures
- Unknown field preservation

### MCP Status
❌ **No MCP server** (by design - it's a library)

### Dependencies
- Python 3.x
- No external dependencies in requirements.txt

### Recommendation
**KEEP SEPARATE** - This is infrastructure, not a skill. Other skills import it as a library.

---

## 2. Ontology-Enrich

### Classification
**Layer 1 Skill** - Enriches existing ontology files

### Implementation Status
✅ **Fully Implemented** (Python)

**Files**:
- `src/enrichment_workflow.py` (12,711 bytes) - Main workflow
- `src/link_validator.py` (8,899 bytes) - WikiLink validation
- `src/perplexity_client.py` (10,615 bytes) - Perplexity API integration

### Features
- In-place validation and enrichment
- Perplexity API integration (UK English focus)
- Broken link detection/repair
- Field preservation via ontology-core
- Automatic rollback on validation failures

### MCP Status
❌ **No MCP server** - Needs migration to expose via MCP

### Dependencies
- **ontology-core** (Python library)
- Perplexity API key
- Git for rollback

### Recommendation
**MIGRATE TO MCP** - High priority
- Create FastMCP wrapper (like web-summary)
- Expose tools: `validate_file`, `enrich_field`, `fix_links`, `batch_enrich`
- Keep Python implementation, add MCP layer
- Priority: **HIGH** (needed for ontology workflows)

---

## 3. Import-to-Ontology

### Classification
**Migration Tool** - Batch import with validation

### Implementation Status
✅ **Fully Implemented** (Node.js + Python bridge)

**Files**:
- `import-engine.js` (22,315 bytes) - Main import logic
- `destructive-import.js` (18,976 bytes) - Destructive operations
- `llm-matcher.js` (8,660 bytes) - Semantic targeting
- `asset-handler.js` (7,954 bytes) - Asset management
- `src/validation_bridge.js` (9,164 bytes) - Node → Python bridge

### Features
- Batch markdown import (200+ files)
- Semantic targeting via in-memory index
- OWL2 validation before/after moves
- WikiLink and URL stub detection
- Web-summary integration for enrichment
- Destructive operations with rollback
- Progress tracking with resume

### MCP Status
❌ **No MCP server** - Uses validation_bridge.js to call ontology-core

### Dependencies
- **ontology-core** (Python, via validation_bridge.js)
- **web-summary** (for URL enrichment)
- Node.js runtime
- No npm dependencies (pure Node)

### Issues
- Hardcoded path to validator: `../../../logseq/skills/ontology-augmenter/src/owl2_validator.py`
- Should reference `ontology-core` dynamically

### Recommendation
**MIGRATE TO MCP** - Medium priority
- Create MCP wrapper for import tools
- Fix validator path to use relative or configured path
- Expose tools: `import_file`, `import_batch`, `dry_run`, `progress_status`
- Keep Node.js implementation, add MCP layer
- Priority: **MEDIUM** (useful but not critical for daily workflows)

---

## 4. Web-Summary (Reference Implementation)

### Classification
**FastMCP Reference** - Exemplar implementation

### Implementation Status
✅ **Fully Implemented** (FastMCP + Python)

**Files**:
- `mcp-server/server.py` (381 lines, complete)
- Clean FastMCP implementation
- Z.AI integration on port 9600

### Features
- URL content summarization
- YouTube transcript extraction
- Semantic topic generation (Logseq/Obsidian format)
- Health check tool
- VisionFlow integration via MCP resources

### MCP Status
✅ **FastMCP server** - Production ready

**Tools**:
- `summarize_url` - Web/YouTube summarization
- `youtube_transcript` - Full transcript extraction
- `generate_topics` - Semantic topic links
- `health_check` - Z.AI connectivity check

### Dependencies
- Z.AI service (port 9600)
- httpx (async HTTP client)
- youtube-transcript-api

### Recommendation
**REFERENCE IMPLEMENTATION** - Use as template
- This is the gold standard for FastMCP skills
- Copy architecture for ontology-enrich and import-to-ontology migrations
- No changes needed - keep as-is

---

## 5. DeepSeek-Reasoning (Stub)

### Classification
**MCP Stub** - Needs client implementation

### Implementation Status
⚠️ **Stub Only** (MCP wrapper without client)

**Files**:
- `mcp-server/server.js` (7,528 bytes) - MCP wrapper only
- `tools/` directory exists but no client implementation

### Declared Features (Not Implemented)
- Complex multi-step reasoning
- Code/system analysis
- Task planning with dependencies
- User isolation (devuser → deepseek-user bridge)

### MCP Status
✅ **MCP SDK wrapper exists** - But missing actual client

**Declared Tools** (stubs only):
- `deepseek_reason` - Not implemented
- `deepseek_analyze` - Not implemented
- `deepseek_plan` - Not implemented

### Missing Implementation
```
tools/deepseek_client.js - MISSING
```

The MCP server spawns this client but it doesn't exist.

### Recommendation
**IMPLEMENT CLIENT** - Low-medium priority
- Create `tools/deepseek_client.js`
- Implement DeepSeek API integration
- Add user bridge: `sudo -u deepseek-user`
- Or consider removing if not needed
- Priority: **LOW-MEDIUM** (useful for complex reasoning but not critical)

---

## 6. Perplexity (Stub)

### Classification
**MCP Stub** - Needs client implementation

### Implementation Status
⚠️ **Stub Only** (MCP wrapper without client)

**Files**:
- `mcp-server/server.js` (7,628 bytes) - MCP wrapper only
- `tools/` directory exists but no client implementation

### Declared Features (Not Implemented)
- Real-time web search
- Deep research with citations
- UK-centric focus
- Prompt optimization

### MCP Status
✅ **MCP SDK wrapper exists** - But missing actual client

**Declared Tools** (stubs only):
- `perplexity_search` - Not implemented
- `perplexity_research` - Not implemented
- `perplexity_generate_prompt` - Not implemented

### Missing Implementation
```
tools/perplexity_client.py - MISSING
```

The MCP server spawns this Python client but it doesn't exist.

### Conflict with Ontology-Enrich
⚠️ **DUPLICATE FUNCTIONALITY**

`ontology-enrich/src/perplexity_client.py` EXISTS and is fully implemented!

### Recommendation
**MERGE OR REUSE** - High priority decision needed

**Option A: Merge into ontology-enrich**
- ontology-enrich already has Perplexity client
- Add MCP wrapper to ontology-enrich
- Remove duplicate perplexity skill

**Option B: Extract shared Perplexity client**
- Move perplexity_client.py to perplexity skill
- Make it a standalone MCP service
- Have ontology-enrich import it

**Recommended: Option A** - Less complexity, ontology-enrich already has the implementation
- Priority: **HIGH** (resolve duplicate code)

---

## Migration Priorities

### Phase 1: Critical (Week 1)
1. ✅ **Resolve Perplexity duplicate** - Merge into ontology-enrich or extract
2. **Migrate ontology-enrich to MCP** - Create FastMCP wrapper
   - Use web-summary as template
   - Expose 4-5 core tools
   - Test with VisionFlow

### Phase 2: Important (Week 2)
3. **Migrate import-to-ontology to MCP** - Create MCP wrapper
   - Fix validator path references
   - Expose import tools
   - Add progress tracking tool

### Phase 3: Optional (Week 3-4)
4. **Implement deepseek-reasoning client** - Or remove if not needed
5. **Document ontology-core** - Add API reference for library users

---

## Dependency Graph

```
ontology-core (Python library)
    ↑
    ├── ontology-enrich (Python + Perplexity client)
    │       ↑
    │       └── (future MCP wrapper)
    │
    └── import-to-ontology (Node.js)
            ↑
            ├── ontology-core (via validation_bridge.js)
            ├── web-summary (for URL enrichment)
            └── (future MCP wrapper)

web-summary (FastMCP) ← Reference implementation
    ↑
    └── Z.AI service (port 9600)

deepseek-reasoning (MCP stub) ← Needs client implementation
perplexity (MCP stub) ← Duplicate of ontology-enrich/perplexity_client.py
```

---

## Key Architectural Decisions

### 1. Keep ontology-core as Library
**Rationale**: It's infrastructure, not a user-facing skill. Multiple skills depend on it.

### 2. Merge Perplexity Implementations
**Rationale**:
- ontology-enrich already has complete Perplexity client
- perplexity skill is just MCP wrapper stub
- No value in maintaining duplicate code

**Action**: Add MCP wrapper to ontology-enrich, expose Perplexity tools, remove standalone perplexity skill

### 3. Use FastMCP for Python Skills
**Rationale**:
- web-summary proves FastMCP works well
- Pure Python, no Node.js complexity
- Clean architecture, easy to maintain

**Action**: Migrate ontology-enrich to FastMCP pattern

### 4. Keep import-to-ontology Node.js
**Rationale**:
- Already implemented in Node.js
- Complex logic working well
- Just needs MCP wrapper, not full rewrite

**Action**: Add MCP SDK wrapper (like deepseek-reasoning pattern)

---

## Implementation Checklist

### Immediate Actions

- [ ] **Audit perplexity vs ontology-enrich** - Verify functionality overlap
- [ ] **Choose merge strategy** - Document decision
- [ ] **Create ontology-enrich MCP wrapper** - Use web-summary template
- [ ] **Test ontology-enrich MCP** - Verify all tools work
- [ ] **Update docs** - Reflect new MCP architecture

### Follow-up Actions

- [ ] **Fix import-to-ontology validator path** - Use relative path
- [ ] **Create import-to-ontology MCP wrapper** - Expose tools
- [ ] **Implement or remove deepseek-reasoning** - Document decision
- [ ] **Document ontology-core API** - For skill developers
- [ ] **Create migration guide** - For other skills to follow

---

## Technical Debt

### High Priority
1. **Perplexity duplication** - Two implementations of same API client
2. **Hardcoded paths** - import-to-ontology uses brittle validator path
3. **Missing MCP wrappers** - 2 fully implemented skills not exposed via MCP

### Medium Priority
4. **Incomplete stubs** - deepseek-reasoning and perplexity MCP stubs without clients
5. **Documentation gaps** - ontology-core lacks API reference docs

### Low Priority
6. **Test coverage** - ontology-core has tests, others may not
7. **Error handling** - Standardize across skills

---

## Recommendations Summary

1. **ontology-core**: Keep as library, improve documentation
2. **ontology-enrich**: Migrate to FastMCP (HIGH priority)
3. **import-to-ontology**: Migrate to MCP SDK (MEDIUM priority)
4. **web-summary**: Keep as-is (reference implementation)
5. **deepseek-reasoning**: Implement client or remove (LOW priority)
6. **perplexity**: Merge into ontology-enrich (HIGH priority)

**Total Skills After Consolidation**: 4-5 skills
- ontology-core (library)
- ontology-enrich (MCP with Perplexity tools)
- import-to-ontology (MCP)
- web-summary (MCP reference)
- (optional) deepseek-reasoning (MCP)

---

## Migration Timeline

**Week 1** (Critical Path):
- Day 1-2: Merge Perplexity implementations
- Day 3-5: Create ontology-enrich FastMCP wrapper
- Test and document

**Week 2** (Important):
- Day 1-3: Create import-to-ontology MCP wrapper
- Day 4-5: Fix validator path, test end-to-end

**Week 3** (Optional):
- Implement deepseek-reasoning client OR remove skill
- Document ontology-core API

**Week 4** (Polish):
- Update all documentation
- Create examples and tutorials
- Verify VisionFlow integration

---

## Success Metrics

- [ ] Zero duplicate code (merge perplexity)
- [ ] All implemented skills have MCP wrappers
- [ ] All MCP servers start successfully
- [ ] VisionFlow can discover and use all tools
- [ ] Documentation updated for all skills
- [ ] Clear library vs skill separation

---

## Appendix: File Counts

| Skill | Total Files | Python | Node.js | MCP Server |
|-------|-------------|--------|---------|------------|
| ontology-core | 6 | 4 | 0 | No |
| ontology-enrich | 7 | 4 | 0 | No |
| import-to-ontology | 13 | 0 | 6 | No |
| web-summary | 5 | 1 | 0 | Yes (FastMCP) |
| deepseek-reasoning | 4 | 0 | 1 | Yes (stub) |
| perplexity | 4 | 0 | 1 | Yes (stub) |

**Total**: 39 files across 6 skills
