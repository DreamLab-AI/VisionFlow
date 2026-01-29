---
title: Ontology & Knowledge Skills Cluster Analysis
description: Analysis reveals **significant architectural inconsistency** across the ontology skills cluster.  Only **web-summary** implements the "Best in Class" FastMCP pattern (Python-only with direct stdio).
category: explanation
tags:
  - architecture
  - patterns
  - structure
  - api
  - api
related-docs:
  - analysis/ontology-knowledge-skills-analysis.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Node.js runtime
---

# Ontology & Knowledge Skills Cluster Analysis

**Date:** 2025-12-18
**Analyst:** Research Agent
**Scope:** 7 skills in ontology/knowledge domain

## Executive Summary

Analysis reveals **significant architectural inconsistency** across the ontology skills cluster. Only **web-summary** implements the "Best in Class" FastMCP pattern (Python-only with direct stdio). The remaining skills are either missing, incomplete, or use hybrid architectures that conflict with the FastMCP standard established by the Blender skill.

### Critical Findings

1. **2/7 skills missing entirely** (ontology-core, ontology-enrich)
2. **1/7 uses Best in Class architecture** (web-summary with FastMCP)
3. **4/7 are stubs with no MCP implementation** (import-to-ontology, logseq-formatted, deepseek-reasoning, perplexity)
4. **Dependency confusion:** import-to-ontology references non-existent ontology-core MCP server

---

## Skill-by-Skill Analysis

### 1. import-to-ontology

**Status:** ❌ Script collection, no MCP server
**Architecture Type:** Node.js scripts with validation bridge
**MCP Server:** None
**Dependencies:** Expects ontology-core as MCP server (doesn't exist)

**File Structure:**
```
import-to-ontology/
├── SKILL.md (26KB - extensive documentation)
├── import-engine.js (standalone script)
├── destructive-import.js (standalone script)
├── llm-matcher.js (helper)
├── asset-handler.js (helper)
├── package.json (no dependencies!)
└── src/validation_bridge.js (expects ontology-core)
```

**Problems:**
- **No MCP implementation** - Just standalone scripts
- **Circular dependency** - References ontology-core for validation via Python subprocess
- **Manual execution required** - No tool exposure to Claude Code
- **Inconsistent with Best in Class** - Should be FastMCP Python server

**Recommended Architecture:**
```python
# import-to-ontology/mcp-server/server.py (FastMCP)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("import-to-ontology")

@mcp.tool()
async def import_content(source_file: str, target_ontology: str, dry_run: bool) -> dict:
    """Import content blocks with OWL2 validation and semantic targeting."""
    # Use ontology-core library directly (not as MCP server)
    from ontology_core.src.owl2_validator import OWL2Validator
    from ontology_core.src.ontology_parser import parse_ontology_block
    # Implementation...
```

---

### 2. ontology-core

**Status:** ❌ Missing SKILL.md, Python library only
**Architecture Type:** Python library (not MCP server)
**MCP Server:** None (should not be one)
**Dependencies:** None

**File Structure:**
```
ontology-core/
├── skill.md (2KB - minimal, not SKILL.md)
├── src/
│   ├── owl2_validator.py (validation logic)
│   ├── ontology_parser.py (parsing logic)
│   └── ontology_modifier.py (modification logic)
├── tests/ (comprehensive test suite)
└── requirements.txt
```

**Problems:**
- **Missing SKILL.md** - Has old "skill.md" instead
- **Not a skill** - This is a library, not an MCP server
- **Confusion in architecture** - import-to-ontology expects this to be an MCP server

**Correct Role:**
- **Should be a Python library dependency** for other skills
- **Should NOT have MCP server** - just pure Python functions
- **Should be imported by import-to-ontology and ontology-enrich**

**Recommendation:**
```bash
# ontology-core should be:
ontology-core/
├── README.md (library documentation)
├── pyproject.toml (Python package metadata)
├── src/ontology_core/ (importable package)
└── tests/

# NOT a skill - remove from skills/ directory
# Install as: pip install -e ./ontology-core
```

---

### 3. ontology-enrich

**Status:** ❌ Missing SKILL.md, incomplete implementation
**Architecture Type:** Hybrid Python (no MCP server)
**MCP Server:** None
**Dependencies:** ontology-core (as library), Perplexity API

**File Structure:**
```
ontology-enrich/
├── skill.md (9KB - old format)
├── README.md
├── config/
├── src/
│   ├── enrichment_workflow.py (imports from ontology_core)
│   ├── perplexity_client.py
│   └── link_validator.py
└── requirements.txt
```

**Problems:**
- **Missing SKILL.md** - Has old "skill.md"
- **No MCP server** - Just Python scripts
- **Unclear how to invoke** - No tool exposure

**Recommended Architecture:**
```python
# ontology-enrich/mcp-server/server.py (FastMCP)
from mcp.server.fastmcp import FastMCP
from ontology_core.src.owl2_validator import OWL2Validator
from .enrichment_workflow import EnrichmentWorkflow

mcp = FastMCP("ontology-enrich")

@mcp.tool()
async def enrich_definition(file_path: str, field: str, query: str) -> dict:
    """Enrich ontology field using Perplexity with OWL2 validation and rollback."""
    # Implementation using EnrichmentWorkflow
```

---

### 4. web-summary ✅

**Status:** ✅ Best in Class FastMCP implementation
**Architecture Type:** FastMCP (Python-only)
**MCP Server:** server.py (381 lines, complete)
**Dependencies:** httpx, youtube-transcript-api, Z.AI service (port 9600)

**File Structure:**
```
web-summary/
├── SKILL.md (3.4KB - complete)
├── mcp-server/
│   └── server.py (FastMCP implementation)
├── examples/
└── tools/
```

**Strengths:**
- **Pure Python FastMCP** - Single server.py file
- **4 MCP tools exposed:** summarize_url, youtube_transcript, generate_topics, health_check
- **Clear protocol:** stdio with Pydantic models
- **Z.AI integration** - Cost-effective Claude API
- **Resource exposure** - VisionFlow discovery via web-summary://capabilities
- **Complete documentation** - SKILL.md matches implementation

**This is the standard all other skills should follow.**

---

### 5. logseq-formatted

**Status:** ⚠️ Skill documentation only, no MCP server
**Architecture Type:** None (pure documentation skill)
**MCP Server:** None
**Dependencies:** None

**File Structure:**
```
logseq-formatted/
└── SKILL.md (835 lines - comprehensive guide)
```

**Analysis:**
- **Not a tool skill** - This is a writing style guide
- **No MCP server needed** - Just prompting instructions
- **Correctly implemented** - Documentation-only skills are valid
- **No changes needed** - This is appropriate as-is

**Verdict:** ✅ Correct architecture for documentation skill

---

### 6. deepseek-reasoning

**Status:** ❌ Stub with documentation, no implementation
**Architecture Type:** Planned MCP-SDK (Node.js)
**MCP Server:** Mentioned in SKILL.md but not implemented
**Dependencies:** deepseek-api, planned user isolation (deepseek-user)

**File Structure:**
```
deepseek-reasoning/
└── SKILL.md (385 lines - detailed spec)
```

**Problems:**
- **No implementation** - Only documentation exists
- **Wrong protocol specified** - Says "mcp-sdk" instead of FastMCP
- **User isolation planned** - Unnecessarily complex
- **Should use FastMCP** - Simpler Python implementation

**Recommended Architecture:**
```python
# deepseek-reasoning/mcp-server/server.py (FastMCP)
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("deepseek-reasoning")

@mcp.tool()
async def deepseek_reason(query: str, context: str, max_steps: int = 10) -> dict:
    """Complex multi-step reasoning via DeepSeek special model."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            os.environ["DEEPSEEK_API_ENDPOINT"],
            json={"prompt": query, "context": context},
            headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}
        )
        return response.json()
```

**No need for user isolation** - Use environment variables like web-summary does with Z.AI.

---

### 7. perplexity

**Status:** ❌ Stub with documentation, no implementation
**Architecture Type:** Planned MCP-SDK (Node.js)
**MCP Server:** Mentioned in SKILL.md but not implemented
**Dependencies:** perplexity-sdk

**File Structure:**
```
perplexity/
└── SKILL.md (248 lines - API documentation)
```

**Problems:**
- **No implementation** - Only documentation
- **Wrong protocol** - Says "mcp-sdk" instead of FastMCP
- **Duplicates web-summary** - Web research already covered

**Recommended Architecture:**
```python
# perplexity/mcp-server/server.py (FastMCP)
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("perplexity")

@mcp.tool()
async def perplexity_search(query: str, uk_focus: bool = True) -> dict:
    """Quick factual search with citations via Perplexity Sonar API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": query}]
            },
            headers={"Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}"}
        )
        return response.json()
```

---

## Interdependencies

```
import-to-ontology (Node.js scripts)
    ├─> ontology-core (expects MCP server, actually Python library)
    └─> web-summary (for URL enrichment) ✅

ontology-enrich (Python scripts, no MCP)
    ├─> ontology-core (correctly imports as Python library) ✅
    └─> perplexity (expects MCP server, doesn't exist) ❌

web-summary (FastMCP Python)
    └─> Z.AI service (port 9600, working) ✅

logseq-formatted (documentation only)
    └─> No dependencies ✅

deepseek-reasoning (stub)
    └─> deepseek-user isolation (unnecessary complexity) ❌

perplexity (stub)
    └─> No dependencies
```

---

## Redundancies

### Content Enrichment Overlap

**web-summary vs perplexity:**
- Both do web research with citations
- web-summary uses Z.AI (cost-effective)
- perplexity uses Perplexity API (paid)

**Recommendation:**
- Keep web-summary for general summarization (YouTube, web pages)
- Implement perplexity only for specialized research queries requiring Sonar model

### Validation Redundancy

**import-to-ontology vs ontology-enrich:**
- Both validate OWL2 compliance
- Both use ontology-core library
- Both have rollback mechanisms

**Recommendation:**
- Consolidate validation logic in ontology-core library
- Both skills import from single source of truth

---

## Best in Class Standard: Blender Skill

The Blender skill represents the ideal architecture (despite not using FastMCP):

**Key Characteristics:**
1. **Single entry point** - One server file per protocol
2. **Clear tool definitions** - 52 tools with consistent patterns
3. **Thread-safe execution** - Background server, main thread dispatcher
4. **Simple configuration** - mcp.json for Claude Code integration
5. **Comprehensive documentation** - SKILL.md matches implementation

**For Python skills, FastMCP is even simpler:**

```python
# Minimal FastMCP pattern (from web-summary)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("skill-name")

class ToolParams(BaseModel):
    field: str

@mcp.tool()
async def tool_name(params: ToolParams) -> dict:
    """Tool description."""
    return {"result": "..."}

if __name__ == "__main__":
    mcp.run()
```

---

## Specific Refactoring Recommendations

### Priority 1: Fix Core Architecture

#### 1. ontology-core → Python Library (Not a Skill)

**Action:** Remove from skills/, install as package

```bash
# Current (wrong):
skills/ontology-core/src/owl2_validator.py

# Target (correct):
libs/ontology-core/
├── pyproject.toml
├── src/ontology_core/
│   ├── __init__.py
│   ├── owl2_validator.py
│   ├── ontology_parser.py
│   └── ontology_modifier.py
└── tests/

# Install:
pip install -e ./libs/ontology-core
```

**Files to move:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/ontology-core/` → `./libs/ontology-core/`

#### 2. import-to-ontology → FastMCP Python Server

**Action:** Rewrite as FastMCP server

**Current:**
```
import-to-ontology/
├── import-engine.js (Node.js script)
└── package.json
```

**Target:**
```
import-to-ontology/
├── SKILL.md (keep existing)
├── mcp-server/
│   └── server.py (FastMCP - new)
└── examples/
```

**Implementation:**
```python
# /home/devuser/workspace/project/multi-agent-docker/skills/import-to-ontology/mcp-server/server.py
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from ontology_core.owl2_validator import OWL2Validator
from ontology_core.ontology_parser import parse_ontology_block

mcp = FastMCP("import-to-ontology", version="2.0.0")

class ImportParams(BaseModel):
    source_file: str
    target_ontology: str
    dry_run: bool = True

@mcp.tool()
async def import_content(params: ImportParams) -> dict:
    """Import content blocks with OWL2 validation and semantic targeting."""
    # Parse source
    blocks = parse_markdown_blocks(params.source_file)

    # Validate target
    validator = OWL2Validator()
    pre_validation = validator.validate_file(params.target_ontology)

    if not pre_validation.is_valid:
        return {"success": False, "error": "Target file has OWL2 errors"}

    # Import logic...
    return {"success": True, "blocks_imported": len(blocks)}

if __name__ == "__main__":
    mcp.run()
```

**Files to create:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/import-to-ontology/mcp-server/server.py`

**Files to deprecate (keep for reference):**
- `import-engine.js` → `legacy/import-engine.js`
- `destructive-import.js` → `legacy/destructive-import.js`

#### 3. ontology-enrich → FastMCP Python Server

**Action:** Add FastMCP server to existing Python code

**Current:**
```
ontology-enrich/
├── src/enrichment_workflow.py
└── src/perplexity_client.py
```

**Target:**
```
ontology-enrich/
├── SKILL.md (update with tool docs)
├── mcp-server/
│   └── server.py (FastMCP - new)
├── src/ (keep as library code)
└── config/
```

**Implementation:**
```python
# /home/devuser/workspace/project/multi-agent-docker/skills/ontology-enrich/mcp-server/server.py
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from ..src.enrichment_workflow import EnrichmentWorkflow
import os

mcp = FastMCP("ontology-enrich", version="1.0.0")

class EnrichParams(BaseModel):
    file_path: str
    field: str
    query: str

@mcp.tool()
async def enrich_field(params: EnrichParams) -> dict:
    """Enrich ontology field with Perplexity research and OWL2 validation."""
    workflow = EnrichmentWorkflow(
        api_key=os.environ["PERPLEXITY_API_KEY"]
    )

    result = await workflow.enrich_field(
        params.file_path,
        params.field,
        params.query
    )

    return result.to_dict()

if __name__ == "__main__":
    mcp.run()
```

**Files to create:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/ontology-enrich/mcp-server/server.py`
- `/home/devuser/workspace/project/multi-agent-docker/skills/ontology-enrich/SKILL.md` (replace skill.md)

### Priority 2: Implement Stubs

#### 4. deepseek-reasoning → FastMCP Python Server

**Action:** Implement FastMCP server (remove user isolation complexity)

```python
# /home/devuser/workspace/project/multi-agent-docker/skills/deepseek-reasoning/mcp-server/server.py
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import httpx
import os

mcp = FastMCP("deepseek-reasoning", version="1.0.0")

class ReasonParams(BaseModel):
    query: str
    context: str = ""
    max_steps: int = 10

@mcp.tool()
async def deepseek_reason(params: ReasonParams) -> dict:
    """Complex multi-step reasoning via DeepSeek special model."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            os.environ.get("DEEPSEEK_API_ENDPOINT", "https://api.deepseek.com/chat/completions"),
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Think step by step."},
                    {"role": "user", "content": f"Context: {params.context}\n\nQuery: {params.query}"}
                ],
                "max_tokens": 2000
            },
            headers={
                "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
        return response.json()

if __name__ == "__main__":
    mcp.run()
```

**Files to create:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/deepseek-reasoning/mcp-server/server.py`

**Files to update:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/deepseek-reasoning/SKILL.md` (change protocol from mcp-sdk to fastmcp, remove user isolation)

#### 5. perplexity → FastMCP Python Server

**Action:** Implement FastMCP server with Perplexity Sonar API

```python
# /home/devuser/workspace/project/multi-agent-docker/skills/perplexity/mcp-server/server.py
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import httpx
import os

mcp = FastMCP("perplexity", version="2.0.0")

class SearchParams(BaseModel):
    query: str
    uk_focus: bool = True
    timeframe: str = "30d"

@mcp.tool()
async def perplexity_search(params: SearchParams) -> dict:
    """Quick factual search with citations via Perplexity Sonar API."""
    prompt = params.query
    if params.uk_focus:
        prompt = f"UK-focused: {prompt}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "search_recency_filter": params.timeframe
            },
            headers={
                "Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
        return response.json()

if __name__ == "__main__":
    mcp.run()
```

**Files to create:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/perplexity/mcp-server/server.py`

**Files to update:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/perplexity/SKILL.md` (change protocol to fastmcp)

### Priority 3: Documentation Updates

#### 6. Standardize SKILL.md Format

All skills must have SKILL.md (not skill.md) with:

```yaml
---
name: skill-name
description: Brief description
version: x.y.z
author: Author Name
mcp_server: true/false
protocol: fastmcp|websocket|none
entry_point: mcp-server/server.py
dependencies:
  - package1
  - package2
---

# Skill Name

[Documentation following Blender pattern]
```

**Files to update:**
- `/home/devuser/workspace/project/multi-agent-docker/skills/ontology-enrich/skill.md` → `SKILL.md`
- `/home/devuser/workspace/project/multi-agent-docker/skills/ontology-core/skill.md` → Delete (not a skill)

---

## Summary of Changes

| Skill | Current State | Target State | Effort | Priority |
|-------|--------------|-------------|--------|----------|
| **ontology-core** | Python library in skills/ | Separate package in libs/ | Medium | High |
| **import-to-ontology** | Node.js scripts | FastMCP Python server | High | High |
| **ontology-enrich** | Python scripts, no MCP | FastMCP Python server | Medium | High |
| **web-summary** | FastMCP (complete) ✅ | No changes needed | None | N/A |
| **logseq-formatted** | Documentation only ✅ | No changes needed | None | N/A |
| **deepseek-reasoning** | Stub | FastMCP Python server | Low | Medium |
| **perplexity** | Stub | FastMCP Python server | Low | Medium |

---

## Implementation Roadmap

### Phase 1: Core Architecture (Week 1)

1. **Extract ontology-core as library**
   - Create `libs/ontology-core/` with pyproject.toml
   - Move Python modules to `src/ontology_core/`
   - Install as editable package: `pip install -e ./libs/ontology-core`
   - Update import-to-ontology and ontology-enrich to use package

2. **Rewrite import-to-ontology as FastMCP**
   - Create `mcp-server/server.py` with 3 tools:
     - `import_content()` - Main import with validation
     - `dry_run_analysis()` - Preview without changes
     - `validate_target()` - Pre-check OWL2 compliance
   - Use ontology_core as library
   - Keep legacy Node.js scripts in `legacy/` folder

3. **Add FastMCP server to ontology-enrich**
   - Create `mcp-server/server.py` with 2 tools:
     - `enrich_field()` - Enrich with Perplexity + validation
     - `validate_enrichment()` - Check before applying
   - Wrap existing enrichment_workflow.py
   - Update SKILL.md with tool documentation

### Phase 2: Implement Stubs (Week 2)

4. **Implement deepseek-reasoning FastMCP**
   - Create `mcp-server/server.py` with 3 tools:
     - `deepseek_reason()` - Multi-step reasoning
     - `deepseek_analyze()` - Code/system analysis
     - `deepseek_plan()` - Task planning
   - Use environment variables (no user isolation)
   - Update SKILL.md to match implementation

5. **Implement perplexity FastMCP**
   - Create `mcp-server/server.py` with 3 tools:
     - `perplexity_search()` - Quick search
     - `perplexity_research()` - Deep research
     - `perplexity_generate_prompt()` - Prompt optimization
   - Follow web-summary pattern
   - Update SKILL.md to match implementation

### Phase 3: Testing & Integration (Week 3)

6. **Integration testing**
   - Test import-to-ontology with ontology-core library
   - Test ontology-enrich with Perplexity API
   - Test deepseek-reasoning with DeepSeek API
   - Test perplexity with Perplexity API

7. **Documentation audit**
   - Standardize all SKILL.md files
   - Add mcp.json for each skill
   - Create integration examples
   - Update main SKILLS.md with new architecture

---

---

## Related Documentation

- [Ontology/Knowledge Skills Analysis](ontology-knowledge-skills-analysis.md)
- [VisionFlow Documentation Modernization - Final Report](../DOCUMENTATION_MODERNIZATION_COMPLETE.md)
- [X-FluxAgent Integration Plan for ComfyUI MCP Skill](../multi-agent-docker/x-fluxagent-adaptation-plan.md)
- [Blender MCP Unified System Architecture](../architecture/blender-mcp-unified-architecture.md)
- [Server Architecture](../concepts/architecture/core/server.md)

## Conclusion

The ontology skills cluster requires **significant refactoring** to achieve architectural consistency. Only web-summary currently implements the FastMCP standard correctly. The primary issues are:

1. **ontology-core is misplaced** - Should be a library, not a skill
2. **import-to-ontology has wrong architecture** - Node.js scripts instead of FastMCP
3. **Two stubs need implementation** - deepseek-reasoning and perplexity
4. **ontology-enrich needs MCP server** - Has library code but no tools

Total estimated effort: **3-4 weeks** for complete refactoring with testing.

**Immediate Action Items:**
1. Move ontology-core to libs/ and install as package
2. Rewrite import-to-ontology as FastMCP Python server
3. Add FastMCP server to ontology-enrich
4. Implement deepseek-reasoning and perplexity as FastMCP servers
5. Standardize all SKILL.md documentation

This will bring the entire cluster into alignment with the Blender skill's "Best in Class" pattern, ensuring consistent tooling for Claude Code integration.
