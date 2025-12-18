# Skills Directory Refactoring Plan
**Generated**: 2025-12-18
**Status**: Research Complete, Implementation Pending

## Executive Summary

**Total Skills**: 40
**FastMCP Modern**: 3 (imagemagick, qgis, web-summary) - **7.5%**
**Node.js MCP**: 4 (playwright, comfyui, deepseek-reasoning, jupyter-notebooks)
**Script Collections**: 5 (pdf, xlsx, docx, pptx, docs-alignment)
**Prompt-Only**: 28+ (correct pattern, no changes needed)

**Critical Issues Found**: 1
- jupyter-notebooks: No kernel persistence (CRITICAL state management failure)

**High-Priority Refactorings**: 4
**Medium-Priority**: 3
**Token Waste Reduction**: 8,700 tokens/session (96.5% efficiency gain)

---

## Research Findings Summary

### 1. Ontology & Knowledge Cluster

**Status**: Fragmented across 7 skills with inconsistent architecture

| Skill | Type | Status | Action |
|-------|------|--------|--------|
| ontology-core | Library (should be in libs/) | Misplaced | Move to /libs |
| import-to-ontology | Node.js scripts | Outdated | Rewrite FastMCP |
| ontology-enrich | Python library | No MCP | Add FastMCP server |
| web-summary | FastMCP ✅ | Gold Standard | Keep as reference |
| logseq-formatted | Prompt-only | Correct | Keep |
| deepseek-reasoning | Node.js MCP | Functional | Rewrite FastMCP |
| perplexity | Stub | Missing | Implement FastMCP |

**Recommendation**: Merge import-to-ontology + ontology-enrich → **knowledge-graph-mcp**

### 2. Web Engineering Cluster

**Critical Finding**: 100% use case overlap between frontend-design and web-artifacts-builder

| Merger Candidate | Justification | Priority | Effort |
|-----------------|--------------|----------|--------|
| frontend-design + web-artifacts-builder → **frontend-creator** | Guidelines + implementation = complete skill | **HIGH** | Low (2-3h) |
| webapp-testing → playwright | Eliminate script bucket, 80% overlap | MEDIUM | Medium (4-6h) |

**Keep Separate**: chrome-devtools vs playwright (different protocols/use cases)

### 3. Hardware/Science Cluster

**CRITICAL ISSUE**: jupyter-notebooks spawns fresh Python per cell
- No kernel persistence
- Variables lost between cells
- Violates fundamental Jupyter workflow

| Skill | Implementation | Issue | Priority |
|-------|---------------|-------|----------|
| jupyter-notebooks | Node.js MCP | **CRITICAL**: No state persistence | **URGENT** |
| kicad | stdin/stdout wrapper | High subprocess overhead | HIGH |
| ngspice | stdin/stdout wrapper | Fragile tempfile management | HIGH |
| qgis | FastMCP ✅ | Already modern | Keep |

### 4. Document/Media Cluster

**Token Waste**: 4,494 LOC of scripts read per session

| Skill | LOC | Tools Needed | Token Savings | Priority |
|-------|-----|--------------|---------------|----------|
| docx | 2,651 | 6-8 | ~3,800 tokens | **P1** |
| pptx | 2,086 | 6-8 | ~3,100 tokens | **P2** |
| pdf | 757 | 8-10 | ~1,200 tokens | P3 |
| xlsx | ~400 | 5-6 | ~900 tokens | P4 |

**Gold Standard**: latex-documents (comprehensive SKILL.md, no script reading)

### 5. Strategic Cluster

**Platinum Standard**: wardley-maps
- 4-layer architecture (NLP → Heuristics → Analysis → Visualization)
- 2,465 LOC production Python
- spaCy NLP + D3.js interactive output
- Benchmark for complex analytical skills

**Context-Only** (Correct): internal-comms, brand-guidelines

---

## Consolidated Refactoring Plan

### Phase 1: Critical Fixes (Week 1) - **URGENT**

#### 1.1 Fix jupyter-notebooks Kernel Persistence
**Issue**: CRITICAL - No state between cell executions
**Impact**: Breaks fundamental Jupyter workflow
**Effort**: 6-8 hours

```python
# Current (BROKEN):
@mcp.tool()
def execute_cell(code):
    subprocess.run(["/opt/venv/bin/python", "-c", code])  # Fresh process!

# Fixed (PERSISTENT):
kernel_managers = {}  # notebook_path -> KernelManager

@mcp.tool()
def execute_cell(notebook_path, cell_index):
    km = kernel_managers.get(notebook_path) or create_kernel(notebook_path)
    result = km.execute(cell_code)  # Same kernel = persistent state
```

**Files**:
- Rewrite: `skills/jupyter-notebooks/server.js` → `skills/jupyter-notebooks/mcp-server/server.py`
- Add: `requirements.txt` (jupyter_client, ipykernel)
- Update: `SKILL.md` with FastMCP migration notes

#### 1.2 Merge frontend-design + web-artifacts-builder
**Justification**: 100% use case overlap, guidelines without implementation = incomplete
**Effort**: 2-3 hours

**Actions**:
```bash
# Create merged skill
mkdir -p skills/frontend-creator
mv skills/web-artifacts-builder/* skills/frontend-creator/
cat skills/frontend-design/SKILL.md >> skills/frontend-creator/SKILL.md

# Deprecate old skills
mv skills/frontend-design skills/frontend-design.deprecated
mv skills/web-artifacts-builder skills/web-artifacts-builder.deprecated
```

**Update Dockerfile.unified**: Remove old skill copies

---

### Phase 2: High-Impact Wrappers (Weeks 2-3)

#### 2.1 Wrap docx in FastMCP (Priority 1)
**Token Savings**: ~3,800 per session
**Effort**: 8-10 hours

**Tools to expose**:
```python
@mcp.tool()
def create_docx(template: str, data: dict) -> dict

@mcp.tool()
def read_docx(path: str) -> dict

@mcp.tool()
def edit_docx_redline(path: str, changes: list) -> dict

@mcp.tool()
def extract_tracked_changes(path: str) -> dict
```

**Files**:
- Create: `skills/docx/mcp-server/server.py`
- Keep: `skills/docx/tools/document.py` (internal library)
- Update: `SKILL.md` (reference tools, not scripts)

#### 2.2 Wrap pptx in FastMCP (Priority 2)
**Token Savings**: ~3,100 per session
**Effort**: 8-10 hours

**Tools to expose**:
```python
@mcp.tool()
def create_pptx(template: str, slides: list) -> dict

@mcp.tool()
def edit_pptx_template(path: str, replacements: dict) -> dict

@mcp.tool()
def generate_thumbnails(path: str) -> dict

@mcp.tool()
def extract_text(path: str) -> dict
```

#### 2.3 Convert kicad/ngspice to FastMCP
**Benefit**: Eliminate subprocess overhead, add connection pooling
**Effort**: 6-8 hours each

**Pattern**:
```python
# Process pool for kicad-cli
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=2)

@mcp.tool()
def create_schematic(params: SchematicParams) -> dict:
    future = executor.submit(run_kicad_cli, params)
    return future.result(timeout=60)
```

---

### Phase 3: Node.js → FastMCP Migrations (Weeks 3-4)

#### 3.1 Rewrite comfyui as FastMCP
**Current**: Node.js WebSocket client
**Target**: Python httpx async client
**Effort**: 6-8 hours

**Benefits**:
- Z.AI integration consistency
- Python ecosystem alignment
- Async workflow generation
- Better error handling

#### 3.2 Rewrite deepseek-reasoning as FastMCP
**Current**: Node.js with sudo bridging
**Target**: Python subprocess with user switching
**Effort**: 4-6 hours

**Benefits**:
- Simpler sudo management
- No Node.js dependency
- Pydantic validation

#### 3.3 Migrate playwright to FastMCP (Optional)
**Current**: Node.js @mcp/sdk (functional)
**Priority**: Low (works well, only for consistency)
**Effort**: 8-10 hours

---

### Phase 4: VisionFlow Integration (Week 4)

Add VisionFlow resources to all FastMCP skills:

```python
@mcp.resource("skill://capabilities")
def get_capabilities() -> str:
    return json.dumps({
        "name": "skill-name",
        "version": "2.0.0",
        "protocol": "fastmcp",
        "tools": [list_of_tools],
        "visionflow_compatible": True
    })
```

**Target skills**: All FastMCP servers (12 total after migrations)

---

## Docker Build Integration

### Dockerfile.unified Changes Required

#### 1. Remove deprecated skills (Phase 1)
```dockerfile
# DELETE these copy operations:
# COPY --chown=devuser:devuser skills/frontend-design /home/devuser/.claude/skills/frontend-design
# COPY --chown=devuser:devuser skills/web-artifacts-builder /home/devuser/.claude/skills/web-artifacts-builder

# ADD merged skill:
COPY --chown=devuser:devuser skills/frontend-creator /home/devuser/.claude/skills/frontend-creator
```

#### 2. Add FastMCP Python dependencies
```dockerfile
# Phase 2: Document processing dependencies
RUN /opt/venv/bin/pip install \
    python-docx>=0.8.11 \
    python-pptx>=0.6.21 \
    pypdf>=3.0.0 \
    openpyxl>=3.1.0 \
    # Jupyter kernel management (Phase 1)
    jupyter-client>=8.0.0 \
    ipykernel>=6.25.0 \
    # KiCAD/ngspice (Phase 2)
    psutil>=5.9.0
```

#### 3. Update supervisord.conf
```ini
# UPDATE jupyter-notebooks to use Python FastMCP
[program:jupyter-notebooks-mcp]
command=/opt/venv/bin/python /home/devuser/.claude/skills/jupyter-notebooks/mcp-server/server.py
directory=/home/devuser/.claude/skills/jupyter-notebooks/mcp-server
user=devuser
environment=HOME="/home/devuser",JUPYTER_NOTEBOOK_DIR="/home/devuser/workspace"

# ADD docx MCP server (Phase 2)
[program:docx-mcp]
command=/opt/venv/bin/python /home/devuser/.claude/skills/docx/mcp-server/server.py
directory=/home/devuser/.claude/skills/docx/mcp-server
user=devuser

# ADD pptx MCP server (Phase 2)
[program:pptx-mcp]
command=/opt/venv/bin/python /home/devuser/.claude/skills/pptx/mcp-server/server.py
directory=/home/devuser/.claude/skills/pptx/mcp-server
user=devuser
```

---

## Testing Strategy

### After Each Phase:

```bash
# 1. Rebuild container
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker
docker build --no-cache -f Dockerfile.unified -t agentic-workstation:latest .

# 2. Restart container
docker-compose -f docker-compose.unified.yml down
docker-compose -f docker-compose.unified.yml up -d

# 3. Verify skills loaded
docker exec agentic-workstation /opt/venv/bin/supervisorctl status | grep mcp

# 4. Test specific skill
docker exec -u devuser agentic-workstation bash -c "
  cd ~/.claude/skills/<skill-name> && \
  /opt/venv/bin/python mcp-server/server.py --test
"
```

### Validation Checklist per Skill:

- [ ] MCP server starts without errors
- [ ] Tools discoverable via MCP protocol
- [ ] VisionFlow resource endpoint responds
- [ ] No token waste (SKILL.md references tools, not scripts)
- [ ] supervisorctl shows RUNNING status
- [ ] Integration test passes (end-to-end workflow)

---

## Success Metrics

### Quantitative:
- **Token efficiency**: 96.5% reduction (8,700 → 300 tokens/session)
- **FastMCP coverage**: 7.5% → 30% (3 → 12 skills)
- **Critical bugs fixed**: 1 (jupyter-notebooks kernel persistence)
- **Deprecated skills**: 2 (frontend-design, web-artifacts-builder)

### Qualitative:
- Consistent FastMCP architecture across document/media skills
- No Node.js dependencies for Python-native workflows
- VisionFlow resource discovery on all MCP servers
- Jupyter kernel persistence matches user expectations

---

## Risk Mitigation

### Rollback Strategy:
- Keep `.deprecated` folders until Phase 4 complete
- Git tag before each phase: `git tag phase-1-pre-refactor`
- Volume mount preserves data during container rebuilds

### Compatibility:
- Maintain old SKILL.md examples during transition
- Add deprecation notices to old skills
- Document migration guide for users

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 (Critical) | Week 1 | jupyter-notebooks fixed, frontend-creator merged |
| Phase 2 (Wrappers) | Weeks 2-3 | docx, pptx, pdf, kicad, ngspice FastMCP |
| Phase 3 (Node→Python) | Weeks 3-4 | comfyui, deepseek-reasoning FastMCP |
| Phase 4 (VisionFlow) | Week 4 | All skills have resources |

**Total**: 4 weeks

---

## Next Steps

1. **Immediate**: Fix jupyter-notebooks (CRITICAL)
2. **Week 1**: Merge frontend-creator, test rebuild
3. **Week 2**: Start docx/pptx wrappers
4. **Continuous**: Update Dockerfile.unified with each change
5. **Weekly**: Container rebuild + validation tests
