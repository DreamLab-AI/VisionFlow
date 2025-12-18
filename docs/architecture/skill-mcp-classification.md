---
title: Skill MCP Implementation Classification
description: Based on file structure analysis and code inspection of 6 representative skills:
category: explanation
tags:
  - architecture
  - design
  - patterns
  - structure
  - api
related-docs:
  - architecture/HEXAGONAL_ARCHITECTURE_STATUS.md
  - architecture/blender-mcp-unified-architecture.md
  - architecture/phase1-completion.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Node.js runtime
---

# Skill MCP Implementation Classification

**Analysis Date**: 2025-12-18
**Total Skills**: 40
**Scope**: /home/devuser/workspace/project/multi-agent-docker/skills/

## Summary

Based on file structure analysis and code inspection of 6 representative skills:

### Implementation Types

| Type | Count | Examples | Description |
|------|-------|----------|-------------|
| **FastMCP (Python)** | 3 | imagemagick, qgis, web-summary | Modern Python SDK with Pydantic models |
| **@mcp/sdk (Node.js)** | 4 | playwright, comfyui, deepseek-reasoning, jupyter-notebooks | Node.js SDK with JSON-RPC |
| **Prompt-Only** | 28+ | algorithmic-art, brand-guidelines, canvas-design | SKILL.md only, no server code |
| **Script Collections** | 5+ | docs-alignment, wardley-maps, slack-gif-creator | Utility scripts, no MCP server |

---

## Category 1: FastMCP Python Servers (Modern)

**Pattern**: Uses `mcp.server.fastmcp.FastMCP` with Pydantic parameter validation

### ✅ imagemagick (FastMCP 2.0)
**Location**: `skills/imagemagick/mcp-server/server.py`

**Features**:
- FastMCP SDK with Pydantic models
- 7 tools: create_image, convert_image, resize_image, crop_image, composite_images, identify_image, batch_process
- Structured error handling
- Environment configuration (ImageMagick version detection)
- VisionFlow integration via `@mcp.resource()` decorator
- No TCP proxy required (stdio transport)

**Key Code Pattern**:
```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("imagemagick", version="2.0.0", description="...")

class ResizeParams(BaseModel):
    input_path: str = Field(..., description="...")
    width: int = Field(..., ge=1, le=10000)

@mcp.tool()
def resize_image(params: ResizeParams) -> dict:
    return run_imagemagick([...])

if __name__ == "__main__":
    mcp.run()
```

**Migration Path**: ✅ Already modernized (Era 3)

---

### ✅ qgis (FastMCP 2.0 + TCP Client)
**Location**: `skills/qgis/mcp-server/server.py`

**Features**:
- FastMCP SDK + Pydantic models
- TCP socket client to QGIS instance (port 9877)
- 11 tools: load_layer, buffer_analysis, calculate_distance, transform_coordinates, export_map, etc.
- Connection pooling with context managers
- Custom exceptions: QGISConnectionError, QGISCommandError
- Health check with connection diagnostics
- VisionFlow resources: `qgis://capabilities`, `qgis://status`

**Key Code Pattern**:
```python
from mcp.server.fastmcp import FastMCP
from contextlib import contextmanager

@contextmanager
def qgis_connection(host: str = QGIS_HOST, port: int = QGIS_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    yield sock
    sock.close()

def send_qgis_command(command_type: str, params: Dict) -> Dict:
    with qgis_connection() as sock:
        sock.sendall(json.dumps({"type": command_type, "params": params}).encode())
        return json.loads(sock.recv(4096).decode())

@mcp.tool()
def buffer_analysis(params: BufferParams) -> dict:
    result = send_qgis_command("buffer", {...})
    return {"success": True, "result": result}
```

**Architecture**:
- MCP Server (stdio) → TCP → QGIS instance (port 9877)
- QGIS runs with addon that listens on TCP socket
- Two-layer protocol: MCP (stdio) + QGIS (TCP)

**Migration Path**: ✅ Already modernized (Era 3)

---

### ✅ web-summary (FastMCP 2.0 + Z.AI Integration)
**Location**: `skills/web-summary/mcp-server/server.py`

**Features**:
- FastMCP SDK with async/await
- 4 tools: summarize_url, youtube_transcript, generate_topics, health_check
- Integrated with Z.AI service (port 9600) for cost-effective LLM calls
- YouTube transcript extraction via `youtube-transcript-api`
- Logseq/Obsidian topic formatting
- VisionFlow resource: `web-summary://capabilities`

**Key Code Pattern**:
```python
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("web-summary", version="2.0.0")

async def call_zai(prompt: str, max_tokens: int = 2000) -> dict:
    async with httpx.AsyncClient(timeout=ZAI_TIMEOUT) as client:
        response = await client.post(ZAI_URL, json={"prompt": prompt})
        return {"success": True, "content": response.json()["content"]}

@mcp.tool()
async def summarize_url(params: SummarizeUrlParams) -> dict:
    # Fetch URL content
    if is_youtube_url(params.url):
        content = (await fetch_youtube_transcript(video_id))["transcript"]
    else:
        content = (await fetch_url_content(params.url))["content"]

    # Summarize via Z.AI
    summary = await call_zai(f"Summarize: {content}")
    return {"success": True, "summary": summary["content"]}
```

**Architecture**:
- MCP Server (stdio) → Z.AI (HTTP port 9600) → Anthropic API
- Eliminated Node.js wrapper from Era 1
- Direct Python implementation

**Migration Path**: ✅ Already modernized (Era 3)

---

## Category 2: @mcp/sdk Node.js Servers

**Pattern**: Uses `@modelcontextprotocol/sdk` with JSON-RPC over stdio

### ✅ playwright (MCP SDK 2.0)
**Location**: `skills/playwright/mcp-server/server.js`

**Features**:
- @modelcontextprotocol/sdk/server
- StdioServerTransport (no TCP proxy)
- 10 tools: navigate, screenshot, click, type, evaluate, wait_for_selector, etc.
- Direct browser launch on Display :1 (VNC)
- Screenshot capture to /tmp/playwright-screenshots
- VisionFlow resources: `playwright://capabilities`, `playwright://status`
- Lazy-loaded playwright dependency

**Key Code Pattern**:
```javascript
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');

const server = new Server(
    { name: 'playwright', version: '2.0.0' },
    { capabilities: { tools: {}, resources: {} } }
);

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    if (name === 'navigate') {
        await page.goto(args.url);
        return { content: [{ type: 'text', text: JSON.stringify({success: true}) }] };
    }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

**Architecture**:
- Consolidated from 3 separate scripts (client, proxy, local)
- Direct browser control (no TCP intermediary)
- Eliminated proxy complexity

**Migration Path**: ✅ Already modernized (Era 2)

---

### ✅ comfyui (MCP SDK 1.0)
**Location**: `skills/comfyui/mcp-server/server.js`

**Features**:
- @modelcontextprotocol/sdk/server
- WebSocket client to ComfyUI instance (port 8188)
- 9 tools: workflow_submit, workflow_status, image_generate, video_generate, model_list, etc.
- Job tracking with progress monitoring
- LLM-powered workflow generation (chat_workflow)
- Display capture via ImageMagick
- Structured response formatting

**Key Code Pattern**:
```javascript
class ComfyUIClient extends EventEmitter {
    constructor(serverUrl = 'http://localhost:8188') {
        this.ws = new WebSocket(serverUrl.replace('http', 'ws') + '/ws');
        this.jobTracking = new Map();
    }

    async submitWorkflow(workflow) {
        const response = await this.httpRequest('/prompt', 'POST', { prompt: workflow });
        this.jobTracking.set(response.prompt_id, { status: 'queued' });
        return { jobId: response.prompt_id };
    }

    async chatToWorkflow(prompt, llmEndpoint = 'http://localhost:9600/chat') {
        const response = await fetch(llmEndpoint, {
            method: 'POST',
            body: JSON.stringify({ prompt: `Convert to ComfyUI workflow: ${prompt}` })
        });
        return { workflow: await response.json() };
    }
}

class ComfyUIMCPServer {
    async handleToolCall(toolName, params) {
        if (toolName === 'image_generate') {
            const workflow = this.client.buildText2ImgWorkflow(params);
            return await this.client.submitWorkflow(workflow);
        }
    }
}
```

**Architecture**:
- MCP Server (stdio) → ComfyUI (HTTP + WebSocket port 8188)
- Optional Z.AI integration for workflow generation
- Progress tracking via WebSocket events

**Migration Path**: Consider FastMCP Python rewrite for consistency

---

### ✅ deepseek-reasoning (MCP SDK 1.0 + Multi-User)
**Location**: `skills/deepseek-reasoning/mcp-server/server.js`

**Features**:
- @modelcontextprotocol/sdk/server
- Multi-user bridge: devuser → deepseek-user
- 3 tools: deepseek_reason, deepseek_analyze, deepseek_plan
- Executes client as deepseek-user via sudo
- JSON-RPC over stdin/stdout
- Structured reasoning with chain-of-thought

**Key Code Pattern**:
```javascript
async function _executeAsDeepSeekUser(args) {
    const sudoArgs = ['-u', 'deepseek-user', 'node', TOOL_PATH, ...args];
    const proc = spawn('sudo', sudoArgs);

    let stdout = '';
    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.on('close', (code) => {
        const result = JSON.parse(stdout);
        resolve(result);
    });
}

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const result = await _executeAsDeepSeekUser(['--tool', name, '--params', JSON.stringify(args)]);
    return { content: [{ type: 'text', text: JSON.stringify(result) }] };
});
```

**Architecture**:
- MCP Server (devuser) → sudo → deepseek-user → DeepSeek API
- Credential isolation via Linux user separation
- JSON-RPC protocol maintained across sudo boundary

**Migration Path**: Consider FastMCP Python rewrite with user switching

---

### ✅ jupyter-notebooks (MCP SDK - Minimal)
**Location**: `skills/jupyter-notebooks/server.js`

**Features**:
- Assumed @modelcontextprotocol/sdk (based on directory structure)
- Notebook execution and manipulation
- Cell-level operations

**Migration Path**: Needs inspection (not yet reviewed)

---

## Category 3: Prompt-Only Skills (No Server)

**Pattern**: Only SKILL.md file with Claude Code instructions, no executable server

### Examples (28+ skills)
- algorithmic-art
- brand-guidelines
- canvas-design
- frontend-design
- internal-comms
- latex-documents
- logseq-formatted
- rust-development
- theme-factory
- web-artifacts-builder
- pyplot-ml
- ffmpeg-processing
- And 16+ more...

**Characteristics**:
- No server.py or server.js
- No mcp.json configuration
- Purely instructional prompts for Claude Code
- Use built-in Claude Code tools (Bash, Read, Write, Edit)

**Migration Path**: No migration needed (by design)

---

## Category 4: Script Collections (No MCP Server)

**Pattern**: Utility scripts without MCP protocol wrapper

### Examples

#### docs-alignment
**Location**: `skills/docs-alignment/scripts/`

**Scripts**:
- archive_working_docs.py
- validate_links.py
- check_mermaid.py
- docs_alignment.py
- generate_report.py
- detect_ascii.py
- scan_stubs.py

**Migration Path**: Could wrap as FastMCP server with tools for each script

---

#### wardley-maps
**Location**: `skills/wardley-maps/tools/`

**Scripts**:
- generate_wardley_map.py
- strategic_analyzer.py
- heuristics_engine.py
- interactive_map_generator.py
- advanced_nlp_parser.py
- wardley_mapper.py
- quick_map.py

**Migration Path**: Could wrap as FastMCP server with SVG generation tools

---

#### slack-gif-creator
**Location**: `skills/slack-gif-creator/`

**Structure**:
- core/ (gif_builder.py, typography.py, visual_effects.py, etc.)
- templates/ (bounce.py, spin.py, zoom.py, etc.)

**Migration Path**: Could wrap as FastMCP server with GIF generation tool

---

## Category 5: Legacy/Hybrid Implementations

### blender (Unified Architecture - In Progress)
**Location**: `skills/blender/`

**Current State**:
- Unified addon architecture (dispatcher.py, server.py)
- TCP socket server (port 2800) in Blender addon
- Tools split across addon/tools/ modules
- FastMCP client planned but not yet implemented

**Files**:
- addon/__init__.py - Blender addon registration
- addon/dispatcher.py - Command routing
- addon/server.py - TCP socket server
- addon/tools/ - 9 tool modules (core, creation, materials, render, etc.)
- scripts/headless_start.py - Blender headless launcher
- mcp.json - Configuration (not yet active)

**Architecture** (Current):
```
Claude Code → [FastMCP Client?] → TCP (port 2800) → Blender Addon → Tool Modules
```

**Migration Path**:
1. Create skills/blender/mcp-server/server.py (FastMCP)
2. Implement TCP client similar to QGIS pattern
3. Keep addon TCP server as-is
4. Map MCP tools to TCP commands
5. Add VisionFlow resources

---

## Migration Priority Matrix

### High Priority (User-Facing, Complex)

| Skill | Current | Target | Complexity | Impact |
|-------|---------|--------|------------|--------|
| **blender** | Unified addon + TCP | FastMCP + TCP | Medium | High (3D workflows) |
| **comfyui** | MCP SDK Node.js | FastMCP Python | Medium | High (image/video gen) |
| **deepseek-reasoning** | MCP SDK Node.js | FastMCP Python | Low | Medium (reasoning) |

### Medium Priority (Feature Complete)

| Skill | Status | Notes |
|-------|--------|-------|
| imagemagick | ✅ Era 3 | Modern FastMCP reference implementation |
| qgis | ✅ Era 3 | FastMCP + TCP pattern complete |
| web-summary | ✅ Era 3 | FastMCP + Z.AI integration complete |
| playwright | ✅ Era 2 | MCP SDK consolidated, works well |

### Low Priority (Script Collections)

| Skill | Action | Priority |
|-------|--------|----------|
| docs-alignment | Wrap as FastMCP | Low |
| wardley-maps | Wrap as FastMCP | Low |
| slack-gif-creator | Wrap as FastMCP | Very Low |

### No Action (Prompt-Only)

28+ skills with SKILL.md only - no migration needed

---

## FastMCP Migration Template

For converting Node.js MCP SDK to FastMCP Python:

```python
#!/usr/bin/env python3
"""
{Skill Name} MCP Server - FastMCP Implementation
"""

import os
import json
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Environment configuration
CONFIG = {
    "host": os.environ.get("SERVICE_HOST", "localhost"),
    "port": int(os.environ.get("SERVICE_PORT", "8188")),
    "timeout": int(os.environ.get("SERVICE_TIMEOUT", "60"))
}

# Initialize FastMCP server
mcp = FastMCP(
    "{skill-name}",
    version="2.0.0",
    description="{description}"
)

# =============================================================================
# Pydantic Models
# =============================================================================

class ToolParams(BaseModel):
    """Parameters for tool."""
    param1: str = Field(..., description="...")
    param2: int = Field(default=10, ge=1, le=100)

# =============================================================================
# Service Client
# =============================================================================

def send_command(command: str, params: Dict[str, Any]) -> Dict:
    """Send command to external service."""
    # Implement HTTP, TCP, or WebSocket client here
    pass

# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
def tool_name(params: ToolParams) -> dict:
    """
    Tool description for Claude Code.

    Use when you need to...
    """
    try:
        result = send_command("command_type", params.dict())
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def health_check() -> dict:
    """Check service connection health."""
    try:
        result = send_command("ping", {})
        return {"success": True, "status": "connected", "result": result}
    except Exception as e:
        return {"success": False, "status": "disconnected", "error": str(e)}

# =============================================================================
# MCP Resources (VisionFlow Integration)
# =============================================================================

@mcp.resource("{skill}://capabilities")
def get_capabilities() -> str:
    """Return capabilities for VisionFlow discovery."""
    capabilities = {
        "name": "{skill-name}",
        "version": "2.0.0",
        "protocol": "fastmcp",
        "tools": ["tool_name", "health_check"],
        "visionflow_compatible": True
    }
    return json.dumps(capabilities, indent=2)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()
```

---

## Recommended Next Steps

1. **Blender**: Complete FastMCP migration (highest priority for VisionFlow)
2. **ComfyUI**: Rewrite as FastMCP Python (consistency + Z.AI integration)
3. **DeepSeek**: Rewrite as FastMCP Python (simplify user switching)
4. **Script Collections**: Low priority wrapping as needed
5. **Documentation**: Update SKILLS.md with FastMCP patterns

---

## Key Findings

### Pattern Distribution
- **3** FastMCP Python servers (modern, reference implementations)
- **4** MCP SDK Node.js servers (functional, some legacy)
- **28+** Prompt-only skills (no migration needed)
- **5+** Script collections (potential FastMCP candidates)
- **1** Unified architecture in progress (Blender)

### TCP Proxy Usage
Only **2 skills** use TCP proxies:
- **qgis** (port 9877) - Necessary (QGIS requires TCP)
- **blender** (port 2800) - Necessary (Blender addon architecture)

All others use stdio transport (modern pattern).

### Z.AI Integration
- **web-summary** - Full integration for summarization
- **comfyui** - Optional for workflow generation
- Opportunity: Extend to other LLM-heavy skills

### VisionFlow Readiness
**Ready** (3):
- imagemagick - Full resources, capabilities
- qgis - Full resources, status endpoint
- web-summary - Capabilities resource

**Partial** (2):
- playwright - Has resources
- comfyui - No resources yet

**Not Ready** (others)

---

## Conclusion

The skill ecosystem shows clear evolution:
- **Era 1** (Legacy): stdin/stdout JSON, Node.js proxies
- **Era 2** (Consolidated): @mcp/sdk, reduced complexity
- **Era 3** (Modern): FastMCP, Pydantic, VisionFlow integration

**Recommendation**: Standardize on FastMCP Python for all new skills and migrate high-priority Node.js implementations.
