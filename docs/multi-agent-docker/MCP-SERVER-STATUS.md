# MCP Server Status

## Current Working Configuration

Last Updated: 2025-09-30

### ✅ All Systems Operational

| Server | Status | Type | Purpose |
|--------|--------|------|---------|
| **claude-flow** | ✅ Connected | Stdio | AI workflow orchestration and agent coordination |
| **flow-nexus** | ✅ Connected | Stdio | Flow-based programming and task automation (94 tools) |
| **playwright** | ✅ Connected | Stdio | Browser automation and web testing |
| **ruv-swarm** | ✅ Connected | Stdio | Multi-agent swarm coordination (25 tools) |

### 🔧 Core Services (Supervisord)

| Service | Port | Status | Purpose |
|---------|------|--------|---------|
| **mcp-tcp-server** | 9500 | ✅ Running | Primary MCP TCP server (shared instance) |
| **mcp-ws-bridge** | 3002 | ✅ Running | WebSocket-to-stdio MCP bridge |
| **claude-flow-tcp** | 9502 | ✅ Running | Isolated claude-flow sessions per client |
| **playwright-mcp-proxy** | - | ✅ Running | Proxy to GUI container playwright |
| **blender-mcp-proxy** | - | ✅ Running | Proxy to GUI container Blender (9876) |
| **qgis-mcp-proxy** | - | ✅ Running | Proxy to GUI container QGIS (9877) |
| **pbr-mcp-proxy** | - | ✅ Running | Proxy to GUI container PBR Generator (9878) |
| **claude-md-watcher** | - | ✅ Running | Auto-repairs CLAUDE.md tool manifest |

### 📊 Context Usage

**Current Load**: ~75k tokens (within limits)
- flow-nexus: 94 tools (~59,052 tokens)
- ruv-swarm: 25 tools (~15,935 tokens)

### 🎯 Quick Commands

```bash
# View MCP server status
/mcp

# List all available tools
# (Inside Claude CLI)

# Check service status
docker exec multi-agent-container supervisorctl status

# View logs
docker logs -f multi-agent-container

# Test MCP TCP server
echo '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}' | nc localhost 9500
```

## Configuration Files

### Local Config (per-project)
**Path**: `/home/dev/.claude/.claude.json` [project: /workspace]

Contains project-specific MCP server configurations:
- claude-flow
- flow-nexus
- playwright
- ruv-swarm

### Project Config (shared)
**Path**: `/workspace/.mcp.json`

Contains base MCP server definitions for the project.

### User Config (global)
**Path**: `/home/dev/.claude/.claude.json`

User-level settings available across all projects.

## Recent Fixes

### 2025-09-30: MCP Configuration Cleanup

**Issues Resolved**:
1. ✅ Fixed claude-flow version (alpha.120 → @alpha)
2. ✅ Removed duplicate playwright-mcp entry (kept `playwright`)
3. ✅ Removed broken blender-mcp, kicad-mcp, imagemagick-mcp from local config
4. ✅ Documented expected soft-fail behavior for GUI-dependent tools

**Configuration Changes**:
- claude-flow: Fixed command format (`npx claude-flow@alpha mcp start`)
- playwright: Kept correct format (removed duplicate)
- GUI tools: Documented as proxy-based (will soft-fail until GUI container ready)

## GUI-Dependent Tools (Optional)

These tools connect to services in the `gui-tools-container` and will timeout during initial startup:

| Tool | Port | Expected Behavior |
|------|------|-------------------|
| Blender MCP | 9876 | 30-60s timeout during GUI container startup |
| QGIS MCP | 9877 | 30-60s timeout during GUI container startup |
| PBR Generator | 9878 | 30-60s timeout during GUI container startup |
| Playwright (GUI) | 9879 | Separate from local playwright MCP |

**Note**: These timeouts are **expected** and will resolve once the GUI container finishes initialization. The local MCP servers (claude-flow, flow-nexus, etc.) work independently.

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Claude CLI (Inside Container)           │
│                                                 │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ claude-flow │  │ flow-nexus  │             │
│  │   (stdio)   │  │   (stdio)   │             │
│  └─────────────┘  └─────────────┘             │
│                                                 │
│  ┌─────────────┐  ┌─────────────┐             │
│  │ playwright  │  │  ruv-swarm  │             │
│  │   (stdio)   │  │   (stdio)   │             │
│  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────┘
                      │
                      │ Docker Network
                      │
┌─────────────────────────────────────────────────┐
│         External Access (Host Machine)          │
│                                                 │
│  TCP 9500 ──► mcp-tcp-server (shared instance) │
│  TCP 9502 ──► claude-flow-tcp (isolated)       │
│  WS  3002 ──► mcp-ws-bridge                    │
└─────────────────────────────────────────────────┘
```

## Troubleshooting

### MCP Server Won't Connect

1. Check Claude CLI version:
   ```bash
   claude --version
   ```

2. Check MCP logs:
   ```bash
   ls -la /home/dev/.cache/claude-cli-nodejs/-workspace/mcp-logs-*/
   ```

3. Restart Claude CLI:
   ```bash
   exit
   docker exec -it multi-agent-container bash
   ```

### Tools Not Showing Up

1. Verify MCP config:
   ```bash
   cat /home/dev/.claude/.claude.json | jq '.projects["/workspace"].mcpServers | keys'
   ```

2. Check for errors:
   ```bash
   claude --debug
   ```

### High Context Usage Warning

If you see warnings about large MCP tools context (>25k tokens):

**Current**: ~75k tokens (flow-nexus: 94 tools, ruv-swarm: 25 tools)

**Options**:
1. Disable unused servers temporarily
2. Use specific tools instead of loading all
3. This is informational - functionality not impacted

## See Also

- [PORT-CONFIGURATION.md](./PORT-CONFIGURATION.md) - Complete port documentation
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
- [DOCUMENTATION-AUDIT-REPORT.md](./DOCUMENTATION-AUDIT-REPORT.md) - Recent doc updates