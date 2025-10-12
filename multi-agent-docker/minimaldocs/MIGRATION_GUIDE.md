# Migration Guide: HTTP to Stdio Architecture

## Overview

This guide helps you migrate from the deprecated HTTP-based MCP architecture to the simplified stdio-based architecture.

## What Changed

### Architecture Shift

**Old (HTTP-based)**:
```
Client → Management API → HTTP Request → MCP Server (persistent) → Tool
```

**New (stdio-based)**:
```
Client → Management API → Worker Session → MCP Tool (on-demand via stdio)
```

### Key Differences

| Aspect | HTTP Architecture | Stdio Architecture |
|--------|------------------|-------------------|
| Tool Execution | Persistent HTTP servers | On-demand stdio processes |
| Communication | HTTP/JSON over TCP | stdio/JSON over pipes |
| Resource Usage | Always running | Only when needed |
| Port Management | Multiple ports (9876-9882) | No ports needed |
| Configuration | Complex bridge pattern | Direct stdio invocation |
| Scalability | Limited by port availability | Limited by process count |
| Startup Time | Slower (server init) | Faster (direct spawn) |
| Debugging | Network traces | Process logs |

## Files Removed

### Configuration Files
- ❌ `config/supervisord-unified.conf`
- ❌ `config/mcp-unified.json`
- ❌ `docker-compose.workstation.yml`
- ✅ `config/supervisord.conf` (formerly supervisord-simple.conf)
- ✅ `config/mcp.json` (formerly mcp-stdio.json)
- ✅ `docker-compose.yml` (formerly docker-compose-simple.yml)

### HTTP Server Files
All `*-mcp-server.js` and `*-mcp-server.py` files removed:
- ❌ `blender-mcp-server.js`
- ❌ `qgis-mcp-server.js`
- ❌ `playwright-mcp-server.js`
- ❌ `web-summary-mcp-server.py`
- ❌ `kicad-mcp-server.py`
- ❌ `imagemagick-mcp-server.py`

### Bridge Client Files
All `*-mcp-client.js` and `*-mcp-client.py` files removed:
- ❌ `mcp-client-blender.js`
- ❌ `mcp-client-qgis.js`
- ❌ `mcp-client-playwright.js`
- ❌ `mcp-client-web-summary.py`
- ❌ `mcp-client-kicad.py`
- ❌ `mcp-client-imagemagick.py`

### Redundant Directories
- ❌ `gui-tools-assets/`
- ❌ `assets/gui-tools-assets/`
- ❌ `assets/core-assets/`
- ❌ `assets/claude-zai/`
- ✅ `core-assets/` (consolidated location)

## Migration Steps

### 1. Stop Existing Container

```bash
# Stop and remove old container
docker-compose down

# Remove old image
docker rmi agentic-flow-cachyos:old
```

### 2. Update Configuration

#### docker-compose.yml

**Old**:
```yaml
services:
  cachyos:
    ports:
      - "9876:9876"  # Blender MCP
      - "9877:9877"  # QGIS MCP
      - "9878:9878"  # Playwright MCP
      - "9879:9879"  # Web Summary MCP
      - "9880:9880"  # KiCAD MCP
      - "9881:9881"  # ImageMagick MCP
```

**New**:
```yaml
services:
  cachyos:
    ports:
      - "9090:9090"  # Management API only
      - "8080:8080"  # code-server (optional)
      - "6901:6901"  # noVNC (optional)
```

#### MCP Configuration

**Old (`mcp-unified.json`)**:
```json
{
  "mcpServers": {
    "blender": {
      "command": "node",
      "args": ["/app/scripts/mcp-client-blender.js"],
      "type": "stdio",
      "serverUrl": "http://localhost:9876"
    }
  }
}
```

**New (`mcp.json`)**:
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-playwright"],
      "type": "stdio",
      "env": {
        "DISPLAY": ":1"
      }
    }
  }
}
```

#### Environment Variables

Remove these variables:
```bash
# No longer needed
BLENDER_MCP_PORT=9876
QGIS_MCP_PORT=9877
PLAYWRIGHT_MCP_PORT=9878
WEB_SUMMARY_MCP_PORT=9879
KICAD_MCP_PORT=9880
IMAGEMAGICK_MCP_PORT=9881
```

### 3. Update Custom Tools

If you have custom tools using the HTTP pattern, convert them to stdio.

**Old (HTTP Server)**:
```javascript
// my-tool-server.js
const express = require('express');
const app = express();

app.post('/execute', (req, res) => {
  const { action, params } = req.body;
  // Process request
  res.json({ result: ... });
});

app.listen(process.env.MY_TOOL_PORT);
```

**New (Stdio)**:
```javascript
// my-tool-mcp.js
#!/usr/bin/env node
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  const request = JSON.parse(line);

  // Process request
  const response = {
    jsonrpc: "2.0",
    id: request.id,
    result: { /* your result */ }
  };

  console.log(JSON.stringify(response));
});
```

### 4. Update Client Code

**Old (HTTP Client)**:
```javascript
const response = await fetch('http://localhost:9876/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    action: 'test',
    params: {}
  })
});
const result = await response.json();
```

**New (Session API)**:
```javascript
// Create session with tools
const session = await fetch('http://localhost:9090/api/v1/sessions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    tools: ['playwright', 'filesystem']
  })
});
const { sessionId } = await session.json();

// Execute via session
const result = await fetch(
  `http://localhost:9090/api/v1/sessions/${sessionId}/execute`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tool: 'playwright',
      action: 'navigate',
      params: { url: 'https://example.com' }
    })
  }
);
```

### 5. Rebuild and Start

```bash
# Build new image
docker-compose build

# Start with new architecture
docker-compose up -d

# Verify services
docker-compose ps
docker exec agentic-flow-cachyos supervisorctl status
```

### 6. Verify Migration

```bash
# Check Management API
curl http://localhost:9090/health

# Verify MCP configuration
docker exec agentic-flow-cachyos \
  jq '.' /home/devuser/.config/claude/mcp.json

# Test tool availability
docker exec agentic-flow-cachyos \
  npx -y @modelcontextprotocol/server-playwright --help

# Create test session
curl -X POST http://localhost:9090/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"tools": ["playwright"]}'
```

## Tool Equivalents

### Blender
**Old**: Custom HTTP server
**New**: No official MCP server yet

If you need Blender:
1. Keep custom implementation
2. Convert to stdio pattern
3. Add to `mcp.json`

### QGIS
**Old**: Custom HTTP server
**New**: No official MCP server yet

Similar to Blender - convert custom implementation.

### Playwright
**Old**: Custom HTTP bridge
**New**: `@modelcontextprotocol/server-playwright`

```json
{
  "playwright": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-playwright"],
    "type": "stdio"
  }
}
```

### Web Scraping / YouTube
**Old**: Custom HTTP server
**New**: Use Playwright + custom logic or Fetch

```json
{
  "fetch": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-fetch"],
    "type": "stdio"
  }
}
```

### KiCAD
**Old**: Custom HTTP server
**New**: No official MCP server

Convert custom implementation to stdio.

### ImageMagick
**Old**: Custom HTTP server
**New**: No official MCP server

Convert custom implementation or use system ImageMagick via shell.

## Common Issues

### Port Conflicts

**Old Error**:
```
Error: Port 9876 already in use
```

**Solution**:
Ports no longer needed - remove from docker-compose.yml

### Tool Not Found

**Old**:
```
Connection refused to http://localhost:9876
```

**New Error**:
```
Tool 'blender' not found in mcp.json
```

**Solution**:
Add tool to `mcp.json` with stdio configuration

### Bridge Client Errors

**Old Error**:
```
Error in mcp-client-playwright.js: Cannot connect to server
```

**Solution**:
Remove bridge clients, use direct stdio tools

### Environment Variables

**Old**:
```yaml
environment:
  - PLAYWRIGHT_MCP_PORT=9878
```

**New**:
```yaml
environment:
  - DISPLAY=:1  # For GUI tools only
```

## Rollback Plan

If you need to rollback:

```bash
# Stop new container
docker-compose down

# Checkout old version
git checkout <old-commit>

# Rebuild old image
docker-compose build

# Start old version
docker-compose up -d
```

## Performance Comparison

### Resource Usage

**HTTP Architecture**:
- Base memory: ~2GB
- Per tool overhead: ~100-200MB (idle)
- Total memory (8 tools): ~3.5GB

**Stdio Architecture**:
- Base memory: ~1GB
- Per tool overhead: 0MB (idle), ~50-100MB (active)
- Total memory (0 tools): ~1GB
- Total memory (2 active tools): ~1.2GB

### Startup Time

**HTTP Architecture**:
- Container start: 30-45 seconds
- Tool availability: 5-10 seconds per tool
- Total: 40-95 seconds

**Stdio Architecture**:
- Container start: 15-20 seconds
- Tool availability: 0 seconds (on-demand)
- Total: 15-20 seconds

### Request Latency

**HTTP Architecture**:
- HTTP overhead: 5-10ms
- Request processing: varies by tool
- Total: 5-10ms + processing time

**Stdio Architecture**:
- Spawn overhead: 100-500ms (first request only)
- Request processing: varies by tool
- Total: 0-500ms + processing time

## Benefits of New Architecture

### Simplified
- Fewer configuration files
- No port management
- Direct tool invocation
- Clear communication pattern

### Efficient
- Lower idle resource usage
- Faster container startup
- On-demand tool spawning
- Automatic cleanup

### Secure
- No exposed ports for tools
- Process isolation per session
- Reduced attack surface
- Standard Unix permissions

### Maintainable
- Fewer moving parts
- Standard MCP packages
- Clear upgrade path
- Better debugging

## Next Steps

After successful migration:

1. Remove old configuration backups
2. Update documentation references
3. Train team on new architecture
4. Monitor performance metrics
5. Report any issues on GitHub

## Support

If you encounter migration issues:

- Check troubleshooting section in [ARCHITECTURE.md](./ARCHITECTURE.md)
- Review examples in [MCP_TOOLS.md](./MCP_TOOLS.md)
- Open GitHub issue with migration logs
- Join Discord for real-time help
