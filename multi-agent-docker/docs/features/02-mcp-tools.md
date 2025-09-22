# MCP Tools Guide

The Multi-Agent Docker environment provides a comprehensive set of Model Context Protocol (MCP) tools for AI-assisted development.

## Available MCP Tools

### Core Development Tools
- **claude-flow**: Advanced agent orchestration (Goal Planner, Neural agents)
- **ruv-swarm**: Swarm intelligence coordination
- **imagemagick**: Image manipulation and processing
- **kicad**: Electronic design automation
- **ngspice**: Circuit simulation

### Visual GUI Tools (via GUI Container)
- **blender**: 3D modeling and rendering (port 9876)
- **qgis**: Geographic information system (port 9877)
- **pbr-generator**: Physically based rendering materials (port 9878)
- **playwright**: Browser automation with visual debugging (port 9879)

## Using MCP Tools

### Through Claude Code

All MCP tools are accessible via Claude Code using the `mcp__` prefix:

```bash
# In Claude Code
claude

# Use a tool
> Use mcp__claude-flow__goal_create to create a new goal
> Use mcp__blender__create_cube to create a 3D cube
> Use mcp__playwright__navigate to open a webpage
```

### Direct CLI Access

Some tools can be accessed directly:

```bash
# Claude-Flow agents
npx claude-flow@alpha goal create --name "Build feature"
npx claude-flow@alpha neural train

# Test MCP connections
echo '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}' | nc localhost 9500
```

## MCP Architecture

### Connection Methods

1. **Local Stdio**: Direct process communication for tools in main container
2. **TCP Proxy**: Network communication for GUI container tools
3. **WebSocket Bridge**: Browser-based tool access on port 3002

### Tool Discovery

List all available tools:

```bash
# Through TCP
echo '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}' | nc localhost 9500

# Check specific tool
mcp-test-tcp 9876  # Test Blender
mcp-test-tcp 9877  # Test QGIS
```

## GUI Tools Integration

### Blender MCP (Port 9876)
- Create and manipulate 3D objects
- Render scenes
- Export models
- Access via: `mcp__blender__`

### QGIS MCP (Port 9877)
- Load geographic data
- Perform spatial analysis
- Create maps
- Access via: `mcp__qgis__`

### Playwright MCP (Port 9879)
- Browser automation with visual feedback
- Screenshot capture
- Form interaction
- Access via: `mcp__playwright__`

### Visual Access
All GUI tools can be viewed through VNC:
```bash
# Connect to VNC
vncviewer localhost:5901
# Or use any VNC client
```

## Advanced Usage

### Creating Custom MCP Tools

1. Create tool script in `/workspace/mcp-tools/`
2. Add to `.mcp.json` configuration
3. Implement MCP protocol handlers

Example Python MCP tool:
```python
#!/usr/bin/env python3
import json
import sys

def handle_request(request):
    method = request.get('method')
    if method == 'tools/list':
        return {
            'tools': [{
                'name': 'my_tool',
                'description': 'Custom tool',
                'inputSchema': {}
            }]
        }
    # Handle other methods...

# Main loop
for line in sys.stdin:
    request = json.loads(line)
    response = handle_request(request)
    print(json.dumps(response))
    sys.stdout.flush()
```

### Monitoring and Debugging

```bash
# View all MCP logs
mcp-logs

# Monitor specific tool
tail -f /app/mcp-logs/blender-mcp-server.log

# Test tool connectivity
playwright-stack-test  # Test Playwright stack
mcp-test-health       # Test health endpoints
```

## Best Practices

1. **Use appropriate tools**: GUI tools for visual tasks, CLI tools for automation
2. **Monitor resources**: Check logs if tools become unresponsive
3. **Handle errors**: Tools may timeout or fail - check logs
4. **Visual debugging**: Use VNC for GUI tools when debugging

## Troubleshooting

### Tool Not Responding
```bash
# Restart all MCP services
mcp-restart

# Check specific service
supervisorctl status
```

### Connection Refused
```bash
# Ensure GUI container is running
docker-compose ps gui-tools-service

# Test network connectivity
ping gui-tools-service
```

### Tool Not Found
```bash
# Verify tool in configuration
cat /workspace/.mcp.json | jq

# Check if service is running
mcp-status
```