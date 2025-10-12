# CachyOS Docker Architecture

## Overview

The CachyOS Docker environment implements a **simplified stdio-based MCP architecture** where a single Management API (port 9090) spawns isolated worker sessions that invoke MCP tools on-demand via stdio.

## Design Principles

1. **Single Entry Point**: Management API on port 9090 is the only required external interface
2. **On-Demand Tool Execution**: MCP tools spawn only when needed via stdio
3. **Session Isolation**: Each worker gets its own isolated tool instances
4. **No HTTP Overhead**: Direct stdio communication eliminates HTTP server overhead
5. **Minimal Resource Footprint**: Tools don't consume resources when idle

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  Docker Container                    │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │         Management API (Port 9090)          │   │
│  │  - Session creation/management               │   │
│  │  - Worker lifecycle control                  │   │
│  │  - Resource allocation                       │   │
│  └────────┬────────────────────────────────────┘   │
│           │                                          │
│           │ Spawns workers as needed                │
│           ▼                                          │
│  ┌─────────────────────────────────────────────┐   │
│  │           Worker Sessions                    │   │
│  │  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │  Worker 1    │  │  Worker 2    │  ...    │   │
│  │  └──────┬───────┘  └──────┬───────┘         │   │
│  │         │ stdio            │ stdio           │   │
│  │         ▼                  ▼                 │   │
│  │  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │ MCP Tools    │  │ MCP Tools    │  ...    │   │
│  │  │ - Playwright │  │ - Context7   │         │   │
│  │  │ - Filesystem │  │ - GitHub     │         │   │
│  │  │ - Git        │  │ - Fetch      │         │   │
│  │  └──────────────┘  └──────────────┘         │   │
│  └─────────────────────────────────────────────┘   │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │      Optional Desktop Environment            │   │
│  │  - VNC Server (port 5901)                   │   │
│  │  - noVNC (port 6901)                        │   │
│  │  - XFCE4                                     │   │
│  └─────────────────────────────────────────────┘   │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │      Optional Development Tools              │   │
│  │  - code-server (port 8080)                  │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Key Components

### Management API (Required)
- **Port**: 9090
- **Purpose**: Central orchestration point for worker sessions
- **Responsibilities**:
  - Create and destroy worker sessions
  - Allocate resources to workers
  - Monitor worker health
  - Clean up stale sessions

### Worker Sessions
- **Lifecycle**: Created on-demand, destroyed when idle
- **Isolation**: Each worker has its own process space
- **Tool Access**: Spawns MCP tools via stdio as configured in `mcp.json`
- **State**: Ephemeral - no persistent state between sessions

### MCP Tools (stdio-based)
- **Invocation**: Spawned via `npx` or direct command execution
- **Communication**: Standard input/output (stdio)
- **Configuration**: Defined in `/home/devuser/.config/claude/mcp.json`
- **Examples**:
  - `@modelcontextprotocol/server-playwright`: Browser automation
  - `@upstash/context7-mcp`: Documentation lookup
  - `@modelcontextprotocol/server-filesystem`: File operations
  - `@modelcontextprotocol/server-git`: Git operations
  - `@modelcontextprotocol/server-github`: GitHub API
  - `@modelcontextprotocol/server-fetch`: Web content fetching

### Desktop Environment (Optional)
- **VNC Port**: 5901
- **noVNC Port**: 6901 (web-based VNC client)
- **Display**: :1
- **Window Manager**: XFCE4
- **Control**: Enabled via `ENABLE_DESKTOP=true` environment variable

### Code Server (Optional)
- **Port**: 8080
- **Purpose**: Web-based VS Code IDE
- **Control**: Enabled via `ENABLE_CODE_SERVER=true` environment variable

## Configuration Files

### Primary Configuration Files
- `docker-compose.yml`: Container orchestration
- `Dockerfile.workstation`: Container image definition
- `config/supervisord.conf`: Process management
- `config/mcp.json`: MCP tool definitions (stdio-based)
- `config/router.config.json`: Agentic flow routing configuration

### Deprecated (Removed)
- ~~`supervisord-unified.conf`~~ - HTTP-based architecture
- ~~`mcp-unified.json`~~ - HTTP bridge pattern
- ~~`docker-compose.workstation.yml`~~ - Complex architecture

## Workflow

1. **Container Start**: supervisord launches Management API and optional services
2. **Client Request**: External client connects to Management API (port 9090)
3. **Worker Creation**: API spawns isolated worker session
4. **Tool Invocation**: Worker spawns MCP tools via stdio as needed
5. **Task Execution**: Tools process requests and return results via stdio
6. **Session Cleanup**: Worker terminated after idle timeout or completion
7. **Tool Termination**: All child tool processes automatically cleaned up

## Advantages of Stdio Architecture

### Resource Efficiency
- **No Idle Overhead**: Tools only run when actively processing requests
- **Memory Savings**: No persistent HTTP servers consuming RAM
- **Fast Startup**: Direct process spawning without HTTP server initialization
- **Automatic Cleanup**: Process tree termination ensures no orphaned processes

### Simplicity
- **Fewer Moving Parts**: No HTTP servers, no port management, no network stack
- **Direct Communication**: Standard Unix pipes for IPC
- **Clear Lifecycle**: Process spawn = start, process exit = cleanup
- **Easier Debugging**: Standard streams for input/output/errors

### Security
- **Reduced Attack Surface**: No exposed ports for MCP tools
- **Process Isolation**: Each worker's tools isolated from others
- **No Network Layer**: Communication doesn't traverse network stack
- **Resource Limits**: Standard Unix resource limits apply directly

### Scalability
- **Session Isolation**: Multiple workers don't interfere
- **Resource Pooling**: OS handles process scheduling efficiently
- **Elastic Scaling**: Workers created/destroyed based on demand
- **No Port Conflicts**: Stdio eliminates port assignment issues

## Dependency Management

### Local Dependencies Pattern
All auxiliary tools (pm2, gemini-flow, claude-code) are installed as `devDependencies` in the project's `package.json` rather than as global npm packages. This provides:

#### Benefits
- **Version Locking**: All dependencies pinned to specific versions in package-lock.json
- **Reproducibility**: Identical environment across builds and deployments
- **No Conflicts**: Different projects can use different versions of tools
- **Hermetic Builds**: All dependencies self-contained within project
- **Easier Updates**: Single `npm update` updates all tools consistently

#### Implementation
Tools are available via:
1. **npm scripts**: `npm run pm2`, `npm run gemini-flow`, `npm run claude-code`
2. **PATH**: `/tmp/agentic-flow/node_modules/.bin` added to PATH
3. **Symlinks**: Binaries symlinked to `/usr/local/bin` for convenience

Only the main `agentic-flow` package itself is installed globally to provide the `agentic-flow` command system-wide.

## Environment Variables

### Core Settings
- `ENABLE_DESKTOP`: Enable VNC/noVNC/XFCE4 (default: true)
- `ENABLE_CODE_SERVER`: Enable web-based VS Code (default: true)
- `WORKSPACE`: Workspace directory path (default: /home/devuser/workspace)
- `DISPLAY`: X11 display for GUI applications (default: :1)

### AI Provider Configuration
- `ENABLE_GEMINI`: Enable Gemini AI provider
- `ENABLE_OPENAI`: Enable OpenAI provider
- `ENABLE_CLAUDE`: Enable Anthropic Claude provider
- `ENABLE_OPENROUTER`: Enable OpenRouter provider
- `ROUTER_MODE`: Routing strategy (performance/cost/quality)

### Gemini Flow Settings
- `GEMINI_FLOW_ENABLED`: Enable Gemini Flow orchestration
- `GEMINI_FLOW_PROTOCOLS`: Protocols to enable (a2a,mcp)
- `GEMINI_FLOW_TOPOLOGY`: Swarm topology (hierarchical/mesh/adaptive)
- `GEMINI_FLOW_MAX_AGENTS`: Maximum concurrent agents

## Port Mappings

| Port | Service | Required | Purpose |
|------|---------|----------|---------|
| 9090 | Management API | Yes | Worker orchestration |
| 8080 | code-server | Optional | Web IDE |
| 6901 | noVNC | Optional | Web-based VNC |
| 5901 | VNC | Optional | Direct VNC access |
| 3000 | Dev Server | Optional | Development preview |

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| ./workspace | /home/devuser/workspace | Project files |
| ./models | /home/devuser/models | AI model cache |
| ./logs | /home/devuser/logs | Application logs |
| ./.config | /home/devuser/.config | Configuration |
| ./.claude-flow | /home/devuser/.claude-flow | Claude Flow state |

## Migration from HTTP Architecture

### What Was Removed
1. **HTTP MCP Servers**: All `*-mcp-server.js` and `*-mcp-server.py` files
2. **Bridge Clients**: All `*-mcp-client.js` and `*-mcp-client.py` files
3. **Port Configurations**: Environment variables like `BLENDER_MCP_PORT`, `QGIS_MCP_PORT`
4. **supervisord Programs**: Removed MCP tool programs from supervisord
5. **Redundant Directories**: Removed `gui-tools-assets` and duplicated asset directories

### What Stayed
1. **Management API**: Core orchestration remains
2. **Desktop Environment**: VNC/noVNC optional services
3. **Core Assets**: Configuration, scripts, and documentation
4. **Docker Compose**: Simplified to single compose file

### Migration Path
If you have existing deployments:
1. Stop existing containers
2. Remove old images: `docker rmi <old-image>`
3. Update `mcp.json` to use stdio commands
4. Rebuild with new Dockerfile: `docker-compose build`
5. Start with new configuration: `docker-compose up`

## Best Practices

### Development
- Use `docker-compose.yml` for standard deployments
- Mount workspace directory for persistent project files
- Enable desktop environment only when GUI tools needed
- Use code-server for remote development access

### Production
- Disable optional services (desktop, code-server)
- Set appropriate resource limits in docker-compose
- Monitor Management API health endpoint
- Configure session cleanup intervals
- Enable logging with appropriate rotation

### Debugging
- Check Management API logs: `docker logs agentic-flow-cachyos`
- Inspect supervisord status: `docker exec agentic-flow-cachyos supervisorctl status`
- Monitor worker sessions: Check Management API `/sessions` endpoint
- Verify MCP config: Review `/home/devuser/.config/claude/mcp.json`

## Troubleshooting

### Worker Sessions Not Starting
- Check Management API logs for errors
- Verify `mcp.json` syntax and tool availability
- Ensure required npm packages installed globally
- Check system resources (memory, CPU)

### MCP Tools Failing
- Verify tool command is correct in `mcp.json`
- Check if npm package is installed: `docker exec <container> npm list -g`
- Review worker logs for stdio errors
- Test tool manually: `docker exec <container> npx <tool-name>`

### Desktop Environment Issues
- Verify `ENABLE_DESKTOP=true` is set
- Check VNC port not already in use
- Ensure X11 display :1 is available
- Review xvnc logs: `docker exec <container> cat /home/devuser/logs/xvnc.log`

### Performance Problems
- Monitor container resource usage: `docker stats`
- Check worker session cleanup is running
- Verify no stale processes: `docker exec <container> ps aux`
- Review session idle timeout settings

## Future Enhancements

### Planned Features
- Dynamic tool discovery and registration
- Worker pool management with warm standby
- Tool usage metrics and analytics
- Automatic scaling based on load
- Multi-tenant session isolation

### Potential Optimizations
- Tool result caching
- Worker process reuse
- Lazy tool initialization
- Predictive tool preloading
- Resource usage profiling
