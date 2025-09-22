# Multi-Agent Docker Documentation

Welcome to the Multi-Agent Docker environment - a comprehensive AI-assisted development platform with visual tools, browser automation, and advanced agent orchestration.

## 🚀 Quick Start

```bash
# Clone and start the environment
git clone <repository>
cd multi-agent-docker
./multi-agent.sh start

# Run setup inside container
/app/setup-workspace.sh

# Initialize AI agents
claude-flow-init-agents
```

## 📚 Documentation Structure

### Getting Started
- **[Quick Start Guide](getting-started/01-quick-start.md)** - Get up and running in 5 minutes
- **[Configuration](getting-started/02-configuration.md)** - Environment variables and settings

### Architecture & Design
- **[System Overview](architecture/01-overview.md)** - Components and architecture
- **[Networking](architecture/02-networking.md)** - Ports, services, and connectivity
- **[Security](architecture/03-security.md)** - Authentication and security features

### Core Features
- **[Claude Authentication](features/01-claude-authentication.md)** - Setting up Claude Code access
- **[MCP Tools](features/02-mcp-tools.md)** - Using Model Context Protocol tools
- **[Visual Browser Automation](features/03-playwright-visual.md)** - Playwright with VNC access
- **[AI Agents](features/04-ai-agents.md)** - Claude-Flow Goal Planner and Neural agents
- **[GUI Tools](features/05-gui-tools.md)** - Blender, QGIS, and PBR Generator

### Reference
- **[Environment Variables](reference/01-environment-variables.md)** - Complete configuration reference
- **[API Reference](reference/02-api-reference.md)** - MCP protocols and endpoints
- **[CLI Commands](reference/03-cli-commands.md)** - Available commands and aliases
- **[Troubleshooting](reference/04-troubleshooting.md)** - Common issues and solutions

## 🔧 Key Features

### Multi-Container Architecture
- **Main Container**: Development environment with Claude Code and MCP tools
- **GUI Container**: Visual tools with VNC access (Blender, QGIS, Playwright)
- **Shared Network**: Seamless communication between containers

### AI Capabilities
- **Claude-Flow v110**: Advanced agent orchestration
- **Goal Planner**: GOAP-based task planning
- **Neural Agent**: Self-aware learning system
- **MCP Integration**: 87+ tools for development

### Visual Tools
- **Browser Automation**: Playwright with real-time visual debugging
- **3D Modeling**: Blender with MCP integration
- **GIS Tools**: QGIS for geographic data
- **VNC Access**: Port 5901 for visual interaction

## 📡 Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Claude Flow UI | 3000 | Web interface |
| WebSocket Bridge | 3002 | Real-time communication |
| VNC | 5901 | Visual access to GUI tools |
| MCP TCP (Shared) | 9500 | Shared Claude-Flow instance |
| Claude-Flow TCP | 9502 | Isolated sessions |
| Blender MCP | 9876 | 3D modeling tools |
| QGIS MCP | 9877 | GIS tools |
| PBR MCP | 9878 | Material generation |
| Playwright MCP | 9879 | Browser automation |

## 🛠️ Common Tasks

### Check Service Status
```bash
mcp-status          # All MCP services
cf-tcp-status       # Claude-Flow TCP
playwright-proxy-status  # Playwright proxy
```

### View Logs
```bash
mcp-tcp-logs        # MCP TCP server
cf-logs             # Claude-Flow logs
playwright-proxy-logs  # Playwright proxy
```

### Test Connections
```bash
playwright-stack-test  # Test Playwright stack
cf-test-tcp         # Test Claude-Flow TCP
mcp-test-health     # Test MCP health
```

## 🐛 Quick Troubleshooting

- **Services not starting?** Run `mcp-restart`
- **Can't connect to GUI tools?** Check with `docker-compose ps gui-tools-service`
- **Authentication issues?** See [Claude Authentication](features/01-claude-authentication.md)
- **Browser automation failing?** Run `playwright-stack-test`

## 📞 Support

- Check [Troubleshooting Guide](reference/04-troubleshooting.md)
- Review logs with the commands above
- Ensure both containers are running: `docker-compose ps`