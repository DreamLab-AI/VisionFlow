# CachyOS Docker Documentation

## Overview

Complete documentation for the CachyOS Docker environment with simplified stdio-based MCP architecture.

## Documentation Structure

### Getting Started
- **[QUICKSTART.md](./QUICKSTART.md)** - Quick setup and basic usage (5 minutes)
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System design and architecture overview
- **[CONFIGURATION.md](./CONFIGURATION.md)** - Detailed configuration guide

### Reference Documentation
- **[MCP_TOOLS.md](./MCP_TOOLS.md)** - Available MCP tools and usage examples
- **[DEPENDENCIES.md](./DEPENDENCIES.md)** - Dependency management and best practices
- **[MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)** - Migrating from HTTP to stdio architecture

### Additional Resources
- **[ARCHITECTURE-SIMPLIFIED.md](../ARCHITECTURE-SIMPLIFIED.md)** - Original architecture proposal
- **[FINAL-ARCHITECTURE.md](../FINAL-ARCHITECTURE.md)** - Complete architecture specification

## Quick Links

### For New Users
1. Read [QUICKSTART.md](./QUICKSTART.md)
2. Follow setup instructions
3. Explore [MCP_TOOLS.md](./MCP_TOOLS.md) for available tools

### For Developers
1. Review [ARCHITECTURE.md](./ARCHITECTURE.md)
2. Study [CONFIGURATION.md](./CONFIGURATION.md)
3. Reference [MCP_TOOLS.md](./MCP_TOOLS.md) for tool integration

### For Existing Users
1. Check [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)
2. Follow migration steps
3. Verify new configuration

## Key Concepts

### Stdio Architecture
All MCP tools communicate via standard input/output (stdio) instead of HTTP. This provides:
- Lower resource usage
- Faster startup
- Better security
- Simpler configuration

### Management API
Single entry point (port 9090) for:
- Worker session creation
- Tool orchestration
- Resource management
- Session lifecycle control

### Worker Sessions
Isolated execution environments that:
- Spawn MCP tools on-demand
- Process tasks independently
- Clean up automatically
- Scale elastically

### MCP Tools
Standard MCP-compliant packages:
- Installed via npm
- Invoked via stdio
- Process JSON-RPC requests
- No persistent HTTP servers

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Docker Container                 │
│                                          │
│  ┌────────────────────────────────┐    │
│  │   Management API (Port 9090)   │    │
│  └────────┬───────────────────────┘    │
│           │                              │
│           │ spawns                       │
│           ▼                              │
│  ┌────────────────────────────────┐    │
│  │      Worker Sessions            │    │
│  │   ┌──────────┐  ┌──────────┐  │    │
│  │   │ Worker 1 │  │ Worker 2 │  │    │
│  │   └────┬─────┘  └────┬─────┘  │    │
│  │        │stdio         │stdio    │    │
│  │        ▼              ▼         │    │
│  │   ┌────────┐    ┌────────┐    │    │
│  │   │  MCP   │    │  MCP   │    │    │
│  │   │ Tools  │    │ Tools  │    │    │
│  │   └────────┘    └────────┘    │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Common Tasks

### Start System (Recommended)
```bash
# From repository root
./start-agentic-flow.sh --build  # First time
./start-agentic-flow.sh          # Regular startup
./start-agentic-flow.sh --status # Check health
./start-agentic-flow.sh --logs   # View logs
```

### Manual Start
```bash
cd docker/cachyos
docker-compose up -d
```

### Create Session
```bash
curl -X POST http://localhost:9090/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"tools": ["playwright", "filesystem"]}'
```

### Execute Task
```bash
curl -X POST http://localhost:9090/api/v1/sessions/{id}/execute \
  -H "Content-Type: application/json" \
  -d '{"tool": "playwright", "action": "navigate", "params": {"url": "https://example.com"}}'
```

### Access IDE
Open browser: http://localhost:8080

### Access Desktop
Open browser: http://localhost:6901

### Interactive Shell
```bash
docker exec -it agentic-flow-cachyos zsh
```

## Configuration Files

```
docker/cachyos/
├── config/
│   ├── supervisord.conf      # Service management
│   ├── mcp.json              # MCP tool definitions
│   ├── router.config.json    # AI router config
│   └── .zshrc                # Shell configuration
├── docker-compose.yml         # Container orchestration
├── Dockerfile.workstation     # Image definition
└── docs/                      # This documentation
```

## Environment Variables

### Essential
```bash
ANTHROPIC_API_KEY=sk-ant-xxx  # Required for Claude
ZAI_API_KEY=xxx               # Required for Z.AI semantic features
GOOGLE_API_KEY=AIza-xxx       # Required for web-summary tool
ENABLE_DESKTOP=true           # Optional desktop
ENABLE_CODE_SERVER=true       # Optional web IDE
```

### Optional
```bash
OPENAI_API_KEY=sk-xxx
GOOGLE_GEMINI_API_KEY=AIza-xxx
GITHUB_TOKEN=ghp_xxx
BRAVE_API_KEY=xxx
CONTEXT7_API_KEY=xxx
```

## Port Mappings

| Port | Service | Required |
|------|---------|----------|
| 9090 | Management API | Yes |
| 9600 | Claude-ZAI Service | Yes |
| 8080 | code-server | Optional |
| 6901 | noVNC | Optional |
| 5901 | VNC | Optional |
| 3000 | Dev server | Optional |

## Available MCP Tools

- **claude-flow** - Workflow orchestration
- **context7** - Documentation lookup
- **playwright** - Browser automation
- **filesystem** - File operations
- **git** - Git operations
- **github** - GitHub API
- **fetch** - Web content fetching
- **brave-search** - Web search
- **web-summary** - Web/YouTube summarization with Z.AI semantic topic matching

See [MCP_TOOLS.md](./MCP_TOOLS.md) for complete reference.

## Resource Requirements

### Minimum
- 4GB RAM
- 2 CPU cores
- 10GB disk space

### Recommended
- 8GB RAM
- 4 CPU cores
- 20GB disk space

### Production
- 16GB RAM
- 8 CPU cores
- 50GB disk space

## Support

### Documentation
- Architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- Configuration: [CONFIGURATION.md](./CONFIGURATION.md)
- Tools: [MCP_TOOLS.md](./MCP_TOOLS.md)
- Migration: [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)

### Community
- GitHub Issues: Report bugs and feature requests
- GitHub Discussions: Ask questions and share ideas
- Discord: Real-time community support

### Contributing
We welcome contributions! See CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Changelog

### v2.1.0 - Z.AI Integration
- ✅ Added Z.AI semantic topic matching service
- ✅ Integrated web-summary MCP tool for URL/YouTube summarization
- ✅ RAGFlow network auto-connection support
- ✅ Unified startup script for combined system
- ✅ Enhanced documentation for Z.AI features
- ✅ Python dependencies for web summarization

### v2.0.0 - Stdio Architecture
- ✅ Migrated from HTTP to stdio-based MCP tools
- ✅ Removed persistent HTTP servers
- ✅ Consolidated configuration files
- ✅ Optimized Dockerfile for layer caching
- ✅ Reduced idle resource usage by 60%
- ✅ Improved container startup time by 50%
- ✅ Added comprehensive documentation

### v1.0.0 - HTTP Architecture (Deprecated)
- ❌ HTTP-based MCP servers
- ❌ Bridge client pattern
- ❌ Multiple configuration files
- ❌ Complex supervisord setup

## Related Projects

- [Agentic Flow](https://github.com/yourusername/agentic-flow) - Main project
- [Claude Flow](https://github.com/claude-flow) - Workflow orchestration
- [Gemini Flow](https://github.com/gemini-flow) - AI swarm framework
- [Model Context Protocol](https://github.com/modelcontextprotocol) - MCP specification
