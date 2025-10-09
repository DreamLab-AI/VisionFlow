# Multi-Agent Docker System Documentation

**Last Updated**: 2025-10-09
**Version**: 2.2 (Z.AI Integration + Automatic Topics Generation)

## ⚠️ Critical: Database Isolation Required

**NEVER run `claude-flow init --force` from `/workspace`** - This creates a shared database that causes container crashes.

The system uses automatic database isolation. No manual initialization is needed.

## Overview

The Multi-Agent Docker System is a containerized AI orchestration platform that provides:

- **Hive-Mind Task Orchestration** with UUID-based session isolation
- **Database Isolation** preventing SQLite lock conflicts
- **MCP (Model Context Protocol)** server infrastructure for AI tool integration
- **Web Summary MCP Server** with Z.AI GLM-4.6 integration for intelligent markdown processing
- **GPU-Accelerated Spring Visualization** telemetry streaming
- **VNC Desktop Environment** for GUI-based AI tools (Blender, QGIS, Playwright)
- **Persistent Logging** for debugging and monitoring
- **Automatic Topics.json Generation** from markdown directory at container startup

## Quick Start

### Prerequisites

Before starting the container, ensure these API keys are configured in your `.env` file:

```bash
# REQUIRED API Keys
GOOGLE_API_KEY=your_google_ai_studio_key_here     # Required for web content summarization
ZAI_API_KEY=your_z_ai_api_key_here                # Required for semantic topic matching
ANTHROPIC_API_KEY=your_anthropic_key_here         # Optional: For Claude API access

# IMPORTANT: Enable YouTube Data API v3 in Google Cloud Console
# The GOOGLE_API_KEY must have YouTube Data API v3 enabled for video transcript extraction
# https://console.cloud.google.com/apis/library/youtube.googleapis.com
```

### Starting the Container

```bash
# Start the container
docker-compose up -d

# Verify API keys are loaded
docker exec multi-agent-container env | grep -E "GOOGLE_API_KEY|ZAI_API_KEY"

# Check topics.json was generated
docker exec multi-agent-container cat /app/core-assets/config/topics.json | jq '.metadata'

# Create a hive-mind task session
UUID=$(docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh create "your task here" "high")

# Start the task
docker exec -d multi-agent-container \
  /app/scripts/hive-session-manager.sh start $UUID

# Monitor status
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID

# If experiencing container crashes
docker exec multi-agent-container rm -f /workspace/.swarm/memory.db*
docker-compose restart
```

## Documentation Structure

### Core Concepts
- **[Architecture Overview](01-architecture.md)** - System design and components
- **[Session Isolation](02-session-isolation.md)** - Database isolation and UUID tracking
- **[MCP Infrastructure](03-mcp-infrastructure.md)** - Model Context Protocol servers

### Integration Guides
- **[External Integration](04-external-integration.md)** - Rust/external system integration
- **[Session API Reference](05-session-api.md)** - Complete API documentation
- **[TCP/MCP Telemetry](06-tcp-mcp-telemetry.md)** - Real-time monitoring and visualization

### Operations
- **[Logging System](07-logging.md)** - Persistent logs and debugging
- **[VNC Access](08-vnc-access.md)** - Remote desktop for GUI tools
- **[Security](09-security.md)** - Authentication and access control

### Troubleshooting
- **[Common Issues](10-troubleshooting.md)** - Solutions to known problems
- **[Database Conflicts](11-database-troubleshooting.md)** - SQLite lock resolution

### AI Tools & Services
- **[Web Summary MCP Server](12-web-summary-mcp-server.md)** - Automated markdown link expansion with Z.AI topic matching
  - **Z.AI GLM-4.6 Integration**: Intelligent semantic topic matching (40x cheaper than Claude API)
  - **Google Gemini 2.0 Flash**: Web content and YouTube video summarization
  - **Automatic Topics Generation**: Extracts topics from markdown filenames at container startup
  - **Logseq Compliance**: Proper bullet formatting and CRLF line endings

## Key Features

### 1. Session-Based Isolation
Each external task spawn gets:
- Unique UUID identifier
- Isolated SQLite database
- Dedicated output directory
- Separate log file

### 2. Hybrid Architecture
- **Control Plane**: Docker exec for task spawning (reliable, fast)
- **Data Plane**: TCP/MCP for telemetry streaming (rich data)
- **Visualization**: WebSocket for real-time updates

### 3. Concurrent Task Support
- No SQLite lock conflicts
- Unlimited parallel spawns
- Independent task lifecycles
- Resource isolation

## Directory Structure

```
multi-agent-docker/
├── docs/                       # Documentation (you are here)
├── scripts/                    # Utility scripts
│   ├── hive-session-manager.sh # Session lifecycle management
│   ├── configure-claude-mcp.sh # MCP configuration
│   └── generate-topics-from-markdown.py # Auto-generate topics.json
├── core-assets/
│   ├── config/
│   │   └── topics.json         # Auto-generated from ../data/markdown/*.md
│   ├── scripts/
│   │   └── mcp-tcp-server.js   # TCP MCP server
│   └── mcp.json                # MCP server registry
├── claude-zai/                 # Z.AI container for topic matching
│   ├── Dockerfile
│   ├── wrapper/server.js       # HTTP wrapper for Claude Code
│   └── claude-config.json      # Z.AI configuration
├── logs/                       # Persistent logs (mounted)
│   ├── mcp/                    # MCP server logs
│   ├── supervisor/             # Process manager logs
│   └── entrypoint.log          # Container startup log
├── workspace/                  # Persistent workspace (mounted)
│   ├── .swarm/
│   │   ├── sessions/{UUID}/    # Session working directories
│   │   └── tcp-server-instance/ # TCP server isolation
│   └── ext/
│       ├── data/markdown/      # Markdown files (mounted from ../data/markdown)
│       └── hive-sessions/{UUID}/ # Session outputs
└── docker-compose.yml
```

### Markdown Directory Mount

The system expects a `../data/markdown/` directory (relative to `multi-agent-docker/`) containing your markdown files. This directory is:

1. **Mounted read-only** into the container at `/workspace/ext/data/markdown`
2. **Parsed at container startup** to generate `/app/core-assets/config/topics.json`
3. **Used by Web Summary MCP Server** for semantic topic matching

**Host Directory Structure**:
```
/home/user/
├── multi-agent-docker/    # Docker build context
│   └── docker-compose.yml
└── data/
    └── markdown/          # Your .md files (180+ files)
        ├── AI Video.md
        ├── Stable Diffusion.md
        ├── 3D and 4D.md
        └── ...
```

**Container View**:
- Host: `../data/markdown/*.md` → Container: `/workspace/ext/data/markdown/*.md`
- Generated: `/app/core-assets/config/topics.json` (created at startup)

## Container Ports

| Port | Service | Purpose |
|------|---------|---------|
| 9500 | TCP MCP Server | Primary MCP communication |
| 9502 | Claude-Flow TCP | Isolated session MCP |
| 3002 | WebSocket Bridge | Real-time telemetry streaming |
| 5901 | VNC Server | Remote desktop access |
| 6901 | noVNC Web | Browser-based VNC |
| 9876-9880 | GUI MCP Servers | Blender, QGIS, Playwright, etc. |

## Environment Variables

### Required API Keys

| Variable | Required | Purpose | Notes |
|----------|----------|---------|-------|
| `GOOGLE_API_KEY` | **YES** | Google Gemini 2.0 Flash API access | **Must have YouTube Data API v3 enabled** in Google Cloud Console |
| `ZAI_API_KEY` | **YES** | Z.AI GLM-4.6 for semantic topic matching | Used by claude-zai container for intelligent link analysis |
| `ANTHROPIC_API_KEY` | Optional | Claude API access | Only needed if using Claude API directly |

### System Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `WS_AUTH_TOKEN` | - | WebSocket authentication |
| `TCP_AUTH_TOKEN` | - | TCP MCP authentication |
| `NODE_ENV` | production | Environment mode |

### YouTube API Setup

**CRITICAL**: The `GOOGLE_API_KEY` must have YouTube Data API v3 enabled:

1. Visit [Google Cloud Console APIs Library](https://console.cloud.google.com/apis/library/youtube.googleapis.com)
2. Select your project
3. Click "Enable" on YouTube Data API v3
4. Verify with: `curl "https://www.googleapis.com/youtube/v3/videos?part=snippet&id=dQw4w9WgXcQ&key=YOUR_KEY"`

Without YouTube API enabled, video transcript extraction will fail silently.

## Getting Help

- Check the relevant doc in this directory
- Review logs: `tail -f logs/mcp/*.log`
- Inspect sessions: `/app/scripts/hive-session-manager.sh list`
- Report issues with full logs and session UUID

### Common Setup Issues

#### API Keys Not Working

```bash
# Verify API keys are loaded correctly
docker exec multi-agent-container env | grep -E "GOOGLE_API_KEY|ZAI_API_KEY"

# Test Google API key
docker exec multi-agent-container bash -c 'curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" | jq -r ".models[0].name"'

# Test Z.AI container
curl -X POST http://localhost:9600/prompt -H "Content-Type: application/json" -d '{"prompt":"test"}'
```

#### Topics.json Not Generated

```bash
# Check if markdown directory is mounted
docker exec multi-agent-container ls -la /workspace/ext/data/markdown | head -10

# Verify topics.json exists
docker exec multi-agent-container cat /app/core-assets/config/topics.json | jq '.metadata'

# Manually regenerate topics.json
docker exec multi-agent-container python3 /app/scripts/generate-topics-from-markdown.py \
  /workspace/ext/data/markdown \
  /app/core-assets/config/topics.json

# Check entrypoint logs
docker exec multi-agent-container tail -50 /workspace/.setup.log | grep -A 5 "topics.json"
```

#### YouTube Transcript Failures

If YouTube video summaries are failing:

1. **Enable YouTube Data API v3** in Google Cloud Console
2. **Verify API quota**: Check [Google Cloud Console Quotas](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas)
3. **Test the API**:
   ```bash
   docker exec multi-agent-container bash -c 'curl -s "https://www.googleapis.com/youtube/v3/videos?part=snippet&id=dQw4w9WgXcQ&key=$GOOGLE_API_KEY" | jq ".items[0].snippet.title"'
   ```

## Important Usage Notes

### Do NOT Use These Commands
- ❌ `claude-flow init --force` (creates shared database conflicts)
- ❌ `claude-flow-init-agents` alias (same issue)
- ❌ Direct `claude-flow hive-mind spawn` from `/workspace` (bypasses isolation)

### DO Use These Patterns
- ✅ Session manager API for all task spawns
- ✅ MCP servers handle initialization automatically
- ✅ Wrapper handles CLI commands transparently
- ✅ Each session gets isolated database automatically

### Database Health Check
```bash
# Should show ONLY isolated databases
docker exec multi-agent-container find /workspace/.swarm -name "memory.db" -ls

# Correct output:
#   tcp-server-instance/.swarm/memory.db
#   sessions/{UUID}/.swarm/memory.db
#   root-cli-instance/.swarm/memory.db

# If you see /workspace/.swarm/memory.db → DELETE IT
```

## Next Steps

1. Read [Architecture Overview](01-architecture.md) to understand the system
2. Follow [External Integration](04-external-integration.md) to integrate with your Rust system
3. Check [Session API Reference](05-session-api.md) for detailed API documentation
