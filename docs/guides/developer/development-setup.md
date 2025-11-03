---
title: Development Setup
description: Complete guide for setting up your Turbo Flow Claude development environment
category: developer-guide
tags: [setup, installation, environment, configuration]
difficulty: beginner
last_updated: 2025-10-27
related:
  - /docs/guides/developer/adding-a-feature.md
  - /docs/guides/developer/testing-guide.md
  - /docs/getting-started/README.md
---

# Development Setup

> **⚠️ Work in Progress**: This guide is currently under development. Content will be expanded in future updates.

## Overview

This guide walks you through setting up a complete development environment for Turbo Flow Claude, including all dependencies, tools, and configurations needed for productive development.

## Prerequisites

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+ recommended)
- macOS 11+
- Windows 10+ with WSL2

**Hardware:**
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space
- Multi-core processor

### Required Software

**Essential tools:**
- Node.js 18+ and npm
- Git 2.30+
- Docker 20.10+ (for containerized development)
- Python 3.9+ (for agent scripts)

**Optional but recommended:**
- tmux or screen (terminal multiplexing)
- Visual Studio Code or similar IDE
- curl and jq (API testing)

## Quick Start Installation

### 1. Clone the Repository

```bash
# Clone the project
git clone https://github.com/ruvnet/claude-flow.git
cd claude-flow

# Or if using Turbo Flow Claude directly
cd /home/devuser/workspace/project
```

### 2. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Claude Flow globally
npm install -g claude-flow@alpha

# Verify installation
npx claude-flow --version
```

### 3. Configure MCP Servers

**Add required MCP server (Claude Flow):**
```bash
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

**Add optional MCP servers:**
```bash
# Enhanced coordination (optional)
claude mcp add ruv-swarm npx ruv-swarm mcp start

# Cloud features (optional, requires authentication)
claude mcp add flow-nexus npx flow-nexus@latest mcp start
```

### 4. Verify Setup

```bash
# Run health check
npm run build
npm run test

# Check MCP tools availability
npx claude-flow mcp status
```

## Detailed Configuration

### Environment Variables

**Create `.env` file in project root:**

```bash
# Core configuration
NODE_ENV=development
LOG_LEVEL=debug

# API Keys (never commit these!)
ANTHROPIC_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here

# Optional: Flow Nexus (if using cloud features)
FLOW_NEXUS_USER_ID=your_user_id
FLOW_NEXUS_API_KEY=your_api_key

# Z.AI Service (if running locally)
ZAI_PORT=9600
ZAI_WORKERS=4
```

**Security:** Never commit `.env` files to version control. Use `.env.example` for templates.

### Project Structure Setup

**Ensure proper directory structure:**

```bash
# Create required directories
mkdir -p src tests docs config scripts examples
mkdir -p docs/{guides,tutorials,reference,architecture}
mkdir -p tests/{unit,integration,e2e}
```

### IDE Configuration

#### Visual Studio Code

**Recommended extensions:**
- ESLint
- Prettier
- TypeScript and JavaScript
- GitLens
- Docker

**Create `.vscode/settings.json`:**
```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.tsdk": "node_modules/typescript/lib",
  "files.exclude": {
    "node_modules": true,
    "dist": true
  }
}
```

### Git Configuration

**Configure commit hooks:**

```bash
# Install husky for git hooks
npm install --save-dev husky
npx husky install

# Add pre-commit hook
npx husky add .husky/pre-commit "npm run lint && npm run typecheck"
```

**Set up Git user:**
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Container Development (Docker)

### Using the Turbo Flow Unified Container

**Start the container:**
```bash
docker run -d \
  --name turbo-flow-unified \
  -p 2222:22 \
  -p 5901:5901 \
  -p 8080:8080 \
  -p 9090:9090 \
  -v $(pwd):/home/devuser/workspace/project \
  turbo-flow-unified:latest
```

**Access methods:**
- SSH: `ssh devuser@localhost -p 2222` (password: `turboflow`)
- VNC: `localhost:5901` (password: `turboflow`)
- code-server: `http://localhost:8080`
- Management API: `http://localhost:9090`

### Multi-User Environment

**Switch between users:**
```bash
# Switch to Gemini user
as-gemini

# Switch to OpenAI user
as-openai

# Switch to Z.AI service user
as-zai

# Return to devuser
exit
```

### tmux Workspace

**Attach to development workspace:**
```bash
tmux attach -t workspace
```

**Window layout:**
- Win 0: Claude-Main (primary workspace)
- Win 1: Claude-Agent (agent execution)
- Win 2: Services (supervisord monitoring)
- Win 3: Development (coding)
- Win 4: Logs (service logs)
- Win 5: System (htop monitoring)

## Agent Templates

### Using 610 Claude Sub-Agents

**Location:** `/home/devuser/agents/*.md`

**Load specific agents:**
```bash
# List available agents
ls /home/devuser/agents/

# Load agent template
cat /home/devuser/agents/tdd-london-swarm.md

# Use in development
npx claude-flow sparc run tdd "feature description"
```

**Key agent categories:**
- Core development: coder, reviewer, tester
- SPARC methodology: specification, architecture, refinement
- GitHub integration: pr-manager, code-review-swarm
- Performance: perf-analyzer, performance-benchmarker

## Service Configuration

### Z.AI Service

**Start Z.AI service:**
```bash
# Service runs on port 9600 (internal only)
curl http://localhost:9600/health

# Test chat endpoint
curl -X POST http://localhost:9600/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt", "timeout": 30000}'
```

### Management API

**Base URL:** `http://localhost:9090`

**Test endpoints:**
```bash
# Health check (no auth)
curl http://localhost:9090/health

# System status (requires API key)
curl -H "X-API-Key: change-this-secret-key" \
  http://localhost:9090/api/status
```

### Gemini Flow Commands

**Initialize Gemini Flow:**
```bash
gf-init        # Initialize coordination
gf-swarm       # Spawn 66 agents
gf-status      # Check status
gf-monitor     # Monitor performance
```

## Development Workflow

### Daily Setup Routine

```bash
# 1. Start development environment
tmux attach -t workspace

# 2. Check service status
sudo supervisorctl status

# 3. Pull latest changes
git pull origin main

# 4. Update dependencies
npm install

# 5. Run tests
npm run test
```

### Agent-Driven Development

**SPARC workflow:**
```bash
# Full TDD workflow
npx claude-flow sparc tdd "feature description"

# Individual phases
npx claude-flow sparc run spec-pseudocode "task"
npx claude-flow sparc run architect "task"
npx claude-flow sparc run integration "task"
```

**Parallel agent execution (Claude Code Task tool):**
```javascript
// Spawn multiple agents concurrently
Task("Researcher", "Analyze requirements", "researcher")
Task("Architect", "Design system", "system-architect")
Task("Coder", "Implement features", "coder")
Task("Tester", "Create tests", "tester")
```

## Testing Setup

### Running Tests

```bash
# Run all tests
npm run test

# Run specific test suite
npm run test -- tests/unit/

# Run with coverage
npm run test -- --coverage

# Watch mode
npm run test -- --watch
```

### Test Configuration

**See:** [Testing Guide](/docs/guides/developer/testing-guide.md) for comprehensive testing strategies.

## Troubleshooting

### Common Issues

**Node modules issues:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Permission errors in container:**
```bash
# Fix ownership
sudo chown -R devuser:devuser /home/devuser/workspace/project
```

**Service not starting:**
```bash
# Check supervisord status
sudo supervisorctl status

# Restart service
sudo supervisorctl restart <service-name>

# View logs
sudo supervisorctl tail -f <service-name>
```

**MCP server connection issues:**
```bash
# Restart MCP servers
claude mcp restart claude-flow

# Check MCP logs
npx claude-flow mcp logs
```

### Getting Help

**Diagnostic commands:**
```bash
# System diagnostics
docker stats turbo-flow-unified

# Container diagnostics
docker exec turbo-flow-unified supervisorctl status

# View logs
tail -f /var/log/supervisord.log
```

## Hook Integration Examples

### Pre-task Hook
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Integration Engineer: Connect OntologyReasoningService to data pipeline"
```

### Post-edit Hook (for each file)
```bash
npx claude-flow@alpha hooks post-edit \
  --file "src/services/ontology_pipeline_service.rs" \
  --memory-key "swarm/integration-engineer/pipeline-service-created"
```

### Post-task Hook
```bash
npx claude-flow@alpha hooks post-task \
  --task-id "integration-engineer-pipeline-integration"
```

### Notify Completion Hook
```bash
npx claude-flow@alpha hooks notify \
  --message "Integration Engineer: Ontology pipeline integration complete"
```

### Session Management Hooks
```bash
# Session restore
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"

# Session end with metrics export
npx claude-flow@alpha hooks session-end --export-metrics true
```

### Memory Storage
All integration activities are stored in `.swarm/memory.db`:
- Task descriptions and IDs
- File modifications
- Completion status
- Notifications

This enables cross-session context restoration and swarm coordination.

## Best Practices

### File Organization
- **Never** save working files to root directory
- Use appropriate subdirectories (src, tests, docs, config)
- Keep modules under 500 lines
- Separate concerns clearly

### Concurrent Operations
- Batch all related operations in single messages
- Use Claude Code's Task tool for agent spawning
- Batch TodoWrite operations (5-10+ todos minimum)
- Parallel file operations when possible

### Security
- Never commit secrets or API keys
- Use environment variables for configuration
- Change default passwords before production
- Regularly update dependencies

### Development Efficiency
- Use agent coordination hooks
- Store decisions in memory
- Enable auto-formatting
- Monitor performance metrics

## Next Steps

After completing setup:
1. Review the [Adding a Feature](/docs/guides/developer/adding-a-feature.md) guide
2. Familiarize yourself with [Testing Guide](/docs/guides/developer/testing-guide.md)
3. Explore the [Architecture Overview](/docs/architecture/README.md)
4. Join the community and contribute

## Related Resources

- [Getting Started](/docs/getting-started/README.md)
- [Adding a Feature](/docs/guides/developer/adding-a-feature.md)
- [Testing Guide](/docs/guides/developer/testing-guide.md)
- [SPARC Methodology](/docs/reference/sparc-methodology.md)

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Community: https://github.com/ruvnet/claude-flow/discussions
