# Turbo Flow Unified Container

## Discovery Commands

```bash
# System capabilities
aisp init                    # AISP 5.1 Platinum protocol
npx claude-flow@v3alpha doctor  # Full system diagnostics
supervisorctl status         # Running services

# Available resources
ls /home/devuser/agents/     # 610 agent templates
ls ~/.claude/skills/         # Claude Code skills
cat /opt/aisp/index.js       # AISP implementation
```

## Core Protocols

| Protocol | Spec | CLI |
|----------|------|-----|
| **AISP 5.1** | [aisp.md](aisp.md) | `aisp validate`, `aisp binding A B` |
| **Claude Flow v3** | [docs](https://github.com/ruvnet/claude-flow) | `npx claude-flow@v3alpha` |

## Multi-Provider System (zsh aliases)

| User | Switch | Endpoint |
|------|--------|----------|
| devuser | - | Claude Code CLI |
| gemini-user | `as-gemini` | `gf-swarm` |
| zai-user | `as-zai` | `localhost:9600` |
| deepseek-user | `as-deepseek` | DeepSeek API |

## Services

| Port | Service |
|------|---------|
| 2222→22 | SSH |
| 5901 | VNC |
| 8080 | code-server |
| 9090 | Management API |
| 9600 | Z.AI (internal) |

## Rules

1. **Batch**: 1 message = all related operations
2. **Files**: Never save to `/`. Use `/src`, `/tests`, `/docs`
3. **Swarm**: MCP coordinates, Task tool executes
4. **AISP**: Required for specs, instructions, coordination

## Quick Start

```bash
# Initialize
npx claude-flow@v3alpha init --wizard
aisp init

# Swarm
npx claude-flow@v3alpha swarm init --topology hierarchical-mesh
aisp binding coder tester  # Check compatibility before spawn

# Cross-provider
as-zai  # Switch to Z.AI
gf-swarm  # Gemini 66-agent swarm
```

## Credentials (Dev Only)

SSH/VNC: `turboflow` | API: `X-API-Key: change-this-secret-key`

---

## 🚀 Project-Specific: Turbo Flow Claude

### 610 Claude Sub-Agents
- **Repository**: https://github.com/ChrisRoyse/610ClaudeSubagents
- **Location**: `/home/devuser/agents/*.md` (610+ templates)
- **Usage**: Load specific agents with `cat agents/<agent-name>.md`
- **Key Agents**: doc-planner, microtask-breakdown, github-pr-manager, tdd-london-swarm

### Z.AI Service (Cost-Effective Claude API)
**Port**: 9600 (internal only) | **User**: zai-user | **Worker Pool**: 4 concurrent
```bash
# Health check
curl http://localhost:9600/health

# Chat request
curl -X POST http://localhost:9600/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your prompt here", "timeout": 30000}'

# Switch to zai-user
as-zai
```

### Gemini Flow Commands
```bash
gf-init        # Initialize (protocols: a2a,mcp, topology: hierarchical)
gf-swarm       # 66 agents with intelligent coordination
gf-architect   # 5 system architects
gf-coder       # 12 master coders
gf-status      # Swarm status
gf-monitor     # Protocols and performance
gf-health      # Health check
```

### Multi-User System
| User | UID | Purpose | Switch |
|------|-----|---------|--------|
| devuser | 1000 | Claude Code, primary dev | - |
| gemini-user | 1001 | Google Gemini, gemini-flow | `as-gemini` |
| openai-user | 1002 | OpenAI Codex | `as-openai` |
| zai-user | 1003 | Z.AI service (port 9600) | `as-zai` |

### tmux Workspace (8 Windows)
**Attach**: `tmux attach -t workspace`
| Win | Name | Purpose |
|-----|------|---------|
| 0 | Claude-Main | Primary workspace |
| 1 | Claude-Agent | Agent execution |
| 2 | Services | supervisord monitoring |
| 3 | Development | Python/Rust/CUDA dev |
| 4 | Logs | Service logs (split) |
| 5 | System | htop monitoring |
| 6 | VNC-Status | VNC info |
| 7 | SSH-Shell | General shell |

### Management API
**Base**: http://localhost:9090 | **Auth**: `X-API-Key: <MANAGEMENT_API_KEY>`
```bash
GET  /health              # Health (no auth)
GET  /api/status          # System status
POST /api/tasks           # Create task
GET  /api/tasks/:id       # Task status
GET  /metrics             # Prometheus metrics
GET  /documentation       # Swagger UI
```

### Diagnostic Commands
```bash
# Service status
sudo supervisorctl status

# Container diagnostics
docker exec turbo-flow-unified supervisorctl status
docker stats turbo-flow-unified

# Logs
sudo supervisorctl tail -f management-api
sudo supervisorctl tail -f claude-zai
tail -f /var/log/supervisord.log

# User switching test
as-gemini whoami  # Should output: gemini-user
```

### Service Ports
| Port | Service | Access |
|------|---------|--------|
| 22 | SSH | Public (mapped to 2222) |
| 5901 | VNC | Public |
| 8080 | code-server | Public |
| 9090 | Management API | Public |
| 9600 | Z.AI | Internal only |

**Security**: Default creds are DEVELOPMENT ONLY. Change before production:
- SSH: `devuser:turboflow`
- VNC: `turboflow`
- Management API: `X-API-Key: change-this-secret-key`

### Development Environment Notes

**Container Modification Best Practices**:
- ✅ **DO**: Modify Dockerfile and entrypoint scripts DIRECTLY in the project
- ❌ **DON'T**: Create patching scripts or temporary fixes
- ✅ **DO**: Edit /home/devuser/workspace/project/multi-agent-docker/ files
- ❌ **DON'T**: Use workarounds - fix the root cause

**Isolated Docker Environment**:
- This container is isolated from external build systems
- Only these validation tools work:
  - \`cargo test\` - Rust project testing
  - \`npm run check\` / \`npm test\` - Node.js validation
  - \`pytest\` - Python testing
- **DO NOT** attempt to:
  - Build external projects directly
  - Run production builds inside container
  - Execute deployment scripts
  - Access external build infrastructure
- **Instead**: Test, validate, and export artifacts

**File Organization**:
- Never save working files to root (/)
- Use appropriate subdirectories:
  - /docs - Documentation
  - /scripts - Helper scripts
  - /tests - Test files
  - /config - Configuration
