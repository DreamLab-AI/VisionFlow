---
title: DeepSeek User Setup - Complete
description: Successfully configured `deepseek-user` (UID 1004) in the unified docker container with full agentic-flow integration and DeepSeek API credentials.
type: archive
status: archived
---

# DeepSeek User Setup - Complete

## Overview
Successfully configured `deepseek-user` (UID 1004) in the unified docker container with full agentic-flow integration and DeepSeek API credentials.

## What Was Done

### 1. User Creation
- Created `deepseek-user` (UID 1004) in the running container
- Configured home directory structure: `/home/deepseek-user/{workspace,agentic-flow,.config,.cache}`
- Set up zsh as default shell
- Configured sudo access from devuser: `sudo -u deepseek-user -i`

### 2. Agentic-Flow Setup
- Cloned [ruvnet/agentic-flow](https://github.com/ruvnet/agentic-flow) to `/home/deepseek-user/agentic-flow`
- Installed all npm dependencies (965 packages)
- Version: agentic-flow v1.10.2

### 3. Configuration Files

#### DeepSeek API Config
**Location:** `/home/deepseek-user/.config/deepseek/config.json`
```json
{
  "apiKey": "sk-[your deepseek api key]",
  "baseUrl": "https://api.deepseek.com",
  "model": "deepseek-chat",
  "maxTokens": 4096,
  "temperature": 0.7
}
```

#### Agentic-Flow .env
**Location:** `/home/deepseek-user/agentic-flow/.env`
```env
DEEPSEEK_API_KEY=sk-[your deepseek api key]
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_MAX_TOKENS=4096
DEEPSEEK_TEMPERATURE=0.7

AI_PROVIDER=deepseek
API_KEY=sk-[your deepseek api key]
API_BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
```

### 4. Terminal Integration

#### Virtual Desktop (VNC)
Added DeepSeek terminal to autostart grid:
- **Row 4**: DeepSeek-Shell terminal
- Launches automatically on VNC startup
- Title: "🧠 DeepSeek-Shell"

#### Tmux Workspace
Added Window 11: DeepSeek-Shell
- Location: `/home/deepseek-user/workspace`
- Auto-switches to deepseek-user
- Pre-configured with quick start commands

### 5. Quick Access Commands

Created system-wide alias:
```bash
as-deepseek     # Switch to deepseek-user
```

## Usage

### Method 1: Docker Exec (Direct)
```bash
# Switch to deepseek user
docker exec -u deepseek-user -it agentic-workstation zsh

# Run agentic-flow
docker exec -u deepseek-user agentic-workstation bash -c \
  "cd ~/agentic-flow && npx agentic-flow --help"
```

### Method 2: From Inside Container
```bash
# From devuser, switch to deepseek
sudo -u deepseek-user -i

# Or use the alias
as-deepseek
```

### Method 3: VNC Desktop
1. Connect to VNC: `vnc://localhost:5901`
2. Look for "🧠 DeepSeek-Shell" terminal (bottom row)
3. Already switched to deepseek-user context

### Method 4: Tmux Workspace
```bash
# Attach to tmux
tmux attach -t workspace

# Navigate to window 11
Ctrl+B :11

# Or from command line
tmux select-window -t workspace:11
```

## Testing Commands

### Verify Setup
```bash
docker exec -u deepseek-user agentic-workstation bash -c "
  echo '=== DeepSeek User Test ==='
  whoami
  pwd
  echo
  echo '=== Agentic-Flow Version ==='
  cd ~/agentic-flow && npx agentic-flow --version
  echo
  echo '=== Configuration ==='
  cat ~/.config/deepseek/config.json
"
```

### List Available Agents
```bash
docker exec -u deepseek-user agentic-workstation bash -c \
  "cd ~/agentic-flow && npx agentic-flow agent list summary"
```

### Test Agentic-Flow Commands
```bash
docker exec -u deepseek-user agentic-workstation bash -c \
  "cd ~/agentic-flow && npx agentic-flow config list"
```

## Files Modified

### Container Files (via docker cp)
1. `/home/devuser/.config/init-deepseek.sh` - Terminal initialization script
2. `/home/devuser/.config/autostart-terminals.sh` - Updated with DeepSeek terminal
3. `/home/devuser/.config/tmux-autostart.sh` - Added Window 11 for DeepSeek
4. `/usr/local/bin/as-deepseek` - Quick switch command

### Host Files
1. `multi-agent-docker/unified-config/terminal-init/init-deepseek.sh`
2. `multi-agent-docker/unified-config/scripts/setup-deepseek-user.sh`
3. `multi-agent-docker/unified-config/autostart-terminals.sh`
4. `multi-agent-docker/unified-config/tmux-autostart.sh`

## User Isolation

All users are isolated:
| User | UID | Purpose | Access |
|------|-----|---------|--------|
| devuser | 1000 | Primary development | sudo, all tools |
| gemini-user | 1001 | Google Gemini | `as-gemini` |
| openai-user | 1002 | OpenAI | `as-openai` |
| zai-user | 1003 | Z.AI service | `as-zai` |
| **deepseek-user** | **1004** | **DeepSeek AI** | **`as-deepseek`** |

## Architecture Integration

### Agentic-Flow Features Available:
- ✅ Multi-provider AI support (configured for DeepSeek)
- ✅ 600+ agent templates
- ✅ MCP server management
- ✅ Federation hub capabilities
- ✅ QUIC transport proxy
- ✅ Claude Code integration
- ✅ AgentDB support (if installed)

### Agent Categories:
- Analysis (2 agents)
- Architecture (1 agent)
- Consensus (7 agents)
- Core (5 agents)
- Custom agents
- Development tools
- Flow Nexus integration
- GitHub integration
- Performance optimization
- Security scanning
- Testing frameworks

## Persistence

All changes are live in the running container:
- User persists across container restarts
- Configuration files persist in mounted volumes
- Workspace data persists in `/home/deepseek-user/workspace`
- Agentic-flow installation persists

## Next Steps

### For Permanent Integration:
To make this setup permanent in future container builds, update `Dockerfile.unified`:

```dockerfile
# Add after zai-user creation (around line 118)
RUN useradd -m -u 1004 -s /usr/bin/zsh deepseek-user && \
    mkdir -p /home/deepseek-user/{workspace,.config,.cache}

# Update sudo configuration (around line 121)
RUN echo "devuser ALL=(gemini-user,openai-user,zai-user,deepseek-user) NOPASSWD: ALL" >> /etc/sudoers
```

Then copy the configuration scripts during build phase.

## Verification Status

✅ User created (UID 1004)
✅ Agentic-flow cloned and installed
✅ Dependencies installed (965 packages)
✅ Configuration files created
✅ API credentials configured
✅ Terminal integration complete
✅ Tmux workspace updated
✅ Quick access command created
✅ Tested agent listing
✅ Verified user switching
✅ Confirmed workspace access

## API Credentials Source

Credentials loaded from `.env`:
```env
DEEPSEEK_API_KEY=sk-[your deepseek api key]
DEEPSEEK_BASE_URL=https://api.deepseek.com/v3.2_speciale_expires_on_20251215
```

Note: The base URL in .env includes version/special path, but agentic-flow uses the base domain.

## Support

For issues or questions:
- Agentic-Flow: https://github.com/ruvnet/agentic-flow
- Container: Check supervisord logs
- DeepSeek API: https://api.deepseek.com

---

**Setup completed:** December 2, 2025
**Container:** agentic-workstation (Up 24 hours, healthy)
**Status:** Fully operational ✅
