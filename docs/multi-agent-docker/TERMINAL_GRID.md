# Terminal Grid Configuration

## Overview

The VNC desktop automatically launches 9 color-coded terminal windows in a 3x3 grid, each with a custom banner identifying its purpose and providing helpful commands.

## Terminal Windows

### 1. 🤖 Claude-Main (Cyan)
- **Directory**: `/home/devuser/workspace`
- **User**: devuser
- **Purpose**: Primary Claude Code development workspace
- **Quick Commands**:
  - `dsp` - Start Claude Code
  - `cd project` - Go to external project mount
  - `tmux attach -t workspace` - Connect to SSH tmux session

### 2. 🤖 Claude-Agent (Magenta)
- **Directory**: `/home/devuser/agents`
- **User**: devuser
- **Purpose**: Agent testing and execution
- **Quick Commands**:
  - `ls *.md | wc -l` - Count available agents
  - `find . -name '*github*'` - Find GitHub agents
  - `cf-swarm "task"` - Launch claude-flow swarm

### 3. ⚙️ Services (Yellow)
- **Directory**: `/home/devuser`
- **User**: devuser (with sudo)
- **Purpose**: Monitor and manage system services
- **Auto-displays**: First 10 services on startup
- **Quick Commands**:
  - `sudo supervisorctl status` - Check all services
  - `sudo supervisorctl tail -f <name>` - View logs
  - `sudo supervisorctl restart <name>` - Restart service
  - `curl http://localhost:9090/health` - Management API health

### 4. 💻 Development (Green)
- **Directory**: `/home/devuser/workspace/project`
- **User**: devuser
- **Purpose**: External project development
- **Mount**: `/mnt/mldata/githubs/AR-AI-Knowledge-Graph` (host)
- **Auto-displays**: Git branch and status if available
- **Features**:
  - Changes persist to host (read-write mount)
  - GPU acceleration enabled
  - Full development toolchain (Python, Rust, Node.js, Docker)

### 5. 🐳 Docker Manager (Magenta)
- **Directory**: `/home/devuser/workspace`
- **User**: devuser (UID 1000)
- **Purpose**: Container and image management
- **Quick Commands**:
  - `docker ps` - List running containers
  - `docker images` - List images
  - `docker compose ps` - Show compose services
  - `docker stats` - Resource usage

### 6. 🔀 Git Version Control (Blue)
- **Directory**: `/home/devuser/workspace/project`
- **User**: devuser (UID 1000)
- **Purpose**: Git operations and version control
- **Auto-displays**: Repository status, current branch, recent changes
- **Quick Commands**:
  - `git status` - Check repository status
  - `git log --oneline -10` - Recent commits
  - `git branch -a` - List all branches
  - `gh pr list` - GitHub PR list

### 7. 🔮 Gemini-Shell (Light Magenta)
- **Directory**: `/home/gemini-user/workspace`
- **User**: gemini-user (UID 1001)
- **Purpose**: Isolated Google Gemini API operations
- **Credentials**: `~/.config/gemini/config.json`
- **Quick Commands**:
  - `gemini-flow --version`
  - `gf-init` - Initialize project
  - `gf-swarm` - Launch 66-agent swarm
  - `gf-status` - Check swarm status

### 8. 🧠 OpenAI-Shell (Light Blue)
- **Directory**: `/home/openai-user/workspace`
- **User**: openai-user (UID 1002)
- **Purpose**: Isolated OpenAI API operations
- **Credentials**: `~/.config/openai/config.json`
- **Features**:
  - Isolated from other users
  - Dedicated workspace volume

### 9. ⚡ Z.AI-Shell (Bright Yellow)
- **Directory**: `/home/zai-user`
- **User**: zai-user (UID 1003)
- **Purpose**: Z.AI service management
- **Service**: http://localhost:9600 (internal only)
- **Credentials**: `~/.config/zai/config.json`
- **Quick Commands**:
  - `curl http://localhost:9600/health` - Check service
- **Features**:
  - 4-worker pool with 50-request queue
  - Cost-effective Claude API wrapper
  - Used by web-summary skill

## Implementation

### Init Scripts Location
```
/home/devuser/.config/terminal-init/
├── init-claude-main.sh
├── init-claude-agent.sh
├── init-services.sh
├── init-development.sh
├── init-docker.sh
├── init-git.sh
├── init-gemini.sh
├── init-openai.sh
└── init-zai.sh
```

### Autostart Script
- **Location**: `/home/devuser/.config/autostart-terminals.sh`
- **Triggered by**: supervisord `terminal-grid` service (priority 300)
- **Delay**: 5 seconds after desktop starts

### Terminal Specifications
- **Emulator**: xfce4-terminal
- **Geometry**: 80x24 characters (optimized for 2048x2048 resolution)
- **Layout**: 3x3 grid (9 terminals total)
- **Shell**: zsh (no tmux in VNC terminals)
- **Colors**: ANSI escape codes for headers and information

## Customization

To modify a terminal's banner:
1. Edit the corresponding init script in `/home/devuser/.config/terminal-init/`
2. Restart the terminal-grid service: `sudo supervisorctl restart terminal-grid`
3. Or manually close and relaunch terminals

## Note on tmux

VNC terminals do NOT use tmux - they are independent shell sessions. This prevents the confusing behavior of multiple terminals showing the same tmux session.

tmux is still available for SSH access:
```bash
ssh devuser@localhost -p 2222
tmux attach -t workspace
```

The tmux workspace has 11 windows (0-10) optimized for SSH usage.
