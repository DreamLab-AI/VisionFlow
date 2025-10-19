# Container Access Methods

**All the ways to access and interact with the Turbo Flow container.**

---

## Access Methods Overview

| Method | Use Case | Login User | Working Directory |
|--------|----------|------------|-------------------|
| **VNC** | GUI apps, desktop environment | devuser | /home/devuser |
| **SSH** | Remote shell, file transfer | devuser | /home/devuser |
| **docker exec** | Quick commands, debugging | root (default) | /home/zai-user |
| **docker exec -u devuser** | Development work | devuser | /home/devuser |
| **Management API** | Automated task submission | N/A (HTTP) | Task-specific |

---

## VNC Desktop (Recommended for GUI Work)

### Connect

**Server**: `localhost:5901`
**Password**: `turboflow`

**VNC Clients**:
- TigerVNC (Linux/Windows/macOS)
- Remmina (Linux)
- RealVNC Viewer (all platforms)
- macOS Screen Sharing

```bash
# Linux
vncviewer localhost:5901

# macOS
open vnc://localhost:5901

# Or use Screen Sharing.app and connect to localhost:5901
```

### Features

- Full XFCE4 desktop environment
- Chromium and Firefox browsers
- Blender, QGIS, KiCAD (GUI apps)
- File manager (Thunar)
- Multiple terminal windows
- Pre-configured 8-window tmux session

### Default Layout

The tmux workspace auto-starts with 8 windows:

```bash
# Attach to tmux workspace
tmux attach-session -t workspace

# Windows:
# 0: Claude-Main        - Primary Claude Code workspace
# 1: Claude-Agent       - Agent execution and testing
# 2: Services          - Supervisord monitoring
# 3: Development       - Python/Rust/CUDA development
# 4: Logs              - Service logs (split panes)
# 5: System            - htop resource monitoring
# 6: VNC-Status        - VNC connection info
# 7: SSH-Shell         - General purpose shell
```

**tmux Commands**:
- `Ctrl+B` then `0-7` - Switch to window
- `Ctrl+B` then `d` - Detach
- `Ctrl+B` then `c` - Create new window
- `Ctrl+B` then `"` - Split pane horizontally
- `Ctrl+B` then `%` - Split pane vertically

---

## SSH Access

### Connect

```bash
ssh -p 2222 devuser@localhost
# Password: turboflow
```

### With Key-Based Auth (Recommended)

```bash
# Copy your public key to container
ssh-copy-id -p 2222 devuser@localhost

# Or manually
cat ~/.ssh/id_rsa.pub | ssh -p 2222 devuser@localhost \
  'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'

# Connect without password
ssh -p 2222 devuser@localhost
```

### SSH Features

- **Persistent sessions**: Use screen or tmux
- **X11 forwarding**: `ssh -X -p 2222 devuser@localhost`
- **File transfer**: `scp -P 2222 file.txt devuser@localhost:~/`
- **Port forwarding**: `ssh -L 8000:localhost:8000 -p 2222 devuser@localhost`

### SSHFS (Mount Container Filesystem)

```bash
# Install sshfs
sudo apt install sshfs  # Debian/Ubuntu
brew install macfuse && brew install sshfs  # macOS

# Mount container workspace on host
mkdir ~/turbo-flow-workspace
sshfs -p 2222 devuser@localhost:/home/devuser/workspace ~/turbo-flow-workspace

# Unmount
fusermount -u ~/turbo-flow-workspace  # Linux
umount ~/turbo-flow-workspace  # macOS
```

---

## Docker Exec

### Quick Access

```bash
# Default login (as root)
docker exec -it turbo-flow-unified zsh
# You'll be: root at /home/zai-user

# Login as devuser (recommended for development)
docker exec -u devuser -it turbo-flow-unified zsh
# You'll be: devuser at /home/devuser

# Direct to workspace
docker exec -u devuser -it turbo-flow-unified zsh -c 'cd ~/workspace && zsh'

# Start Claude Code directly
docker exec -u devuser -it turbo-flow-unified claude
```

### Running Single Commands

```bash
# Check status
docker exec -u devuser turbo-flow-unified whoami

# List files
docker exec -u devuser turbo-flow-unified ls -la ~/workspace

# Count agents
docker exec -u devuser turbo-flow-unified bash -c 'ls ~/agents/*.md | wc -l'

# Check services
docker exec turbo-flow-unified supervisorctl status

# View logs
docker exec turbo-flow-unified supervisorctl tail -f management-api

# Check API health
docker exec turbo-flow-unified curl -s http://localhost:9090/health
```

### Understanding Login Locations

**Default `docker exec -it turbo-flow-unified zsh`**:
- **User**: root
- **Working Directory**: `/home/zai-user`
- **Why**: WORKDIR in Dockerfile points to zai-user

**Recommended `docker exec -u devuser -it turbo-flow-unified zsh`**:
- **User**: devuser
- **Working Directory**: `/home/devuser`
- **Environment**: Full dev environment with Claude, skills, agents

**Switch after login**:
```bash
# If you logged in as root
su - devuser
```

---

## Multi-User Switching

### Inside the Container

Once logged in as devuser, switch to other AI users:

```bash
# Switch to Gemini
as-gemini
whoami  # gemini-user
pwd     # /home/gemini-user

# Switch to OpenAI
as-openai
whoami  # openai-user

# Switch to Z.AI service user
as-zai
whoami  # zai-user
```

### User Isolation

Each user has:

| User | UID:GID | Credentials Location | Purpose |
|------|---------|---------------------|---------|
| **devuser** | 1000:1000 | `~/.config/claude/` | Primary Claude Code development |
| **gemini-user** | 1001:1001 | `~/.config/gemini/` | Google Gemini tools |
| **openai-user** | 1002:1002 | `~/.config/openai/` | OpenAI tools |
| **zai-user** | 1003:1003 | `~/.config/zai/` | Z.AI service (port 9600) |

### Direct Login as Specific User

```bash
docker exec -u gemini-user -it turbo-flow-unified zsh
docker exec -u openai-user -it turbo-flow-unified zsh
docker exec -u zai-user -it turbo-flow-unified zsh
```

---

## Service Management

### Check All Services

```bash
# As root or devuser with sudo
docker exec turbo-flow-unified supervisorctl status

# Output:
# claude-zai         RUNNING   pid 47, uptime 1:23:45
# dbus               RUNNING   pid 41, uptime 1:23:45
# dbus-user          RUNNING   pid 42, uptime 1:23:45
# management-api     RUNNING   pid 46, uptime 1:23:45
# sshd               RUNNING   pid 43, uptime 1:23:45
# xfce4              RUNNING   pid 45, uptime 1:23:45
# xvnc               RUNNING   pid 44, uptime 1:23:45
```

### Restart Services

```bash
docker exec turbo-flow-unified supervisorctl restart xfce4
docker exec turbo-flow-unified supervisorctl restart management-api
docker exec turbo-flow-unified supervisorctl restart claude-zai
```

### View Service Logs

```bash
# Follow logs in real-time
docker exec turbo-flow-unified supervisorctl tail -f management-api
docker exec turbo-flow-unified supervisorctl tail -f claude-zai

# View last 100 lines
docker exec turbo-flow-unified supervisorctl tail management-api

# Direct log file access
docker exec turbo-flow-unified tail -f /var/log/management-api.log
docker exec turbo-flow-unified tail -f /var/log/supervisord.log
```

---

## File Operations

### Copy Files Into Container

```bash
# Copy local file to container workspace
docker cp local-file.txt turbo-flow-unified:/home/devuser/workspace/

# Copy directory
docker cp local-dir/ turbo-flow-unified:/home/devuser/workspace/

# Set correct ownership
docker exec turbo-flow-unified chown -R devuser:devuser /home/devuser/workspace/
```

### Copy Files From Container

```bash
# Copy file from container to host
docker cp turbo-flow-unified:/home/devuser/workspace/output.txt ./

# Copy entire workspace
docker cp turbo-flow-unified:/home/devuser/workspace/ ./turbo-workspace/
```

### Edit Files in Container

```bash
# Using vim
docker exec -u devuser -it turbo-flow-unified vim ~/workspace/file.txt

# Using nano
docker exec -u devuser -it turbo-flow-unified nano ~/workspace/file.txt

# Using code-server (web IDE)
# Open browser: http://localhost:8080
```

---

## External Project Mounting

**Note**: Currently requires manual volume mount configuration.

### Docker Run Method

```bash
docker run --rm --network docker_ragflow \
  --name turbo-flow-unified -d \
  -p 2222:22 -p 5901:5901 -p 8080:8080 -p 9090:9090 \
  --env-file .env \
  --runtime=nvidia \
  --shm-size=32g \
  -v ${HOME}/.claude:/mnt/host-claude:ro \
  -v /path/to/your/project:/home/devuser/workspace/project:rw \
  -v workspace:/home/devuser/workspace \
  -v agents:/home/devuser/agents \
  -v claude-config:/home/devuser/.claude \
  turbo-flow-unified:latest
```

### Verify Mount

```bash
# Inside container
docker exec -u devuser turbo-flow-unified ls -la ~/workspace/project

# If empty, mount not configured
# Add -v flag to docker run command above
```

---

## Management API (HTTP)

### Create Task

```bash
curl -X POST http://localhost:9090/v1/tasks \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  -d '{
    "agent": "python-developer",
    "task": "Create a FastAPI REST API",
    "provider": "claude-flow"
  }'
```

### Check Task Status

```bash
curl http://localhost:9090/v1/tasks/{taskId} \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure'
```

See [Management API Guide](management-api.md) for complete reference.

---

## Comparison Table

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **VNC** | Full desktop, GUI apps, persistent | Requires VNC client | GUI work, Blender, browsers |
| **SSH** | Native terminal, X11 forwarding | Network required | Remote work, file transfer |
| **docker exec** | No password, any user, fast | Less persistent | Quick commands, debugging |
| **Management API** | Automation, CI/CD integration | HTTP only, no interactive | Automated tasks, webhooks |

---

## Troubleshooting

### VNC Shows Black Screen

```bash
# Restart desktop environment
docker exec turbo-flow-unified supervisorctl restart xfce4 xvnc

# Check logs
docker exec turbo-flow-unified supervisorctl tail -f xfce4
```

### SSH Connection Refused

```bash
# Check SSH service
docker exec turbo-flow-unified supervisorctl status sshd

# Restart SSH
docker exec turbo-flow-unified supervisorctl restart sshd
```

### "Command not found" in docker exec

```bash
# Commands available depend on user's PATH
# Use absolute paths or login shell:

# Wrong
docker exec turbo-flow-unified claude

# Right
docker exec -u devuser -it turbo-flow-unified zsh -l -c claude
```

### Can't Access ~/workspace Files

```bash
# Check you're logged in as correct user
whoami  # Should be: devuser

# Check working directory
pwd  # Should be: /home/devuser or /home/devuser/workspace

# If root, switch to devuser
su - devuser
```

---

## Quick Reference

### Access Methods

```bash
# VNC
vncviewer localhost:5901

# SSH
ssh -p 2222 devuser@localhost

# Docker exec as devuser
docker exec -u devuser -it turbo-flow-unified zsh

# Docker exec as root
docker exec -it turbo-flow-unified zsh

# Start Claude directly
docker exec -u devuser -it turbo-flow-unified claude
```

### Default Credentials

```bash
# VNC password
turboflow

# SSH
devuser:turboflow

# Management API
Authorization: Bearer change-this-secret-key-to-something-secure
```

### User Switching

```bash
as-gemini          # Switch to gemini-user
as-openai          # Switch to openai-user
as-zai             # Switch to zai-user
```

### Service Management

```bash
# Status
docker exec turbo-flow-unified supervisorctl status

# Restart
docker exec turbo-flow-unified supervisorctl restart {service}

# Logs
docker exec turbo-flow-unified supervisorctl tail -f {service}
```

---

**Choose the access method that fits your workflow!** Most users prefer VNC for development and SSH for remote access.
