# Skills Installation Guide

This document describes the custom skills added to the agentic-workstation container.

## Installed Skills

### 1. Docker Manager Skill
**Location**: `/home/devuser/.claude/skills/docker-manager/`

**Purpose**: Manage VisionFlow container from within agentic-workstation via Docker socket access.

**Capabilities**:
- Build, start, stop, restart VisionFlow container
- Execute commands in VisionFlow
- Stream logs and check status
- Container discovery on docker_ragflow network
- Full integration with `scripts/launch.sh`

**Usage**:
```
Use Docker Manager to build and restart VisionFlow in dev mode
```

**Documentation**:
- `SKILL.md` - Claude Code skill documentation
- `README.md` - Developer guide
- `QUICKSTART.md` - Quick reference

**Files**:
- `tools/docker_manager.py` - Python Docker SDK client
- `tools/visionflow_ctl.sh` - Zsh wrapper script
- `config/docker-auth.json` - Container mappings

### 2. Chrome DevTools Skill
**Location**: `/home/devuser/.claude/skills/chrome-devtools/`

**Purpose**: Debug web pages directly in Chrome using official Chrome DevTools MCP server.

**Capabilities**:
- Performance tracing and analysis
- Network request inspection
- Console log/error viewing
- DOM/CSS inspection
- JavaScript execution in page context
- Screenshot capture
- Code coverage analysis

**Usage**:
```
Use Chrome DevTools to check console errors on http://localhost:3001
```

**Documentation**:
- `SKILL.md` - Claude Code skill documentation
- `README.md` - Setup and troubleshooting guide

**Files**:
- `config/mcp-config.json` - MCP server configuration

**MCP Server**: `chrome-devtools-mcp@latest` (installed globally via NPM)

## Installation

Both skills are automatically installed when building the agentic-workstation container.

### Rebuild Container with Skills

```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker

# Option 1: Use convenience script
./rebuild-with-docker-manager.sh

# Option 2: Manual rebuild
docker-compose down
docker-compose build --no-cache agentic-workstation
docker-compose up -d agentic-workstation
```

## Verification

### Check Skills Are Installed

```bash
# SSH into container
ssh devuser@localhost -p 2222
# Password: turboflow

# Or docker exec
docker exec -it agentic-workstation /bin/zsh

# Verify Docker Manager
ls -la /home/devuser/.claude/skills/docker-manager/
/home/devuser/.claude/skills/docker-manager/test-skill.sh

# Verify Chrome DevTools
ls -la /home/devuser/.claude/skills/chrome-devtools/
which chrome-devtools-mcp
```

### Test Docker Manager

```bash
# Test from shell
visionflow_ctl.sh status
visionflow_ctl.sh discover

# Test from Claude Code
# In Claude session:
Use Docker Manager to check VisionFlow status
```

### Test Chrome DevTools

```bash
# Test Chrome availability
which chromium
chromium --version
echo $DISPLAY

# Test from Claude Code
# In Claude session:
Use Chrome DevTools to navigate to https://example.com
```

## Dependencies

### Docker Manager Dependencies

**System**:
- Docker socket access (`/var/run/docker.sock`)
- Python 3.x
- jq (for JSON parsing)

**Python**:
- `docker` (Python Docker SDK)

**Mounted**:
- Project path: `/home/devuser/workspace/project/`
- Launch script: `scripts/launch.sh`

### Chrome DevTools Dependencies

**System**:
- Chromium browser (`/usr/bin/chromium`)
- Node.js and NPM
- X11 display (`:1` via VNC)

**NPM**:
- `chrome-devtools-mcp@latest` (global)

**Environment**:
- `DISPLAY=:1`
- VNC running on port 5901

## Configuration Files

### Docker Manager

**docker-auth.json**:
```json
{
  "containers": {
    "visionflow": {
      "name": "visionflow_container",
      "network": "docker_ragflow",
      "image_prefix": "ar-ai-knowledge-graph-webxr"
    }
  },
  "docker_socket": "/var/run/docker.sock",
  "project_path": "/home/devuser/workspace/project",
  "launch_script": "scripts/launch.sh"
}
```

### Chrome DevTools

**mcp-config.json**:
```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "npx",
      "args": ["-y", "chrome-devtools-mcp@latest"],
      "env": {
        "CHROME_PATH": "/usr/bin/chromium",
        "DISPLAY": ":1"
      }
    }
  }
}
```

## Troubleshooting

### Docker Manager Issues

**Socket Permission Denied**:
```bash
sudo chmod 666 /var/run/docker.sock
sudo usermod -aG docker devuser
```

**Container Not Found**:
```bash
docker ps -a | grep visionflow
visionflow_ctl.sh discover
```

**Python Dependencies Missing**:
```bash
pip3 install docker --break-system-packages
```

### Chrome DevTools Issues

**Chrome Won't Launch**:
```bash
# Check Chromium
which chromium
chromium --version

# Test display
echo $DISPLAY
DISPLAY=:1 xdpyinfo
```

**MCP Server Not Found**:
```bash
# Check global installation
npm list -g chrome-devtools-mcp

# Reinstall
npm install -g chrome-devtools-mcp@latest
```

**Display Connection Failed**:
```bash
# Check VNC is running
sudo supervisorctl status xvnc

# Restart VNC
sudo supervisorctl restart xvnc
```

## Usage Examples

### Combined Workflow: Debug VisionFlow

```
I need to debug VisionFlow. Please:
1. Use Docker Manager to restart VisionFlow in dev mode
2. Wait for it to start
3. Use Chrome DevTools to navigate to http://localhost:3001
4. Check console for errors
5. Record a 10-second performance trace
6. Report any issues found
```

### Docker Manager Only

```
Use Docker Manager to:
1. Stop VisionFlow
2. Build with no cache
3. Start in dev mode
4. Show status and logs
```

### Chrome DevTools Only

```
Use Chrome DevTools to:
1. Load VisionFlow homepage
2. Get all network requests
3. Check for failed requests
4. Show console errors
5. Take a screenshot
```

## Integration with Development Workflow

### Typical Development Cycle

1. **Edit Code** - Make changes to VisionFlow source
2. **Rebuild Container** - Use Docker Manager to restart
3. **Verify Changes** - Use Chrome DevTools to test
4. **Debug Issues** - Inspect console, network, performance
5. **Iterate** - Fix and repeat

### Example Session

```
# User edits backend/src/main.rs

Use Docker Manager to restart VisionFlow with rebuild

# Wait for restart...

Use Chrome DevTools to:
1. Navigate to http://localhost:3001
2. Check for console errors
3. Test WebXR initialization
4. Record performance trace

# Report findings and fix any issues
```

## Skill Architecture

### Docker Manager Architecture

```
Claude Code → docker_manager.py → Docker Socket → Docker Daemon → visionflow_container
                 ↓
          launch.sh wrapper → scripts/launch.sh
```

### Chrome DevTools Architecture

```
Claude Code → MCP Client → chrome-devtools-mcp (NPX)
                              ↓
                          Chromium (headless)
                              ↓
                         Remote Debugging
                              ↓
                    VisionFlow (localhost:3001)
```

## Dockerfile Integration

### Skills Copy Phase (PHASE 13)

```dockerfile
# Copy skills
COPY --chown=devuser:devuser skills/ /home/devuser/.claude/skills/

# Copy docker-manager skill from multi-agent-docker
COPY --chown=devuser:devuser multi-agent-docker/skills/docker-manager/ /home/devuser/.claude/skills/docker-manager/

# Copy chrome-devtools skill from multi-agent-docker
COPY --chown=devuser:devuser multi-agent-docker/skills/chrome-devtools/ /home/devuser/.claude/skills/chrome-devtools/
```

### Dependencies Installation

```dockerfile
# Node.js packages (PHASE 5)
RUN npm install -g \
    @anthropic-ai/claude-code \
    chrome-devtools-mcp@latest \
    ...

# Python packages (PHASE 13)
RUN pip3 install --no-cache-dir --break-system-packages \
    docker \
    ...

# Make tools executable
RUN find /home/devuser/.claude/skills -name "*.py" -path "*/tools/*" -exec chmod +x {} \;
RUN find /home/devuser/.claude/skills -name "*.sh" -path "*/tools/*" -exec chmod +x {} \;
```

## Resources

### Docker Manager
- **SKILL.md**: Full skill documentation
- **README.md**: Developer guide and troubleshooting
- **QUICKSTART.md**: Quick reference guide
- **test-skill.sh**: Installation verification script

### Chrome DevTools
- **SKILL.md**: Full skill documentation
- **README.md**: Setup and troubleshooting guide
- **GitHub**: https://github.com/ChromeDevTools/chrome-devtools-mcp
- **Blog Post**: https://developer.chrome.com/blog/chrome-devtools-mcp

## Future Enhancements

### Docker Manager
- [ ] Automated rollback on failed deployments
- [ ] Multi-container orchestration
- [ ] Performance metrics collection
- [ ] Blue-green deployment support

### Chrome DevTools
- [ ] Mobile device emulation
- [ ] Lighthouse performance audits
- [ ] Accessibility testing
- [ ] Visual regression testing

## Support

For issues:
1. Check skill documentation (SKILL.md, README.md)
2. Run test scripts if available
3. Verify dependencies are installed
4. Check container logs: `docker logs agentic-workstation`
5. Report issues in project repository
