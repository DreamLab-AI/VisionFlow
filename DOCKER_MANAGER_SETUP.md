# Docker Manager & Chrome DevTools Setup

## Summary of Changes

Two new skills have been added to the agentic-workstation container:

### 1. Docker Manager Skill
**Purpose**: Control VisionFlow container from within agentic-workstation

**Features**:
- Build, start, stop, restart VisionFlow
- Execute commands in VisionFlow
- Stream logs and check status
- Container discovery

### 2. Chrome DevTools Skill
**Purpose**: Debug web pages using official Chrome DevTools MCP

**Features**:
- Performance tracing
- Network inspection
- Console debugging
- DOM/CSS inspection
- Screenshots

## Critical Fix Applied

**Issue**: Docker socket was not mounted in the container.

**Solution**: Added Docker socket mount to `docker-compose.unified.yml`:
```yaml
volumes:
  # ... other volumes ...
  # Docker socket for Docker Manager skill (inter-container control)
  - /var/run/docker.sock:/var/run/docker.sock:rw
```

## Files Modified

1. **multi-agent-docker/docker-compose.unified.yml**
   - Added Docker socket mount (line 70)

2. **multi-agent-docker/Dockerfile.unified**
   - Added `chrome-devtools-mcp@latest` NPM package (line 125)
   - Added `docker` Python package (line 295)
   - Added docker-manager skill copy (line 286)
   - Added chrome-devtools skill copy (line 289)

3. **Created Skills**:
   - `multi-agent-docker/skills/docker-manager/` (complete skill)
   - `multi-agent-docker/skills/chrome-devtools/` (complete skill)

4. **Updated Scripts**:
   - `multi-agent-docker/build-unified.sh` (enhanced with skills verification)

## Rebuild Instructions

### Quick Rebuild (Recommended)

```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker
./build-unified.sh
```

The script will:
1. Verify skills exist
2. Build Docker image (with cache for speed)
3. Launch container with docker-compose
4. Verify skills and Docker socket are installed
5. Display access information

To rebuild without cache (slower):
```bash
./build-unified.sh --no-cache
```

### Manual Rebuild

```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker

# Stop container
docker stop agentic-workstation
docker rm agentic-workstation

# Build (disable BuildKit to avoid --allow flag bug)
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
docker compose -f docker-compose.unified.yml build --no-cache agentic-workstation

# Start
docker compose -f docker-compose.unified.yml up -d agentic-workstation
```

## Verification

### 1. Check Container is Running

```bash
docker ps | grep agentic-workstation
```

### 2. Access Container

```bash
# SSH
ssh devuser@localhost -p 2222
# Password: turboflow

# Or docker exec
docker exec -it agentic-workstation /bin/zsh
```

### 3. Verify Docker Socket

```bash
# Inside container
ls -la /var/run/docker.sock
docker ps
```

Expected output:
```
srw-rw---- 1 root docker 0 Oct 23 15:00 /var/run/docker.sock
```

### 4. Test Docker Manager Skill

```bash
# Inside container
/home/devuser/.claude/skills/docker-manager/test-skill.sh
```

Expected: All tests pass ✓

### 5. Test VisionFlow Control

```bash
# Inside container
visionflow_ctl.sh status
visionflow_ctl.sh discover
```

### 6. Test Chrome DevTools

```bash
# Inside container
which chrome-devtools-mcp
chromium --version
echo $DISPLAY
```

## Usage Examples

### From Claude Code

Once inside the container with Claude Code:

**Docker Manager**:
```
Use Docker Manager to check VisionFlow status
```

```
Use Docker Manager to build and restart VisionFlow in dev mode
```

```
Use Docker Manager to show the last 50 lines of VisionFlow logs
```

**Chrome DevTools**:
```
Use Chrome DevTools to check console errors on http://localhost:3001
```

```
Use Chrome DevTools to record a 10-second performance trace of VisionFlow
```

**Combined Workflow**:
```
I need to debug VisionFlow. Please:
1. Use Docker Manager to restart VisionFlow in dev mode
2. Wait for it to start
3. Use Chrome DevTools to navigate to http://localhost:3001
4. Check console for errors
5. Record a performance trace
6. Report any issues found
```

### From Shell

```bash
# Docker Manager commands
visionflow_ctl.sh status
visionflow_ctl.sh build --no-cache
visionflow_ctl.sh restart --rebuild
visionflow_ctl.sh logs -n 100 -f
visionflow_ctl.sh exec "npm run test"
visionflow_ctl.sh discover

# Chrome DevTools (via NPX)
npx -y chrome-devtools-mcp@latest
```

## Troubleshooting

### Docker Socket Permission Denied

```bash
# Inside container
sudo chmod 666 /var/run/docker.sock
# or
sudo usermod -aG docker devuser
exec zsh
```

### Container Not Found

```bash
# Check if VisionFlow is running
docker ps -a | grep visionflow

# Discover all containers
visionflow_ctl.sh discover
```

### Chrome Won't Launch

```bash
# Check Chromium
which chromium
chromium --version

# Check display
echo $DISPLAY
DISPLAY=:1 xdpyinfo

# Check VNC is running
sudo supervisorctl status xvnc
```

### Build Fails with "--allow" Error

This is a Docker Compose 2.40.x bug. The rebuild script automatically disables BuildKit:

```bash
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
```

If building manually, set these variables first.

### Skills Not Found After Rebuild

```bash
# Check skills are copied
docker exec agentic-workstation ls -la /home/devuser/.claude/skills/

# Should see:
# docker-manager/
# chrome-devtools/
# (plus other skills)
```

## Architecture Diagram

```
┌────────────────────────────────────────────────────┐
│                    Host Machine                    │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  agentic-workstation container               │ │
│  │                                              │ │
│  │  Claude Code + Docker Manager + Chrome DT    │ │
│  │       ↓                      ↓               │ │
│  │  /var/run/docker.sock   Chromium (headless) │ │
│  └──────────────────────────────────────────────┘ │
│         │                                          │
│         ↓ (Docker socket access)                  │
│  ┌──────────────────────────────────────────────┐ │
│  │      Docker Daemon                           │ │
│  └──────────────────────────────────────────────┘ │
│         │                                          │
│         ↓                                          │
│  ┌──────────────────────────────────────────────┐ │
│  │  visionflow_container                        │ │
│  │  (VisionFlow Application)                    │ │
│  │  Port 3001 (dev)                             │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Network: docker_ragflow (bridge)                 │
└────────────────────────────────────────────────────┘
```

## Environment Variables

Set in `.env` file or docker-compose:

```bash
# Project directory to mount
PROJECT_DIR=/mnt/mldata/githubs/AR-AI-Knowledge-Graph

# VisionFlow profile
VISIONFLOW_PROFILE=dev

# Docker socket (already default)
DOCKER_HOST=unix:///var/run/docker.sock

# Chrome/Chromium
CHROME_PATH=/usr/bin/chromium
DISPLAY=:1
```

## Security Notes

**Docker Socket Access**:
- Provides full control over host Docker daemon
- User `devuser` has docker group membership (GID 970)
- All Docker operations are logged
- Restricted to `docker_ragflow` network for container operations

**Chrome DevTools**:
- Chrome launched with `--disable-web-security` for local development
- Remote debugging port (9222) only accessible within container
- No external network access
- Browser data cleared between sessions

## Next Steps

After successful rebuild:

1. **Verify Installation**
   ```bash
   docker exec -it agentic-workstation /home/devuser/.claude/skills/docker-manager/test-skill.sh
   ```

2. **Start Claude Code Session**
   ```bash
   docker exec -it agentic-workstation claude
   ```

3. **Test Docker Manager**
   ```
   Use Docker Manager to check VisionFlow status
   ```

4. **Test Chrome DevTools**
   ```
   Use Chrome DevTools to navigate to http://localhost:3001
   ```

5. **Start Development**
   - Edit VisionFlow code
   - Use Docker Manager to rebuild
   - Use Chrome DevTools to test
   - Iterate!

## Documentation

- **Docker Manager**: `multi-agent-docker/skills/docker-manager/SKILL.md`
- **Chrome DevTools**: `multi-agent-docker/skills/chrome-devtools/SKILL.md`
- **Quick Reference**: `multi-agent-docker/skills/docker-manager/QUICKSTART.md`
- **Complete Guide**: `multi-agent-docker/SKILLS_INSTALLATION.md`

## Support

If issues persist:
1. Check Docker daemon is running: `docker info`
2. Verify socket permissions: `ls -la /var/run/docker.sock`
3. Check container logs: `docker logs agentic-workstation`
4. Review skill documentation in the `skills/` directories
5. Run test suite: `test-skill.sh`

## Known Issues

### Docker Compose 2.40.x --allow Flag Bug
**Symptom**: Build fails with "unknown flag: --allow"
**Solution**: Scripts automatically set `DOCKER_BUILDKIT=0`

### VisionFlow Container Not Found
**Symptom**: "visionflow_container not found"
**Solution**: Ensure VisionFlow is running and on `docker_ragflow` network

### Chrome Display Issues
**Symptom**: Chrome won't launch or display errors
**Solution**: Verify VNC is running and `DISPLAY=:1` is set

## Success Criteria

✅ Container builds without errors
✅ Docker socket is mounted and accessible
✅ Docker Manager test suite passes
✅ Can discover VisionFlow container
✅ Can execute commands in VisionFlow
✅ Chrome DevTools MCP server starts
✅ Chromium launches in headless mode

All criteria met = Ready for development! 🎉
