# Agentic Workstation - Skills Setup

## Quick Start

Build and launch the container with Docker Manager and Chrome DevTools skills:

```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker
./build-unified.sh
```

**Use cache for fast builds** (default). Use `--no-cache` only if needed.

## What's Included

### Docker Manager Skill
- Control VisionFlow container from within agentic-workstation
- Build, start, stop, restart VisionFlow
- Execute commands in VisionFlow
- Stream logs and check status
- Requires: Docker socket mounted at `/var/run/docker.sock` ✅ (now configured)

### Chrome DevTools Skill
- Debug web pages using official Chrome DevTools MCP
- Performance tracing, network inspection, console debugging
- DOM/CSS inspection, screenshots
- Requires: Chromium browser ✅ (installed)

## Usage

### From Claude Code

```
Use Docker Manager to check VisionFlow status
Use Chrome DevTools to debug http://localhost:3001
```

### From Shell

```bash
# Access container
docker exec -it agentic-workstation /bin/zsh

# Test Docker Manager
/home/devuser/.claude/skills/docker-manager/test-skill.sh
visionflow_ctl.sh status

# Test Chrome DevTools
which chrome-devtools-mcp
chromium --version
```

## Verification

After build completes, the script will show:

```
✅ Docker Manager skill installed
✅ Chrome DevTools skill installed
✅ Docker socket mounted
```

## Configuration

### Docker Socket Mount
Location: `docker-compose.unified.yml` line 70
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:rw
```

### BuildKit Workaround
The build script automatically disables BuildKit to avoid Docker Compose 2.40.x `--allow` flag bug.

## Documentation

- Full setup guide: `/DOCKER_MANAGER_SETUP.md`
- Docker Manager: `skills/docker-manager/SKILL.md`
- Chrome DevTools: `skills/chrome-devtools/SKILL.md`
- All skills overview: `SKILLS_INSTALLATION.md`

## Troubleshooting

**Docker socket not found:**
```bash
docker exec agentic-workstation ls -la /var/run/docker.sock
```
Should show: `srw-rw---- 1 root docker`

**Skills not installed:**
```bash
docker exec agentic-workstation ls -la /home/devuser/.claude/skills/
```
Should show `docker-manager/` and `chrome-devtools/`

**Build fails:**
- Check Docker daemon is running: `docker info`
- Verify compose file: `cat docker-compose.unified.yml | grep docker.sock`
- Try rebuild: `./build-unified.sh`

## Development Workflow

1. Edit VisionFlow code
2. Use Docker Manager: `visionflow_ctl.sh restart --rebuild`
3. Test with Chrome DevTools
4. Iterate!

All ready to go! 🚀
