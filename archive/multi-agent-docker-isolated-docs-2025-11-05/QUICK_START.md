# Quick Start - Agentic Workstation

## Build & Launch

```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker
./build-unified.sh
```

**Default: Uses cache (fast rebuild)**
Add `--no-cache` only if you need a clean build.

## What You Get

✅ **Docker Manager Skill** - Control VisionFlow from Claude Code
✅ **Chrome DevTools Skill** - Debug web apps with Chrome DevTools MCP
✅ **Docker Socket Mounted** - Inter-container communication enabled
✅ **All Original Skills** - Web summary, Playwright, Blender, etc.

## After Build

The script verifies and shows:
```
✅ Docker Manager skill installed
✅ Chrome DevTools skill installed
✅ Docker socket mounted
```

## Access Container

**SSH:**
```bash
ssh -p 2222 devuser@localhost
# Password: turboflow
```

**Docker Exec:**
```bash
docker exec -it agentic-workstation /bin/zsh
```

**VNC:** `vnc://localhost:5901` (password: turboflow)
**Code Server:** http://localhost:8080
**Management API:** http://localhost:9090/health

## Test Skills

```bash
# Inside container
/home/devuser/.claude/skills/docker-manager/test-skill.sh
visionflow_ctl.sh status
```

## Use from Claude Code

```
Use Docker Manager to check VisionFlow status
```

```
Use Chrome DevTools to debug http://localhost:3001
```

## Development Workflow

1. Edit VisionFlow code
2. `visionflow_ctl.sh restart --rebuild`
3. Use Chrome DevTools to test
4. Iterate!

## Troubleshooting

**Skills not found:**
```bash
docker exec agentic-workstation ls -la /home/devuser/.claude/skills/
```

**Docker socket missing:**
```bash
docker exec agentic-workstation ls -la /var/run/docker.sock
```

**Rebuild container:**
```bash
docker stop agentic-workstation
docker rm agentic-workstation
./build-unified.sh
```

## Documentation

- Full guide: `../DOCKER_MANAGER_SETUP.md`
- Skills overview: `SKILLS_INSTALLATION.md`
- Skills details: `README_SKILLS.md`

Ready to develop! 🚀
