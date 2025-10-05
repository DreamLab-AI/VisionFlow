# Cleanup Candidates - Multi-Agent Docker

This document lists legacy files and development artifacts that are candidates for removal after the unified container migration.

## Confirmed Legacy - Safe to Remove

### Old GUI Container Assets (Integrated into Main Container)
- `gui-based-tools-docker.old/` - **ENTIRE DIRECTORY**
  - `Dockerfile` - Old separate GUI container
  - `Dockerfile.minimal` - Minimal GUI variant
  - `blender-4.5.1-linux-x64.tar.xz` - Now in main container
  - `addon.py` - Blender addon (copied to main)
  - `autostart.py` - Blender startup (copied to main)
  - `playwright-mcp-server.js` - Now in main container
  - `qgis-mcp-server.js` - Now in main container
  - `pbr-mcp-simple.py` - Now in main container
  - `xstartup` - Old VNC startup
  - `startup.sh` - Old container startup
  - `tessellating-pbr-generator/` - Now in main container

### Backup Files (23 total)
```bash
# MCP config backups
workspace/.mcp.json.bak.20251004_182533
workspace/.mcp.json.bak.20251002_171951
workspace/.mcp.json.bak.20251004_110625
workspace/.mcp.json.backup

# Script backups
setup-workspace.sh.bak
CLAUDE.md.old

# Other backups (run to find all)
find . -name "*.bak*" -o -name "*backup*"
```

### Old Docker Compose Files
Check for legacy compose files:
```bash
docker-compose.dev.yml          # If exists, check if still used
docker-compose.production.yml   # If exists, check if still used
docker-compose.override.yml     # If exists, check if still used
```

## Review Required - Verify Before Removal

### Documentation Files
- `AGENT-BRIEFING.md` - Review: Agent-specific docs, may still be relevant
- `RESILIENCE-STRATEGY.md` - Review: Strategy doc, may still be relevant
- `SECURITY.md` - **KEEP**: Security documentation is important

### Helper Scripts
- `mcp-helper.sh` - Review: Check if still used or replaced by setup-workspace.sh
- `multi-agent.sh` - Review: Check if wrapper script still needed

### Core Assets Proxy Scripts (May be Legacy)
Check `core-assets/scripts/` for old proxy scripts:
- `blender-mcp-proxy.js` - **LEGACY**: Direct MCP now, no proxy needed
- `qgis-mcp-proxy.js` - **LEGACY**: Direct MCP now, no proxy needed
- `pbr-mcp-proxy.js` - **LEGACY**: Direct MCP now, no proxy needed
- `playwright-mcp-proxy.js` - **LEGACY**: Local MCP server used
- `chrome-devtools-mcp-proxy.js` - If exists, legacy

### Workspace Development Files
- `workspace/` subdirectories - Review for dev artifacts:
  - `workspace/docs/` - May contain outdated architecture docs
  - `workspace/tests/` - Check for old test artifacts
  - `workspace/gpu_test/` - Development test directory
  - `workspace/multi-agent-docker/` - Nested duplicate?

## Cleanup Commands

### Safe Cleanup (Backups Only)
```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker

# Remove backup files
find . -name "*.bak*" -delete
find . -name "*backup*" -delete

# Remove old CLAUDE.md
rm -f CLAUDE.md.old
```

### Complete Legacy Cleanup (After Verification)
```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker

# Remove old GUI container directory (ENTIRE TREE)
rm -rf gui-based-tools-docker.old/

# Remove legacy proxy scripts
rm -f core-assets/scripts/*-proxy.js

# Remove setup script backup
rm -f setup-workspace.sh.bak
```

### Aggressive Cleanup (Review First!)
```bash
# Remove potentially unused helper scripts
rm -f mcp-helper.sh
rm -f multi-agent.sh

# Clean workspace dev artifacts
rm -rf workspace/gpu_test/
rm -rf workspace/docs/architecture/  # Only if outdated
```

## Post-Cleanup Verification

After cleanup, verify the container still builds and runs:

```bash
# Rebuild container
docker-compose down
docker-compose up --build -d

# Verify services
docker exec multi-agent-container supervisorctl status

# Test VNC
# Connect to localhost:5901

# Test MCP tools
docker exec -u dev multi-agent-container python3 /app/core-assets/mcp-tools/imagemagick_mcp.py
```

## Size Impact Estimate

```bash
# Check size of legacy directories
du -sh gui-based-tools-docker.old/
# Expected: ~2-3GB (includes Blender tarball)

# Check size of backup files
find . -name "*.bak*" -o -name "*backup*" -exec du -ch {} + | tail -1
# Expected: ~100KB-1MB

# Total estimated cleanup: ~2-3GB
```

## Migration Complete Checklist

- [ ] All services running in unified container
- [ ] VNC desktop accessible with XFCE
- [ ] MCP tools verified (imagemagick, kicad, ngspice)
- [ ] External directory mounted at /workspace/ext
- [ ] Blender accessible via CLI
- [ ] Documentation updated
- [ ] Legacy files removed
- [ ] Container size optimized
