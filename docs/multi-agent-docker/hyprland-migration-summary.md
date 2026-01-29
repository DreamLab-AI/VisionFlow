---
title: Hyprland Migration Summary
description: Successfully migrated the Turbo Flow Unified Container from XFCE4 + Xvnc to **Hyprland + wayvnc** with 4K (3840x2160) support, high-contrast fonts, and comprehensive desktop automation.
category: explanation
tags:
  - api
  - api
  - docker
  - testing
  - playwright
related-docs:
  - multi-agent-docker/ANTIGRAVITY.md
  - multi-agent-docker/SKILLS.md
  - multi-agent-docker/TERMINAL_GRID.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Docker installation
---

# Hyprland Migration Summary

## Overview

Successfully migrated the Turbo Flow Unified Container from XFCE4 + Xvnc to **Hyprland + wayvnc** with 4K (3840x2160) support, high-contrast fonts, and comprehensive desktop automation.

## Major Changes

### 1. Desktop Environment

**Before**:
- XFCE4 window manager
- Xvnc server @ 1920x1080
- No tiling, manual window management

**After**:
- **Hyprland** Wayland compositor
- **wayvnc** server @ 3840x2160 (4K)
- Automated terminal tiling
- GPU-accelerated rendering

### 2. Terminal Setup

**Before**:
- 8 tmux windows
- Manual terminal launching

**After**:
- **11 tmux windows** (3 new user shells)
- Automated Kitty terminal tiles
- Large fonts (20pt FiraCode Nerd Font)
- High contrast Dracula theme

### 3. User Shells

**Added 3 dedicated shells** (windows 8-10):
- **Window 8**: `gemini-user` shell (UID 1001)
- **Window 9**: `openai-user` shell (UID 1002)
- **Window 10**: `zai-user` shell (UID 1003)

Each pre-configured with user-specific environment and tools.

### 4. Application Autostart

Hyprland now auto-launches:
- **7 Kitty terminals** (tmux windows 0-3, 8-10)
- **Chromium** with remote debugging (port 9222)
- **Blender** (minimized, MCP server active)
- **QGIS** (minimized, MCP server active)

### 5. Claude Credentials

**Before**:
- Copied from `/mnt/host-claude` (read-only)
- Stored in volume `claude-config`

**After**:
- **Direct mount**: `${HOME}/.claude` → `/home/devuser/.claude` (rw)
- Single source of truth
- No copying, instant sync

### 6. MCP Configuration

**Updated paths**:
- Chrome DevTools MCP: Uses remote debugging port 9222
- All MCP servers: Correct node path (`/usr/local/bin/node`)
- Blender/QGIS: Enabled autostart

## Configuration Files Created

| File | Purpose |
|------|---------|
| `unified-config/hyprland.conf` | Hyprland window manager config (4K, tiling, autostart) |
| `unified-config/kitty.conf` | Kitty terminal config (20pt font, Dracula theme) |
| `unified-config/supervisord.unified.conf` | Updated with Hyprland, wayvnc, correct node paths |
| `unified-config/tmux-autostart.sh` | Extended with 3 user shell windows |
| `unified-config/entrypoint-unified.sh` | Simplified credential logic, updated MCP config |
| `docker-compose.unified.yml` | Direct `.claude` mount, removed `claude-config` volume |
| `Dockerfile.unified` | Hyprland packages, fonts, configs |

## Packages Added

### Desktop (Hyprland Stack)
```
hyprland wayvnc kitty
xdg-desktop-portal-hyprland
qt5-wayland qt6-wayland
wl-clipboard grim slurp
polkit dunst
```

### Fonts (High Contrast 4K)
```
ttf-firacode-nerd ttf-liberation ttf-dejavu
ttf-font-awesome ttf-hack
noto-fonts noto-fonts-emoji
```

### Upstream Tools (from turbo-flow-claude)
```
agentic-qe agentic-flow agentic-jujutsu
@playwright/mcp
```

## Supervisord Services

### Updated Services
| Service | Before | After |
|---------|--------|-------|
| Desktop | `xvnc` + `xfce4` | `hyprland` + `wayvnc` |
| Node path | `/usr/bin/node` | `/usr/local/bin/node` |
| Blender MCP | autostart=false | autostart=true |
| QGIS MCP | autostart=false | autostart=true |
| Gemini Flow | `/usr/bin/gemini-flow` | `/usr/local/bin/gemini-flow` |

### Service Priority (Startup Order)
1. **Priority 10**: dbus (system)
2. **Priority 15**: dbus-user
3. **Priority 50**: sshd
4. **Priority 100**: hyprland
5. **Priority 200**: wayvnc
6. **Priority 300**: management-api
7. **Priority 400**: code-server
8. **Priority 500**: claude-zai
9. **Priority 510-517**: MCP servers (50+ skills, 15+ with MCP servers)
10. **Priority 600**: gemini-flow
11. **Priority 900**: tmux-autostart

## Hyprland Configuration Highlights

### Resolution
```
monitor=,3840x2160@60,0x0,1
```

### Autostart Applications
```bash
# 7 Kitty terminals (tmux windows)
exec-once = kitty --class kitty-tmux -e tmux attach-session -t workspace:0
exec-once = kitty --class kitty-tmux -e tmux attach-session -t workspace:1
# ... (windows 2, 3, 8, 9, 10)

# Chromium with remote debugging
exec-once = chromium --remote-debugging-port=9222 --user-data-dir=/home/devuser/.config/chromium-mcp

# Blender (minimized)
exec-once = blender

# QGIS (minimized)
exec-once = qgis
```

### Window Rules
```
# Tmux terminals - tile across workspace
windowrule = tile,^(kitty-tmux)$

# Chromium - maximize on workspace 2
windowrulev2 = workspace 2,class:^(chromium)$
windowrulev2 = maximize,class:^(chromium)$

# Blender - workspace 3, minimized
windowrulev2 = workspace 3,class:^(blender)$
windowrulev2 = minimize,class:^(blender)$

# QGIS - workspace 3, minimized
windowrulev2 = workspace 3,class:^(qgis)$
windowrulev2 = minimize,class:^(qgis)$
```

## Kitty Terminal Configuration

```ini
font_family      FiraCode Nerd Font Mono
font_size        20.0
foreground       #f8f8f2
background       #1e1e2e
cursor_shape     block
scrollback_lines 50000
```

## Testing Procedure

### 1. Service Verification
```bash
docker exec agentic-workstation /opt/venv/bin/supervisorctl status
```

Expected: All services RUNNING except autostart=false

### 2. Process Check
```bash
docker exec agentic-workstation ps auxf | grep -E "(hyprland|wayvnc|chromium|blender|qgis)"
```

Expected: All processes running as devuser

### 3. Log Inspection
```bash
docker exec agentic-workstation tail -f /var/log/hyprland.log
docker exec agentic-workstation tail -f /var/log/wayvnc.log
```

Expected: No critical errors

### 4. VNC Connection
```bash
# Connect via VNC client
vnc://localhost:5901
# Password: turboflow
```

Expected: 4K desktop with tiled terminals

### 5. Claude Code Test
```bash
# Inside container (tmux window 0)
dsp  # alias for claude --dangerously-skip-permissions
```

Expected: Claude Code starts, credentials loaded from `/home/devuser/.claude`

### 6. MCP Server Test
```bash
# Check MCP servers
docker exec agentic-workstation /opt/venv/bin/supervisorctl status | grep mcp
```

Expected: All MCP servers RUNNING

### 7. Chrome DevTools MCP Test
```bash
# Inside Claude Code
# Use Chrome DevTools MCP to inspect Chromium
```

Expected: Remote debugging connection successful

## Troubleshooting

### Hyprland Won't Start
Check `/var/log/hyprland.log` for errors. Common issues:
- XDG_RUNTIME_DIR not writable
- dbus-user not running
- GPU permissions

### Wayvnc Black Screen
```bash
# Check wayvnc logs
docker exec agentic-workstation cat /var/log/wayvnc.log

# Ensure Hyprland is running first
docker exec agentic-workstation ps aux | grep Hyprland
```

### MCP Servers Failing
```bash
# Check node path
docker exec agentic-workstation which node
# Should output: /usr/local/bin/node

# Restart service
docker exec agentic-workstation /opt/venv/bin/supervisorctl restart web-summary-mcp
```

### Tmux Windows Missing
```bash
# Check tmux-autostart log
docker exec agentic-workstation cat /var/log/tmux-autostart.log

# Manually run
docker exec -u devuser agentic-workstation /home/devuser/.config/tmux-autostart.sh
```

## Performance Notes

### 4K Resolution
- Wayvnc capped at 30 FPS to reduce bandwidth
- Use `--max-fps=60` for smoother experience if LAN allows

### GPU Acceleration
- Hyprland uses NVIDIA GPU if available
- Fallback to software rendering if no GPU
- Check `nvidia-smi` inside container

### Memory Usage
- Hyprland: ~200MB
- wayvnc: ~100MB
- Chromium: ~500MB
- Blender (idle): ~150MB
- QGIS (idle): ~100MB

**Total increase**: ~1GB vs XFCE4

## Next Steps

1. **Verify all services** start cleanly
2. **Test VNC connection** at 4K
3. **Launch Claude Code** in tmux window 0
4. **Exercise MCP servers** (chrome-devtools, blender, qgis)
5. **Confirm claude-flow** integration works
6. **Document user workflow** for new Hyprland desktop

## Known Limitations

1. **Wayvnc resolution** must match Hyprland output
2. **Xwayland** required for Blender/QGIS (legacy X11 apps)
3. **Remote debugging** only one Chromium instance
4. **GPU passthrough** requires host NVIDIA drivers

---

## Related Documentation

- [Terminal Grid Configuration](TERMINAL_GRID.md)
- [Upstream Turbo-Flow-Claude Analysis](upstream-analysis.md)
- [Final Status - Turbo Flow Unified Container Upgrade](development-notes/SESSION_2025-11-15.md)
- [X-FluxAgent Integration Plan for ComfyUI MCP Skill](x-fluxagent-adaptation-plan.md)
- [GPU-Only Build Status Report](fixes/GPU_BUILD_STATUS.md)

## Future Enhancements

- **Waybar** status bar for system info
- **Rofi** application launcher
- **Swaylock** screen locking
- **Gammastep** night light
- **Multiple outputs** for multi-monitor setups
- **OBS** screen recording integration
