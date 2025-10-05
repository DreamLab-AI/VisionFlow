# VNC Remote Desktop Access

## Overview

The container includes a VNC server for accessing GUI-based AI tools (Blender, QGIS, Playwright browsers, etc.).

## Access Methods

### 1. VNC Client (Recommended)
**Port**: 5901
**Password**: `password`

```bash
# Using vncviewer
vncviewer localhost:5901

# Using RealVNC
# Connect to: localhost:5901
# Password: password
```

### 2. noVNC (Browser-based)
**URL**: `http://localhost:6901`
**Password**: `password`

Open in web browser, no VNC client needed.

## Desktop Environment

- **Display Manager**: XFCE4
- **Display**: `:1`
- **Resolution**: 1920x1080 (configurable)

## Available GUI Tools

| Tool | Launch Command | Purpose |
|------|---------------|---------|
| Blender | `blender` | 3D modeling and rendering |
| QGIS | `qgis` | Geographic Information Systems |
| Chromium | `chromium` | Playwright browser automation |
| Terminal | `xfce4-terminal` | Command line access |

## Common Issues

### Screensaver Lock
If locked, kill screensaver:
```bash
docker exec multi-agent-container bash -c "DISPLAY=:1 killall xfce4-screensaver"
```

### Black Screen
Restart XFCE:
```bash
docker exec multi-agent-container supervisorctl restart xfce
```

### Performance
Disable desktop effects:
```bash
docker exec multi-agent-container bash -c "
  DISPLAY=:1 xfconf-query -c xfwm4 -p /general/use_compositing -s false
"
```

See `/VNC-ACCESS.md` and `/VNC-POPUP-FIX.md` for more details.
