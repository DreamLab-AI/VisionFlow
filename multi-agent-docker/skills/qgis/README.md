# QGIS Skill - Quick Reference

## Current Status (2026-01-19)

✅ **QGIS 3.44.6-Solothurn** - Fully operational
✅ **Python 3.14 Support** - PyQt5 configured
✅ **MCP Server** - Running on port 9877
✅ **Display** - VNC :1
✅ **Qgis2threejs Plugin** - 3D visualization installed (v2.8)

## Quick Start

```bash
# Check QGIS MCP server
python3.14 -c "import socket, json
s = socket.socket()
s.connect(('localhost', 9877))
s.send((json.dumps({'type': 'health_check'}) + '\n').encode())
print(json.loads(s.recv(4096).decode()))"

# Launch QGIS GUI
qgis-with-python &

# View on VNC (port 5901)
vncviewer localhost:5901
```

## Architecture

```
Claude Code
    ↓ (MCP stdin/stdout)
FastMCP Server (mcp-server/server.py)
    ↓ (TCP socket)
QGIS MCP Standalone (port 9877)
    ↓ (QGIS Python API)
QGIS 3.44.6 Desktop (Display :1)
```

## Available Operations

| Command | Description |
|---------|-------------|
| `health_check` | Server status |
| `calculate_distance` | Distance between points (meters/km) |
| `transform_coordinates` | CRS transformation (EPSG codes) |
| `list_layers` | List loaded layers |
| `buffer_analysis` | Create buffer zones (planned) |
| `load_layer` | Load geospatial data (planned) |

## Python 3.14 Fix

The container configures `PYTHONPATH=/usr/lib/python3.14/site-packages` automatically.

If you encounter SIP errors:
```bash
# Verify PyQt5
python3.14 -c "from PyQt5 import QtCore; print('OK')"

# Wrapper script (already configured)
export PYTHONPATH=/usr/lib/python3.14/site-packages:$PYTHONPATH
qgis --nologo
```

## Docker Integration

### Dockerfile.unified Changes

```dockerfile
# Install PyQt5 for Python 3.14 (line ~53)
python-pyqt5 python-pyqt5-sip \
```

### entrypoint-unified.sh Changes

```bash
# Export PYTHONPATH in .zshrc (line ~464)
export PYTHONPATH=/usr/lib/python3.14/site-packages:\$PYTHONPATH
```

## Files Modified

- `/home/devuser/workspace/project/multi-agent-docker/Dockerfile.unified`
- `/home/devuser/workspace/project/multi-agent-docker/unified-config/entrypoint-unified.sh`
- `/home/devuser/workspace/project/multi-agent-docker/skills/qgis/SKILL.md`
- `/tmp/qgis_mcp_standalone.py` (runtime)

## 3D Visualization (Qgis2threejs)

**Plugin**: Qgis2threejs v2.8
**Location**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/qgis2threejs/`

### Features
- 3D terrain from DEM data
- 3D vector visualization (points, lines, polygons)
- Web export (three.js format)
- glTF export for 3D printing/AR/VR
- Camera animations and narratives
- Lighting and fog effects

### Access
```bash
# Launch QGIS GUI
DISPLAY=:1 qgis-with-python

# Access via VNC on port 5901
# Menu: Web → Qgis2threejs Exporter
```

## Next Build

```bash
cd /home/devuser/workspace/project/multi-agent-docker
./build-unified.sh
```

Changes will be included in next container build.
