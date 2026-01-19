# QGIS Python 3.14 Fix - Changelog

## Issue
QGIS 3.44.6 displayed "Couldn't load SIP module. Python support will be disabled."

**Root Cause**: QGIS Python bindings are in `/usr/lib/python3.14/site-packages/` but QGIS's internal Python path doesn't include `site-packages`, only:
- `/usr/share/qgis/python`
- `/usr/lib/python3.14`
- `/usr/lib/python3.14/lib-dynload`

## Solution

### 1. Docker Build (`Dockerfile.unified`)
```dockerfile
# Line ~53: Install PyQt5 from Arch repos
python-pyqt5 python-pyqt5-sip \

# Phase 16.5: Runtime fix for Python 3.14
RUN python3.14 -m pip install --break-system-packages PyQt5-sip PyQt5
```

### 2. Entrypoint (`entrypoint-unified.sh`)
```bash
# Line ~464: Export PYTHONPATH in devuser .zshrc
export PYTHONPATH=/usr/lib/python3.14/site-packages:\$PYTHONPATH
```

### 3. QGIS MCP Standalone Server
Created `/opt/qgis/qgis_mcp_standalone.py`:
- Runs independently of QGIS GUI
- Provides MCP API on port 9877
- Uses QGIS Python bindings directly

### 4. Wrapper Script
Created `/usr/local/bin/qgis-with-python`:
```bash
export PYTHONPATH=/usr/lib/python3.14/site-packages:$PYTHONPATH
exec /usr/bin/qgis "$@"
```

## Verification

```bash
# Test PyQt5 import
python3.14 -c "from PyQt5 import QtCore; print('PyQt5 OK')"

# Test QGIS bindings
python3.14 -c "import sys; sys.path.insert(0, '/usr/lib/python3.14/site-packages'); from qgis.core import Qgis; print(Qgis.version())"

# Test MCP server
python3.14 -c "import socket, json; s = socket.socket(); s.connect(('localhost', 9877)); s.send((json.dumps({'type': 'health_check'}) + '\n').encode()); print(json.loads(s.recv(4096).decode()))"
```

## Files Modified

1. `Dockerfile.unified` - Added PyQt5 install and standalone server
2. `unified-config/entrypoint-unified.sh` - Added PYTHONPATH export
3. `skills/qgis/SKILL.md` - Added troubleshooting section
4. `skills/qgis/README.md` - Created quick reference guide
5. `skills/qgis/qgis_mcp_standalone.py` - Added standalone MCP server

## Testing Results

✅ QGIS GUI starts without SIP errors
✅ Python console functional in QGIS
✅ MCP server operational on port 9877
✅ Coordinate transform: WGS84 → British National Grid working
✅ Distance calculation: Cumbria ↔ Liverpool = 112.23 km

## Next Steps

1. Rebuild container: `./build-unified.sh`
2. Test QGIS plugin loading
3. Implement remaining MCP operations (buffer, load_layer, etc.)
4. Add supervisord service for automatic MCP server start

---
**Date**: 2026-01-19
**Tested with**: QGIS 3.44.6, Python 3.14.2, Arch Linux (CachyOS)
