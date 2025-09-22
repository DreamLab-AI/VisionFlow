#!/bin/bash
set -e  # Exit on any error

echo "--- Setting up environment ---"
export DISPLAY=:1
export MCP_HOST=${MCP_HOST:-0.0.0.0}
echo "DISPLAY=$DISPLAY"
echo "MCP_HOST=$MCP_HOST"

echo "--- Starting virtual framebuffer ---"
# Clean up any stale lock files
rm -f /tmp/.X1-lock
rm -f /tmp/.X11-unix/X1

Xvfb :1 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 3

echo "--- Checking if Xvfb started ---"
if ps -p $XVFB_PID > /dev/null; then
    echo "Xvfb started successfully (PID: $XVFB_PID)"
else
    echo "ERROR: Xvfb failed to start"
    exit 1
fi

echo "--- Starting XFCE desktop ---"
startxfce4 &
sleep 3

echo "--- Installing Blender MCP addon ---"
# Get Blender version and create appropriate addon directory
BLENDER_VERSION=$(/opt/blender-4.5/blender --version | head -n1 | grep -oP '(?<=Blender )\d+\.\d+')
ADDON_DIR="/home/blender/.config/blender/${BLENDER_VERSION}/scripts/addons"
mkdir -p "$ADDON_DIR"
cp /home/blender/addon.py "$ADDON_DIR/addon.py"
echo "Blender MCP addon installed for version $BLENDER_VERSION"

echo "--- Starting Blender ---"
/opt/blender-4.5/blender --python /home/blender/autostart.py &
sleep 3

echo "--- Starting QGIS ---"
qgis &
QGIS_PID=$!
sleep 3

echo "--- Starting QGIS MCP Server ---"
node /opt/qgis-mcp-server.js &
QGIS_MCP_PID=$!
sleep 2

echo "--- Starting PBR Generator MCP Server ---"
# Use simple PBR server until the full PBR generator dependencies are fixed
python3 /opt/pbr-mcp-simple.py &
PBR_PID=$!
sleep 2

echo "--- Starting Playwright MCP Server ---"
export PLAYWRIGHT_MCP_PORT=${PLAYWRIGHT_MCP_PORT:-9879}
export PLAYWRIGHT_MCP_HOST=${PLAYWRIGHT_MCP_HOST:-0.0.0.0}
export PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
node /opt/playwright-mcp/server.js &
PLAYWRIGHT_PID=$!
sleep 3

echo "--- Checking if Playwright MCP started ---"
if ps -p $PLAYWRIGHT_PID > /dev/null; then
    echo "Playwright MCP started successfully (PID: $PLAYWRIGHT_PID)"
else
    echo "ERROR: Playwright MCP failed to start"
fi

echo "--- Starting x11vnc ---"
x11vnc -display :1 -nopw -forever -xkb -listen 0.0.0.0 -rfbport 5901 -verbose &
VNC_PID=$!
sleep 5

echo "--- Checking if x11vnc started ---"
if ps -p $VNC_PID > /dev/null; then
    echo "x11vnc started successfully (PID: $VNC_PID)"
else
    echo "ERROR: x11vnc failed to start"
    exit 1
fi

echo "--- Checking all processes ---"
ps aux | grep -E "(Xvfb|startxfce4|blender|qgis|x11vnc|playwright)" | grep -v grep

echo "--- Checking network listeners ---"
netstat -tuln

echo "--- Startup complete, container is running ---"
sleep infinity