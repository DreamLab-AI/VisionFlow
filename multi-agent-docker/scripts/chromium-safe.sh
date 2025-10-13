#!/bin/bash
# Chromium wrapper for VNC/Docker environment
# Disables sandbox and GPU for container compatibility

# Use the real chromium binary if it exists, otherwise use the standard path
CHROMIUM_BIN="/usr/bin/chromium.real"
if [ ! -f "$CHROMIUM_BIN" ]; then
    CHROMIUM_BIN="/usr/bin/chromium"
fi

exec "$CHROMIUM_BIN" \
  --no-sandbox \
  --disable-dev-shm-usage \
  --enable-gpu-rasterization \
  --enable-webgl \
  --enable-webgl2 \
  --enable-accelerated-2d-canvas \
  --ignore-gpu-blocklist \
  --use-gl=angle \
  --use-angle=gl \
  --enable-features=VaapiVideoDecoder,VaapiVideoEncoder \
  --disable-features=UseChromeOSDirectVideoDecoder \
  "$@"
