#!/bin/bash
# Launch Chromium with configured startup URL and resource limits

# Wait for XFCE to fully start
sleep 5

# Get startup URL from environment or use default
STARTUP_URL="${CHROMIUM_STARTUP_URL:-about:blank}"

# Launch Chromium with resource limits and GPU acceleration disabled
if [ -n "$STARTUP_URL" ] && [ "$STARTUP_URL" != "" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Chromium with URL: $STARTUP_URL"

    # Block NVIDIA libraries to prevent driver mismatch crashes
    export LD_PRELOAD=""
    unset NVIDIA_VISIBLE_DEVICES
    unset NVIDIA_DRIVER_CAPABILITIES

    # Enable software WebGL rendering via SwiftShader
    chromium \
        --no-sandbox \
        --disable-dev-shm-usage \
        --disable-gpu \
        --use-gl=swiftshader-webgl \
        --enable-webgl-software-rendering \
        --enable-unsafe-swiftshader \
        --use-angle=swiftshader \
        --disable-accelerated-2d-canvas \
        --disable-accelerated-video-decode \
        --disable-gpu-compositing \
        --in-process-gpu \
        --disable-features=VizDisplayCompositor \
        --enable-features=UseSkiaRenderer \
        --no-first-run \
        --no-default-browser-check \
        --user-data-dir=/home/dev/.config/chromium \
        --disk-cache-size=52428800 \
        --media-cache-size=52428800 \
        "$STARTUP_URL" &
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No startup URL configured, skipping Chromium launch"
fi
