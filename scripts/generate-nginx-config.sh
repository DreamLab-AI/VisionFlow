#!/bin/bash
# Generate nginx configuration from environment variables

set -e

# Input args
TEMPLATE="${1:-/app/nginx.conf.template}"
ENV_FILE="${2:-/dev/null}"  # Ignored, kept for compatibility
OUTPUT="${3:-/etc/nginx/nginx.conf}"

echo "[nginx-config] Generating nginx configuration..."

# Check if template exists
if [ ! -f "$TEMPLATE" ]; then
    echo "[nginx-config] ERROR: Template not found: $TEMPLATE"
    exit 1
fi

# Set default values for all nginx variables if not already set
export NGINX_WORKER_PROCESSES="${NGINX_WORKER_PROCESSES:-auto}"
export NGINX_WORKER_CONNECTIONS="${NGINX_WORKER_CONNECTIONS:-1024}"
export NGINX_KEEPALIVE_TIMEOUT="${NGINX_KEEPALIVE_TIMEOUT:-65}"
export NGINX_CLIENT_MAX_BODY_SIZE="${NGINX_CLIENT_MAX_BODY_SIZE:-100M}"
export NGINX_PROXY_BUFFER_SIZE="${NGINX_PROXY_BUFFER_SIZE:-8k}"
export NGINX_PROXY_BUFFERS="${NGINX_PROXY_BUFFERS:-16 8k}"
export NGINX_PROXY_BUSY_BUFFERS_SIZE="${NGINX_PROXY_BUSY_BUFFERS_SIZE:-32k}"
export NGINX_GZIP="${NGINX_GZIP:-on}"
export NGINX_GZIP_COMP_LEVEL="${NGINX_GZIP_COMP_LEVEL:-6}"
export NGINX_ACCESS_LOG="${NGINX_ACCESS_LOG:-/var/log/nginx/access.log combined}"
export NGINX_ERROR_LOG="${NGINX_ERROR_LOG:-/var/log/nginx/error.log warn}"
export NGINX_CSP="${NGINX_CSP:-default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'}"
export NGINX_REAL_IP_FROM_CF="${NGINX_REAL_IP_FROM_CF:-\$remote_addr}"
export NGINX_FORWARDED_PROTO="${NGINX_FORWARDED_PROTO:-\$scheme}"

# Additional variables that may be missing
export NGINX_LOG_LEVEL="${NGINX_LOG_LEVEL:-warn}"
export NGINX_MULTI_ACCEPT="${NGINX_MULTI_ACCEPT:-on}"
export NGINX_MAX_BODY_SIZE="${NGINX_MAX_BODY_SIZE:-100M}"
export NGINX_KEEPALIVE_REQUESTS="${NGINX_KEEPALIVE_REQUESTS:-100}"
export BACKEND_PORT="${BACKEND_PORT:-4000}"
export FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export NGINX_PORT="${NGINX_PORT:-3001}"
export NGINX_SERVER_NAME="${NGINX_SERVER_NAME:-_}"
export NGINX_ROOT="${NGINX_ROOT:-/app/client/dist}"
export NGINX_FRAME_OPTIONS="${NGINX_FRAME_OPTIONS:-SAMEORIGIN}"
export NGINX_REFERRER_POLICY="${NGINX_REFERRER_POLICY:-same-origin}"
export NGINX_HSTS_PRELOAD="${NGINX_HSTS_PRELOAD:-}"
export NGINX_API_BUFFERING="${NGINX_API_BUFFERING:-off}"
export NGINX_WS_ACCESS_LOG="${NGINX_WS_ACCESS_LOG:-/var/log/nginx/websocket.log}"
export NGINX_WS_LOG_FORMAT="${NGINX_WS_LOG_FORMAT:-debug_format}"
export NGINX_WS_ERROR_LOG="${NGINX_WS_ERROR_LOG:-/var/log/nginx/websocket-error.log}"
export NGINX_WS_ERROR_LEVEL="${NGINX_WS_ERROR_LEVEL:-debug}"
export NGINX_STATIC_FALLBACK="${NGINX_STATIC_FALLBACK:-development_static}"
export NGINX_STATIC_EXPIRES="${NGINX_STATIC_EXPIRES:-30d}"
export NGINX_STATIC_CACHE_CONTROL="${NGINX_STATIC_CACHE_CONTROL:-public, immutable}"
export NGINX_HTML_FALLBACK="${NGINX_HTML_FALLBACK:-/index.html}"
export NGINX_ROOT_FALLBACK="${NGINX_ROOT_FALLBACK:-development_root}"
export NGINX_ROOT_EXPIRES="${NGINX_ROOT_EXPIRES:--1}"
export NGINX_ROOT_CACHE_CONTROL="${NGINX_ROOT_CACHE_CONTROL:-no-cache, no-store, must-revalidate}"

# Generate the config with envsubst
# Use specific variable list to avoid replacing nginx variables like $uri
VARS='$NGINX_LOG_LEVEL $NGINX_WORKER_PROCESSES $NGINX_WORKER_CONNECTIONS $NGINX_MULTI_ACCEPT
$NGINX_KEEPALIVE_TIMEOUT $NGINX_KEEPALIVE_REQUESTS $NGINX_CLIENT_MAX_BODY_SIZE $NGINX_MAX_BODY_SIZE
$NGINX_PROXY_BUFFER_SIZE $NGINX_PROXY_BUFFERS $NGINX_PROXY_BUSY_BUFFERS_SIZE 
$NGINX_GZIP $NGINX_GZIP_COMP_LEVEL $NGINX_ACCESS_LOG $NGINX_ERROR_LOG 
$NGINX_CSP $NGINX_REAL_IP_FROM_CF $NGINX_FORWARDED_PROTO
$BACKEND_PORT $FRONTEND_PORT $NGINX_PORT $NGINX_SERVER_NAME $NGINX_ROOT
$NGINX_FRAME_OPTIONS $NGINX_REFERRER_POLICY $NGINX_HSTS_PRELOAD
$NGINX_API_BUFFERING $NGINX_WS_ACCESS_LOG $NGINX_WS_LOG_FORMAT
$NGINX_WS_ERROR_LOG $NGINX_WS_ERROR_LEVEL
$NGINX_STATIC_FALLBACK $NGINX_STATIC_EXPIRES $NGINX_STATIC_CACHE_CONTROL
$NGINX_HTML_FALLBACK $NGINX_ROOT_FALLBACK $NGINX_ROOT_EXPIRES $NGINX_ROOT_CACHE_CONTROL'

envsubst "$VARS" < "$TEMPLATE" > "$OUTPUT"

echo "[nginx-config] Configuration generated successfully at $OUTPUT"

# Validate nginx config if nginx is available
if command -v nginx &> /dev/null; then
    echo "[nginx-config] Validating configuration..."
    if nginx -t -c "$OUTPUT" 2>&1; then
        echo "[nginx-config] Configuration is valid"
    else
        echo "[nginx-config] WARNING: Configuration validation failed, but file was generated"
        # Don't exit with error since the config was generated
    fi
else
    echo "[nginx-config] WARNING: nginx command not found, skipping validation"
fi

exit 0