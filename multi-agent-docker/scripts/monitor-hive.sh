#!/bin/bash
# Monitor Hive Mind swarms via docker exec

CONTAINER_NAME="${1:-multi-agent-container}"
INTERVAL="${2:-5}"

echo "🐝 Hive Mind Monitor"
echo "Container: $CONTAINER_NAME"
echo "Update interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════"
    echo "🐝 HIVE MIND SWARM MONITOR - $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # Status
    docker exec -u dev "$CONTAINER_NAME" bash -c 'claude-flow hive-mind status' 2>/dev/null || {
        echo "❌ Container not running or claude-flow unavailable"
        exit 1
    }

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "📊 PERFORMANCE METRICS"
    echo "───────────────────────────────────────────────────────────"
    docker exec -u dev "$CONTAINER_NAME" bash -c 'claude-flow hive-mind metrics' 2>/dev/null

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "💾 DATABASE STATUS"
    echo "───────────────────────────────────────────────────────────"
    docker exec "$CONTAINER_NAME" bash -c 'ls -lh /workspace/.swarm/*.db 2>/dev/null | tail -5'

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "🔌 TCP SERVER"
    echo "───────────────────────────────────────────────────────────"
    docker exec "$CONTAINER_NAME" bash -c 'ss -tlnp | grep 9500 || echo "Port 9500 not listening"'

    echo ""
    echo "Refreshing in ${INTERVAL}s... (Ctrl+C to stop)"
    sleep "$INTERVAL"
done
