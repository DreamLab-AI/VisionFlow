#!/bin/bash

# Continuous monitoring script for Agent Control Interface

echo "======================================"
echo "Agent Control Interface Monitor"
echo "======================================"
echo ""
echo "Monitoring server status and logs..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    echo "----------------------------------------"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Status Check"
    echo "----------------------------------------"
    
    # Check if process is running
    if ps aux | grep -v grep | grep -q "node src/index.js"; then
        PID=$(ps aux | grep -v grep | grep "node src/index.js" | awk '{print $2}')
        echo "✓ Server running (PID: $PID)"
        
        # Check memory and CPU
        PS_INFO=$(ps aux | grep -v grep | grep "node src/index.js" | awk '{print "  CPU: "$3"%, MEM: "$4"%"}')
        echo "$PS_INFO"
    else
        echo "✗ Server not running!"
    fi
    
    # Check port
    if netstat -tlnp 2>/dev/null | grep -q 9500 || ss -tlnp 2>/dev/null | grep -q 9500; then
        echo "✓ Port 9500 is listening"
    else
        echo "✗ Port 9500 not listening!"
    fi
    
    # Check for connections
    CONNECTIONS=$(netstat -tnp 2>/dev/null | grep :9500 | grep ESTABLISHED | wc -l || ss -tnp | grep :9500 | grep ESTAB | wc -l)
    if [ "$CONNECTIONS" -gt 0 ]; then
        echo "✓ Active connections: $CONNECTIONS"
        netstat -tnp 2>/dev/null | grep :9500 | grep ESTABLISHED || ss -tnp | grep :9500 | grep ESTAB
    else
        echo "  No active connections"
    fi
    
    # Show recent logs
    echo ""
    echo "Recent logs:"
    tail -n 5 logs/agent-control.log 2>/dev/null || echo "  No recent logs"
    
    echo ""
    sleep 10
done