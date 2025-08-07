#!/bin/bash

# Stop script for Agent Control Interface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f agent-control.pid ]; then
    PID=$(cat agent-control.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping Agent Control Interface (PID: $PID)..."
        kill -TERM $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "Force stopping..."
            kill -9 $PID
        fi
        rm agent-control.pid
        echo "Stopped."
    else
        echo "Process not running (stale PID file)"
        rm agent-control.pid
    fi
else
    echo "No PID file found. Checking for running processes..."
    PIDS=$(ps aux | grep "node src/index.js" | grep -v grep | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo "Found processes: $PIDS"
        kill -TERM $PIDS
        echo "Stopped."
    else
        echo "No running processes found."
    fi
fi