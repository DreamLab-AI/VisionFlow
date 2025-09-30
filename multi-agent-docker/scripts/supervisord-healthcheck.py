#!/usr/bin/env python3
"""
Supervisord Health Check Event Listener
Monitors service health and performs automatic recovery
"""

import sys
import os
import time
import json
import requests
from datetime import datetime

def write_stdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def write_stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

def log(msg):
    timestamp = datetime.now().isoformat()
    write_stderr(f"[{timestamp}] {msg}\n")

def check_mcp_tcp_health():
    """Check MCP TCP server health"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 9500))
        sock.close()
        return result == 0
    except Exception as e:
        log(f"MCP TCP health check failed: {e}")
        return False

def check_mcp_ws_health():
    """Check MCP WebSocket bridge health"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 3002))
        sock.close()
        return result == 0
    except Exception as e:
        log(f"MCP WS health check failed: {e}")
        return False

def check_health_endpoint():
    """Check HTTP health endpoint"""
    try:
        response = requests.get('http://localhost:9501/health', timeout=5)
        return response.status_code == 200
    except Exception as e:
        log(f"Health endpoint check failed: {e}")
        return False

def restart_service(service_name):
    """Restart a failed service"""
    log(f"Attempting to restart service: {service_name}")
    os.system(f"supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart {service_name}")

def main():
    log("Health check listener started")

    while True:
        # Wait for TICK_60 events from supervisord
        write_stdout('READY\n')
        line = sys.stdin.readline()

        # Process the event
        headers = dict([x.split(':') for x in line.split()])

        # Perform health checks
        health_status = {
            'mcp_tcp': check_mcp_tcp_health(),
            'mcp_ws': check_mcp_ws_health(),
            'health_endpoint': check_health_endpoint()
        }

        log(f"Health status: {json.dumps(health_status)}")

        # Restart services if unhealthy
        if not health_status['mcp_tcp']:
            restart_service('mcp-tcp-server')

        if not health_status['mcp_ws']:
            restart_service('mcp-ws-bridge')

        # Acknowledge the event
        write_stdout('RESULT 2\nOK')

if __name__ == '__main__':
    main()