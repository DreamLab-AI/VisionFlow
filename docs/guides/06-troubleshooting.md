# Troubleshooting Guide

*[← Back to Guides](index.md)*

This comprehensive guide helps you diagnose and resolve common issues with VisionFlow, covering installation problems, runtime errors, performance issues, and more.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Container and Docker Issues](#container-and-docker-issues)
4. [Network and Connectivity](#network-and-connectivity)
5. [Agent System Problems](#agent-system-problems)
6. [GPU and Performance Issues](#gpu-and-performance-issues)
7. [MCP Tool Failures](#mcp-tool-failures)
8. [Database and Storage](#database-and-storage)
9. [Frontend and UI Issues](#frontend-and-ui-issues)
10. [Debugging Techniques](#debugging-techniques)
11. [Log Analysis](#log-analysis)
12. [Recovery Procedures](#recovery-procedures)

## Quick Diagnostics

### System Health Check Script

```bash
#!/bin/bash
# health-check.sh - Comprehensive system health check

echo "=== VisionFlow System Health Check ==="
echo "Date: $(date)"
echo

# Check Docker
echo "1. Docker Status:"
docker --version
docker-compose --version
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

# Check services
echo "2. Service Health:"
services=("postgres" "redis" "server" "client" "multi-agent-container")
for service in "${services[@]}"; do
    if docker ps | grep -q $service; then
        echo "✓ $service: Running"
        # Check health endpoint if available
        if [[ $service == "server" ]]; then
            curl -s http://localhost:3001/health > /dev/null && echo "  └─ API: Healthy" || echo "  └─ API: Unhealthy"
        fi
    else
        echo "✗ $service: Not running"
    fi
done
echo

# Check resources
echo "3. Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
echo

# Check disk space
echo "4. Disk Space:"
df -h | grep -E "Filesystem|docker|workspace"
echo

# Check network
echo "5. Network Status:"
docker network ls
echo

# Check logs for errors
echo "6. Recent Errors:"
docker-compose logs --tail=10 2>&1 | grep -i "error\|failed\|exception" | tail -5
echo

echo "=== Health Check Complete ==="
```

### Quick Fix Commands

```bash
# Restart all services
docker-compose restart

# Reset specific service
docker-compose restart server

# Clear all data and restart
docker-compose down -v
docker-compose up -d

# Check resource usage
docker system df

# Free up space
docker system prune -a
```

## Installation Issues

### Docker Installation Problems

**Issue: Docker not found**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in

# macOS
brew install docker docker-compose
open /Applications/Docker.app

# Windows WSL2
wsl --update
# Install Docker Desktop for Windows
```

**Issue: Docker Compose version mismatch**
```bash
# Check version
docker-compose --version

# Update Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Or use Docker Compose V2
docker compose version  # Note: no hyphen
```

### Permission Issues

**Issue: Permission denied errors**
```bash
# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Fix volume permissions
sudo chown -R $USER:$USER ./data ./workspace

# Fix in container
docker exec -u root container_name chown -R dev:dev /workspace
```

### Dependency Problems

**Issue: Missing dependencies**
```bash
# Install system dependencies
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    python3-pip \
    nodejs \
    npm

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
cd client && npm install

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Container and Docker Issues

### Container Won't Start

**Issue: Container exits immediately**
```bash
# Check logs
docker logs container_name

# Debug with shell
docker run -it --entrypoint /bin/bash image_name

# Check exit codes
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.ExitCode}}"

# Common fixes:
# 1. Check Dockerfile CMD/ENTRYPOINT
# 2. Verify environment variables
# 3. Check file permissions
# 4. Ensure ports are available
```

### Build Failures

**Issue: Docker build fails**
```bash
# Build with no cache
docker-compose build --no-cache

# Build specific service
docker-compose build server

# Debug build
docker build -t debug . --progress=plain

# Common issues:
# - Network timeouts: use --network=host
# - Out of space: docker system prune
# - Cache issues: use --no-cache
```

### Memory Issues

**Issue: Out of memory errors**
```bash
# Check memory usage
docker stats

# Increase memory limits
docker-compose down
# Edit docker-compose.yml:
services:
  server:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

# Restart with new limits
docker-compose up -d

# For Docker Desktop, increase in settings
```

## Network and Connectivity

### Port Conflicts

**Issue: Port already in use**
```bash
# Find process using port
sudo lsof -i :3001
sudo netstat -tulpn | grep 3001

# Kill process
sudo kill -9 $(sudo lsof -t -i:3001)

# Or change port in .env
HOST_PORT=3002
```

### Container Communication

**Issue: Containers can't communicate**
```bash
# Check network
docker network ls
docker network inspect docker_ragflow

# Test connectivity
docker exec multi-agent-container ping postgres
docker exec multi-agent-container curl http://server:8080/health

# Recreate network
docker-compose down
docker network rm docker_ragflow
docker-compose up -d
```

### External Access Issues

**Issue: Can't access from browser**
```bash
# Check firewall
sudo ufw status
sudo ufw allow 3001/tcp

# Check Docker port mapping
docker ps --format "table {{.Names}}\t{{.Ports}}"

# Test locally
curl http://localhost:3001/health

# Check nginx/proxy configuration
docker exec nginx cat /etc/nginx/nginx.conf
```

## Agent System Problems

### Agents Not Spawning

**Issue: Failed to spawn agent**
```python
# Debug agent spawning
import logging
logging.basicConfig(level=logging.DEBUG)

# Check orchestrator status
curl http://localhost:3001/api/agents/orchestrator/status

# Manually spawn agent
curl -X POST http://localhost:3001/api/agents \
  -H "Content-Type: application/json" \
  -d '{"type": "researcher", "config": {}}'

# Common issues:
# - Resource limits reached
# - Invalid agent configuration
# - Task queue full
```

### Agent Communication Failures

**Issue: Agents can't communicate**
```bash
# Check message bus
docker exec multi-agent-container redis-cli ping

# Monitor messages
docker exec multi-agent-container redis-cli monitor

# Check agent logs
docker logs multi-agent-container | grep "agent_id"

# Test direct communication
echo '{"type": "test", "to": "agent_123"}' | \
  docker exec -i multi-agent-container redis-cli -x PUBLISH agent.messages
```

### Task Processing Issues

**Issue: Tasks stuck in queue**
```bash
# Check task queue
curl http://localhost:3001/api/tasks/queue/status

# List stuck tasks
curl http://localhost:3001/api/tasks?status=stuck

# Clear stuck tasks
curl -X POST http://localhost:3001/api/tasks/queue/clear-stuck

# Restart task processor
docker exec multi-agent-container supervisorctl restart task-processor
```

## GPU and Performance Issues

### GPU Not Detected

**Issue: NVIDIA GPU not available**
```bash
# Check GPU on host
nvidia-smi

# Check in container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Update docker-compose.yml
services:
  server:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Performance Degradation

**Issue: System running slowly**
```bash
# Profile system
docker exec server /app/scripts/profile.sh

# Check bottlenecks
htop
iotop
iftop

# Optimize database
docker exec postgres psql -U visionflow -c "VACUUM ANALYZE;"

# Clear caches
docker exec redis redis-cli FLUSHALL

# Restart with performance monitoring
ENABLE_PROFILING=true docker-compose up -d
```

### Memory Leaks

**Issue: Increasing memory usage**
```python
# Monitor memory usage
import psutil
import time

def monitor_memory():
    while True:
        process = psutil.Process()
        mem = process.memory_info()
        print(f"RSS: {mem.rss / 1024 / 1024:.2f} MB")
        time.sleep(60)

# Enable memory profiling
import tracemalloc
tracemalloc.start()

# Take snapshots
snapshot1 = tracemalloc.take_snapshot()
# ... run code ...
snapshot2 = tracemalloc.take_snapshot()

# Compare snapshots
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
```

## MCP Tool Failures

### Tool Not Found

**Issue: MCP tool not available**
```bash
# List available tools
docker exec multi-agent-container ./mcp-helper.sh list-tools

# Refresh tool configuration
docker exec multi-agent-container /app/setup-workspace.sh --force

# Check tool definition
docker exec multi-agent-container cat .mcp.json | jq '.tools'

# Test tool directly
echo '{"method": "test"}' | \
  docker exec -i multi-agent-container python3 mcp-tools/tool_name.py
```

### Bridge Tool Connection Issues

**Issue: Can't connect to external application**
```bash
# Check GUI tools container
docker ps | grep gui-tools

# Test TCP connection
docker exec multi-agent-container nc -zv gui-tools-service 9876

# Check application status
docker exec gui-tools-container ps aux | grep blender

# Restart services
docker-compose restart gui-tools-service

# Check logs
docker logs gui-tools-container | grep ERROR
```

### Tool Execution Errors

**Issue: Tool fails during execution**
```python
# Debug tool execution
#!/usr/bin/env python3
import sys
import json
import traceback

def debug_wrapper():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            # Log request
            with open('/tmp/tool_debug.log', 'a') as f:
                f.write(f"Request: {request}\n")
            
            # Process request
            result = process_request(request)
            print(json.dumps({'result': result}), flush=True)
            
        except Exception as e:
            # Log full traceback
            with open('/tmp/tool_debug.log', 'a') as f:
                f.write(f"Error: {traceback.format_exc()}\n")
            
            print(json.dumps({'error': str(e)}), flush=True)

if __name__ == '__main__':
    debug_wrapper()
```

## Database and Storage

### Database Connection Issues

**Issue: Can't connect to database**
```bash
# Check PostgreSQL status
docker exec postgres pg_isready

# Test connection
docker exec postgres psql -U visionflow -d visionflow -c "SELECT 1;"

# Check configuration
docker exec server cat /app/.env | grep DATABASE_URL

# Reset database
docker-compose down -v
docker-compose up -d postgres
docker-compose exec postgres createdb -U visionflow visionflow
docker-compose up -d
```

### Data Corruption

**Issue: Database corruption detected**
```bash
# Backup current data
docker exec postgres pg_dump -U visionflow visionflow > backup.sql

# Check database integrity
docker exec postgres psql -U visionflow -d visionflow -c "
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Repair tables
docker exec postgres psql -U visionflow -d visionflow -c "REINDEX DATABASE visionflow;"

# Restore from backup if needed
docker exec -i postgres psql -U visionflow visionflow < backup.sql
```

### Storage Full

**Issue: No space left on device**
```bash
# Check disk usage
df -h
du -sh /var/lib/docker/*

# Clean Docker resources
docker system prune -a --volumes

# Remove old logs
find /var/log -type f -name "*.log" -mtime +7 -delete

# Clean build cache
docker builder prune

# Move Docker root
sudo systemctl stop docker
sudo mv /var/lib/docker /new/path/docker
sudo ln -s /new/path/docker /var/lib/docker
sudo systemctl start docker
```

## Frontend and UI Issues

### Blank Page

**Issue: Frontend shows blank page**
```bash
# Check browser console for errors
# Open Developer Tools → Console

# Check API connectivity
curl http://localhost:3001/api/health

# Rebuild frontend
cd client
npm run build

# Clear browser cache
# Ctrl+Shift+R (Chrome/Firefox)
# Cmd+Shift+R (Safari)

# Check WebSocket connection
wscat -c ws://localhost:3001/ws
```

### Rendering Issues

**Issue: 3D visualization not working**
```javascript
// Debug Three.js issues
window.__THREE__ = THREE;
console.log('Three.js version:', THREE.REVISION);

// Check WebGL support
const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
if (!gl) {
    console.error('WebGL not supported');
}

// Monitor performance
const stats = new Stats();
document.body.appendChild(stats.dom);
function animate() {
    stats.begin();
    // render code
    stats.end();
    requestAnimationFrame(animate);
}
```

### State Management Issues

**Issue: UI state inconsistent**
```typescript
// Enable Redux DevTools
const store = createStore(
  rootReducer,
  window.__REDUX_DEVTOOLS_EXTENSION__ && window.__REDUX_DEVTOOLS_EXTENSION__()
);

// Debug state updates
store.subscribe(() => {
  console.log('State changed:', store.getState());
});

// Reset state
localStorage.clear();
sessionStorage.clear();
window.location.reload();
```

## Debugging Techniques

### Enable Debug Mode

```bash
# Set debug environment
cat >> .env << EOF
DEBUG_MODE=true
RUST_LOG=debug
NODE_ENV=development
REACT_APP_DEBUG=true
EOF

# Restart with debug mode
docker-compose down
docker-compose up

# Enable verbose logging
docker-compose logs -f --tail=100
```

### Remote Debugging

**Backend (Rust)**
```bash
# Build with debug symbols
docker-compose exec server cargo build

# Attach debugger
docker-compose exec server rust-gdb target/debug/visionflow

# Or use VS Code remote debugging
# .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [{
    "type": "lldb",
    "request": "attach",
    "name": "Attach to Docker",
    "pid": "${command:pickRemoteProcess}",
    "sourceLanguages": ["rust"]
  }]
}
```

**Frontend (React)**
```bash
# Start with source maps
cd client
GENERATE_SOURCEMAP=true npm start

# Use React DevTools
# Install browser extension

# Profile performance
# React DevTools Profiler tab
```

### Network Debugging

```bash
# Monitor all network traffic
docker exec server tcpdump -i any -w capture.pcap

# Analyze WebSocket traffic
docker exec server tshark -i any -f "port 3002" -Y websocket

# Check HTTP requests
docker exec server tcpdump -i any -A -s 0 'tcp port 80'

# Use mitmproxy for debugging
docker run -it -p 8080:8080 mitmproxy/mitmproxy
```

## Log Analysis

### Centralized Logging

```bash
# View all logs
docker-compose logs

# Follow specific service
docker-compose logs -f server

# Filter by time
docker-compose logs --since 1h

# Search for patterns
docker-compose logs | grep -i error | less

# Export logs
docker-compose logs > visionflow-logs-$(date +%Y%m%d).log
```

### Log Aggregation

```yaml
# docker-compose.logging.yml
services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/lib/docker/containers:/var/lib/docker/containers:ro

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Log Patterns

```bash
# Common error patterns to search for:
grep -E "ERROR|FATAL|PANIC|Exception|Failed|Unable|Cannot" logs.txt

# Memory issues
grep -E "OOM|OutOfMemory|heap|memory leak" logs.txt

# Network issues
grep -E "timeout|refused|unreachable|disconnect" logs.txt

# Database issues
grep -E "deadlock|constraint violation|connection pool" logs.txt
```

## Recovery Procedures

### Disaster Recovery

```bash
#!/bin/bash
# disaster-recovery.sh

echo "Starting disaster recovery..."

# 1. Stop all services
docker-compose down

# 2. Backup current state
mkdir -p backups/disaster-$(date +%Y%m%d)
docker run --rm -v visionflow_postgres_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/disaster-$(date +%Y%m%d)/postgres-data.tar.gz -C /data .

# 3. Reset to clean state
docker system prune -a --volumes -f

# 4. Restore from last known good backup
if [ -f "backups/last-known-good/postgres-data.tar.gz" ]; then
  docker run --rm -v visionflow_postgres_data:/data -v $(pwd)/backups:/backup \
    alpine tar xzf /backup/last-known-good/postgres-data.tar.gz -C /data
fi

# 5. Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# 6. Verify services
sleep 30
./health-check.sh

echo "Disaster recovery complete"
```

### Data Recovery

```sql
-- Recover deleted data from PostgreSQL
-- Enable row visibility
ALTER SYSTEM SET track_commit_timestamp = on;

-- Find recently deleted rows
SELECT *
FROM agents
WHERE xmin::text::bigint > (txid_current() - 1000);

-- Restore from transaction log
SELECT pg_create_restore_point('before_delete');

-- Use pg_rewind if replication is set up
pg_rewind --target-pgdata=/var/lib/postgresql/data \
          --source-server="host=replica port=5432"
```

### State Recovery

```python
# Recover agent state
import pickle
import json
from datetime import datetime

class StateRecovery:
    def __init__(self):
        self.backup_dir = "/workspace/state-backups"
    
    def save_state(self, agent_id: str, state: dict):
        """Save agent state periodically."""
        timestamp = datetime.utcnow().isoformat()
        filename = f"{self.backup_dir}/{agent_id}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'agent_id': agent_id,
                'timestamp': timestamp,
                'state': state
            }, f)
    
    def recover_state(self, agent_id: str):
        """Recover last known good state."""
        import glob
        
        # Find latest state file
        pattern = f"{self.backup_dir}/{agent_id}_*.json"
        files = sorted(glob.glob(pattern))
        
        if files:
            with open(files[-1], 'r') as f:
                return json.load(f)
        
        return None
```

## Common Error Messages

### Error Reference Table

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `ECONNREFUSED` | Service not running | Start the service: `docker-compose up -d` |
| `EADDRINUSE` | Port already in use | Change port or kill process |
| `OOMKilled` | Out of memory | Increase memory limits |
| `No space left on device` | Disk full | Clean up with `docker system prune` |
| `Permission denied` | File permissions | Fix with `chown` or `chmod` |
| `Module not found` | Missing dependency | Install with `npm install` or `pip install` |
| `Connection timeout` | Network issue | Check firewall and connectivity |
| `CORS error` | Cross-origin issue | Configure CORS in backend |

## Preventive Maintenance

### Regular Maintenance Tasks

```bash
#!/bin/bash
# maintenance.sh - Weekly maintenance script

echo "Starting weekly maintenance..."

# 1. Backup data
./backup.sh

# 2. Clean up old logs
find /var/log -name "*.log" -mtime +30 -delete

# 3. Optimize database
docker exec postgres vacuumdb -U visionflow -d visionflow -z

# 4. Update dependencies
cd client && npm update
cd ../server && cargo update

# 5. Clean Docker resources
docker system prune -f

# 6. Check for security updates
docker exec server cargo audit

echo "Maintenance complete"
```

### Monitoring Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'visionflow'
    static_configs:
      - targets: ['server:9090']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

## Getting Further Help

### Resources

1. **Documentation**
   - [API Reference](../reference/README.md)
   - [Architecture Guide](../architecture/README.md)
   - [GitHub Issues](https://github.com/your-org/visionflow/issues)

2. **Community**
   - Discord: [Join Server](https://discord.gg/visionflow)
   - Forum: [community.visionflow.dev](https://community.visionflow.dev)
   - Stack Overflow: Tag `visionflow`

3. **Support**
   - Email: support@visionflow.dev
   - Emergency: security@visionflow.dev

### Reporting Issues

When reporting issues, include:
1. System information: `./health-check.sh > system-info.txt`
2. Logs: `docker-compose logs > logs.txt`
3. Steps to reproduce
4. Expected vs actual behavior
5. Any error messages

```bash
# Generate debug report
./scripts/generate-debug-report.sh
# Creates debug-report-TIMESTAMP.tar.gz
```

---

*[← Extending the System](05-extending-the-system.md) | [Back to Guides](index.md)*