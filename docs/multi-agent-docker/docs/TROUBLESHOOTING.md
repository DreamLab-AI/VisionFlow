# Troubleshooting Guide

---
**Version:** 1.0.0
**Last Updated:** 2025-10-12
**Status:** Active
**Category:** Operations
**Tags:** [troubleshooting, debugging, support]
---

## Overview

This guide covers common issues, diagnostic procedures, and solutions for the Agentic Flow Docker environment. Follow the structured approach to identify and resolve problems efficiently.

---

## Quick Diagnostic Checklist

Before diving into specific issues, run these quick checks:

```bash
# 1. Check service health
curl http://localhost:9090/health
curl http://localhost:9600/health

# 2. Verify containers are running
docker ps | grep -E 'agentic-flow|claude-zai'

# 3. Check Docker logs
docker logs agentic-flow-cachyos --tail 50
docker logs claude-zai-service --tail 50

# 4. Verify environment configuration
cat .env | grep -E 'API_KEY|PROVIDER'

# 5. Check resource usage
docker stats --no-stream

# 6. Check GPU (if applicable)
docker exec agentic-flow-cachyos nvidia-smi
```

---

## 1. Services Not Starting

### Symptom: Containers Fail to Start or Exit Immediately

**Check Container Status:**
```bash
docker ps -a | grep -E 'agentic-flow|claude-zai'
```

**View Container Logs:**
```bash
# Management API logs
docker logs agentic-flow-cachyos

# Claude-ZAI logs
docker logs claude-zai-service

# Follow logs in real-time
docker logs -f agentic-flow-cachyos
```

**Common Causes and Solutions:**

#### Port Already in Use
```bash
# Error: "address already in use"

# Check what's using the ports
sudo lsof -i :9090
sudo lsof -i :9600

# Solution 1: Stop conflicting service
sudo kill $(sudo lsof -t -i:9090)

# Solution 2: Change port in .env
MANAGEMENT_API_PORT=9091
```

#### Missing Environment Variables
```bash
# Error: "API key not configured"

# Verify .env file exists
ls -la .env

# Check if API keys are set
grep -E 'ANTHROPIC_API_KEY|GOOGLE_GEMINI_API_KEY' .env

# Solution: Configure at least one provider
cp .env.example .env
nano .env  # Add your API keys
```

#### Docker Daemon Not Running
```bash
# Error: "Cannot connect to Docker daemon"

# Check Docker service status
sudo systemctl status docker

# Solution: Start Docker service
sudo systemctl start docker

# Enable on boot
sudo systemctl enable docker
```

#### Insufficient Resources
```bash
# Error: "out of memory" or "resource exhausted"

# Check system resources
free -h
df -h

# Solution: Increase Docker resource limits
# Edit docker-compose.yml resources section
deploy:
  resources:
    limits:
      memory: 32G  # Reduce if needed
      cpus: '16'   # Reduce if needed
```

---

## 2. GPU Not Detected

### Symptom: GPU Acceleration Unavailable or CUDA Errors

**Check GPU Status:**
```bash
# From host
nvidia-smi

# From container
docker exec agentic-flow-cachyos nvidia-smi

# Check CUDA availability
docker exec agentic-flow-cachyos python -c "import torch; print(torch.cuda.is_available())"
```

**Common Causes and Solutions:**

#### NVIDIA Docker Runtime Not Installed
```bash
# Check if nvidia-docker2 is installed
dpkg -l | grep nvidia-docker

# Solution: Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### GPU Not Exposed to Container
```bash
# Error: "no CUDA-capable device is detected"

# Verify docker-compose.yml configuration
grep -A 10 'runtime:' docker-compose.yml

# Solution: Ensure runtime and devices are configured
runtime: nvidia
devices:
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
```

#### Driver Version Mismatch
```bash
# Check driver and CUDA version
nvidia-smi | grep "Driver Version"

# Verify container CUDA version
docker exec agentic-flow-cachyos nvcc --version

# Solution: Update NVIDIA drivers on host
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install --only-upgrade nvidia-driver-XXX

# Arch/CachyOS
sudo pacman -Syu nvidia nvidia-utils
```

#### GPU Memory Exhausted
```bash
# Check GPU memory usage
docker exec agentic-flow-cachyos nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Solution: Clear GPU memory or reduce batch sizes
# Restart container to free GPU memory
docker restart agentic-flow-cachyos
```

---

## 3. API Authentication Failures

### Symptom: 401 Unauthorized or 403 Forbidden Errors

**Test Authentication:**
```bash
# Test with API key
export API_KEY="your-management-api-key"
curl -H "Authorization: Bearer $API_KEY" http://localhost:9090/v1/status

# Test without auth (should fail)
curl http://localhost:9090/v1/status
```

**Common Causes and Solutions:**

#### Incorrect API Key
```bash
# Error: "Invalid API key"

# Check configured API key
docker exec agentic-flow-cachyos env | grep MANAGEMENT_API_KEY

# Solution: Update .env with correct key
MANAGEMENT_API_KEY=your-secure-key-here

# Restart services
docker restart agentic-flow-cachyos
```

#### Using Wrong Header Format
```bash
# Wrong formats:
curl -H "X-API-Key: key" http://localhost:9090/v1/status
curl -H "Api-Key: key" http://localhost:9090/v1/status

# Correct format:
curl -H "Authorization: Bearer your-key" http://localhost:9090/v1/status
```

#### Provider API Key Issues
```bash
# Error: "API key invalid" from provider

# Verify provider keys are set
docker exec agentic-flow-cachyos env | grep -E 'ANTHROPIC|GOOGLE|OPENAI|OPENROUTER'

# Test provider key directly
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-3-sonnet-20240229","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
```

---

## 4. Provider Connection Issues

### Symptom: Timeouts, Connection Refused, or Provider Errors

**Check Provider Status:**
```bash
# Check provider configuration
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq '.providers'

# Test individual provider
docker exec agentic-flow-cachyos curl -I https://api.anthropic.com
docker exec agentic-flow-cachyos curl -I https://api.openai.com
docker exec agentic-flow-cachyos curl -I https://generativelanguage.googleapis.com
```

**Common Causes and Solutions:**

#### Network Connectivity
```bash
# Error: "getaddrinfo ENOTFOUND" or "connect ETIMEDOUT"

# Check DNS resolution
docker exec agentic-flow-cachyos nslookup api.anthropic.com

# Check internet connectivity
docker exec agentic-flow-cachyos ping -c 3 8.8.8.8

# Solution: Check firewall rules
sudo ufw status
sudo ufw allow out 443/tcp
```

#### Rate Limiting
```bash
# Error: "429 Too Many Requests"

# Check rate limit headers in logs
docker logs agentic-flow-cachyos | grep "429"

# Solution: Implement backoff or switch provider
ROUTER_MODE=performance
FALLBACK_CHAIN=gemini,openai,claude,openrouter
```

#### Provider API Outage
```bash
# Check provider status pages:
# - Anthropic: https://status.anthropic.com
# - OpenAI: https://status.openai.com
# - Google: https://status.cloud.google.com

# Solution: Configure fallback providers
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=openai,claude,openrouter
```

#### SSL Certificate Issues
```bash
# Error: "unable to verify SSL certificate"

# Check certificate validity
docker exec agentic-flow-cachyos openssl s_client -connect api.anthropic.com:443

# Solution: Update CA certificates
docker exec agentic-flow-cachyos sh -c "apt-get update && apt-get install -y ca-certificates"
docker restart agentic-flow-cachyos
```

---

## 5. Task Failures

### Symptom: Tasks Fail, Timeout, or Hang

**Check Task Status:**
```bash
# List all tasks
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks | jq

# Get specific task details
TASK_ID="your-task-id"
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks/$TASK_ID | jq

# View task logs
docker exec agentic-flow-cachyos cat ~/logs/tasks/$TASK_ID.log
```

**Common Causes and Solutions:**

#### Task Timeout
```bash
# Error: "Task exceeded timeout"

# Check task duration
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks/$TASK_ID | jq '.duration'

# Solution: Increase timeout or optimize task
# Edit task creation request
{
  "agent": "coder",
  "task": "your task",
  "timeout": 300000  // 5 minutes in ms
}
```

#### Process Crashes
```bash
# Check process manager logs
docker logs agentic-flow-cachyos | grep "process-manager"

# Check for core dumps
docker exec agentic-flow-cachyos ls -la /tmp/core.*

# Solution: Check memory limits and restart
docker restart agentic-flow-cachyos
```

#### Invalid Task Parameters
```bash
# Error: "Invalid agent type" or "Missing required field"

# Verify task schema
curl http://localhost:9090/docs

# Solution: Use correct task format
{
  "agent": "coder",        // Valid agent type
  "task": "description",   // Required
  "provider": "gemini",    // Optional
  "timeout": 60000         // Optional
}
```

---

## 6. Performance Issues

### Symptom: Slow Response Times, High Latency, or Stuttering

**Check Performance Metrics:**
```bash
# System status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq '.system'

# Prometheus metrics
curl http://localhost:9090/metrics

# Real-time resource usage
docker stats agentic-flow-cachyos
```

**Common Causes and Solutions:**

#### High CPU Usage
```bash
# Check CPU load
docker exec agentic-flow-cachyos uptime

# Identify CPU-heavy processes
docker exec agentic-flow-cachyos top -bn1

# Solution: Reduce concurrent tasks
CLAUDE_WORKER_POOL_SIZE=2  # Reduce from 4
CLAUDE_MAX_QUEUE_SIZE=25   # Reduce from 50
```

#### Memory Pressure
```bash
# Check memory usage
docker exec agentic-flow-cachyos free -h

# Check for memory leaks
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}"

# Solution: Increase memory limit or restart
docker-compose down
# Edit docker-compose.yml memory limits
docker-compose up -d
```

#### Disk I/O Bottleneck
```bash
# Check disk usage and performance
docker exec agentic-flow-cachyos df -h
docker exec agentic-flow-cachyos iostat -x 1 5

# Solution: Use faster storage or clean up
docker exec agentic-flow-cachyos rm -rf ~/logs/tasks/*.old
docker system prune -a
```

#### Network Latency
```bash
# Test provider latency
docker exec agentic-flow-cachyos ping -c 5 api.anthropic.com

# Measure API response time
time curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status

# Solution: Optimize routing
ROUTER_MODE=performance  # Prefer fastest provider
```

---

## 7. Desktop/VNC Issues

### Symptom: VNC Not Accessible or Black Screen

**Check Desktop Service:**
```bash
# Verify desktop is enabled
grep ENABLE_DESKTOP .env

# Check if VNC is running
docker exec agentic-flow-cachyos ps aux | grep vnc

# Test VNC port
curl -v http://localhost:5901
curl -v http://localhost:6901
```

**Common Causes and Solutions:**

#### Desktop Not Enabled
```bash
# Solution: Enable in .env
ENABLE_DESKTOP=true

# Rebuild and restart
./start-agentic-flow.sh --build
```

#### VNC Server Not Started
```bash
# Check VNC logs
docker exec agentic-flow-cachyos cat /var/log/vnc.log

# Manually start VNC (if needed)
docker exec agentic-flow-cachyos vncserver :1 -geometry 1920x1080 -depth 24

# Solution: Restart container
docker restart agentic-flow-cachyos
```

#### Display Issues
```bash
# Check X11 display
docker exec agentic-flow-cachyos echo $DISPLAY

# Test X11 connectivity
docker exec agentic-flow-cachyos xdpyinfo

# Solution: Reset display
DISPLAY=:0
```

#### Browser Connection Issues
```bash
# Access noVNC web interface
open http://localhost:6901

# If not loading, check noVNC process
docker exec agentic-flow-cachyos ps aux | grep websockify

# Solution: Port forward correctly
# Ensure ports are mapped in docker-compose.yml
ports:
  - "5901:5901"
  - "6901:6901"
```

---

## 8. Database Locking Issues

### Symptom: SQLite Database Locked Errors

**Check Database Status:**
```bash
# List active database connections
docker exec agentic-flow-cachyos lsof | grep .db

# Check database file permissions
docker exec agentic-flow-cachyos ls -la ~/.agentic-flow/*.db
```

**Common Causes and Solutions:**

#### Concurrent Write Conflicts
```bash
# Error: "database is locked"

# Check for stale locks
docker exec agentic-flow-cachyos fuser ~/.agentic-flow/tasks.db

# Solution: Use task isolation (already implemented)
# Each task gets its own database in separate workspace
```

#### Database Corruption
```bash
# Check database integrity
docker exec agentic-flow-cachyos sqlite3 ~/.agentic-flow/tasks.db "PRAGMA integrity_check;"

# Solution: Restore from backup or recreate
docker exec agentic-flow-cachyos mv ~/.agentic-flow/tasks.db ~/.agentic-flow/tasks.db.bak
docker restart agentic-flow-cachyos
```

#### Long-Running Transactions
```bash
# Check for blocking transactions
docker exec agentic-flow-cachyos sqlite3 ~/.agentic-flow/tasks.db "SELECT * FROM pragma_database_list;"

# Solution: Implement timeout
# Add to database configuration
PRAGMA busy_timeout = 5000;
```

---

## 9. Memory/Resource Exhaustion

### Symptom: Out of Memory, Container Crashes, or System Slowdown

**Check Resource Usage:**
```bash
# Overall system resources
free -h
df -h

# Container resource usage
docker stats --no-stream

# Memory breakdown
docker exec agentic-flow-cachyos cat /proc/meminfo
```

**Common Causes and Solutions:**

#### Container Memory Limit Reached
```bash
# Check memory limit
docker inspect agentic-flow-cachyos | jq '.[0].HostConfig.Memory'

# Solution: Increase memory limit
# Edit docker-compose.yml
deploy:
  resources:
    limits:
      memory: 64G  # Increase as needed
    reservations:
      memory: 16G
```

#### Memory Leak in Node Process
```bash
# Monitor Node.js memory
docker exec agentic-flow-cachyos node -e "console.log(process.memoryUsage())"

# Check heap usage over time
docker exec agentic-flow-cachyos node --expose-gc -e "global.gc(); console.log(process.memoryUsage())"

# Solution: Restart services periodically
docker restart agentic-flow-cachyos
```

#### Large Model Cache
```bash
# Check model cache size
docker exec agentic-flow-cachyos du -sh ~/models

# Solution: Clear model cache
docker exec agentic-flow-cachyos rm -rf ~/models/*
docker volume rm model-cache
```

#### Too Many Concurrent Tasks
```bash
# Check active task count
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq '.tasks.active'

# Solution: Limit concurrent tasks
CLAUDE_WORKER_POOL_SIZE=2
CLAUDE_MAX_QUEUE_SIZE=10
```

---

## 10. Network Connectivity Issues

### Symptom: Cannot Access Services, Connection Timeouts

**Check Network Configuration:**
```bash
# List Docker networks
docker network ls

# Inspect agentic network
docker network inspect agentic-network

# Check container networking
docker exec agentic-flow-cachyos ip addr
docker exec agentic-flow-cachyos netstat -tlnp
```

**Common Causes and Solutions:**

#### Firewall Blocking Ports
```bash
# Check firewall status
sudo ufw status verbose

# Solution: Allow required ports
sudo ufw allow 9090/tcp  # Management API
sudo ufw allow 9600/tcp  # Claude-ZAI
sudo ufw allow 5901/tcp  # VNC
sudo ufw allow 6901/tcp  # noVNC
sudo ufw reload
```

#### Network Bridge Issues
```bash
# Recreate Docker bridge
docker network rm agentic-network
docker network create agentic-network

# Restart containers
./start-agentic-flow.sh --restart
```

#### DNS Resolution Failures
```bash
# Test DNS from container
docker exec agentic-flow-cachyos nslookup google.com

# Solution: Configure DNS
# Edit /etc/docker/daemon.json
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}

sudo systemctl restart docker
```

#### Container-to-Container Communication
```bash
# Test connectivity between containers
docker exec claude-zai-service ping -c 3 agentic-flow-cachyos

# Solution: Ensure both containers on same network
docker network connect agentic-network claude-zai-service
docker network connect agentic-network agentic-flow-cachyos
```

---

## 11. Log Investigation Guide

### Finding Logs

**Management API Logs:**
```bash
# Container stdout/stderr
docker logs agentic-flow-cachyos

# Management API structured logs
docker logs agentic-flow-cachyos | grep management-api

# Task-specific logs
docker exec agentic-flow-cachyos ls ~/logs/tasks/
docker exec agentic-flow-cachyos cat ~/logs/tasks/<task-id>.log
```

**Claude-ZAI Logs:**
```bash
# Service logs
docker logs claude-zai-service

# Follow logs in real-time
docker logs -f claude-zai-service --tail 100
```

**System Logs:**
```bash
# Docker daemon logs
journalctl -u docker.service -f

# Container resource events
docker events --filter container=agentic-flow-cachyos

# Kernel logs (for GPU issues)
dmesg | grep -i nvidia
```

### Log Levels

**Change Log Level:**
```bash
# Edit .env
LOG_LEVEL=debug  # Options: error, warn, info, debug

# Restart services
docker restart agentic-flow-cachyos
```

**Enable Debug Mode:**
```bash
# Full debug output
NODE_ENV=development
DEBUG=*

# Specific namespace
DEBUG=agentic-flow:*
```

### Log Analysis

**Filter by Error Severity:**
```bash
# Errors only
docker logs agentic-flow-cachyos | grep -i error

# Warnings and errors
docker logs agentic-flow-cachyos | grep -iE 'error|warn'

# Request failures
docker logs agentic-flow-cachyos | grep '"statusCode":5'
```

**Search for Specific Issues:**
```bash
# Connection errors
docker logs agentic-flow-cachyos | grep -i "ECONNREFUSED\|ETIMEDOUT\|ENOTFOUND"

# Memory issues
docker logs agentic-flow-cachyos | grep -i "out of memory\|heap\|allocation"

# GPU errors
docker logs agentic-flow-cachyos | grep -i "cuda\|gpu\|nvidia"
```

---

## 12. Debug Mode

### Enable Debug Mode

**Environment Variables:**
```bash
# Edit .env
NODE_ENV=development
LOG_LEVEL=debug
DEBUG=agentic-flow:*

# Restart services
./start-agentic-flow.sh --restart
```

**Runtime Debug:**
```bash
# Attach debugger to running process
docker exec agentic-flow-cachyos kill -USR1 $(pidof node)

# Enable Node.js inspector
docker exec agentic-flow-cachyos node --inspect=0.0.0.0:9229 /path/to/server.js
```

### Debug Endpoints

**Health Check Endpoints:**
```bash
# Basic health
curl http://localhost:9090/health

# Readiness probe
curl http://localhost:9090/ready

# Detailed status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq
```

**Metrics Endpoint:**
```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# Specific metric
curl http://localhost:9090/metrics | grep http_request
```

### Interactive Debugging

**Shell Access:**
```bash
# Open interactive shell
docker exec -it agentic-flow-cachyos zsh

# Run diagnostics interactively
docker exec -it agentic-flow-cachyos bash
```

**Node.js REPL:**
```bash
# Start Node.js REPL in container
docker exec -it agentic-flow-cachyos node

# Load modules and test
> const SystemMonitor = require('./management-api/utils/system-monitor');
> const monitor = new SystemMonitor(console);
> monitor.getStatus().then(console.log);
```

---

## 13. Common Error Messages and Solutions

### Management API Errors

**"EADDRINUSE: address already in use"**
```bash
# Port conflict
sudo lsof -i :9090 | grep LISTEN
sudo kill $(sudo lsof -t -i:9090)
```

**"ENOENT: no such file or directory"**
```bash
# Missing file or directory
docker exec agentic-flow-cachyos ls -la ~/logs/
mkdir -p ~/logs/tasks
```

**"Failed to authenticate"**
```bash
# Invalid API key
echo "Check MANAGEMENT_API_KEY in .env"
grep MANAGEMENT_API_KEY .env
```

### Provider Errors

**"API key invalid"**
```bash
# Verify provider API key
docker exec agentic-flow-cachyos env | grep ANTHROPIC_API_KEY
# Test key manually at provider's website
```

**"Rate limit exceeded"**
```bash
# Too many requests
# Wait or switch provider
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq '.providers'
```

**"Model not found"**
```bash
# Invalid model name
# Check available models in provider documentation
# Use correct model identifier
```

### Docker Errors

**"Cannot connect to Docker daemon"**
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

**"No space left on device"**
```bash
docker system prune -a
docker volume prune
df -h
```

**"Container unhealthy"**
```bash
docker ps --filter health=unhealthy
docker logs <container-id>
docker restart <container-id>
```

### GPU Errors

**"CUDA out of memory"**
```bash
docker restart agentic-flow-cachyos
# Or reduce batch size / concurrent tasks
```

**"no CUDA-capable device"**
```bash
nvidia-smi  # Check GPU visible on host
# Check Docker GPU runtime configuration
```

**"CUDA driver version insufficient"**
```bash
# Update NVIDIA drivers
sudo apt-get update
sudo apt-get install --only-upgrade nvidia-driver-XXX
```

---

## 14. When to File an Issue

### Before Filing an Issue

1. **Search Existing Issues:** Check if your problem is already reported
   - [GitHub Issues](https://github.com/ruvnet/agentic-flow/issues)

2. **Verify Setup:** Ensure you've followed the documentation
   - [Getting Started](../getting-started/README.md)
   - [Deployment Guide](../guides/deployment.md)

3. **Collect Information:**
   ```bash
   # System information
   uname -a
   docker --version
   docker-compose --version

   # Container status
   docker ps -a

   # Logs
   docker logs agentic-flow-cachyos > logs.txt
   docker logs claude-zai-service >> logs.txt

   # Configuration (remove sensitive data)
   cat .env | sed 's/=.*/=***/' > config.txt
   ```

### When to File

**File an issue if:**
- You've followed this troubleshooting guide without resolution
- You've found a bug or unexpected behavior
- Documentation is incorrect or missing
- Feature requests or enhancements

**Include in Your Issue:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Docker version, GPU)
- Relevant logs and error messages
- Configuration (sanitized)

### Issue Template

```markdown
**Description:**
Brief description of the issue

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Docker: [e.g., 24.0.7]
- Docker Compose: [e.g., 2.23.0]
- GPU: [e.g., NVIDIA RTX 3090]
- CUDA: [e.g., 12.2]

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happened

**Logs:**
```
Paste relevant logs here
```

**Configuration:**
```yaml
Paste sanitized configuration here
```

**Additional Context:**
Any other relevant information
```

---

## Emergency Procedures

### Complete System Reset

```bash
# Stop all services
./start-agentic-flow.sh --stop

# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi $(docker images -q 'agentic-flow*')

# Clean Docker system
docker system prune -a --volumes -f

# Rebuild from scratch
./start-agentic-flow.sh --build
```

### Backup Before Reset

```bash
# Backup volumes
docker run --rm -v workspace:/data -v $(pwd):/backup \
  alpine tar czf /backup/workspace-backup-$(date +%Y%m%d).tar.gz /data

docker run --rm -v agent-memory:/data -v $(pwd):/backup \
  alpine tar czf /backup/agent-memory-backup-$(date +%Y%m%d).tar.gz /data

# Backup configuration
cp .env .env.backup
cp docker-compose.yml docker-compose.yml.backup
```

### Quick Recovery

```bash
# Restore from backup
tar xzf workspace-backup-YYYYMMDD.tar.gz -C /tmp
docker run --rm -v workspace:/data -v /tmp:/backup \
  alpine sh -c "cd /data && tar xzf /backup/workspace-backup-YYYYMMDD.tar.gz --strip 1"

# Restore configuration
cp .env.backup .env

# Restart services
./start-agentic-flow.sh --restart
```

---

## Support Resources

### Documentation
- [Architecture Overview](../reference/architecture/README.md)
- [Getting Started Guide](../getting-started/README.md)
- [API Reference](API_REFERENCE.md)
- [Deployment Guide](../guides/deployment.md)

### Community
- [GitHub Issues](https://github.com/ruvnet/agentic-flow/issues)
- [Discussions](https://github.com/ruvnet/agentic-flow/discussions)
- [Discord Community](https://discord.gg/agentic-flow)

### Contact
- Report bugs via GitHub Issues
- Feature requests via GitHub Discussions
- Security issues: security@ruvnet.com

---

## Appendix: Diagnostic Commands Reference

### Container Management
```bash
docker ps                              # List running containers
docker ps -a                           # List all containers
docker logs <container>                # View logs
docker logs -f <container>             # Follow logs
docker exec -it <container> bash       # Shell access
docker inspect <container>             # Detailed info
docker stats                           # Resource usage
docker restart <container>             # Restart container
```

### Network Diagnostics
```bash
docker network ls                      # List networks
docker network inspect <network>       # Network details
curl -v http://localhost:9090/health   # Test endpoint
netstat -tlnp | grep 9090             # Check port listening
```

### GPU Diagnostics
```bash
nvidia-smi                             # GPU status
nvidia-smi -l 1                        # Monitor GPU
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
docker exec <container> nvidia-smi     # GPU in container
```

### System Diagnostics
```bash
free -h                                # Memory usage
df -h                                  # Disk usage
top                                    # Process monitor
iostat -x 1                            # I/O statistics
uptime                                 # Load average
```

---

**Last Updated:** 2025-10-12
**Next Review:** 2025-11-12
**Maintainer:** Agentic Flow Team
