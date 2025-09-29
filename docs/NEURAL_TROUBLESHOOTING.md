# Neural Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps diagnose and resolve issues with the Neural-Enhanced Swarm Controller. From common configuration problems to complex distributed system failures, this guide provides systematic approaches to problem resolution.

## Quick Diagnostic Commands

### System Health Check

```bash
#!/bin/bash
# neural-health-check.sh

echo "=== Neural Swarm Health Check ==="
echo "Date: $(date)"
echo ""

# Check API availability
echo "1. API Health:"
if curl -s http://localhost:8080/health > /dev/null; then
    echo "   ✓ API is responding"
    curl -s http://localhost:8080/health | jq .
else
    echo "   ✗ API is not responding"
fi
echo ""

# Check GPU availability
echo "2. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
else
    echo "   ! NVIDIA GPU not detected"
fi
echo ""

# Check memory usage
echo "3. Memory Usage:"
free -h
echo ""

# Check disk space
echo "4. Disk Usage:"
df -h /
echo ""

# Check container status
echo "5. Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Check neural metrics
echo "6. Neural Metrics:"
if curl -s http://localhost:9090/metrics > /dev/null; then
    echo "   Collective Intelligence: $(curl -s http://localhost:9090/metrics | grep neural_collective_intelligence | tail -1 | awk '{print $2}')"
    echo "   Active Agents: $(curl -s http://localhost:9090/metrics | grep neural_active_agents | tail -1 | awk '{print $2}')"
    echo "   Active Tasks: $(curl -s http://localhost:9090/metrics | grep neural_active_tasks | tail -1 | awk '{print $2}')"
else
    echo "   ✗ Metrics endpoint not available"
fi
echo ""

# Check logs for errors
echo "7. Recent Errors:"
docker logs neural-controller 2>&1 | grep -i error | tail -5
echo ""

echo "Health check completed."
```

### Performance Diagnostics

```bash
# neural-perf-check.sh
#!/bin/bash

echo "=== Neural Performance Diagnostics ==="

# CPU utilization
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

# Memory pressure
echo "Memory Pressure:"
cat /proc/pressure/memory | grep avg10

# Network connections
echo "Network Connections:"
ss -tuln | grep -E ':(8080|8081|9090|6379)'

# GPU utilization
echo "GPU Utilization:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

# Swarm metrics
echo "Swarm Performance:"
curl -s http://localhost:8080/api/v1/neural/swarms | jq '.[] | {id, metrics: {collective_intelligence, task_throughput, adaptation_rate}}'
```

## Common Issues and Solutions

### Issue 1: API Not Responding

**Symptoms:**
- HTTP 500 errors
- Connection timeouts
- "Connection refused" errors

**Diagnosis:**
```bash
# Check if service is running
docker ps | grep neural-controller

# Check logs
docker logs neural-controller --tail 50

# Check port binding
netstat -tlnp | grep 8080

# Test local connectivity
curl -v http://localhost:8080/health
```

**Solutions:**

1. **Service Not Running:**
```bash
# Restart the service
docker-compose restart neural-controller

# Or restart entire stack
docker-compose down && docker-compose up -d
```

2. **Port Conflicts:**
```bash
# Check what's using the port
lsof -i :8080

# Change port in configuration
export NEURAL_API_PORT=8081
docker-compose up -d
```

3. **Configuration Issues:**
```bash
# Validate configuration
docker exec neural-controller neural-swarm validate-config

# Reset to default configuration
docker exec neural-controller cp /app/config/neural.default.toml /app/config/neural.toml
docker-compose restart neural-controller
```

### Issue 2: GPU Not Detected

**Symptoms:**
- GPU acceleration disabled warnings
- Slow neural processing
- "No CUDA devices found" errors

**Diagnosis:**
```bash
# Check GPU hardware
lspci | grep -i nvidia

# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check container GPU access
docker exec neural-controller nvidia-smi
```

**Solutions:**

1. **Driver Issues:**
```bash
# Reinstall NVIDIA drivers
sudo apt purge nvidia-*
sudo apt autoremove
sudo apt install nvidia-driver-525
sudo reboot
```

2. **Docker GPU Support:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

3. **Container Configuration:**
```yaml
# Add to docker-compose.yml
services:
  neural-controller:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Issue 3: High Memory Usage

**Symptoms:**
- System becomes unresponsive
- Out of memory errors
- Container restarts

**Diagnosis:**
```bash
# Check system memory
free -h

# Check container memory usage
docker stats neural-controller

# Check memory pressure
cat /proc/pressure/memory

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full docker exec neural-controller neural-swarm
```

**Solutions:**

1. **Increase Memory Limits:**
```bash
# Increase Docker memory limit
docker update --memory=8g neural-controller

# Or update docker-compose.yml
services:
  neural-controller:
    deploy:
      resources:
        limits:
          memory: 8G
```

2. **Memory Configuration:**
```toml
# In neural.toml
[memory]
max_memory_size = "4GB"
retention_days = 7  # Reduce retention
consolidation_interval = "30m"  # More frequent consolidation
```

3. **Garbage Collection:**
```bash
# Force memory cleanup
docker exec neural-controller neural-swarm memory cleanup

# Restart with memory optimization
export NEURAL_MEMORY_OPTIMIZATION=true
docker-compose restart neural-controller
```

### Issue 4: Low Collective Intelligence

**Symptoms:**
- Collective intelligence below 0.6
- Poor task performance
- Slow decision-making

**Diagnosis:**
```bash
# Check swarm metrics
curl http://localhost:8080/api/v1/neural/swarms/status | jq '.metrics'

# Check agent distribution
curl http://localhost:8080/api/v1/neural/agents | jq 'group_by(.cognitive_pattern) | map({pattern: .[0].cognitive_pattern, count: length})'

# Check topology connectivity
curl http://localhost:8080/api/v1/neural/topology/analysis
```

**Solutions:**

1. **Increase Cognitive Diversity:**
```bash
# Add diverse agents
curl -X POST http://localhost:8080/api/v1/neural/agents \
  -d '{
    "role": "researcher",
    "cognitive_pattern": "divergent",
    "capabilities": ["creative_thinking", "exploration"]
  }'

curl -X POST http://localhost:8080/api/v1/neural/agents \
  -d '{
    "role": "analyzer",
    "cognitive_pattern": "critical_analysis",
    "capabilities": ["evaluation", "validation"]
  }'
```

2. **Optimize Topology:**
```bash
# Switch to mesh topology for better connectivity
curl -X PUT http://localhost:8080/api/v1/neural/topology \
  -d '{
    "type": "mesh",
    "connectivity": 0.8,
    "redundancy": 3
  }'
```

3. **Enable Learning:**
```bash
# Increase learning rate
curl -X PUT http://localhost:8080/api/v1/neural/config \
  -d '{
    "learning_rate": 0.02,
    "neural_plasticity": 0.8
  }'
```

### Issue 5: Task Assignment Failures

**Symptoms:**
- Tasks stuck in "pending" state
- "No suitable agents" errors
- Uneven task distribution

**Diagnosis:**
```bash
# Check task queue
curl http://localhost:8080/api/v1/neural/tasks?status=pending

# Check agent availability
curl http://localhost:8080/api/v1/neural/agents | jq 'map(select(.workload < 0.8))'

# Check cognitive pattern distribution
curl http://localhost:8080/api/v1/neural/agents | jq 'group_by(.cognitive_pattern) | map({pattern: .[0].cognitive_pattern, count: length, avg_workload: (map(.workload) | add / length)})'
```

**Solutions:**

1. **Add Missing Cognitive Patterns:**
```bash
# Identify required patterns
curl http://localhost:8080/api/v1/neural/tasks?status=pending | jq '.[].cognitive_requirements[]' | sort | uniq -c

# Add agents with required patterns
curl -X POST http://localhost:8080/api/v1/neural/agents \
  -d '{
    "role": "specialist",
    "cognitive_pattern": "systems_thinking",
    "capabilities": ["architecture", "integration"]
  }'
```

2. **Adjust Task Constraints:**
```bash
# Relax task constraints
curl -X PUT http://localhost:8080/api/v1/neural/tasks/{task_id} \
  -d '{
    "neural_constraints": {
      "min_activation_level": 0.5,
      "max_cognitive_load": 0.9,
      "required_trust_score": 0.5
    }
  }'
```

3. **Rebalance Workload:**
```bash
# Trigger workload rebalancing
curl -X POST http://localhost:8080/api/v1/neural/operations/rebalance
```

## Performance Issues

### Slow Neural Processing

**Symptoms:**
- High response times (>1s)
- CPU usage consistently above 80%
- Task completion timeouts

**Diagnosis:**
```bash
# CPU profiling
perf top -p $(pgrep neural-controller)

# Memory profiling
valgrind --tool=massif docker exec neural-controller neural-swarm

# Network latency
ping -c 10 neural-peer-node

# GPU bottlenecks
nvidia-smi dmon -s u
```

**Solutions:**

1. **CPU Optimization:**
```bash
# Increase CPU allocation
docker update --cpus="4.0" neural-controller

# Enable CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

2. **GPU Optimization:**
```bash
# Increase GPU batch size
export NEURAL_GPU_BATCH_SIZE=64
docker-compose restart neural-controller

# Enable GPU memory optimization
export NEURAL_GPU_MEMORY_OPTIMIZATION=true
```

3. **Network Optimization:**
```bash
# Optimize network stack
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p
```

### Memory Leaks

**Symptoms:**
- Memory usage continuously increasing
- System becomes slower over time
- Out of memory crashes

**Diagnosis:**
```bash
# Memory leak detection
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all docker exec neural-controller neural-swarm

# Memory profiling over time
while true; do
  echo "$(date): $(docker stats neural-controller --no-stream --format '{{.MemUsage}}')"
  sleep 60
done
```

**Solutions:**

1. **Memory Limits:**
```bash
# Set strict memory limits
docker update --memory=4g --memory-swap=4g neural-controller
```

2. **Garbage Collection:**
```bash
# Enable aggressive GC
export NEURAL_GC_AGGRESSIVE=true
export NEURAL_MEMORY_CLEANUP_INTERVAL=300
```

3. **Memory Pool Configuration:**
```toml
# In neural.toml
[memory]
pool_size = "2GB"
cleanup_threshold = 0.8
compaction_interval = "10m"
```

## Network Issues

### Connection Timeouts

**Symptoms:**
- "Connection timed out" errors
- Intermittent API failures
- Agents disconnecting

**Diagnosis:**
```bash
# Network connectivity
telnet localhost 8080

# Check firewall
iptables -L
ufw status

# Check network congestion
ss -i

# Check DNS resolution
nslookup neural-peer-node
```

**Solutions:**

1. **Firewall Configuration:**
```bash
# Open required ports
sudo ufw allow 8080/tcp
sudo ufw allow 8081:8089/tcp
sudo ufw allow 9090/tcp
```

2. **Network Tuning:**
```bash
# Increase connection limits
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
sysctl -p
```

3. **Load Balancer Configuration:**
```yaml
# nginx.conf
upstream neural_backend {
    server neural-node-1:8080 max_fails=3 fail_timeout=30s;
    server neural-node-2:8080 max_fails=3 fail_timeout=30s;
    server neural-node-3:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://neural_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

### Distributed System Issues

#### Split Brain Scenarios

**Symptoms:**
- Multiple swarm leaders
- Inconsistent state across nodes
- Conflicting decisions

**Diagnosis:**
```bash
# Check cluster status
curl http://node1:8080/api/v1/neural/cluster/status
curl http://node2:8080/api/v1/neural/cluster/status
curl http://node3:8080/api/v1/neural/cluster/status

# Check consensus state
curl http://localhost:8080/api/v1/neural/consensus/status
```

**Solutions:**

1. **Force Leader Election:**
```bash
# Stop all nodes
docker-compose down

# Start with single node
docker-compose up -d neural-controller-1

# Wait for stabilization
sleep 30

# Start remaining nodes
docker-compose up -d neural-controller-2 neural-controller-3
```

2. **Reset Cluster State:**
```bash
# Clear cluster state
docker exec neural-controller-1 neural-swarm cluster reset
docker exec neural-controller-2 neural-swarm cluster reset
docker exec neural-controller-3 neural-swarm cluster reset

# Restart cluster
docker-compose restart
```

#### Data Inconsistency

**Symptoms:**
- Different agent counts on nodes
- Inconsistent memory state
- Conflicting task assignments

**Diagnosis:**
```bash
# Compare node states
for node in node1 node2 node3; do
  echo "=== $node ==="
  curl http://$node:8080/api/v1/neural/agents | jq 'length'
  curl http://$node:8080/api/v1/neural/tasks | jq 'length'
done
```

**Solutions:**

1. **Force Synchronization:**
```bash
# Trigger full sync
curl -X POST http://localhost:8080/api/v1/neural/cluster/sync
```

2. **Rebuild from Backup:**
```bash
# Restore from backup
./neural-restore.sh /backup/neural-swarm/20240115-120000.tar.gz
```

## Monitoring and Alerting

### Prometheus Alerts

```yaml
# neural-alerts.yml
groups:
- name: neural-swarm
  rules:
  - alert: NeuralAPIDown
    expr: up{job="neural-controller"} == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Neural API is down"
      description: "Neural controller API has been down for more than 30 seconds"

  - alert: LowCollectiveIntelligence
    expr: neural_collective_intelligence < 0.6
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low collective intelligence"
      description: "Collective intelligence is {{ $value }}, below threshold of 0.6"

  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes{name="neural-controller"} / container_spec_memory_limit_bytes > 0.9
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value | humanizePercentage }}"

  - alert: GPUUtilizationHigh
    expr: nvidia_gpu_utilization > 95
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU utilization is very high"
      description: "GPU utilization is {{ $value }}% for more than 5 minutes"

  - alert: TaskBacklogHigh
    expr: neural_pending_tasks > 10
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "High task backlog"
      description: "{{ $value }} tasks are pending assignment"
```

### Health Check Scripts

```bash
#!/bin/bash
# neural-monitor.sh

LOG_FILE="/var/log/neural-monitor.log"
ALERT_THRESHOLD_INTELLIGENCE=0.6
ALERT_THRESHOLD_MEMORY=90
ALERT_THRESHOLD_GPU=95

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a $LOG_FILE
}

check_api() {
    if ! curl -s http://localhost:8080/health > /dev/null; then
        log "CRITICAL: Neural API is not responding"
        return 1
    fi
    return 0
}

check_intelligence() {
    local intelligence=$(curl -s http://localhost:9090/metrics | grep neural_collective_intelligence | tail -1 | awk '{print $2}')
    if (( $(echo "$intelligence < $ALERT_THRESHOLD_INTELLIGENCE" | bc -l) )); then
        log "WARNING: Low collective intelligence: $intelligence"
        return 1
    fi
    return 0
}

check_memory() {
    local memory_percent=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$memory_percent" -gt "$ALERT_THRESHOLD_MEMORY" ]; then
        log "WARNING: High memory usage: ${memory_percent}%"
        return 1
    fi
    return 0
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [ "$gpu_util" -gt "$ALERT_THRESHOLD_GPU" ]; then
            log "WARNING: High GPU utilization: ${gpu_util}%"
            return 1
        fi
    fi
    return 0
}

# Main monitoring loop
while true; do
    check_api
    check_intelligence
    check_memory
    check_gpu
    
    sleep 60
done
```

## Debugging Tools

### Log Analysis

```bash
#!/bin/bash
# neural-log-analyzer.sh

LOG_FILE="/var/log/neural-swarm/neural.log"

echo "=== Neural Log Analysis ==="
echo "Log file: $LOG_FILE"
echo "Analysis time: $(date)"
echo ""

# Error summary
echo "Error Summary (last 24 hours):"
grep -i error $LOG_FILE | grep "$(date -d '1 day ago' '+%Y-%m-%d')\|$(date '+%Y-%m-%d')" | \
    awk '{print $4}' | sort | uniq -c | sort -nr
echo ""

# Warning summary
echo "Warning Summary (last 24 hours):"
grep -i warn $LOG_FILE | grep "$(date -d '1 day ago' '+%Y-%m-%d')\|$(date '+%Y-%m-%d')" | \
    awk '{print $4}' | sort | uniq -c | sort -nr
echo ""

# Performance issues
echo "Performance Issues:"
grep -i "slow\|timeout\|latency" $LOG_FILE | tail -10
echo ""

# Memory issues
echo "Memory Issues:"
grep -i "memory\|oom\|allocation" $LOG_FILE | tail -10
echo ""

# Network issues
echo "Network Issues:"
grep -i "connection\|network\|timeout" $LOG_FILE | tail -10
echo ""

# Recent critical events
echo "Recent Critical Events:"
grep -i "critical\|fatal\|panic" $LOG_FILE | tail -5
```

### Configuration Validator

```bash
#!/bin/bash
# validate-neural-config.sh

CONFIG_FILE="/etc/neural-swarm/neural.toml"

echo "=== Neural Configuration Validation ==="

# Check file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Validate TOML syntax
if ! toml-validator "$CONFIG_FILE"; then
    echo "ERROR: Invalid TOML syntax in configuration file"
    exit 1
fi

# Check required sections
required_sections=("neural" "topology" "memory" "api")
for section in "${required_sections[@]}"; do
    if ! grep -q "\[$section\]" "$CONFIG_FILE"; then
        echo "ERROR: Missing required section: [$section]"
        exit 1
    fi
done

# Validate memory configuration
memory_size=$(grep "max_memory_size" "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' "')
if [[ ! $memory_size =~ ^[0-9]+[GMK]?B?$ ]]; then
    echo "WARNING: Invalid memory size format: $memory_size"
fi

# Validate GPU configuration
if grep -q "gpu_acceleration = true" "$CONFIG_FILE"; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: GPU acceleration enabled but NVIDIA drivers not found"
    fi
fi

# Validate network ports
api_port=$(grep "port" "$CONFIG_FILE" | head -1 | cut -d'=' -f2 | tr -d ' ')
if [ "$api_port" -lt 1024 ] || [ "$api_port" -gt 65535 ]; then
    echo "ERROR: Invalid API port: $api_port"
fi

echo "Configuration validation completed"
```

### Performance Profiler

```bash
#!/bin/bash
# neural-profiler.sh

PROFILE_DURATION=60
OUTPUT_DIR="/tmp/neural-profile"

mkdir -p $OUTPUT_DIR

echo "Starting neural performance profiling for ${PROFILE_DURATION}s..."

# CPU profiling
perf record -p $(pgrep neural-controller) -o $OUTPUT_DIR/cpu.perf &
CPU_PID=$!

# Memory profiling
valgrind --tool=massif --massif-out-file=$OUTPUT_DIR/memory.massif \
    docker exec neural-controller neural-swarm profile-memory &
MEM_PID=$!

# Network profiling
tcpdump -i any -w $OUTPUT_DIR/network.pcap port 8080 &
NET_PID=$!

# GPU profiling
nvidia-smi dmon -s u -o T -f $OUTPUT_DIR/gpu.csv &
GPU_PID=$!

# Wait for profiling duration
sleep $PROFILE_DURATION

# Stop profiling
kill $CPU_PID $MEM_PID $NET_PID $GPU_PID 2>/dev/null

# Generate reports
echo "Generating profiling reports..."

# CPU report
perf report -i $OUTPUT_DIR/cpu.perf > $OUTPUT_DIR/cpu-report.txt

# Memory report
ms_print $OUTPUT_DIR/memory.massif > $OUTPUT_DIR/memory-report.txt

# Network statistics
echo "Network Statistics:" > $OUTPUT_DIR/network-report.txt
tshark -r $OUTPUT_DIR/network.pcap -q -z conv,tcp >> $OUTPUT_DIR/network-report.txt

echo "Profiling completed. Reports available in: $OUTPUT_DIR"
```

## Recovery Procedures

### Emergency Shutdown

```bash
#!/bin/bash
# neural-emergency-shutdown.sh

echo "=== NEURAL EMERGENCY SHUTDOWN ==="
echo "This will stop all neural services immediately."
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Shutdown cancelled"
    exit 0
fi

echo "Initiating emergency shutdown..."

# Stop accepting new requests
iptables -A INPUT -p tcp --dport 8080 -j REJECT

# Graceful shutdown of swarm
curl -X POST http://localhost:8080/api/v1/neural/shutdown

# Wait for graceful shutdown
sleep 10

# Force stop containers
docker-compose down --timeout 5

# Kill any remaining processes
pkill -f neural-controller

# Save state for recovery
mkdir -p /backup/emergency-$(date +%Y%m%d-%H%M%S)
docker cp neural-memory:/data /backup/emergency-$(date +%Y%m%d-%H%M%S)/

echo "Emergency shutdown completed"
```

### Disaster Recovery

```bash
#!/bin/bash
# neural-disaster-recovery.sh

BACKUP_FILE=$1
RECOVERY_MODE=${2:-"full"}

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file> [recovery_mode]"
    echo "Recovery modes: full, config-only, memory-only"
    exit 1
fi

echo "=== NEURAL DISASTER RECOVERY ==="
echo "Backup file: $BACKUP_FILE"
echo "Recovery mode: $RECOVERY_MODE"
echo ""

# Stop all services
echo "Stopping all neural services..."
docker-compose down

# Extract backup
echo "Extracting backup..."
RECOVERY_DIR="/tmp/neural-recovery"
mkdir -p $RECOVERY_DIR
tar -xzf $BACKUP_FILE -C $RECOVERY_DIR

case $RECOVERY_MODE in
    "full")
        echo "Performing full recovery..."
        
        # Restore configuration
        cp $RECOVERY_DIR/neural-config.yaml /etc/neural-swarm/
        
        # Restore memory
        docker volume rm neural_redis
        docker volume create neural_redis
        docker run --rm -v neural_redis:/data -v $RECOVERY_DIR:/backup \
            redis:7-alpine sh -c "cp /backup/neural-memory.rdb /data/dump.rdb"
        
        # Restore persistent volumes
        kubectl apply -f $RECOVERY_DIR/persistent-volumes.yaml
        kubectl apply -f $RECOVERY_DIR/persistent-volume-claims.yaml
        
        # Restore deployment
        kubectl apply -f $RECOVERY_DIR/deployment-snapshot.yaml
        ;;
        
    "config-only")
        echo "Restoring configuration only..."
        cp $RECOVERY_DIR/neural-config.yaml /etc/neural-swarm/
        ;;
        
    "memory-only")
        echo "Restoring memory only..."
        docker volume rm neural_redis
        docker volume create neural_redis
        docker run --rm -v neural_redis:/data -v $RECOVERY_DIR:/backup \
            redis:7-alpine sh -c "cp /backup/neural-memory.rdb /data/dump.rdb"
        ;;
esac

# Restart services
echo "Restarting neural services..."
docker-compose up -d

# Wait for services to stabilize
echo "Waiting for services to stabilize..."
sleep 30

# Verify recovery
echo "Verifying recovery..."
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✓ Recovery successful - API is responding"
else
    echo "✗ Recovery failed - API is not responding"
    exit 1
fi

# Clean up
rm -rf $RECOVERY_DIR

echo "Disaster recovery completed successfully"
```

This comprehensive troubleshooting guide provides systematic approaches to diagnosing and resolving issues in the Neural-Enhanced Swarm Controller, ensuring reliable operation in production environments.