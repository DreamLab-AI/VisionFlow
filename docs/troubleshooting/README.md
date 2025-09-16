# Troubleshooting Guide

Common issues and solutions for VisionFlow deployment and operation.

## Quick Fixes

### Common Issues

#### GPU Not Detected
**Problem**: VisionFlow can't access GPU acceleration

**Solutions**:
1. Check CUDA installation: `nvidia-smi`
2. Verify CUDA version compatibility (11.8+)
3. Check environment variable: `VISIONFLOW_GPU_ENABLED=true`
4. Restart Docker with GPU support: `docker run --gpus all ...`

#### WebSocket Connection Failed
**Problem**: Frontend can't connect to backend WebSocket

**Solutions**:
1. Verify port is open: `netstat -tuln | grep 3001`
2. Check firewall settings
3. Ensure CORS is configured correctly
4. Test with: `wscat -c ws://localhost:3001/wss`

#### High Memory Usage
**Problem**: System consuming excessive memory

**Solutions**:
1. Reduce `max_nodes` in configuration
2. Adjust GPU memory pool: `gpu.memory_pool_size = "1GB"`
3. Enable memory monitoring: `VISIONFLOW_LOG_LEVEL=debug`
4. Check for memory leaks in logs

#### Agent Connection Issues
**Problem**: AI agents not connecting or responding

**Solutions**:
1. Verify MCP server is running: `curl localhost:9500/health`
2. Check TCP connection: `telnet localhost 9500`
3. Review agent logs: `docker logs claude-flow-container`
4. Restart MCP service

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, Windows 10/11 with WSL2
- **CPU**: 4 cores, 2.5 GHz
- **Memory**: 8GB RAM
- **GPU**: CUDA-compatible (optional but recommended)
- **Storage**: 10GB free space

### Recommended Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **Memory**: 16GB+ RAM
- **GPU**: RTX 3060 or better
- **Storage**: 50GB+ SSD

## Configuration Issues

### Environment Variables Not Loading
```bash
# Check current environment
env | grep VISIONFLOW

# Load from .env file
export $(cat .env | xargs)

# Verify specific variables
echo $VISIONFLOW_GPU_ENABLED
```

### Docker Issues
```bash
# Check container status
docker ps -a

# View container logs
docker logs visionflow-app

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Permission Issues
```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data

# Fix Docker socket permissions
sudo usermod -a -G docker $USER
```

## Performance Issues

### Slow Graph Rendering
1. **Reduce node count**: Limit to <10,000 nodes for smooth performance
2. **Enable GPU acceleration**: Set `VISIONFLOW_GPU_ENABLED=true`
3. **Optimize physics settings**:
   ```toml
   [physics]
   iterations_per_frame = 5
   enable_adaptive_timestep = true
   ```
4. **Use binary protocol**: Ensure WebSocket binary mode is enabled

### High CPU Usage
1. **Check actor system**: Review actor mailbox sizes
2. **Optimize WebSocket connections**: Limit concurrent connections
3. **Profile with tools**:
   ```bash
   # Install profiling tools
   cargo install flamegraph

   # Profile the application
   cargo flamegraph --bin visionflow
   ```

### Network Latency
1. **Enable compression**: `websocket.compression = true`
2. **Adjust update rates**: Lower physics FPS if needed
3. **Use local deployment**: Deploy closer to users
4. **Check network configuration**: Verify routing and DNS

## Database Issues

### Connection Errors
```bash
# Check database file permissions
ls -la data/visionflow.db

# Test database connection
sqlite3 data/visionflow.db ".tables"

# Repair corrupt database
sqlite3 data/visionflow.db ".recover"
```

### Migration Failures
```bash
# Check migration status
visionflow db status

# Force migration
visionflow db migrate --force

# Rollback if needed
visionflow db rollback
```

## Logging and Debugging

### Enable Debug Logging
```bash
# Set environment variable
export RUST_LOG=debug

# Or in configuration
log_level = "debug"
```

### Structured Logging
```bash
# View logs with jq for JSON formatting
docker logs visionflow-app | jq .

# Filter by component
docker logs visionflow-app | grep "GraphService"

# Follow logs in real-time
docker logs -f visionflow-app
```

### Performance Monitoring
```bash
# Monitor system resources
htop
nvidia-smi -l 1

# Monitor VisionFlow metrics
curl localhost:3001/api/metrics
```

## Network Debugging

### Port Issues
```bash
# Check if ports are open
nmap localhost -p 3001,9500

# Test specific port
nc -zv localhost 3001

# Check what's using a port
lsof -i :3001
```

### Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 3001
sudo ufw allow 9500

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=3001/tcp
sudo firewall-cmd --reload
```

## Recovery Procedures

### Backup and Restore
```bash
# Backup configuration
tar -czf config-backup.tar.gz config/ data/

# Backup database
sqlite3 data/visionflow.db ".backup backup.db"

# Restore from backup
tar -xzf config-backup.tar.gz
```

### Reset to Defaults
```bash
# Reset configuration
visionflow config reset

# Clear cache and temporary files
rm -rf data/cache/*
rm -rf data/temp/*

# Restart services
docker-compose restart
```

### Emergency Recovery
```bash
# Stop all services
docker-compose down

# Check system resources
df -h
free -h

# Clean up Docker
docker system prune -a

# Restart with minimal configuration
docker-compose up visionflow
```

## Getting Help

### Support Channels
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/visionflow/visionflow/issues)
- **Documentation**: [Complete documentation](../README.md)
- **Community Forum**: [Discussion and Q&A](https://github.com/visionflow/visionflow/discussions)

### Information to Include
When seeking help, include:
1. Operating system and version
2. VisionFlow version
3. Hardware specifications
4. Complete error messages
5. Relevant log files
6. Steps to reproduce the issue

### Diagnostic Commands
```bash
# System information
uname -a
nvidia-smi
docker --version

# VisionFlow status
visionflow status
visionflow config check

# Generate diagnostic report
visionflow diagnose > diagnostic-report.txt
```

---

[← Back to Documentation](../README.md)