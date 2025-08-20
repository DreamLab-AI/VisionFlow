# Troubleshooting Guide

## Common Issues and Solutions

### Connection Issues

#### Problem: Connection Refused
```
Error: TCP connection failed: Connection refused (os error 111)
```

**Causes:**
- TCP server not running
- Wrong port number
- Firewall blocking connection
- Container not on same network

**Solutions:**

1. **Check for deadlock recovery issues:**
```bash
# If graph appears frozen with nodes at boundary
docker logs visionflow-container | grep "deadlock"
# Expected: "Deadlock detected: 177 nodes at boundary"
```

2. **Verify TCP server is running:**
```bash
docker exec multi-agent-container /app/status-mcp-tcp.sh
```

2. **Start TCP server if needed:**
```bash
docker exec multi-agent-container /app/start-mcp-tcp.sh
```

3. **Check port binding:**
```bash
docker exec multi-agent-container netstat -tuln | grep 9500
```

4. **Verify network connectivity:**
```bash
docker network inspect docker_ragflow
docker exec visionflow ping multi-agent-container
```

---

#### Problem: Connection Timeout
```
Error: TCP connection timeout after 30s
```

**Causes:**
- Network latency
- Server overloaded
- Deadlock in graph physics engine

#### Problem: Graph Visualization Frozen
```
Issue: All nodes stuck at boundary position (980 units)
```

**Symptoms:**
- All 177 nodes positioned at viewport edge
- No movement despite physics enabled
- Zero kinetic energy in system

**Automatic Recovery:**
The system includes automatic deadlock recovery with:
- Enhanced detection (kinetic energy threshold: 0.001)
- Aggressive recovery parameters (8x stronger repulsion)
- Symmetry breaking via random perturbation
- Expanded viewport boundaries (1500 units)

**Manual Recovery:**
```bash
# Force physics parameter reset
curl -X POST http://localhost:4000/api/graph/physics/reset
```
- Incorrect host configuration

**Solutions:**

1. **Increase timeout:**
```bash
export MCP_CONNECTION_TIMEOUT=60000  # 60 seconds
```

2. **Check network latency:**
```bash
docker exec visionflow ping -c 10 multi-agent-container
```

3. **Verify DNS resolution:**
```bash
docker exec visionflow nslookup multi-agent-container
```

---

#### Problem: Broken Pipe
```
Error: TCP connection closed by server
```

**Causes:**
- Server crashed
- Connection idle timeout
- Network interruption

**Solutions:**

1. **Check server logs:**
```bash
docker logs multi-agent-container | tail -100
docker exec multi-agent-container tail -f /app/mcp-logs/tcp-server.log
```

2. **Enable automatic reconnection:**
```rust
let client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .with_retry(5, Duration::from_secs(2))
    .build()
    .await?;
```

3. **Implement keep-alive:**
```rust
// TCP keep-alive is enabled by default in TcpTransport
```

---

### Protocol Issues

#### Problem: Protocol Version Mismatch
```
Error: Unknown protocol version
```

**Solutions:**

1. **Verify protocol version:**
```bash
echo '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2024-11-05"}}' | \
  nc multi-agent-container 9500
```

2. **Update client if needed:**
```bash
cargo update
```

---

#### Problem: Invalid JSON Response
```
Error: Failed to parse response: expected value at line 1
```

**Causes:**
- Server sending non-JSON data
- Multiple responses concatenated
- Encoding issues

**Solutions:**

1. **Enable debug logging:**
```bash
export RUST_LOG=trace,claude_flow=trace,tcp=trace
cargo run 2>&1 | tee debug.log
```

2. **Test with netcat:**
```bash
echo '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}' | \
  nc multi-agent-container 9500 | jq .
```

3. **Check for line delimiters:**
```rust
// Ensure newline after each message
writer.write_all(b"\n").await?;
```

---

### Performance Issues

#### Problem: High Latency
```
Average request latency > 10ms
```

**Solutions:**

1. **Check network performance:**
```bash
# Measure round-trip time
docker exec visionflow ping -c 100 multi-agent-container | tail -3

# Test TCP throughput
docker exec visionflow iperf3 -c multi-agent-container -p 9500
```

2. **Enable TCP nodelay:**
```rust
// Already enabled in TcpTransport
stream.set_nodelay(true)?;
```

3. **Use connection pooling:**
```rust
// Implement connection pool for high-throughput scenarios
struct ConnectionPool {
    connections: Vec<TcpTransport>,
}
```

---

#### Problem: Memory Leak
```
Memory usage continuously increasing
```

**Solutions:**

1. **Check for connection leaks:**
```bash
# Monitor open connections
watch -n 1 'netstat -an | grep 9500 | wc -l'
```

2. **Ensure proper cleanup:**
```rust
// Client automatically cleans up on drop
impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Connection closed automatically
    }
}
```

3. **Monitor memory usage:**
```bash
docker stats visionflow
```

---

### Docker Issues

#### Problem: Container Can't Find multi-agent-container
```
Error: failed to lookup address information: Name or service not known
```

**Solutions:**

1. **Verify both containers on same network:**
```bash
docker network inspect docker_ragflow | jq '.Containers'
```

2. **Add container to network:**
```bash
docker network connect docker_ragflow visionflow
```

3. **Use IP address instead of hostname:**
```bash
# Get container IP
docker inspect multi-agent-container | jq '.[0].NetworkSettings.Networks.docker_ragflow.IPAddress'

# Use IP in configuration
export CLAUDE_FLOW_HOST=172.18.0.2  # Use actual IP
```

---

#### Problem: Health Check Failing
```
container is unhealthy
```

**Solutions:**

1. **Check health endpoint:**
```bash
docker exec visionflow curl -v http://localhost:4000/api/health
```

2. **Verify TCP connection in health check:**
```rust
// Add TCP check to health endpoint
async fn health_check() -> Result<impl Responder> {
    let tcp_ok = TcpStream::connect("multi-agent-container:9500")
        .await
        .is_ok();
    
    Ok(Json(json!({
        "status": if tcp_ok { "healthy" } else { "degraded" },
        "tcp_connection": tcp_ok
    })))
}
```

---

### Development Issues

#### Problem: Tests Failing
```
cargo test fails with connection errors
```

**Solutions:**

1. **Use test configuration:**
```bash
# .env.test
CLAUDE_FLOW_HOST=localhost
MCP_TCP_PORT=9501  # Different port for tests
```

2. **Mock TCP server for tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    async fn start_mock_server() -> u16 {
        // Start mock TCP server
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        
        tokio::spawn(async move {
            // Handle test connections
        });
        
        port
    }
}
```

---

#### Problem: Can't Build Project
```
error: failed to compile
```

**Solutions:**

1. **Check Rust version:**
```bash
rustc --version  # Should be 1.75+
```

2. **Update dependencies:**
```bash
cargo update
cargo clean
cargo build
```

3. **Check for breaking changes:**
```bash
cargo check
```

---

## Debugging Tools

### Network Debugging

```bash
# TCP connection test
nc -zv multi-agent-container 9500

# Send test request
echo '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}' | nc -w 2 multi-agent-container 9500

# Monitor TCP traffic
tcpdump -i any -n port 9500

# Check socket states
ss -tan | grep 9500
```

### Application Debugging

```rust
// Enable detailed logging
use log::{debug, trace};

debug!("Connecting to {}:{}", host, port);
trace!("Request: {:?}", request);

// Add breakpoints
#[cfg(debug_assertions)]
{
    eprintln!("Debug: Connection state = {:?}", self.is_connected());
}

// Use debug binary
cargo build
RUST_LOG=trace ./target/debug/visionflow
```

### Docker Debugging

```bash
# Container logs
docker logs -f visionflow

# Execute commands in container
docker exec -it visionflow bash

# Copy files for inspection
docker cp visionflow:/app/logs/error.log ./

# Check resource usage
docker stats visionflow

# Inspect container configuration
docker inspect visionflow | jq '.[] | {NetworkMode, Networks, HostConfig}'
```

### Performance Profiling

```bash
# CPU profiling
perf record -F 99 -p $(pgrep visionflow)
perf report

# Memory profiling
valgrind --leak-check=full ./target/debug/visionflow

# Flame graphs
cargo install flamegraph
cargo flamegraph
```

## Log Analysis

### Important Log Patterns

```bash
# Connection failures
grep "TCP connection failed" /app/logs/visionflow.log

# Timeout issues
grep -i timeout /app/logs/visionflow.log

# Protocol errors
grep "Failed to parse" /app/logs/visionflow.log

# Reconnection attempts
grep "Retrying" /app/logs/visionflow.log
```

### Log Aggregation

```bash
# Combine logs from multiple sources
tail -f /app/logs/*.log | grep -E 'ERROR|WARN|TCP'

# JSON log parsing
cat /app/logs/visionflow.json | jq 'select(.level=="ERROR")'

# Time-based filtering
journalctl -u visionflow --since "1 hour ago"
```

## Recovery Procedures

### Full System Restart

```bash
# 1. Stop all services
docker-compose down

# 2. Clean up
docker system prune -f

# 3. Rebuild
docker-compose build --no-cache

# 4. Start services
docker-compose up -d

# 5. Verify health
docker-compose ps
docker exec visionflow curl http://localhost:4000/api/health
```

### Connection Reset

```rust
// Force reconnection in code
if let Some(mut client) = self.client.take() {
    client.disconnect().await?;
}

self.client = Some(
    ClaudeFlowClientBuilder::new()
        .with_tcp()
        .build()
        .await?
);
```

### Clear State

```bash
# Clear persistent data
docker exec visionflow rm -rf /app/data/cache/*

# Reset configuration
docker exec visionflow rm /app/.env
docker-compose up -d  # Recreates from environment
```

## Getting Help

### Diagnostic Information to Collect

When reporting issues, include:

1. **Error messages:**
```bash
docker logs visionflow | tail -100
```

2. **Configuration:**
```bash
docker exec visionflow env | grep MCP
```

3. **Network status:**
```bash
docker network inspect docker_ragflow
```

4. **Version information:**
```bash
docker exec visionflow visionflow --version
cargo --version
rustc --version
```

5. **Reproduction steps:**
- Exact commands run
- Expected vs actual behavior
- Frequency of issue

### Support Channels

- GitHub Issues: [github.com/your-org/visionflow/issues](https://github.com)
- Documentation: This guide
- Community Discord: [discord.gg/visionflow](https://discord.gg)

---

*Troubleshooting Guide Version: 1.0*
*Last Updated: 2025-08-12*