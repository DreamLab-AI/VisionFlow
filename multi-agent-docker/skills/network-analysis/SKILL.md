---
name: network-analysis
version: 1.0.0
description: Network diagnostics and traffic analysis - tcpdump, ss, nmap, curl testing
author: agentic-workstation
tags: [network, tcpdump, wireshark, nmap, diagnostics, traffic]
mcp_server: true
---

# Network Analysis Skill

Comprehensive network diagnostics and traffic analysis using native Linux tools. Arch Linux compatible alternative to wireshark-mcp.

## Overview

This skill provides network analysis capabilities through standard Linux networking tools:
- Connection monitoring (ss, netstat replacement)
- Traffic capture (tcpdump)
- Port scanning (nmap)
- HTTP testing (curl with timing)
- DNS diagnostics (dig, host)
- SSL/TLS certificate inspection (openssl)
- Network path analysis (ping, traceroute)

## Available Tools

### Connection Monitoring
- `connections(state, protocol)` - List active connections with filtering
- `listening_ports()` - Show listening TCP/UDP ports
- `interface_stats()` - Network interface statistics
- `route_table()` - Display routing table

### DNS & Discovery
- `dns_lookup(host, type)` - DNS queries (A, AAAA, MX, TXT, etc.)
- `arp_table()` - ARP cache (neighbor table)
- `whois_lookup(domain)` - Domain registration info

### Connectivity Testing
- `ping_test(host, count)` - ICMP echo test
- `traceroute(host, max_hops)` - Network path trace
- `http_test(url, method, headers, data)` - HTTP request testing
- `http_timing(url)` - HTTP with timing breakdown
- `ssl_check(host, port)` - SSL/TLS certificate inspection

### Advanced Analysis
- `port_scan(host, ports, quick)` - Port scanning (requires nmap)
- `tcp_dump(interface, filter, count, timeout)` - Packet capture
- `bandwidth_test(host)` - Bandwidth testing (requires iperf3)

## Capture Examples

### Basic Connection Monitoring
```bash
# List all established TCP connections
connections(state="ESTABLISHED", protocol="tcp")

# Show listening ports
listening_ports()

# View interface statistics
interface_stats()
```

### Traffic Capture
```bash
# Capture 100 packets on all interfaces
tcp_dump(interface="any", count=100)

# Capture HTTP traffic on eth0
tcp_dump(interface="eth0", filter="tcp port 80", count=50)

# Capture DNS queries
tcp_dump(filter="udp port 53", count=20)
```

### HTTP Testing
```bash
# Simple GET request
http_test("https://example.com")

# POST with JSON data
http_test("https://api.example.com/data",
          method="POST",
          headers={"Content-Type": "application/json"},
          data='{"key": "value"}')

# Timing analysis
http_timing("https://example.com")
```

### Network Diagnostics
```bash
# DNS lookup
dns_lookup("example.com", type="A")
dns_lookup("example.com", type="MX")

# Check SSL certificate
ssl_check("example.com", port=443)

# Trace route
traceroute("8.8.8.8", max_hops=15)
```

### Port Scanning
```bash
# Quick scan common ports
port_scan("192.168.1.1", quick=True)

# Scan specific range
port_scan("192.168.1.1", ports="20-443")
```

## Security Notes

### Privilege Requirements

**Root/Capabilities Required:**
- `tcp_dump()` - Requires CAP_NET_RAW or root for packet capture
- `port_scan()` - May require privileges for SYN scanning
- `bandwidth_test()` - Server mode requires binding privileged ports

**User-Level Tools:**
- `connections()` - Readable by all users
- `listening_ports()` - Readable by all users
- `http_test()`, `http_timing()` - No special privileges
- `dns_lookup()` - No special privileges
- `ping_test()` - Usually available to all users
- `traceroute()` - Usually available to all users
- `ssl_check()` - No special privileges

### Running with Capabilities

For Docker containers, grant specific capabilities instead of full root:

```dockerfile
# Minimal capabilities for packet capture
CAP_NET_RAW, CAP_NET_ADMIN
```

```bash
# Run container with specific capabilities
docker run --cap-add=NET_RAW --cap-add=NET_ADMIN ...
```

### Security Best Practices

1. **Rate Limiting**: Implement rate limits on scan operations
2. **Network Isolation**: Use in isolated networks when testing
3. **Audit Logging**: Log all capture and scan operations
4. **Data Sanitization**: Filter sensitive data from captures
5. **Permission Validation**: Check capabilities before attempting privileged operations

### Ethical Usage

- Only scan networks you own or have permission to test
- Respect rate limits and avoid DoS conditions
- Follow organizational security policies
- Document all network analysis activities
- Use for legitimate diagnostics and security testing only

## Installation Requirements

### Arch Linux Packages
```bash
pacman -S tcpdump iproute2 bind-tools curl openssl whois nmap iperf3 traceroute
```

### Minimum Required
- `iproute2` (ss, ip commands)
- `curl` (HTTP testing)
- `bind-tools` (dig, host)

### Optional Tools
- `tcpdump` (packet capture)
- `nmap` (port scanning)
- `iperf3` (bandwidth testing)
- `whois` (domain lookup)
- `traceroute` (path analysis)

## MCP Integration

This skill runs as an MCP server, exposing all tools through the Model Context Protocol:

```json
{
  "mcpServers": {
    "network-analysis": {
      "command": "python",
      "args": ["/path/to/skills/network-analysis/server.py"]
    }
  }
}
```

## Usage Examples

### Diagnosing Connection Issues
```python
# Check if service is listening
ports = listening_ports()
# Verify connectivity
ping = ping_test("server.example.com")
# Check routing
routes = route_table()
# Test application layer
http = http_test("http://server.example.com:8080/health")
```

### Performance Analysis
```python
# Measure HTTP timing
timing = http_timing("https://api.example.com")
# Check interface statistics
stats = interface_stats()
# Run bandwidth test
bandwidth = bandwidth_test("iperf.server.com")
```

### Security Auditing
```python
# Discover listening services
ports = listening_ports()
# Scan for open ports
scan = port_scan("192.168.1.1", ports="1-65535")
# Verify SSL configuration
ssl = ssl_check("example.com", 443)
```

## Troubleshooting

### Permission Denied
- Check if tool requires root/capabilities
- Verify user is in required groups (e.g., wireshark for tcpdump)
- Use `getcap` to check file capabilities

### Tool Not Found
- Install missing packages using pacman
- Verify tool is in PATH
- Check for alternative tool names (e.g., ss vs netstat)

### Capture Issues
- Verify interface name with `ip link`
- Check for firewall rules blocking capture
- Ensure sufficient disk space for captures
- Use correct BPF filter syntax

## Performance Considerations

- Use filters to reduce capture volume
- Set appropriate timeouts and count limits
- Consider network impact of scans
- Monitor CPU/memory during packet capture
- Use quick scan mode for reconnaissance

## Related Skills

- **system-monitoring**: System-level resource monitoring
- **security-audit**: Security scanning and auditing
- **docker-networking**: Container network analysis
- **log-analysis**: Parse and analyze network logs
