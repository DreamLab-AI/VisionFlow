---
name: linux-admin
version: 1.0.0
description: Read-only Linux system diagnostics for Arch/CachyOS - systemd, journalctl, pacman, network, storage
author: agentic-workstation
tags: [linux, arch, cachyos, systemd, pacman, diagnostics, network, storage]
mcp_server: true
---

# Linux Admin Skill

Comprehensive read-only system diagnostics and monitoring for Arch Linux/CachyOS systems. This skill provides safe, non-invasive access to system information through standardized tools.

## Overview

The linux-admin skill enables AI agents to gather system information and diagnose issues on Arch/CachyOS systems without making any modifications. All operations are read-only for safety.

### Key Features

- **Systemd Management**: Query service status and unit states
- **Journal Access**: Search and filter system logs with journalctl
- **Network Diagnostics**: View network configuration and connections
- **Storage Monitoring**: Check disk usage and block devices
- **Process Management**: List and analyze running processes
- **Resource Monitoring**: CPU, memory, and system load information
- **Package Management**: Query installed packages and search repositories
- **Kernel Information**: View kernel version and boot logs

## Tools

### systemd_status
Query the status of a systemd service or unit.

**Parameters:**
- `service` (string, required): Service or unit name (e.g., "sshd.service", "nginx")

**Returns:** Service status including state, PID, memory usage, and recent log entries

**Example:**
```json
{
  "service": "docker.service"
}
```

### systemd_list
List systemd units filtered by state.

**Parameters:**
- `state` (string, optional): Filter by state (running, failed, active, inactive, enabled, disabled)

**Returns:** List of units matching the filter criteria

**Example:**
```json
{
  "state": "failed"
}
```

### journal_query
Query system logs using journalctl with flexible filtering.

**Parameters:**
- `unit` (string, optional): Filter by systemd unit name
- `lines` (integer, optional): Number of lines to return (default: 100)
- `since` (string, optional): Time filter (e.g., "1 hour ago", "2024-01-01", "today")
- `priority` (string, optional): Log priority filter (emerg, alert, crit, err, warning, notice, info, debug)

**Returns:** Filtered journal entries

**Example:**
```json
{
  "unit": "sshd.service",
  "lines": 50,
  "since": "1 hour ago",
  "priority": "err"
}
```

### network_info
Get comprehensive network configuration and connection information.

**Parameters:** None

**Returns:** Network interfaces, IP addresses, routing table, and active connections

**Example:**
```json
{}
```

### disk_usage
Check disk usage and block device information.

**Parameters:**
- `path` (string, optional): Path to check (default: "/")

**Returns:** Filesystem usage statistics and block device layout

**Example:**
```json
{
  "path": "/home"
}
```

### process_list
List running processes with detailed information.

**Parameters:**
- `sort_by` (string, optional): Sort criteria (cpu, mem, pid, time) - default: "cpu"
- `limit` (integer, optional): Number of processes to return (default: 20)

**Returns:** Top processes sorted by specified criteria

**Example:**
```json
{
  "sort_by": "mem",
  "limit": 10
}
```

### memory_info
Get detailed memory and swap usage information.

**Parameters:** None

**Returns:** Memory statistics including total, used, free, buffers, cache, and swap

**Example:**
```json
{}
```

### cpu_info
Get CPU information and current load statistics.

**Parameters:** None

**Returns:** CPU model, core count, architecture, and load averages

**Example:**
```json
{}
```

### pacman_query
Query installed packages or get detailed package information.

**Parameters:**
- `package` (string, optional): Specific package name for detailed info

**Returns:** List of installed packages or detailed info for specific package

**Example:**
```json
{
  "package": "linux"
}
```

### pacman_search
Search available packages in Arch repositories.

**Parameters:**
- `term` (string, required): Search term

**Returns:** Matching packages from repositories

**Example:**
```json
{
  "term": "docker"
}
```

### kernel_info
Get kernel version and configuration information.

**Parameters:** None

**Returns:** Kernel version, build info, and command line parameters

**Example:**
```json
{}
```

### boot_log
Get boot logs from current or previous boot.

**Parameters:**
- `lines` (integer, optional): Number of log lines to return (default: 50)

**Returns:** Boot log entries from journalctl

**Example:**
```json
{
  "lines": 100
}
```

## Usage Examples

### Diagnosing Service Issues

```python
# Check if a service is running
status = systemd_status("nginx.service")

# Find all failed services
failed = systemd_list("failed")

# Get recent error logs for a service
logs = journal_query(
    unit="nginx.service",
    since="1 hour ago",
    priority="err"
)
```

### Network Troubleshooting

```python
# Get network configuration
network = network_info()

# Check for connection issues in logs
conn_logs = journal_query(
    since="today",
    priority="err"
)
```

### Performance Monitoring

```python
# Check top CPU consumers
processes = process_list(sort_by="cpu", limit=10)

# Check memory usage
memory = memory_info()

# Check disk space
disk = disk_usage("/")

# Get system load
cpu = cpu_info()
```

### Package Investigation

```python
# Check if package is installed
pkg_info = pacman_query("docker")

# Search for available packages
search_results = pacman_search("container")

# List all installed packages
all_packages = pacman_query()
```

### System Health Check

```python
# Quick system overview
kernel = kernel_info()
cpu = cpu_info()
memory = memory_info()
disk = disk_usage("/")
failed_services = systemd_list("failed")
recent_errors = journal_query(
    since="1 hour ago",
    priority="err",
    lines=50
)
```

## Security

This skill is designed with security as a priority:

### Read-Only Operations
All tools perform read-only operations. No system modifications are possible through this skill.

### Safe Command Execution
- Commands use `subprocess` with `shell=False` to prevent injection attacks
- All parameters are validated before execution
- Commands run with current user permissions (no privilege escalation)

### Information Disclosure
While this skill is read-only, it can access sensitive system information:
- Log files may contain sensitive data
- Process lists may reveal running applications
- Network information shows configuration details

**Best Practices:**
- Use in trusted environments only
- Review log outputs before sharing
- Limit access to authorized personnel
- Be aware that some operations require appropriate user permissions

### Permission Requirements

Most operations work with regular user permissions, but some require elevated privileges:
- **No special permissions**: CPU info, memory info, kernel info, pacman queries
- **May require sudo**: Full journal access, some network details, detailed process info
- **Always allowed**: Service status checks, boot logs, disk usage

## MCP Server Configuration

This skill runs as an MCP server. Configuration:

```json
{
  "mcpServers": {
    "linux-admin": {
      "command": "python",
      "args": ["/path/to/skills/linux-admin/server.py"]
    }
  }
}
```

## Dependencies

- Python 3.8+
- mcp package
- psutil package
- Standard Linux utilities (systemctl, journalctl, ip, df, etc.)

## Platform Support

- **Primary**: Arch Linux, CachyOS
- **Compatible**: Manjaro, EndeavourOS, other Arch derivatives
- **Partial**: Other systemd-based Linux distributions (tool paths may vary)

## Error Handling

All tools return structured error information when commands fail:
- Permission denied errors
- Command not found errors
- Invalid parameter errors
- Timeout errors for long-running queries

## Performance Considerations

- Journal queries can be slow on systems with large logs (use `since` parameter)
- Process listing is cached for 1 second to avoid overhead
- Large disk operations may take time on slow storage
- Network queries are generally fast but depend on system state

## Limitations

- Cannot modify system configuration
- Cannot restart services or manage systemd units
- Cannot install or remove packages
- Cannot modify network configuration
- Limited by user permissions (some data may require sudo)

## Future Enhancements

Potential additions for future versions:
- Performance metrics collection over time
- Log pattern analysis and anomaly detection
- Automated health check reporting
- Custom alert thresholds
- Hardware sensor monitoring
- Container and VM introspection
