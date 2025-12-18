#!/usr/bin/env python3
"""
Linux Admin MCP Server
Provides read-only system diagnostics for Arch/CachyOS systems
"""

import subprocess
import json
import sys
from typing import Optional, Dict, Any, List
import psutil
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
except ImportError:
    print("Error: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

app = Server("linux-admin")


def run_command(cmd: List[str], timeout: int = 10) -> Dict[str, Any]:
    """
    Safely execute a command and return structured output.

    Args:
        cmd: Command as list of strings (no shell=True for security)
        timeout: Command timeout in seconds

    Returns:
        Dict with success, output, and error fields
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False  # Security: prevent shell injection
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Command timed out after {timeout} seconds",
            "returncode": -1
        }
    except FileNotFoundError:
        return {
            "success": False,
            "output": "",
            "error": f"Command not found: {cmd[0]}",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "returncode": -1
        }


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available Linux admin tools."""
    return [
        Tool(
            name="systemd_status",
            description="Get status of a systemd service or unit",
            inputSchema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service or unit name (e.g., 'sshd.service', 'nginx')"
                    }
                },
                "required": ["service"]
            }
        ),
        Tool(
            name="systemd_list",
            description="List systemd units filtered by state",
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Filter by state: running, failed, active, inactive, enabled, disabled",
                        "enum": ["running", "failed", "active", "inactive", "enabled", "disabled"]
                    }
                }
            }
        ),
        Tool(
            name="journal_query",
            description="Query system logs with journalctl",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "description": "Filter by systemd unit name"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to return",
                        "default": 100
                    },
                    "since": {
                        "type": "string",
                        "description": "Time filter (e.g., '1 hour ago', '2024-01-01', 'today')"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Log priority filter",
                        "enum": ["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"]
                    }
                }
            }
        ),
        Tool(
            name="network_info",
            description="Get network configuration and connection information",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="disk_usage",
            description="Check disk usage and block device information",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to check (default: /)",
                        "default": "/"
                    }
                }
            }
        ),
        Tool(
            name="process_list",
            description="List running processes with detailed information",
            inputSchema={
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "description": "Sort criteria",
                        "enum": ["cpu", "mem", "pid", "time"],
                        "default": "cpu"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of processes to return",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="memory_info",
            description="Get detailed memory and swap usage information",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="cpu_info",
            description="Get CPU information and current load statistics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="pacman_query",
            description="Query installed packages or get detailed package information",
            inputSchema={
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Specific package name for detailed info"
                    }
                }
            }
        ),
        Tool(
            name="pacman_search",
            description="Search available packages in Arch repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Search term"
                    }
                },
                "required": ["term"]
            }
        ),
        Tool(
            name="kernel_info",
            description="Get kernel version and configuration information",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="boot_log",
            description="Get boot logs from current boot",
            inputSchema={
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "integer",
                        "description": "Number of log lines to return",
                        "default": 50
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls."""

    if name == "systemd_status":
        service = arguments["service"]
        result = run_command(["systemctl", "status", service])
        return [TextContent(
            type="text",
            text=json.dumps({
                "service": service,
                "status": result["output"] if result["success"] else result["error"],
                "success": result["success"]
            }, indent=2)
        )]

    elif name == "systemd_list":
        state = arguments.get("state")
        cmd = ["systemctl", "list-units"]
        if state:
            cmd.extend(["--state", state])
        cmd.append("--no-pager")

        result = run_command(cmd)
        return [TextContent(
            type="text",
            text=json.dumps({
                "state_filter": state or "all",
                "units": result["output"] if result["success"] else result["error"],
                "success": result["success"]
            }, indent=2)
        )]

    elif name == "journal_query":
        cmd = ["journalctl", "--no-pager"]

        if arguments.get("unit"):
            cmd.extend(["-u", arguments["unit"]])

        lines = arguments.get("lines", 100)
        cmd.extend(["-n", str(lines)])

        if arguments.get("since"):
            cmd.extend(["--since", arguments["since"]])

        if arguments.get("priority"):
            cmd.extend(["-p", arguments["priority"]])

        result = run_command(cmd, timeout=30)
        return [TextContent(
            type="text",
            text=json.dumps({
                "query": {
                    "unit": arguments.get("unit"),
                    "lines": lines,
                    "since": arguments.get("since"),
                    "priority": arguments.get("priority")
                },
                "logs": result["output"] if result["success"] else result["error"],
                "success": result["success"]
            }, indent=2)
        )]

    elif name == "network_info":
        # Get IP addresses
        ip_addr = run_command(["ip", "addr"])
        # Get routing table
        ip_route = run_command(["ip", "route"])
        # Get listening ports and connections
        ss_result = run_command(["ss", "-tuln"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "interfaces": ip_addr["output"] if ip_addr["success"] else ip_addr["error"],
                "routes": ip_route["output"] if ip_route["success"] else ip_route["error"],
                "connections": ss_result["output"] if ss_result["success"] else ss_result["error"],
                "success": all([ip_addr["success"], ip_route["success"], ss_result["success"]])
            }, indent=2)
        )]

    elif name == "disk_usage":
        path = arguments.get("path", "/")

        # Get filesystem usage
        df_result = run_command(["df", "-h", path])
        # Get block devices
        lsblk_result = run_command(["lsblk"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "path": path,
                "usage": df_result["output"] if df_result["success"] else df_result["error"],
                "block_devices": lsblk_result["output"] if lsblk_result["success"] else lsblk_result["error"],
                "success": df_result["success"] and lsblk_result["success"]
            }, indent=2)
        )]

    elif name == "process_list":
        sort_by = arguments.get("sort_by", "cpu")
        limit = arguments.get("limit", 20)

        # Map sort_by to ps sort format
        sort_map = {
            "cpu": "-pcpu",
            "mem": "-pmem",
            "pid": "pid",
            "time": "-time"
        }
        sort_arg = sort_map.get(sort_by, "-pcpu")

        result = run_command(["ps", "aux", f"--sort={sort_arg}"])

        if result["success"]:
            lines = result["output"].split("\n")
            # Keep header + limited number of processes
            output = "\n".join(lines[:limit+1])
        else:
            output = result["error"]

        return [TextContent(
            type="text",
            text=json.dumps({
                "sort_by": sort_by,
                "limit": limit,
                "processes": output,
                "success": result["success"]
            }, indent=2)
        )]

    elif name == "memory_info":
        # Get memory info using free command
        free_result = run_command(["free", "-h"])

        # Also get detailed info from psutil
        try:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            memory_details = {
                "total": f"{vm.total / (1024**3):.2f} GB",
                "available": f"{vm.available / (1024**3):.2f} GB",
                "used": f"{vm.used / (1024**3):.2f} GB",
                "percent": f"{vm.percent}%",
                "swap_total": f"{swap.total / (1024**3):.2f} GB",
                "swap_used": f"{swap.used / (1024**3):.2f} GB",
                "swap_percent": f"{swap.percent}%"
            }
        except Exception as e:
            memory_details = {"error": str(e)}

        return [TextContent(
            type="text",
            text=json.dumps({
                "summary": free_result["output"] if free_result["success"] else free_result["error"],
                "details": memory_details,
                "success": free_result["success"]
            }, indent=2)
        )]

    elif name == "cpu_info":
        # Get CPU info
        lscpu_result = run_command(["lscpu"])

        # Get load average using psutil
        try:
            load_avg = psutil.getloadavg()
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_details = {
                "load_average_1min": load_avg[0],
                "load_average_5min": load_avg[1],
                "load_average_15min": load_avg[2],
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "per_cpu_percent": cpu_percent
            }
        except Exception as e:
            cpu_details = {"error": str(e)}

        return [TextContent(
            type="text",
            text=json.dumps({
                "info": lscpu_result["output"] if lscpu_result["success"] else lscpu_result["error"],
                "details": cpu_details,
                "success": lscpu_result["success"]
            }, indent=2)
        )]

    elif name == "pacman_query":
        package = arguments.get("package")

        if package:
            # Get detailed info for specific package
            result = run_command(["pacman", "-Qi", package])
        else:
            # List all installed packages
            result = run_command(["pacman", "-Q"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "package": package or "all",
                "info": result["output"] if result["success"] else result["error"],
                "success": result["success"]
            }, indent=2)
        )]

    elif name == "pacman_search":
        term = arguments["term"]
        result = run_command(["pacman", "-Ss", term])

        return [TextContent(
            type="text",
            text=json.dumps({
                "search_term": term,
                "results": result["output"] if result["success"] else result["error"],
                "success": result["success"]
            }, indent=2)
        )]

    elif name == "kernel_info":
        # Get kernel version
        uname_result = run_command(["uname", "-a"])

        # Try to read /proc/version
        try:
            proc_version = Path("/proc/version").read_text()
        except Exception as e:
            proc_version = f"Error reading /proc/version: {e}"

        # Try to read kernel command line
        try:
            cmdline = Path("/proc/cmdline").read_text().strip()
        except Exception as e:
            cmdline = f"Error reading /proc/cmdline: {e}"

        return [TextContent(
            type="text",
            text=json.dumps({
                "uname": uname_result["output"] if uname_result["success"] else uname_result["error"],
                "proc_version": proc_version,
                "cmdline": cmdline,
                "success": uname_result["success"]
            }, indent=2)
        )]

    elif name == "boot_log":
        lines = arguments.get("lines", 50)
        result = run_command(["journalctl", "-b", "--no-pager", "-n", str(lines)])

        return [TextContent(
            type="text",
            text=json.dumps({
                "lines": lines,
                "log": result["output"] if result["success"] else result["error"],
                "success": result["success"]
            }, indent=2)
        )]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Unknown tool: {name}"
            })
        )]


def main():
    """Run the MCP server."""
    import asyncio

    async def run_server():
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
