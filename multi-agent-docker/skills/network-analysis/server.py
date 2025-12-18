#!/usr/bin/env python3
"""
Network Analysis MCP Server
Provides network diagnostics and traffic analysis tools for Arch Linux
"""

import subprocess
import json
import shutil
from typing import Optional, Dict, Any, List
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

app = Server("network-analysis")

def run_command(cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
    """Execute shell command and return result"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def check_tool_available(tool: str) -> bool:
    """Check if a command-line tool is available"""
    return shutil.which(tool) is not None

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available network analysis tools"""
    return [
        Tool(
            name="connections",
            description="List active network connections with optional filtering by state (ESTABLISHED, LISTEN, etc.) and protocol (tcp, udp)",
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "Connection state filter (ESTABLISHED, LISTEN, etc.)"},
                    "protocol": {"type": "string", "description": "Protocol filter (tcp, udp)"}
                }
            }
        ),
        Tool(
            name="listening_ports",
            description="Show all listening TCP and UDP ports with process information",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="interface_stats",
            description="Display network interface statistics including bytes, packets, and errors",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="route_table",
            description="Display the system routing table",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="dns_lookup",
            description="Perform DNS lookup for a host with optional record type (A, AAAA, MX, TXT, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Hostname to lookup"},
                    "type": {"type": "string", "description": "DNS record type (A, AAAA, MX, TXT, etc.)", "default": "A"}
                },
                "required": ["host"]
            }
        ),
        Tool(
            name="ping_test",
            description="Test connectivity to a host using ICMP echo requests",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Host to ping"},
                    "count": {"type": "integer", "description": "Number of packets", "default": 4}
                },
                "required": ["host"]
            }
        ),
        Tool(
            name="traceroute",
            description="Trace network path to a destination host",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Destination host"},
                    "max_hops": {"type": "integer", "description": "Maximum hops", "default": 30}
                },
                "required": ["host"]
            }
        ),
        Tool(
            name="port_scan",
            description="Scan ports on a host (requires nmap). Use quick mode for common ports only.",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Host to scan"},
                    "ports": {"type": "string", "description": "Port range (e.g., 1-1000, 80,443)", "default": "1-1000"},
                    "quick": {"type": "boolean", "description": "Quick scan of common ports", "default": True}
                },
                "required": ["host"]
            }
        ),
        Tool(
            name="http_test",
            description="Test HTTP/HTTPS endpoint with custom method, headers, and data",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to test"},
                    "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    "headers": {"type": "object", "description": "HTTP headers as key-value pairs"},
                    "data": {"type": "string", "description": "Request body data"}
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="http_timing",
            description="Measure HTTP request timing breakdown (DNS, connect, transfer, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to measure"}
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="tcp_dump",
            description="Capture network packets (requires CAP_NET_RAW or root). Use BPF filters to limit capture.",
            inputSchema={
                "type": "object",
                "properties": {
                    "interface": {"type": "string", "description": "Network interface", "default": "any"},
                    "filter": {"type": "string", "description": "BPF filter expression"},
                    "count": {"type": "integer", "description": "Number of packets", "default": 100},
                    "timeout": {"type": "integer", "description": "Capture timeout in seconds", "default": 10}
                }
            }
        ),
        Tool(
            name="arp_table",
            description="Display ARP cache (neighbor table)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="bandwidth_test",
            description="Test bandwidth using iperf3 (requires iperf3 server)",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "iperf3 server host"}
                }
            }
        ),
        Tool(
            name="ssl_check",
            description="Check SSL/TLS certificate information",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Host to check"},
                    "port": {"type": "integer", "description": "Port number", "default": 443}
                },
                "required": ["host"]
            }
        ),
        Tool(
            name="whois_lookup",
            description="Lookup domain registration information",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain to lookup"}
                },
                "required": ["domain"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution"""

    if name == "connections":
        state = arguments.get("state")
        protocol = arguments.get("protocol")

        cmd = ["ss", "-tunapl"]
        result = run_command(cmd)

        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]

        output = result["stdout"]

        # Filter by state and protocol
        if state or protocol:
            lines = output.split('\n')
            filtered = [lines[0]]  # Keep header
            for line in lines[1:]:
                if state and state.upper() not in line.upper():
                    continue
                if protocol and not line.lower().startswith(protocol.lower()):
                    continue
                filtered.append(line)
            output = '\n'.join(filtered)

        return [TextContent(type="text", text=output)]

    elif name == "listening_ports":
        result = run_command(["ss", "-tlnp"])
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "interface_stats":
        result = run_command(["ip", "-s", "link"])
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "route_table":
        result = run_command(["ip", "route"])
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "dns_lookup":
        host = arguments["host"]
        dns_type = arguments.get("type", "A")

        if check_tool_available("dig"):
            result = run_command(["dig", "+short", host, dns_type])
        elif check_tool_available("host"):
            result = run_command(["host", "-t", dns_type, host])
        else:
            return [TextContent(type="text", text="Error: Neither dig nor host command available")]

        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "ping_test":
        host = arguments["host"]
        count = arguments.get("count", 4)

        result = run_command(["ping", "-c", str(count), host])
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "traceroute":
        host = arguments["host"]
        max_hops = arguments.get("max_hops", 30)

        if not check_tool_available("traceroute"):
            return [TextContent(type="text", text="Error: traceroute not installed")]

        result = run_command(["traceroute", "-m", str(max_hops), host], timeout=120)
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "port_scan":
        if not check_tool_available("nmap"):
            return [TextContent(type="text", text="Error: nmap not installed. Install with: pacman -S nmap")]

        host = arguments["host"]
        ports = arguments.get("ports", "1-1000")
        quick = arguments.get("quick", True)

        cmd = ["nmap"]
        if quick:
            cmd.append("-F")  # Fast scan
        else:
            cmd.extend(["-p", ports])
        cmd.append(host)

        result = run_command(cmd, timeout=300)
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "http_test":
        url = arguments["url"]
        method = arguments.get("method", "GET")
        headers = arguments.get("headers", {})
        data = arguments.get("data")

        cmd = ["curl", "-X", method, "-i", "-s"]

        for key, value in headers.items():
            cmd.extend(["-H", f"{key}: {value}"])

        if data:
            cmd.extend(["-d", data])

        cmd.append(url)

        result = run_command(cmd)
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "http_timing":
        url = arguments["url"]

        timing_format = (
            "time_namelookup:  %{time_namelookup}s\n"
            "time_connect:     %{time_connect}s\n"
            "time_appconnect:  %{time_appconnect}s\n"
            "time_pretransfer: %{time_pretransfer}s\n"
            "time_redirect:    %{time_redirect}s\n"
            "time_starttransfer: %{time_starttransfer}s\n"
            "time_total:       %{time_total}s\n"
            "http_code:        %{http_code}\n"
            "size_download:    %{size_download} bytes\n"
            "speed_download:   %{speed_download} bytes/s\n"
        )

        cmd = ["curl", "-w", timing_format, "-o", "/dev/null", "-s", url]
        result = run_command(cmd)

        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "tcp_dump":
        if not check_tool_available("tcpdump"):
            return [TextContent(type="text", text="Error: tcpdump not installed. Install with: pacman -S tcpdump")]

        interface = arguments.get("interface", "any")
        filter_expr = arguments.get("filter")
        count = arguments.get("count", 100)
        timeout = arguments.get("timeout", 10)

        cmd = ["tcpdump", "-i", interface, "-c", str(count), "-n"]

        if filter_expr:
            cmd.append(filter_expr)

        result = run_command(cmd, timeout=timeout + 5)

        if result.get("returncode") == 1 and "permission denied" in result.get("stderr", "").lower():
            return [TextContent(type="text", text="Error: Packet capture requires CAP_NET_RAW capability or root privileges")]

        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]

        return [TextContent(type="text", text=result["stdout"] + "\n\n" + result["stderr"])]

    elif name == "arp_table":
        result = run_command(["ip", "neigh"])
        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "bandwidth_test":
        if not check_tool_available("iperf3"):
            return [TextContent(type="text", text="Error: iperf3 not installed. Install with: pacman -S iperf3")]

        host = arguments.get("host")
        if not host:
            return [TextContent(type="text", text="Error: host parameter required for bandwidth test")]

        cmd = ["iperf3", "-c", host, "-J"]  # JSON output
        result = run_command(cmd, timeout=60)

        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    elif name == "ssl_check":
        host = arguments["host"]
        port = arguments.get("port", 443)

        cmd = ["openssl", "s_client", "-connect", f"{host}:{port}", "-servername", host]

        # Send empty input and close connection
        result = subprocess.run(
            cmd,
            input="",
            capture_output=True,
            text=True,
            timeout=10
        )

        # Extract certificate info
        cert_info = []
        in_cert = False
        for line in result.stdout.split('\n'):
            if 'Certificate chain' in line:
                in_cert = True
            if in_cert:
                cert_info.append(line)
            if 'Server certificate' in line:
                break

        output = '\n'.join(cert_info) if cert_info else result.stdout
        return [TextContent(type="text", text=output)]

    elif name == "whois_lookup":
        if not check_tool_available("whois"):
            return [TextContent(type="text", text="Error: whois not installed. Install with: pacman -S whois")]

        domain = arguments["domain"]
        result = run_command(["whois", domain])

        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result.get('error', result.get('stderr'))}")]
        return [TextContent(type="text", text=result["stdout"])]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
