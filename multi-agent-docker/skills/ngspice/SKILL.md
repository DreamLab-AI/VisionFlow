---
name: ngspice
description: Electronic circuit simulation and SPICE analysis via ngspice FastMCP server
version: 2.0.0
mcp_server: true
protocol: fastmcp
entry_point: mcp-server/server.py
---

# NGSpice Circuit Simulator Skill

Execute SPICE circuit simulations using ngspice via FastMCP protocol.

## Capabilities

- Run SPICE netlists with full simulation output
- DC sweep analysis with data extraction
- AC frequency response analysis
- Transient (time-domain) analysis
- DC operating point calculation
- Structured result parsing

## MCP Tools

| Tool | Description |
|------|-------------|
| `simulate_netlist` | Run raw SPICE netlist and capture output |
| `dc_analysis` | DC sweep with configurable source and range |
| `ac_analysis` | Frequency response with magnitude/phase |
| `transient_analysis` | Time-domain simulation |
| `operating_point` | Calculate DC operating point |
| `check_ngspice` | Verify ngspice installation |

## Prerequisites

- ngspice installed (version 38+ recommended)
- Python 3.11+ with FastMCP: `pip install mcp`

## Usage

```python
# Simple netlist simulation
simulate_netlist(netlist="""
RC Low Pass Filter
V1 in 0 DC 1 AC 1
R1 in out 1k
C1 out 0 1u
.end
""")

# DC sweep analysis
dc_analysis(
    netlist="...",
    source="V1",
    start=0,
    stop=5,
    step=0.1,
    output_node="out"
)

# Transient analysis
transient_analysis(
    netlist="...",
    step=1e-6,
    stop=1e-3,
    output_nodes=["out", "in"]
)
```

## MCP Configuration

```json
{
  "mcpServers": {
    "ngspice": {
      "command": "python",
      "args": ["/home/devuser/.claude/skills/ngspice/mcp-server/server.py"]
    }
  }
}
```
