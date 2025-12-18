---
name: kicad
description: Control KiCad for electronic circuit design, schematic creation, and PCB layout operations
version: 2.0.0
mcp_server: true
protocol: fastmcp
entry_point: mcp-server/server.py
---

# KiCad PCB Skill

Interact with KiCad for electronic circuit design, schematic manipulation, and PCB layout via FastMCP.

## Capabilities

- Create and manage KiCad 8.x projects
- Export Gerber fabrication files
- Generate Bill of Materials (BOM)
- Export schematics to PDF/SVG
- Run arbitrary kicad-cli commands
- List component libraries

## MCP Tools

| Tool | Description |
|------|-------------|
| `create_project` | Create new KiCad project with schematic and PCB files |
| `export_gerbers` | Export Gerber fabrication files from PCB |
| `export_bom` | Generate Bill of Materials from schematic |
| `export_schematic` | Export schematic to PDF or SVG |
| `run_kicad_cli` | Run validated kicad-cli commands |
| `list_libraries` | List available symbol/footprint libraries |

## Prerequisites

- KiCad 8.x installed with `kicad-cli` in PATH
- Python 3.11+ with FastMCP: `pip install mcp`

## Usage

```python
# Create a new project
create_project(project_name="my_board", project_dir="/workspace/electronics")

# Export Gerbers for fabrication
export_gerbers(pcb_file="/workspace/my_board/my_board.kicad_pcb", output_dir="/workspace/gerbers")

# Generate BOM
export_bom(schematic_file="/workspace/my_board/my_board.kicad_sch", output_file="/workspace/bom.csv")
```

## MCP Configuration

```json
{
  "mcpServers": {
    "kicad": {
      "command": "python",
      "args": ["/home/devuser/.claude/skills/kicad/mcp-server/server.py"]
    }
  }
}
```
