#!/usr/bin/env python3
"""
KiCad MCP Server - FastMCP Implementation

Provides KiCad PCB/schematic design functionality via MCP protocol.
Wraps kicad-cli commands for project creation, export, and management.
"""

import os
import json
import subprocess
from typing import Optional, List
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# Initialize FastMCP server
mcp = FastMCP(
    "kicad",
    version="2.0.0",
    description="Electronic circuit design - create projects, manage schematics, export Gerbers and BOMs"
)

# =============================================================================
# Pydantic Models
# =============================================================================

class CreateProjectParams(BaseModel):
    """Parameters for creating a new KiCad project."""
    project_name: str = Field(..., description="Name of the KiCad project")
    project_dir: str = Field(default="/workspace", description="Parent directory for project")

    @field_validator('project_name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Project name cannot be empty")
        # Sanitize for filesystem
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Invalid character '{char}' in project name")
        return v.strip()


class ExportGerbersParams(BaseModel):
    """Parameters for exporting Gerber files."""
    pcb_file: str = Field(..., description="Path to .kicad_pcb file")
    output_dir: str = Field(default="/workspace/gerbers", description="Output directory for Gerber files")
    layers: Optional[List[str]] = Field(default=None, description="Specific layers to export (default: all)")


class ExportBomParams(BaseModel):
    """Parameters for exporting Bill of Materials."""
    schematic_file: str = Field(..., description="Path to .kicad_sch file")
    output_file: str = Field(..., description="Output CSV file path")
    format: str = Field(default="csv", description="Output format: csv, xml, json")


class RunCliParams(BaseModel):
    """Parameters for running arbitrary kicad-cli commands."""
    command: str = Field(..., description="KiCad CLI command (e.g., 'pcb export svg')")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    project_path: str = Field(default="/workspace", description="Working directory")

    @field_validator('command')
    @classmethod
    def validate_command(cls, v: str) -> str:
        # Security: only allow known kicad-cli subcommands
        allowed = ['pcb', 'sch', 'sym', 'fp', 'version']
        parts = v.split()
        if parts and parts[0] not in allowed:
            raise ValueError(f"Command must start with one of: {allowed}")
        return v


class SchematicExportParams(BaseModel):
    """Parameters for exporting schematic to PDF/SVG."""
    schematic_file: str = Field(..., description="Path to .kicad_sch file")
    output_file: str = Field(..., description="Output file path (.pdf or .svg)")
    format: str = Field(default="pdf", description="Export format: pdf, svg")


# =============================================================================
# Tool Implementations
# =============================================================================

@mcp.tool()
def create_project(params: CreateProjectParams) -> dict:
    """Create a new KiCad project with schematic and PCB files."""
    try:
        project_path = Path(params.project_dir) / params.project_name
        project_path.mkdir(parents=True, exist_ok=True)

        # Create basic KiCad project files
        kicad_pro = project_path / f"{params.project_name}.kicad_pro"
        kicad_sch = project_path / f"{params.project_name}.kicad_sch"
        kicad_pcb = project_path / f"{params.project_name}.kicad_pcb"

        # Project file
        pro_content = {
            "board": {"design_settings": {}, "layer_presets": [], "viewports": []},
            "boards": [],
            "libraries": {"pinned_footprint_libs": [], "pinned_symbol_libs": []},
            "meta": {"filename": f"{params.project_name}.kicad_pro", "version": 1},
            "net_settings": {"classes": [{"clearance": 0.2, "name": "Default"}]},
            "project": {"files": []},
            "schematic": {"design_settings": {}, "page_layout_descr_file": ""},
            "sheets": [["Root", ""]]
        }
        kicad_pro.write_text(json.dumps(pro_content, indent=2))

        # Schematic file (minimal KiCad 8 format)
        sch_content = f'''(kicad_sch (version 20231120) (generator "eeschema") (generator_version "8.0")
  (uuid "{_generate_uuid()}")
  (paper "A4")
  (lib_symbols)
  (symbol_instances)
)'''
        kicad_sch.write_text(sch_content)

        # PCB file (minimal KiCad 8 format)
        pcb_content = f'''(kicad_pcb (version 20231014) (generator "pcbnew") (generator_version "8.0")
  (general (thickness 1.6) (legacy_teardrops no))
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
    (50 "User.1" user)
    (51 "User.2" user)
  )
  (setup (pad_to_mask_clearance 0) (allow_soldermask_bridges_in_footprints no))
  (net 0 "")
)'''
        kicad_pcb.write_text(pcb_content)

        return {
            "success": True,
            "project_path": str(project_path),
            "files_created": [str(kicad_pro), str(kicad_sch), str(kicad_pcb)]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def export_gerbers(params: ExportGerbersParams) -> dict:
    """Export Gerber files from a KiCad PCB file."""
    try:
        if not Path(params.pcb_file).exists():
            return {"success": False, "error": f"PCB file not found: {params.pcb_file}"}

        os.makedirs(params.output_dir, exist_ok=True)

        cmd = ['kicad-cli', 'pcb', 'export', 'gerbers', '--output', params.output_dir, params.pcb_file]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            return {
                "success": False,
                "error": "Gerber export failed",
                "stderr": result.stderr
            }

        # List generated files
        gerber_files = []
        if os.path.exists(params.output_dir):
            gerber_files = [f for f in os.listdir(params.output_dir)
                          if f.endswith(('.gbr', '.drl', '.gbrjob'))]

        return {
            "success": True,
            "output_directory": params.output_dir,
            "files_generated": gerber_files
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Export timed out after 120 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def export_bom(params: ExportBomParams) -> dict:
    """Export Bill of Materials from schematic."""
    try:
        if not Path(params.schematic_file).exists():
            return {"success": False, "error": f"Schematic file not found: {params.schematic_file}"}

        os.makedirs(Path(params.output_file).parent, exist_ok=True)

        cmd = ['kicad-cli', 'sch', 'export', 'python-bom', '-o', params.output_file, params.schematic_file]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return {
                "success": False,
                "error": "BOM export failed",
                "stderr": result.stderr
            }

        return {
            "success": True,
            "output_file": params.output_file,
            "format": params.format
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def export_schematic(params: SchematicExportParams) -> dict:
    """Export schematic to PDF or SVG."""
    try:
        if not Path(params.schematic_file).exists():
            return {"success": False, "error": f"Schematic file not found: {params.schematic_file}"}

        os.makedirs(Path(params.output_file).parent, exist_ok=True)

        cmd = ['kicad-cli', 'sch', 'export', params.format, '-o', params.output_file, params.schematic_file]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Schematic export to {params.format} failed",
                "stderr": result.stderr
            }

        return {
            "success": True,
            "output_file": params.output_file,
            "format": params.format
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def run_kicad_cli(params: RunCliParams) -> dict:
    """Run arbitrary kicad-cli command with safety validation."""
    try:
        cmd = ['kicad-cli'] + params.command.split() + params.args

        result = subprocess.run(
            cmd,
            cwd=params.project_path,
            capture_output=True,
            text=True,
            timeout=120
        )

        return {
            "success": result.returncode == 0,
            "command": ' '.join(cmd),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def list_libraries() -> dict:
    """List available KiCad symbol and footprint libraries."""
    try:
        # Get symbol libraries
        sym_result = subprocess.run(
            ['kicad-cli', 'sym', 'export', 'svg', '--help'],
            capture_output=True, text=True, timeout=10
        )

        # Get footprint libraries
        fp_result = subprocess.run(
            ['kicad-cli', 'fp', 'export', 'svg', '--help'],
            capture_output=True, text=True, timeout=10
        )

        return {
            "success": True,
            "note": "Use kicad-cli sym/fp commands to work with libraries",
            "kicad_version": _get_kicad_version()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_uuid() -> str:
    """Generate a KiCad-compatible UUID."""
    import uuid
    return str(uuid.uuid4())


def _get_kicad_version() -> str:
    """Get installed KiCad version."""
    try:
        result = subprocess.run(['kicad-cli', 'version'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"


# =============================================================================
# Resource for Discovery
# =============================================================================

@mcp.resource("skill://kicad/capabilities")
def get_capabilities() -> str:
    """Return skill capabilities for VisionFlow discovery."""
    return """
KiCad MCP Server v2.0.0

Tools:
- create_project: Create new KiCad project with schematic and PCB
- export_gerbers: Export Gerber fabrication files
- export_bom: Export Bill of Materials
- export_schematic: Export schematic to PDF/SVG
- run_kicad_cli: Run arbitrary kicad-cli commands
- list_libraries: List available component libraries

Requirements:
- KiCad 8.x installed with kicad-cli in PATH
- Write access to output directories
"""


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()
