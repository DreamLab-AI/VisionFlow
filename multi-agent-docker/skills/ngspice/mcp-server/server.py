#!/usr/bin/env python3
"""
NGSpice MCP Server - FastMCP Implementation

Provides circuit simulation functionality via MCP protocol.
Wraps ngspice for SPICE simulations, DC/AC/transient analysis.
"""

import os
import subprocess
import tempfile
import re
from typing import Optional, List, Dict, Any
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# Initialize FastMCP server
mcp = FastMCP(
    "ngspice",
    version="2.0.0",
    description="Electronic circuit simulation - run SPICE netlists, DC/AC/transient analysis"
)

# =============================================================================
# Pydantic Models
# =============================================================================

class SimulateNetlistParams(BaseModel):
    """Parameters for running a SPICE netlist simulation."""
    netlist: str = Field(..., description="SPICE netlist content")
    timeout: int = Field(default=60, ge=1, le=600, description="Simulation timeout in seconds")

    @field_validator('netlist')
    @classmethod
    def validate_netlist(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Netlist cannot be empty")
        # Basic validation - should have title line
        lines = v.strip().split('\n')
        if len(lines) < 2:
            raise ValueError("Netlist must have at least a title and one component")
        return v


class DcAnalysisParams(BaseModel):
    """Parameters for DC analysis sweep."""
    netlist: str = Field(..., description="SPICE netlist content")
    source: str = Field(..., description="Voltage/current source name (e.g., 'V1')")
    start: float = Field(..., description="Start voltage/current value")
    stop: float = Field(..., description="Stop voltage/current value")
    step: float = Field(..., description="Step increment")
    output_node: str = Field(..., description="Node to measure (e.g., 'out')")


class AcAnalysisParams(BaseModel):
    """Parameters for AC frequency analysis."""
    netlist: str = Field(..., description="SPICE netlist content")
    analysis_type: str = Field(default="dec", description="Sweep type: dec, oct, lin")
    points: int = Field(default=10, ge=1, description="Points per decade/octave or total")
    fstart: float = Field(..., ge=0.001, description="Start frequency in Hz")
    fstop: float = Field(..., description="Stop frequency in Hz")
    output_node: str = Field(..., description="Node to measure")


class TransientParams(BaseModel):
    """Parameters for transient (time-domain) analysis."""
    netlist: str = Field(..., description="SPICE netlist content")
    step: float = Field(..., gt=0, description="Time step in seconds")
    stop: float = Field(..., gt=0, description="Stop time in seconds")
    start: float = Field(default=0, ge=0, description="Start time in seconds")
    output_nodes: List[str] = Field(..., description="Nodes to measure")


class OpPointParams(BaseModel):
    """Parameters for DC operating point analysis."""
    netlist: str = Field(..., description="SPICE netlist content")


# =============================================================================
# Tool Implementations
# =============================================================================

@mcp.tool()
def simulate_netlist(params: SimulateNetlistParams) -> dict:
    """Run a SPICE netlist simulation and return results."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cir') as f:
            f.write(params.netlist)
            netlist_path = f.name

        try:
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=params.timeout
            )

            output_data = _parse_spice_output(result.stdout)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "parsed_data": output_data,
                "returncode": result.returncode
            }
        finally:
            os.unlink(netlist_path)

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Simulation timed out after {params.timeout} seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def dc_analysis(params: DcAnalysisParams) -> dict:
    """Run DC sweep analysis on a circuit."""
    try:
        # Append DC analysis command to netlist
        analysis_cmd = f".dc {params.source} {params.start} {params.stop} {params.step}"
        print_cmd = f".print dc v({params.output_node})"
        control_block = f"""
.control
run
print v({params.output_node})
.endc
"""

        # Check if netlist already has .end
        netlist = params.netlist.strip()
        if netlist.lower().endswith('.end'):
            netlist = netlist[:-4].strip()

        full_netlist = f"{netlist}\n{analysis_cmd}\n{print_cmd}\n{control_block}\n.end"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cir') as f:
            f.write(full_netlist)
            netlist_path = f.name

        try:
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            data_points = _extract_dc_data(result.stdout, params.source, params.output_node)

            return {
                "success": result.returncode == 0,
                "analysis_type": "dc",
                "source": params.source,
                "output_node": params.output_node,
                "data_points": data_points,
                "raw_output": result.stdout
            }
        finally:
            os.unlink(netlist_path)

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def ac_analysis(params: AcAnalysisParams) -> dict:
    """Run AC frequency analysis on a circuit."""
    try:
        analysis_cmd = f".ac {params.analysis_type} {params.points} {params.fstart} {params.fstop}"
        control_block = f"""
.control
run
print vdb({params.output_node}) vp({params.output_node})
.endc
"""

        netlist = params.netlist.strip()
        if netlist.lower().endswith('.end'):
            netlist = netlist[:-4].strip()

        full_netlist = f"{netlist}\n{analysis_cmd}\n{control_block}\n.end"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cir') as f:
            f.write(full_netlist)
            netlist_path = f.name

        try:
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            return {
                "success": result.returncode == 0,
                "analysis_type": "ac",
                "frequency_range": {"start": params.fstart, "stop": params.fstop},
                "output_node": params.output_node,
                "raw_output": result.stdout
            }
        finally:
            os.unlink(netlist_path)

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def transient_analysis(params: TransientParams) -> dict:
    """Run transient (time-domain) analysis on a circuit."""
    try:
        analysis_cmd = f".tran {params.step} {params.stop} {params.start}"
        prints = ' '.join([f"v({node})" for node in params.output_nodes])
        control_block = f"""
.control
run
print {prints}
.endc
"""

        netlist = params.netlist.strip()
        if netlist.lower().endswith('.end'):
            netlist = netlist[:-4].strip()

        full_netlist = f"{netlist}\n{analysis_cmd}\n{control_block}\n.end"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cir') as f:
            f.write(full_netlist)
            netlist_path = f.name

        try:
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=300
            )

            return {
                "success": result.returncode == 0,
                "analysis_type": "transient",
                "time_range": {"start": params.start, "stop": params.stop, "step": params.step},
                "output_nodes": params.output_nodes,
                "raw_output": result.stdout
            }
        finally:
            os.unlink(netlist_path)

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def operating_point(params: OpPointParams) -> dict:
    """Calculate DC operating point of a circuit."""
    try:
        control_block = """
.control
op
print all
.endc
"""

        netlist = params.netlist.strip()
        if netlist.lower().endswith('.end'):
            netlist = netlist[:-4].strip()

        full_netlist = f"{netlist}\n{control_block}\n.end"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cir') as f:
            f.write(full_netlist)
            netlist_path = f.name

        try:
            result = subprocess.run(
                ['ngspice', '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            op_data = _extract_op_point(result.stdout)

            return {
                "success": result.returncode == 0,
                "analysis_type": "op",
                "operating_point": op_data,
                "raw_output": result.stdout
            }
        finally:
            os.unlink(netlist_path)

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def check_ngspice() -> dict:
    """Check ngspice installation and version."""
    try:
        result = subprocess.run(
            ['ngspice', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        version_match = re.search(r'ngspice[- ](\d+)', result.stdout + result.stderr, re.IGNORECASE)
        version = version_match.group(1) if version_match else "unknown"

        return {
            "success": True,
            "installed": True,
            "version": version,
            "output": result.stdout + result.stderr
        }
    except FileNotFoundError:
        return {"success": False, "installed": False, "error": "ngspice not found in PATH"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_spice_output(output: str) -> Dict[str, Any]:
    """Parse ngspice output into structured data."""
    data = {
        "analysis_complete": "Analysis complete" in output or "run complete" in output.lower(),
        "errors": [],
        "warnings": []
    }

    for line in output.split('\n'):
        if 'error' in line.lower():
            data["errors"].append(line.strip())
        elif 'warning' in line.lower():
            data["warnings"].append(line.strip())

    return data


def _extract_dc_data(output: str, source: str, node: str) -> List[Dict[str, float]]:
    """Extract DC sweep data points from output."""
    data_points = []
    # Simple extraction - real implementation would be more sophisticated
    lines = output.split('\n')
    for line in lines:
        # Look for numeric data lines
        parts = line.split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                data_points.append({"input": x, "output": y})
            except ValueError:
                continue
    return data_points


def _extract_op_point(output: str) -> Dict[str, float]:
    """Extract operating point voltages and currents."""
    op_data = {}
    # Look for node voltages in output
    voltage_pattern = re.compile(r'v\((\w+)\)\s*=\s*([-\d.e+]+)', re.IGNORECASE)
    current_pattern = re.compile(r'i\((\w+)\)\s*=\s*([-\d.e+]+)', re.IGNORECASE)

    for match in voltage_pattern.finditer(output):
        op_data[f"v({match.group(1)})"] = float(match.group(2))

    for match in current_pattern.finditer(output):
        op_data[f"i({match.group(1)})"] = float(match.group(2))

    return op_data


# =============================================================================
# Resource for Discovery
# =============================================================================

@mcp.resource("skill://ngspice/capabilities")
def get_capabilities() -> str:
    """Return skill capabilities for VisionFlow discovery."""
    return """
NGSpice MCP Server v2.0.0

Tools:
- simulate_netlist: Run raw SPICE netlist simulation
- dc_analysis: DC sweep analysis
- ac_analysis: AC frequency response analysis
- transient_analysis: Time-domain transient analysis
- operating_point: DC operating point calculation
- check_ngspice: Verify ngspice installation

Requirements:
- ngspice installed and in PATH
- Typically version 38+ recommended
"""


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()
