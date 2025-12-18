#!/usr/bin/env python3
"""
tmux-ops MCP Server
Terminal multiplexer operations for persistent background sessions
"""

import json
import subprocess
import time
import os
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("tmux-ops")


def run_tmux_command(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Execute tmux command and return result"""
    cmd = ["tmux"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tmux command failed: {e.stderr}")


def parse_tmux_list(output: str, format_spec: str) -> List[Dict[str, Any]]:
    """Parse tmux list output based on format specification"""
    if not output.strip():
        return []

    lines = output.strip().split("\n")
    keys = format_spec.split(",")

    results = []
    for line in lines:
        values = line.split(",")
        item = {}
        for i, key in enumerate(keys):
            if i < len(values):
                item[key] = values[i]
        results.append(item)

    return results


@mcp.tool()
def session_list() -> List[Dict[str, str]]:
    """List all tmux sessions

    Returns:
        List of session info dicts with: name, created, attached status
    """
    result = run_tmux_command(
        ["list-sessions", "-F", "#{session_name},#{session_created},#{session_attached}"],
        check=False
    )

    if result.returncode != 0:
        return []

    sessions = parse_tmux_list(result.stdout, "name,created,attached")

    # Convert attached to boolean
    for session in sessions:
        session["attached"] = session.get("attached", "0") != "0"

    return sessions


@mcp.tool()
def session_new(name: str, command: Optional[str] = None, detached: bool = True) -> Dict[str, Any]:
    """Create new tmux session

    Args:
        name: Session name
        command: Optional command to run in session
        detached: If True, create detached session (default)

    Returns:
        Success status and session info
    """
    args = ["new-session", "-s", name]

    if detached:
        args.append("-d")

    if command:
        args.append(command)

    result = run_tmux_command(args, check=False)

    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr.strip()
        }

    return {
        "success": True,
        "name": name,
        "detached": detached,
        "command": command
    }


@mcp.tool()
def session_kill(name: str) -> Dict[str, Any]:
    """Kill tmux session

    Args:
        name: Session name to kill

    Returns:
        Success status
    """
    result = run_tmux_command(["kill-session", "-t", name], check=False)

    return {
        "success": result.returncode == 0,
        "name": name,
        "error": result.stderr.strip() if result.returncode != 0 else None
    }


@mcp.tool()
def session_exists(name: str) -> bool:
    """Check if tmux session exists

    Args:
        name: Session name to check

    Returns:
        True if session exists
    """
    result = run_tmux_command(["has-session", "-t", name], check=False)
    return result.returncode == 0


@mcp.tool()
def window_list(session: str) -> List[Dict[str, Any]]:
    """List windows in session

    Args:
        session: Session name

    Returns:
        List of window info dicts
    """
    result = run_tmux_command(
        ["list-windows", "-t", session, "-F", "#{window_index},#{window_name},#{window_active}"],
        check=False
    )

    if result.returncode != 0:
        return []

    windows = parse_tmux_list(result.stdout, "index,name,active")

    # Convert types
    for window in windows:
        window["index"] = int(window.get("index", 0))
        window["active"] = window.get("active", "0") != "0"

    return windows


@mcp.tool()
def window_new(session: str, name: Optional[str] = None, command: Optional[str] = None) -> Dict[str, Any]:
    """Create new window in session

    Args:
        session: Session name
        name: Optional window name
        command: Optional command to run

    Returns:
        Success status and window info
    """
    args = ["new-window", "-t", session]

    if name:
        args.extend(["-n", name])

    if command:
        args.append(command)

    result = run_tmux_command(args, check=False)

    return {
        "success": result.returncode == 0,
        "session": session,
        "name": name,
        "command": command,
        "error": result.stderr.strip() if result.returncode != 0 else None
    }


@mcp.tool()
def pane_list(session: str, window: Optional[int] = None) -> List[Dict[str, Any]]:
    """List panes in session or specific window

    Args:
        session: Session name
        window: Optional window index

    Returns:
        List of pane info dicts
    """
    target = f"{session}:{window}" if window is not None else session

    result = run_tmux_command(
        ["list-panes", "-t", target, "-F", "#{pane_id},#{pane_active},#{pane_width},#{pane_height}"],
        check=False
    )

    if result.returncode != 0:
        return []

    panes = parse_tmux_list(result.stdout, "pane_id,active,width,height")

    # Convert types
    for pane in panes:
        pane["active"] = pane.get("active", "0") != "0"
        pane["width"] = int(pane.get("width", 0))
        pane["height"] = int(pane.get("height", 0))

    return panes


@mcp.tool()
def send_keys(target: str, keys: str, enter: bool = True) -> Dict[str, Any]:
    """Send keystrokes to pane

    Args:
        target: Target session:window.pane (e.g., "build:0.0")
        keys: Keys to send
        enter: If True, send Enter after keys

    Returns:
        Success status
    """
    args = ["send-keys", "-t", target, keys]

    if enter:
        args.append("Enter")

    result = run_tmux_command(args, check=False)

    return {
        "success": result.returncode == 0,
        "target": target,
        "keys": keys,
        "error": result.stderr.strip() if result.returncode != 0 else None
    }


@mcp.tool()
def capture_pane(target: str, lines: int = 100, start: Optional[int] = None) -> str:
    """Capture pane output

    Args:
        target: Target session:window.pane
        lines: Number of lines to capture (default 100)
        start: Optional start line (negative for from end)

    Returns:
        Captured output text
    """
    if start is not None:
        range_spec = f"-S {start} -E {start + lines - 1}"
    else:
        range_spec = f"-S -{lines}"

    args = ["capture-pane", "-t", target, "-p"] + range_spec.split()

    result = run_tmux_command(args, check=False)

    if result.returncode != 0:
        return ""

    return result.stdout


@mcp.tool()
def get_pane_pid(target: str) -> Optional[int]:
    """Get PID of process running in pane

    Args:
        target: Target session:window.pane

    Returns:
        Process PID or None if not found
    """
    result = run_tmux_command(
        ["display-message", "-t", target, "-p", "#{pane_pid}"],
        check=False
    )

    if result.returncode != 0:
        return None

    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


@mcp.tool()
def run_detached(name: str, command: str) -> Dict[str, Any]:
    """Create session, run command, return immediately (non-blocking)

    Args:
        name: Session name
        command: Command to run

    Returns:
        Session info for later checking
    """
    # Create detached session with command
    result = session_new(name, command, detached=True)

    if not result["success"]:
        return result

    # Give command a moment to start
    time.sleep(0.5)

    return {
        "success": True,
        "name": name,
        "command": command,
        "message": "Session created and running in background. Use capture_pane or wait_and_capture to check progress."
    }


@mcp.tool()
def wait_and_capture(name: str, timeout: int = 30, poll_interval: int = 2) -> Dict[str, Any]:
    """Poll session until command completes or timeout

    Args:
        name: Session name to monitor
        timeout: Max seconds to wait
        poll_interval: Seconds between checks

    Returns:
        Completion status, output, and exit code if available
    """
    start_time = time.time()
    last_output = ""

    while time.time() - start_time < timeout:
        # Check if session still exists
        if not session_exists(name):
            return {
                "completed": True,
                "output": last_output,
                "message": "Session no longer exists (command completed or killed)"
            }

        # Capture current output
        output = capture_pane(f"{name}:0.0", lines=100)
        last_output = output

        # Check if process is still running
        pid = get_pane_pid(f"{name}:0.0")
        if pid:
            # Check if PID still exists
            try:
                os.kill(pid, 0)  # Signal 0 just checks existence
                # Process still running
                time.sleep(poll_interval)
                continue
            except OSError:
                # Process finished
                return {
                    "completed": True,
                    "output": output,
                    "pid": pid,
                    "message": "Process completed"
                }
        else:
            # No PID means likely finished
            time.sleep(poll_interval)

    # Timeout reached
    return {
        "completed": False,
        "output": last_output,
        "message": f"Timeout reached after {timeout}s. Session still running.",
        "timeout": timeout
    }


if __name__ == "__main__":
    # Run MCP server
    mcp.run()
