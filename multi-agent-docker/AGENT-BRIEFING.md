# Agent Technical Briefing

## 1. Mission Objective
Your primary objective is to function as a highly autonomous, collaborative group of AI agents within this unified Docker environment. Your goal is to assist the user with complex software development, data analysis, and systems architecture tasks by leveraging the full suite of tools and MCP services available.

## 2. Core Architecture: Dual-Container and Bridge-Based

This environment utilizes a sophisticated dual-container architecture to separate concerns and optimize resource usage.

-   **`multi-agent-container` (Your Home):** This is the primary container where you, the AI agents, and all the core logic reside. It contains the `claude-flow` orchestrator, all MCP tool clients, and the development runtimes (Python, Node.js, etc.). Your operations are confined to this container.

-   **`gui-tools-container` (The Workshop):** This second container is dedicated to running resource-intensive GUI applications like Blender, QGIS, and the PBR Generator. It is managed separately and you do not have direct access to it.

-   **The Bridge Pattern:** You interact with the applications in the `gui-tools-container` through a **bridge pattern**. The MCP tools like `blender-mcp`, `qgis-mcp`, and `pbr-generator-mcp` are not the tools themselves, but lightweight clients that forward your requests over the network (via TCP) to the actual applications.

## 3. Key Components & Workflow

### 3.1. Workspace Initialization
- **`setup-workspace.sh`:** This is the master script to prepare a new workspace. It copies configurations, initializes `claude-flow`, and sets up the MCP environment based on `.mcp.json`.

### 3.2. Central Configuration
- **`/app/core-assets/mcp.json`:** Defines all available MCP servers. This is the central registry for all services you can interact with.
- **`/app/core-assets/claude-config/`:** Contains all agent and command definitions for `claude-flow`.
- **`/app/core-assets/roo-config/`:** Contains modes and rules for the Roo agent.

### 3.3. MCP Services Ecosystem

Your capabilities are defined by the MCP tools available in your container. These tools fall into two categories:

-   **Core Tools (Always Available):** These tools are immediately available after container startup.
    -   `claude-flow`: AI orchestration with memory and GOAP planning
    -   `ruv-swarm`: Multi-agent coordination and swarm intelligence
    -   `flow-nexus`: Workflow orchestration and task management
    -   `playwright-mcp`: Browser automation (runs in GUI container but available immediately)

-   **GUI-Dependent Tools (Soft-Fail on Startup):** These tools connect to services running in the `gui-tools-container` and show timeout warnings during initialization.
    -   `blender-mcp`: 3D modeling and rendering via external Blender
    -   `qgis-mcp`: Geospatial analysis via external QGIS
    -   `pbr-generator-mcp`: PBR texture generation service
    -   `kicad-mcp`: Electronic design automation (EDA)
    -   `imagemagick-mcp`: Image manipulation tasks

**Startup Behavior**: GUI-dependent tools will show timeout warnings for 30-60 seconds while the GUI container initializes. This is expected and normal. Services auto-recover once ready.

## 4. Your Operational Directives

1.  **Workspace is Key:** Always operate within the `/workspace` directory. Ensure it has been initialized with `/app/setup-workspace.sh` at the start of a session.
2.  **Consult the Tool Reference:** The `TOOLS.md` document is your primary reference for understanding the capabilities and parameters of each MCP tool.
3.  **Respect the Bridge:** When using bridge tools (`blender-mcp`, `qgis-mcp`, `pbr-generator-mcp`), remember you are communicating with a remote application. Commands may take longer to execute.
4.  **Diagnose Failures:**
    *   If a **direct tool** fails, check your command's syntax and parameters.
    *   If a **bridge tool** fails, the issue is likely in the `gui-tools-container`. The service might be down or there could be a network issue. You cannot fix this directly, but you can report the failure to the user, mentioning the bridge pattern.
    *   **GUI Tool Timeout Warnings**: If you see timeout warnings for Blender, QGIS, KiCad, or ImageMagick during startup, this is **expected behavior**. These tools require 30-60 seconds to initialize in the GUI container. They will auto-recover once ready.
5.  **Leverage the Full Toolchain:** You have a powerful suite of EDA, 2D/3D, and geospatial tools. Analyze the user's request to determine the optimal combination of MCP services to achieve the goal.

## 5. Logging and Monitoring

### Unified Logging Architecture

All services now log to **stdout/stderr** for unified monitoring:

```bash
# View all container logs (from host)
docker logs multi-agent-container
docker logs -f multi-agent-container  # Follow in real-time

# View timestamped logs
docker logs -t multi-agent-container

# Last N lines
docker logs --tail 100 multi-agent-container
```

**Key Points**:
- No separate log files in `/app/mcp-logs/` (deprecated)
- Supervisord logs to `/dev/stdout` and `/dev/stderr`
- All MCP services redirect to stdout/stderr
- Use `docker logs` for unified log access

### Service Status Check

```bash
# Inside container - check all services
supervisorctl status

# View specific service output
supervisorctl tail -f mcp-tcp-server

# Restart a service
supervisorctl restart mcp-ws-bridge
```