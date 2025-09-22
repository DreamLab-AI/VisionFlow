# Agent Technical Briefing

This document is intended for AI agents operating within this environment. It outlines your mission, architecture, and operational directives.

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
- **Automated Setup:** The container runs an automated setup process on startup that configures Claude CLI, sets up workspace projects, and verifies all MCP services.

### 3.2. Central Configuration
- **`/workspace/.mcp.json`:** This is your primary service registry. It defines all available MCP servers and how to execute them.
- **`/app/core-assets/`:** This directory contains the source-of-truth for all configurations and scripts.

### 3.3. MCP Services Ecosystem

Your capabilities are defined by the MCP tools available in your container. These tools fall into two categories:

-   **Direct Tools:** These are self-contained command-line tools that run directly within your container.
    -   `imagemagick-mcp`: For all image manipulation tasks.
    -   `kicad-mcp`: For electronic design automation (EDA).
    -   `ngspice-mcp`: For circuit simulation.

-   **Bridge Tools:** These tools connect to services running in the `gui-tools-container`.
    -   `blender-mcp`: Your interface to the Blender 3D application.
    -   `qgis-mcp`: Your interface to the QGIS geospatial application.
    -   `pbr-generator-mcp`: Your interface to the PBR texture generation service.

## 4. Your Operational Directives

1.  **Workspace is Key:** Always operate within the `/workspace` directory. The automated setup ensures it's properly initialized.
2.  **Consult the Tool Reference:** The **[MCP Tools API Reference](../reference/02-mcp-tools-api.md)** is your primary reference for understanding the capabilities and parameters of each MCP tool.
3.  **Respect the Bridge:** When using bridge tools (`blender-mcp`, `qgis-mcp`, `pbr-generator-mcp`), remember you are communicating with a remote application. Commands may take longer to execute.
4.  **Diagnose Failures:**
    *   If a **direct tool** fails, check your command's syntax and parameters.
    *   If a **bridge tool** fails, the issue is likely in the `gui-tools-container`. The service might be down or there could be a network issue. Use `multi-agent status` to check service health.
5.  **Leverage the Full Toolchain:** You have a powerful suite of EDA, 2D/3D, and geospatial tools. Analyze the user's request to determine the optimal combination of MCP services to achieve the goal.

## 5. Available Helper Commands

From within the container, you have access to these helper commands:

- `multi-agent status` - Check setup and service status
- `multi-agent health` - Run comprehensive health check
- `multi-agent services` - Show supervisor service status
- `multi-agent test-mcp` - Test MCP TCP connection
- `mcp-status` - Quick status of all MCP services
- `mcp-restart` - Restart all MCP services
- `setup-status` - Check automated setup status
- `setup-logs` - View automated setup logs

## 6. Authentication & Access

- **Claude Authentication:** The container mounts Claude credentials from the host at `~/.claude`. This provides persistent authentication across container restarts.
- **Service Authentication:** MCP services use token-based authentication. The tokens are configured via environment variables and managed by the security layer.

## 7. Best Practices

1. **Error Handling:** Always check tool responses for errors and provide meaningful feedback to users.
2. **Resource Awareness:** Be mindful that GUI tools consume more resources. Batch operations when possible.
3. **Security:** Never expose authentication tokens or attempt to bypass security measures.
4. **Logging:** Important operations are logged to `/app/mcp-logs/`. Check logs when debugging issues.
5. **Performance:** Use the TCP server (port 9500) for high-frequency operations, WebSocket bridge (port 3002) for interactive sessions.

Remember: You are operating in a sophisticated, security-hardened environment designed for enterprise-grade AI agent deployments. Use the tools responsibly and efficiently to deliver maximum value to the user.