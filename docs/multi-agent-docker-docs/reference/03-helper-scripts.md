# Reference: Helper Scripts

This document provides a reference for the key shell scripts that manage and interact with the environment.

### `multi-agent.sh`

-   **Location**: Project Root (Host Machine)
-   **Purpose**: The main entry point for managing the Docker Compose environment from your host machine.

#### Commands

| Command | Description |
| :--- | :--- |
| `build` | Builds or rebuilds the Docker images for both services. |
| `start` | Starts both containers in detached mode and opens a shell into the `multi-agent-container`. |
| `stop` | Stops and removes the running containers. |
| `restart` | A convenient shortcut for `stop` followed by `start`. |
| `status` | Displays the current status of the containers. |
| `logs [service]` | Tails the logs from all containers, or a specific one (e.g., `logs gui-tools-service`). |
| `shell` | Opens a new bash shell inside an already running `multi-agent-container`. |
| `cleanup` | Stops containers and removes all associated volumes, **deleting all data in the workspace**. |

#### Usage Examples

```bash
# Build the containers for the first time
./multi-agent.sh build

# Start the environment and enter the container
./multi-agent.sh start

# Check container status
./multi-agent.sh status

# View logs from a specific service
./multi-agent.sh logs gui-tools-service

# Open additional shell in running container
./multi-agent.sh shell
```

---

### `mcp-helper.sh`

-   **Location**: `/workspace/mcp-helper.sh` (Inside Container)
-   **Purpose**: A user-friendly wrapper for interacting with `claude-flow` and testing MCP tools.

#### Commands

| Command | Description |
| :--- | :--- |
| `list-tools` | Lists all MCP tools registered in `/workspace/.mcp.json`. |
| `test-tool <name>` | Checks if a specific tool is available and configured correctly. |
| `run-tool <name> '<json>'` | Executes a tool with the provided JSON payload. |
| `test-all` | Runs a series of automated tests to verify the environment's health. |
| `claude-instructions` | Displays a pre-formatted message to give to an AI agent on how to use the tools. |
| `test-imagemagick` | Runs specific tests for the ImageMagick MCP tool. |
| `test-kicad` | Runs specific tests for the KiCad MCP tool. |

#### Usage Examples

```bash
# List all available tools
./mcp-helper.sh list-tools

# Test a specific tool
./mcp-helper.sh test-tool blender-mcp

# Run a tool with JSON payload
./mcp-helper.sh run-tool imagemagick-mcp '{
  "method": "create",
  "params": {
    "width": 200,
    "height": 200,
    "color": "blue",
    "output": "blue_square.png"
  }
}'

# Run all tests
./mcp-helper.sh test-all
```

---

### `setup-workspace.sh`

-   **Location**: `/app/setup-workspace.sh` (Inside Container)
-   **Purpose**: Initializes a new or existing workspace. It is run automatically on first start but can be run manually to reset or update the workspace.

#### Actions Performed

1.  **Copies Core Assets**: Copies the latest versions of scripts and tools from `/app/core-assets` into `/workspace`.
2.  **Merges Configuration**: Intelligently merges the base `.mcp.json` config with any existing user configuration in the workspace.
3.  **Sets Permissions**: Ensures all scripts are executable.
4.  **Installs Dependencies**: Runs `npm install` to install `claude-flow` and other Node.js dependencies in the workspace.
5.  **Verifies Services**: Checks that all configured MCP services are responsive and reports their status.
6.  **Creates Completion Marker**: Creates a `.setup_completed` file to prevent the welcome message from showing on subsequent shell logins.

#### Options

-   `--force`: Overwrites existing files in the workspace with fresh copies from `/app/core-assets`.
-   `--quiet`: Suppresses most of the informational output.

#### Usage Examples

```bash
# Normal setup (preserves existing configurations)
/app/setup-workspace.sh

# Force update all files from core-assets
/app/setup-workspace.sh --force

# Quiet mode (minimal output)
/app/setup-workspace.sh --quiet
```

---

### `entrypoint-wrapper.sh`

-   **Location**: `/entrypoint-wrapper.sh` (Inside Container)
-   **Purpose**: This is the container's main entrypoint. It performs critical first-time setup tasks before starting the main services.

#### Actions Performed

1.  **Security Initialization**: Checks for default security tokens and warns the user.
2.  **Permission & Ownership Fixes**: Ensures the `dev` user owns its home directory and workspace directories.
3.  **Claude Authentication**: Sets up Claude credentials based on the mounted host file or environment variables.
4.  **Directory Creation**: Creates necessary directories like `/workspace/.swarm` and `/app/mcp-logs`.
5.  **Automatic Setup**: Triggers `setup-workspace.sh` and `automated-setup.sh` on the first run.
6.  **Starts Supervisord**: The final step is to execute `supervisord`, which launches and manages all background services (MCP TCP Server, WebSocket Bridge, etc.).

---

### `automated-setup.sh`

-   **Location**: `/app/scripts/automated-setup.sh` (Inside Container)
-   **Purpose**: Performs automated Claude CLI setup and workspace configuration on container startup.

#### Actions Performed

1.  **Claude CLI Setup**: Configures Claude CLI for the container user
2.  **Project Creation**: Creates and configures initial Claude projects
3.  **MCP Service Testing**: Validates all MCP services are operational
4.  **Health Check**: Performs comprehensive system health check

---

### `health-check.sh`

-   **Location**: `/app/scripts/health-check.sh` (Inside Container)
-   **Purpose**: Comprehensive health checking script for monitoring container status.

#### Checks Performed

1.  **Service Status**: Verifies supervisor services are running
2.  **Port Availability**: Tests MCP TCP and WebSocket ports
3.  **MCP Service Health**: Validates each MCP tool responds correctly
4.  **Resource Usage**: Reports CPU and memory utilization
5.  **Authentication**: Verifies security tokens are configured

#### Usage

```bash
# Run full health check
/app/scripts/health-check.sh

# Quick status check
multi-agent health
```

---

### Container Alias Commands

The container includes several helpful aliases defined in `/home/dev/.bashrc`:

| Alias | Command | Description |
| :--- | :--- | :--- |
| `multi-agent` | `/app/scripts/multi-agent-utils.sh` | Multi-purpose utility command |
| `mcp-status` | `supervisorctl status` | Show status of all supervisor services |
| `mcp-restart` | `supervisorctl restart all` | Restart all MCP services |
| `mcp-logs` | Function | View logs for specific MCP service |
| `setup-status` | `/app/scripts/setup-status.sh` | Check automated setup status |
| `setup-logs` | `tail -f /app/logs/automated-setup.log` | View setup logs |
| `rerun-setup` | `/app/scripts/automated-setup.sh` | Manually re-run automated setup |
| `blender-log` | `tail -f /app/mcp-logs/blender-mcp.log` | View Blender MCP logs |
| `tcp-log` | `tail -f /app/mcp-logs/mcp-tcp-server.log` | View TCP server logs |

---

### Security Scripts

Additional scripts for security management:

| Script | Purpose |
| :--- | :--- |
| `mcp-security-audit` | View security audit logs |
| `mcp-connections` | Show active connections to MCP services |
| `mcp-health` | Run health check endpoint test |
| `mcp-test-ws` | Test WebSocket connectivity |
| `mcp-test-tcp` | Test TCP server connectivity |

These scripts help monitor and maintain the security posture of the environment.