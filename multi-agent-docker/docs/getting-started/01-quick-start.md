# Quick Start Guide

This guide provides the fastest path to getting the Multi-Agent Docker Environment up and running.

### Prerequisites

- [ ] Docker Engine & Docker Compose
- [ ] Git
- [ ] 8GB+ RAM
- [ ] 20GB+ disk space

### Step 1: Clone the Repository

Open your terminal and clone the project repository.

```bash
git clone <repository-url>
cd multi-agent-docker
```

### Step 2: Build and Start the Environment

The `multi-agent.sh` script is your primary tool for managing the environment.

First, build the Docker images. This may take some time on the first run.
```bash
./multi-agent.sh build
```

Next, start the services. This command starts both containers in the background and drops you into a shell inside the main `multi-agent-container`.
```bash
./multi-agent.sh start
```

### Step 3: Initialize the Workspace

Once you are inside the container's shell (`dev@multi-agent-container:/workspace$`), run the one-time setup script. This prepares your workspace by copying tools, installing dependencies, and verifying the environment.

```bash
/app/setup-workspace.sh
```

### Step 4: Verify the Setup

Use the `mcp-helper.sh` script to test that all components are working correctly.

```bash
# List all available tools
./mcp-helper.sh list-tools

# Run an automated test suite for all tools
./mcp-helper.sh test-all
```

### 🎉 Success!

If all tests pass, your environment is fully operational. You can confirm this by:

1.  **Checking Container Status**: In a new terminal on your host machine, run `./multi-agent.sh status`. Both `multi-agent-container` and `gui-tools-container` should show an `Up` status.
2.  **Accessing the GUI Tools**: Open a VNC client (e.g., RealVNC, TigerVNC) and connect to `localhost:5901`. You should see the XFCE desktop of the `gui-tools-container`.

### Next Steps

- Learn how to customize your setup in the **[Configuration Guide](./02-configuration.md)**.
- Explore what you can do with the tools in the **[Using MCP Tools Guide](../guides/02-using-mcp-tools.md)**.