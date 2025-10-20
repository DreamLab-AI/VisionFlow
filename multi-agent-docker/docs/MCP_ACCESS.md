# Accessing MCP Servers in Docker Container

## Problem

MCP servers running inside the Docker container are not visible to Claude Code running on the host because MCP uses stdio (stdin/stdout) communication by default, which doesn't cross container boundaries.

## Solutions

### Option 1: Use Claude Code Inside the Container (Recommended)

SSH into the container and run Claude Code from inside where MCP servers are configured:

```bash
# SSH into container
ssh -p 2222 devuser@localhost
# Password: turboflow

# Once inside, Claude Code will see all 8 MCP servers:
# - web-summary
# - qgis
# - blender
# - imagemagick
# - kicad
# - ngspice
# - pbr-rendering
# - playwright

# Run Claude Code
claude
```

**MCP Config Location Inside Container:**
`/home/devuser/.config/claude/mcp_settings.json`

### Option 2: Expose MCP Servers via TCP

Modify MCP servers to listen on TCP ports and use SSH tunneling:

#### Step 1: Start MCP TCP Gateway in Container

```bash
docker exec -it agentic-workstation bash
cd /home/devuser/mcp-infrastructure/servers
node mcp-gateway.js
```

#### Step 2: Create SSH Tunnels on Host

```bash
# Forward each MCP port
ssh -p 2222 -L 9510:localhost:9510 devuser@localhost  # web-summary
ssh -p 2222 -L 9511:localhost:9511 devuser@localhost  # imagemagick
ssh -p 2222 -L 9512:localhost:9512 devuser@localhost  # playwright
# ... etc
```

#### Step 3: Configure Host Claude Code

Add to `~/.config/claude/mcp_settings.json` on HOST:

```json
{
  "mcpServers": {
    "web-summary-docker": {
      "command": "nc",
      "args": ["localhost", "9510"]
    },
    "imagemagick-docker": {
      "command": "nc",
      "args": ["localhost", "9511"]
    },
    "playwright-docker": {
      "command": "nc",
      "args": ["localhost", "9512"]
    }
  }
}
```

### Option 3: Use code-server (Web-based VSCode)

Access Claude Code via web browser through code-server:

```bash
# Access code-server at:
http://localhost:8080

# MCP servers will be available inside the web IDE
```

**Note:** code-server may need additional configuration for Claude Code extension.

## Current MCP Servers in Container

All configured in `/home/devuser/.config/claude/mcp_settings.json`:

1. **web-summary** - Web content and YouTube summarization with Z.AI
2. **qgis** - Geographic information system operations
3. **blender** - 3D modeling and rendering
4. **imagemagick** - Image manipulation (always-on)
5. **kicad** - Circuit design
6. **ngspice** - Circuit simulation
7. **pbr-rendering** - PBR material generation
8. **playwright** - Browser automation (always-on)

## Recommended Workflow

**For Development:** Use SSH to container and run Claude Code inside
```bash
ssh -p 2222 devuser@localhost
claude
```

**For GUI Access:** Use VNC to container desktop
```bash
# VNC Viewer: localhost:5901
# Password: turboflow
# Then run Claude Code in terminal
```

**For Remote Access:** Use code-server web interface
```bash
# Browser: http://localhost:8080
```

## Architecture

```
Host Machine                   Docker Container (agentic-workstation)
├── Claude Code (host)         ├── Claude Code (container) ← Can see MCP servers
├── SSH client                 ├── SSH server (port 2222)
└── VNC client                 ├── VNC server (port 5901)
                               ├── code-server (port 8080)
                               └── MCP Servers (stdio)
                                   ├── web-summary
                                   ├── imagemagick
                                   ├── playwright
                                   └── ...
```

## Verification

Check MCP servers are available inside container:

```bash
docker exec agentic-workstation bash -c \
  "cat /home/devuser/.config/claude/mcp_settings.json | jq -r '.mcpServers | keys[]'"
```

Should output:
```
web-summary
qgis
blender
imagemagick
kicad
ngspice
pbr-rendering
playwright
```
