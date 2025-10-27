# Desktop Environment Guide

Complete guide to using the integrated desktop environment in the Agentic Flow workstation for GUI-based development, browser automation, and visual debugging.

## Quick Start

Enable desktop environment:

```bash
# In .env file
ENABLE_DESKTOP=true
```

Access desktop:
- **noVNC (Browser)**: http://localhost:6901
- **VNC Client**: localhost:5901

## Table of Contents

1. [Enabling Desktop Environment](#1-enabling-desktop-environment)
2. [VNC and noVNC Access](#2-vnc-and-novnc-access)
3. [XFCE4 Desktop Features](#3-xfce4-desktop-features)
4. [Available GUI Applications](#4-available-gui-applications)
5. [Browser Automation with Playwright](#5-browser-automation-with-playwright)
6. [Code Server (VS Code in Browser)](#6-code-server-vs-code-in-browser)
7. [X11 Forwarding](#7-x11-forwarding)
8. [Troubleshooting Desktop Issues](#8-troubleshooting-desktop-issues)

---

## 1. Enabling Desktop Environment

### Configuration

Edit your `.env` file:

```bash
# Desktop environment (VNC/noVNC)
ENABLE_DESKTOP=true

# Optional: Also enable VS Code Server
ENABLE_CODE_SERVER=true
```

### Environment Variables

The desktop environment uses these variables:

```bash
DISPLAY=:1                    # X11 display number (default)
ENABLE_DESKTOP=true           # Enable desktop components
```

### Service Architecture

Desktop environment consists of:

1. **Xvnc** - X server with VNC capability (port 5901)
2. **D-Bus** - Inter-process communication
3. **XFCE4** - Lightweight desktop environment
4. **noVNC** - HTML5 VNC client (port 6901)

### Verify Desktop Services

Check desktop service status:

```bash
docker exec -it agentic-flow-cachyos supervisorctl status
```

Expected output:
```
desktop:dbus                     RUNNING   pid 123
desktop:novnc                    RUNNING   pid 456
desktop:xfce4                    RUNNING   pid 789
desktop:xvnc                     RUNNING   pid 234
```

---

## 2. VNC and noVNC Access

### noVNC (Browser Access)

**Recommended for most users** - no client installation required.

1. Open browser: http://localhost:6901
2. Click "Connect"
3. Full desktop in browser

**Features:**
- No VNC client installation needed
- Cross-platform (any modern browser)
- Clipboard integration
- Fullscreen mode available

### Native VNC Client Access

For better performance and features, use a native VNC client.

**Popular VNC Clients:**
- **TigerVNC** (Linux/Windows/macOS)
- **RealVNC Viewer** (all platforms)
- **TightVNC** (Windows)
- **Screens** (macOS)

**Connection Details:**
- Host: `localhost`
- Port: `5901`
- Display: `:1`
- Password: None (no authentication)

**Example with TigerVNC:**
```bash
vncviewer localhost:5901
```

### Port Configuration

Default ports exposed in docker-compose.yml:

```yaml
ports:
  - "5901:5901"   # VNC
  - "6901:6901"   # noVNC
```

Change ports if conflicts exist:
```yaml
ports:
  - "15901:5901"  # Custom VNC port
  - "16901:6901"  # Custom noVNC port
```

---

## 3. XFCE4 Desktop Features

### Desktop Environment

XFCE4 provides a lightweight, full-featured desktop:

- **Panel**: Top panel with application menu, launchers
- **File Manager**: Thunar for browsing workspace
- **Terminal**: XFCE Terminal with zsh
- **Settings**: Desktop preferences and appearance

### Key Shortcuts

```
Alt + F2              - Run application dialog
Alt + F4              - Close window
Ctrl + Alt + T        - Open terminal
Super (Windows) Key   - Application finder
```

### File Manager (Thunar)

Access your workspace:
- `/home/devuser/workspace` - Main development directory
- `/home/devuser/models` - Model cache
- `/home/devuser/.config` - Configuration files
- `/home/devuser/logs` - Service logs

### Terminal Access

Open terminal from desktop:
1. Click Applications menu → Terminal
2. Or use keyboard shortcut: `Ctrl + Alt + T`

Configured with:
- **Shell**: zsh with Oh-My-Zsh
- **Color scheme**: 256 colors
- **Features**: Git integration, syntax highlighting

### Clipboard Integration

Copy/paste between host and container:
- noVNC has clipboard sync in sidebar
- VNC clients vary (check client settings)

---

## 4. Available GUI Applications

### Chromium Browser

Pre-installed for web access and testing.

**Launch:**
```bash
chromium --no-sandbox &
```

**Features:**
- Software rendering (no GPU conflicts)
- Persistent profile: `/home/devuser/.config/chromium`
- Extensions supported
- DevTools available

**Auto-launch on startup:**
```bash
# In .env or docker-compose environment
CHROMIUM_STARTUP_URL=https://example.com
```

### Code Editor (VS Code Server)

Browser-based VS Code for in-container development.

**Enable:**
```bash
ENABLE_CODE_SERVER=true
```

**Access:**
- URL: http://localhost:8080
- No authentication by default
- Full VS Code features
- Opens to `/home/devuser/workspace`

**See:** [Code Server section](#6-code-server-vs-code-in-browser) for details.

### Playwright Browser (Chromium)

Headless browser automation with visual debugging.

**Visual Mode:**
```javascript
const browser = await chromium.launch({
  headless: false,
  args: ['--no-sandbox']
});
```

Browser opens in desktop environment for debugging.

**See:** [Browser Automation section](#5-browser-automation-with-playwright)

### Terminal Applications

GUI terminals available:
- **XFCE4 Terminal** - Lightweight, configurable
- **xterm** - Fallback terminal

### File Managers

- **Thunar** - XFCE default file manager
- **Command line**: `ranger`, `mc` (midnight commander)

---

## 5. Browser Automation with Playwright

### Overview

Playwright provides browser automation with two modes:

1. **Headless** - Fast, no display (default)
2. **Visual** - Shows browser in desktop (debugging)

### Headless Automation (Default)

Fast automation without GUI:

```javascript
const { chromium } = require('playwright');

const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox']
});

const page = await browser.newPage();
await page.goto('https://example.com');
```

### Visual Debugging Mode

See browser actions in real-time:

```javascript
const browser = await chromium.launch({
  headless: false,
  args: ['--no-sandbox']
});
```

Browser window appears in VNC desktop.

### MCP Integration

Playwright available as MCP server for AI agents.

**Configuration:** `~/.config/claude/mcp.json`
```json
{
  "mcpServers": {
    "playwright": {
      "command": "node",
      "args": ["/app/core-assets/scripts/playwright-mcp-proxy.js"]
    }
  }
}
```

**MCP Tools:**
- `playwright_navigate` - Navigate to URL
- `playwright_screenshot` - Capture screenshot
- `playwright_click` - Click elements
- `playwright_fill` - Fill forms
- `playwright_evaluate` - Execute JavaScript

### Testing Scripts

Test Playwright setup:

```bash
# Headless test
node /app/core-assets/scripts/playwright-demo.js

# Full stack test (including MCP)
bash /app/core-assets/scripts/test-playwright-stack.sh
```

### Resource Limits

Playwright configured with software rendering:

```javascript
args: [
  '--no-sandbox',
  '--disable-gpu',
  '--use-gl=swiftshader-webgl',
  '--enable-webgl-software-rendering',
  '--disable-dev-shm-usage'
]
```

Prevents GPU conflicts in containerized environment.

---

## 6. Code Server (VS Code in Browser)

### Overview

Full VS Code experience running in container, accessible via browser.

### Enabling Code Server

**Configuration:**
```bash
# .env
ENABLE_CODE_SERVER=true
```

**Port:**
```yaml
# docker-compose.yml
ports:
  - "8080:8080"   # code-server
```

### Accessing Code Server

1. Open browser: http://localhost:8080
2. No password required (development setup)
3. Workspace: `/home/devuser/workspace`

### Features

**Available:**
- Full editor features
- Integrated terminal
- Extension marketplace
- Git integration
- Debugging support
- Multi-file editing

**Not Available:**
- Some native extensions (platform-dependent)
- System-level debugging

### Configuration

**User settings:** `/home/devuser/.config/code-server/config.yaml`

```yaml
bind-addr: 0.0.0.0:8080
auth: none
cert: false
```

### Installing Extensions

```bash
# Inside container
code-server --install-extension ms-python.python
```

Or use Extensions panel in browser interface.

### Performance Tips

1. **Disable heavy extensions** for better performance
2. **Use workspace sync** sparingly
3. **Terminal preference**: Use native terminal for intensive tasks

### Security Considerations

**Development setup:**
- No authentication (localhost only)
- Bind to 0.0.0.0 (docker networking)

**Production setup:**
```yaml
# Add authentication
auth: password
password: your-secure-password
```

---

## 7. X11 Forwarding

### Overview

X11 forwarding allows GUI apps in container to display on host.

### Host Configuration

**Mount X11 socket:**

Already configured in docker-compose.yml:
```yaml
volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix:rw
environment:
  DISPLAY: ${DISPLAY:-:0}
```

### Using X11 Forwarding

**On Linux host:**

1. Allow container to access X server:
```bash
xhost +local:docker
```

2. Applications display on host:
```bash
docker exec -it agentic-flow-cachyos chromium --no-sandbox
```

Browser appears on host display.

### Display Configuration

**Container display:**
- `:1` - VNC desktop (internal)
- `:0` - Host display (X11 forwarding)

**Switch displays:**
```bash
# Use VNC desktop
export DISPLAY=:1

# Use host display (X11 forwarding)
export DISPLAY=:0
```

### X11 vs VNC

**X11 Forwarding:**
- ✓ Native performance
- ✓ Host window manager
- ✗ Linux host only
- ✗ Requires xhost configuration

**VNC Desktop:**
- ✓ Cross-platform
- ✓ Full desktop environment
- ✓ No host configuration
- ✗ Slight latency

### Security Note

X11 forwarding opens security risks:

```bash
# Restrict to container
xhost +local:docker

# Remove access when done
xhost -local:docker
```

---

## 8. Troubleshooting Desktop Issues

### Desktop Not Starting

**Check service status:**
```bash
docker exec -it agentic-flow-cachyos supervisorctl status desktop:
```

**Common issues:**

1. **ENABLE_DESKTOP not set**
   ```bash
   # Check environment
   docker exec -it agentic-flow-cachyos env | grep ENABLE_DESKTOP
   # Should show: ENABLE_DESKTOP=true
   ```

2. **Services not running**
   ```bash
   # Restart desktop group
   docker exec -it agentic-flow-cachyos supervisorctl restart desktop:
   ```

3. **Port conflicts**
   ```bash
   # Check if ports already in use
   netstat -an | grep -E '5901|6901'
   # Change ports in docker-compose.yml if needed
   ```

### VNC Connection Issues

**Cannot connect to VNC:**

1. **Verify port mapping:**
   ```bash
   docker port agentic-flow-cachyos
   ```

2. **Check firewall:**
   ```bash
   # Linux
   sudo ufw allow 5901
   sudo ufw allow 6901
   ```

3. **Test locally:**
   ```bash
   docker exec -it agentic-flow-cachyos netstat -tlnp | grep -E '5901|6901'
   ```

**Black screen in VNC:**

Desktop may be starting up. Wait 30 seconds and refresh.

If persists:
```bash
# Check Xvnc logs
docker exec -it agentic-flow-cachyos cat /home/devuser/logs/xvnc.log

# Restart desktop services
docker exec -it agentic-flow-cachyos supervisorctl restart desktop:
```

### noVNC Web Client Issues

**Blank page:**
1. Check browser console for errors
2. Verify noVNC service:
   ```bash
   docker exec -it agentic-flow-cachyos supervisorctl status desktop:novnc
   ```
3. Check logs:
   ```bash
   docker exec -it agentic-flow-cachyos cat /home/devuser/logs/novnc.log
   ```

**Connection refused:**
- Xvnc may not be running
- Restart desktop services

### Performance Issues

**Slow desktop response:**

1. **Reduce resolution:**
   Edit `/etc/supervisord.conf`:
   ```ini
   [program:xvnc]
   command=/usr/bin/Xvnc :1 -geometry 1280x720 -depth 24 ...
   ```

2. **Reduce color depth:**
   ```ini
   -depth 16  # Instead of -depth 24
   ```

3. **Close unused applications**

4. **Use native VNC client** instead of noVNC

**Browser rendering issues:**

Chromium uses software rendering to avoid GPU conflicts:
```bash
# Already configured in launch script
--disable-gpu
--use-gl=swiftshader-webgl
```

### Clipboard Not Working

**noVNC:**
- Click sidebar icon (clipboard/keyboard)
- Paste into text area
- Copies to container clipboard

**VNC client:**
- Check client clipboard settings
- Some clients don't support bidirectional sync

### Font Rendering Issues

**Blurry fonts:**

1. **Increase VNC color depth:**
   ```ini
   -depth 24
   ```

2. **Adjust DPI:**
   ```bash
   xfce4-appearance-settings
   # Adjust DPI in Fonts tab
   ```

### Application Crashes

**Chromium crashes:**

Common with GPU access issues. Solution:
```bash
# Force software rendering
chromium --no-sandbox --disable-gpu --use-gl=swiftshader-webgl
```

**Check logs:**
```bash
docker exec -it agentic-flow-cachyos cat /home/devuser/logs/xfce4.log
```

### Display Configuration Issues

**Wrong DISPLAY variable:**

```bash
# Check current display
echo $DISPLAY

# For VNC desktop
export DISPLAY=:1

# For X11 forwarding
export DISPLAY=:0
```

**Multiple displays conflict:**

Container uses `:1` by default. If host uses `:1`, change container:

```yaml
environment:
  DISPLAY: :2
```

And update supervisord.conf:
```ini
command=/usr/bin/Xvnc :2 -geometry 1920x1080 ...
```

### Service Logs

Access all desktop service logs:

```bash
# View logs in container
docker exec -it agentic-flow-cachyos ls /home/devuser/logs/

# Tail specific log
docker exec -it agentic-flow-cachyos tail -f /home/devuser/logs/xvnc.log
docker exec -it agentic-flow-cachyos tail -f /home/devuser/logs/xfce4.log
docker exec -it agentic-flow-cachyos tail -f /home/devuser/logs/novnc.log
docker exec -it agentic-flow-cachyos tail -f /home/devuser/logs/dbus.log
```

### Complete Desktop Reset

If all else fails:

```bash
# Stop container
docker-compose down

# Remove config volume (if configured)
docker volume rm multi-agent-docker_config-persist

# Restart
docker-compose up -d

# Monitor startup
docker logs -f agentic-flow-cachyos
```

### Getting Help

When reporting issues, include:

1. **Service status:**
   ```bash
   docker exec -it agentic-flow-cachyos supervisorctl status
   ```

2. **Recent logs:**
   ```bash
   docker exec -it agentic-flow-cachyos tail -100 /home/devuser/logs/xvnc.log
   ```

3. **Environment:**
   ```bash
   docker exec -it agentic-flow-cachyos env | grep -E 'DISPLAY|ENABLE'
   ```

4. **Port bindings:**
   ```bash
   docker port agentic-flow-cachyos
   ```

---

## Additional Resources

- **Playwright Documentation**: https://playwright.dev
- **XFCE Documentation**: https://docs.xfce.org
- **noVNC Project**: https://novnc.com
- **Code Server**: https://github.com/coder/code-server

## Related Guides

- [Getting Started](../getting-started/README.md) - Initial setup
- [Configuration](../CONFIGURATION.md) - Environment variables
- [Troubleshooting](../TROUBLESHOOTING.md) - General issues
- [MCP Tools](../../minimaldocs/MCP_TOOLS.md) - MCP server setup
