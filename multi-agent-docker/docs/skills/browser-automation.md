# Browser Automation Skills

Browser automation and debugging tools for Turbo Flow Claude, providing screenshot capture, web scraping, and debugging capabilities via MCP.

## Available Skills

| Skill | Purpose | MCP Server |
|-------|---------|------------|
| `playwright` | Browser automation, screenshots, web scraping | `mcp__playwright__*` |
| `host-webserver-debug` | Debug host servers from containers | `mcp__host-webserver-debug__*` |
| `chrome-devtools` | Chrome DevTools Protocol debugging | `mcp__chrome-devtools__*` |

## Requirements

- **Display**: Virtual display `:1` (VNC on port 5901)
- **Browser**: Chromium installed at `/usr/bin/chromium`
- **VNC Access**: Connect to `localhost:5901` (password: `turboflow`)

## Playwright Skill

### Installation

```bash
# Add MCP server
claude mcp add playwright -- node /home/devuser/.claude/skills/playwright/mcp-server/server.js

# Install dependencies
cd /home/devuser/.claude/skills/playwright/mcp-server
npm install
```

### Tools

| Tool | Description |
|------|-------------|
| `navigate` | Navigate browser to URL |
| `screenshot` | Capture screenshot (viewport or full page) |
| `click` | Click element by selector |
| `type` | Type text into input field |
| `evaluate` | Execute JavaScript in page context |
| `wait_for_selector` | Wait for element to appear |
| `get_content` | Get page HTML content |
| `get_url` | Get current URL and title |
| `close_browser` | Close browser instance |
| `health_check` | Check browser connection health |

### Usage Examples

```javascript
// Navigate and screenshot
mcp__playwright__navigate({ url: "https://example.com" })
mcp__playwright__screenshot({ filename: "page.png", fullPage: true })

// Form interaction
mcp__playwright__type({ selector: "#email", text: "user@example.com" })
mcp__playwright__click({ selector: "button[type=submit]" })

// JavaScript evaluation
mcp__playwright__evaluate({ script: "document.title" })
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY` | `:1` | X display for browser |
| `CHROMIUM_PATH` | `/usr/bin/chromium` | Chromium binary path |
| `PLAYWRIGHT_HEADLESS` | `false` | Headless mode |
| `SCREENSHOT_DIR` | `/tmp/playwright-screenshots` | Screenshot output |
| `PLAYWRIGHT_TIMEOUT` | `30000` | Default timeout (ms) |
| `VIEWPORT_WIDTH` | `1920` | Browser viewport width |
| `VIEWPORT_HEIGHT` | `1080` | Browser viewport height |

## Host Webserver Debug Skill

Bridges HTTPS to HTTP for debugging host development servers from within Docker containers.

### Installation

```bash
# Add MCP server
claude mcp add host-webserver-debug -- node /home/devuser/.claude/skills/host-webserver-debug/mcp-server/server.js

# Install dependencies
cd /home/devuser/.claude/skills/host-webserver-debug
npm install
```

### Tools

| Tool | Description |
|------|-------------|
| `bridge_start` | Start HTTPS bridge proxy |
| `bridge_status` | Check bridge status |
| `bridge_stop` | Stop bridge proxy |
| `screenshot` | Take screenshot via bridge |
| `debug_cors` | Analyse CORS headers |
| `health_check` | Check host server reachability |
| `get_host_ip` | Detect Docker gateway IP |

### Usage Examples

```javascript
// Start bridge to host dev server
mcp__host-webserver-debug__bridge_start({
  host_ip: "192.168.0.51",
  https_port: 3001,
  target_port: 3001
})

// Take screenshot
mcp__host-webserver-debug__screenshot({
  url: "https://localhost:3001",
  full_page: true
})

// Debug CORS
mcp__host-webserver-debug__debug_cors({
  url: "https://localhost:3001/api/data",
  origin: "https://localhost:3001"
})
```

## Chrome DevTools Skill

Direct access to Chrome DevTools Protocol for advanced debugging.

### Installation

```bash
# Add MCP server (uses npx, no local install needed)
claude mcp add chrome-devtools -- npx -y chrome-devtools-mcp@latest
```

### Tools

- `performance_start_trace` - Record performance trace
- `network_get_requests` - Get network requests
- `console_get_messages` - Get console logs/errors
- `dom_query_selector` - Query DOM elements
- `dom_get_computed_style` - Get computed styles
- `runtime_evaluate` - Execute JavaScript
- `page_screenshot` - Capture screenshot
- `coverage_start` / `coverage_stop` - Code coverage analysis

## Troubleshooting

### Display Issues

```bash
# Verify X display
DISPLAY=:1 xdpyinfo

# Check VNC status
sudo supervisorctl status xvnc
```

### Browser Launch Failures

```bash
# Test Chromium
DISPLAY=:1 chromium --no-sandbox --disable-gpu https://example.com

# Check Chromium version
chromium --version
```

### MCP Connection Issues

```bash
# List MCP servers
claude mcp list

# Test MCP server directly
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  DISPLAY=:1 node /home/devuser/.claude/skills/playwright/mcp-server/server.js
```

### Screenshot Permissions

```bash
# Ensure screenshot directory exists
mkdir -p /tmp/playwright-screenshots
chmod 755 /tmp/playwright-screenshots
```

## Visual Verification

Connect to VNC to see browser activity:

```bash
# Using VNC client
vncviewer localhost:5901
# Password: turboflow

# Or via web browser (if noVNC configured)
http://localhost:6080
```

## Integration with Other Skills

These browser skills work well with:

- **imagemagick** - Post-process screenshots
- **web-summary** - Summarise web content
- **perplexity** - Research context for URLs
- **agentic-qe** - Automated testing workflows
