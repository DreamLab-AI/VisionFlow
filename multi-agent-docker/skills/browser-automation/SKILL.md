---
name: browser-automation
description: >
  Unified browser automation meta-skill with progressive disclosure. Helps agents choose
  between agent-browser (AI snapshots), Playwright (full API + visual), Chrome CDP (live sessions),
  and host-webserver-debug (Docker bridge). Covers headless and VNC Display :1 desktop modes.
  Use when you need to automate browsers, scrape pages, test UIs, debug web apps, or interact
  with live browser sessions. Invoke for any browser-related task to get the optimal tool selection.
---

# Browser Automation

Unified decision framework for browser automation in this container. Five tools are available, each optimised for different scenarios. This skill helps you choose -- or combine -- them.

## When Not To Use

- If you already know which browser tool to use -- invoke that skill directly (browser, playwright, chrome-cdp, host-webserver-debug)
- For fetching page content without browser interaction -- use WebFetch, curl, web-summary, or gemini-url-context
- For API testing without a real browser -- use curl or httpx directly
- For building UI components -- use the daisyui or ui-ux-pro-max-skill skills instead

## Quick Decision

**What do you need?**

| Need | Best Tool | Invoke |
|------|-----------|--------|
| Fill a form, click buttons, scrape data | **agent-browser** | `agent-browser open <url>` |
| Screenshot, visual test, full Playwright API | **Playwright** | MCP `playwright` tools |
| Inspect a live Chromium session, logged-in pages | **Chrome CDP** | `scripts/cdp.mjs list` |
| Access host web server from Docker | **host-webserver-debug** | MCP `host-webserver-debug` tools |
| Read page content without interaction | **WebFetch** or `curl` | Direct tool call |

If unsure, start with **agent-browser** -- it has the smallest context footprint and covers 80% of use cases.

---

## Environment

This container has two browser execution modes:

### Headless Mode (Default)
- No display required
- All tools work: agent-browser, Playwright (headless), Chrome CDP (headless)
- Fastest execution, lowest resource usage
- Use for: scraping, form automation, API testing, data extraction

### VNC Display :1 (Visual Desktop)
- Full desktop visible via VNC on port 5901 (password: `turboflow`)
- Chromium renders visually -- you can see what the browser is doing
- Use for: visual regression testing, debugging CSS layouts, interactive debugging, screenshots of rendered state
- Connect: `vncviewer localhost:5901`

```bash
# Launch browser on VNC display
DISPLAY=:1 chromium --no-sandbox https://example.com &

# Launch headless
chromium --headless --no-sandbox --disable-gpu about:blank &
```

---

## Tool Details

### 1. agent-browser (Vercel)

**Best for**: Quick interactions, form filling, scraping, AI-friendly page understanding.

**Strengths**:
- AI-optimised accessibility snapshots (93% smaller than full DOM)
- Element refs (@e1, @e2) for reliable interaction
- Multi-session support for parallel work
- CDP connect mode for attaching to existing browsers
- Smallest context footprint of all tools

**Limitations**:
- No visual rendering pipeline (headless only by default)
- No screenshot comparison / visual regression
- No full Playwright API (no network interception, no tracing)

```bash
# Basic workflow
agent-browser open https://example.com
agent-browser snapshot -i          # interactive elements only
agent-browser fill @e2 "text"
agent-browser click @e3
agent-browser snapshot -i          # re-check after action

# Connect to existing CDP session
agent-browser connect 9222
agent-browser snapshot -i

# Isolated sessions for parallel agents
agent-browser --session agent1 open https://site-a.com
agent-browser --session agent2 open https://site-b.com
```

**MCP tools**: `browser/open`, `browser/snapshot`, `browser/click`, `browser/fill`, `browser/screenshot`, `browser/close`

---

### 2. Playwright

**Best for**: Visual testing, full browser control, complex multi-page flows, Display :1 rendering.

**Strengths**:
- Full Playwright API (network interception, tracing, HAR recording)
- Visual rendering on Display :1 (VNC accessible)
- Screenshot + full-page PDF capture
- CSS selector + XPath + text selectors
- Wait strategies (networkidle, domcontentloaded, selector)
- MCP server with structured tool interface

**Limitations**:
- Heavier than agent-browser (larger context per operation)
- Launches fresh browser instance (no live session reuse)
- Requires Display :1 for visual mode

```bash
# Add MCP server (if not already configured)
claude mcp add playwright -- node /home/devuser/.claude/skills/playwright/mcp-server/server.js
```

```javascript
// MCP tool usage
await navigate({ url: "https://example.com" })
await screenshot({ filename: "page.png", fullPage: true })
await click({ selector: "#submit" })
await type({ selector: "#email", text: "user@example.com" })
await wait_for_selector({ selector: ".loaded" })
await evaluate({ script: "document.querySelectorAll('.item').length" })
```

**Environment**:
| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY` | `:1` | X display for visual mode |
| `PLAYWRIGHT_HEADLESS` | `false` | Set `true` for headless |
| `CHROMIUM_PATH` | `/usr/bin/chromium` | Browser path |

---

### 3. Chrome CDP (Live Session)

**Best for**: Inspecting running Chromium, debugging live pages, logged-in session access, raw CDP commands.

**Strengths**:
- Connects to already-running Chromium (tabs you have open)
- Persistent per-tab daemons (no reconnection overhead)
- Handles 100+ tabs reliably
- Raw CDP command passthrough
- Works in both headless and visual modes
- No Puppeteer dependency, Node.js 22+ only

**Limitations**:
- Requires Chromium to be running with remote debugging enabled
- No built-in element refs like agent-browser
- Manual target selection via ID prefixes

```bash
# Start Chromium with CDP enabled
chromium --remote-debugging-port=9222 --no-sandbox --headless &
# Or on VNC: DISPLAY=:1 chromium --remote-debugging-port=9222 --no-sandbox &

# Use CDP
/home/devuser/.claude/skills/chrome-cdp/scripts/cdp.mjs list
/home/devuser/.claude/skills/chrome-cdp/scripts/cdp.mjs snap <target>
/home/devuser/.claude/skills/chrome-cdp/scripts/cdp.mjs eval <target> "document.title"
/home/devuser/.claude/skills/chrome-cdp/scripts/cdp.mjs click <target> "#button"
/home/devuser/.claude/skills/chrome-cdp/scripts/cdp.mjs shot <target>
```

---

### 4. host-webserver-debug

**Best for**: Accessing web servers running on the Docker host from inside the container.

**Strengths**:
- HTTPS bridge bypasses browser security restrictions
- Auto-detects Docker host gateway IP
- Solves CORS issues between container and host
- Self-signed certificate generation
- Supervisord-managed service

**Limitations**:
- Only useful for Docker host access scenarios
- Development certificates only (not for production)

```bash
# Start HTTPS bridge
node /opt/https-bridge/https-proxy.js

# Access host server via bridge
curl -sk https://localhost:3001
DISPLAY=:1 chromium --ignore-certificate-errors https://localhost:3001
```

**MCP tools**: `bridge_start`, `bridge_status`, `bridge_stop`, `screenshot`, `navigate`, `debug_cors`, `health_check`

---

## Combination Patterns

### Pattern 1: Scrape + Visual Verify

Use agent-browser for fast scraping, Playwright for visual verification.

```bash
# Fast scrape with agent-browser
agent-browser open https://example.com/products
agent-browser snapshot -i
agent-browser get text @e1  # product name
agent-browser get text @e2  # price

# Visual verify with Playwright
await navigate({ url: "https://example.com/products" })
await screenshot({ filename: "products.png", fullPage: true })
```

### Pattern 2: CDP Debug + agent-browser Automate

Use CDP to inspect live state, agent-browser to automate.

```bash
# Inspect via CDP
/home/devuser/.claude/skills/chrome-cdp/scripts/cdp.mjs list
/home/devuser/.claude/skills/chrome-cdp/scripts/cdp.mjs eval <target> "performance.timing.toJSON()"

# Connect agent-browser to same CDP session
agent-browser connect 9222
agent-browser snapshot -i
agent-browser click @e5
```

### Pattern 3: Host Debug + Visual Test

Use host-webserver-debug for bridge, Playwright for testing.

```bash
# Bridge to host server
node /opt/https-bridge/https-proxy.js &

# Test via Playwright on Display :1
await navigate({ url: "https://localhost:3001" })
await screenshot({ filename: "host-app.png" })
await click({ selector: "#login-btn" })
await wait_for_selector({ selector: ".dashboard" })
```

### Pattern 4: Parallel Multi-Agent Browser Work

Spawn multiple agents with isolated sessions for parallel scraping.

```bash
# Agent 1: scrape site A
agent-browser --session scraper1 open https://site-a.com
agent-browser --session scraper1 snapshot -i

# Agent 2: scrape site B (parallel)
agent-browser --session scraper2 open https://site-b.com
agent-browser --session scraper2 snapshot -i

# Agent 3: visual regression on Display :1 (parallel)
# Uses Playwright MCP tools
```

---

## Decision Tree

```
START: What browser task?
│
├─ "Read page content, no interaction"
│  └─ Use WebFetch or curl (no browser needed)
│
├─ "Fill forms, click buttons, scrape data"
│  ├─ Need visual rendering? → Playwright (Display :1)
│  └─ Headless OK? → agent-browser (fastest)
│
├─ "Take screenshots, visual testing"
│  ├─ Compare screenshots? → Playwright (full API)
│  └─ Quick screenshot? → agent-browser screenshot or CDP shot
│
├─ "Debug a live running page"
│  ├─ Page is on host machine? → host-webserver-debug + Playwright
│  ├─ Page in container Chromium? → Chrome CDP
│  └─ Need to see it visually? → VNC Display :1 + Chrome CDP
│
├─ "Execute JavaScript in page"
│  ├─ One-off eval? → Chrome CDP eval or agent-browser eval
│  └─ Complex JS with DOM manipulation? → Playwright evaluate
│
├─ "Network inspection, HAR, tracing"
│  └─ Playwright (only tool with full network API)
│
├─ "Interact with logged-in session"
│  └─ Chrome CDP (connects to existing tabs with cookies)
│
├─ "Multi-page parallel scraping"
│  └─ agent-browser with --session (isolated contexts)
│
└─ "Complex: multiple needs above"
   └─ Combine tools (see Combination Patterns above)
```

## Tool Availability Check

```bash
# Verify all tools are available
agent-browser --version          # Should show 0.21.2+
chromium --version               # Should show 146+
playwright --version             # Should show 1.57+
node --version                   # Should show 22+ (for CDP)
curl -s http://localhost:9222/json/version  # CDP if Chromium running
```

## Tips

1. **Start with agent-browser** unless you know you need something else -- smallest context, fastest
2. **Use Display :1 for debugging** -- seeing the browser helps diagnose layout issues
3. **CDP is for live sessions** -- if you need to inspect what the user is looking at, use CDP
4. **Playwright for testing** -- its wait strategies and screenshot comparison are unmatched
5. **Sessions are isolated** -- agent-browser sessions and Playwright contexts do not share cookies
6. **VNC password** is `turboflow` on port 5901
7. **Combine freely** -- these tools can coexist. Run CDP on port 9222 while agent-browser uses its own Chromium instance
8. **For swarm agents**: spawn each with their own `--session` to avoid collision
