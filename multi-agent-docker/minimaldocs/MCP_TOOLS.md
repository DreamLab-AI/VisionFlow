# MCP Tools Reference

## Overview

The CachyOS Docker environment supports Model Context Protocol (MCP) tools via stdio communication. Tools spawn on-demand and communicate through standard input/output streams.

## Available Tools

### Claude Flow
**Package**: `claude-flow`
**Category**: workflows
**Description**: Claude Flow MCP integration for agentic workflows

```json
{
  "command": "npx",
  "args": ["-y", "claude-flow", "mcp", "start"],
  "type": "stdio"
}
```

#### Capabilities
- Workflow orchestration
- Agent coordination
- Task decomposition
- Result aggregation

#### Example Usage
```javascript
// Spawn worker with claude-flow tool
const session = await createSession({
  tools: ["claude-flow"]
});

// Execute workflow
await session.execute({
  tool: "claude-flow",
  action: "workflow.create",
  params: {
    name: "data-pipeline",
    steps: [...]
  }
});
```

---

### Context7
**Package**: `@upstash/context7-mcp`
**Category**: documentation
**Description**: Up-to-date code documentation - fetches current API docs and examples

```json
{
  "command": "npx",
  "args": ["-y", "@upstash/context7-mcp"],
  "type": "stdio",
  "env": {
    "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"
  }
}
```

#### Capabilities
- API documentation lookup
- Code example retrieval
- Framework reference
- Package documentation

#### Configuration
```bash
# Required environment variable
export CONTEXT7_API_KEY=your-key-here
```

#### Example Usage
```bash
# Query documentation
echo '{"jsonrpc":"2.0","method":"docs/search","params":{"query":"react hooks"},"id":1}' | \
  npx -y @upstash/context7-mcp
```

---

### Playwright
**Package**: `@modelcontextprotocol/server-playwright`
**Category**: automation
**Description**: Browser automation via Playwright

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-playwright"],
  "type": "stdio",
  "env": {
    "DISPLAY": ":1"
  }
}
```

#### Capabilities
- Web page navigation
- Element interaction
- Screenshot capture
- PDF generation
- Form automation
- Multi-browser support (Chromium, Firefox, WebKit)

#### Example Usage
```javascript
// Navigate and capture screenshot
await session.execute({
  tool: "playwright",
  action: "navigate",
  params: {
    url: "https://example.com"
  }
});

await session.execute({
  tool: "playwright",
  action: "screenshot",
  params: {
    path: "/tmp/screenshot.png",
    fullPage: true
  }
});
```

#### Requirements
- Desktop environment with DISPLAY=:1 (for headed mode)
- Or run in headless mode (default)

---

### Filesystem
**Package**: `@modelcontextprotocol/server-filesystem`
**Category**: filesystem
**Description**: Read and write files in workspace

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/devuser/workspace"],
  "type": "stdio"
}
```

#### Capabilities
- Read files
- Write files
- List directories
- Create directories
- Delete files
- File metadata

#### Security
- Restricted to workspace directory (`/home/devuser/workspace`)
- Cannot access files outside workspace
- No symlink traversal outside workspace

#### Example Usage
```javascript
// Read file
await session.execute({
  tool: "filesystem",
  action: "read",
  params: {
    path: "package.json"
  }
});

// Write file
await session.execute({
  tool: "filesystem",
  action: "write",
  params: {
    path: "output.txt",
    content: "Hello, World!"
  }
});

// List directory
await session.execute({
  tool: "filesystem",
  action: "list",
  params: {
    path: "src/"
  }
});
```

---

### Git
**Package**: `@modelcontextprotocol/server-git`
**Category**: filesystem
**Description**: Git operations

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-git"],
  "type": "stdio"
}
```

#### Capabilities
- Repository initialization
- Commit creation
- Branch management
- Status checking
- Diff viewing
- Log browsing
- Remote operations

#### Example Usage
```javascript
// Initialize repository
await session.execute({
  tool: "git",
  action: "init",
  params: {
    path: "/home/devuser/workspace/myproject"
  }
});

// Create commit
await session.execute({
  tool: "git",
  action: "commit",
  params: {
    message: "Initial commit",
    files: ["src/", "package.json"]
  }
});

// Check status
await session.execute({
  tool: "git",
  action: "status"
});
```

---

### GitHub
**Package**: `@modelcontextprotocol/server-github`
**Category**: github
**Description**: GitHub API operations

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "type": "stdio",
  "env": {
    "GITHUB_TOKEN": "${GITHUB_TOKEN}"
  }
}
```

#### Capabilities
- Repository management
- Issue tracking
- Pull request operations
- Workflow runs
- Release management
- Gist operations

#### Configuration
```bash
# Required environment variable
export GITHUB_TOKEN=ghp_your-token-here
```

Token permissions needed:
- `repo`: Full repository access
- `workflow`: GitHub Actions workflow access
- `read:org`: Organization read access (for org repos)

#### Example Usage
```javascript
// Create issue
await session.execute({
  tool: "github",
  action: "issues.create",
  params: {
    owner: "username",
    repo: "repository",
    title: "Bug report",
    body: "Description of the issue"
  }
});

// List pull requests
await session.execute({
  tool: "github",
  action: "pulls.list",
  params: {
    owner: "username",
    repo: "repository",
    state: "open"
  }
});

// Trigger workflow
await session.execute({
  tool: "github",
  action: "actions.createWorkflowDispatch",
  params: {
    owner: "username",
    repo: "repository",
    workflow_id: "deploy.yml",
    ref: "main"
  }
});
```

---

### Fetch
**Package**: `@modelcontextprotocol/server-fetch`
**Category**: web
**Description**: Fetch web content

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-fetch"],
  "type": "stdio"
}
```

#### Capabilities
- HTTP GET requests
- HTTP POST requests
- Header management
- Response parsing
- Error handling
- Timeout configuration

#### Example Usage
```javascript
// GET request
await session.execute({
  tool: "fetch",
  action: "get",
  params: {
    url: "https://api.example.com/data",
    headers: {
      "Authorization": "Bearer token"
    }
  }
});

// POST request
await session.execute({
  tool: "fetch",
  action: "post",
  params: {
    url: "https://api.example.com/submit",
    body: JSON.stringify({ key: "value" }),
    headers: {
      "Content-Type": "application/json"
    }
  }
});
```

---

### Brave Search
**Package**: `@modelcontextprotocol/server-brave-search`
**Category**: web
**Description**: Web search via Brave API

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-brave-search"],
  "type": "stdio",
  "env": {
    "BRAVE_API_KEY": "${BRAVE_API_KEY}"
  }
}
```

#### Capabilities
- Web search
- News search
- Safe search filtering
- Result ranking
- Snippet extraction

#### Configuration
```bash
# Required environment variable
export BRAVE_API_KEY=your-key-here
```

Get API key: https://brave.com/search/api/

#### Example Usage
```javascript
// Web search
await session.execute({
  tool: "brave-search",
  action: "search",
  params: {
    query: "machine learning tutorials",
    count: 10
  }
});

// News search
await session.execute({
  tool: "brave-search",
  action: "news",
  params: {
    query: "latest AI developments",
    freshness: "24h"
  }
});
```

---

### Web Summary
**Package**: Custom Python MCP server
**Category**: web
**Description**: Fetches and summarizes web content, including YouTube videos, with Z.AI semantic topic matching

```json
{
  "command": "/opt/venv/bin/python3",
  "args": ["-u", "/app/core-assets/scripts/web-summary-mcp-server.py"],
  "type": "stdio",
  "env": {
    "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
    "ZAI_CONTAINER_URL": "http://claude-zai-service:9600"
  }
}
```

#### Capabilities
- Web page summarization
- YouTube video transcript extraction and summarization
- Semantic topic matching via Z.AI
- Logseq-formatted output with automatic [[topic links]]
- Supports any URL accessible via HTTP/HTTPS

#### Dependencies
- **Z.AI Service**: Claude-zai container must be running
- **Google API Key**: For YouTube transcript access
- **Topics Database**: Pre-configured topic list in `/app/core-assets/config/topics.json`

#### Configuration
```bash
# Required environment variables
export GOOGLE_API_KEY=AIza...      # Google API key
export ZAI_API_KEY=your-zai-key    # Z.AI API key (for container)
```

Get API keys:
- Google: https://console.cloud.google.com/apis/credentials
- Z.AI: https://z.ai/

#### Example Usage
```javascript
// Summarize web page
await session.execute({
  tool: "web-summary",
  action: "summarize_url",
  params: {
    url: "https://www.gemini.com"
  }
});

// Summarize YouTube video
await session.execute({
  tool: "web-summary",
  action: "summarize_url",
  params: {
    url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  }
});

// Result format (Logseq-compatible markdown)
{
  "title": "Page Title",
  "url": "https://...",
  "summary": "Concise summary...",
  "topics": ["[[AI]]", "[[Machine Learning]]"],
  "formatted": "## Page Title\n\n**Summary:** ...\n\n**Topics:** [[AI]], [[Machine Learning]]\n\n**Source:** https://..."
}
```

#### How It Works
1. Fetches URL content (web page or YouTube transcript)
2. Uses Gemini API to generate concise summary
3. Sends summary to Z.AI service for semantic topic extraction
4. Matches extracted topics against permitted topics database
5. Returns Logseq-formatted markdown with [[topic links]]

#### Troubleshooting
```bash
# Check Z.AI service health
curl http://localhost:9600/health

# Test web-summary tool manually
docker exec -it agentic-flow-cachyos zsh
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"web-summary.summarize_url","arguments":{"url":"https://example.com"}}}' | \
  /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py

# Check topics database
docker exec agentic-flow-cachyos cat /app/core-assets/config/topics.json
```

---

## Adding Custom Tools

### 1. Create Tool Package

Create an MCP-compliant npm package:

```javascript
// my-tool-mcp/index.js
#!/usr/bin/env node
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  const request = JSON.parse(line);

  // Process request
  const response = {
    jsonrpc: "2.0",
    id: request.id,
    result: {
      // Your result here
    }
  };

  console.log(JSON.stringify(response));
});
```

### 2. Add to MCP Configuration

Edit `config/mcp.json`:

```json
{
  "mcpServers": {
    "my-tool": {
      "command": "npx",
      "args": ["-y", "@myorg/my-tool-mcp"],
      "type": "stdio",
      "description": "My custom MCP tool",
      "env": {
        "MY_TOOL_CONFIG": "${MY_TOOL_CONFIG}"
      }
    }
  },
  "toolCategories": {
    "custom": ["my-tool"]
  }
}
```

### 3. Test Tool

```bash
# Manual test
echo '{"jsonrpc":"2.0","method":"test","params":{},"id":1}' | \
  npx -y @myorg/my-tool-mcp

# Container test
docker exec agentic-flow-cachyos npx -y @myorg/my-tool-mcp
```

### 4. Use in Sessions

```javascript
const session = await createSession({
  tools: ["my-tool"]
});

await session.execute({
  tool: "my-tool",
  action: "test",
  params: {}
});
```

## Tool Categories

Tools are organized into logical categories:

```json
{
  "toolCategories": {
    "documentation": ["context7"],
    "automation": ["playwright"],
    "filesystem": ["filesystem", "git"],
    "web": ["fetch", "brave-search"],
    "github": ["github"],
    "workflows": ["claude-flow"]
  }
}
```

## Best Practices

### Tool Selection
- Only include tools needed for specific tasks
- Minimize tool count per session
- Group related tools together

### Environment Variables
- Store API keys in `.env` file
- Never commit API keys to version control
- Use environment variable substitution in `mcp.json`

### Error Handling
- Tools may fail to spawn (missing package, network issues)
- Tools may timeout (long-running operations)
- Tools may return errors (invalid params, API errors)
- Always check response status

### Resource Management
- Tools are automatically cleaned up when session ends
- Set appropriate session timeouts
- Monitor tool resource usage
- Kill stale sessions periodically

### Security
- Restrict filesystem tool to workspace only
- Validate tool inputs before execution
- Use minimal permission API tokens
- Audit tool access logs

## Troubleshooting

### Tool Not Found

```bash
# Verify package exists
docker exec agentic-flow-cachyos npm search <package-name>

# Test installation
docker exec agentic-flow-cachyos npx -y <package-name> --help
```

### Tool Fails to Start

Check logs:
```bash
docker exec agentic-flow-cachyos cat /home/devuser/logs/management-api.log
```

Verify configuration:
```bash
docker exec agentic-flow-cachyos jq '.mcpServers' /home/devuser/.config/claude/mcp.json
```

### Tool Timeout

Increase timeout in session configuration:
```javascript
const session = await createSession({
  tools: ["playwright"],
  timeout: 60000  // 60 seconds
});
```

### Environment Variable Not Set

Verify in container:
```bash
docker exec agentic-flow-cachyos env | grep GITHUB_TOKEN
```

Check docker-compose.yml has environment variable passed through.

### Permission Denied

Filesystem tool is restricted to workspace:
```javascript
// This works
await session.execute({
  tool: "filesystem",
  action: "read",
  params: { path: "file.txt" }  // Relative to workspace
});

// This fails
await session.execute({
  tool: "filesystem",
  action: "read",
  params: { path: "/etc/passwd" }  // Outside workspace
});
```

## Performance Tips

### Tool Reuse
Keep sessions alive and reuse for multiple operations:
```javascript
const session = await createSession({ tools: ["git", "filesystem"] });

// Multiple operations in same session
await session.execute({ tool: "git", action: "status" });
await session.execute({ tool: "filesystem", action: "read", params: {...} });
await session.execute({ tool: "git", action: "commit", params: {...} });
```

### Parallel Execution
Create multiple sessions for parallel tool execution:
```javascript
const sessions = await Promise.all([
  createSession({ tools: ["playwright"] }),
  createSession({ tools: ["fetch"] }),
  createSession({ tools: ["brave-search"] })
]);

const results = await Promise.all([
  sessions[0].execute({ tool: "playwright", action: "navigate", params: {...} }),
  sessions[1].execute({ tool: "fetch", action: "get", params: {...} }),
  sessions[2].execute({ tool: "brave-search", action: "search", params: {...} })
]);
```

### Lazy Loading
Only create sessions when needed:
```javascript
// Bad: Create all sessions upfront
const allSessions = await createAllSessions();

// Good: Create on demand
async function getToolSession(tool) {
  if (!sessionCache[tool]) {
    sessionCache[tool] = await createSession({ tools: [tool] });
  }
  return sessionCache[tool];
}
```
