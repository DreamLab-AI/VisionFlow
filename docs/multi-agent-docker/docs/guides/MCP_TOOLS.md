# MCP Tools Guide

## What is MCP (Model Context Protocol)

Model Context Protocol (MCP) is a standardized protocol that enables AI models to interact with external tools and services through a unified interface. MCP servers expose capabilities as tools, resources, and prompts that can be dynamically discovered and invoked by AI agents.

### Key Benefits

- **Standardization**: Consistent interface for all tools regardless of implementation
- **Isolation**: Each worker session spawns independent tool instances
- **On-Demand**: Tools only run when needed, minimizing resource usage
- **Extensibility**: Easy to add new capabilities through additional servers

### Architecture

In this system:
- All MCP servers communicate via stdio (standard input/output)
- No HTTP servers or exposed ports for MCP tools
- Each worker session spawns its own tool instances for isolation
- Tools spawn on-demand when tasks require them

## Available MCP Tools

### Documentation Tools

#### context7
```json
{
  "command": "npx",
  "args": ["-y", "@upstash/context7-mcp"],
  "env": {
    "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"
  }
}
```

**Purpose**: Fetches up-to-date code documentation, API references, and examples.

**Use Cases**:
- Getting current API documentation for libraries
- Finding code examples for specific frameworks
- Accessing latest package documentation

**Configuration**:
- Requires `CONTEXT7_API_KEY` environment variable
- Get API key from [Upstash Context7](https://upstash.com)

### Automation Tools

#### playwright
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-playwright"],
  "env": {
    "DISPLAY": ":1"
  }
}
```

**Purpose**: Browser automation for web scraping, testing, and interaction.

**Use Cases**:
- Automated web testing
- Scraping dynamic content requiring JavaScript
- Form filling and submission
- Screenshot capture
- PDF generation from web pages

**Configuration**:
- `DISPLAY` environment variable points to X11 display server
- Runs headless Chromium in container environment

**Example Tasks**:
- Navigate to URLs and extract content
- Click buttons and interact with page elements
- Wait for elements to load
- Execute JavaScript in page context

### Filesystem Tools

#### filesystem
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/devuser/workspace"],
}
```

**Purpose**: Read and write files in the workspace directory.

**Use Cases**:
- Reading file contents
- Writing new files
- Editing existing files
- Creating directories
- Listing directory contents

**Configuration**:
- Scoped to `/home/devuser/workspace` for security
- Cannot access files outside workspace directory

**Capabilities**:
- `read_file`: Read file contents
- `write_file`: Create or overwrite files
- `edit_file`: Modify existing files
- `list_directory`: List files and directories
- `create_directory`: Create new directories

#### git
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-git"]
}
```

**Purpose**: Git version control operations.

**Use Cases**:
- Checking repository status
- Creating commits
- Managing branches
- Viewing diffs and logs
- Resolving merge conflicts

**Capabilities**:
- `git_status`: Check working tree status
- `git_diff`: View changes
- `git_log`: View commit history
- `git_commit`: Create commits
- `git_add`: Stage changes
- `git_branch`: Manage branches
- `git_checkout`: Switch branches
- `git_pull`: Pull changes
- `git_push`: Push commits

### Web Tools

#### fetch
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-fetch"]
}
```

**Purpose**: Fetch web content via HTTP/HTTPS.

**Use Cases**:
- Downloading web pages
- Making API requests
- Fetching JSON/XML data
- Downloading files

**Capabilities**:
- GET requests
- POST requests with body
- Custom headers
- Response parsing

#### brave-search
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-brave-search"],
  "env": {
    "BRAVE_API_KEY": "${BRAVE_API_KEY}"
  }
}
```

**Purpose**: Web search via Brave Search API.

**Use Cases**:
- Finding information online
- Research tasks
- Discovering relevant URLs
- Getting current information

**Configuration**:
- Requires `BRAVE_API_KEY` environment variable
- Get API key from [Brave Search API](https://brave.com/search/api/)

**Features**:
- Web search results
- Search snippets
- URL ranking
- Safe search options

#### web-summary
```json
{
  "command": "/opt/venv/bin/python3",
  "args": ["-u", "/app/core-assets/scripts/web-summary-mcp-server.py"],
  "env": {
    "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
    "ZAI_CONTAINER_URL": "http://claude-zai-service:9600"
  }
}
```

**Purpose**: Fetches and summarizes web content, including YouTube videos, with semantic topic links for Logseq.

**Use Cases**:
- Summarizing articles and blog posts
- Extracting key points from YouTube videos
- Creating knowledge graph entries
- Generating semantic links for PKM systems

**Configuration**:
- Requires `GOOGLE_API_KEY` for Gemini API access
- Uses `ZAI_CONTAINER_URL` for AI processing
- Custom implementation in Python

**Features**:
- Web page summarization
- YouTube video transcript extraction and summary
- Semantic topic extraction
- Logseq-compatible output format
- Automatic topic linking

### GitHub Tools

#### github
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_TOKEN": "${GITHUB_TOKEN}"
  }
}
```

**Purpose**: GitHub API operations for repository management.

**Use Cases**:
- Managing issues
- Creating pull requests
- Reviewing code
- Managing releases
- Repository operations

**Configuration**:
- Requires `GITHUB_TOKEN` environment variable
- Token needs appropriate scopes for operations

**Capabilities**:
- Issue management (create, update, close)
- Pull request operations
- Repository information
- File operations in repos
- Branch management
- Release management
- Workflow management

### Workflow Tools

#### claude-flow
```json
{
  "command": "npx",
  "args": ["-y", "claude-flow", "mcp", "start"]
}
```

**Purpose**: Claude Flow MCP integration for agentic workflows.

**Use Cases**:
- Orchestrating multi-step workflows
- Managing task dependencies
- Coordinating multiple tools
- Implementing complex automation

**Features**:
- Workflow definition and execution
- Task coordination
- State management
- Error handling and retries

## Tool Categories

The configuration organizes tools into logical categories:

```json
{
  "documentation": ["context7"],
  "automation": ["playwright"],
  "filesystem": ["filesystem", "git"],
  "web": ["fetch", "brave-search", "web-summary"],
  "github": ["github"],
  "workflows": ["claude-flow"]
}
```

This categorization helps with:
- Understanding tool relationships
- Selecting appropriate tools for tasks
- Managing tool permissions
- Documentation organization

## Configuring mcp.json

### Basic Structure

```json
{
  "mcpServers": {
    "server-name": {
      "command": "executable",
      "args": ["arg1", "arg2"],
      "type": "stdio",
      "description": "What this server does",
      "env": {
        "VAR_NAME": "${VAR_NAME}"
      }
    }
  },
  "toolCategories": {
    "category": ["server-name"]
  },
  "config": {
    "defaultTimeout": 30000,
    "retryAttempts": 3,
    "logLevel": "info"
  }
}
```

### Server Configuration Fields

- **command**: Executable to run (npx, python, node, etc.)
- **args**: Command-line arguments as array
- **type**: Communication type (always "stdio" in this system)
- **description**: Human-readable description of server purpose
- **env**: Environment variables (use `${VAR}` for substitution)

### Global Configuration

```json
{
  "config": {
    "defaultTimeout": 30000,      // Tool execution timeout in ms
    "retryAttempts": 3,            // Number of retry attempts on failure
    "logLevel": "info"             // Logging verbosity (debug, info, warn, error)
  }
}
```

### Environment Variable Substitution

Environment variables in `mcp.json` use `${VARIABLE_NAME}` syntax:

```json
{
  "env": {
    "API_KEY": "${MY_API_KEY}"
  }
}
```

Set variables in:
- `.env` file in project root
- Docker Compose environment section
- Container environment configuration

## Using Each Tool Effectively

### Documentation Retrieval Pattern

```
Use context7 when you need:
1. Current API documentation
2. Up-to-date code examples
3. Framework-specific patterns
4. Library usage guides

Example: "Get the latest React hooks documentation"
```

### Browser Automation Pattern

```
Use playwright when you need:
1. Interact with dynamic web pages
2. Execute JavaScript on pages
3. Handle SPAs (Single Page Applications)
4. Take screenshots or generate PDFs
5. Test web applications

Example: "Navigate to example.com, fill login form, and capture dashboard"
```

### File Management Pattern

```
Use filesystem for:
1. Reading project files
2. Writing code or configuration
3. Managing project structure
4. Batch file operations

Use git for:
1. Version control operations
2. Tracking changes
3. Collaboration workflows
4. Repository management

Example: "Read config.json, modify settings, write back, and commit changes"
```

### Web Content Pattern

```
Use fetch for:
1. Simple HTTP requests
2. API consumption
3. Static content retrieval

Use brave-search for:
1. Finding information
2. Research tasks
3. Discovery

Use web-summary for:
1. Understanding web content
2. Extracting key information
3. Creating knowledge entries
4. YouTube content analysis

Example: "Search for 'Docker security best practices', fetch top result, and summarize"
```

### GitHub Operations Pattern

```
Use github when you need:
1. Repository management
2. Issue tracking
3. Pull request workflows
4. Code review automation
5. Release management

Example: "Create issue for bug, assign to team member, add labels"
```

### Workflow Orchestration Pattern

```
Use claude-flow for:
1. Multi-step processes
2. Complex task dependencies
3. Coordinated tool usage
4. Error recovery workflows

Example: "Fetch repo, analyze code, create issues for problems, generate report"
```

## Adding Custom MCP Servers

### Step 1: Implement MCP Server

Create a server that implements the MCP protocol. Options:

**Node.js Example**:
```javascript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'my-custom-server',
  version: '1.0.0'
});

// Define tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: 'my_tool',
    description: 'Does something useful',
    inputSchema: {
      type: 'object',
      properties: {
        input: { type: 'string' }
      }
    }
  }]
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  // Implementation
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

**Python Example**:
```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("my-custom-server")

@app.list_tools()
async def list_tools():
    return [{
        "name": "my_tool",
        "description": "Does something useful",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            }
        }
    }]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    # Implementation
    pass

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Add to mcp.json

```json
{
  "mcpServers": {
    "my-custom-server": {
      "command": "node",
      "args": ["/path/to/my-server.js"],
      "type": "stdio",
      "description": "My custom MCP server",
      "env": {
        "MY_API_KEY": "${MY_API_KEY}"
      }
    }
  },
  "toolCategories": {
    "custom": ["my-custom-server"]
  }
}
```

### Step 3: Make Available in Container

**Option 1: Include in Docker image**
```dockerfile
COPY my-server.js /app/custom-tools/
RUN npm install dependencies
```

**Option 2: Mount as volume**
```yaml
volumes:
  - ./custom-tools:/app/custom-tools
```

### Step 4: Test Server

```bash
# Test standalone
echo '{"jsonrpc":"2.0","method":"initialize","id":1}' | node my-server.js

# Test in container
docker exec workstation-dev node /app/custom-tools/my-server.js
```

## Troubleshooting MCP Issues

### Server Won't Start

**Symptoms**: Tool unavailable, timeout errors

**Diagnosis**:
```bash
# Check if command exists
docker exec workstation-dev which npx

# Test server directly
docker exec workstation-dev npx -y @modelcontextprotocol/server-filesystem /home/devuser/workspace

# Check logs
docker logs workstation-dev | grep -i mcp
```

**Solutions**:
1. Verify command path is correct
2. Check Node.js/Python is installed
3. Ensure dependencies are available
4. Verify network access for npx downloads

### Environment Variables Not Set

**Symptoms**: Authentication errors, API key missing

**Diagnosis**:
```bash
# Check environment in container
docker exec workstation-dev env | grep API_KEY

# Check .env file
cat multi-agent-docker/.env
```

**Solutions**:
1. Add variables to `.env` file
2. Update docker-compose.yml environment section
3. Restart containers after changes
4. Use `${VARIABLE}` syntax in mcp.json

### Tool Execution Timeout

**Symptoms**: Operations timeout, no response

**Diagnosis**:
- Check defaultTimeout value
- Monitor tool resource usage
- Review tool logs for hangs

**Solutions**:
```json
{
  "config": {
    "defaultTimeout": 60000,  // Increase timeout
    "retryAttempts": 1        // Reduce retries
  }
}
```

### Stdio Communication Issues

**Symptoms**: Garbled output, protocol errors

**Diagnosis**:
- Check for stdout/stderr mixing
- Verify JSON-RPC format
- Look for debug prints in server code

**Solutions**:
1. Remove debug prints from server code
2. Use stderr for logging, stdout for protocol
3. Python: Use `python -u` for unbuffered output
4. Validate JSON-RPC messages

### Permission Errors

**Symptoms**: Cannot read/write files, access denied

**Diagnosis**:
```bash
# Check file permissions
docker exec workstation-dev ls -la /home/devuser/workspace

# Check user
docker exec workstation-dev whoami
```

**Solutions**:
1. Fix file ownership: `chown -R devuser:devuser /path`
2. Adjust filesystem server scope in mcp.json
3. Run tools as correct user
4. Check Docker volume mount permissions

### Tool Not Found in Category

**Symptoms**: Tool doesn't appear in expected category

**Solutions**:
1. Verify tool name matches server key in mcpServers
2. Check toolCategories mapping
3. Restart system to reload configuration

### Network/API Errors

**Symptoms**: Cannot reach external APIs, DNS failures

**Diagnosis**:
```bash
# Test network from container
docker exec workstation-dev curl -I https://api.github.com

# Check DNS
docker exec workstation-dev nslookup github.com
```

**Solutions**:
1. Check internet connectivity
2. Verify API endpoints are accessible
3. Check firewall rules
4. Validate API keys and tokens
5. Review rate limiting

### Debugging Workflow

1. **Verify Configuration**
   - Check mcp.json syntax
   - Validate environment variables
   - Confirm tool paths

2. **Test Standalone**
   - Run server outside MCP framework
   - Send manual JSON-RPC messages
   - Verify basic functionality

3. **Check Integration**
   - Test in container environment
   - Review logs for errors
   - Monitor resource usage

4. **Enable Verbose Logging**
   ```json
   {
     "config": {
       "logLevel": "debug"
     }
   }
   ```

5. **Isolate Issues**
   - Test one tool at a time
   - Disable other servers
   - Use minimal configuration

### Common Error Messages

**"Server initialization failed"**
- Command path incorrect
- Dependencies missing
- Environment variables not set

**"Tool execution timeout"**
- Operation taking too long
- Server hung or crashed
- Network request stalled

**"Invalid JSON-RPC response"**
- Server output contaminated
- Protocol implementation incorrect
- Parsing errors

**"Permission denied"**
- File access restricted
- User permissions insufficient
- Path outside allowed scope

**"API authentication failed"**
- API key missing or invalid
- Token expired
- Incorrect environment variable name

### Getting Help

When reporting MCP issues, include:
1. Server name and configuration
2. Complete error message
3. Docker logs
4. Environment variable status (redact secrets)
5. Steps to reproduce
6. Expected vs actual behavior

Check documentation:
- [MCP Specification](https://modelcontextprotocol.io)
- [SDK Documentation](https://github.com/modelcontextprotocol)
- Project-specific docs in `/docs/`

## Best Practices

1. **Security**
   - Never commit API keys to repository
   - Use environment variables for secrets
   - Scope filesystem access appropriately
   - Review tool permissions regularly

2. **Performance**
   - Set appropriate timeouts
   - Minimize retry attempts for fast-failing operations
   - Use on-demand tool spawning
   - Monitor resource usage

3. **Reliability**
   - Implement error handling in custom servers
   - Test tools in isolation before integration
   - Use descriptive error messages
   - Log important operations

4. **Maintainability**
   - Document custom servers
   - Use consistent naming conventions
   - Organize tools in logical categories
   - Keep mcp.json well-structured

5. **Development**
   - Test servers standalone first
   - Use version control for configurations
   - Validate JSON syntax
   - Monitor logs during development
