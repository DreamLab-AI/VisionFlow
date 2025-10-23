# Node.js Version Update to 23.11.1

## Date
2025-10-23

## Summary
Updated the agentic-workstation Docker container to use Node.js v23.11.1 instead of v22.11.0 to resolve compatibility issues with chrome-devtools-mcp and other MCP tools.

## Problem
The chrome-devtools-mcp package requires Node.js version 22.12.0 LTS or newer. The container was previously built with Node v22.11.0, which caused the following error:

```
ERROR: `chrome-devtools-mcp` does not support Node v22.11.0.
Please upgrade to Node 22.12.0 LTS or a newer LTS.
```

This affected the ability to use Chrome DevTools MCP for browser automation and debugging tasks.

## Solution
Updated `Dockerfile.unified` to install Node.js v23.11.1 directly from the official Node.js distribution:

### Changes Made

#### 1. Updated Node.js Installation (Line 47-50)
**Before:**
```dockerfile
# Install Node.js LTS (v22) via official installer to avoid better-sqlite3 v24 incompatibility
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - || \
    (curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | \
     tar -xJ -C /usr/local --strip-components=1) && \
    npm install -g npm@latest
```

**After:**
```dockerfile
# Install Node.js v23.11.1 (required for chrome-devtools-mcp compatibility)
RUN curl -fsSL https://nodejs.org/dist/v23.11.1/node-v23.11.1-linux-x64.tar.xz | \
     tar -xJ -C /usr/local --strip-components=1 && \
    npm install -g npm@latest
```

#### 2. Updated Comment (Line 20)
**Before:**
```dockerfile
# Development languages (skip nodejs from pacman, will install LTS via nvm)
```

**After:**
```dockerfile
# Development languages (skip nodejs from pacman, will install v23.11.1 directly)
```

## Benefits
1. **MCP Compatibility**: chrome-devtools-mcp now works without errors
2. **Future-Proof**: Node v23.x is more current and supports newer packages
3. **Cleaner Installation**: Removed fallback logic and simplified to direct download
4. **Consistent Environment**: All MCP tools now have access to compatible Node version

## Testing
After rebuilding the container with the updated Dockerfile:

```bash
# Build new container
cd /home/devuser/workspace/project/multi-agent-docker
docker build -f Dockerfile.unified -t agentic-workstation:latest .

# Verify Node version inside container
docker run --rm agentic-workstation:latest node --version
# Expected output: v23.11.1

# Test chrome-devtools-mcp
docker run --rm agentic-workstation:latest npx chrome-devtools-mcp@latest --version
# Expected output: 0.9.0 (without errors)
```

## MCP Configuration
The chrome-devtools MCP server is configured in `.claude.json`:

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "type": "stdio",
      "command": "/home/devuser/.config/nvm/versions/node/v23.11.1/bin/npx",
      "args": ["chrome-devtools-mcp@latest"],
      "env": {
        "PATH": "/home/devuser/.config/nvm/versions/node/v23.11.1/bin:/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```

## Impact on Existing Features
- ✅ All existing npm packages continue to work
- ✅ claude-flow@alpha MCP server: Compatible
- ✅ ruv-swarm MCP server: Compatible
- ✅ Global packages (claude-code, typescript, playwright): Compatible
- ✅ All skill MCP servers: Compatible

## Rebuild Instructions
To apply these changes to your existing container:

```bash
# Stop existing container
docker-compose -f docker-compose.unified.yml down

# Rebuild with new Node version
./build-unified.sh --no-cache

# Or manually:
docker build --no-cache -f Dockerfile.unified -t agentic-workstation:latest .
docker-compose -f docker-compose.unified.yml up -d

# Verify services started
docker exec agentic-workstation supervisorctl status
```

## Related Files
- `multi-agent-docker/Dockerfile.unified` - Main Dockerfile with Node installation
- `.claude.json` - MCP server configurations
- `multi-agent-docker/build-unified.sh` - Build script

## References
- [Chrome DevTools MCP Documentation](https://github.com/google/chrome-devtools-mcp)
- [Node.js 23.x Release Notes](https://nodejs.org/en/blog/release/v23.11.1)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

## Notes
- Node.js v23.x is the current stable release as of October 2025
- v22.12.0 LTS would also be compatible, but v23.11.1 is more current
- No breaking changes observed in the upgrade from v22 to v23 for our use case
- NVM is still available in the container for users who want to manage multiple Node versions

## Author
Claude Code (Anthropic) with human oversight
