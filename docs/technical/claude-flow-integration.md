# Claude Flow MCP Fix - Integration Guide

## Overview
This package contains pre-fixed MCP server files that resolve the mock data issue where `agent_list` was returning fake agents (agent-1, agent-2, agent-3) instead of real spawned agents.

## Files Included

### 1. `/workspace/ext/fixed-mcp-files/mcp-server.js`
- **Purpose**: Fixed MCP server that queries memoryStore for real agents
- **Target Location**: `/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js`
- **Key Changes**:
  - Removed mock data fallback from `agent_list` function (was at line 1525)
  - Replaced with memoryStore query to return real agents
  - Fixed syntax errors from incorrect console.error placement

### 2. `/workspace/ext/fixed-mcp-files/mcp-tcp-server.js`
- **Purpose**: TCP server that uses global claude-flow binary
- **Target Location**: `/app/core-assets/scripts/mcp-tcp-server.js`
- **Key Changes**:
  - Changed spawn from `npx claude-flow@alpha` to `/usr/bin/claude-flow`
  - Added CLAUDE_FLOW_GLOBAL environment variable
  - Fixed syntax error in spawn command arguments

### 3. `/workspace/ext/setup-workspace-simple.sh`
- **Purpose**: Simplified setup script using copy approach
- **Features**:
  - Cleans npx cache to prevent version conflicts
  - Copies pre-fixed files to correct locations
  - Creates wrapper to prevent npx from downloading new versions
  - Restarts TCP server with fixed version

## Manual Integration Steps

### Option 1: Use the Simple Setup Script
```bash
# Make script executable
chmod +x /workspace/ext/setup-workspace-simple.sh

# Run the setup
bash /workspace/ext/setup-workspace-simple.sh
```

### Option 2: Manual File Copy
```bash
# 1. Backup original files
cp /usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js \
   /usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js.backup

cp /app/core-assets/scripts/mcp-tcp-server.js \
   /app/core-assets/scripts/mcp-tcp-server.js.backup

# 2. Copy fixed files
cp /workspace/ext/fixed-mcp-files/mcp-server.js \
   /usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js

cp /workspace/ext/fixed-mcp-files/mcp-tcp-server.js \
   /app/core-assets/scripts/mcp-tcp-server.js

# 3. Restart TCP server
pkill -f mcp-tcp-server.js
node /app/core-assets/scripts/mcp-tcp-server.js &
```

### Option 3: Add to Core Assets Build
To permanently include these fixes in your container image:

1. Copy fixed files to your build directory:
```bash
cp /workspace/ext/fixed-mcp-files/mcp-server.js \
   [your-build-dir]/core-assets/patches/mcp-server.js

cp /workspace/ext/fixed-mcp-files/mcp-tcp-server.js \
   [your-build-dir]/core-assets/scripts/mcp-tcp-server.js
```

2. Update your Dockerfile or build script:
```dockerfile
# After installing claude-flow
RUN npm install -g claude-flow@alpha

# Apply fixes
COPY core-assets/patches/mcp-server.js \
     /usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js

COPY core-assets/scripts/mcp-tcp-server.js \
     /app/core-assets/scripts/mcp-tcp-server.js
```

## Testing the Fix

Run the test script to verify agents are being tracked:
```bash
bash /workspace/ext/final-test.sh
```

Expected output:
- ✅ Agent spawning returns real agent IDs
- ✅ Agent list returns spawned agents (not mock data)
- ✅ No more agent-1, agent-2, agent-3 mock agents

## Key Issue Resolved

### Problem
- Multiple claude-flow versions: global install vs npx cache
- MCP server had hardcoded mock data fallback
- TCP server was using npx which downloaded different version
- Patches applied to global version weren't being used

### Solution
- Force use of global claude-flow at `/usr/bin/claude-flow`
- Remove mock data fallback from agent_list
- Prevent npx from downloading cached versions
- Ensure all services use the same fixed version

## Troubleshooting

### If agents still show as mock data:
1. Check which claude-flow is being used:
   ```bash
   ps aux | grep claude-flow
   ```
   Should show `/usr/bin/claude-flow`, not `/home/ubuntu/.npm/_npx/`

2. Clear npx cache:
   ```bash
   rm -rf /home/ubuntu/.npm/_npx/*claude-flow*
   ```

3. Verify TCP server is using correct binary:
   ```bash
   grep spawn /app/core-assets/scripts/mcp-tcp-server.js
   ```
   Should show `/usr/bin/claude-flow`

### If TCP server won't start:
1. Check for syntax errors:
   ```bash
   node -c /app/core-assets/scripts/mcp-tcp-server.js
   node -c /usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js
   ```

2. Check port 9500:
   ```bash
   netstat -tlnp | grep 9500
   ```

3. View logs:
   ```bash
   tail -f /tmp/mcp-tcp.log
   ```

## Version Compatibility
- Tested with: claude-flow v2.0.0-alpha.108
- Node.js: v22.x or later
- TCP Port: 9500 (default)

## Support
For issues or questions about this fix:
- Check the test output: `bash /workspace/ext/final-test.sh`
- Review the fixed files in `/workspace/ext/fixed-mcp-files/`
- Ensure only one version of claude-flow is installed