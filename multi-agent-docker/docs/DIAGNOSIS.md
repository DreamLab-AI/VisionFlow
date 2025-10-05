# MCP Connection Failure Diagnosis

**Date**: 2025-10-05 17:11
**Issue**: Claude-flow MCP server fails to connect from Claude Code

## 🔍 Root Cause Found

### The Problem
Claude-flow MCP server **starts successfully** but immediately sees **stdin closed** and shuts down:

```
✅ Starting Claude Flow MCP server in stdio mode...
{"jsonrpc":"2.0","method":"server.initialized","params":{...}}
🔌 Connection closed: session-cf-1759684219031-alhy
MCP: stdin closed, shutting down...
```

**Timeline:**
1. Claude Code spawns `/usr/sbin/claude-flow mcp start`
2. Server initializes and sends `server.initialized` message
3. **Stdin immediately closes** (Claude Code closes the pipe?)
4. Server detects closed stdin and shuts down
5. Claude Code marks server as "failed"

### Why This Happens

**Possible causes:**
1. **Claude Code closes stdin too early** - Before server can read initialize request
2. **TTY/pipe issue** - stdio communication broken
3. **Process spawn issue** - Claude Code not keeping pipes open
4. **Timeout** - 30 second timeout expires before handshake completes

### Evidence from Logs

**claude-flow log:**
```json
{
  "debug": "Starting connection with timeout of 30000ms",
  "timestamp": "2025-10-05T17:04:08.945Z"
}
// No further entries - connection never established
```

**Server output:**
```
[claude-flow-mcp] Claude-Flow MCP server starting in stdio mode
[claude-flow-mcp] 🔌 Connection closed: session-cf-...
[claude-flow-mcp] MCP: stdin closed, shutting down...
```

## 🧪 Test Results

### ✅ Server Works Standalone
```bash
$ /usr/sbin/claude-flow mcp start
✅ Starting Claude Flow MCP server in stdio mode...
# Waits for input...
```

### ✅ Server Responds to JSON-RPC
```bash
$ echo '{"jsonrpc":"2.0",...}' | /usr/sbin/claude-flow mcp start
# Sends initialize response correctly
```

### ❌ Claude Code Connection Fails
```
Status: failed
Log: Connection timeout after 30000ms
```

## 🔧 Potential Fixes

### Fix 1: Check Claude Code Spawn Settings

The issue may be in how Claude Code spawns the process. Check:
- `stdio: ['pipe', 'pipe', 'pipe']` configuration
- Keep stdin open until server confirms ready
- Don't close stdin immediately after spawn

### Fix 2: Add Keepalive to Server

Modify claude-flow to:
- Send heartbeat messages
- Don't exit on first EOF
- Buffer stdin reads

### Fix 3: Use Different Transport

Instead of stdio, use:
- TCP socket (port 9500 - already working!)
- Unix socket
- WebSocket

### Fix 4: Debug Claude Code Spawn

```bash
# Add debugging to see what's happening
export DEBUG=*
claude --debug
```

## 🎯 Recommended Solution

**Use TCP MCP instead of stdio:**

The TCP MCP server on port 9500 is **already working**. We can connect Claude Code to it instead of spawning a stdio process.

### Update `.mcp.json`:
```json
{
  "claude-flow": {
    "command": "node",
    "args": ["-e", "
      const net = require('net');
      const client = net.connect(9500, '127.0.0.1');
      process.stdin.pipe(client);
      client.pipe(process.stdout);
    "],
    "type": "stdio"
  }
}
```

Or create a proxy script that bridges stdio to TCP.

## 📊 Other MCP Server Issues

### ruv-swarm
**Error:** Invalid JSON-RPC messages (missing `id`, `method`)
**Fix:** Update ruv-swarm to send valid messages

### flow-nexus
**Error:** Sends plain text to stdout instead of JSON-RPC
**Fix:** Suppress informational output when in MCP mode

## 🚀 Immediate Workaround

**Use the working MCP servers:**
- ✅ playwright - Connected and working
- ✅ TCP MCP at port 9500 - Working (tested)

**Skip the broken ones for now:**
- ❌ claude-flow (stdio broken)
- ❌ ruv-swarm (protocol errors)
- ❌ flow-nexus (output corruption)

## 📝 Next Steps

1. **Test TCP bridge** - Create stdio-to-TCP proxy
2. **Report upstream** - File issue with claude-flow about stdin closing
3. **Alternative transport** - Use TCP or Unix socket
4. **Debug Claude Code** - Check why it closes stdin immediately
