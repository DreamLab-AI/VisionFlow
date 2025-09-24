# Multi-Agent Docker System Issues and Fixes

## Overview

This document tracks issues and solutions for the multi-agent Docker system, which consists of:
1. **multi-agent-container**: Hosts MCP server, supervisord services, and development environment
2. **gui-tools-service**: Provides GUI application support (Chrome, QGIS, etc.)
3. **visionflow_container**: AR/VR application that connects to the MCP server

## Issue 1: Docker Initialization Race Conditions (RESOLVED)

### Problem
The multi-agent Docker system had initialization race conditions from multiple overlapping scripts:
- `entrypoint-wrapper.sh`: Complex initialization logic
- `entrypoint.sh`: 240+ lines of redundant functionality
- `setup-workspace.sh`: Service management conflicts
- `automated-setup.sh`: Redundant operations

### Root Causes
1. Multiple scripts trying to manage the same services
2. Read-only mounted files causing permission errors
3. Docker-compose command overriding supervisord
4. Claude authentication order dependencies
5. Redundant Rust toolchain validation

### Implemented Solutions

#### 1. Simplified entrypoint.sh
Reduced from 240+ lines to 36 lines, focusing only on:
- Creating necessary directories
- Setting permissions (skipping .claude* files)
- Starting supervisord in background
- Launching bash for interactive mode

#### 2. Fixed Permission Errors
```bash
# Skip mounted files that might be read-only
find /home/dev -maxdepth 1 ! -name '.claude*' -exec chown dev:dev {} \; 2>/dev/null || true
```

#### 3. Fixed Supervisord Management
```bash
# Start supervisord in the background for all cases
echo "Starting supervisord in background..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &
```

#### 4. Fixed Claude Authentication Order
Added automatic Claude initialization in background to ensure auth is ready before setup-workspace.sh runs.

#### 5. Fixed claude-flow-tcp-proxy.js
```javascript
const TCP_PORT = parseInt(process.env.CLAUDE_FLOW_TCP_PORT || '9502');
```

#### 6. Reverted to npx for Reliability
Per user preference for "slower startup" but more reliable operation, all claude-flow references use npx.

## Issue 2: Client UI "MCP Disconnected" Error (RESOLVED)

### Problem
The VisionFlow client UI shows "MCP Disconnected" despite the MCP TCP server working correctly on port 9500.

### Root Cause
The client was calling the wrong REST API endpoint:
- **Previous (incorrect)**: `/api/bots/mcp-status` (returns 404)
- **Correct endpoint**: `/api/bots/status` (as defined in `/src/handlers/api_handler/bots/mod.rs`)

### Fix Applied
✅ Updated `/client/src/features/bots/components/MultiAgentInitializationPrompt.tsx` line 45:
```typescript
// Changed from:
const response = await fetch(`${apiService.getBaseUrl()}/bots/mcp-status`);
// To:
const response = await fetch(`${apiService.getBaseUrl()}/bots/status`);
```

### Verification
- REST API endpoint now correctly connects
- The endpoint polls every 3 seconds to check connection status
- Auth is disabled via .env (AUTH_ENABLED=false)

## Issue 3: MCP TCP Client Method Call Format (RESOLVED)

### Previous State

### Infrastructure
- **MCP Server**: Running in `multi-agent-container` on port 9500
- **MCP Client**: Running in `visionflow_container`
- **Network**: Both containers communicate via Docker network using hostname `multi-agent-container`
- **Connection Status**: TCP connections work but receive brief health checks every 2 seconds

### Error Pattern
```
[2025-09-23T19:44:59Z ERROR webxr::actors::claude_flow_actor] MCP TCP query failed, falling back to JSON-RPC: Request failed after 6 attempts: MCP Error -32601: Method not found (data: None)
```

### Connection Testing Results
```bash
# From visionflow container to multi-agent-container:9500
$ docker exec visionflow_container sh -c 'nc -zv multi-agent-container 9500'
Connection to multi-agent-container (172.18.0.9) 9500 port [tcp/*] succeeded!

# MCP server logs show brief connections every 2 seconds:
[INFO] [claude-flow] New TCP connection from 172.18.0.10:45678
[INFO] [claude-flow] TCP client disconnected
```

## Root Cause

The MCP client in `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/src/utils/mcp_tcp_client.rs` is sending direct method calls like:

```json
{
  "jsonrpc": "2.0",
  "method": "agent_list",
  "params": {...},
  "id": 1
}
```

However, the MCP server only recognizes these top-level methods:
- `initialize`
- `tools/list`
- `tools/call`

All actual functionality must be accessed through `tools/call`.

## Fix Applied

### Solution Implemented

The MCP TCP client in `/src/utils/mcp_tcp_client.rs` was verified and found to **already have the correct implementation**:

✅ **`send_tool_call()` method** (lines 162-197) properly wraps all tool calls:
```rust
let wrapped_params = json!({
    "name": tool_name,
    "arguments": arguments
});
let response = self.send_request("tools/call", wrapped_params).await?;
```

✅ **All query methods use the correct wrapper**:
- `query_agent_list()` - Uses `send_tool_call("agent_list", params)`
- `query_swarm_status()` - Uses `send_tool_call("swarm_status", params)`
- `query_server_info()` - Uses `send_tool_call("server_info", params)`

✅ **Response parsing correctly handles nested format**:
- Extracts content from `content[0].text`
- Parses nested JSON response properly

### Verification Completed

✅ **Cargo check passes** - All Rust code compiles successfully with only minor warnings
✅ **Client UI endpoint fixed** - Now uses correct `/bots/status` endpoint
✅ **MCP protocol compliance verified** - All calls properly wrapped in `tools/call`

## Success Criteria

- No more "Method not found" errors in logs
- Agent status successfully displayed in VisionFlow UI
- TCP connection remains stable without fallback to JSON-RPC
- All MCP tool calls working correctly

## System Architecture

### Container Roles

1. **multi-agent-container**
   - MCP server on port 9500 (claude-flow-tcp-proxy.js)
   - Supervisord managing multiple services
   - Development environment with Claude CLI
   - Mounts: `/logs`, `/home/dev/.claude.json`

2. **gui-tools-service**
   - Chrome browser for web-based tools
   - QGIS on port 9875
   - Other GUI applications
   - Necessary for the multi-agent system's visual tools

3. **visionflow_container**
   - AR/VR application
   - MCP client connecting to multi-agent-container:9500
   - Uses environment variables:
     - `CLAUDE_FLOW_HOST` or `MCP_HOST` (defaults to "multi-agent-container")
     - `MCP_TCP_PORT` (defaults to 9500)

### Key Configuration Files

1. **docker-compose.yml**
   - Removed redundant entrypoint-wrapper.sh
   - Removed unused named volumes
   - Uses simplified entrypoint.sh

2. **mcp.json**
   - All claude-flow references use npx
   - GUI tools configured with correct container hostnames
   - QGIS port changed from 9877 to 9875

3. **setup-workspace.sh**
   - No longer manages services (removed supervisorctl commands)
   - Retains security checks and Claude auth
   - Creates multi-agent helper script

### MCP Server Details

The MCP server provides 90+ tools for hive mind operations including:
- `swarm_init`, `agent_spawn`, `task_orchestrate`
- `neural_train`, `neural_patterns`, `neural_predict`
- `memory_usage`, `performance_report`, `bottleneck_analyze`
- And many more...

All of these must be accessed through the `tools/call` wrapper method.

## Verification Commands

### Check MCP Server
```bash
# Test MCP server directly
docker exec multi-agent-container ps aux | grep claude-flow

# Check MCP port
docker exec multi-agent-container netstat -tlnp | grep 9500

# Test from visionflow
docker exec visionflow_container sh -c 'echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"agent_list\",\"arguments\":{}}}" | nc -w 2 multi-agent-container 9500'
```

### Check Client UI
```bash
# Monitor client API calls
docker logs visionflow_container -f | grep "api/bots"

# Check server routes
grep -r "mcp-status" /mnt/mldata/githubs/AR-AI-Knowledge-Graph/src/
```

## Summary

### Completed
1. ✅ Docker initialization race conditions fixed
2. ✅ Permission errors resolved
3. ✅ Supervisord management fixed
4. ✅ Claude authentication order fixed
5. ✅ TCP port parsing fixed
6. ✅ Reverted to npx for reliability
7. ✅ Client UI endpoint fixed (changed `/api/bots/mcp-status` to `/api/bots/status`)
8. ✅ MCP TCP client verified (already using correct `tools/call` format)
9. ✅ Rust code compilation verified with `cargo check`
10. ✅ Hive mind swarm deployed to coordinate fixes
11. ✅ Fixed "body stream already read" error in apiService.ts
12. ✅ Fixed API endpoint mismatch (/bots/initialize-multi-agent → /bots/initialize-swarm)

### All Issues Resolved
- Multi-agent container connection working
- MCP TCP server responding correctly
- Client UI connecting to correct endpoints
- MCP protocol compliance verified
- Client-side API error handling fixed
- Hive mind spawn endpoint corrected

### Latest Fixes (2025-09-24 12:55)
- **apiService.ts**: Fixed response body stream error by removing duplicate `response.text()` call after `response.json()`
- **MultiAgentInitializationPrompt.tsx**: Updated API endpoint from `/bots/initialize-multi-agent` to `/bots/initialize-swarm` to match server route

### MCP Server Status (2025-09-24 13:00)
- **MCP TCP Server**: Running on port 9500 ✓
- **WebSocket Bridge**: Running on port 3002 ✓
- **Task Submission**: Confirmed working with test task
- **Task ID Format**: `task_[timestamp]_[random]` (e.g., task_1758718778897_xtbo2y3ei)
- **Swarm Status**: Active with ID `swarm_1758717360460_2w0daysss`
- **Connection**: All components verified and operational

## Issue 4: Mock Agents Appearing in UI (RESOLVED ✅)

### Problem (2025-09-24 13:10)
The UI shows mock agents (coordinator-1, researcher-1, coder-1) instead of real agents spawned by the MCP server. Investigation revealed:
- The MCP TCP server was spawning new `npx claude-flow@alpha` processes on each connection
- Each npx invocation downloads and installs a fresh, unpatched version of claude-flow
- The unpatched version contains hardcoded mock agents in the `agent_list` response
- Multiple MCP server processes were running simultaneously

### Root Cause
The file `/app/core-assets/scripts/mcp-tcp-server.js` in the container was using:
```javascript
const mcpCommand = process.env.MCP_TCP_COMMAND || 'npx';
const mcpArgs = process.env.MCP_TCP_ARGS ? process.env.MCP_TCP_ARGS.split(' ') : ['claude-flow@alpha', 'mcp', 'start'];
```

This caused every external connection to download a new unpatched claude-flow package.

### Solution
Updated `multi-agent-docker/core-assets/scripts/mcp-tcp-server.js` to use the global patched installation:
```javascript
const mcpCommand = process.env.MCP_TCP_COMMAND || '/usr/bin/claude-flow';
const mcpArgs = process.env.MCP_TCP_ARGS ? process.env.MCP_TCP_ARGS.split(' ') : ['mcp', 'start'];
```

This ensures:
- The patched global installation is used instead of downloading via npx
- No mock agents in the response
- Consistent behavior across all connections
- Proper agent tracking and real agent data

### Verification Steps
1. Copy updated mcp-tcp-server.js to container
2. Restart mcp-core:mcp-tcp-server via supervisorctl
3. Test agent_list returns real agents, not mock ones
4. Verify UI shows actual spawned agents

### Permanent Solution (2025-09-24 13:35)
To prevent ephemeral npx installations that cannot be patched:
1. Updated `mcp-tcp-server.js` to use `/app/node_modules/.bin/claude-flow` instead of npx
2. Updated `entrypoint.sh` to create symlink from node_modules installation
3. Updated `setup-workspace.sh` to patch `/app/node_modules/claude-flow/src/mcp/mcp-server.js`
4. This uses the claude-flow already installed via package.json dependencies

Benefits:
- No more ephemeral npx downloads on each connection
- Patches can be applied reliably to a known location
- Consistent version across all usages
- Faster startup (no download needed)

### Complete Fix Implementation (2025-09-24 13:40)
Updated ALL npx references across the codebase:
- **mcp-tcp-server.js**: Uses `/app/node_modules/.bin/claude-flow`
- **claude-flow-tcp-proxy.js**: Uses `/app/node_modules/.bin/claude-flow`
- **package.json**: Removed `npx claude-flow@alpha` from all scripts
- **init-claude-flow-agents.sh**: Replaced all npx calls with direct `claude-flow`
- **automated-setup.sh**: Replaced all npx calls with direct `claude-flow`
- **setup-workspace.sh**: Updated all aliases and version checks
- **entrypoint.sh**: Creates symlink `/usr/bin/claude-flow -> /app/node_modules/.bin/claude-flow`

### Verification Results ✅
1. **Symlink created**: `/usr/bin/claude-flow -> /app/node_modules/.bin/claude-flow`
2. **Patches applied successfully**: All mock agents removed from mcp-server.js
3. **MCP server running**: Using patched node_modules version
4. **No mock agents**: Returns empty array when no agents spawned
5. **Real agents work**: Successfully spawned `agent_1758721150001_hgdowu` (system-coordinator)
6. **API working**: Returns real agent data, no more hardcoded mocks

### Summary
The multi-agent system now uses a single, patchable installation of claude-flow from node_modules. This eliminates the problem of ephemeral npx installs that download fresh, unpatched versions on every connection. All components now reference the same patched instance, ensuring consistent behavior and real agent data throughout the system.

## Issue 5: MCP Orchestrator Environment Inheritance (IN PROGRESS)

### Problem (2025-09-24 14:00)
The MCP orchestrator doesn't inherit the dev user's runtime environment, preventing agents from executing tools like LaTeX that are available in the container.

### Symptoms
- LaTeX is installed and works: `/usr/bin/pdflatex` exists
- Manual LaTeX compilation works in the container
- MCP tools like `terminal_execute` return generic success messages without actual output
- Agents cannot execute real commands or access the dev user's tools

### Investigation
- MCP TCP server runs as dev user ✓
- LaTeX tools are in PATH ✓
- Created LaTeX specialist agent with capabilities ✓
- But agents cannot actually execute LaTeX commands

### Root Cause
The MCP server process needs to inherit the full dev user environment including:
- Proper PATH with all tool directories
- HOME directory set correctly
- Access to CLAUDE.md configuration
- Environment variables for tool access

### Attempted Fix
Updated `mcp-tcp-server.js` to set explicit environment:
```javascript
const devEnv = {
  ...process.env,
  HOME: '/home/dev',
  USER: 'dev',
  PATH: `/home/dev/.local/bin:/opt/venv312/bin:/home/dev/.cargo/bin:/home/dev/.deno/bin:/app/core-assets/scripts:/app/core-assets/mcp-tools:/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin`,
  CLAUDE_FLOW_DIRECT_MODE: 'true',
  CLAUDE_FLOW_DB_PATH: '/workspace/.swarm/memory.db',
  CLAUDE_CONFIG_DIR: '/home/dev/.claude',
  CLAUDE_PROJECT_ROOT: '/workspace'
};
```

### Next Steps
Need to verify if the MCP tools have actual implementations or if they're returning mock responses, and ensure the orchestrator can execute real commands in the dev user environment.

### Investigation Results (2025-09-24 14:05)
Found that many MCP tools including `terminal_execute` are NOT actually implemented. In `/app/node_modules/claude-flow/src/mcp/mcp-server.js`, the executeTool function has a default case that returns mock success for any unimplemented tool:

```javascript
default:
  return {
    success: true,
    tool: name,
    message: `Tool ${name} executed successfully`,
    args: args,
    timestamp: new Date().toISOString(),
  };
```

This explains why:
- `terminal_execute` returns success but no output
- LaTeX commands can't actually be run
- The orchestrator can't execute real commands

Implemented tools include:
- `swarm_init`, `agent_spawn`, `agent_list` (basic swarm management)
- Some workflow and performance monitoring tools
- Memory storage operations

NOT implemented (return mock responses):
- `terminal_execute` - Can't run commands
- `config_manage` - Can't manage configs
- Most system interaction tools

### Solution Required
To enable LaTeX PDF generation and other real command execution, we need to:
1. Implement `terminal_execute` to actually run commands with proper environment
2. Or find an alternative way for agents to execute tasks
3. Or use a different orchestration system that has real execution capabilities

### Implementation Attempt (2025-09-24 14:10)
Added patch to setup-workspace.sh that implements terminal_execute:
- Uses Node.js child_process.spawn to execute commands
- Sets proper environment variables and PATH
- Returns stdout, stderr, and exit code

### Terminal Execute Implementation Success (2025-09-24 14:26)
The patch was applied successfully and terminal_execute now works:
- Tested with `pdflatex --version` - returns proper output
- Commands execute with correct dev user environment
- LaTeX PDF generation confirmed working (test.pdf created)

### Critical Architecture Issue Discovered (2025-09-24 14:30)
The MCP TCP server has a fundamental design flaw:
- **Each TCP connection spawns a new isolated MCP process**
- Tasks submitted through one connection are invisible to other connections
- No persistence or state sharing between connections
- Agents spawned in one process don't exist in others

This explains why:
- The Rust "hello world" task submitted via client UI is not visible via CLI
- Agent lists show empty when queried from command line
- Each `nc` command creates a fresh, isolated environment
- Tasks appear to "disappear" between connections

The current `mcp-tcp-server.js` implementation:
```javascript
// Line 57-63: Each connection spawns a new process
this.mcpProcess = spawn(mcpCommand, mcpArgs, {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: '/workspace',
    env: devEnv,
    uid: process.getuid(),
    gid: process.getgid()
});
```

### Impact
This architecture makes the multi-agent system effectively unusable for persistent tasks:
- Tasks submitted are isolated to their connection
- Agents can't collaborate across connections
- No way to monitor or manage tasks after submission
- Each API call or CLI command sees a different state

### Required Fix
The MCP TCP server needs fundamental redesign:
1. Maintain a single persistent MCP process
2. Route all TCP connections through this single process
3. Implement proper connection multiplexing
4. Add state persistence between connections

Until this is fixed, the multi-agent system cannot function as intended for coordinated task execution.

### Hive-Mind Alternative Approach (2025-09-24 14:47)
Tested using `claude-flow hive-mind` system as an alternative to MCP:
- Successfully spawns persistent swarms with Queen and Worker agents
- Maintains state across sessions (unlike MCP TCP server)
- Has proper task orchestration capabilities
- **BUT**: Tasks are created but not executed

Issues discovered:
1. Hive-mind creates swarms and tasks but no actual execution happens
2. Worker agents exist but don't execute terminal commands
3. The `terminal_execute` implementation works but isn't being called by agents
4. No integration between hive-mind agents and the MCP tool execution layer

The hive-mind system shows promise but needs:
- Integration between worker agents and terminal_execute
- Automatic task routing to appropriate workers
- Execution feedback loop to update task status


this is how we should launch claude-flow hive-mind which in theory we can monitor using the same tools

 claude-flow hive-mind

🧠 Claude Flow Hive Mind System

USAGE:
  claude-flow hive-mind [subcommand] [options]

SUBCOMMANDS:
  init         Initialize hive mind system
  spawn        Spawn hive mind swarm for a task
  status       Show hive mind status
  resume       Resume a paused hive mind session
  stop         Stop a running hive mind session
  sessions     List all hive mind sessions
  consensus    View consensus decisions
  memory       Manage collective memory
  metrics      View performance metrics
  wizard       Interactive hive mind wizard

EXAMPLES:
  # Initialize hive mind
  claude-flow hive-mind init

  # Spawn swarm with interactive wizard
  claude-flow hive-mind spawn

  # Quick spawn with objective
  claude-flow hive-mind spawn "Build microservices architecture"

  # View current status
  claude-flow hive-mind status

  # Interactive wizard
  claude-flow hive-mind wizard

  # Spawn with Claude Code coordination
  claude-flow hive-mind spawn "Build REST API" --claude

  # Auto-spawn coordinated Claude Code instances
  claude-flow hive-mind spawn "Research AI trends" --auto-spawn --verbose

  # List all sessions
  claude-flow hive-mind sessions

  # Resume a paused session
  claude-flow hive-mind resume session-1234567890-abc123

KEY FEATURES:
  🐝 Queen-led coordination with worker specialization
  🧠 Collective memory and knowledge sharing
  🤝 Consensus building for critical decisions
  ⚡ Parallel task execution with auto-scaling
  🔄 Work stealing and load balancing
  📊 Real-time metrics and performance tracking
  🛡️ Fault tolerance and self-healing
  🔒 Secure communication between agents

OPTIONS:
  --queen-type <type>    Queen coordinator type (strategic, tactical, adaptive)
  --max-workers <n>      Maximum worker agents (default: 8)
  --consensus <type>     Consensus algorithm (majority, weighted, byzantine)
  --memory-size <mb>     Collective memory size in MB (default: 100)
  --auto-scale           Enable auto-scaling based on workload
  --encryption           Enable encrypted communication
  --monitor              Real-time monitoring dashboard
  --verbose              Detailed logging
  --claude               Generate Claude Code spawn commands with coordination
  --spawn                Alias for --claude
  --auto-spawn           Automatically spawn Claude Code instances
  --execute              Execute Claude Code spawn commands immediately

For more information:
https://github.com/ruvnet/claude-flow/tree/main/docs/hive-mind

so we should use terminal exec
claude-flow hive-mind spawn "task from the client passed through" --claude

and then use the monitoring tools in the same way. The isolation of the processes might mean the problem needs a new architechture but it's possible this is enough.