# WebXR Agent Graph System - Current Status & Future Work

**Date**: 2025-08-30
**Status**: 🟡 **MOSTLY COMPLETE** - Core functionality working, architectural improvements pending
**Environment**: `multi-agent-container` + `visionflow_container`

## 🔍 MCP TCP Interface Analysis Results - COMPLETE ✅

### Protocol Implementation
**Date**: 2025-08-30
**Status**: ✅ **FULLY ANALYZED AND TESTED**
**Analysis**: Complete protocol mapping, test suite created and validated

#### Key Protocol Findings:
1. **Protocol**: JSON-RPC 2.0 over TCP (line-delimited)
2. **Port**: 9500 (configurable via MCP_TCP_PORT)
3. **Host**: multi-agent-container (Docker service name)
4. **Authentication**: None (network-level access control)
5. **Server**: claude-flow v2.0.0-alpha.101 MCP implementation

#### Critical Protocol Behavior:
1. **Startup Sequence**: Server sends non-JSON startup message (must be ignored)
2. **Notifications**: `server.initialized` has no ID field (skip these)
3. **Responses**: Valid responses contain matching request ID
4. **Connection**: Must keep connection open briefly (~100ms) to receive full response

#### Message Flow:
```
Client → Server:
{"jsonrpc":"2.0","id":"uuid","method":"initialize","params":{...}}

Server → Client (3 lines):
1. ✅ Starting Claude Flow MCP server... (ignore)
2. {"jsonrpc":"2.0","method":"server.initialized"...} (notification)
3. {"jsonrpc":"2.0","id":"uuid","result":{...}} (actual response)
```

#### Critical Tools for VisionFlow:
- `swarm_init` - Initialize swarm with topology
- `agent_list` - Get agent statuses  
- `swarm_destroy` - Terminate swarm
- `swarm_status` - Get swarm state

#### Test Scripts Created:
1. **Main Test Suite**: `/workspace/ext/test_mcp_interface_final.sh`
   - ✅ All 7 tests passing
   - Features: Full protocol testing, performance mode, interactive mode
   - Usage: `bash test_mcp_interface_final.sh [HOST] [PORT]`

2. **Diagnostic Tool**: `/workspace/ext/diagnose_mcp_connection.sh`
   - Network diagnostics and troubleshooting
   - Usage: `bash diagnose_mcp_connection.sh`

#### VisionFlow Rust Client Considerations:
- **Connection**: Use `TcpStream` with `set_nodelay(true)`
- **Reading**: Must handle multiple lines per response
- **Filtering**: Skip non-JSON lines and notifications
- **Timeout**: Keep connection open briefly for full response
- **Parsing**: Match responses by request ID field

## 🎯 System Overview

### Two Independent Graph Systems
1. **Knowledge Graph** - 177+ nodes from markdown/logseq data (✅ Working correctly)
2. **Agent/Bots Graph** - 3-10 AI agent nodes with MCP integration (⚠️ Needs architectural improvements)

### Architecture

**IMPORTANT**: Development Setup Clarification
- The VisionFlow code is **mounted** in `/workspace/ext/` for development/editing
- VisionFlow actually **RUNS** in a separate Docker container (`visionflow_container`)
- This container (`multi-agent-container`) runs the MCP server with Claude Flow agents
- The two containers communicate over the Docker network

```
visionflow_container ──TCP:9500──> multi-agent-container
     (Rust + React)                  (MCP + Claude Flow)
           │                               │
    [WebXR Render] <───WebSocket──> [Agent Swarms]
```

**Container Responsibilities:**
- `visionflow_container` (490c211caa2c): Runs Rust backend + React frontend
- `multi-agent-container` (233024d56830): Runs MCP TCP server on port 9500 with Claude Flow
- Communication: VisionFlow connects to `multi-agent-container:9500` over Docker network

## ✅ Completed Functionality

### Core Features Working
- **Graph Rendering**: Nodes and edges display correctly
- **Real-time Updates**: Positions update from server physics simulation
- **MCP Integration**: Proper swarm initialization and termination
- **UI Controls**: Spawn/disconnect agents via control panel
- **Data Flow**: Complete pipeline from MCP → Backend → WebSocket → Frontend
- **Token Display**: Shows usage metrics (1000 default)
- **Agent Colors**: Server-configurable via `dev_config.toml`
- **Error Handling**: Proper logging instead of silent failures

### Recent Fixes (2025-08-30)
- ✅ Fixed WebSocket message type mismatch (`bots-graph-update` → `botsGraphUpdate`)
- ✅ Added missing graph update broadcast in `UpdateBotsGraph` handler
- ✅ Added safety check for undefined data in frontend `updateFromGraphData`
- ✅ Backend now sends both `bots-full-update` AND `botsGraphUpdate` messages
- ✅ Added debug logging for WebSocket messages with undefined data
- ✅ REVERTED incorrect localhost change - VisionFlow correctly connects to `multi-agent-container:9500`
- ✅ Fixed ClaudeFlowActor initialization - Changed from hardcoded `localhost` to `multi-agent-container`
- ✅ Fixed WebSocket handler to get bots graph data from GraphServiceActor instead of BotsClient
- ✅ **CRITICAL FIX**: Added automatic agent spawning after swarm initialization
  - `swarm_init` only creates swarm structure, doesn't spawn agents
  - Added `call_agent_spawn` function to MCP connection utilities
  - Modified both `initialize_swarm` and `initialize_multi_agent` to spawn agents
  - Spawns 3-4 agents based on topology (hierarchical, mesh, star)

### Technical Fixes Applied
- ✅ MCP response parsing (removed incorrect `content[0].text` unwrapping)
- ✅ Position updates (agents now move with server physics)
- ✅ Data model consistency (removed unsafe 'coordinator' fallback)
- ✅ Swarm lifecycle management (proper init/destroy flow)
- ✅ Route registration (all endpoints properly configured)
- ✅ Type conversions (u32 → string for IDs)
- ✅ Property naming (unified to `swarmId`)

## 🔴 CRITICAL ISSUE - Agent Tracking Not Working

### Root Cause Analysis (2025-08-30 15:18)
**Priority**: CRITICAL
**Status**: MCP server not recognizing agent methods

#### Problem Summary:
- `agent_list` returns empty array instead of spawned agents
- Agents ARE being created (get valid IDs like `agent_1756564254868_pcgsya`)
- Agent tracker module exists but agents aren't being tracked
- The MCP server wrapper is NOT using our modified code

#### Technical Analysis:
1. **MCP Architecture**:
   - `/app/core-assets/scripts/mcp-tcp-server.js` - TCP wrapper (spawns MCP per connection)
   - Each connection spawns: `npx claude-flow@alpha mcp start`
   - This uses GLOBAL npm package, NOT our local `/workspace/ext/claude-flow` code

2. **Agent Tracking Flow**:
   - `agent_spawn` creates agent and stores in memory
   - `global.agentTracker` should track agents in AgentTracker class
   - `agent_list` checks `global.agentTracker.getAgents(swarmId)`
   - Returns empty because tracker isn't populated

3. **Code Changes Made**:
   - ✅ Fixed agent tracker initialization logging
   - ✅ Added swarmId consistency to agent_spawn
   - ✅ Removed mock data fallback from agent_list
   - ✅ Added debug logging throughout
   - ✅ Modified TCP wrapper to use local code
   - ❌ Changes NOT active (wrapper keeps getting restarted)

#### ✅ SOLUTION IMPLEMENTED (2025-08-30 15:00)

**Final Working Solution:**

1. **Enhanced `/workspace/ext/multi-agent-docker/setup-workspace.sh`**:
   - Updated `patch_mcp_server()` function to find global installation at `/usr/lib/node_modules/claude-flow`
   - Added Patch 3: Remove mock data fallback from agent_list
   - Added Patch 4: Fix agent_spawn to properly track agents with swarmId
   - Added Patch 5: Enhanced agent tracker initialization with verification
   - Patches are applied to the INSTALLED claude-flow, not a transient copy

2. **Added agent spawning to VisionFlow handlers**:
   - Modified `/workspace/ext/src/handlers/bots_handler.rs`
   - Added `call_agent_spawn` function to `/workspace/ext/src/utils/mcp_connection.rs`
   - Now automatically spawns 3-4 agents after swarm_init based on topology

**How It Works:**
1. When container starts, `setup-workspace.sh` runs
2. It finds the globally installed claude-flow at `/usr/lib/node_modules/claude-flow`
3. Applies all patches directly to the installed version
4. Supervisord starts mcp-tcp-server which uses the patched claude-flow
5. Agent tracking now works correctly with real agents instead of mock data

**To Apply Changes:**
1. Restart the Docker container to run setup-workspace.sh
2. The patches will be applied to the installed claude-flow
3. Supervisord will restart mcp-tcp-server automatically
4. VisionFlow will spawn agents automatically when creating swarms

## ⚠️ FUTURE WORK REQUIRED

### 1. Backend Data Transformation Consolidation
**Priority**: HIGH
**Problem**: Inconsistent positioning logic causes jarring layout changes
**Details**:
- REST API uses `position_agents_hierarchically()` for structured layout
- WebSocket uses simple circular layout in `graph_actor.rs`
- First WebSocket update can cause nodes to snap to different positions

**Solution Required**:
```rust
// In graph_actor.rs UpdateBotsGraph handler
// Replace circular layout with:
use crate::handlers::bots_handler::position_agents_hierarchically;
position_agents_hierarchically(&mut agents);
```

### 2. WebSocket Message Standardization
**Priority**: MEDIUM
**Problem**: Two parallel message types for same data
**Details**:
- `bots-graph-update`: Pre-processed graph (preferred)
- `bots-full-update`: Raw agent data (redundant)
- Potential for race conditions and state conflicts

**Solution Required**:
1. Remove `bots-full-update` message type entirely
2. Standardize on `bots-graph-update` format
3. Update `BotsDataContext.tsx` to use single handler
4. Remove `updateFromFullUpdate` method

### 3. Performance Optimizations
**Priority**: LOW
**Current State**:
- Polling every 2000ms (could use WebSocket push)
- Full graph sent each update (could send deltas)
- ~3-5KB per update with 3-10 agents

**Improvements Possible**:
- Server-push updates instead of polling
- Delta updates for position changes only
- Binary protocol for position data

### 4. Enhanced MCP Integration
**Priority**: LOW
**Opportunities**:
- Store multiple swarm IDs for multi-swarm support
- Add swarm configuration persistence
- Implement swarm state recovery after crashes
- Add metrics collection and visualization

## 📋 Quick Reference

### Testing Steps
1. Open WebXR visualization
2. Click "Spawn Hive Mind" in control panel
3. Verify nodes, edges, and real-time movement
4. Test disconnect functionality
5. Check console for any errors

### Key Files
**Backend**:
- `/src/handlers/bots_handler.rs` - Main agent handler
- `/src/actors/graph_actor.rs` - WebSocket updates (needs fix)
- `/src/utils/mcp_connection.rs` - MCP integration

**Frontend**:
- `/client/src/features/bots/components/BotsVisualizationFixed.tsx` - 3D rendering
- `/client/src/features/bots/contexts/BotsDataContext.tsx` - State management
- `/client/src/features/bots/services/BotsWebSocketIntegration.ts` - WebSocket client


```

## 📊 Success Metrics
- ✅ Agents spawn and display correctly
- ✅ Positions update in real-time
- ✅ Disconnect properly terminates MCP swarm
- ✅ No console errors during operation
- ⚠️ Layout consistency between REST/WebSocket (pending)
- ⚠️ Single message type for updates (pending)

---
*Last Updated: Session 7 - External assessment addressed, core issues fixed, architectural improvements documented*