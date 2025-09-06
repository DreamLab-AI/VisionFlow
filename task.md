# WebXR Agent Graph System - Current Status & Future Work

**Date**: 2025-09-06
**Status**: 🟡 **MOSTLY COMPLETE** - Core functionality working, architectural improvements pending
**Environment**: `multi-agent-container` + `visionflow_container`

## ✅ RESOLVED - Rust App Running (2025-09-06 20:50)

### Issue Summary:
The Rust backend in the logseq container is now running successfully.

### Root Causes Fixed:
1. **Configuration Issue**: Missing `z` coordinate in camera `lookAt` field in `/workspace/ext/data/settings.yaml` - **FIXED**
2. **CUDA Error**: Previously crashed with `cudaErrorSymbolNotFound` - **RESOLVED**

### Current Status:
✅ Rust app is running without crashes
✅ No more CUDA termination errors
⚠️ GPU compute context warnings remain but are non-fatal
✅ WebXR backend operational

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

## ✅ SPAWN AGENT BUTTON INVESTIGATION - COMPLETE

### Message Flow Analysis (2025-09-06)
**Status**: ✅ **FULLY TRACED AND DOCUMENTED**
**Investigation**: Complete trace from UI button click to MCP TCP communication

#### Frontend → Backend → MCP TCP Flow:

**1. Frontend UI (React/TypeScript)**
- **File**: `/client/src/features/visualisation/components/IntegratedControlPanel.tsx`
- **Buttons**:
  - `"Initialize multi-agent"` (line 934) - when no agents exist
  - `"New multi-agent Task"` (line 983) - when agents exist
- **Action**: Both buttons call `setshowmultiAgentPrompt(true)`

**2. Spawn Dialog Component**
- **File**: `/client/src/features/bots/components/MultiAgentInitializationPrompt.tsx`
- **API Call**: `POST /bots/initialize-multi-agent` (line 118)
- **Payload**:
```typescript
{
  topology: "mesh" | "hierarchical" | "ring" | "star",
  maxAgents: number,
  strategy: "adaptive",
  enableNeural: boolean,
  agentTypes: string[],
  customPrompt: string
}
```

**3. Rust Backend Handler**
- **File**: `/src/handlers/bots_handler.rs`
- **Function**: `initialize_multi_agent()` (line 1329)
- **MCP Connection**: Connects to `multi-agent-container:9500`
- **Process**:
  1. Calls `call_swarm_init()` with topology parameters
  2. Spawns agents via `call_agent_spawn()` based on topology
  3. Stores swarm ID for later disconnection

**4. MCP Connection Utilities**
- **File**: `/src/utils/mcp_connection.rs`
- **Functions**:
  - `call_swarm_init()` (line 315) - Creates swarm
  - `call_agent_spawn()` (line 357) - Spawns individual agents
- **Protocol**: JSON-RPC 2.0 over TCP
- **Message Format**:
```json
{
  "jsonrpc": "2.0",
  "id": "uuid",
  "method": "tools/call",
  "params": {
    "name": "swarm_init",
    "arguments": {
      "topology": "mesh",
      "maxAgents": 8,
      "strategy": "adaptive"
    }
  }
}
```

**5. MCP TCP Server**
- **File**: `/multi-agent-docker/core-assets/scripts/mcp-tcp-server.js`
- **Port**: 9500 (configurable via MCP_TCP_PORT)
- **Process**: Spawns `npx claude-flow@alpha mcp start` per connection
- **Location**: Runs in `multi-agent-container` Docker container

**6. Claude Container Execution**
- **Command**: `npx claude-flow@alpha mcp start --stdio --file /workspace/.mcp.json`
- **Tools**: Executes actual swarm_init, agent_spawn MCP tools
- **Response**: Returns agent IDs and swarm configuration

#### Message Protocol Details:
- **Transport**: TCP socket on port 9500
- **Format**: Line-delimited JSON-RPC 2.0
- **Connection**: Fresh connection per command (no persistent pooling)
- **Timeout**: 10 seconds per command
- **Retry**: Up to 3 attempts with 500ms delay

#### Key Environment Variables:
- `CLAUDE_FLOW_HOST` / `MCP_HOST` → `multi-agent-container`
- `MCP_TCP_PORT` → `9500`
- `CLAUDE_FLOW_DIRECT_MODE` → `true` (in MCP server)

## ⚠️ PARTIALLY RESOLVED - Agent Tracking System Issues Persist

### Resolution Summary (2025-09-06 20:38)
**Priority**: HIGH
**Status**: Configuration fixes applied, agent tracking still returns mock data

#### Problem Summary:
- `agent_list` was returning mock data (agent-1, agent-2, agent-3) instead of real agents
- Root cause: Multi-layered architecture issue with process isolation
- Agents ARE persisted to database but query filtering fails

#### Root Causes Identified:
1. **Process Isolation**:
   - Each TCP connection spawns a NEW MCP process
   - Agent tracker is per-process (not shared)
   - Database persistence works but query fails

2. **Multiple Issues Fixed**:
   - TCP server was using `npx claude-flow@alpha` (downloading fresh copy each time)
   - Agent tracker wasn't initialized in constructor
   - Mock data fallback was masking the real issue
   - Variable scoping conflicts in switch statements
   - Database query filtering doesn't match key structure

3. **Fixes Applied**:
   - ✅ Changed TCP server to use global installation `/usr/bin/claude-flow`
   - ✅ Added agent tracker initialization in constructor
   - ✅ Removed mock data fallback
   - ✅ Fixed variable naming conflicts (swarmId → listSwarmId)
   - ✅ Updated database query to use list() with filtering
   - ⚠️ Database filtering still needs work (finds 29 total agents but 0 for current swarm)

#### ✅ UPDATED SOLUTION (2025-09-06 20:30)

**Updated `/workspace/ext/setup-workspace.sh`**:

The setup script now includes comprehensive patches:

1. **Patch 1**: Fix hardcoded version (dynamic from package.json)
2. **Patch 2**: Fix method routing for direct tool calls  
3. **Patch 3**: Fix agent_list to check database with proper filtering
4. **Patch 4**: Fix agent tracker module export
5. **Patch 5**: Initialize agent tracker in constructor
6. **Patch 6**: Store active swarm ID when creating swarm
7. **Patch 7**: Fix TCP server to use global claude-flow instead of npx

**Key Discoveries**:
- Agents ARE being persisted to SQLite database at `/workspace/.swarm/memory.db`
- Database keys format: `agent:swarm_ID:agent_ID`
- The memoryStore.list() returns all entries but filtering logic needs adjustment
- Each TCP connection gets its own MCP process with isolated memory

**Current Status (2025-09-06 20:50)**:
- ✅ TCP server running on port 9500
- ✅ Multiple swarms created and persisted to database
- ✅ Agents ARE persisted to `/workspace/.swarm/memory.db`
- ❌ Agent tracking STILL returns mock data - verified with test script
- ❌ `agent_list` returns hardcoded agents (agent-1, agent-2, agent-3)
- ❌ `swarm_status` shows 0 agents even for swarms with 5+ agents in DB
- 📝 Found multiple active swarms with real agents in database:
  - `swarm_1757190236313_k0cl75zp7` - Latest with 5+ agents
  - `swarm_1757190098787_ji7urxyhb` - Previous with 5+ agents
  - Database contains real agent data but MCP can't query it properly

**Next Steps for Full Resolution**:
1. Fix the database key filtering logic to match actual key structure
2. Consider implementing cross-process agent sharing via Redis or shared memory
3. Add connection pooling to reuse MCP processes

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

## 📊 Client-Side Rendering System Analysis - COMPLETE ✅

**Date**: 2025-09-06
**Status**: ✅ **FULLY ANALYZED**
**Analysis**: Comprehensive investigation of force-directed graph visualization system

### 🎨 3D Visualization Architecture

#### Core Libraries & Dependencies
1. **Three.js v0.175.0** - Primary 3D rendering engine
2. **@react-three/fiber v8.15.0** - React reconciler for Three.js
3. **@react-three/drei v9.80.0** - Utility components (Html, Text, Billboard, Line)
4. **@react-three/postprocessing v2.15.0** - Post-processing effects
5. **@react-three/xr v6.0.0** - WebXR/AR support for Quest 3

#### WebGL Rendering Pipeline
- **Custom Geometries**: Dynamic sphere/tetrahedron/box based on agent status
- **Material System**: PBR materials with emissive properties for glowing effects
- **Instanced Rendering**: Optimized for multiple similar objects (from dualGraphOptimizations.ts)
- **LOD System**: Distance-based detail levels (high/medium/low/hidden at 20/50/100 unit thresholds)
- **Frustum Culling**: Automatic visibility optimization using THREE.Frustum
- **Spatial Partitioning**: Octree implementation for large node counts (8 levels, 10 objects/node)

### 🔌 WebSocket Connection Architecture

#### Primary WebSocket Service (`WebSocketService.ts`)
- **Connection URL**: Auto-detected (dev: ws://host:3001/wss, prod: /wss)
- **Protocol**: JSON for control messages, Binary (ArrayBuffer) for position updates
- **Reconnection**: Exponential backoff (base 2s, max 10 attempts, 1.5x multiplier)
- **Message Types**:
  - `connection_established` - Server ready signal
  - `graph-update` - Logseq graph updates (JSON)
  - `botsGraphUpdate` - Agent graph with nodes/edges (JSON)
  - `bots-full-update` - Legacy full agent data (JSON)
  - Binary position streams - Server-computed physics (ArrayBuffer)

#### Bots WebSocket Integration (`BotsWebSocketIntegration.ts`)
- **Polling Strategy**: 2-second interval for graph data requests
- **Event System**: Observable pattern with typed handlers
- **Message Flow**:
  1. `requestBotsGraph` → Backend
  2. Backend → `botsGraphUpdate` with processed graph data
  3. Context updates → React re-renders

### 🔄 Real-Time Update Mechanism

#### Server-Authoritative Physics Model
1. **Backend Computation**: Rust server computes all physics (force-directed layout)
2. **Binary Protocol**: 28-byte records (nodeId:4, position:12, velocity:12)
3. **Compression**: Optional zlib compression for large datasets
4. **Client Interpolation**: Smooth animation between server positions using exponential smoothing

#### Data Flow Pipeline
```
MCP Server → Rust Backend → Graph Actor → WebSocket → React Context → Three.js Rendering
    ↓            ↓             ↓            ↓            ↓              ↓
Agent Spawn  Physics Sim   JSON Format   Binary Pos   State Update   Visual Update
```

#### Binary Position Protocol (`binaryProtocol.ts`)
- **Node Type Flags**: Bit 31 (agent), Bit 30 (knowledge), Bits 0-29 (actual ID)
- **Format**: Little-endian IEEE 754 float32
- **Detection**: Automatic agent vs knowledge node classification
- **Validation**: NaN/Infinity checks, corrupted data recovery

### 🎯 Force-Directed Graph Visualization

#### Core Component: `BotsVisualizationFixed.tsx`
- **Server-Authoritative Positioning**: No client-side physics computation
- **Position Interpolation**: Smooth animation using `useFrame` hook
- **Node Representations**:
  - Geometry: Status-based (sphere/tetrahedron/box)
  - Size: CPU usage influenced (1.0 + cpuUsage/100 * 0.5)
  - Color: Agent type mapping from server config
  - Glow: Health-based color (green/orange/red at 80/50% thresholds)

#### Visual Enhancement Features
1. **Animated Status Badges**: HTML overlays with real-time metrics
2. **Token Flow Visualization**: Edge thickness/color based on communication rate
3. **Activity Indicators**: Pulsing/rotation based on token rate and agent status
4. **Processing Logs**: Mock activity generation per agent type
5. **3D Text Labels**: Billboard-aligned agent names
6. **Connection Lines**: Dynamic edge rendering with activity-based styling

### ⚡ Performance Optimizations

#### `dualGraphOptimizations.ts` Features
1. **Frustum Culling**: THREE.Frustum-based visibility testing
2. **Level of Detail**: Distance-based geometry quality (32→16→8 segments)
3. **Instanced Rendering**: Batched rendering for identical objects
4. **SharedArrayBuffer**: Worker thread communication (when supported)
5. **Spatial Partitioning**: Octree for efficient neighbor queries
6. **Memory Management**: Geometry/material pooling and disposal

#### Worker Thread Architecture (`graph.worker.ts`)
- **Comlink Integration**: Type-safe worker communication
- **Animation Interpolation**: Client-side smoothing of server positions
- **Physics Modes**: Server-authoritative vs local simulation toggle
- **User Interaction**: Drag handling with position pinning
- **State Management**: Current/target position buffers with velocity

### 🔧 Data Context & State Management

#### `BotsDataContext.tsx`
- **Provider Pattern**: React context for global state
- **Data Transformation**: Backend graph format → React component props
- **Edge Processing**: u32 node IDs → string agent IDs mapping
- **Metadata Parsing**: Server snake_case → client camelCase conversion
- **Real-time Updates**: WebSocket event → context update → component re-render

#### Agent Data Structure (`BotsTypes.ts`)
- **15+ Agent Types**: Including queen, coordinator, researcher, coder, etc.
- **Comprehensive Metrics**: Health, CPU, memory, tokens, success rate
- **3D Properties**: Position, velocity, force vectors
- **Communication**: Edge data with volume/rate/timestamp
- **Visual Config**: Colors, physics, sizes per agent type

### 🚀 Key Performance Metrics

#### Rendering Optimizations
- **Instance Batching**: Up to 5000 objects per batch
- **Culling Efficiency**: Distance-based hiding beyond 150 units
- **LOD Transitions**: Smooth quality changes at 20/50/100 unit boundaries
- **Memory Usage**: Geometry/material pooling reduces allocation overhead

#### Network Efficiency
- **Binary Protocol**: ~28 bytes per agent position update
- **Compression**: Zlib for large position datasets
- **Polling Rate**: 2-second intervals (configurable)
- **Message Size**: ~3-5KB for typical 3-10 agent graphs

#### Animation Smoothness
- **Interpolation**: Exponential smoothing (factor: 1.0 - exp(-8.0 * dt))
- **Frame Rate**: 60fps target with deltaTime clamping (max 16ms)
- **State Synchronization**: Server positions → client interpolation → smooth movement

### 💡 Advanced Features

#### WebXR Integration
- **Quest 3 Support**: Native VR/AR graph exploration
- **Hand Tracking**: Direct agent interaction in 3D space
- **Spatial Computing**: Real-world coordinate anchoring

#### Accessibility & Debug
- **Debug System**: Configurable logging levels and data inspection
- **Error Resilience**: Graceful fallbacks for corrupted data/connection issues
- **Performance Monitoring**: Real-time metrics collection and analysis

---
*Last Updated: Session 8 - Client-side rendering system fully analyzed and documented*