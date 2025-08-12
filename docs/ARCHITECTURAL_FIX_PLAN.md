# Architectural Fix Plan - Remove Incorrect Claude-Flow NPM Integration

## Problem Summary

The Rust backend server is incorrectly trying to spawn `claude-flow` as a subprocess via `npx` instead of connecting to the separate `multi-agent-container` service running on the Docker network. This causes:

1. **JSON Parse Errors**: The backend expects JSON but gets stdout text like "✅ Starting Claude Flow MCP server in stdio mode..."
2. **Method Not Found Errors**: Protocol mismatch between what backend expects and what the subprocess provides
3. **502 Bad Gateway**: Nginx can't reach the backend because it's crashing from these errors

## Root Cause

The `StdioTransport` in `/workspace/ext/src/services/claude_flow/transport/stdio.rs` is spawning:
```rust
Command::new("npx")
    .args(&["claude-flow@alpha", "mcp", "start", "--stdio"])
```

This is WRONG - the backend should connect to `multi-agent-container:3002` via WebSocket.

## Architecture Clarification

### Correct Architecture:
```
Client (Browser) 
    ↓ (via Nginx proxy)
Rust Backend (webxr container)
    ↓ (WebSocket to port 3002)
multi-agent-container (separate Docker container)
    - Runs the actual MCP/Claude-Flow server
    - Provides agent telemetry and control
```

### Current (Incorrect) Architecture:
```
Rust Backend trying to spawn claude-flow as subprocess
    ↓ (stdio pipe)
npx claude-flow (doesn't exist in Rust container!)
```

## Fix Implementation

### 1. Remove StdioTransport Usage
- Delete or disable `StdioTransport` class entirely
- Remove `use_stdio()` method from `ClaudeFlowClientBuilder`
- Ensure NO fallback to stdio when WebSocket fails

### 2. Fix WebSocket Connection
- Always use WebSocket to connect to `multi-agent-container:3002`
- Proper endpoints: `/ws` or `/mcp` (need to verify with container)
- Handle connection failures gracefully without subprocess fallback

### 3. Update Environment Variables
```bash
CLAUDE_FLOW_HOST=multi-agent-container
CLAUDE_FLOW_PORT=3002
BOTS_ORCHESTRATOR_URL=ws://multi-agent-container:3002/ws
```

### 4. Remove NPM Dependencies from Rust Container
- Remove any `package.json` or `node_modules` from the Rust backend
- Update Dockerfiles to not install Node.js/NPM in backend container
- Backend should be pure Rust + GPU libraries only

### 5. Fix bots_handler.rs
- Ensure it uses the WebSocket client, not stdio subprocess
- Remove any references to spawning processes

## Files to Modify

1. **Remove/Disable StdioTransport**:
   - `/workspace/ext/src/services/claude_flow/transport/stdio.rs` - Delete or comment out
   - `/workspace/ext/src/services/claude_flow/transport/mod.rs` - Remove stdio export
   - `/workspace/ext/src/services/claude_flow/client.rs` - Remove stdio builder methods

2. **Fix Connection Logic**:
   - `/workspace/ext/src/app_state.rs` - Ensure WebSocket only, no fallback
   - `/workspace/ext/src/actors/claude_flow_actor_enhanced.rs` - Use WebSocket to multi-agent-container
   - `/workspace/ext/src/services/bots_client.rs` - Already correct, connects to ws://multi-agent-container:3002/ws

3. **Clean Dockerfiles**:
   - `/workspace/ext/Dockerfile.dev` - Remove Node.js/NPM installation
   - `/workspace/ext/Dockerfile.production` - Remove Node.js/NPM installation

4. **Update Scripts**:
   - `/workspace/ext/scripts/start-backend-with-claude-flow.sh` - Remove or update

## Testing

After fixes:
1. Backend should connect to `ws://multi-agent-container:3002/ws`
2. No more "Starting Claude Flow MCP server" messages in logs
3. No more JSON parse errors
4. API endpoints should return 200 OK
5. WebSocket should maintain stable connection

## Commands to Verify Multi-Agent Container

```bash
# Check if multi-agent-container is running
docker ps | grep multi-agent-container

# Test connection from backend container
docker exec -it <webxr-container> curl http://multi-agent-container:3002/health

# Check multi-agent-container logs
docker logs multi-agent-container
```