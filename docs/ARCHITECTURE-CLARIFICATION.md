# Architecture Clarification - Multi-Agent Docker System

## 🎯 WHERE WE ARE NOW
**We are currently inside: `multi-agent-container` (172.18.0.3)**
- This is the MCP/Agent orchestration container
- Contains the MCP TCP server on port 9500
- Has claude-flow, ruv-swarm, flow-nexus running
- This is where agents are created and managed

## 🖥️ WHERE WEBXR RUNS
**WebXR application runs in: `logseq container` (NOT listed in network inspect)**
- The logseq container is on the same docker_ragflow network
- WebXR code is at `/workspace/ext` in THAT container
- WebXR needs to connect HERE to `multi-agent-container:9500`

## 🔄 THE CONNECTION FLOW

```
┌─────────────────────────────────────┐
│  LOGSEQ CONTAINER (WebXR)           │
│  - Rust backend (Actix)             │
│  - TypeScript frontend              │
│  - BotsClient                       │
│  - ClaudeFlowActor                  │
└─────────────┬───────────────────────┘
              │
              │ TCP Connection to port 9500
              │ (needs to connect HERE)
              ↓
┌─────────────────────────────────────┐
│  MULTI-AGENT-CONTAINER (HERE)       │
│  IP: 172.18.0.3                     │
│  - MCP TCP Server (port 9500) ✅    │
│  - claude-flow MCP server ✅        │
│  - Agent storage/memory ✅          │
│  - We are working HERE now          │
└─────────────────────────────────────┘
```

## 📍 CRITICAL UNDERSTANDING

### From WebXR's Perspective (in logseq container):
- **Correct connection**: `multi-agent-container:9500` or `172.18.0.3:9500`
- **Wrong connection**: `localhost:9500` (would look for MCP in logseq container)

### From Our Perspective (HERE in multi-agent-container):
- MCP server is on `localhost:9500` (because we're inside this container)
- WebXR must connect to us at `multi-agent-container:9500`

## 🔧 WHAT NEEDS TO BE FIXED IN WEBXR

The WebXR code (in the logseq container) needs to:

1. **BotsClient** (`/workspace/ext/src/services/bots_client.rs` in logseq):
   ```rust
   // CORRECT - connects to multi-agent-container
   let host = "multi-agent-container";
   let port = 9500;
   ```

2. **ClaudeFlowActor** (`/workspace/ext/src/actors/claude_flow_actor.rs` in logseq):
   ```rust
   // CORRECT - uses multi-agent-container
   let host = std::env::var("CLAUDE_FLOW_HOST")
       .unwrap_or_else(|_| "multi-agent-container".to_string());
   ```

## 🧪 TEST COMMANDS

### From THIS container (multi-agent-container):
```bash
# Test MCP locally (works because we're here)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | nc localhost 9500

# Check what's listening
netstat -tuln | grep 9500
```

### From LOGSEQ container (where WebXR runs):
```bash
# Test connection to multi-agent-container
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | nc multi-agent-container 9500

# Or using IP
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | nc 172.18.0.3 9500
```

## 📝 SUMMARY FOR CLARITY

- **We are HERE**: multi-agent-container (172.18.0.3)
- **MCP server runs HERE**: port 9500 in THIS container
- **WebXR runs THERE**: logseq container (different container)
- **WebXR connects to US**: via `multi-agent-container:9500`
- **We're fixing**: WebXR code to properly connect HERE

## 🎯 ACTION ITEMS

1. **Update WebXR connection strings** to use `multi-agent-container` not `localhost`
2. **Ensure network connectivity** between logseq and multi-agent containers
3. **Test from logseq container** to verify connection works
4. **Update UpdateBotsGraph** flow to send agent data to WebXR visualization

This is the correct architecture - WebXR in logseq container needs to connect to the MCP server running HERE in multi-agent-container!