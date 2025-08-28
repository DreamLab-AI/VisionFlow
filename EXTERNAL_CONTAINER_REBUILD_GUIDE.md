# External Container (visionflow_container) Rebuild Guide

## Summary of Fixes Applied

The MCP connection between the WebXR application (visionflow_container) and the Claude Flow multi-agent system (multi-agent-container) is now working. The following changes need to be incorporated into the external container rebuild:

## 1. Rust Backend Changes

### File: `/app/src/handlers/bots_handler.rs`

**Key Changes:**
- Enhanced MCP response handling to skip notifications and wait for matching response IDs
- Direct method calls (no tools/invoke wrapper needed with patched server)
- Increased timeout to 10 seconds with 10 retry attempts
- Extensive debug logging for troubleshooting

**Critical Code Section (lines 1250-1392):**
```rust
// Send swarm_init with direct method call
let json_request = json!({
    "jsonrpc": "2.0",
    "method": "swarm_init",  // Direct method call
    "params": {
        "topology": request.topology.clone(),
        "maxAgents": request.max_agents,
        "strategy": request.strategy.clone()
    },
    "id": request_id
});

// Enhanced response handling with notification filtering
loop {
    let mut response = String::new();
    match reader.read_line(&mut response) {
        Ok(0) => {
            error!("MCP connection closed while waiting for response");
            break;
        }
        Ok(_) => {
            // Parse and check if this is our response
            if let Ok(parsed) = serde_json::from_str::<Value>(&response) {
                if let Some(id) = parsed.get("id") {
                    if id == &json!(request_id) {
                        // This is our response!
                        break;
                    }
                }
            }
        }
        Err(e) => {
            error!("Error reading MCP response: {}", e);
            break;
        }
    }
}
```

## 2. Environment Configuration

### File: `/etc/supervisor/conf.d/supervisord.conf`

**Critical Fix:**
```conf
environment=CLAUDE_FLOW_HOST=multi-agent-container
```
**NOT** `CLAUDE_FLOW_HOST=172.18.0.10` (hardcoded IP was wrong)

## 3. Test Scripts to Include

Copy these test scripts to `/app/` in the visionflow_container:

### `/app/test_visionflow.sh`
- Basic MCP connection test
- Tests DNS resolution, TCP connectivity, and basic request/response

### `/app/test_swarm_init_direct.sh`
- Tests direct method calls to swarm_init
- Tests both direct and tools/call formats
- Validates the patches are working

## 4. Expected Working Flow

1. **WebXR Frontend** clicks "Spawn Hive Mind"
2. **Rust Backend** receives POST to `/api/v1/tools/agents/init`
3. **Backend** connects to `multi-agent-container:9500` via TCP
4. **Backend** sends initialize request, then swarm_init request
5. **MCP Server** (with patches) accepts direct method call
6. **Swarm** is initialized with unique ID like `swarm_1756398867416_j2i7fu2ve`
7. **Backend** returns success to frontend
8. **Frontend** updates UI from "MOCK" to "LIVE" with green indicator

## 5. Verification Steps

After rebuilding the external container:

1. Check connection:
```bash
docker exec -it visionflow_container bash
cd /app
./test_visionflow.sh
```

2. Test swarm initialization:
```bash
./test_swarm_init_direct.sh
```

3. Check from WebXR UI:
- Click "Spawn Hive Mind" button
- Should see status change to "VisionFlow (LIVE)" with green dot
- Check browser console for successful swarm ID

## 6. Known Issues Still Pending

1. **GPU Initialization Warning**: 
   - Error: "GPU NOT INITIALIZED! Cannot update graph data"
   - Affects graph visualization but not core functionality

2. **Multiple Swarm Instances**:
   - Need to track and manage multiple swarm IDs
   - UI needs to display/select active swarm

## 7. Docker Network Configuration

Ensure both containers are on the same network:
```yaml
networks:
  default:
    external:
      name: docker_ragflow
```

Both containers should be able to resolve each other by name:
- `multi-agent-container` → MCP server host
- `visionflow_container` → WebXR client

## Success Criteria

✅ MCP connection established (no more "Connection refused")
✅ Swarm initialization returns valid swarm ID
✅ UI updates to show "LIVE" status
✅ No hardcoded IPs in configuration
✅ Proper error handling and logging

## Notes on MCP Server Patches

The multi-agent-container has patched MCP server to support:
1. Dynamic version reading (shows correct 2.0.0-alpha.101)
2. Direct method routing (swarm_init works without tools/call wrapper)

These patches are automatically applied via `/app/setup-workspace.sh` on container start.