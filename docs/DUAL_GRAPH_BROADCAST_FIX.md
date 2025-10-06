# Dual-Graph WebSocket Broadcast Fix

**Date:** 2025-10-06
**Status:** ✅ FIXED
**Last Updated:** 2025-10-06 (Removed conflicting immediate GPU broadcast)

## Problem Summary

The system has dual parallel graph structures (knowledge graph + agent graph) but was broadcasting them **separately**, causing WebSocket conflicts and incorrect visualization movement.

### Root Cause (Fixed in Two Phases)

**Graph Actor Structure:**
- `graph_data` / `node_map`: Knowledge graph (185 nodes)
- `bots_graph_data`: Agent graph (3 nodes)

**Phase 1 - Original Broken Flow:**
1. **Physics Loop** → Broadcast ONLY knowledge nodes → NO type flags
2. **UpdateBotsGraph Handler** → Broadcast ONLY agent nodes → WITH AGENT_NODE_FLAG
3. **Client receives TWO conflicting broadcasts**

**Phase 2 - Regression Bug (FIXED):**
After initial fix, an "immediate GPU broadcast" was added that:
1. **Unified Physics Loop** → Broadcast BOTH graphs with flags ✅
2. **Immediate GPU Broadcast** → Broadcast ONLY knowledge nodes → NO type flags ❌
3. **Client receives conflicting messages again**

This regression was introduced during agent management refactoring and has now been removed.

## Solution Implemented

### Change 1: Unified Physics Broadcast (`graph_actor.rs:2087-2122`)

**Modified** the physics broadcast loop to collect BOTH graphs in ONE message:

```rust
// Collect knowledge graph nodes
for (node_id, node) in self.node_map.iter() {
    position_data.push(...);
    knowledge_ids.push(*node_id);
}

// ALSO collect agent graph nodes
for node in &self.bots_graph_data.nodes {
    position_data.push(...);
    agent_ids.push(node.id);
}

// Encode with BOTH type flags
let binary_data = encode_node_data_with_types(
    &position_data,
    &agent_ids,
    &knowledge_ids
);
```

**Result:**
- Single broadcast with 188 nodes (185 knowledge + 3 agents)
- Knowledge nodes flagged with `KNOWLEDGE_NODE_FLAG` (bit 30 = 0x40000000)
- Agent nodes flagged with `AGENT_NODE_FLAG` (bit 31 = 0x80000000)

### Change 2: Remove Duplicate Broadcast (`graph_actor.rs:3186-3204`)

**Removed** the separate agent broadcast from UpdateBotsGraph handler:

```rust
// OLD CODE (removed):
// - Collected agent positions
// - Called encode_node_data_with_flags()
// - Broadcast separately

// NEW CODE:
// Just update bots_graph_data
// Physics loop will pick it up automatically
debug!("Agent graph data updated. Physics loop will broadcast with AGENT_NODE_FLAG.");
```

## Binary Protocol Usage

### Type Flags (Protocol V2)

Defined in `src/utils/binary_protocol.rs`:

```rust
const AGENT_NODE_FLAG: u32 = 0x80000000;       // Bit 31
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000;   // Bit 30
const NODE_ID_MASK: u32 = 0x3FFFFFFF;          // Bits 0-29
```

### Encoding Function

```rust
pub fn encode_node_data_with_types(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32]
) -> Vec<u8>
```

- Checks each node ID against both lists
- Sets `AGENT_NODE_FLAG` if in `agent_node_ids`
- Sets `KNOWLEDGE_NODE_FLAG` if in `knowledge_node_ids`
- Client decodes flags to separate graph types

### Client-Side Decoding

Client code (`BinaryWebSocketProtocol.ts`) uses:

```typescript
const AGENT_NODE_FLAG = 0x8000;       // Bit 15 in V1 (0x80000000 in V2)
const KNOWLEDGE_NODE_FLAG = 0x4000;   // Bit 14 in V1 (0x40000000 in V2)

const isAgent = (nodeId & AGENT_NODE_FLAG) !== 0;
const isKnowledge = (nodeId & KNOWLEDGE_NODE_FLAG) !== 0;
```

## Architecture Compliance

This fix brings the implementation in line with the documented architecture:

**From `docs/architecture/core/visualization.md`:**
> VisionFlow supports dual-graph visualisation where both **Knowledge Graphs** and **Agent Graphs** coexist in the same 3D scene.

**From `docs/reference/api/binary-protocol.md`:**
> Node types are distinguished by control bits in the node_id field

**From `docs/architecture/core/client.md` line 292:**
> The system implements a unified graph with protocol support for dual types

## Expected Behavior After Fix

### Server Side

**Physics Loop Logs:**
```
INFO Sent initial unified graph positions to clients (188 nodes: 185 knowledge + 3 agents)
DEBUG Broadcast unified positions: 188 total (185 knowledge + 3 agents), stable: false, pending: 1/10
```

**UpdateBotsGraph Logs:**
```
INFO Updated bots graph with 3 agents and 6 edges - data will be broadcast in next physics cycle
DEBUG Agent graph data updated (3 nodes). Physics loop will broadcast with AGENT_NODE_FLAG.
```

### Client Side

**WebSocket Receives:**
- ONE binary message with 188 nodes
- Each node has proper type flag
- Client separates via `is_agent_node()` / `is_knowledge_node()`
- Knowledge graph and agent graph render separately but synchronized

### Visualization

**Knowledge Graph (185 nodes):**
- GPU physics simulation
- Force-directed layout
- Rendered with instanced meshes

**Agent Graph (3 nodes):**
- GPU physics simulation (coexisting)
- Server-computed topology positions
- Rendered with individual meshes for animations

**No more conflicts or erratic movement!**

## Files Modified

### Phase 1 (Initial Fix)
1. **src/actors/graph_actor.rs**
   - Lines 2087-2122: Modified physics broadcast to collect both graphs
   - Lines 2132-2149: Updated log messages
   - Lines 3186-3204: Removed duplicate broadcast from UpdateBotsGraph

### Phase 2 (Regression Fix - Complete)
1. **src/actors/graph_actor.rs**
   - Lines 2155-2160: **REMOVED** conflicting "immediate GPU broadcast" section
   - Lines 2717-2770: **FIXED** `ForcePositionBroadcast` handler to use unified dual-graph encoding
   - Lines 2776-2828: **FIXED** `InitialClientSync` handler to use unified dual-graph encoding

**Issue:** Three separate broadcast paths were conflicting:
   1. ✅ Unified physics loop broadcast (lines 2087-2150) - **CORRECT**
   2. ❌ Immediate GPU broadcast (removed) - **WAS BROKEN**
   3. ❌ ForcePositionBroadcast handler - **WAS BROKEN**
   4. ❌ InitialClientSync handler - **WAS BROKEN**

**Fix:** Removed #2, fixed #3 and #4 to use `encode_node_data_with_types()` with both agent and knowledge node lists

## Testing Checklist

- [x] Code compiles successfully (`cargo check`)
- [ ] Backend starts without errors
- [ ] Spawn agents via UI
- [ ] Check backend logs for unified broadcast messages
- [ ] Verify client receives single WebSocket stream
- [ ] Confirm both graphs render correctly without conflicts
- [ ] Test agent movement is smooth and synchronized

## Next Steps

1. Restart visionflow docker container
2. Spawn test agents via UI
3. Monitor logs for unified broadcast messages
4. Verify visualization shows both graphs moving correctly

---

**Implementation Status:** READY FOR TESTING
**Compilation:** ✅ PASSED
**Architecture Compliance:** ✅ ALIGNED
