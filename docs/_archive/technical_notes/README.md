# Development Notes - October 2025

This directory contains development reports and fix documentation from October 2025. These documents capture the technical decisions and debugging process during major system upgrades.

## Contents

### AGENT_CONTROL_AUDIT.md
**Topic:** Agent management implementation audit
**Date:** 2025-10-06
**Status:** Integrated into `docs/architecture/hybrid_docker_mcp_architecture.md`

Documents the implementation of real MCP session spawning, agent control endpoints, UUID ↔ swarm_id correlation, and GPU integration for agent visualization.

### DUAL_GRAPH_BROADCAST_FIX.md
**Topic:** Dual-graph WebSocket broadcast conflict resolution
**Date:** 2025-10-06
**Status:** Integrated into `docs/architecture/components/websocket-protocol.md`

Describes the fix for conflicting WebSocket broadcasts between knowledge graph and agent graph. Solution: unified broadcast with type flags at bits 31/30.

### PROTOCOL_V2_UPGRADE.md
**Topic:** Binary protocol upgrade from V1 to V2
**Date:** 2025-10-06
**Status:** Integrated into `docs/reference/api/binary-protocol.md`

Documents the upgrade from 34-byte (u16 IDs) to 36-byte (u32 IDs) binary protocol, fixing node ID truncation bug and the critical 38→36 byte documentation error.

### REFACTOR-SUMMARY.md
**Topic:** UI refactoring summary
**Date:** Prior to 2025-10-06
**Status:** Historical record

Summarizes UI component restructuring and client-side architecture changes. Relevant details should be integrated into client architecture documentation.

### SYSTEM_STATUS_REPORT.md
**Topic:** Overall system status report
**Date:** Prior to 2025-10-06
**Status:** Historical record

Comprehensive system status covering all components. Information has been integrated into permanent documentation locations.

## Integration Status

All information from these reports has been integrated into the permanent documentation corpus:

- **Agent Control**: → `docs/architecture/hybrid_docker_mcp_architecture.md`
- **Binary Protocol V2**: → `docs/reference/api/binary-protocol.md`
- **Dual-Graph Broadcasting**: → `docs/architecture/components/websocket-protocol.md`
- **UI Refactoring**: → (Pending) `docs/architecture/core/client.md`

## Usage

These files are kept for historical reference and debugging context. For current information, always refer to the permanent documentation locations listed above.

---

**Archive Date:** 2025-10-06
**Archived By:** Documentation consolidation effort
