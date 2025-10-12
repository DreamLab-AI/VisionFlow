# Pre-Refactoring Documentation Archive

**Archive Date**: 2025-10-12
**Archive Reason**: Phases 0-3 refactoring complete (Management API migration)

## Contents

This directory contains documentation that described the **previous architecture** before the Management API refactoring was completed. These documents are preserved for historical reference but are **no longer accurate** descriptions of the current system.

### What Changed

**Before (Hybrid Architecture)**:
- Docker exec-based task spawning
- MCP TCP (port 9500) for agent monitoring
- Complex actor hierarchy (TcpConnectionActor, JsonRpcClient)
- Session correlation bridge
- Docker hive mind orchestration

**After (Simplified Architecture)**:
- HTTP REST API (port 9090) for task management
- Management API polling for agent monitoring
- Simplified actor model (TaskOrchestratorActor, AgentMonitorActor)
- Direct HTTP communication
- No Docker exec required

### Archived Documents

#### Architecture
- `multi-agent-integration.md` - Described dual-channel (HTTP + MCP TCP) architecture
- `hybrid_docker_mcp_architecture.md` - Detailed hybrid Docker exec + MCP system
- `hybrid-implementation-plan.md` - Implementation plan for hybrid approach

#### Implementation
- `api_handlers_hybrid_migration.md` - Migration strategy for hybrid API handlers
- `speech_service_migration_strategy.md` - Speech service integration with hybrid system

#### Migration
- `testing-strategy.md` - Testing strategy for hybrid architecture

## Current Documentation

For up-to-date architecture information, see:
- `/docs/REFACTORING-PHASES-0-3-COMPLETE.md` - Comprehensive refactoring documentation
- `/task.md` - Current implementation status and remaining work

## Deleted Modules

The following Rust modules were deleted as part of this refactoring:
- `src/utils/docker_hive_mind.rs` - Docker exec-based orchestration
- `src/actors/tcp_connection_actor.rs` - TCP connection management
- `src/actors/jsonrpc_client.rs` - JSON-RPC client for MCP
- `src/services/mcp_session_bridge.rs` - Session management bridge
- `src/services/session_correlation_bridge.rs` - Session correlation

## Why Preserve?

These documents are kept for:
1. **Historical context** - Understanding design decisions that led to current architecture
2. **Rollback reference** - If critical issues discovered, reference old implementation
3. **Onboarding** - New team members can see architectural evolution
4. **Documentation patterns** - Reuse documentation structure and diagrams

## Do Not Use

❌ Do not use these documents as implementation guides
❌ Do not reference these architectures in new PRs or issues
❌ Do not update these files (they are frozen snapshots)

✅ Use `/docs/REFACTORING-PHASES-0-3-COMPLETE.md` instead
✅ Update `/task.md` with remaining work
✅ Create new architecture docs in `/docs/architecture/` if needed
