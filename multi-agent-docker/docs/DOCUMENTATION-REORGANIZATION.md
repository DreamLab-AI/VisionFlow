# Documentation Reorganization - 2025-10-05

## Summary

Completely reorganized `/docs` directory to reflect the **current session-isolated architecture** with hybrid control/data plane integration.

## What Was Removed

All legacy documentation has been moved to `docs/archive/`:
- Old debugging notes (DIAGNOSIS.md, CHECK-RESULTS.md, etc.)
- Migration guides for deprecated systems
- Outdated MCP fix documentation
- Preliminary resilience strategies

**Reason**: These documents described problems that no longer exist or solutions that have been superseded by the session isolation system.

## New Documentation Structure

### docs/
```
00-quick-reference.md         # One-page quick start
README.md                      # Documentation index and overview
01-architecture.md             # System design and components
02-session-isolation.md        # UUID-based isolation system
03-mcp-infrastructure.md       # MCP servers and protocols
04-external-integration.md     # Rust/external system integration guide
05-session-api.md             # Complete session manager API reference
06-tcp-mcp-telemetry.md       # Real-time monitoring and visualization
07-logging.md                  # Persistent logging system
08-vnc-access.md              # Remote desktop access
09-security.md                 # Authentication and access control
10-troubleshooting.md          # Common issues and solutions
11-database-troubleshooting.md # SQLite-specific troubleshooting
archive/                       # Old documentation (for reference)
```

## Documentation Philosophy

Each document follows these principles:

1. **Current State Only**: Describes the system as it exists now, not historical iterations
2. **Practical Focus**: Code examples and command snippets, not theory
3. **Integration-Oriented**: Shows how external systems (particularly Rust) integrate
4. **Troubleshooting-Ready**: Every feature includes debugging information

## Key Topics Covered

### 1. Architecture (01-architecture.md)
- Hybrid control/data plane design
- Session isolation mechanism
- Database isolation strategy
- Process management
- Failure modes and recovery

### 2. Session Isolation (02-session-isolation.md)
- UUID-based session tracking
- Working directory isolation
- Database separation
- Session lifecycle management
- Recovery after container restarts

### 3. MCP Infrastructure (03-mcp-infrastructure.md)
- TCP MCP Server (port 9500)
- WebSocket Bridge (port 3002)
- GUI MCP servers (Playwright, QGIS, Blender, etc.)
- stdio-to-TCP bridge
- Available tools and protocols

### 4. External Integration (04-external-integration.md)
- Complete Rust integration examples
- Control plane (Docker exec) patterns
- Data plane (TCP/MCP) patterns
- File system access for results
- WebSocket streaming for visualization
- Error handling and retry logic

### 5. Session API (05-session-api.md)
- Full API reference for `hive-session-manager.sh`
- All commands with examples
- Return value specifications
- Workflow examples
- Rust integration patterns

### 6. TCP/MCP Telemetry (06-tcp-mcp-telemetry.md)
- Real-time agent monitoring
- Session-specific queries
- WebSocket subscription filters
- GPU spring system integration
- Bandwidth optimization
- Telemetry data structures

### 7. Logging (07-logging.md)
- Persistent log architecture
- Log rotation configuration
- Per-session logs
- Debugging workflows
- Log analysis examples

### 8. VNC Access (08-vnc-access.md)
- Remote desktop setup
- GUI tool access
- Common VNC issues
- Performance optimization

### 9. Security (09-security.md)
- Authentication mechanisms
- Token-based access control
- Container isolation
- Process ownership
- Network security

### 10. Troubleshooting (10-troubleshooting.md)
- Container crashes
- Session failures
- MCP connectivity
- High memory usage
- VNC issues
- Recovery procedures

### 11. Database Troubleshooting (11-database-troubleshooting.md)
- SQLite lock conflicts (prevention and detection)
- Database corruption recovery
- WAL file management
- Performance optimization
- Backup and restore
- Health monitoring

## Integration Focus

Documentation emphasizes **external system integration**, particularly:

### Docker Exec (Control Plane)
```rust
// Create session
let uuid = create_session("task").await?;

// Start task
start_session(uuid).await?;

// Monitor completion
let status = wait_for_completion(uuid).await?;
```

### TCP/MCP (Data Plane)
```rust
// Connect to telemetry
let mut mcp = McpClient::connect("localhost:9500").await?;

// Query session metrics
let metrics = mcp.query_session_metrics(uuid).await?;
```

### WebSocket (Visualization)
```rust
// Stream to GPU spring system
let ws = connect_websocket("ws://localhost:3002").await?;
subscribe_to_session(ws, uuid).await?;
```

## Usage Patterns Documented

### 1. Voice Command Integration
How to spawn tasks from voice commands and track them by UUID.

### 2. Multi-Session Monitoring
Tracking multiple concurrent tasks with telemetry aggregation.

### 3. Real-Time Visualization
Streaming agent positions and topology to GPU spring systems.

### 4. Error Recovery
Handling container restarts, network failures, and process crashes.

### 5. Performance Optimization
Bandwidth management, connection pooling, and caching strategies.

## What's NOT Documented

- Internal claude-flow implementation details (out of scope)
- Legacy migration paths (use git history if needed)
- Unimplemented features (session resume, load balancing, etc.)
- Development/build process (see README.md in project root)

## How to Use This Documentation

### For New Users
1. Start with `00-quick-reference.md` for basic commands
2. Read `README.md` for overview
3. Follow `04-external-integration.md` to integrate your system

### For External Integration
1. Read `01-architecture.md` to understand design
2. Study `04-external-integration.md` for Rust examples
3. Reference `05-session-api.md` for API details
4. Use `06-tcp-mcp-telemetry.md` for real-time monitoring

### For Troubleshooting
1. Check `10-troubleshooting.md` for common issues
2. Consult `11-database-troubleshooting.md` for database problems
3. Review logs with guidance from `07-logging.md`

### For Operations
1. Use `00-quick-reference.md` for daily commands
2. Monitor with `06-tcp-mcp-telemetry.md` patterns
3. Maintain with `11-database-troubleshooting.md` health checks

## Maintenance

This documentation should be updated when:

- **New features added**: Update relevant section + 00-quick-reference.md
- **API changes**: Update 05-session-api.md immediately
- **Architecture changes**: Update 01-architecture.md and dependent docs
- **Bug fixes**: Add to troubleshooting if pattern is repeatable

## Documentation Standards

All documentation follows:

- **Markdown format**: GitHub-flavored markdown
- **Code examples**: Executable, tested snippets
- **Command examples**: Copy-paste ready bash commands
- **Error examples**: Real error messages with solutions
- **Navigation**: Cross-references between related docs

## Archive Policy

Old documentation in `docs/archive/` is kept for:
- Historical reference
- Understanding evolution of the system
- Debugging legacy deployments

But should **not** be referenced in new integrations.

## Related Documentation

### Outside docs/
- `/SESSION-API.md` - Detailed session manager docs (duplicates 05-session-api.md)
- `/SESSION-ISOLATION-COMPLETE.md` - Implementation summary
- `/QUICK-START-SESSION-API.md` - Quick reference (duplicates 00-quick-reference.md)
- `/SECURITY.md` - Detailed security guide (duplicates 09-security.md)

**Note**: Root-level docs may be consolidated into `docs/` in future.

## Feedback

If documentation is unclear or incomplete:
1. Check `archive/` for additional context
2. Review code in `/app/scripts/` and `/core-assets/`
3. File issue with specific gaps noted

## Version

This documentation reflects:
- **System Version**: 2.0 (Session-Isolated Architecture)
- **Last Updated**: 2025-10-05
- **Major Changes**: Complete reorganization, removal of legacy material
