# Claude Flow Actor Refactoring Architecture

## Overview

The ClaudeFlowActorTcp has been refactored to follow the single responsibility principle by breaking down its responsibilities into separate, focused actors. This improves maintainability, testability, and system resilience.

## Architecture Components

### 1. TcpConnectionActor
**Responsibility**: Low-level TCP stream management

**Features**:
- TCP connection establishment with exponential backoff
- Raw TCP read/write operations  
- Connection lifecycle management
- Network resilience patterns (circuit breaker, connection pooling)
- Resource monitoring to prevent file descriptor exhaustion
- Connection event notifications to subscribers

**Key Methods**:
- `EstablishConnection` - Initiates TCP connection to Claude Flow server
- `SendRawMessage` - Sends raw bytes over TCP
- `SendJsonMessage` - Sends JSON data over TCP
- `SubscribeToEvents` - Subscribe to connection events

### 2. JsonRpcClient  
**Responsibility**: MCP protocol handling and message correlation

**Features**:
- JSON-RPC message formatting and parsing
- Request/response correlation using unique IDs  
- MCP protocol initialization
- Tool calling abstractions
- Timeout handling for requests
- Response validation

**Key Methods**:
- `InitializeMcpSession` - Performs MCP initialization handshake
- `CallTool` - Calls MCP tools with response correlation
- `SendRequest` - Sends general JSON-RPC requests
- `SendNotification` - Sends notifications (no response expected)

### 3. ClaudeFlowActorTcp (Refactored)
**Responsibility**: Application logic only

**Features**:
- Agent data management and caching
- Polling coordination and business logic
- System metrics and telemetry  
- Graph service integration
- Application-level error handling
- Coordination patterns management

**Delegated Concerns**:
- TCP connection management → TcpConnectionActor
- JSON-RPC protocol → JsonRpcClient

## Message Flow

```
┌─────────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│                     │     │                   │     │                      │
│ ClaudeFlowActorTcp  │────▶│   JsonRpcClient   │────▶│ TcpConnectionActor   │
│   (Application)     │     │   (Protocol)      │     │   (Transport)        │
│                     │     │                   │     │                      │
└─────────────────────┘     └───────────────────┘     └──────────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│                     │     │                   │     │                      │
│ • Agent Caching     │     │ • Request/Response│     │ • TCP Streams        │
│ • Polling Logic     │     │   Correlation     │     │ • Connection Pool    │
│ • System Metrics    │     │ • MCP Protocol    │     │ • Circuit Breaker    │
│ • Graph Updates     │     │ • Tool Calls      │     │ • Resilience         │
│ • Business Rules    │     │ • JSON-RPC Format │     │ • Resource Monitor   │
│                     │     │                   │     │                      │
└─────────────────────┘     └───────────────────┘     └──────────────────────┘
```

## Event Flow

1. **Connection Establishment**:
   - ClaudeFlowActorTcp starts TcpConnectionActor
   - TcpConnectionActor establishes TCP connection to port 9500
   - Connection events are published to subscribers
   - JsonRpcClient receives connection notification
   - MCP session initialization is triggered

2. **Agent Status Polling**:
   - ClaudeFlowActorTcp initiates polling cycle
   - Uses JsonRpcClient to call `agent_list` tool
   - JsonRpcClient formats JSON-RPC request with correlation ID
   - TcpConnectionActor sends raw TCP data
   - Response flows back through the chain
   - ClaudeFlowActorTcp processes agent data and updates graph

3. **Error Handling**:
   - TcpConnectionActor handles connection failures with exponential backoff
   - JsonRpcClient handles protocol errors and timeouts
   - ClaudeFlowActorTcp maintains application-level circuit breaker

## Benefits of Refactoring

### Single Responsibility Principle
- Each actor has one clear purpose
- Easier to test individual components
- Reduced coupling between concerns

### Improved Maintainability
- TCP logic isolated in TcpConnectionActor
- Protocol concerns contained in JsonRpcClient  
- Business logic focused in ClaudeFlowActorTcp

### Enhanced Resilience
- Network failures handled at transport layer
- Protocol errors handled at JSON-RPC layer
- Application logic remains clean and focused

### Better Testability
- Mock TCP connections for JsonRpcClient tests
- Mock JSON-RPC for application logic tests
- Integration tests can test component interactions

### Scalability
- TcpConnectionActor can manage multiple connections
- JsonRpcClient can handle concurrent requests
- Clear separation allows for independent optimization

## Backwards Compatibility

The refactored system maintains full backwards compatibility:
- All existing message handlers preserved
- Same external API surface
- MCP TCP connection on port 9500 preserved
- Polling functionality maintained
- Graph integration unchanged

## Configuration

The system uses the same environment variables:
- `CLAUDE_FLOW_HOST` / `MCP_HOST` - Server hostname
- `MCP_TCP_PORT` - TCP port (default: 9500)
- `DOCKER_ENV` - Docker networking flag

## Migration Path

For a gradual migration:

1. **Phase 1**: Deploy refactored actors alongside original (DONE)
2. **Phase 2**: Switch default export to refactored version (DONE)  
3. **Phase 3**: Monitor system stability
4. **Phase 4**: Remove original implementation after validation

The system currently exports both versions:
- `ClaudeFlowActorTcp` - Original implementation
- `ClaudeFlowActorRefactored` - New implementation  
- `ClaudeFlowActor` - Defaults to refactored version

## Critical Functionality Preservation

✅ **MCP TCP Connection**: Port 9500 connection maintained  
✅ **Agent Polling**: 1Hz polling cycle preserved  
✅ **Graph Updates**: Real-time agent visualization maintained  
✅ **Error Resilience**: Circuit breaker and retry logic preserved  
✅ **Resource Management**: Connection pooling and monitoring maintained  

The refactored system ensures that all critical MCP TCP functionality remains operational while providing a cleaner, more maintainable architecture.