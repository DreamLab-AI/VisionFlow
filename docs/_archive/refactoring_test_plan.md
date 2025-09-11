# Claude Flow Actor Refactoring Test Plan

## Testing Strategy

### Unit Testing

#### TcpConnectionActor Tests
```rust
#[cfg(test)]
mod tcp_connection_tests {
    use super::*;
    use actix::test;
    use tokio::net::TcpListener;

    #[actix::test]
    async fn test_connection_establishment() {
        // Start mock TCP server on localhost
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        
        let actor = TcpConnectionActor::new(
            addr.ip().to_string(), 
            addr.port()
        ).start();
        
        let result = actor.send(EstablishConnection).await;
        assert!(result.is_ok());
    }

    #[actix::test]  
    async fn test_connection_resilience() {
        // Test connection with no server running
        let actor = TcpConnectionActor::new(
            "127.0.0.1".to_string(), 
            19999 // Non-existent port
        ).start();
        
        // Should handle failure gracefully
        let result = actor.send(EstablishConnection).await;
        // Actor should not crash, will retry internally
    }
}
```

#### JsonRpcClient Tests  
```rust
#[cfg(test)]
mod jsonrpc_tests {
    use super::*;
    use actix::test;

    #[actix::test]
    async fn test_request_correlation() {
        let client = JsonRpcClient::new().start();
        
        // Test request/response correlation
        let params = json!({"test": "value"});
        let request = JsonRpcClient::create_request(
            "test-id".to_string(),
            "test_method".to_string(), 
            params
        );
        
        assert_eq!(request["id"], "test-id");
        assert_eq!(request["method"], "test_method");
        assert_eq!(request["jsonrpc"], "2.0");
    }

    #[actix::test]
    async fn test_mcp_initialization() {
        let client_info = ClientInfo {
            name: "test-client".to_string(),
            version: "1.0.0".to_string(),
        };
        
        let request = JsonRpcClient::create_initialize_request(
            "init-id".to_string(),
            &client_info,
            "1.0.0"
        );
        
        assert_eq!(request["method"], "initialize");
        assert!(request["params"]["capabilities"].is_object());
    }
}
```

### Integration Testing

#### Full Stack Integration
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use actix::test;

    #[actix::test]
    async fn test_full_refactored_stack() {
        // Start mock Claude Flow server
        let mock_server = start_mock_claude_flow_server().await;
        
        // Create refactored actor system
        let graph_addr = GraphServiceActor::new().start();
        let client = ClaudeFlowClient::new();
        
        let claude_flow_actor = ClaudeFlowActorTcp::new(
            client, 
            graph_addr.clone()
        ).start();
        
        // Test swarm initialization
        let result = claude_flow_actor.send(InitializeSwarm {
            topology: "mesh".to_string(),
            max_agents: 5,
            strategy: "balanced".to_string(),
        }).await;
        
        assert!(result.is_ok());
        
        // Test agent status polling  
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        let status = claude_flow_actor.send(GetSwarmStatus).await;
        assert!(status.is_ok());
    }
}
```

### Manual Testing Scenarios

#### 1. Connection Establishment
- **Test**: Start system and verify TCP connection to port 9500
- **Expected**: Logs show "TCP connection established to Claude Flow"
- **Verify**: No connection errors in logs

#### 2. MCP Protocol Initialization  
- **Test**: After connection, verify MCP initialization
- **Expected**: Logs show "MCP session initialized successfully"
- **Verify**: JSON-RPC initialize request/response logged

#### 3. Agent Polling Cycle
- **Test**: Let system run for 60 seconds
- **Expected**: Regular "Polling agent statuses" log entries
- **Verify**: Graph updates received every ~1 second

#### 4. Error Recovery
- **Test**: Kill Claude Flow service and restart it
- **Expected**: System detects connection loss and reconnects
- **Verify**: Exponential backoff in reconnection attempts

#### 5. Message Correlation
- **Test**: Send multiple concurrent tool calls
- **Expected**: Responses correctly matched to requests
- **Verify**: No timeout errors or mismatched responses

### Performance Testing

#### Memory Usage
```bash
# Monitor memory usage of refactored system
ps aux | grep visionflow
# Should show stable memory usage without leaks
```

#### Connection Pool Efficiency
```bash
# Monitor TCP connections
netstat -an | grep 9500
# Should show controlled number of connections
```

#### CPU Usage
```bash  
# Monitor CPU during heavy polling
top -p <visionflow-pid>
# Should show reasonable CPU usage
```

### Regression Testing

#### Critical Path Verification
- [ ] Agent status polling works at 1Hz frequency
- [ ] Graph visualization updates in real-time
- [ ] Swarm initialization returns valid swarm ID
- [ ] System metrics are calculated correctly
- [ ] Connection resilience works with network failures
- [ ] MCP tool calls succeed with proper correlation
- [ ] Resource monitoring prevents file descriptor leaks

#### API Compatibility
- [ ] All existing message handlers work unchanged
- [ ] External API surface identical to original
- [ ] Environment variables processed correctly
- [ ] Docker networking support maintained

### Load Testing

#### High Agent Count
- Test with 100+ agents in swarm
- Verify polling performance remains stable
- Check memory usage scales linearly

#### Extended Runtime
- Run system for 24+ hours
- Verify no memory leaks
- Check connection stability over time

#### Concurrent Operations
- Multiple simultaneous swarm initializations
- Concurrent tool calls from different contexts
- Verify no race conditions or deadlocks

## Success Criteria

### Functional Requirements
✅ All original functionality preserved  
✅ TCP connection to port 9500 maintained  
✅ Agent polling at 1Hz frequency  
✅ Graph updates work in real-time  
✅ MCP protocol compliance maintained  

### Non-Functional Requirements  
✅ Improved code maintainability  
✅ Clear separation of concerns  
✅ Enhanced testability  
✅ Better error handling and resilience  
✅ Preserved performance characteristics  

### Quality Attributes
✅ Single responsibility principle followed  
✅ Loose coupling between actors  
✅ High cohesion within actors  
✅ Comprehensive error handling  
✅ Resource leak prevention  

## Known Considerations

### Backward Compatibility
- Original implementation remains available as `ClaudeFlowActorTcp`
- Refactored version available as `ClaudeFlowActorRefactored`  
- Default export switched to refactored version
- Gradual migration path supported

### Dependencies
- Preserved all existing dependencies
- No new external crates introduced
- Uses existing network resilience utilities
- Compatible with current Actix version

### Configuration
- Same environment variables supported
- Docker networking logic preserved  
- Port 9500 default maintained
- No breaking configuration changes

## Deployment Strategy

1. **Deploy** refactored actors alongside original
2. **Switch** default export to refactored version
3. **Monitor** system stability for 48 hours  
4. **Validate** all critical functionality working
5. **Remove** original implementation after validation

The refactoring maintains full backward compatibility while providing a cleaner, more maintainable architecture that follows SOLID principles.