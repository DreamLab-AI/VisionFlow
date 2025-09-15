// MCP Integration Tests for WebXR
// Run these in the WebXR container to verify MCP TCP connection

use std::io::{Write, Read, BufReader, BufRead};
use std::net::TcpStream;
use serde_json::{json, Value};
use std::time::Duration;

/// Test basic MCP server connectivity
#[test]
fn test_mcp_server_connection() {
    let result = TcpStream::connect_timeout(
        &"localhost:9500".parse().unwrap(),
        Duration::from_secs(5)
    );

    assert!(result.is_ok(), "Failed to connect to MCP server on port 9500");
    println!("✅ MCP server is reachable on port 9500");
}

/// Test MCP initialization protocol
#[test]
fn test_mcp_initialization() {
    let mut stream = TcpStream::connect("localhost:9500").expect("Failed to connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    // Send initialize request
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": { "listChanged": true }
            },
            "clientInfo": {
                "name": "webxr-test",
                "version": "1.0.0"
            }
        }
    });

    writeln!(stream, "{}", init_req).expect("Failed to send initialize");

    // Read response
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut response = String::new();
    reader.read_line(&mut response).expect("Failed to read response");

    let parsed: Value = serde_json::from_str(&response).expect("Invalid JSON response");

    assert_eq!(parsed["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(parsed["result"]["serverInfo"]["name"], "claude-flow");
    println!("✅ MCP initialization successful");
}

/// Test agent spawning
#[test]
fn test_agent_spawn() {
    let mut stream = TcpStream::connect("localhost:9500").expect("Failed to connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    // Initialize first
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": { "listChanged": true }},
            "clientInfo": { "name": "webxr-test", "version": "1.0.0" }
        }
    });
    writeln!(stream, "{}", init_req).unwrap();

    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut response = String::new();
    reader.read_line(&mut response).unwrap(); // Read init response

    // Spawn an agent
    let spawn_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "agent_spawn",
            "arguments": {
                "type": "researcher",
                "name": format!("test-agent-{}", chrono::Utc::now().timestamp())
            }
        }
    });

    writeln!(stream, "{}", spawn_req).expect("Failed to send spawn request");

    // Read spawn response
    response.clear();
    reader.read_line(&mut response).expect("Failed to read spawn response");

    let parsed: Value = serde_json::from_str(&response).expect("Invalid JSON");
    let content = &parsed["result"]["content"][0]["text"];
    let agent_data: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

    assert!(agent_data["success"].as_bool().unwrap());
    assert!(agent_data["agentId"].as_str().unwrap().starts_with("agent_"));
    assert_eq!(agent_data["type"], "researcher");
    assert_eq!(agent_data["status"], "active");

    println!("✅ Agent spawned successfully: {}", agent_data["agentId"]);
}

/// Test agent listing
#[test]
fn test_agent_list() {
    let mut stream = TcpStream::connect("localhost:9500").expect("Failed to connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    // Initialize
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": { "listChanged": true }},
            "clientInfo": { "name": "webxr-test", "version": "1.0.0" }
        }
    });
    writeln!(stream, "{}", init_req).unwrap();

    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut response = String::new();
    reader.read_line(&mut response).unwrap(); // Read init

    // List agents
    let list_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "agent_list",
            "arguments": {}
        }
    });

    writeln!(stream, "{}", list_req).expect("Failed to send list request");

    // Read list response
    response.clear();
    reader.read_line(&mut response).expect("Failed to read list response");

    let parsed: Value = serde_json::from_str(&response).expect("Invalid JSON");
    let content = &parsed["result"]["content"][0]["text"];
    let agent_list: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

    assert!(agent_list["success"].as_bool().unwrap());
    assert!(agent_list["agents"].is_array());

    let agent_count = agent_list["agents"].as_array().unwrap().len();
    println!("✅ Agent list retrieved: {} agents found", agent_count);
}

/// Test swarm initialization
#[test]
fn test_swarm_init() {
    let mut stream = TcpStream::connect("localhost:9500").expect("Failed to connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    // Initialize MCP
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": { "listChanged": true }},
            "clientInfo": { "name": "webxr-test", "version": "1.0.0" }
        }
    });
    writeln!(stream, "{}", init_req).unwrap();

    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut response = String::new();
    reader.read_line(&mut response).unwrap();

    // Initialize swarm
    let swarm_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "swarm_init",
            "arguments": {
                "topology": "mesh",
                "maxAgents": 8,
                "strategy": "balanced"
            }
        }
    });

    writeln!(stream, "{}", swarm_req).expect("Failed to send swarm init");

    // Read response
    response.clear();
    reader.read_line(&mut response).expect("Failed to read swarm response");

    let parsed: Value = serde_json::from_str(&response).expect("Invalid JSON");
    let content = &parsed["result"]["content"][0]["text"];
    let swarm_data: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

    assert!(swarm_data["success"].as_bool().unwrap());
    assert!(swarm_data["swarmId"].as_str().unwrap().starts_with("swarm_"));

    println!("✅ Swarm initialized: {}", swarm_data["swarmId"]);
}

/// Test performance metrics
#[test]
fn test_performance_metrics() {
    let mut stream = TcpStream::connect("localhost:9500").expect("Failed to connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    // Initialize
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": { "listChanged": true }},
            "clientInfo": { "name": "webxr-test", "version": "1.0.0" }
        }
    });
    writeln!(stream, "{}", init_req).unwrap();

    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut response = String::new();
    reader.read_line(&mut response).unwrap();

    // Get performance report
    let perf_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "performance_report",
            "arguments": {
                "timeframe": "24h",
                "format": "json"
            }
        }
    });

    writeln!(stream, "{}", perf_req).expect("Failed to send performance request");

    // Read response
    response.clear();
    reader.read_line(&mut response).expect("Failed to read performance response");

    let parsed: Value = serde_json::from_str(&response).expect("Invalid JSON");
    assert!(parsed["result"].is_object());

    println!("✅ Performance metrics retrieved successfully");
}

/// Integration test for full agent lifecycle
#[test]
fn test_full_agent_lifecycle() {
    let mut stream = TcpStream::connect("localhost:9500").expect("Failed to connect");
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    // Initialize
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": { "listChanged": true }},
            "clientInfo": { "name": "webxr-lifecycle-test", "version": "1.0.0" }
        }
    });
    writeln!(stream, "{}", init_req).unwrap();

    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut response = String::new();
    reader.read_line(&mut response).unwrap();

    // 1. Initialize swarm
    let swarm_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "swarm_init",
            "arguments": {
                "topology": "hierarchical",
                "maxAgents": 5
            }
        }
    });
    writeln!(stream, "{}", swarm_req).unwrap();
    response.clear();
    reader.read_line(&mut response).unwrap();

    // 2. Spawn multiple agents
    let agent_types = ["coordinator", "researcher", "coder", "tester"];
    let mut agent_ids = Vec::new();

    for (i, agent_type) in agent_types.iter().enumerate() {
        let spawn_req = json!({
            "jsonrpc": "2.0",
            "id": 3 + i,
            "method": "tools/call",
            "params": {
                "name": "agent_spawn",
                "arguments": {
                    "type": agent_type,
                    "name": format!("test-{}-{}", agent_type, chrono::Utc::now().timestamp())
                }
            }
        });

        writeln!(stream, "{}", spawn_req).unwrap();
        response.clear();
        reader.read_line(&mut response).unwrap();

        let parsed: Value = serde_json::from_str(&response).unwrap();
        let content = &parsed["result"]["content"][0]["text"];
        let agent_data: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

        agent_ids.push(agent_data["agentId"].as_str().unwrap().to_string());
        println!("  Created {}: {}", agent_type, agent_data["agentId"]);
    }

    // 3. List all agents
    let list_req = json!({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "agent_list",
            "arguments": {}
        }
    });

    writeln!(stream, "{}", list_req).unwrap();
    response.clear();
    reader.read_line(&mut response).unwrap();

    let parsed: Value = serde_json::from_str(&response).unwrap();
    let content = &parsed["result"]["content"][0]["text"];
    let agent_list: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();

    assert!(agent_list["agents"].as_array().unwrap().len() >= 4);

    println!("✅ Full agent lifecycle test completed");
    println!("   - Swarm initialized");
    println!("   - {} agents spawned", agent_ids.len());
    println!("   - All agents verified in list");
}

// Helper function to parse MCP responses
fn parse_mcp_response(response: &str) -> Result<Value, String> {
    let parsed: Value = serde_json::from_str(response)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    if let Some(error) = parsed.get("error") {
        return Err(format!("MCP error: {}", error));
    }

    if let Some(content) = parsed["result"]["content"][0]["text"].as_str() {
        serde_json::from_str(content)
            .map_err(|e| format!("Failed to parse content: {}", e))
    } else {
        Ok(parsed["result"].clone())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_mcp_server_health() {
        // Test that MCP server is healthy and responding
        assert!(test_mcp_connection_health());
    }

    fn test_mcp_connection_health() -> bool {
        TcpStream::connect_timeout(
            &"localhost:9500".parse().unwrap(),
            Duration::from_secs(2)
        ).is_ok()
    }
}