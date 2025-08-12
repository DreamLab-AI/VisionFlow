use tokio;
use log::{info, error};
use env_logger;
use serde_json::json;
use std::time::Instant;

// Import from your crate
use visionflow::services::claude_flow::{
    client_builder::ClaudeFlowClientBuilder,
    error::Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    info!("Testing MCP TCP Connection...");
    
    // Load environment
    dotenv::dotenv().ok();
    
    let host = std::env::var("CLAUDE_FLOW_HOST")
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let port = std::env::var("MCP_TCP_PORT")
        .unwrap_or_else(|_| "9500".to_string())
        .parse()
        .unwrap_or(9500);
    
    info!("Connecting to TCP server at {}:{}...", host, port);
    
    let start = Instant::now();
    
    // Build client with TCP transport
    let mut client = match ClaudeFlowClientBuilder::new()
        .with_tcp()
        .host(&host)
        .port(port)
        .build()
        .await
    {
        Ok(client) => {
            info!("✅ Connected successfully in {:?}", start.elapsed());
            client
        }
        Err(e) => {
            error!("❌ Connection failed: {}", e);
            return Err(e);
        }
    };
    
    // Test 1: List tools
    info!("\nTest 1: Listing available tools...");
    let tools_start = Instant::now();
    match client.list_tools().await {
        Ok(tools) => {
            info!("✅ Found {} tools in {:?}:", tools.len(), tools_start.elapsed());
            for (i, tool) in tools.iter().enumerate().take(5) {
                info!("   {}. {}", i + 1, tool.name);
            }
            if tools.len() > 5 {
                info!("   ... and {} more", tools.len() - 5);
            }
        }
        Err(e) => {
            error!("❌ Failed to list tools: {}", e);
        }
    }
    
    // Test 2: Spawn an agent
    info!("\nTest 2: Spawning a test agent...");
    let spawn_start = Instant::now();
    match client.call_tool("agent_spawn", json!({
        "type": "researcher",
        "name": "tcp-test-agent",
        "capabilities": ["test", "tcp", "connection"]
    })).await {
        Ok(result) => {
            info!("✅ Agent spawned in {:?}: {:?}", spawn_start.elapsed(), result);
        }
        Err(e) => {
            error!("❌ Failed to spawn agent: {}", e);
        }
    }
    
    // Test 3: Check swarm status
    info!("\nTest 3: Checking swarm status...");
    let status_start = Instant::now();
    match client.call_tool("swarm_status", json!({})).await {
        Ok(status) => {
            info!("✅ Swarm status retrieved in {:?}", status_start.elapsed());
            if let Some(topology) = status.get("topology") {
                info!("   Topology: {}", topology);
            }
            if let Some(agents) = status.get("totalAgents") {
                info!("   Total agents: {}", agents);
            }
        }
        Err(e) => {
            error!("❌ Failed to get swarm status: {}", e);
        }
    }
    
    // Test 4: Performance test - rapid requests
    info!("\nTest 4: Performance test (10 rapid requests)...");
    let perf_start = Instant::now();
    let mut successes = 0;
    let mut total_time = Duration::from_secs(0);
    
    for i in 0..10 {
        let req_start = Instant::now();
        match client.call_tool("memory_usage", json!({
            "action": "list",
            "namespace": "test"
        })).await {
            Ok(_) => {
                successes += 1;
                let elapsed = req_start.elapsed();
                total_time += elapsed;
                info!("   Request {}: {:?}", i + 1, elapsed);
            }
            Err(e) => {
                error!("   Request {} failed: {}", i + 1, e);
            }
        }
    }
    
    info!("✅ Performance test complete in {:?}", perf_start.elapsed());
    info!("   Success rate: {}/10", successes);
    if successes > 0 {
        info!("   Average latency: {:?}", total_time / successes);
    }
    
    // Disconnect
    info!("\nDisconnecting...");
    match client.disconnect().await {
        Ok(_) => info!("✅ Disconnected cleanly"),
        Err(e) => error!("❌ Disconnect error: {}", e),
    }
    
    info!("\n=== TCP Connection Test Complete ===");
    info!("Total test duration: {:?}", start.elapsed());
    
    Ok(())
}

use std::time::Duration;