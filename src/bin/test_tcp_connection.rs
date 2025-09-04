use log::info;
use serde_json::json;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    info!("Starting TCP connection test for Claude Flow MCP");
    
    let host = std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
    let port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
    
    info!("Connecting to {}:{}...", host, port);
    
    let start = Instant::now();
    let mut stream = TcpStream::connect(format!("{}:{}", host, port)).await?;
    let connect_time = start.elapsed();
    
    info!("Connected in {:?}", connect_time);
    
    // Set TCP options
    stream.set_nodelay(true)?;
    
    // Create reader - for Tokio we need a second connection
    let reader_stream = TcpStream::connect(format!("{}:{}", host, port)).await?;
    reader_stream.set_nodelay(true)?;
    let mut reader = BufReader::new(reader_stream);
    
    // Send initialization
    let init_msg = json!({
        "jsonrpc": "2.0",
        "id": Uuid::new_v4().to_string(),
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0.0",
            "capabilities": {
                "roots": true,
                "sampling": true
            },
            "clientInfo": {
                "name": "visionflow-test",
                "version": "1.0.0"
            }
        }
    });
    
    let msg_str = format!("{}\n", serde_json::to_string(&init_msg)?);
    stream.write_all(msg_str.as_bytes()).await?;
    stream.flush().await?;
    
    info!("Sent initialization message");
    
    // Read response
    let mut response = String::new();
    reader.read_line(&mut response).await?;
    
    info!("Received response: {}", response);
    
    // Test listing tools
    let list_tools = json!({
        "jsonrpc": "2.0",
        "id": Uuid::new_v4().to_string(),
        "method": "tools/list"
    });
    
    let msg_str = format!("{}\n", serde_json::to_string(&list_tools)?);
    stream.write_all(msg_str.as_bytes()).await?;
    stream.flush().await?;
    
    info!("Requested tool list");
    
    // Read tools response
    let mut tools_response = String::new();
    reader.read_line(&mut tools_response).await?;
    
    info!("Available tools: {}", tools_response);
    
    // Test swarm initialization
    let swarm_init = json!({
        "jsonrpc": "2.0",
        "id": Uuid::new_v4().to_string(),
        "method": "tools/call",
        "params": {
            "name": "swarm_init",
            "arguments": {
                "objective": "Test swarm",
                "maxAgents": 3,
                "strategy": "balanced"
            }
        }
    });
    
    let msg_str = format!("{}\n", serde_json::to_string(&swarm_init)?);
    let send_start = Instant::now();
    stream.write_all(msg_str.as_bytes()).await?;
    stream.flush().await?;
    let send_time = send_start.elapsed();
    
    info!("Swarm initialization sent in {:?}", send_time);
    
    // Read swarm response
    let mut swarm_response = String::new();
    reader.read_line(&mut swarm_response).await?;
    
    info!("Swarm response: {}", swarm_response);
    
    let total_time = start.elapsed();
    info!("Test completed successfully in {:?}", total_time);
    
    // Performance summary
    println!("\n=== Performance Summary ===");
    println!("Connection time: {:?}", connect_time);
    println!("Message send time: {:?}", send_time);
    println!("Total test time: {:?}", total_time);
    
    Ok(())
}