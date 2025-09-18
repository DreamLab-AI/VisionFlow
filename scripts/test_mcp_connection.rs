use std::env;

#[path = "src/utils/mcp_connection.rs"]
mod mcp_connection;

use mcp_connection::call_agent_list;

#[tokio::main]
async fn main() {
    env_logger::init();

    // Use the multi-agent-container IP
    let host = "172.18.0.4";
    let port = "9500";

    println!("Testing MCP connection to {}:{}", host, port);

    match call_agent_list(host, port, "all").await {
        Ok(result) => {
            println!("✅ Successfully connected to MCP!");
            println!("Agent list response: {}", serde_json::to_string_pretty(&result).unwrap());
        }
        Err(e) => {
            println!("❌ Failed to connect to MCP: {}", e);
        }
    }
}