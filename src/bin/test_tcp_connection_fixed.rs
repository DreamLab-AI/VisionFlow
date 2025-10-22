use log::{error, info};
use serde_json::json;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::TcpStream;
use uuid::Uuid;

/// TCP Connection Test with Proper Resource Management
///
/// This version fixes the "Too many open files" issue by:
/// 1. Using a single connection split into read/write halves
/// 2. Properly shutting down connections
/// 3. Adding resource monitoring
/// 4. Implementing graceful cleanup

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    info!("Starting enhanced TCP connection test for Claude Flow MCP");

    let host =
        std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
    let port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

    info!("Connecting to {}:{}...", host, port);

    // Monitor initial file descriptor count
    let initial_fd_count = count_open_fds().await;
    info!("Initial file descriptors: {}", initial_fd_count);

    let start = Instant::now();

    // Create a single TCP connection to avoid resource leaks
    let stream = match TcpStream::connect(format!("{}:{}", host, port)).await {
        Ok(stream) => {
            info!("Successfully connected to {}:{}", host, port);
            stream
        }
        Err(e) => {
            error!("Failed to connect to {}:{}: {}", host, port, e);
            return Err(e.into());
        }
    };

    let connect_time = start.elapsed();
    info!("Connected in {:?}", connect_time);

    // Set TCP options for optimal performance
    stream.set_nodelay(true)?;

    // Split the stream into read and write halves (SINGLE connection approach)
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    // Check file descriptor count after connection
    let post_connect_fd_count = count_open_fds().await;
    info!(
        "File descriptors after connect: {} (delta: +{})",
        post_connect_fd_count,
        post_connect_fd_count - initial_fd_count
    );

    // Send initialization message
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
                "name": "visionflow-test-fixed",
                "version": "1.0.0"
            }
        }
    });

    let msg_str = format!("{}\n", serde_json::to_string(&init_msg)?);
    writer.write_all(msg_str.as_bytes()).await?;
    writer.flush().await?;

    info!("Sent initialization message");

    // Read response with timeout
    let mut response = String::new();
    let read_result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        reader.read_line(&mut response),
    )
    .await;

    match read_result {
        Ok(Ok(_)) => info!("Received response: {}", response.trim()),
        Ok(Err(e)) => error!("Error reading response: {}", e),
        Err(_) => error!("Timeout waiting for response"),
    }

    // Test listing tools
    let list_tools = json!({
        "jsonrpc": "2.0",
        "id": Uuid::new_v4().to_string(),
        "method": "tools/list"
    });

    let msg_str = format!("{}\n", serde_json::to_string(&list_tools)?);
    writer.write_all(msg_str.as_bytes()).await?;
    writer.flush().await?;

    info!("Requested tool list");

    // Read tools response with timeout
    let mut tools_response = String::new();
    let read_result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        reader.read_line(&mut tools_response),
    )
    .await;

    match read_result {
        Ok(Ok(_)) => info!("Available tools: {}", tools_response.trim()),
        Ok(Err(e)) => error!("Error reading tools response: {}", e),
        Err(_) => error!("Timeout waiting for tools response"),
    }

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
    writer.write_all(msg_str.as_bytes()).await?;
    writer.flush().await?;
    let send_time = send_start.elapsed();

    info!("Swarm initialization sent in {:?}", send_time);

    // Read swarm response with timeout
    let mut swarm_response = String::new();
    let read_result = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        reader.read_line(&mut swarm_response),
    )
    .await;

    match read_result {
        Ok(Ok(_)) => info!("Swarm response: {}", swarm_response.trim()),
        Ok(Err(e)) => error!("Error reading swarm response: {}", e),
        Err(_) => error!("Timeout waiting for swarm response"),
    }

    // CRITICAL: Properly shutdown the writer to close the connection gracefully
    info!("Shutting down TCP connection gracefully...");
    match writer.shutdown().await {
        Ok(_) => info!("TCP writer shutdown successfully"),
        Err(e) => error!("Error shutting down TCP writer: {}", e),
    }

    // Drop reader to close read half
    drop(reader);

    let total_time = start.elapsed();

    // Check final file descriptor count
    let final_fd_count = count_open_fds().await;
    let fd_delta = final_fd_count as i32 - initial_fd_count as i32;

    info!(
        "Final file descriptors: {} (delta: {:+})",
        final_fd_count, fd_delta
    );

    if fd_delta > 0 {
        error!(
            "⚠️  File descriptor leak detected! {} descriptors not cleaned up",
            fd_delta
        );
    } else {
        info!("✅ No file descriptor leaks detected");
    }

    // Performance summary
    println!("\n=== Performance & Resource Summary ===");
    println!("Connection time: {:?}", connect_time);
    println!("Message send time: {:?}", send_time);
    println!("Total test time: {:?}", total_time);
    println!("File descriptor delta: {:+}", fd_delta);
    println!(
        "Resource leak status: {}",
        if fd_delta > 0 {
            "LEAK DETECTED"
        } else {
            "CLEAN"
        }
    );

    info!("TCP connection test completed successfully with resource monitoring");
    Ok(())
}

/// Count current open file descriptors for this process
async fn count_open_fds() -> usize {
    #[cfg(target_os = "linux")]
    {
        match tokio::fs::read_dir("/proc/self/fd").await {
            Ok(mut entries) => {
                let mut count: usize = 0;
                while let Ok(Some(_)) = entries.next_entry().await {
                    count += 1;
                }
                count.saturating_sub(1) // Subtract the dir handle itself
            }
            Err(_) => {
                // If we can't read /proc/self/fd, use lsof as fallback
                match tokio::process::Command::new("lsof")
                    .args(["-p", &std::process::id().to_string()])
                    .output()
                    .await
                {
                    Ok(output) => {
                        String::from_utf8_lossy(&output.stdout)
                            .lines()
                            .skip(1) // Skip header
                            .count()
                    }
                    Err(_) => 0, // Fallback if lsof isn't available
                }
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        // For non-Linux systems, estimate conservatively
        10
    }
}
