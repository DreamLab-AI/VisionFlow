//! Multi-MCP Agent Visualization Integration Demo
//! 
//! This example demonstrates how to use the comprehensive agent visualization system
//! to discover, monitor, and visualize agents across multiple MCP servers.
//!
//! Usage:
//! ```bash
//! cargo run --example multi_mcp_integration_demo
//! ```

use actix::prelude::*;
use actix_web::{web, App, HttpServer, middleware::Logger};
use log::info;
use std::time::Duration;
use tokio::time::sleep;

// Import our visualization system components
use visionflow_ext::services::{
    multi_mcp_agent_discovery::{MultiMcpAgentDiscovery, McpServerConfig},
    agent_visualization_protocol::{McpServerType, AgentVisualizationProtocol},
    topology_visualization_engine::{TopologyVisualizationEngine, TopologyConfig, TopologyType},
};
use visionflow_ext::actors::{
    MultiMcpVisualizationActor, GraphServiceActor
};
use visionflow_ext::handlers::multi_mcp_websocket_handler;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging
    env_logger::init();
    
    info!("🚀 Starting Multi-MCP Agent Visualization Demo");
    
    // Start actor system
    let graph_service = GraphServiceActor::new().start();
    let visualization_actor = MultiMcpVisualizationActor::new(graph_service.clone()).start();
    
    // Demonstrate the discovery service
    demo_discovery_service().await;
    
    // Demonstrate topology visualization
    demo_topology_visualization().await;
    
    // Demonstrate real-time monitoring
    demo_realtime_monitoring().await;
    
    // Start HTTP server with WebSocket endpoints
    info!("🌐 Starting HTTP server with WebSocket endpoints on http://127.0.0.1:8080");
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(DemoAppState {
                visualization_actor: visualization_actor.clone(),
                graph_service: graph_service.clone(),
            }))
            .wrap(Logger::default())
            .configure(multi_mcp_websocket_handler::configure_multi_mcp_routes)
            .service(web::resource("/").to(serve_demo_page))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

#[derive(Clone)]
struct DemoAppState {
    visualization_actor: Addr<MultiMcpVisualizationActor>,
    graph_service: Addr<GraphServiceActor>,
}

/// Demonstrate the multi-MCP discovery service
async fn demo_discovery_service() {
    info!("📡 Demonstrating Multi-MCP Discovery Service");
    
    let discovery = MultiMcpAgentDiscovery::new();
    
    // Initialize with default servers
    discovery.initialize_default_servers().await;
    
    // Add a custom MCP server
    let custom_config = McpServerConfig {
        server_id: "demo-server".to_string(),
        server_type: McpServerType::Custom("demo".to_string()),
        host: "127.0.0.1".to_string(),
        port: 9503,
        enabled: false, // Disabled for demo since server doesn't exist
        discovery_interval_ms: 2000,
        timeout_ms: 5000,
        retry_attempts: 2,
    };
    
    discovery.add_server(custom_config).await;
    
    // Start discovery (won't actually find agents without real MCP servers)
    discovery.start_discovery().await;
    
    // Wait a bit to let discovery attempt to run
    sleep(Duration::from_secs(2)).await;
    
    // Get discovery statistics
    let stats = discovery.get_discovery_stats().await;
    info!("📊 Discovery Stats: successful={}, failed={}, total={}", 
          stats.successful_discoveries, stats.failed_discoveries, stats.total_discoveries);
    
    // Get server statuses
    let server_statuses = discovery.get_server_statuses().await;
    info!("🖥️ Found {} MCP servers", server_statuses.len());
    for server in server_statuses {
        info!("  - {}: {}:{} (connected: {})", 
              server.server_id, server.host, server.port, server.is_connected);
    }
    
    // Stop discovery for demo
    discovery.stop_discovery().await;
}

/// Demonstrate topology visualization engine
async fn demo_topology_visualization() {
    info!("🕸️ Demonstrating Topology Visualization Engine");
    
    // Create mock agents for visualization
    let mock_agents = create_mock_agents();
    
    let config = TopologyConfig {
        topology_type: TopologyType::Hierarchical,
        layout_params: Default::default(),
        visual_params: Default::default(),
    };
    
    let mut engine = TopologyVisualizationEngine::new(config);
    
    // Generate layouts for different topologies
    let topologies = vec![
        TopologyType::Hierarchical,
        TopologyType::Mesh,
        TopologyType::Ring,
        TopologyType::Star,
    ];
    
    for topology in topologies {
        info!("  📐 Generating {:?} topology layout", topology);
        let layout = engine.generate_layout(
            "demo-swarm".to_string(),
            mock_agents.clone(),
            topology,
        );
        
        info!("    - {} agents positioned", layout.agent_positions.len());
        info!("    - {} connections created", layout.connections.len());
        info!("    - {} coordination layers", layout.coordination_layers.len());
        info!("    - Performance metrics: efficiency={:.2}, fault_tolerance={:.2}", 
              layout.performance_metrics.coordination_efficiency,
              layout.performance_metrics.fault_tolerance);
    }
}

/// Demonstrate real-time monitoring
async fn demo_realtime_monitoring() {
    info!("📈 Demonstrating Real-time Monitoring");
    
    let mut protocol = AgentVisualizationProtocol::new();
    
    // Register mock MCP servers
    let claude_flow_server = visionflow_ext::services::agent_visualization_protocol::McpServerInfo {
        server_id: "claude-flow".to_string(),
        server_type: McpServerType::ClaudeFlow,
        host: "localhost".to_string(),
        port: 9500,
        is_connected: true,
        last_heartbeat: chrono::Utc::now().timestamp(),
        supported_tools: vec!["swarm_init".to_string(), "agent_list".to_string()],
        agent_count: 4,
    };
    
    protocol.register_mcp_server(claude_flow_server);
    
    // Simulate agent updates
    let mock_agents = create_mock_agents();
    protocol.update_agents_from_server(McpServerType::ClaudeFlow, mock_agents.clone());
    
    // Generate discovery message
    let discovery_message = protocol.create_discovery_message();
    info!("  📤 Generated discovery message ({} bytes)", discovery_message.len());
    
    // Generate agent update message
    let update_message = protocol.create_agent_update_message(mock_agents);
    info!("  📤 Generated agent update message ({} bytes)", update_message.len());
    
    // Generate performance analysis
    let performance_message = protocol.create_performance_analysis();
    info!("  📤 Generated performance analysis ({} bytes)", performance_message.len());
}

/// Create mock agents for demonstration
fn create_mock_agents() -> Vec<visionflow_ext::services::agent_visualization_protocol::MultiMcpAgentStatus> {
    use visionflow_ext::services::agent_visualization_protocol::*;
    
    vec![
        MultiMcpAgentStatus {
            agent_id: "coordinator-001".to_string(),
            swarm_id: "demo-swarm".to_string(),
            server_source: McpServerType::ClaudeFlow,
            name: "Main Coordinator".to_string(),
            agent_type: "coordinator".to_string(),
            status: "active".to_string(),
            capabilities: vec!["coordination".to_string(), "task_management".to_string()],
            metadata: AgentExtendedMetadata {
                session_id: Some("session-001".to_string()),
                parent_id: None,
                topology_position: Some(TopologyPosition {
                    layer: 0,
                    index_in_layer: 0,
                    connections: vec!["worker-001".to_string(), "worker-002".to_string()],
                    is_coordinator: true,
                    coordination_level: 1,
                }),
                coordination_role: Some("primary".to_string()),
                task_queue_size: 5,
                error_count: 0,
                warning_count: 1,
                tags: vec!["coordinator".to_string(), "primary".to_string()],
            },
            performance: AgentPerformanceData {
                cpu_usage: 0.45,
                memory_usage: 0.62,
                health_score: 0.95,
                activity_level: 0.8,
                tasks_active: 5,
                tasks_completed: 123,
                tasks_failed: 2,
                success_rate: 0.98,
                token_usage: 45000,
                token_rate: 150.0,
                response_time_ms: 45.0,
                throughput: 25.5,
            },
            neural_info: Some(NeuralAgentData {
                model_type: "claude-3-sonnet".to_string(),
                model_size: "medium".to_string(),
                training_status: "active".to_string(),
                cognitive_pattern: "coordination".to_string(),
                learning_rate: 0.01,
                adaptation_score: 0.85,
                memory_capacity: 1000000,
                knowledge_domains: vec!["project_management".to_string(), "system_architecture".to_string()],
            }),
            created_at: chrono::Utc::now().timestamp() - 3600,
            last_active: chrono::Utc::now().timestamp(),
        },
        MultiMcpAgentStatus {
            agent_id: "worker-001".to_string(),
            swarm_id: "demo-swarm".to_string(),
            server_source: McpServerType::ClaudeFlow,
            name: "Code Worker Alpha".to_string(),
            agent_type: "coder".to_string(),
            status: "busy".to_string(),
            capabilities: vec!["rust_development".to_string(), "testing".to_string()],
            metadata: AgentExtendedMetadata {
                session_id: Some("session-002".to_string()),
                parent_id: Some("coordinator-001".to_string()),
                topology_position: Some(TopologyPosition {
                    layer: 1,
                    index_in_layer: 0,
                    connections: vec!["coordinator-001".to_string()],
                    is_coordinator: false,
                    coordination_level: 0,
                }),
                coordination_role: None,
                task_queue_size: 3,
                error_count: 1,
                warning_count: 0,
                tags: vec!["worker".to_string(), "coder".to_string()],
            },
            performance: AgentPerformanceData {
                cpu_usage: 0.82,
                memory_usage: 0.71,
                health_score: 0.88,
                activity_level: 0.95,
                tasks_active: 3,
                tasks_completed: 87,
                tasks_failed: 5,
                success_rate: 0.94,
                token_usage: 32000,
                token_rate: 200.0,
                response_time_ms: 120.0,
                throughput: 18.2,
            },
            neural_info: None,
            created_at: chrono::Utc::now().timestamp() - 1800,
            last_active: chrono::Utc::now().timestamp() - 30,
        },
        MultiMcpAgentStatus {
            agent_id: "monitor-001".to_string(),
            swarm_id: "demo-swarm".to_string(),
            server_source: McpServerType::RuvSwarm,
            name: "System Monitor".to_string(),
            agent_type: "monitor".to_string(),
            status: "idle".to_string(),
            capabilities: vec!["performance_monitoring".to_string(), "health_checks".to_string()],
            metadata: AgentExtendedMetadata {
                session_id: Some("session-003".to_string()),
                parent_id: None,
                topology_position: Some(TopologyPosition {
                    layer: 2,
                    index_in_layer: 0,
                    connections: vec!["coordinator-001".to_string(), "worker-001".to_string()],
                    is_coordinator: false,
                    coordination_level: 0,
                }),
                coordination_role: None,
                task_queue_size: 0,
                error_count: 0,
                warning_count: 0,
                tags: vec!["monitor".to_string(), "system".to_string()],
            },
            performance: AgentPerformanceData {
                cpu_usage: 0.15,
                memory_usage: 0.25,
                health_score: 1.0,
                activity_level: 0.3,
                tasks_active: 0,
                tasks_completed: 245,
                tasks_failed: 0,
                success_rate: 1.0,
                token_usage: 8000,
                token_rate: 20.0,
                response_time_ms: 25.0,
                throughput: 5.8,
            },
            neural_info: None,
            created_at: chrono::Utc::now().timestamp() - 7200,
            last_active: chrono::Utc::now().timestamp() - 120,
        },
    ]
}

/// Serve a simple demo page
async fn serve_demo_page() -> impl actix_web::Responder {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Multi-MCP Agent Visualization Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
        .container { max-width: 800px; margin: 0 auto; }
        .server-status { background: #2a2a2a; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .online { color: #4CAF50; }
        .offline { color: #f44336; }
        .button { background: #007acc; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .button:hover { background: #005fa3; }
        #log { background: #000; color: #0f0; padding: 20px; border-radius: 8px; height: 300px; overflow-y: auto; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Multi-MCP Agent Visualization Demo</h1>
        
        <p>This demo showcases the comprehensive agent visualization system that discovers and monitors agents across multiple MCP servers.</p>
        
        <div class="server-status">
            <h3>📡 MCP Server Status</h3>
            <div id="server-list">
                <div>🟢 <span class="online">Claude Flow</span> - localhost:9500 (4 agents)</div>
                <div>🔴 <span class="offline">RuvSwarm</span> - localhost:9501 (0 agents)</div>
                <div>🔴 <span class="offline">DAA</span> - localhost:9502 (0 agents)</div>
            </div>
        </div>
        
        <div>
            <h3>🎮 Demo Controls</h3>
            <button class="button" onclick="connectWebSocket()">Connect WebSocket</button>
            <button class="button" onclick="requestDiscovery()">Request Discovery</button>
            <button class="button" onclick="requestAgents()">Request Agents</button>
            <button class="button" onclick="requestPerformance()">Request Performance</button>
            <button class="button" onclick="clearLog()">Clear Log</button>
        </div>
        
        <div id="log"></div>
    </div>

    <script>
        let ws = null;
        
        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}\n`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function connectWebSocket() {
            if (ws) {
                ws.close();
            }
            
            ws = new WebSocket('ws://127.0.0.1:8080/api/multi-mcp/ws');
            
            ws.onopen = function() {
                log('🟢 WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    log(`📥 Received: ${data.type || 'unknown'} (${event.data.length} bytes)`);
                    
                    if (data.type === 'discovery') {
                        log(`   📡 Found ${data.total_agents} agents across ${data.servers.length} servers`);
                    } else if (data.type === 'multi_agent_update') {
                        log(`   👥 Agent update: ${data.agents.length} agents`);
                    } else if (data.type === 'performance_analysis') {
                        log(`   📊 Performance: efficiency=${data.global_metrics.system_efficiency}`);
                    }
                } catch (e) {
                    log(`📥 Raw message: ${event.data.substring(0, 100)}...`);
                }
            };
            
            ws.onclose = function() {
                log('🔴 WebSocket disconnected');
            };
            
            ws.onerror = function(error) {
                log(`❌ WebSocket error: ${error}`);
            };
        }
        
        function requestDiscovery() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: 'request_discovery'}));
                log('📤 Requested discovery data');
            } else {
                log('❌ WebSocket not connected');
            }
        }
        
        function requestAgents() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: 'request_agents'}));
                log('📤 Requested agent data');
            } else {
                log('❌ WebSocket not connected');
            }
        }
        
        function requestPerformance() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: 'request_performance'}));
                log('📤 Requested performance data');
            } else {
                log('❌ WebSocket not connected');
            }
        }
        
        function clearLog() {
            document.getElementById('log').innerHTML = '';
        }
        
        // Auto-connect on page load
        log('🚀 Multi-MCP Agent Visualization Demo loaded');
        log('💡 Click "Connect WebSocket" to start monitoring');
    </script>
</body>
</html>
    "#;
    
    actix_web::HttpResponse::Ok()
        .content_type("text/html")
        .body(html)
}