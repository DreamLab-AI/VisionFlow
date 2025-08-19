use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Initial JSON payload sent to clients for agent visualization setup
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentVisualizationMessage {
    /// Initial complete state - sent once on connection
    #[serde(rename = "init")]
    Initialize(InitializeMessage),
    
    /// Incremental position updates - sent frequently
    #[serde(rename = "positions")]
    PositionUpdate(PositionUpdateMessage),
    
    /// Agent state changes - sent on status/health changes
    #[serde(rename = "state")]
    StateUpdate(StateUpdateMessage),
    
    /// Connection changes - sent when edges are added/removed
    #[serde(rename = "connections")]
    ConnectionUpdate(ConnectionUpdateMessage),
    
    /// Performance metrics - sent periodically
    #[serde(rename = "metrics")]
    MetricsUpdate(MetricsUpdateMessage),
}

/// Complete initialization data for client setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeMessage {
    pub timestamp: i64, // Unix timestamp
    pub swarm_id: String,
    pub topology: String,
    
    /// All agents with full metadata
    pub agents: Vec<AgentInit>,
    
    /// All connections between agents
    pub connections: Vec<ConnectionInit>,
    
    /// Visual configuration for rendering
    pub visual_config: VisualConfig,
    
    /// Physics configuration for GPU solver
    pub physics_config: PhysicsConfig,
    
    /// Initial positions (optional - can be calculated client-side)
    pub positions: HashMap<String, Position>,
}

/// Agent initialization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInit {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub agent_type: String,
    pub status: String,
    
    /// Visual properties
    pub color: String,
    pub shape: String,  // "sphere", "cube", "cone", etc.
    pub size: f32,
    
    /// Performance metrics (0-1 normalized)
    pub health: f32,
    pub cpu: f32,
    pub memory: f32,
    pub activity: f32,
    
    /// Task information
    pub tasks_active: u32,
    pub tasks_completed: u32,
    pub success_rate: f32,
    
    /// Token usage
    pub tokens: u64,
    pub token_rate: f32,
    
    /// Additional metadata
    pub capabilities: Vec<String>,
    pub created_at: i64,
}

/// Connection initialization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInit {
    pub id: String,
    pub source: String,
    pub target: String,
    pub strength: f32,      // 0-1
    pub flow_rate: f32,     // 0-1
    pub color: String,
    pub active: bool,
}

/// Position update - sent frequently via WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdateMessage {
    pub timestamp: i64,
    pub positions: Vec<PositionUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Optional velocity for client-side interpolation
    pub vx: Option<f32>,
    pub vy: Option<f32>,
    pub vz: Option<f32>,
}

/// State changes for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUpdateMessage {
    pub timestamp: i64,
    pub updates: Vec<AgentStateUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateUpdate {
    pub id: String,
    pub status: Option<String>,
    pub health: Option<f32>,
    pub cpu: Option<f32>,
    pub memory: Option<f32>,
    pub activity: Option<f32>,
    pub tasks_active: Option<u32>,
    pub current_task: Option<String>,
}

/// Connection updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionUpdateMessage {
    pub timestamp: i64,
    pub added: Vec<ConnectionInit>,
    pub removed: Vec<String>, // connection IDs
    pub updated: Vec<ConnectionStateUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStateUpdate {
    pub id: String,
    pub active: Option<bool>,
    pub flow_rate: Option<f32>,
    pub strength: Option<f32>,
}

/// Performance metrics update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsUpdateMessage {
    pub timestamp: i64,
    pub overall: SwarmMetrics,
    pub agent_metrics: Vec<AgentMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
    pub total_agents: u32,
    pub active_agents: u32,
    pub health_avg: f32,
    pub cpu_total: f32,
    pub memory_total: f32,
    pub tokens_total: u64,
    pub tokens_per_second: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub id: String,
    pub tokens: u64,
    pub token_rate: f32,
    pub tasks_completed: u32,
    pub success_rate: f32,
}

/// Position type used in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Visual configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualConfig {
    pub colors: HashMap<String, String>,
    pub sizes: HashMap<String, f32>,
    pub animations: HashMap<String, AnimationConfig>,
    pub effects: EffectsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    pub speed: f32,
    pub amplitude: f32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsConfig {
    pub glow: bool,
    pub particles: bool,
    pub bloom: bool,
    pub shadows: bool,
}

/// Physics configuration for GPU solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub spring_k: f32,
    pub link_distance: f32,
    pub damping: f32,
    pub repel_k: f32,
    pub gravity_k: f32,
    pub max_velocity: f32,
}

/// Multi-MCP Agent Discovery and Monitoring System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub server_id: String,
    pub server_type: McpServerType,
    pub host: String,
    pub port: u16,
    pub is_connected: bool,
    pub last_heartbeat: i64,
    pub supported_tools: Vec<String>,
    pub agent_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum McpServerType {
    ClaudeFlow,
    RuvSwarm,
    Daa,
    Custom(String),
}

/// Enhanced agent data with MCP server context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiMcpAgentStatus {
    pub agent_id: String,
    pub swarm_id: String,
    pub server_source: McpServerType,
    pub name: String,
    pub agent_type: String,
    pub status: String,
    pub capabilities: Vec<String>,
    pub metadata: AgentExtendedMetadata,
    pub performance: AgentPerformanceData,
    pub neural_info: Option<NeuralAgentData>,
    pub created_at: i64,
    pub last_active: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExtendedMetadata {
    pub session_id: Option<String>,
    pub parent_id: Option<String>,
    pub topology_position: Option<TopologyPosition>,
    pub coordination_role: Option<String>,
    pub task_queue_size: u32,
    pub error_count: u32,
    pub warning_count: u32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyPosition {
    pub layer: u32,
    pub index_in_layer: u32,
    pub connections: Vec<String>, // Connected agent IDs
    pub is_coordinator: bool,
    pub coordination_level: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceData {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub health_score: f32,
    pub activity_level: f32,
    pub tasks_active: u32,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub success_rate: f32,
    pub token_usage: u64,
    pub token_rate: f32,
    pub response_time_ms: f32,
    pub throughput: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAgentData {
    pub model_type: String,
    pub model_size: String,
    pub training_status: String,
    pub cognitive_pattern: String,
    pub learning_rate: f32,
    pub adaptation_score: f32,
    pub memory_capacity: u64,
    pub knowledge_domains: Vec<String>,
}

/// Swarm topology visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmTopologyData {
    pub topology_type: String,
    pub total_agents: u32,
    pub coordination_layers: u32,
    pub efficiency_score: f32,
    pub load_distribution: Vec<LayerLoad>,
    pub critical_paths: Vec<CriticalPath>,
    pub bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLoad {
    pub layer_id: u32,
    pub agent_count: u32,
    pub average_load: f32,
    pub max_capacity: u32,
    pub utilization: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub path_id: String,
    pub agent_sequence: Vec<String>,
    pub total_latency_ms: f32,
    pub bottleneck_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub agent_id: String,
    pub bottleneck_type: String,
    pub severity: f32,
    pub impact_agents: Vec<String>,
    pub suggested_action: String,
}

/// Enhanced message types for multi-MCP coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MultiMcpVisualizationMessage {
    /// Full discovery of all MCP servers and agents
    #[serde(rename = "discovery")]
    Discovery(DiscoveryMessage),
    
    /// Agent status updates from multiple servers
    #[serde(rename = "multi_agent_update")]
    MultiAgentUpdate(MultiAgentUpdateMessage),
    
    /// Topology changes and coordination updates
    #[serde(rename = "topology_update")]
    TopologyUpdate(TopologyUpdateMessage),
    
    /// Neural agent learning and adaptation updates
    #[serde(rename = "neural_update")]
    NeuralUpdate(NeuralUpdateMessage),
    
    /// Performance and bottleneck analysis
    #[serde(rename = "performance_analysis")]
    PerformanceAnalysis(PerformanceAnalysisMessage),
    
    /// Real-time coordination events
    #[serde(rename = "coordination_event")]
    CoordinationEvent(CoordinationEventMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryMessage {
    pub timestamp: i64,
    pub servers: Vec<McpServerInfo>,
    pub total_agents: u32,
    pub swarms: Vec<SwarmInfo>,
    pub global_topology: GlobalTopology,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmInfo {
    pub swarm_id: String,
    pub server_source: McpServerType,
    pub topology: String,
    pub agent_count: u32,
    pub health_score: f32,
    pub coordination_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalTopology {
    pub inter_swarm_connections: Vec<InterSwarmConnection>,
    pub coordination_hierarchy: Vec<CoordinationLevel>,
    pub data_flow_patterns: Vec<DataFlowPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterSwarmConnection {
    pub source_swarm: String,
    pub target_swarm: String,
    pub connection_strength: f32,
    pub message_rate: f32,
    pub coordination_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationLevel {
    pub level: u32,
    pub coordinator_agents: Vec<String>,
    pub managed_agents: Vec<String>,
    pub coordination_load: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowPattern {
    pub pattern_id: String,
    pub source_agents: Vec<String>,
    pub target_agents: Vec<String>,
    pub flow_rate: f32,
    pub data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentUpdateMessage {
    pub timestamp: i64,
    pub agents: Vec<MultiMcpAgentStatus>,
    pub differential_updates: Vec<AgentDifferentialUpdate>,
    pub removed_agents: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDifferentialUpdate {
    pub agent_id: String,
    pub field_updates: std::collections::HashMap<String, serde_json::Value>,
    pub performance_delta: Option<PerformanceDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDelta {
    pub cpu_change: f32,
    pub memory_change: f32,
    pub task_completion_rate: f32,
    pub error_rate_change: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyUpdateMessage {
    pub timestamp: i64,
    pub swarm_id: String,
    pub topology_changes: Vec<TopologyChange>,
    pub new_connections: Vec<AgentConnection>,
    pub removed_connections: Vec<String>,
    pub coordination_updates: Vec<CoordinationUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyChange {
    pub change_type: String,
    pub affected_agents: Vec<String>,
    pub new_structure: Option<serde_json::Value>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConnection {
    pub connection_id: String,
    pub source_agent: String,
    pub target_agent: String,
    pub connection_type: String,
    pub strength: f32,
    pub bidirectional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationUpdate {
    pub coordinator_id: String,
    pub managed_agents: Vec<String>,
    pub coordination_load: f32,
    pub efficiency_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralUpdateMessage {
    pub timestamp: i64,
    pub neural_agents: Vec<NeuralAgentUpdate>,
    pub learning_events: Vec<LearningEvent>,
    pub adaptation_metrics: Vec<AdaptationMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAgentUpdate {
    pub agent_id: String,
    pub neural_data: NeuralAgentData,
    pub learning_progress: f32,
    pub recent_adaptations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    pub event_id: String,
    pub agent_id: String,
    pub event_type: String,
    pub learning_data: serde_json::Value,
    pub performance_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetric {
    pub metric_name: String,
    pub current_value: f32,
    pub target_value: f32,
    pub progress: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysisMessage {
    pub timestamp: i64,
    pub global_metrics: GlobalPerformanceMetrics,
    pub bottlenecks: Vec<Bottleneck>,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    pub trend_analysis: Vec<TrendAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalPerformanceMetrics {
    pub total_throughput: f32,
    pub average_latency: f32,
    pub system_efficiency: f32,
    pub resource_utilization: f32,
    pub error_rate: f32,
    pub coordination_overhead: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_id: String,
    pub target_component: String,
    pub optimization_type: String,
    pub expected_improvement: f32,
    pub implementation_complexity: String,
    pub risk_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub metric_name: String,
    pub trend_direction: String,
    pub rate_of_change: f32,
    pub confidence: f32,
    pub prediction_horizon_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationEventMessage {
    pub timestamp: i64,
    pub event_type: String,
    pub source_agent: String,
    pub target_agents: Vec<String>,
    pub event_data: serde_json::Value,
    pub coordination_impact: f32,
}

/// WebSocket protocol handler with multi-MCP support
pub struct AgentVisualizationProtocol {
    _update_interval_ms: u64,
    position_buffer: Vec<PositionUpdate>,
    mcp_servers: std::collections::HashMap<String, McpServerInfo>,
    agent_cache: std::collections::HashMap<String, MultiMcpAgentStatus>,
    topology_cache: std::collections::HashMap<String, SwarmTopologyData>,
    last_discovery: Option<chrono::DateTime<chrono::Utc>>,
}

impl AgentVisualizationProtocol {
    pub fn new() -> Self {
        Self {
            _update_interval_ms: 16, // ~60fps
            position_buffer: Vec::new(),
            mcp_servers: std::collections::HashMap::new(),
            agent_cache: std::collections::HashMap::new(),
            topology_cache: std::collections::HashMap::new(),
            last_discovery: None,
        }
    }
    
    /// Register an MCP server for agent discovery
    pub fn register_mcp_server(&mut self, server_info: McpServerInfo) {
        log::info!("Registering MCP server: {} ({}:{})", server_info.server_id, server_info.host, server_info.port);
        self.mcp_servers.insert(server_info.server_id.clone(), server_info);
    }
    
    /// Update agent data from MCP server
    pub fn update_agents_from_server(&mut self, server_type: McpServerType, agents: Vec<MultiMcpAgentStatus>) {
        for agent in agents {
            self.agent_cache.insert(agent.agent_id.clone(), agent);
        }
        log::debug!("Updated {} agents from {:?} server", self.agent_cache.len(), server_type);
    }
    
    /// Create discovery message with all MCP servers and agents
    pub fn create_discovery_message(&mut self) -> String {
        let timestamp = chrono::Utc::now();
        self.last_discovery = Some(timestamp);
        
        let servers: Vec<McpServerInfo> = self.mcp_servers.values().cloned().collect();
        let total_agents = self.agent_cache.len() as u32;
        
        // Group agents by swarm
        let mut swarms: std::collections::HashMap<String, Vec<&MultiMcpAgentStatus>> = std::collections::HashMap::new();
        for agent in self.agent_cache.values() {
            swarms.entry(agent.swarm_id.clone()).or_insert_with(Vec::new).push(agent);
        }
        
        let swarm_infos: Vec<SwarmInfo> = swarms.into_iter().map(|(swarm_id, agents)| {
            let total_health: f32 = agents.iter().map(|a| a.performance.health_score).sum();
            let avg_health = if !agents.is_empty() { total_health / agents.len() as f32 } else { 0.0 };
            
            SwarmInfo {
                swarm_id,
                server_source: agents.first().map(|a| a.server_source.clone()).unwrap_or(McpServerType::Custom("unknown".to_string())),
                topology: "hierarchical".to_string(), // TODO: Get from actual topology
                agent_count: agents.len() as u32,
                health_score: avg_health,
                coordination_efficiency: 0.85, // TODO: Calculate from actual metrics
            }
        }).collect();
        
        let global_topology = GlobalTopology {
            inter_swarm_connections: vec![], // TODO: Discover inter-swarm connections
            coordination_hierarchy: vec![], // TODO: Build coordination hierarchy
            data_flow_patterns: vec![], // TODO: Analyze data flow patterns
        };
        
        let discovery = DiscoveryMessage {
            timestamp: timestamp.timestamp_millis(),
            servers,
            total_agents,
            swarms: swarm_infos,
            global_topology,
        };
        
        let message = MultiMcpVisualizationMessage::Discovery(discovery);
        serde_json::to_string(&message).unwrap_or_default()
    }
    
    /// Create differential agent update message
    pub fn create_agent_update_message(&self, updated_agents: Vec<MultiMcpAgentStatus>) -> String {
        let differential_updates: Vec<AgentDifferentialUpdate> = updated_agents.iter().map(|agent| {
            let mut field_updates = std::collections::HashMap::new();
            field_updates.insert("status".to_string(), serde_json::json!(agent.status));
            field_updates.insert("last_active".to_string(), serde_json::json!(agent.last_active));
            
            let performance_delta = PerformanceDelta {
                cpu_change: 0.0, // TODO: Calculate actual deltas
                memory_change: 0.0,
                task_completion_rate: agent.performance.success_rate,
                error_rate_change: 0.0,
            };
            
            AgentDifferentialUpdate {
                agent_id: agent.agent_id.clone(),
                field_updates,
                performance_delta: Some(performance_delta),
            }
        }).collect();
        
        let update_msg = MultiAgentUpdateMessage {
            timestamp: chrono::Utc::now().timestamp_millis(),
            agents: updated_agents,
            differential_updates,
            removed_agents: vec![], // TODO: Track removed agents
        };
        
        let message = MultiMcpVisualizationMessage::MultiAgentUpdate(update_msg);
        serde_json::to_string(&message).unwrap_or_default()
    }
    
    /// Create topology update message
    pub fn create_topology_update(&self, swarm_id: String, topology_data: SwarmTopologyData) -> String {
        self.topology_cache.clone().insert(swarm_id.clone(), topology_data.clone());
        
        let topology_update = TopologyUpdateMessage {
            timestamp: chrono::Utc::now().timestamp_millis(),
            swarm_id,
            topology_changes: vec![], // TODO: Track actual topology changes
            new_connections: vec![], // TODO: Track new connections
            removed_connections: vec![], // TODO: Track removed connections
            coordination_updates: vec![], // TODO: Track coordination updates
        };
        
        let message = MultiMcpVisualizationMessage::TopologyUpdate(topology_update);
        serde_json::to_string(&message).unwrap_or_default()
    }
    
    /// Create performance analysis message
    pub fn create_performance_analysis(&self) -> String {
        let agents: Vec<&MultiMcpAgentStatus> = self.agent_cache.values().collect();
        
        let total_throughput: f32 = agents.iter().map(|a| a.performance.throughput).sum();
        let avg_latency: f32 = if !agents.is_empty() {
            agents.iter().map(|a| a.performance.response_time_ms).sum::<f32>() / agents.len() as f32
        } else { 0.0 };
        
        let global_metrics = GlobalPerformanceMetrics {
            total_throughput,
            average_latency: avg_latency,
            system_efficiency: 0.85, // TODO: Calculate from actual metrics
            resource_utilization: agents.iter().map(|a| (a.performance.cpu_usage + a.performance.memory_usage) / 2.0).sum::<f32>() / agents.len().max(1) as f32,
            error_rate: agents.iter().map(|a| a.performance.tasks_failed as f32 / (a.performance.tasks_completed + a.performance.tasks_failed).max(1) as f32).sum::<f32>() / agents.len().max(1) as f32,
            coordination_overhead: 0.15, // TODO: Calculate from coordination metrics
        };
        
        // Identify bottlenecks
        let bottlenecks: Vec<Bottleneck> = agents.iter().filter_map(|agent| {
            if agent.performance.cpu_usage > 0.9 || agent.performance.memory_usage > 0.9 {
                Some(Bottleneck {
                    agent_id: agent.agent_id.clone(),
                    bottleneck_type: if agent.performance.cpu_usage > 0.9 { "cpu" } else { "memory" }.to_string(),
                    severity: (agent.performance.cpu_usage + agent.performance.memory_usage) / 2.0,
                    impact_agents: vec![], // TODO: Calculate impact on other agents
                    suggested_action: "Scale resources or redistribute workload".to_string(),
                })
            } else {
                None
            }
        }).collect();
        
        let performance_analysis = PerformanceAnalysisMessage {
            timestamp: chrono::Utc::now().timestamp_millis(),
            global_metrics,
            bottlenecks,
            optimization_suggestions: vec![], // TODO: Generate optimization suggestions
            trend_analysis: vec![], // TODO: Analyze performance trends
        };
        
        let message = MultiMcpVisualizationMessage::PerformanceAnalysis(performance_analysis);
        serde_json::to_string(&message).unwrap_or_default()
    }
    
    /// Get current agent count by server type
    pub fn get_agent_count_by_server(&self, server_type: &McpServerType) -> u32 {
        self.agent_cache.values()
            .filter(|agent| std::mem::discriminant(&agent.server_source) == std::mem::discriminant(server_type))
            .count() as u32
    }
    
    /// Check if discovery is needed (every 30 seconds)
    pub fn needs_discovery(&self) -> bool {
        self.last_discovery.map_or(true, |last| {
            chrono::Utc::now().signed_duration_since(last).num_seconds() > 30
        })
    }
    
    /// Create initial JSON message for new client (legacy compatibility)
    pub fn create_init_message(
        swarm_id: &str,
        topology: &str,
        agents: Vec<crate::types::claude_flow::AgentStatus>,
    ) -> String {
        use crate::services::agent_visualization_processor::AgentVisualizationProcessor;
        
        let mut processor = AgentVisualizationProcessor::new();
        let viz_data = processor.create_visualization_packet(agents, swarm_id.to_string(), topology.to_string());
        
        // Convert to init message format
        let init_agents: Vec<AgentInit> = viz_data.agents.into_iter().map(|agent| {
            AgentInit {
                id: agent.id,
                name: agent.name,
                agent_type: agent.agent_type,
                status: agent.status,
                color: agent.color,
                shape: match agent.shape_type {
                    crate::services::agent_visualization_processor::ShapeType::Sphere => "sphere",
                    crate::services::agent_visualization_processor::ShapeType::Cube => "cube",
                    crate::services::agent_visualization_processor::ShapeType::Octahedron => "octahedron",
                    crate::services::agent_visualization_processor::ShapeType::Cylinder => "cylinder",
                    crate::services::agent_visualization_processor::ShapeType::Torus => "torus",
                    crate::services::agent_visualization_processor::ShapeType::Cone => "cone",
                    crate::services::agent_visualization_processor::ShapeType::Pyramid => "pyramid",
                }.to_string(),
                size: agent.size,
                health: agent.health,
                cpu: agent.cpu_usage,
                memory: agent.memory_usage,
                activity: agent.activity_level,
                tasks_active: agent.active_tasks,
                tasks_completed: agent.completed_tasks,
                success_rate: agent.success_rate,
                tokens: agent.token_usage,
                token_rate: agent.token_rate,
                capabilities: agent.metadata.capabilities,
                created_at: agent.metadata.created_at.timestamp(),
            }
        }).collect();
        
        let init_connections: Vec<ConnectionInit> = viz_data.connections.into_iter().map(|conn| {
            ConnectionInit {
                id: conn.id,
                source: conn.source_id,
                target: conn.target_id,
                strength: conn.strength,
                flow_rate: conn.flow_rate,
                color: conn.color,
                active: conn.is_active,
            }
        }).collect();
        
        let visual_config = VisualConfig {
            colors: viz_data.visual_config.color_scheme,
            sizes: viz_data.visual_config.size_multipliers,
            animations: HashMap::new(), // TODO: Populate from config
            effects: EffectsConfig {
                glow: true,
                particles: true,
                bloom: true,
                shadows: false,
            },
        };
        
        let init_msg = InitializeMessage {
            timestamp: chrono::Utc::now().timestamp(),
            swarm_id: swarm_id.to_string(),
            topology: topology.to_string(),
            agents: init_agents,
            connections: init_connections,
            visual_config,
            physics_config: viz_data.physics_config,
            positions: HashMap::new(), // Let client calculate initial positions
        };
        
        let message = AgentVisualizationMessage::Initialize(init_msg);
        serde_json::to_string(&message).unwrap_or_default()
    }
    
    /// Buffer position update
    pub fn add_position_update(&mut self, id: String, x: f32, y: f32, z: f32, vx: f32, vy: f32, vz: f32) {
        self.position_buffer.push(PositionUpdate {
            id,
            x,
            y,
            z,
            vx: Some(vx),
            vy: Some(vy),
            vz: Some(vz),
        });
    }
    
    /// Create position update message and clear buffer
    pub fn create_position_update(&mut self) -> Option<String> {
        if self.position_buffer.is_empty() {
            return None;
        }
        
        let msg = PositionUpdateMessage {
            timestamp: chrono::Utc::now().timestamp_millis(),
            positions: std::mem::take(&mut self.position_buffer),
        };
        
        let message = AgentVisualizationMessage::PositionUpdate(msg);
        Some(serde_json::to_string(&message).unwrap_or_default())
    }
    
    /// Create state update message (legacy compatibility)
    pub fn create_state_update(updates: Vec<AgentStateUpdate>) -> String {
        let msg = StateUpdateMessage {
            timestamp: chrono::Utc::now().timestamp_millis(),
            updates,
        };
        
        let message = AgentVisualizationMessage::StateUpdate(msg);
        serde_json::to_string(&message).unwrap_or_default()
    }
}